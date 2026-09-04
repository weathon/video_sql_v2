from __future__ import annotations

import asyncio
import base64
import copy
import json
import math
import os
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import duckdb
import httpx
import numpy as np
from agents import Agent, ItemHelpers, ModelSettings, RunConfig, RunContextWrapper, Runner, function_tool
from agents.items import MessageOutputItem
from agents.models.openai_responses import OpenAIResponsesModel
from agents.tool import ToolOutputImage, ToolOutputText
from openai import AsyncOpenAI
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field


SEARCHABLE_TABLES = {
    "episodes": ("text", "start_s", "end_s"),
    "entities": ("name", "first_seen_s", "last_seen_s"),
    "semantic_triples": ("text", "snapshot_s", "snapshot_s"),
    "topic_facts": ("fact", "time_s", "time_s"),
    "event_steps": ("description", "start_s", "end_s"),
    "object_states": ("text", "start_s", "end_s"),
    "visual_clips": ("text", "start_s", "end_s"),
}
CONTEXT_LIMIT = 262_144
COMPACTION_THRESHOLD = 50_000
TURN_LIMIT = 100
SQL_TIMEOUT_S = 30
DATASET_SNAPSHOT = Path(
    "/raid/huggingface/hub/datasets--MM-Lifelong--MM-Lifelong/"
    "snapshots/248aa82039a574e63a2e524746a7cd8f32330443"
)


class InvalidTrace(RuntimeError):
    pass


class AgentRaised(RuntimeError):
    pass


class CompactionLimit(RuntimeError):
    def __init__(self, message: str, history: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.history = history


class MainAnswer(BaseModel):
    answer: str


class ClipAnswer(BaseModel):
    status: Literal["answered", "unknown"]
    answer: str
    evidence_timestamps_s: list[float] = Field(default_factory=list)


class ReflectionAnswer(BaseModel):
    reflection: str
    meta_reflection: str


class JudgeAnswer(BaseModel):
    score: int = Field(ge=0, le=5)


class SummaryAnswer(BaseModel):
    summary: str


@dataclass
class ModelBundle:
    model: OpenAIResponsesModel
    settings: ModelSettings
    run_config: RunConfig
    name: str


@dataclass
class RolloutContext:
    database_path: Path
    embedding_root: Path
    embedding_url: str
    dataset_root: Path
    skills_dir: Path
    run_dir: Path
    model_bundle: ModelBundle
    video_id: str
    notes: list[str] = field(default_factory=list)
    submitted: Any = None
    raised_reason: str | None = None
    media_counter: int = 0


@dataclass
class UpdaterContext:
    skills_dir: Path
    stopped: bool = False
    stop_reason: str = ""


def json_tool_error(_ctx: RunContextWrapper[Any], error: Exception) -> str:
    if isinstance(error, (InvalidTrace, AgentRaised, CompactionLimit)):
        raise error
    return json.dumps({"error": str(error)}, ensure_ascii=False)


def make_model_bundle(
    model_name: str,
    base_url: str,
    api_key: str,
    reasoning_effort: Literal["low", "medium"],
) -> ModelBundle:
    client = AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=3)
    model = OpenAIResponsesModel(model=model_name, openai_client=client)
    settings = ModelSettings(
        reasoning={"effort": reasoning_effort},
        parallel_tool_calls=True,
        preserve_raw_usage=True,
    )
    run_config = RunConfig(reasoning_item_id_policy="preserve")
    return ModelBundle(model=model, settings=settings, run_config=run_config, name=model_name)


def load_schema(database_path: Path) -> str:
    connection = duckdb.connect(str(database_path), read_only=True)
    tables = connection.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'main' ORDER BY table_name"
    ).fetchall()
    blocks = []
    for (table,) in tables:
        columns = connection.execute(f"PRAGMA table_info('{table}')").fetchall()
        blocks.append(
            f"{table}(" + ", ".join(f"{column[1]} {column[2]}" for column in columns) + ")"
        )
    connection.close()
    return "\n".join(blocks)


def load_skill_catalog(skills_dir: Path) -> str:
    entries = []
    for path in sorted(skills_dir.glob("*.md")):
        if path.name == "META-SKILL.md":
            continue
        text = path.read_text()
        if not text.startswith("---\n"):
            raise ValueError(f"missing YAML frontmatter: {path}")
        _, frontmatter, _ = text.split("---\n", 2)
        values = {}
        for line in frontmatter.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                values[key.strip()] = value.strip()
        entries.append(f"- {values['name']}: {values['description']}")
    return "\n".join(entries)


def sanitize_trace(value: Any, media_dir: Path) -> Any:
    if isinstance(value, dict):
        clean = {}
        for key, item in value.items():
            if key == "image_url" and isinstance(item, str) and item.startswith("data:image/"):
                header, encoded = item.split(",", 1)
                suffix = header.split(";")[0].split("/")[1]
                filename = f"trace-image-{uuid.uuid4().hex}.{suffix}"
                (media_dir / filename).write_bytes(base64.b64decode(encoded))
                clean[key] = filename
            else:
                clean[key] = sanitize_trace(item, media_dir)
        return clean
    if isinstance(value, list):
        return [sanitize_trace(item, media_dir) for item in value]
    if hasattr(value, "model_dump"):
        return sanitize_trace(value.model_dump(mode="json"), media_dir)
    return value


def save_trace(path: Path, result: Any, error: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    media_dir = path.parent.parent / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    payload = {"error": error, "items": sanitize_trace(result.to_input_list(), media_dir)}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


async def stream_agent(
    agent: Agent[Any],
    agent_input: str | list[dict[str, Any]],
    context: Any,
    bundle: ModelBundle,
    trace_path: Path,
    compact_main: bool = False,
    allow_forced_submit: bool = True,
) -> tuple[Any, list[dict[str, Any]], bool]:
    result = Runner.run_streamed(
        agent,
        agent_input,
        context=context,
        max_turns=TURN_LIMIT,
        run_config=bundle.run_config,
    )
    completions: list[str] = []
    compacted = False
    force_submit = False
    try:
        async for event in result.stream_events():
            if event.type == "run_item_stream_event" and event.name == "message_output_created":
                assert isinstance(event.item, MessageOutputItem)
                text = ItemHelpers.text_message_output(event.item)
                completions.append(text)
                if len(completions) >= 3 and completions[-3:] == [text, text, text]:
                    result.cancel()
                    raise InvalidTrace("three consecutive non-reasoning completions were identical")
            if event.type == "raw_response_event":
                response = getattr(event.data, "response", None)
                usage = getattr(response, "usage", None)
                if usage is not None:
                    remaining = CONTEXT_LIMIT - usage.input_tokens - usage.output_tokens
                    if compact_main and remaining <= COMPACTION_THRESHOLD and not compacted:
                        result.cancel(mode="after_turn")
                        compacted = True
                    if (
                        not compact_main
                        and allow_forced_submit
                        and remaining <= COMPACTION_THRESHOLD
                        and not force_submit
                    ):
                        result.cancel(mode="after_turn")
                        force_submit = True
        save_trace(trace_path, result)
    except Exception as error:
        save_trace(trace_path, result, str(error))
        raise
    items = result.to_input_list()
    if force_submit:
        original_tools = agent.tools
        submit_tools = [
            tool
            for tool in original_tools
            if getattr(tool, "name", "") in {"submit_answer", "submit_reflection", "submit_score", "submit_summary", "stop"}
        ]
        if not submit_tools:
            raise InvalidTrace("context threshold reached without a submission tool")
        agent.tools = submit_tools
        try:
            return await stream_agent(
                agent,
                items
                + [
                    {
                        "role": "user",
                        "content": "Context is nearly full. Do not call exploratory tools. Submit the best final output now.",
                    }
                ],
                context,
                bundle,
                trace_path.with_name(trace_path.stem + "-forced" + trace_path.suffix),
                allow_forced_submit=False,
            )
        finally:
            agent.tools = original_tools
    return result.final_output, items, compacted


def resolve_source_rows(context: RolloutContext, start_s: float, end_s: float) -> list[tuple[Any, ...]]:
    connection = duckdb.connect(str(context.database_path), read_only=True)
    rows = connection.execute(
        "SELECT source_video_id, filename, timeline_start_s, timeline_end_s "
        "FROM source_videos WHERE video_id = ? AND timeline_end_s > ? AND timeline_start_s < ? "
        "ORDER BY timeline_start_s",
        [context.video_id, start_s, end_s],
    ).fetchall()
    connection.close()
    return rows


def render_frame(video_path: Path, local_s: float, output_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(local_s),
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            "-y",
            str(output_path),
        ],
        check=True,
    )


def frame_at(context: RolloutContext, timestamp_s: float, output_path: Path) -> None:
    rows = resolve_source_rows(context, timestamp_s, timestamp_s + 0.001)
    if len(rows) != 1:
        raise ValueError(f"timestamp {timestamp_s} maps to {len(rows)} source videos")
    _, filename, timeline_start_s, _ = rows[0]
    render_frame(context.dataset_root / filename, timestamp_s - timeline_start_s, output_path)


def data_image(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def make_contact_sheets(
    context: RolloutContext,
    start_s: float,
    end_s: float,
    fps: float,
    crop_x1: int,
    crop_y1: int,
    crop_x2: int,
    crop_y2: int,
) -> list[Path]:
    timestamps = np.arange(start_s, end_s, 1.0 / fps).tolist()
    if not timestamps:
        timestamps = [start_s]
    request_dir = context.run_dir / "media" / f"sample-{context.media_counter:05d}"
    context.media_counter += 1
    request_dir.mkdir(parents=True, exist_ok=True)
    sheets = []
    for page_index in range(math.ceil(len(timestamps) / 12)):
        page_times = timestamps[page_index * 12 : (page_index + 1) * 12]
        sheet = Image.new("RGB", (1920, 900), "black")
        for index, timestamp_s in enumerate(page_times):
            raw_path = request_dir / f"frame-{page_index:03d}-{index:02d}.jpg"
            frame_at(context, timestamp_s, raw_path)
            with Image.open(raw_path) as raw:
                image = raw.convert("RGB")
                width, height = image.size
                image = image.crop(
                    (
                        width * crop_x1 // 1000,
                        height * crop_y1 // 1000,
                        width * crop_x2 // 1000,
                        height * crop_y2 // 1000,
                    )
                )
                image.thumbnail((480, 270))
                cell = Image.new("RGB", (480, 300), "black")
                cell.paste(image, ((480 - image.width) // 2, 25))
                ImageDraw.Draw(cell).text((8, 5), f"t={timestamp_s:.3f}s", fill="white")
                sheet.paste(cell, ((index % 4) * 480, (index // 4) * 300))
            raw_path.unlink()
        path = request_dir / f"contact-sheet-{page_index:03d}.jpg"
        sheet.save(path, quality=90)
        sheets.append(path)
    return sheets


def query_database(connection: duckdb.DuckDBPyConnection, sql: str) -> tuple[list[str], list[tuple[Any, ...]]]:
    cursor = connection.execute(sql)
    columns = [item[0] for item in cursor.description]
    rows = cursor.fetchall()
    return columns, rows


async def execute_sql(database_path: Path, sql: str) -> tuple[list[str], list[tuple[Any, ...]]]:
    connection = duckdb.connect(str(database_path), read_only=True)
    task = asyncio.create_task(asyncio.to_thread(query_database, connection, sql))
    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout=SQL_TIMEOUT_S)
    except TimeoutError:
        connection.interrupt()
        try:
            await task
        except duckdb.InterruptException:
            pass
        raise TimeoutError(f"SQL exceeded {SQL_TIMEOUT_S} seconds")
    finally:
        connection.close()


def clipped_cell(value: Any) -> dict[str, Any]:
    text = str(value)
    return {"value": text[:500], "original_length": len(text)}


def make_main_tools(context: RolloutContext) -> list[Any]:
    @function_tool(failure_error_function=json_tool_error)
    async def run_sql(sql: str) -> str:
        """Execute one read-only DuckDB SQL statement and return at most 100 rows."""
        columns, rows = await execute_sql(context.database_path, sql)
        return json.dumps(
            {
                "columns": columns,
                "rows": [[clipped_cell(cell) for cell in row] for row in rows[:100]],
                "has_more": len(rows) > 100,
            },
            ensure_ascii=False,
        )

    @function_tool(failure_error_function=json_tool_error)
    async def sql_qa(sql: str, question: str) -> str:
        """Execute SQL and have a separate model answer a question about the full result."""
        columns, rows = await execute_sql(context.database_path, sql)
        raw = json.dumps({"columns": columns, "rows": rows}, ensure_ascii=False, default=str)
        if len(raw) > 50_000:
            raw = raw[:50_000] + "\n[TRUNCATED AT 50000 CHARACTERS]"
        answer_context: dict[str, Any] = {"answer": None}

        @function_tool(failure_error_function=json_tool_error)
        def submit_answer(answer: str) -> str:
            """Submit the final answer to the SQL-result question."""
            answer_context["answer"] = answer
            return answer

        agent = Agent(
            name="sql-result-qa",
            instructions="Answer only from the supplied SQL result. Call submit_answer when done.",
            model=context.model_bundle.model,
            model_settings=context.model_bundle.settings,
            tools=[submit_answer],
            tool_use_behavior={"stop_at_tool_names": ["submit_answer"]},
        )
        trace_path = context.run_dir / "traces" / f"sql-qa-{uuid.uuid4().hex}.json"
        await stream_agent(
            agent,
            f"Question:\n{question}\n\nSQL result:\n{raw}",
            answer_context,
            context.model_bundle,
            trace_path,
        )
        if answer_context["answer"] is None:
            raise ValueError("sql_qa did not call submit_answer")
        return answer_context["answer"]

    @function_tool(failure_error_function=json_tool_error)
    async def semantic_search(
        query: str,
        n: int,
        table: str,
        rank_start: int = 1,
        rank_end: int | None = None,
        threshold_min: float | None = None,
        threshold_max: float | None = None,
    ) -> str:
        """Search an embedded table by cosine similarity with rank or score filters."""
        if table not in SEARCHABLE_TABLES:
            raise ValueError(f"table must be one of {sorted(SEARCHABLE_TABLES)}")
        if rank_start < 1 or n < 1:
            raise ValueError("rank_start and n must be positive")
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                context.embedding_url,
                json={"model": "qwen3-vl-embedding-8b", "input": [query]},
            )
            response.raise_for_status()
        data = response.json()["data"]
        vector = np.asarray(data[0]["embedding"], dtype=np.float32)
        if vector.shape != (4096,) or not np.isfinite(vector).all():
            raise ValueError(f"invalid embedding shape or values: {vector.shape}")
        row_ids = np.load(context.embedding_root / f"{table}.row_ids.npy", mmap_mode="r", allow_pickle=False)
        vectors = np.load(context.embedding_root / f"{table}.vectors.npy", mmap_mode="r", allow_pickle=False)
        scores = vectors @ vector
        order = np.argsort(-scores)
        stop = rank_end if rank_end is not None else rank_start + n - 1
        selected = order[rank_start - 1 : stop]
        hits = []
        connection = duckdb.connect(str(context.database_path), read_only=True)
        text_column, start_column, end_column = SEARCHABLE_TABLES[table]
        for position in selected:
            score = float(scores[position])
            if threshold_min is not None and score < threshold_min:
                continue
            if threshold_max is not None and score > threshold_max:
                continue
            row_id = str(row_ids[position])
            row = connection.execute(
                f"SELECT row_id, video_id, {start_column}, {end_column}, {text_column} "
                f"FROM {table} WHERE row_id = ?",
                [row_id],
            ).fetchone()
            if row is None:
                raise ValueError(f"embedding row_id missing from {table}: {row_id}")
            hits.append(
                {
                    "table": table,
                    "row_id": row[0],
                    "video_id": row[1],
                    "start_s": row[2],
                    "end_s": row[3],
                    "text": str(row[4])[:500],
                    "score": score,
                }
            )
        connection.close()
        return json.dumps(hits, ensure_ascii=False)

    @function_tool(failure_error_function=json_tool_error)
    def add_note(note: str) -> str:
        """Add a working note for this rollout."""
        context.notes.append(note)
        return json.dumps({"saved": len(context.notes)}, ensure_ascii=False)

    @function_tool(failure_error_function=json_tool_error)
    def read_notes() -> str:
        """Read all working notes for this rollout."""
        return json.dumps(context.notes, ensure_ascii=False)

    @function_tool(failure_error_function=json_tool_error)
    def read_skill(name: str) -> str:
        """Read one available skill by its frontmatter name."""
        for path in context.skills_dir.glob("*.md"):
            if path.name == "META-SKILL.md":
                continue
            text = path.read_text()
            if f"name: {name}\n" in text.split("---\n", 2)[1]:
                return text
        raise ValueError(f"unknown skill: {name}")

    @function_tool(failure_error_function=json_tool_error)
    def agent_raise(reason: str) -> str:
        """End this rollout because valid completion is impossible."""
        context.raised_reason = reason
        raise AgentRaised(reason)

    async def clip_qa_impl(
        video_id: str, start_s: float, end_s: float, question: str
    ) -> list[ToolOutputText | ToolOutputImage]:
        if video_id != context.video_id:
            raise ValueError(f"current logical timeline is {context.video_id}, not {video_id}")
        if end_s <= start_s:
            raise ValueError("end_s must be greater than start_s")
        if end_s - start_s > 1200:
            raise ValueError("clip_qa refuses intervals longer than 20 minutes")
        clip_context = copy.copy(context)
        clip_context.notes = []
        clip_context.submitted = None
        clip_context.run_dir = context.run_dir / "clip_qa" / uuid.uuid4().hex

        @function_tool(failure_error_function=json_tool_error)
        async def sample_video(
            start_s: float,
            end_s: float,
            fps: float,
            include_captions: bool,
            crop_x1: int = 0,
            crop_y1: int = 0,
            crop_x2: int = 1000,
            crop_y2: int = 1000,
        ) -> list[ToolOutputText | ToolOutputImage]:
            """Return fully paginated 4x3 contact sheets for a timeline interval."""
            if not (0 <= crop_x1 < crop_x2 <= 1000 and 0 <= crop_y1 < crop_y2 <= 1000):
                raise ValueError("crop coordinates must satisfy 0 <= min < max <= 1000")
            if start_s < 0 or end_s <= start_s or fps <= 0:
                raise ValueError("invalid time range or fps")
            sheets = await asyncio.to_thread(
                make_contact_sheets,
                clip_context,
                start_s,
                end_s,
                fps,
                crop_x1,
                crop_y1,
                crop_x2,
                crop_y2,
            )
            blocks: list[ToolOutputText | ToolOutputImage] = []
            caption_text = ""
            if include_captions:
                columns, rows = await execute_sql(
                    clip_context.database_path,
                    "SELECT start_s, end_s, text FROM episodes "
                    f"WHERE video_id = '{video_id}' AND granularity = '30sec' "
                    f"AND end_s > {float(start_s)} AND start_s < {float(end_s)} ORDER BY start_s",
                )
                caption_text = json.dumps({"columns": columns, "rows": rows}, ensure_ascii=False)
            for sheet in sheets:
                blocks.append(
                    ToolOutputText(
                        text=f"{sheet.name}\n{caption_text}" if caption_text else sheet.name
                    )
                )
                blocks.append(ToolOutputImage(image_url=data_image(sheet), detail="auto"))
            return blocks

        @function_tool(failure_error_function=json_tool_error)
        def clip_add_note(note: str) -> str:
            """Add a working note about the clip."""
            clip_context.notes.append(note)
            return json.dumps({"saved": len(clip_context.notes)})

        @function_tool(failure_error_function=json_tool_error)
        def clip_read_notes() -> str:
            """Read all clip working notes."""
            return json.dumps(clip_context.notes, ensure_ascii=False)

        @function_tool(failure_error_function=json_tool_error)
        def submit_answer(
            status: Literal["answered", "unknown"],
            answer: str,
            evidence_timestamps_s: list[float],
        ) -> str:
            """Submit the clip answer. Answered results require evidence timestamps."""
            if status == "answered" and not evidence_timestamps_s:
                raise ValueError("answered requires at least one evidence timestamp")
            clip_context.submitted = ClipAnswer(
                status=status,
                answer=answer,
                evidence_timestamps_s=evidence_timestamps_s,
            )
            return clip_context.submitted.model_dump_json()

        clip_agent = Agent(
            name="clip-qa",
            instructions=(
                "Inspect the requested video interval to answer the question. Intervals up to ten minutes "
                "are recommended. Use sample_video as needed. Call submit_answer when finished."
            ),
            model=context.model_bundle.model,
            model_settings=context.model_bundle.settings,
            tools=[sample_video, clip_add_note, clip_read_notes, submit_answer],
            tool_use_behavior={"stop_at_tool_names": ["submit_answer"]},
        )
        await stream_agent(
            clip_agent,
            f"Logical timeline: {video_id}\nRange: {start_s:.3f}-{end_s:.3f}s\nQuestion: {question}",
            clip_context,
            context.model_bundle,
            clip_context.run_dir / "traces" / "clip.json",
        )
        answer = clip_context.submitted
        if answer is None:
            raise ValueError("clip_qa did not call submit_answer")
        blocks: list[ToolOutputText | ToolOutputImage] = [
            ToolOutputText(text=answer.model_dump_json())
        ]
        for timestamp_s in answer.evidence_timestamps_s:
            screenshot = clip_context.run_dir / "media" / f"evidence-{timestamp_s:.3f}.jpg"
            screenshot.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(frame_at, clip_context, timestamp_s, screenshot)
            blocks.append(ToolOutputText(text=screenshot.name))
            blocks.append(ToolOutputImage(image_url=data_image(screenshot), detail="high"))
        return blocks

    clip_qa = function_tool(
        clip_qa_impl,
        name_override="clip_qa",
        description_override=(
            "Inspect a logical video interval with an internal visual agent. Intervals under ten minutes "
            "are recommended; intervals over twenty minutes are rejected."
        ),
        failure_error_function=json_tool_error,
    )
    return [run_sql, sql_qa, semantic_search, clip_qa, add_note, read_notes, read_skill, agent_raise]


def make_retrieval_agent(context: RolloutContext) -> Agent[RolloutContext]:
    @function_tool(failure_error_function=json_tool_error)
    def submit_answer(answer: str) -> str:
        """Submit the final answer to the memory question."""
        context.submitted = MainAnswer(answer=answer)
        return context.submitted.model_dump_json()

    tools = make_main_tools(context) + [submit_answer]
    schema = load_schema(context.database_path)
    catalog = load_skill_catalog(context.skills_dir)
    instructions = (
        "You answer questions by retrieving evidence from a video-memory SQL database. "
        "Use the available tools and call submit_answer when finished.\n\n"
        f"Database schema:\n{schema}\n\nAvailable skills:\n{catalog}"
    )
    return Agent(
        name="video-memory-retrieval",
        instructions=instructions,
        model=context.model_bundle.model,
        model_settings=context.model_bundle.settings,
        tools=tools,
        tool_use_behavior={"stop_at_tool_names": ["submit_answer"]},
    )


async def run_main_with_compaction(
    agent: Agent[RolloutContext],
    prompt: str,
    context: RolloutContext,
    trace_dir: Path,
) -> tuple[MainAnswer, list[dict[str, Any]], int]:
    history: str | list[dict[str, Any]] = prompt
    compactions = 0
    while True:
        _, items, needs_compaction = await stream_agent(
            agent,
            history,
            context,
            context.model_bundle,
            trace_dir / f"main-segment-{compactions:02d}.json",
            compact_main=True,
        )
        if context.submitted is not None:
            return context.submitted, items, compactions
        if not needs_compaction:
            raise InvalidTrace("main agent stopped without submit_answer or compaction")
        original_tools = agent.tools
        agent.tools = []
        summary_result, summary_items, _ = await stream_agent(
            agent,
            items
            + [
                {
                    "role": "user",
                    "content": (
                        "Stop using tools. Produce a trusted handoff that preserves the question, all useful "
                        "evidence, SQL findings, visual findings, uncertainties, and the exact remaining work."
                    ),
                }
            ],
            context,
            context.model_bundle,
            trace_dir / f"compaction-{compactions:02d}.json",
            allow_forced_submit=False,
        )
        agent.tools = original_tools
        compactions += 1
        if compactions >= 3:
            trusted_history = [
                {
                    "role": "user",
                    "content": (
                        "The answer attempt reached the third compaction limit. Reflect from this trusted "
                        "handoff and focus on why the search loop failed to converge.\n\n" + str(summary_result)
                    ),
                }
            ]
            raise CompactionLimit("third compaction generated", trusted_history)
        handoff = str(summary_result)
        if not handoff.strip():
            handoff = ItemHelpers.text_message_outputs(
                [item for item in getattr(summary_items, "new_items", []) if isinstance(item, MessageOutputItem)]
            )
        history = (
            "Continue the original task from this trusted handoff. Use tools as needed and call submit_answer.\n\n"
            + handoff
        )


def make_reflection_tools(context: RolloutContext) -> list[Any]:
    tools = make_main_tools(context)

    @function_tool(failure_error_function=json_tool_error)
    def submit_reflection(reflection: str, meta_reflection: str) -> str:
        """Submit abstract task-learning and reflection-learning conclusions."""
        context.submitted = ReflectionAnswer(
            reflection=reflection,
            meta_reflection=meta_reflection,
        )
        return context.submitted.model_dump_json()

    return tools + [submit_reflection]


async def continue_reflection(
    agent: Agent[RolloutContext],
    answer_history: list[dict[str, Any]],
    context: RolloutContext,
    reference_answer: str,
    score: int,
    trace_path: Path,
) -> ReflectionAnswer:
    context.submitted = None
    agent.tools = make_reflection_tools(context)
    agent.tool_use_behavior = {"stop_at_tool_names": ["submit_reflection"]}
    meta_skill = (context.skills_dir / "META-SKILL.md").read_text()
    reflection_prompt = (
        f"The correct answer is: {reference_answer}\nYour score is: {score}/5.\n\n"
        "Continue as the same agent in the same conversation. Revisit your full work and use retrieval tools "
        "again when necessary. Explain abstractly why the approach succeeded or failed and derive reusable "
        "reflection lessons. Remove dataset-specific sensitive details. Update, replace, or delete old advice "
        "when it is redundant, conflicting, or obsolete. Call submit_reflection when finished.\n\n"
        f"META-SKILL.md:\n{meta_skill}"
    )
    await stream_agent(
        agent,
        answer_history + [{"role": "user", "content": reflection_prompt}],
        context,
        context.model_bundle,
        trace_path,
    )
    if context.submitted is None:
        raise InvalidTrace("reflection did not call submit_reflection")
    return context.submitted


async def judge_answer(
    bundle: ModelBundle,
    reference_answer: str,
    candidate_answer: str,
    trace_path: Path,
) -> JudgeAnswer:
    state: dict[str, Any] = {"answer": None}

    @function_tool(failure_error_function=json_tool_error)
    def submit_score(score: int) -> str:
        """Submit an integer semantic-equivalence score from 0 to 5."""
        state["answer"] = JudgeAnswer(score=score)
        return state["answer"].model_dump_json()

    agent = Agent(
        name="answer-judge",
        instructions=(
            "Compare only the reference answer and candidate answer. Score semantic equivalence from 0 to 5. "
            "Ignore attached video IDs, timestamps, row IDs, and other identifiers. Call submit_score."
        ),
        model=bundle.model,
        model_settings=bundle.settings,
        tools=[submit_score],
        tool_use_behavior={"stop_at_tool_names": ["submit_score"]},
    )
    await stream_agent(
        agent,
        f"Reference answer: {reference_answer}\nCandidate answer: {candidate_answer}",
        state,
        bundle,
        trace_path,
    )
    if state["answer"] is None:
        raise InvalidTrace("judge did not call submit_score")
    return state["answer"]


async def summarize_reflections(
    bundle: ModelBundle,
    question_summaries: list[dict[str, Any]],
    trace_path: Path,
) -> str:
    state: dict[str, Any] = {"answer": None}

    @function_tool(failure_error_function=json_tool_error)
    def submit_summary(summary: str) -> str:
        """Submit the consolidated abstract reflection summary."""
        state["answer"] = SummaryAnswer(summary=summary)
        return state["answer"].model_dump_json()

    agent = Agent(
        name="reflection-summarizer",
        instructions=(
            "Consolidate the supplied rollout reflections into reusable, non-duplicative guidance. Preserve "
            "both retrieval lessons and meta-reflection lessons. Call submit_summary."
        ),
        model=bundle.model,
        model_settings=bundle.settings,
        tools=[submit_summary],
        tool_use_behavior={"stop_at_tool_names": ["submit_summary"]},
    )
    await stream_agent(
        agent,
        json.dumps(question_summaries, ensure_ascii=False),
        state,
        bundle,
        trace_path,
    )
    if state["answer"] is None:
        raise InvalidTrace("summarizer did not call submit_summary")
    return state["answer"].summary


def make_updater_agent(context: UpdaterContext, bundle: ModelBundle) -> Agent[UpdaterContext]:
    def skill_path(name: str) -> Path:
        if Path(name).name != name or not name.endswith(".md"):
            raise ValueError("name must be a Markdown filename without directories")
        return context.skills_dir / name

    @function_tool(failure_error_function=json_tool_error)
    def list_skills() -> str:
        """List current skill Markdown filenames."""
        return json.dumps(sorted(path.name for path in context.skills_dir.glob("*.md")))

    @function_tool(failure_error_function=json_tool_error)
    def read_skill_file(name: str) -> str:
        """Read a current skill Markdown file."""
        return skill_path(name).read_text()

    @function_tool(failure_error_function=json_tool_error)
    def write_skill_file(name: str, content: str) -> str:
        """Create or replace a skill Markdown file."""
        skill_path(name).write_text(content)
        return json.dumps({"written": name})

    @function_tool(failure_error_function=json_tool_error)
    def delete_skill_file(name: str) -> str:
        """Delete an obsolete ordinary skill file. META-SKILL.md cannot be deleted."""
        if name == "META-SKILL.md":
            raise ValueError("META-SKILL.md cannot be deleted")
        skill_path(name).unlink()
        return json.dumps({"deleted": name})

    @function_tool(failure_error_function=json_tool_error)
    def git_history(name: str) -> str:
        """Read compact Git history and historical patches for one skill file."""
        process = subprocess.run(
            ["git", "log", "--format=%H %s", "-p", "--", name],
            cwd=context.skills_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        return process.stdout[:50_000]

    @function_tool(failure_error_function=json_tool_error)
    def stop(reason: str) -> str:
        """Stop after all skill file operations are complete."""
        context.stopped = True
        context.stop_reason = reason
        return json.dumps({"stopped": True, "reason": reason}, ensure_ascii=False)

    return Agent(
        name="skill-updater",
        instructions=(
            "Update the skill repository from the supplied reflection summaries. You may add, replace, or "
            "delete ordinary Markdown skills and may update META-SKILL.md, but never delete META-SKILL.md. "
            "Every ordinary skill needs YAML name/description frontmatter plus Introduction, When to use, and "
            "Main content. Put concise **Reason:** provenance after each guidance point, or once after a whole "
            "section when it covers that section. Inspect current files and history as needed. Avoid duplicate, "
            "conflicting, similar, stale, or dataset-specific advice. Call stop only after all edits are done."
        ),
        model=bundle.model,
        model_settings=bundle.settings,
        tools=[list_skills, read_skill_file, write_skill_file, delete_skill_file, git_history, stop],
        tool_use_behavior={"stop_at_tool_names": ["stop"]},
    )


async def update_skills(
    bundle: ModelBundle,
    skills_dir: Path,
    summaries: list[dict[str, Any]],
    rollout_ids: list[str],
    trace_path: Path,
) -> str:
    context = UpdaterContext(skills_dir=skills_dir)
    agent = make_updater_agent(context, bundle)
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=skills_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    prompt = json.dumps(
        {"summaries": summaries, "rollout_ids": rollout_ids, "existing_uncommitted_status": status},
        ensure_ascii=False,
    )
    await stream_agent(agent, prompt, context, bundle, trace_path)
    if not context.stopped:
        raise InvalidTrace("updater did not call stop")
    body = "Rollout IDs:\n" + "\n".join(rollout_ids) + "\n\nOverall reason:\n" + context.stop_reason
    subprocess.run(["git", "add", "-A"], cwd=skills_dir, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=codex",
            "-c",
            "user.email=codex@localhost",
            "commit",
            "--allow-empty",
            "-m",
            "Update learned skills",
            "-m",
            body,
        ],
        cwd=skills_dir,
        check=True,
    )
    return context.stop_reason
