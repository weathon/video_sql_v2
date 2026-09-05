import asyncio
import base64
import dataclasses
import json
import math
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import duckdb
import httpx
import numpy as np
from agents import (
    Agent,
    AgentHooks,
    ModelSettings,
    OpenAIResponsesModel,
    RunConfig,
    Runner,
    StopAtTools,
    ToolOutputImage,
    ToolOutputText,
    function_tool,
)
from openai import AsyncOpenAI
from PIL import Image, ImageDraw, ImageOps
from pydantic import BaseModel

from src.normalize import normalize_text


CONTEXT_LIMIT = 262_144
COMPACTION_THRESHOLD = 50_000
MAX_TURNS = 100
SEARCH_TABLES = {
    "episodes": "text",
    "entities": "name",
    "semantic_triples": "text",
    "topic_facts": "fact",
    "event_steps": "description",
    "object_states": "text",
    "visual_clips": "text",
}
SCHEMA = """
episodes(row_id, start_s, end_s, granularity, text)
entities(row_id, name, normalized_name, first_seen_s, last_seen_s, frequency)
semantic_triples(row_id, caption_30sec_row_id, snapshot_s, subject, predicate, object, text)
topic_chains(row_id, entity, has_lifecycle, level, scope_id, start_s, end_s, summary)
topic_facts(row_id, chain_row_id, caption_30sec_row_id, time_s, fact)
event_chains(row_id, name, chain_type, level, scope_id, start_s, end_s, summary)
event_steps(row_id, chain_row_id, caption_start_row_id, caption_end_row_id, step_index, start_s, end_s, description)
visual_clips(row_id, start_s, end_s, text)
object_states(row_id, episode_3min_row_id, caption_30sec_row_id, start_s, end_s, object, status, text)
graph_edges(row_id, src_row_id, dst_row_id, edge_type, weight)
topic_fact_evidence(row_id, fact_row_id, caption_30sec_row_id)
event_step_evidence(row_id, step_row_id, caption_30sec_row_id)
""".strip()
COMPACTION_INSTRUCTION = """
Stop calling tools. Produce a trusted handoff that preserves the question, established evidence,
unresolved branches, useful row IDs and time intervals, and the exact next investigation steps.
Do not answer the question in this handoff.
""".strip()
FORCED_REFLECTION_INSTRUCTION = """
Stop calling retrieval tools now. Submit the best abstract retrieval reflection and abstract
meta-reflection supported by the conversation so far.
""".strip()


class AgentRaised(Exception):
    pass


class InvalidTrace(Exception):
    pass


class AnswerSubmission(BaseModel):
    answer: str


class ClipSubmission(BaseModel):
    status: Literal["answered", "unknown"]
    answer: str
    evidence_times_s: list[float]


class ReflectionSubmission(BaseModel):
    reflection: str
    meta_reflection: str


@dataclass
class RolloutSession:
    agent: Agent
    history: list[dict[str, Any]]
    answer: str | None
    status: str
    trace_files: list[str]


@dataclass
class HarnessContext:
    database: Path
    embedding_index: Path
    media_map: Path
    skill_dir: Path
    run_dir: Path
    model: OpenAIResponsesModel
    client: AsyncOpenAI
    model_settings: ModelSettings
    run_config: RunConfig
    embedding_url: str
    embedding_model: str
    vectors: dict[str, np.ndarray] = field(init=False)
    row_ids: dict[str, np.ndarray] = field(init=False)
    sources: list[dict[str, Any]] = field(init=False)
    notes_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    trace_counter: int = 0

    def __post_init__(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "traces").mkdir(exist_ok=True)
        (self.run_dir / "media").mkdir(exist_ok=True)
        self.vectors = {}
        self.row_ids = {}
        for table in SEARCH_TABLES:
            self.row_ids[table] = np.load(
                self.embedding_index / f"{table}.row_ids.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            self.vectors[table] = np.load(
                self.embedding_index / f"{table}.vectors.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
        self.sources = json.loads(self.media_map.read_text())["sources"]


def tool_error_json(_context: Any, error: Exception) -> str:
    return json.dumps({"error": str(error)}, ensure_ascii=False)


def json_value(value: Any, string_limit: int | None) -> Any:
    if isinstance(value, str) and string_limit is not None and len(value) > string_limit:
        return {"value": value[:string_limit], "original_length": len(value)}
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def stripped_json(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return stripped_json(value.model_dump(mode="json", exclude_unset=True))
    if dataclasses.is_dataclass(value):
        return stripped_json(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): stripped_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [stripped_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str) and value.startswith("data:image/"):
        return "<base64-image-removed>"
    return value


def completion_characters(response: Any) -> str:
    output = []
    for item in response.output:
        item_type = getattr(item, "type", None)
        if item_type == "reasoning":
            continue
        if isinstance(item, BaseModel):
            payload = item.model_dump(mode="json", exclude_unset=True)
        else:
            payload = stripped_json(item)
        if isinstance(payload, dict):
            for key in ("id", "call_id", "status"):
                payload.pop(key, None)
        output.append(payload)
    return json.dumps(output, sort_keys=True, ensure_ascii=False)


def read_skill_catalog(skill_dir: Path) -> list[dict[str, str]]:
    skills = []
    for path in sorted(skill_dir.glob("*.md")):
        if path.name in {"META-SKILL.md", "ROOT-SKILL.md"}:
            continue
        lines = path.read_text().splitlines()
        if not lines or lines[0] != "---":
            raise ValueError(f"missing frontmatter: {path}")
        name = next(line[6:].strip() for line in lines if line.startswith("name: "))
        description = next(
            line[13:].strip() for line in lines if line.startswith("description: ")
        )
        skills.append({"name": name, "description": description, "filename": path.name})
    return skills


def execute_sql(
    database: Path,
    sql: str,
    row_limit: int | None,
    parameters: list[Any] | None,
    string_limit: int | None,
) -> dict[str, Any]:
    normalized = sql.lstrip().lower()
    if not normalized.startswith(("select ", "with ", "describe ", "show ")):
        raise ValueError("only read-only SELECT, WITH, DESCRIBE, or SHOW statements are allowed")
    connection = duckdb.connect(
        str(database),
        read_only=True,
        config={"enable_external_access": "false", "allow_unsigned_extensions": "false"},
    )
    timer = threading.Timer(30, connection.interrupt)
    timer.start()
    try:
        cursor = connection.execute(sql, parameters)
        columns = [column[0] for column in cursor.description]
        if row_limit is None:
            raw_rows = cursor.fetchall()
            has_more = False
        else:
            raw_rows = cursor.fetchmany(row_limit + 1)
            has_more = len(raw_rows) > row_limit
            raw_rows = raw_rows[:row_limit]
        rows = [
            {
                column: json_value(value, string_limit)
                for column, value in zip(columns, row)
            }
            for row in raw_rows
        ]
        return {"columns": columns, "rows": rows, "has_more": has_more}
    finally:
        timer.cancel()
        connection.close()


def save_trace(context: HarnessContext, label: str, result: Any) -> str:
    path = context.run_dir / "traces" / f"{label}-segment-{context.trace_counter:02d}.json"
    while path.exists():
        context.trace_counter += 1
        path = context.run_dir / "traces" / f"{label}-segment-{context.trace_counter:02d}.json"
    context.trace_counter += 1
    raw_responses = []
    for response in result.raw_responses:
        raw_responses.append(
            {
                "output": stripped_json(response.output),
                "usage": stripped_json(response.usage),
                "response_id": response.response_id,
                "request_id": response.request_id,
                "raw_usage": response.raw_usage,
            }
        )
    payload = {
        "input": stripped_json(result.input),
        "history": stripped_json(result.to_input_list()),
        "raw_responses": raw_responses,
        "final_output": stripped_json(result.final_output),
        "is_complete": result.is_complete,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return str(path)


async def run_stream(
    context: HarnessContext,
    agent: Agent,
    input_items: str | list[dict[str, Any]],
    label: str,
    watch_context: bool,
) -> tuple[Any, bool, str]:
    result = Runner.run_streamed(
        agent,
        input_items,
        max_turns=MAX_TURNS,
        run_config=context.run_config,
    )
    context_triggered = False
    caught: BaseException | None = None
    try:
        async for event in result.stream_events():
            if event.type != "raw_response_event":
                continue
            if getattr(event.data, "type", None) != "response.completed":
                continue
            response = event.data.response
            if response.usage is None:
                raise RuntimeError("completed response did not include usage")
            remaining = (
                CONTEXT_LIMIT
                - response.usage.input_tokens
                - response.usage.output_tokens
            )
            if watch_context and remaining <= COMPACTION_THRESHOLD:
                result.cancel(mode="after_turn")
                context_triggered = True
    except BaseException as error:
        caught = error
    trace_file = save_trace(context, label, result)
    if caught is not None:
        raise caught
    if result.run_loop_exception is not None:
        raise result.run_loop_exception
    completions = [completion_characters(response) for response in result.raw_responses]
    for index in range(2, len(completions)):
        if completions[index] == completions[index - 1] == completions[index - 2]:
            raise InvalidTrace("three consecutive non-reasoning completions were identical")
    return result, context_triggered, trace_file


async def run_one_call(
    context: HarnessContext,
    name: str,
    instructions: str,
    prompt: str,
) -> str:
    agent = Agent(
        name=name,
        instructions=instructions,
        model=context.model,
        model_settings=context.model_settings,
        tools=[],
    )
    result, _, _ = await run_stream(context, agent, prompt, name, False)
    return str(result.final_output)


def source_for_time(context: HarnessContext, time_s: float) -> dict[str, Any]:
    for source in context.sources:
        if source["timeline_start_s"] <= time_s < source["timeline_end_s"]:
            return source
    if time_s == context.sources[-1]["timeline_end_s"]:
        return context.sources[-1]
    raise ValueError(f"no source media at time {time_s}")


async def extract_evidence_frame(
    context: HarnessContext, time_s: float, call_id: str
) -> Path:
    source = source_for_time(context, time_s)
    local_time = time_s - source["timeline_start_s"]
    output = (
        context.run_dir
        / "media"
        / f"evidence-{time_s:.3f}-{call_id}-{uuid.uuid4().hex}.jpg"
    )
    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{local_time:.6f}",
        "-i",
        source["path"],
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(stderr.decode())
    with Image.open(output) as image:
        image.load()
    return output


async def sample_contact_sheets(
    context: HarnessContext,
    clip_start_s: float,
    clip_end_s: float,
    start_s: float,
    end_s: float,
    fps: float,
    include_captions: bool,
    crop_left: int,
    crop_top: int,
    crop_right: int,
    crop_bottom: int,
    call_id: str,
) -> list[ToolOutputText | ToolOutputImage]:
    if not clip_start_s <= start_s < end_s <= clip_end_s:
        raise ValueError("sample interval must lie inside the clip_qa interval")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if not (
        0 <= crop_left < crop_right <= 1000
        and 0 <= crop_top < crop_bottom <= 1000
    ):
        raise ValueError("crop coordinates must satisfy 0 <= left < right <= 1000 and 0 <= top < bottom <= 1000")
    media_root = context.run_dir / "media"
    frame_records: list[tuple[float, Path]] = []
    with tempfile.TemporaryDirectory(dir=media_root) as temporary:
        temporary_root = Path(temporary)
        for source_index, source in enumerate(context.sources):
            overlap_start = max(start_s, source["timeline_start_s"])
            overlap_end = min(end_s, source["timeline_end_s"])
            if overlap_start >= overlap_end:
                continue
            local_start = overlap_start - source["timeline_start_s"]
            duration = overlap_end - overlap_start
            pattern = temporary_root / f"{source_index:06d}-%08d.jpg"
            video_filter = (
                f"crop=iw*{crop_right - crop_left}/1000:"
                f"ih*{crop_bottom - crop_top}/1000:"
                f"iw*{crop_left}/1000:ih*{crop_top}/1000,"
                f"scale=480:-2,fps={fps}"
            )
            process = await asyncio.create_subprocess_exec(
                "ffmpeg",
                "-v",
                "error",
                "-ss",
                f"{local_start:.6f}",
                "-t",
                f"{duration:.6f}",
                "-i",
                source["path"],
                "-vf",
                video_filter,
                "-q:v",
                "3",
                str(pattern),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await process.communicate()
            if process.returncode != 0:
                raise RuntimeError(stderr.decode())
            source_frames = sorted(temporary_root.glob(f"{source_index:06d}-*.jpg"))
            for frame_index, frame_path in enumerate(source_frames):
                frame_records.append(
                    (min(overlap_start + frame_index / fps, overlap_end), frame_path)
                )
        if not frame_records:
            raise RuntimeError("ffmpeg returned no frames")
        outputs: list[ToolOutputText | ToolOutputImage] = []
        page_paths = []
        for page_index in range(math.ceil(len(frame_records) / 12)):
            canvas = Image.new("RGB", (1920, 900), "black")
            draw = ImageDraw.Draw(canvas)
            page_records = frame_records[page_index * 12 : (page_index + 1) * 12]
            for cell_index, (time_s, frame_path) in enumerate(page_records):
                with Image.open(frame_path) as frame:
                    rendered = ImageOps.contain(frame.convert("RGB"), (480, 270))
                x = (cell_index % 4) * 480 + (480 - rendered.width) // 2
                y = (cell_index // 4) * 300 + (270 - rendered.height) // 2
                canvas.paste(rendered, (x, y))
                draw.text(
                    ((cell_index % 4) * 480 + 8, (cell_index // 4) * 300 + 276),
                    f"{time_s:.3f}s",
                    fill="white",
                )
            page_path = (
                media_root
                / (
                    f"contact-{start_s:.3f}-{end_s:.3f}-fps{fps:.4f}-"
                    f"crop{crop_left}-{crop_top}-{crop_right}-{crop_bottom}-"
                    f"captions{int(include_captions)}-{call_id}-page{page_index:04d}.jpg"
                )
            )
            canvas.save(page_path, quality=90)
            with Image.open(page_path) as rendered_page:
                rendered_page.load()
            page_paths.append(page_path)
        captions: list[dict[str, Any]] = []
        if include_captions:
            caption_result = execute_sql(
                context.database,
                (
                    "SELECT row_id, start_s, end_s, text FROM episodes "
                    "WHERE granularity = '30sec' "
                    f"AND end_s > {float(start_s)} AND start_s < {float(end_s)} "
                    "ORDER BY start_s"
                ),
                None,
                None,
                None,
            )
            captions = caption_result["rows"]
        outputs.append(
            ToolOutputText(
                text=json.dumps(
                    {
                        "pages": [path.name for path in page_paths],
                        "frame_count": len(frame_records),
                        "captions": captions,
                    },
                    ensure_ascii=False,
                )
            )
        )
        for page_path in page_paths:
            encoded = base64.b64encode(page_path.read_bytes()).decode()
            if not encoded:
                raise RuntimeError(f"empty image encoding: {page_path}")
            outputs.append(
                ToolOutputImage(
                    image_url=f"data:image/jpeg;base64,{encoded}",
                    detail="high",
                )
            )
        return outputs


def make_tools(context: HarnessContext) -> tuple[list[Any], Any, Any]:
    @function_tool(failure_error_function=tool_error_json)
    async def run_sql(sql: str) -> str:
        """Execute one read-only SQL query. Returns at most 100 rows."""
        result = await asyncio.to_thread(
            execute_sql, context.database, sql, 100, None, 500
        )
        return json.dumps(result, ensure_ascii=False)

    @function_tool(failure_error_function=tool_error_json)
    async def sql_qa(sql: str, question: str) -> str:
        """Execute a broad read-only SQL query and have one model call answer from its result."""
        result = await asyncio.to_thread(
            execute_sql, context.database, sql, None, None, None
        )
        serialized = json.dumps(result, ensure_ascii=False)
        if len(serialized) > 50_000:
            serialized = serialized[:50_000] + "\n[SQL result truncated at 50000 characters]"
        return await run_one_call(
            context,
            "sql-qa",
            "Answer only from the supplied SQL result. State when it does not establish the answer.",
            normalize_text(question) + "\n\nSQL result:\n" + serialized,
        )

    def semantic_search_error(tool_context: Any, error: Exception) -> str:
        return json.dumps(
            {
                "error": str(error),
                "arguments": tool_context.tool_arguments,
                "expected_schema": semantic_search.params_json_schema,
                "correction": (
                    "For argument validation errors, correct the arguments and retry semantic_search. "
                    "Use JSON null for unbounded min_score/max_score, not the string 'None'. "
                    "Use one of the table names in the schema."
                ),
            },
            ensure_ascii=False,
        )

    @function_tool(failure_error_function=semantic_search_error)
    async def semantic_search(
        query: str,
        table: Literal[
            "episodes", "entities", "semantic_triples", "topic_facts",
            "event_steps", "object_states", "visual_clips",
        ],
        n: int,
        rank_start: int,
        min_score: float | None = None,
        max_score: float | None = None,
    ) -> str:
        """Search one embedded table. rank_start is one-based and n is the number of ranks.

        Args:
            query: Semantic search text.
            table: One of the embedded tables listed in the schema.
            n: Number of ranks to return, greater than zero.
            rank_start: First rank to return, starting at one.
            min_score: Inclusive lower score bound; JSON null means no lower bound.
            max_score: Inclusive upper score bound; JSON null means no upper bound.
        """
        if table not in SEARCH_TABLES:
            raise ValueError(f"table must be one of {sorted(SEARCH_TABLES)}")
        if n <= 0 or rank_start <= 0:
            raise ValueError("n and rank_start must be positive")
        async with httpx.AsyncClient(timeout=None) as embedding_client:
            response = await embedding_client.post(
                context.embedding_url.rstrip("/") + "/embeddings",
                json={"model": context.embedding_model, "input": [normalize_text(query)]},
            )
        response.raise_for_status()
        data = response.json()["data"]
        if len(data) != 1 or data[0]["index"] != 0:
            raise RuntimeError(f"invalid embedding response indexes: {data}")
        query_vector = np.asarray(data[0]["embedding"], dtype=np.float32)
        if query_vector.shape != (4096,) or not np.isfinite(query_vector).all():
            raise RuntimeError(f"invalid query vector shape or values: {query_vector.shape}")
        scores = context.vectors[table] @ query_vector
        positions = np.argsort(-scores)
        if min_score is not None:
            positions = positions[scores[positions] >= min_score]
        if max_score is not None:
            positions = positions[scores[positions] <= max_score]
        selected = positions[rank_start - 1 : rank_start - 1 + n]
        selected_ids = [str(context.row_ids[table][position]) for position in selected]
        if not selected_ids:
            return json.dumps({"table": table, "hits": []})
        placeholders = ",".join("?" for _ in selected_ids)
        rows = execute_sql(
            context.database,
            f"SELECT * FROM {table} WHERE row_id IN ({placeholders})",
            None,
            selected_ids,
            500,
        )
        by_id = {row["row_id"]: row for row in rows["rows"]}
        hits = []
        for position, row_id in zip(selected, selected_ids):
            row = by_id[row_id]
            preview_field = SEARCH_TABLES[table]
            if table == "entities":
                start_s = row["first_seen_s"]
                end_s = row["last_seen_s"]
            elif table == "semantic_triples":
                start_s = row["snapshot_s"]
                end_s = row["snapshot_s"]
            elif table == "topic_facts":
                start_s = row["time_s"]
                end_s = row["time_s"]
            else:
                start_s = row["start_s"]
                end_s = row["end_s"]
            hits.append(
                {
                    "table": table,
                    "row_id": row_id,
                    "start_s": start_s,
                    "end_s": end_s,
                    "preview": json_value(row[preview_field], 500),
                    "score": float(scores[position]),
                }
            )
        return json.dumps({"table": table, "hits": hits}, ensure_ascii=False)

    @function_tool(failure_error_function=tool_error_json)
    async def add_note(note: str) -> str:
        """Append a durable investigation note."""
        record = {"note": normalize_text(note)}
        async with context.notes_lock:
            with (context.run_dir / "agent_notes.jsonl").open("a") as output:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
        return json.dumps({"saved": record})

    @function_tool(failure_error_function=tool_error_json)
    async def read_notes() -> str:
        """Read all durable investigation notes."""
        path = context.run_dir / "agent_notes.jsonl"
        if not path.exists():
            return "[]"
        return json.dumps(
            [json.loads(line) for line in path.read_text().splitlines()],
            ensure_ascii=False,
        )

    catalog = read_skill_catalog(context.skill_dir)
    filenames = {skill["name"]: skill["filename"] for skill in catalog}

    @function_tool(failure_error_function=tool_error_json)
    async def read_skill(name: str) -> str:
        """Read the complete Markdown for one listed skill."""
        if name not in filenames:
            raise ValueError(f"unknown skill: {name}")
        return (context.skill_dir / filenames[name]).read_text()

    @function_tool(failure_error_function=None)
    async def agent_raise(reason: str) -> str:
        """End the current rollout because a required dependency or valid path is unavailable."""
        raise AgentRaised(reason)

    @function_tool(failure_error_function=None)
    async def submit_answer(answer: str) -> str:
        """Submit the final answer and stop the retrieval loop."""
        return AnswerSubmission(answer=answer).model_dump_json()

    @function_tool(failure_error_function=tool_error_json)
    async def clip_qa(start_s: float, end_s: float, question: str) -> list[Any]:
        """Inspect at most 20 minutes of source media and return an answer with evidence images."""
        if start_s < 0 or end_s <= start_s:
            raise ValueError("clip interval must be non-negative and increasing")
        if end_s - start_s > 1200:
            raise ValueError("clip_qa rejects intervals longer than 20 minutes")
        call_id = uuid.uuid4().hex

        @function_tool(failure_error_function=tool_error_json)
        async def sample_video(
            sample_start_s: float,
            sample_end_s: float,
            fps: float,
            include_captions: bool,
            crop_left: int,
            crop_top: int,
            crop_right: int,
            crop_bottom: int,
        ) -> list[Any]:
            """Return fully paginated 4 by 3 contact sheets for a subinterval."""
            return await sample_contact_sheets(
                context,
                start_s,
                end_s,
                sample_start_s,
                sample_end_s,
                fps,
                include_captions,
                crop_left,
                crop_top,
                crop_right,
                crop_bottom,
                call_id,
            )

        @function_tool(failure_error_function=tool_error_json)
        async def add_clip_note(note: str) -> str:
            """Add a note for this visual investigation."""
            path = context.run_dir / f"clip-notes-{call_id}.jsonl"
            with path.open("a") as output:
                output.write(json.dumps({"note": normalize_text(note)}) + "\n")
            return json.dumps({"saved": True})

        @function_tool(failure_error_function=tool_error_json)
        async def read_clip_notes() -> str:
            """Read notes for this visual investigation."""
            path = context.run_dir / f"clip-notes-{call_id}.jsonl"
            if not path.exists():
                return "[]"
            return json.dumps([json.loads(line) for line in path.read_text().splitlines()])

        @function_tool(failure_error_function=tool_error_json)
        async def submit_clip_answer(
            status: Literal["answered", "unknown"],
            answer: str,
            evidence_times_s: list[float],
        ) -> str:
            """Submit the visual answer and stop the visual loop."""
            if status == "answered" and not evidence_times_s:
                raise ValueError("answered requires at least one evidence timestamp")
            if any(not start_s <= time_s <= end_s for time_s in evidence_times_s):
                raise ValueError("evidence timestamps must lie inside the clip interval")
            return ClipSubmission(
                status=status,
                answer=answer,
                evidence_times_s=evidence_times_s,
            ).model_dump_json()

        clip_agent = Agent(
            name="clip-qa",
            instructions=(
                "Answer the visual question by sampling the supplied interval. "
                "Use sample_video, notes when useful, and submit_clip_answer. "
                "Do not infer an answered result without visible evidence."
            ),
            model=context.model,
            model_settings=context.model_settings,
            tools=[sample_video, add_clip_note, read_clip_notes, submit_clip_answer],
            tool_use_behavior=StopAtTools(stop_at_tool_names=["submit_clip_answer"]),
        )
        clip_result, _, _ = await run_stream(
            context,
            clip_agent,
            (
                f"Interval: {start_s:.3f}s to {end_s:.3f}s\n"
                f"Visual question: {normalize_text(question)}"
            ),
            f"clip-{call_id}",
            False,
        )
        submission = ClipSubmission.model_validate_json(str(clip_result.final_output))
        outputs: list[Any] = [
            ToolOutputText(
                text=json.dumps(
                    {
                        "status": submission.status,
                        "answer": submission.answer,
                        "evidence_times_s": submission.evidence_times_s,
                    },
                    ensure_ascii=False,
                )
            )
        ]
        if submission.status == "answered":
            for time_s in submission.evidence_times_s:
                evidence = await extract_evidence_frame(context, time_s, call_id)
                outputs.append(
                    ToolOutputText(
                        text=json.dumps(
                            {"evidence_time_s": time_s, "filename": evidence.name}
                        )
                    )
                )
                encoded = base64.b64encode(evidence.read_bytes()).decode()
                if not encoded:
                    raise RuntimeError(f"empty evidence image encoding: {evidence}")
                outputs.append(
                    ToolOutputImage(
                        image_url=f"data:image/jpeg;base64,{encoded}",
                        detail="high",
                    )
                )
        return outputs

    answer_tools = [
        run_sql,
        sql_qa,
        semantic_search,
        clip_qa,
        add_note,
        read_notes,
        read_skill,
        agent_raise,
        submit_answer,
    ]
    return answer_tools, submit_answer, catalog


class VideoSQLHarness:
    def __init__(
        self,
        database: Path,
        embedding_index: Path,
        media_map: Path,
        skill_dir: Path,
        run_dir: Path,
        model_name: str,
        api_key: str,
        base_url: str | None,
        embedding_url: str,
        embedding_model: str,
        reasoning_effort: Literal["low", "medium", "high"],
    ) -> None:
        client_arguments: dict[str, Any] = {"api_key": api_key, "max_retries": 3}
        if base_url is not None:
            client_arguments["base_url"] = base_url
        client = AsyncOpenAI(**client_arguments)
        model = OpenAIResponsesModel(model=model_name, openai_client=client)
        model_settings = ModelSettings(
            reasoning={"effort": reasoning_effort},
            response_include=["reasoning.encrypted_content"],
            preserve_raw_usage=True,
        )
        run_config = RunConfig(
            tracing_disabled=True,
            reasoning_item_id_policy="preserve",
        )
        self.context = HarnessContext(
            database=database,
            embedding_index=embedding_index,
            media_map=media_map,
            skill_dir=skill_dir,
            run_dir=run_dir,
            model=model,
            client=client,
            model_settings=model_settings,
            run_config=run_config,
            embedding_url=embedding_url,
            embedding_model=embedding_model,
        )
        self.answer_tools, self.submit_answer_tool, self.catalog = make_tools(self.context)

    async def answer(self, question: str) -> RolloutSession:
        base_instructions = (
            "Answer the user's question by retrieving evidence from the memory database. "
            "Use submit_answer when finished. The complete database schema is:\n" + SCHEMA
            + "\n\nROOT-SKILL (skill loading guidance):\n"
            + (self.context.skill_dir / "ROOT-SKILL.md").read_text()
        )
        agent = Agent(
            name="memory-retrieval",
            instructions=base_instructions,
            model=self.context.model,
            model_settings=self.context.model_settings,
            tools=self.answer_tools,
            tool_use_behavior=StopAtTools(stop_at_tool_names=["submit_answer"]),
        )
        skills = "\n".join(
            f"- {skill['name']}: {skill['description']}" for skill in self.catalog
        )
        input_items: str | list[dict[str, Any]] = (
            f"Question: {normalize_text(question)}\n\nAvailable skills:\n{skills}"
        )
        trace_files = []
        compactions = 0
        while True:
            result, needs_compaction, trace_file = await run_stream(
                self.context,
                agent,
                input_items,
                "answer",
                True,
            )
            trace_files.append(trace_file)
            history = result.to_input_list()
            if not needs_compaction:
                submission = AnswerSubmission.model_validate_json(str(result.final_output))
                return RolloutSession(
                    agent=agent,
                    history=history,
                    answer=submission.answer,
                    status="answered",
                    trace_files=trace_files,
                )
            compactions += 1
            original_tools = agent.tools
            original_behavior = agent.tool_use_behavior
            agent.tools = []
            agent.tool_use_behavior = "run_llm_again"
            compact_result, _, compact_trace = await run_stream(
                self.context,
                agent,
                history
                + [{"role": "user", "content": COMPACTION_INSTRUCTION}],
                "compaction",
                False,
            )
            trace_files.append(compact_trace)
            handoff = str(compact_result.final_output)
            if compactions == 3:
                return RolloutSession(
                    agent=agent,
                    history=compact_result.to_input_list(),
                    answer=None,
                    status="compaction_limit",
                    trace_files=trace_files,
                )
            agent.tools = original_tools
            agent.tool_use_behavior = original_behavior
            agent.instructions = (
                base_instructions
                + "\n\nTrusted handoff from the same agent's earlier context:\n"
                + handoff
            )
            input_items = "Continue the investigation from the trusted handoff."

    async def reflect(
        self,
        session: RolloutSession,
        reference_answer: str,
        score: int,
    ) -> ReflectionSubmission:
        tool_calls = 0

        class ReflectionHooks(AgentHooks):
            async def on_tool_end(self, context, agent, tool, result) -> None:
                nonlocal tool_calls
                if tool.name not in {"submit_reflection", "agent_raise"}:
                    tool_calls += 1

        @function_tool(
            failure_error_function=None,
            is_enabled=lambda context, agent: tool_calls >= 2,
        )
        async def submit_reflection(reflection: str, meta_reflection: str) -> str:
            """Submit both abstract reflections and stop the reflection loop."""
            return ReflectionSubmission(
                reflection=reflection,
                meta_reflection=meta_reflection,
            ).model_dump_json()

        session.agent.tools = [
            tool
            for tool in self.answer_tools
            if tool.name != self.submit_answer_tool.name
        ] + [submit_reflection]
        session.agent.tool_use_behavior = StopAtTools(
            stop_at_tool_names=["submit_reflection"]
        )
        session.agent.hooks = ReflectionHooks()
        session.agent.model_settings = dataclasses.replace(
            self.context.model_settings, tool_choice="required"
        )
        session.agent.reset_tool_choice = False
        reflection_tools = session.agent.tools
        reflection_behavior = session.agent.tool_use_behavior
        meta_skill = (self.context.skill_dir / "META-SKILL.md").read_text()
        if session.status == "compaction_limit":
            reflection_prompt = (
                f"Reference answer: {reference_answer}\nScore out of 5: 0\n"
                "The rollout ended after its third compaction and received score 0. "
                "Focus on why the investigation looped. "
            )
        else:
            reflection_prompt = (
                f"Reference answer: {reference_answer}\nScore out of 5: {score}\n"
            )
        reflection_prompt += (
            "Reflect on your own answer process in this same conversation. "
            "During reflection, make at least two tool calls to investigate and check your "
            "conclusions before submitting. Earlier answer-stage calls do not count. "
            "Use the returned evidence rather than the reference answer alone to diagnose "
            "the process. Submit abstract reflection and "
            "meta-reflection.\n\nMETA-SKILL:\n"
            + meta_skill
        )
        base_instructions = session.agent.instructions
        input_items = session.history + [{"role": "user", "content": reflection_prompt}]
        compactions = 0
        while True:
            result, threshold, _ = await run_stream(
                self.context,
                session.agent,
                input_items,
                "reflection",
                True,
            )
            if result.final_output is not None:
                if tool_calls < 2:
                    raise InvalidTrace("reflection ended before two reflection-stage tool calls")
                return ReflectionSubmission.model_validate_json(str(result.final_output))
            if not threshold:
                raise InvalidTrace("reflection ended without submit_reflection")
            session.agent.tools = []
            session.agent.tool_use_behavior = "run_llm_again"
            session.agent.model_settings = self.context.model_settings
            compact_result, _, _ = await run_stream(
                self.context,
                session.agent,
                result.to_input_list() + [{
                    "role": "user",
                    "content": (
                        "Stop calling tools. Compact this reflection into a handoff preserving "
                        "the question, submitted/reference answers, score, original reasoning "
                        "and tool evidence, reflection-stage checks and returned evidence, "
                        "competing explanations, unresolved issues, and next checks. "
                        "Do not submit the final reflection yet."
                    ),
                }],
                "reflection-compaction",
                False,
            )
            compactions += 1
            session.agent.instructions = (
                base_instructions + "\n\nReflection handoff:\n"
                + str(compact_result.final_output)
            )
            session.agent.tool_use_behavior = reflection_behavior
            session.agent.model_settings = dataclasses.replace(
                self.context.model_settings, tool_choice="required"
            )
            if compactions == 3:
                if tool_calls < 2:
                    raise InvalidTrace("reflection exhausted three compactions before two tool calls")
                session.agent.tools = [submit_reflection]
                forced, _, _ = await run_stream(
                    self.context,
                    session.agent,
                    reflection_prompt + "\n\n" + FORCED_REFLECTION_INSTRUCTION,
                    "reflection-forced",
                    False,
                )
                return ReflectionSubmission.model_validate_json(str(forced.final_output))
            session.agent.tools = reflection_tools
            input_items = [{
                "role": "user",
                "content": reflection_prompt + f"\nReflection tool calls completed: {tool_calls}. "
                "Continue from the reflection handoff.",
            }]

    async def close(self) -> None:
        await self.context.client.close()
