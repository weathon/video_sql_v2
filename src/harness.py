import asyncio
import base64
import dataclasses
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
from agents import Agent, ModelSettings, OpenAIResponsesModel, RunConfig, RunContextWrapper, Runner, ToolOutputImage, ToolOutputText, function_tool
from agents.exceptions import MaxTurnsExceeded
from agents.result import RunResultStreaming
from openai import AsyncOpenAI
from openai.types.shared import Reasoning
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pydantic import BaseModel


DATABASE_CONFIG = {"enable_external_access": "false"}
QUERY_ERRORS = (duckdb.BinderException, duckdb.CatalogException, duckdb.ConstraintException, duckdb.ConversionException, duckdb.InvalidInputException, duckdb.InvalidTypeException, duckdb.NotImplementedException, duckdb.OutOfRangeException, duckdb.ParserException, duckdb.PermissionException, duckdb.SequenceException, duckdb.SyntaxException, duckdb.TransactionException, duckdb.TypeMismatchException)
COMPACTION_INSTRUCTION = "Stop calling tools. Produce a trusted handoff for a fresh continuation of this same retrieval task. Preserve the exact question, all useful evidence with video IDs and timestamps, SQL findings, unresolved uncertainty, notes, and the next retrieval actions. Do not answer the question unless the evidence already supports the answer."


class AgentRaised(RuntimeError):
    pass


class ClipAnswer(BaseModel):
    status: Literal["answered", "unknown"]
    answer: str
    evidence_timestamps_s: list[float]


@dataclass
class SearchIndex:
    row_ids: np.ndarray
    vectors: np.ndarray


@dataclass
class RolloutResult:
    status: Literal["completed", "compaction_limit"]
    answer: str
    turns: int
    compactions: int
    usage: dict[str, int]


@dataclass
class RolloutContext:
    runtime: "PilotHarness"
    attempt_dir: Path
    clip_max_turns: int
    notes: list[str] = field(default_factory=list)
    note_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass
class ClipContext:
    rollout: RolloutContext
    video_id: str
    start_s: float
    end_s: float
    call_id: str


def open_database(database_path: Path) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(database_path), read_only=True, config=DATABASE_CONFIG)


def root_context(value: RolloutContext | ClipContext) -> RolloutContext:
    return value.rollout if isinstance(value, ClipContext) else value


def json_sanitize(value: Any) -> Any:
    if isinstance(value, str):
        return "<image_data_removed>" if value.startswith("data:image/") else value
    if isinstance(value, Path):
        return value.name
    if isinstance(value, BaseModel):
        return json_sanitize(value.model_dump(mode="json"))
    if dataclasses.is_dataclass(value):
        return json_sanitize(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): json_sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_sanitize(item) for item in value]
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    return value


def write_trace(result: RunResultStreaming, events: list[Any], trace_path: Path) -> None:
    responses = [{"response_id": response.response_id, "request_id": response.request_id, "usage": json_sanitize(response.usage), "output": json_sanitize(response.output)} for response in result.raw_responses]
    items = [json_sanitize(item.to_input_item()) for item in result.new_items]
    payload = {"is_complete": result.is_complete, "current_turn": result.current_turn, "responses": responses, "items": items, "events": events, "usage": json_sanitize(result.context_wrapper.usage), "final_output": json_sanitize(result.final_output)}
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


async def stream_agent(agent: Agent, input_items: str | list[dict[str, Any]], context: RolloutContext | ClipContext, max_turns: int, run_config: RunConfig, trace_path: Path, context_limit: int, allow_compaction: bool) -> tuple[RunResultStreaming, bool]:
    result = Runner.run_streamed(agent, input_items, context=context, max_turns=max_turns, run_config=run_config)
    events = []
    compact_required = False
    try:
        async for event in result.stream_events():
            if event.type == "raw_response_event":
                events.append(json_sanitize(event.data.model_dump(mode="json", exclude_none=True)))
                if allow_compaction and not compact_required and event.data.type == "response.completed":
                    response = event.data.response
                    remaining = context_limit - response.usage.input_tokens - response.usage.output_tokens
                    has_tool_calls = any(item.type == "function_call" for item in response.output)
                    if remaining <= 50000 and has_tool_calls:
                        result.cancel(mode="after_turn")
                        compact_required = True
    finally:
        write_trace(result, events, trace_path)
    return result, compact_required


def render_frame(runtime: "PilotHarness", video_id: str, timestamp_s: float, crop_left: int, crop_top: int, crop_right: int, crop_bottom: int) -> Image.Image:
    connection = open_database(runtime.database_path)
    row = connection.execute("SELECT row_id, start_s, end_s FROM visual_clips WHERE video_id = ? AND start_s <= ? AND end_s >= ? ORDER BY start_s DESC LIMIT 1", [video_id, timestamp_s, timestamp_s]).fetchone()
    connection.close()
    if row is None:
        raise RuntimeError(f"No cached visual clip covers video {video_id} at {timestamp_s}")
    row_id, clip_start_s, _ = row
    clip_index = int(row_id.rsplit(":", 1)[1])
    clip_path = runtime.clip_root / video_id / f"{clip_index:06d}.mp4"
    with tempfile.TemporaryDirectory(dir="/tmp") as temporary_dir:
        frame_path = Path(temporary_dir) / "frame.png"
        result = subprocess.run(["ffmpeg", "-v", "error", "-i", str(clip_path), "-ss", str(timestamp_s - clip_start_s), "-frames:v", "1", str(frame_path)], capture_output=True, text=True)
        if result.returncode != 0:
            print(json.dumps({"command": "ffmpeg", "clip_path": str(clip_path), "timestamp_s": timestamp_s, "stderr": result.stderr}))
            result.check_returncode()
        with Image.open(frame_path) as loaded:
            image = loaded.convert("RGB")
            image.load()
    width, height = image.size
    return image.crop((crop_left * width // 1000, crop_top * height // 1000, math.ceil(crop_right * width / 1000), math.ceil(crop_bottom * height / 1000)))


def render_contact_sheets(context: ClipContext, start_s: float, end_s: float, fps: float, crop_left: int, crop_top: int, crop_right: int, crop_bottom: int, include_captions: bool) -> tuple[list[dict[str, Any]], list[Path]]:
    runtime = context.rollout.runtime
    timestamps = np.arange(start_s, end_s, 1.0 / fps).tolist()
    frames = [render_frame(runtime, context.video_id, timestamp_s, crop_left, crop_top, crop_right, crop_bottom) for timestamp_s in timestamps]
    connection = open_database(runtime.database_path)
    captions = connection.execute("SELECT row_id, start_s, end_s, text FROM episodes WHERE video_id = ? AND granularity = '30sec' AND start_s < ? AND end_s > ? ORDER BY start_s, row_id", [context.video_id, end_s, start_s]).fetchall() if include_captions else []
    connection.close()
    caption_rows = [{"row_id": row_id, "start_s": caption_start_s, "end_s": caption_end_s, "text": text} for row_id, caption_start_s, caption_end_s, text in captions]
    sheets = []
    font = ImageFont.load_default(size=20)
    for page_index in range(0, len(frames), 12):
        canvas = Image.new("RGB", (1920, 900), "white")
        draw = ImageDraw.Draw(canvas)
        for cell_index, frame in enumerate(frames[page_index:page_index + 12]):
            contained = ImageOps.contain(frame, (470, 260), Image.Resampling.LANCZOS)
            column = cell_index % 4
            row = cell_index // 4
            x = column * 480 + (480 - contained.width) // 2
            y = row * 300 + 30 + (260 - contained.height) // 2
            canvas.paste(contained, (x, y))
            draw.text((column * 480 + 8, row * 300 + 5), f"{timestamps[page_index + cell_index]:.3f}s", fill="black", font=font)
        filename = f"contact-{context.video_id}-{start_s:.3f}-{end_s:.3f}-fps{fps:.4f}-crop{crop_left}-{crop_top}-{crop_right}-{crop_bottom}-captions{int(include_captions)}-{context.call_id}-page{page_index // 12:04d}.jpg"
        sheet_path = context.rollout.attempt_dir / "media" / filename
        sheet_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(sheet_path, format="JPEG", quality=90)
        sheets.append(sheet_path)
    return caption_rows, sheets


def image_output(path: Path) -> ToolOutputImage:
    image_bytes = path.read_bytes()
    if not image_bytes:
        raise RuntimeError(f"Generated image is empty: {path}")
    return ToolOutputImage(image_url="data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("ascii"), detail="high")


@function_tool(failure_error_function=None)
async def run_sql(context: RunContextWrapper[RolloutContext], sql: str) -> str:
    """Execute one read-only DuckDB query. Returns at most 100 rows; each cell is truncated to 500 characters with its original length."""
    connection = open_database(context.context.runtime.database_path)

    def execute_query() -> tuple[list[str], list[tuple[Any, ...]]]:
        try:
            cursor = connection.execute(sql)
            columns = [description[0] for description in cursor.description]
            return columns, cursor.fetchmany(101)
        finally:
            connection.close()

    task = asyncio.create_task(asyncio.to_thread(execute_query))
    try:
        columns, rows = await asyncio.wait_for(asyncio.shield(task), timeout=30)
    except TimeoutError:
        connection.interrupt()
        try:
            await task
        except duckdb.InterruptException:
            pass
        return json.dumps({"error": "SQL query exceeded the 30 second timeout"})
    except QUERY_ERRORS as error:
        return json.dumps({"error": str(error)})
    output_rows = []
    for row in rows[:100]:
        output_row = []
        for cell in row:
            text = "NULL" if cell is None else str(cell)
            output_row.append({"text": text[:500], "length": len(text)})
        output_rows.append(output_row)
    return json.dumps({"columns": columns, "rows": output_rows, "has_more": len(rows) > 100})


@function_tool(failure_error_function=None)
async def semantic_search(context: RunContextWrapper[RolloutContext], query: str, n: int, table: str) -> str:
    """Search one generated embedding table by exact cosine similarity."""
    runtime = context.context.runtime
    if table not in runtime.indexes:
        return json.dumps({"error": f"Unknown embedding table {table!r}. Allowed tables: {sorted(runtime.indexes)}"})
    if n <= 0:
        return json.dumps({"error": "n must be positive"})
    try:
        response = await runtime.embedding_client.post("/v1/embeddings", json={"model": "qwen3-vl-embedding-8b", "input": [query]})
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        print(json.dumps({"tool": "semantic_search", "table": table, "query_length": len(query), "status_code": error.response.status_code, "response": error.response.text}))
        raise
    except httpx.RequestError as error:
        print(json.dumps({"tool": "semantic_search", "table": table, "query_length": len(query), "error": str(error)}))
        raise
    data = response.json()["data"]
    if len(data) != 1 or data[0]["index"] != 0:
        raise RuntimeError("Unexpected query embedding response")
    vector = np.asarray(data[0]["embedding"], dtype=np.float32)
    if vector.shape != (4096,) or not np.isfinite(vector).all() or abs(float(np.linalg.norm(vector)) - 1.0) > 0.001:
        raise RuntimeError("Invalid query embedding")
    index = runtime.indexes[table]
    scores = await asyncio.to_thread(np.matmul, index.vectors, vector)
    result_count = min(n, len(scores))
    positions = np.argsort(scores)[::-1][:result_count] if result_count == len(scores) else np.argpartition(scores, -result_count)[-result_count:][np.argsort(scores[np.argpartition(scores, -result_count)[-result_count:]])[::-1]]
    connection = open_database(runtime.database_path)
    results = []
    for position in positions:
        row_id = str(index.row_ids[position])
        video_id, start_s, end_s, text = connection.execute(runtime.metadata_queries[table], [row_id]).fetchone()
        results.append({"table": table, "row_id": row_id, "video_id": video_id, "start_s": start_s, "end_s": end_s, "text": {"preview": text[:500], "length": len(text)}, "cosine_score": float(scores[position])})
    connection.close()
    return json.dumps(results, ensure_ascii=False)


@function_tool(failure_error_function=None)
async def add_note(context: RunContextWrapper[RolloutContext | ClipContext], note: str) -> str:
    """Append a note that can be shared between the main retrieval agent and nested clip agents."""
    rollout = root_context(context.context)
    async with rollout.note_lock:
        rollout.notes.append(note)
        notes_path = rollout.attempt_dir / "agent_notes.jsonl"
        notes_path.parent.mkdir(parents=True, exist_ok=True)
        with notes_path.open("a") as handle:
            handle.write(json.dumps({"note": note}, ensure_ascii=False) + "\n")
    return json.dumps({"saved": True, "note_count": len(rollout.notes)})


@function_tool(failure_error_function=None)
async def read_notes(context: RunContextWrapper[RolloutContext | ClipContext]) -> str:
    """Read all notes shared by the current rollout."""
    rollout = root_context(context.context)
    async with rollout.note_lock:
        return json.dumps({"notes": list(rollout.notes)}, ensure_ascii=False)


@function_tool(failure_error_function=None)
async def sample_video(context: RunContextWrapper[ClipContext], start_s: float, end_s: float, fps: float, crop_left: int, crop_top: int, crop_right: int, crop_bottom: int, include_captions: bool) -> list[ToolOutputText | ToolOutputImage]:
    """Sample the active clip interval into complete 4x3 contact-sheet pages. Crop coordinates are integers from 0 to 1000."""
    clip = context.context
    if not math.isfinite(start_s) or not math.isfinite(end_s) or start_s < clip.start_s or end_s > clip.end_s or end_s <= start_s:
        return [ToolOutputText(text=json.dumps({"error": f"Sampling interval must satisfy {clip.start_s} <= start_s < end_s <= {clip.end_s}"}))]
    if not math.isfinite(fps) or fps <= 0:
        return [ToolOutputText(text=json.dumps({"error": "fps must be positive and finite"}))]
    if not all(isinstance(value, int) for value in [crop_left, crop_top, crop_right, crop_bottom]) or not (0 <= crop_left < crop_right <= 1000 and 0 <= crop_top < crop_bottom <= 1000):
        return [ToolOutputText(text=json.dumps({"error": "Crop coordinates must be integers satisfying 0 <= left < right <= 1000 and 0 <= top < bottom <= 1000"}))]
    captions, sheet_paths = await asyncio.to_thread(render_contact_sheets, clip, start_s, end_s, fps, crop_left, crop_top, crop_right, crop_bottom, include_captions)
    metadata = {"video_id": clip.video_id, "start_s": start_s, "end_s": end_s, "fps": fps, "frames": len(np.arange(start_s, end_s, 1.0 / fps)), "contact_sheets": [path.name for path in sheet_paths], "captions": captions}
    return [ToolOutputText(text=json.dumps(metadata, ensure_ascii=False)), *[image_output(path) for path in sheet_paths]]


@function_tool(failure_error_function=None)
async def agent_raise(context: RunContextWrapper[RolloutContext], reason: str) -> str:
    """Stop the current rollout because required data is missing or a system condition makes a valid answer impossible."""
    raise AgentRaised(reason)


@function_tool(failure_error_function=None)
async def clip_qa(context: RunContextWrapper[RolloutContext], video_id: str, start_s: float, end_s: float, question: str) -> list[ToolOutputText | ToolOutputImage]:
    """Inspect one video interval with a nested visual agent. Intervals over 10 minutes are discouraged and intervals over 20 minutes are rejected."""
    rollout = context.context
    if not math.isfinite(start_s) or not math.isfinite(end_s) or start_s < 0 or end_s <= start_s:
        return [ToolOutputText(text=json.dumps({"error": "clip_qa requires finite values with 0 <= start_s < end_s"}))]
    if end_s - start_s > 1200:
        return [ToolOutputText(text=json.dumps({"error": "clip_qa refuses intervals longer than 20 minutes"}))]
    connection = open_database(rollout.runtime.database_path)
    video = connection.execute("SELECT duration_s FROM videos WHERE video_id = ?", [video_id]).fetchone()
    connection.close()
    if video is None:
        return [ToolOutputText(text=json.dumps({"error": f"Unknown video_id {video_id!r}"}))]
    if end_s > video[0]:
        return [ToolOutputText(text=json.dumps({"error": f"end_s exceeds video duration {video[0]}"}))]
    call_id = uuid.uuid4().hex
    clip_context = ClipContext(rollout=rollout, video_id=video_id, start_s=start_s, end_s=end_s, call_id=call_id)
    instructions = "Answer the visual question only from the active video interval. Use sample_video to inspect the frames and captions. You may use shared notes. Return status answered only when the answer is supported and include at least one exact evidence timestamp; otherwise return unknown."
    clip_agent = Agent(name="clip_qa", instructions=instructions, tools=[sample_video, add_note, read_notes], model=rollout.runtime.model, model_settings=rollout.runtime.model_settings, output_type=ClipAnswer)
    input_items: str | list[dict[str, Any]] = f"Video {video_id}, interval {start_s} to {end_s} seconds. Question: {question}"
    used_turns = 0
    segment = 0
    while True:
        if used_turns >= rollout.clip_max_turns:
            raise MaxTurnsExceeded(f"clip_qa exceeded {rollout.clip_max_turns} total turns")
        trace_path = rollout.attempt_dir / "traces" / f"clip-{call_id}-segment-{segment:02d}.json"
        result, _ = await stream_agent(clip_agent, input_items, clip_context, rollout.clip_max_turns - used_turns, rollout.runtime.run_config, trace_path, rollout.runtime.context_limit, False)
        used_turns += len(result.raw_responses)
        answer = result.final_output_as(ClipAnswer, raise_if_incorrect_type=True)
        validation_error = None
        if answer.status == "answered" and not answer.evidence_timestamps_s:
            validation_error = "Rejected clip answer: status answered requires at least one evidence timestamp."
        elif any(not math.isfinite(timestamp_s) or timestamp_s < start_s or timestamp_s > end_s for timestamp_s in answer.evidence_timestamps_s):
            validation_error = f"Rejected clip answer: every evidence timestamp must be finite and within [{start_s}, {end_s}]."
        if validation_error is None:
            break
        input_items = result.to_input_list() + [{"role": "user", "content": validation_error}]
        segment += 1
    evidence_paths = []
    for timestamp_s in answer.evidence_timestamps_s:
        evidence_path = rollout.attempt_dir / "media" / f"evidence-{video_id}-{timestamp_s:.3f}-{call_id}-{uuid.uuid4().hex}.jpg"
        image = await asyncio.to_thread(render_frame, rollout.runtime, video_id, timestamp_s, 0, 0, 1000, 1000)
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(evidence_path, format="JPEG", quality=95)
        evidence_paths.append(evidence_path)
    payload = {"status": answer.status, "answer": answer.answer, "evidence_timestamps_s": answer.evidence_timestamps_s, "evidence_files": [path.name for path in evidence_paths]}
    return [ToolOutputText(text=json.dumps(payload, ensure_ascii=False)), *[image_output(path) for path in evidence_paths]]


class PilotHarness:
    def __init__(self, database_path: Path, index_root: Path, clip_root: Path, model_base_url: str, model_name: str, embedding_base_url: str, context_limit: int):
        self.database_path = database_path
        self.index_root = index_root
        self.clip_root = clip_root
        self.context_limit = context_limit
        self.schema = (Path(__file__).parent / "schema.sql").read_text()
        self.model_client = AsyncOpenAI(base_url=model_base_url.rstrip("/") + "/v1", api_key=os.environ["OPENAI_API_KEY"])
        self.model = OpenAIResponsesModel(model=model_name, openai_client=self.model_client)
        self.embedding_client = httpx.AsyncClient(base_url=embedding_base_url, timeout=None)
        self.model_settings = ModelSettings(parallel_tool_calls=True, reasoning=Reasoning(effort="medium"), include_usage=True)
        self.run_config = RunConfig(tracing_disabled=True, workflow_name="video_sql_retrieval", reasoning_item_id_policy="preserve")
        self.metadata_queries = {
            "episodes": "SELECT video_id, start_s, end_s, text FROM episodes WHERE row_id = ?",
            "entities": "SELECT video_id, first_seen_s, last_seen_s, name FROM entities WHERE row_id = ?",
            "semantic_triples": "SELECT video_id, snapshot_s, snapshot_s, text FROM semantic_triples WHERE row_id = ?",
            "openie_triples": "SELECT o.video_id, e.start_s, e.end_s, o.text FROM openie_triples o JOIN episodes e ON e.row_id = o.episode_row_id WHERE o.row_id = ?",
            "topic_facts": "SELECT video_id, time_s, time_s, fact FROM topic_facts WHERE row_id = ?",
            "event_steps": "SELECT video_id, start_s, end_s, description FROM event_steps WHERE row_id = ?",
            "visual_clips": "SELECT video_id, start_s, end_s, text FROM visual_clips WHERE row_id = ?",
        }
        self.indexes = {}
        for table in self.metadata_queries:
            row_ids = np.load(index_root / f"{table}.row_ids.npy", mmap_mode="r")
            vectors = np.load(index_root / f"{table}.vectors.npy", mmap_mode="r")
            if row_ids.shape != (vectors.shape[0],) or vectors.shape[1] != 4096:
                raise RuntimeError(f"Invalid semantic index for {table}")
            self.indexes[table] = SearchIndex(row_ids=row_ids, vectors=vectors)

    async def close(self) -> None:
        await self.embedding_client.aclose()
        await self.model_client.close()

    async def run_rollout(self, question: str, skill_text: str, attempt_dir: Path, clip_max_turns: int) -> RolloutResult:
        context = RolloutContext(runtime=self, attempt_dir=attempt_dir, clip_max_turns=clip_max_turns)
        instructions = f"You answer questions about a video memory using the available read-only tools. Return only the final answer when ready.\n\nDatabase schema:\n{self.schema}\n\nCurrent retrieval skill:\n{skill_text}"
        main_agent = Agent(name="video_memory_retrieval", instructions=instructions, tools=[run_sql, semantic_search, clip_qa, add_note, read_notes, agent_raise], model=self.model, model_settings=self.model_settings)
        input_items: str | list[dict[str, Any]] = question
        total_turns = 0
        compactions = 0
        usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        segment = 0
        while True:
            if total_turns >= 50:
                raise MaxTurnsExceeded("Main retrieval agent exceeded 50 total turns")
            result, compact_required = await stream_agent(main_agent, input_items, context, 50 - total_turns, self.run_config, attempt_dir / "traces" / f"main-segment-{segment:02d}.json", self.context_limit, True)
            total_turns += len(result.raw_responses)
            for response in result.raw_responses:
                usage["requests"] += response.usage.requests
                usage["input_tokens"] += response.usage.input_tokens
                usage["output_tokens"] += response.usage.output_tokens
                usage["total_tokens"] += response.usage.total_tokens
            if not compact_required:
                return RolloutResult(status="completed", answer=str(result.final_output), turns=total_turns, compactions=compactions, usage=usage)
            if total_turns >= 50:
                raise MaxTurnsExceeded("Main retrieval agent reached 50 turns before compaction")
            compact_agent = main_agent.clone(tools=[])
            compact_input = result.to_input_list() + [{"role": "user", "content": COMPACTION_INSTRUCTION}]
            compact_result, _ = await stream_agent(compact_agent, compact_input, context, 50 - total_turns, self.run_config, attempt_dir / "traces" / f"compaction-{compactions + 1:02d}.json", self.context_limit, False)
            total_turns += len(compact_result.raw_responses)
            for response in compact_result.raw_responses:
                usage["requests"] += response.usage.requests
                usage["input_tokens"] += response.usage.input_tokens
                usage["output_tokens"] += response.usage.output_tokens
                usage["total_tokens"] += response.usage.total_tokens
            compactions += 1
            handoff = str(compact_result.final_output)
            if compactions == 3:
                return RolloutResult(status="compaction_limit", answer=handoff, turns=total_turns, compactions=compactions, usage=usage)
            input_items = [{"role": "developer", "content": f"Trusted handoff from the prior context of this same rollout:\n{handoff}"}]
            segment += 1
