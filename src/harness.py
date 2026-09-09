import asyncio
import base64
import dataclasses
import io
import json
import math
import re
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import duckdb
import httpx
import httpx2
import numpy as np
from agents import (
    Agent,
    ItemHelpers,
    ModelSettings,
    OpenAIResponsesModel,
    RunConfig,
    Runner,
    StopAtTools,
    ToolOutputImage,
    ToolOutputText,
    ToolsToFinalOutputResult,
    function_tool,
)
from agents.run import ModelInputData
from agents.tool_context import ToolContext
from openai import APIError, AsyncOpenAI, DefaultAsyncHttpxClient
from PIL import Image, ImageOps
from pydantic import BaseModel

from src.normalize import normalize_text


OPENAI_BASE_URL = "https://api.openai.com/v1"
CONTEXT_LIMIT = 314573
COMPACTION_THRESHOLD = 178_000
MAX_TURNS = 100
FRAME_SIZE = (854, 480)
# Keep the existing sampling allowance when switching providers; this is not a
# model-specific image token estimate. Context accounting uses returned usage.
CLIP_IMAGE_LIMIT = 650
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


class AgentRaised(Exception):
    pass


class InvalidTrace(Exception):
    pass


class AnswerSubmission(BaseModel):
    answer: str


class ClipSubmission(BaseModel):
    status: Literal["answered", "unknown"]
    answer: str
    evidence_frame_ids: list[str]


@dataclass
class RolloutSession:
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
    embedding_client: httpx.AsyncClient
    context_limit: int = CONTEXT_LIMIT
    compaction_threshold: int = COMPACTION_THRESHOLD
    compactions: int = 0
    base_instructions: str = ""
    response_states: dict[str, Any] = field(default_factory=dict)
    vectors: dict[str, np.ndarray] = field(init=False)
    row_ids: dict[str, np.ndarray] = field(init=False)
    sources: list[dict[str, Any]] = field(init=False)
    notes_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    trace_counter: int = 0
    tool_counter: int = 0
    tool_calls: dict[int, dict[str, Any]] = field(default_factory=dict)
    tool_ids: dict[str, int] = field(default_factory=dict)
    tool_outputs: dict[int, list[dict[str, Any]]] = field(default_factory=dict)

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


def exception_chain(error: BaseException) -> list[BaseException]:
    chain = []
    current: BaseException | None = error
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def tool_error_json(_context: Any, error: Exception) -> str:
    for item in exception_chain(error):
        if isinstance(item, FileNotFoundError):
            raise item
        if isinstance(item, APIError) and (
            item.code == "insufficient_quota" or "no credits remaining" in str(item).lower()
        ):
            raise item
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
        if path.name == "ROOT-SKILL.md":
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
        if len(columns) != len(set(columns)):
            raise ValueError(
                f"duplicate SQL column names: {columns}; use distinct AS aliases"
            )
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


def save_trace(
    context: HarnessContext,
    label: str,
    result: Any,
    header: dict[str, Any],
    events_path: Path,
    error: BaseException | None,
    tool_calls: list[dict[str, Any]],
) -> str:
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
        if response.response_id in context.response_states:
            raw_responses[-1]["prefix_state"] = context.response_states[response.response_id]
    payload = {
        **header,
        "events_file": events_path.name,
        "error": None if error is None else {"type": type(error).__name__, "message": str(error)},
        "input": stripped_json(result.input),
        "history": stripped_json(result.to_input_list()),
        "raw_responses": raw_responses,
        "tool_calls": stripped_json(tool_calls),
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
    tool_calls = []
    # A fixed-prefix continuation carries the IDs in the actual tool outputs.
    # Restore those calls before allocating any new IDs in a fresh harness.
    if isinstance(input_items, list):
        prior_calls = {item["call_id"]: item for item in input_items
                       if item.get("type") == "function_call"}
        restored = []
        for item in input_items:
            if item.get("type") != "function_call_output" or item["call_id"] in context.tool_ids:
                continue
            output = item["output"]
            if isinstance(output, str):
                header, _, original_output = output.partition("\n")
            else:
                header, original_output = output[0]["text"], output[1:]
            tool_id = json.loads(header)["tool_id"]
            call = prior_calls[item["call_id"]]
            record = {"tool_id": tool_id, "call_id": item["call_id"], "name": call["name"],
                      "arguments": call["arguments"], "original_output": stripped_json(original_output),
                      "replacements": []}
            context.tool_calls[tool_id] = record
            context.tool_ids[item["call_id"]] = tool_id
            context.tool_outputs[tool_id] = [call, item]
            context.tool_counter = max(context.tool_counter, tool_id)
            restored.append(record)
        for record in restored:
            if record["name"] == "replace_tool":
                acknowledgement = json.loads(record["original_output"])
                if "replaced_tool_id" in acknowledgement:
                    arguments = json.loads(record["arguments"])
                    context.tool_calls[arguments["tool_id"]]["replacements"].append(arguments["replacement"])

    @function_tool(failure_error_function=tool_error_json)
    async def replace_tool(tool_context: ToolContext, tool_id: int, replacement: str) -> str:
        """Replace any completed tool result in model context with your replacement text.

        Every tool call has an integer tool_id, numbered from 1 within this rollout
        attempt, shared with nested agents. Any tool's returned content, including
        images and earlier replace_tool results, can be replaced. Prefer replacing
        the latest tool result to preserve as much of the KV cache prefix as possible.

        Use this to remove irrelevant results from a wrong tool call (for example,
        reading the wrong file or executing the wrong SQL): large useless outputs
        consume context and distract attention. Also use it to summarize redundant
        results, such as long SQL output containing only a little useful information.
        Preserve the useful evidence in your replacement. Only the returned content
        changes; this does not undo the tool's side effects. Original results remain
        in the saved trace for reflection and Trace2Skill analysis.
        """
        if tool_id not in context.tool_calls:
            raise ValueError(f"tool_id {tool_id} is not a completed tool call")
        call, original = context.tool_outputs[tool_id]
        replacements = context.tool_calls[tool_id]["replacements"]
        prefix = json.dumps({"tool_id": tool_id}) + "\n"
        before = {**original, "output": prefix + replacements[-1]} if replacements else original
        after = {**original, "output": prefix + replacement}
        (context.run_dir / f"replace-{tool_context.tool_call_id}.json").write_text(json.dumps({
            "tool_id": tool_id, "replace_call_id": tool_context.tool_call_id,
            "model": context.model.model, "reasoning": stripped_json(context.model_settings.reasoning),
            "before": stripped_json([call, before]), "after": stripped_json([call, after]),
        }, ensure_ascii=False) + "\n")
        context.tool_calls[tool_id]["replacements"].append(replacement)
        return json.dumps({"replaced_tool_id": tool_id})

    wrapped_tools = []
    for tool in [*agent.tools, replace_tool] if agent.tools else []:
        async def invoke_tool(tool_context: ToolContext, arguments: str, original=tool.on_invoke_tool):
            context.tool_counter += 1
            tool_id = context.tool_counter
            output = await original(tool_context, arguments)
            record = {"tool_id": tool_id, "call_id": tool_context.tool_call_id,
                      "name": tool_context.tool_name, "arguments": arguments,
                      "original_output": stripped_json(output), "replacements": []}
            context.tool_calls[tool_id] = record
            context.tool_ids[tool_context.tool_call_id] = tool_id
            tool_calls.append(record)
            prefix = json.dumps({"tool_id": tool_id}) + "\n"
            if isinstance(output, str):
                numbered_output = prefix + output
            else:
                numbered_output = [ToolOutputText(text=prefix), *output]
            context.tool_outputs[tool_id] = [
                tool_context.tool_call.model_dump(mode="json", exclude_none=True),
                ItemHelpers.tool_call_output_item(tool_context.tool_call, numbered_output),
            ]
            return numbered_output

        wrapped_tools.append(dataclasses.replace(tool, on_invoke_tool=invoke_tool))
    behavior = agent.tool_use_behavior
    if isinstance(behavior, dict):
        def finish_numbered(_run_context: Any, tool_results: list[Any]) -> ToolsToFinalOutputResult:
            for result in tool_results:
                if result.tool.name in behavior["stop_at_tool_names"]:
                    return ToolsToFinalOutputResult(
                        is_final_output=True,
                        final_output=result.output.partition("\n")[2],
                    )
            return ToolsToFinalOutputResult(is_final_output=False)

        agent = agent.clone(tools=wrapped_tools, tool_use_behavior=finish_numbered)
    else:
        agent = agent.clone(tools=wrapped_tools)
    header = {
        "agent": {
            "name": agent.name,
            "instructions": agent.instructions,
            "model": context.model.model,
            "base_url": str(context.client.base_url),
            "model_settings": stripped_json(agent.model_settings),
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.params_json_schema,
                }
                for tool in agent.tools
            ],
        },
        "run_config": {
            "reasoning_item_id_policy": context.run_config.reasoning_item_id_policy,
            "max_turns": MAX_TURNS,
            "watch_context": watch_context,
            "context_limit": context.context_limit,
            "compaction_threshold": context.compaction_threshold,
        },
    }
    events_path = context.run_dir / "traces" / f"{label}-events-{uuid.uuid4().hex}.jsonl"
    context_triggered = False
    caught: BaseException | None = None
    with events_path.open("x", buffering=1) as events:
        def model_input(data):
            items = []
            for item in data.model_data.input:
                if item.get("type") == "function_call_output":
                    tool_id = context.tool_ids[item["call_id"]]
                    record = context.tool_calls[tool_id]
                    if record["replacements"]:
                        output = json.dumps({"tool_id": tool_id}) + "\n" + record["replacements"][-1]
                        item = {**item, "output": output}
                items.append(item)
            events.write(json.dumps({"type": "model_input", "input": stripped_json(items)}, ensure_ascii=False) + "\n")
            return ModelInputData(input=items, instructions=data.model_data.instructions)

        events.write(json.dumps({
            "type": "run_start", **header, "input": stripped_json(input_items),
        }, ensure_ascii=False) + "\n")
        result = Runner.run_streamed(
            agent,
            input_items,
            max_turns=MAX_TURNS,
            run_config=dataclasses.replace(context.run_config, call_model_input_filter=model_input),
        )
        try:
            async for event in result.stream_events():
                if event.type == "raw_response_event":
                    record = {"type": event.type, "data": stripped_json(event.data)}
                elif event.type == "run_item_stream_event":
                    record = {
                        "type": event.type,
                        "name": event.name,
                        "agent": event.item.agent.name,
                        "item_type": event.item.type,
                        "item": stripped_json(event.item.raw_item),
                    }
                    if event.item.type == "tool_call_output_item":
                        record["tool_id"] = context.tool_ids[event.item.raw_item["call_id"]]
                else:
                    record = {"type": event.type, "agent": event.new_agent.name}
                events.write(json.dumps(record, ensure_ascii=False) + "\n")
                if event.type != "raw_response_event":
                    continue
                if getattr(event.data, "type", None) != "response.completed":
                    continue
                response = event.data.response
                if response.usage is None:
                    raise RuntimeError("completed response did not include usage")
                remaining = (
                    context.context_limit
                    - response.usage.input_tokens
                    - response.usage.output_tokens
                )
                if watch_context and remaining <= context.compaction_threshold:
                    result.cancel(mode="after_turn")
                    context_triggered = True
        except BaseException as error:
            caught = error
            if isinstance(error, asyncio.CancelledError):
                result.cancel(mode="immediate")
                try:
                    await result.run_loop_task
                except asyncio.CancelledError:
                    pass
            events.write(json.dumps({
                "type": "run_error", "error_type": type(error).__name__, "message": str(error),
            }, ensure_ascii=False) + "\n")
            print(f"agent={agent.name} label={label} error={error}", flush=True)
    trace_file = save_trace(
        context, label, result, header, events_path,
        caught if caught is not None else result.run_loop_exception,
        tool_calls,
    )
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


def frame_jpeg_bytes(recipe_path: Path) -> bytes:
    frame = json.loads(recipe_path.read_text())
    left, top, right, bottom = frame["crop"]
    with Image.open(recipe_path.parent / frame["original_path"]) as original:
        width, height = original.size
        cropped = original.convert("RGB").crop((
            math.floor(width * left / 1000),
            math.floor(height * top / 1000),
            math.ceil(width * right / 1000),
            math.ceil(height * bottom / 1000),
        ))
        rendered = ImageOps.contain(cropped, FRAME_SIZE)
        output = io.BytesIO()
        rendered.save(output, format="JPEG", quality=90)
    return output.getvalue()


def render_frame(frame: dict[str, Any]) -> list[ToolOutputText | ToolOutputImage]:
    recipe = {key: value for key, value in frame.items() if key != "recipe_path"}
    recipe["original_path"] = frame["original_path"].relative_to(frame["recipe_path"].parent)
    frame["recipe_path"].write_text(
        json.dumps(stripped_json(recipe), ensure_ascii=False, indent=2) + "\n"
    )
    return [
        ToolOutputText(text=json.dumps({
            "frame_id": frame["frame_id"],
            "time_s": frame["time_s"],
            "filename": frame["recipe_path"].name,
            "frame_recipe": str(frame["recipe_path"]),
            "crop": frame["crop"],
        })),
        ToolOutputImage(
            image_url="data:image/jpeg;base64,"
            + base64.b64encode(frame_jpeg_bytes(frame["recipe_path"])).decode(),
            detail="high",
        ),
    ]


async def sample_frames(
    context: HarnessContext,
    start_s: float,
    end_s: float,
    fps: float,
    crop: tuple[int, int, int, int],
    call_id: str,
    remaining_images: int,
) -> list[dict[str, Any]]:
    if not context.sources[0]["timeline_start_s"] <= start_s < end_s <= context.sources[-1]["timeline_end_s"]:
        raise ValueError("sample interval extends beyond available source media")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if remaining_images == 0:
        raise ValueError("clip image limit reached; remaining_images=0")
    media_root = context.run_dir / "media"
    sample_dir = media_root / f"sample-{call_id}-{uuid.uuid4().hex}"
    sample_dir.mkdir()
    frames = []
    for source_index, source in enumerate(context.sources):
        overlap_start = max(start_s, source["timeline_start_s"])
        overlap_end = min(end_s, source["timeline_end_s"])
        if overlap_start >= overlap_end:
            continue
        local_start = overlap_start - source["timeline_start_s"]
        local_end = overlap_end - source["timeline_start_s"]
        pattern = sample_dir / f"source-{source_index:06d}-%08d.png"
        # select keeps the source PTS; fps would replace it with a synthetic clock.
        video_filter = (
            f"trim=start={local_start:.6f}:end={local_end:.6f},"
            f"select='isnan(prev_selected_t)+"
            f"gt(floor((t-{local_start:.6f})*{fps}),"
            f"floor((prev_selected_t-{local_start:.6f})*{fps}))',showinfo"
        )
        command = [
            "ffmpeg", "-hide_banner", "-nostdin", "-nostats", "-v", "info",
            "-copyts", "-start_at_zero",
            "-ss", f"{local_start:.6f}", "-t", f"{overlap_end - overlap_start:.6f}",
            "-threads", "1", "-i", source["path"],
            "-map", "0:v:0", "-vf", video_filter,
            "-filter_threads", "1", "-fps_mode", "passthrough",
            # Decode one extra frame to detect excess; never return a partial sample.
            "-frames:v", str(remaining_images - len(frames) + 1),
            "-threads", "1", str(pattern),
        ]
        log_path = sample_dir / f"ffmpeg-{source_index:06d}.log"
        (sample_dir / f"ffmpeg-{source_index:06d}.json").write_text(
            json.dumps(command, ensure_ascii=False, indent=2) + "\n"
        )
        print(
            f"sample_video start={overlap_start} end={overlap_end} fps={fps} "
            f"remaining_images={remaining_images - len(frames)} log={log_path}",
            flush=True,
        )
        with log_path.open("wb") as log:
            process = await asyncio.create_subprocess_exec(
                *command, stdout=asyncio.subprocess.DEVNULL, stderr=log,
            )
            await process.wait()
        if process.returncode != 0:
            print(f"ffmpeg exit={process.returncode} log={log_path}", flush=True)
            raise RuntimeError(f"ffmpeg exit={process.returncode}; see {log_path.name}")
        source_frames = sorted(sample_dir.glob(f"source-{source_index:06d}-*.png"))
        if len(frames) + len(source_frames) > remaining_images:
            raise ValueError(
                f"sample exceeds clip image limit; remaining_images={remaining_images}. "
                "Request a shorter interval or a lower fps."
            )
        log_text = log_path.read_text()
        time_base_match = re.search(r"config in time_base: (\d+)/(\d+)", log_text)
        timestamps = re.findall(r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:", log_text)
        if time_base_match is None or len(timestamps) != len(source_frames):
            raise RuntimeError(f"ffmpeg frame/PTS mismatch; see {log_path.name}")
        time_base_num, time_base_den = map(int, time_base_match.groups())
        for frame_index, (frame_path, (sequence, pts)) in enumerate(zip(source_frames, timestamps)):
            if int(sequence) != frame_index:
                raise RuntimeError(f"ffmpeg frame sequence mismatch; see {log_path.name}")
            time_s = source["timeline_start_s"] + int(pts) * time_base_num / time_base_den
            frame_id = uuid.uuid4().hex
            frames.append({
                "frame_id": frame_id,
                "time_s": time_s,
                "pts": int(pts),
                "time_base": [time_base_num, time_base_den],
                "source_timeline_start_s": source["timeline_start_s"],
                "original_path": frame_path,
                "recipe_path": media_root / f"frame-{frame_id}.json",
                "crop": crop,
            })
    if not frames:
        raise RuntimeError("ffmpeg returned no frames")
    return frames


def make_tools(context: HarnessContext) -> tuple[list[Any], Any]:
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
        return await run_one_call(
            context,
            "sql-qa",
            "Answer only from the supplied SQL result. State when it does not establish the answer.",
            normalize_text(question) + "\n\nSQL result:\n" + serialized,
        )

    def semantic_search_error(tool_context: Any, error: Exception) -> str:
        if isinstance(error, (httpx.HTTPError, RuntimeError, FileNotFoundError)):
            raise error
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
        response = await context.embedding_client.post(
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
        if not math.isfinite(start_s) or not math.isfinite(end_s) or start_s < 0 or end_s <= start_s:
            raise ValueError("clip interval must be finite, non-negative and increasing")
        if end_s - start_s > 1200:
            raise ValueError("clip_qa rejects intervals longer than 20 minutes")
        call_id = uuid.uuid4().hex
        frames: dict[str, dict[str, Any]] = {}
        images_lock = asyncio.Lock()

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
            """Return individual frames, their IDs and actual source times.

            Crop coordinates refer to the original full frame, in 0-1000 units.
            Full frame: left=0, top=0, right=1000, bottom=1000.
            Require 0 <= left < right <= 1000 and 0 <= top < bottom <= 1000.
            Sampling and crops share a cumulative 650-image limit per clip_qa.
            An over-budget request returns an error without partial images.
            """
            if not start_s <= sample_start_s < sample_end_s <= end_s:
                raise ValueError("sample interval must lie inside the clip_qa interval")
            if not (
                0 <= crop_left < crop_right <= 1000
                and 0 <= crop_top < crop_bottom <= 1000
            ):
                raise ValueError("crop coordinates must satisfy 0 <= left < right <= 1000 and 0 <= top < bottom <= 1000")
            async with images_lock:
                new_frames = await sample_frames(
                    context, sample_start_s, sample_end_s, fps,
                    (crop_left, crop_top, crop_right, crop_bottom),
                    call_id, CLIP_IMAGE_LIMIT - len(frames),
                )
                captions = []
                if include_captions:
                    result = await asyncio.to_thread(
                        execute_sql,
                        context.database,
                        "SELECT row_id, start_s, end_s, text FROM episodes "
                        "WHERE granularity = '30sec' AND end_s > ? AND start_s < ? "
                        "ORDER BY start_s",
                        None, [sample_start_s, sample_end_s], None,
                    )
                    captions = result["rows"]
                count = len(frames) + len(new_frames)
                outputs: list[Any] = [ToolOutputText(text=json.dumps({
                    "frame_count": len(new_frames),
                    "images_used": count,
                    "remaining_images": CLIP_IMAGE_LIMIT - count,
                    "captions": captions,
                }, ensure_ascii=False))]
                for frame in new_frames:
                    outputs.extend(await asyncio.to_thread(render_frame, frame))
                frames.update({frame["frame_id"]: frame for frame in new_frames})
                return outputs

        @function_tool(failure_error_function=tool_error_json)
        async def crop_frame(
            frame_id: str,
            crop_left: int,
            crop_top: int,
            crop_right: int,
            crop_bottom: int,
        ) -> list[Any]:
            """Crop a sampled frame using its saved original resolution.

            Coordinates always refer to the original full frame, even when frame_id
            names an earlier crop. Units are 0-1000; full frame is 0,0,1000,1000.
            Require 0 <= left < right <= 1000 and 0 <= top < bottom <= 1000.
            Each returned crop consumes one image from this clip's shared budget.
            """
            if not (
                0 <= crop_left < crop_right <= 1000
                and 0 <= crop_top < crop_bottom <= 1000
            ):
                raise ValueError("crop coordinates must satisfy 0 <= left < right <= 1000 and 0 <= top < bottom <= 1000")
            async with images_lock:
                if frame_id not in frames:
                    raise ValueError("frame_id must refer to an image returned in this clip_qa")
                if len(frames) == CLIP_IMAGE_LIMIT:
                    raise ValueError("clip image limit reached; remaining_images=0")
                original = frames[frame_id]
                new_id = uuid.uuid4().hex
                frame = {
                    **original,
                    "parent_frame_id": frame_id,
                    "frame_id": new_id,
                    "time_s": original["time_s"],
                    "original_path": original["original_path"],
                    "recipe_path": context.run_dir / "media" / f"frame-{new_id}.json",
                    "crop": (crop_left, crop_top, crop_right, crop_bottom),
                }
                outputs = await asyncio.to_thread(render_frame, frame)
                count = len(frames) + 1
                outputs.append(ToolOutputText(text=json.dumps({
                    "images_used": count,
                    "remaining_images": CLIP_IMAGE_LIMIT - count,
                })))
                frames[new_id] = frame
                return outputs

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
            evidence_frame_ids: list[str],
        ) -> str:
            """Submit the visual answer citing frame IDs returned by sample_video or crop_frame."""
            if status == "answered" and not evidence_frame_ids:
                raise ValueError("answered requires at least one evidence frame")
            if any(frame_id not in frames for frame_id in evidence_frame_ids):
                raise ValueError("evidence_frame_ids must refer to images returned in this clip_qa")
            return ClipSubmission(
                status=status,
                answer=answer,
                evidence_frame_ids=evidence_frame_ids,
            ).model_dump_json()

        def finish_clip(_run_context: Any, tool_results: list[Any]) -> ToolsToFinalOutputResult:
            for result in tool_results:
                if result.tool.name == "submit_clip_answer":
                    payload = json.loads(result.output.partition("\n")[2])
                    if "error" not in payload:
                        submission = ClipSubmission.model_validate(payload)
                        return ToolsToFinalOutputResult(
                            is_final_output=True,
                            final_output=submission.model_dump_json(),
                        )
            return ToolsToFinalOutputResult(is_final_output=False)

        clip_agent = Agent(
            name="clip-qa",
            instructions=(
                "Answer the visual question by sampling the supplied interval. "
                "Use sample_video, crop_frame for fine details, notes when useful, and "
                "submit_clip_answer with evidence_frame_ids from returned images. "
                f"Sampling and crops share {CLIP_IMAGE_LIMIT} images across this entire investigation. "
                "Each image is at most 854 by 480 pixels; crop_frame reads its original resolution. "
                "Do not infer an answered result without visible evidence."
            ),
            model=context.model,
            model_settings=context.model_settings,
            tools=[sample_video, crop_frame, add_clip_note, read_clip_notes, submit_clip_answer],
            tool_use_behavior=finish_clip,
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
                        "evidence_times_s": [
                            frames[frame_id]["time_s"]
                            for frame_id in submission.evidence_frame_ids
                        ],
                    },
                    ensure_ascii=False,
                )
            )
        ]
        if submission.status == "answered":
            for frame_id in submission.evidence_frame_ids:
                frame = frames[frame_id]
                outputs.append(
                    ToolOutputText(text=json.dumps({
                        "frame_id": frame_id,
                        "evidence_time_s": frame["time_s"],
                        "filename": frame["recipe_path"].name,
                        "frame_recipe": str(frame["recipe_path"]),
                        "crop": frame["crop"],
                    }))
                )
                outputs.append(
                    ToolOutputImage(
                        image_url="data:image/jpeg;base64,"
                        + base64.b64encode(await asyncio.to_thread(frame_jpeg_bytes, frame["recipe_path"])).decode(),
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
    return answer_tools, catalog


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
        embedding_url: str,
        embedding_model: str,
        *,
        base_url: str = OPENAI_BASE_URL,
        context_limit: int = CONTEXT_LIMIT,
        compaction_threshold: int = COMPACTION_THRESHOLD,
    ) -> None:
        client = AsyncOpenAI(
            api_key=api_key, base_url=base_url, max_retries=3,
            http_client=DefaultAsyncHttpxClient(
                timeout=httpx2.Timeout(connect=60, read=100000, write=100000, pool=100000),
                limits=httpx2.Limits(max_connections=1000, max_keepalive_connections=256),
            ),
        )
        model = OpenAIResponsesModel(model=model_name, openai_client=client)
        model_settings = ModelSettings(
            reasoning={"effort": "high", "summary": "auto"},
            store=False,
            response_include=["reasoning.encrypted_content"],
            preserve_raw_usage=True,
        )
        run_config = RunConfig(
            tracing_disabled=False,
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
            embedding_client=httpx.AsyncClient(
                timeout=httpx.Timeout(connect=60, read=100000, write=100000, pool=100000),
                limits=httpx.Limits(max_connections=1000, max_keepalive_connections=256),
            ),
            context_limit=context_limit,
            compaction_threshold=compaction_threshold,
        )
        self.answer_tools, self.catalog = make_tools(self.context)

    async def answer(
        self,
        question: str,
        *,
        input_items: list[dict[str, Any]] | None = None,
        instructions: str | None = None,
        base_instructions: str | None = None,
        compactions: int = 0,
    ) -> RolloutSession:
        if base_instructions is None:
            base_instructions = instructions if instructions is not None else (
                "Answer the user's question by retrieving evidence from the memory database. "
                "Use submit_answer when finished. The complete database schema is:\n" + SCHEMA
                + "\n\nROOT-SKILL (skill loading guidance):\n"
                + (self.context.skill_dir / "ROOT-SKILL.md").read_text()
            )
        self.context.base_instructions = base_instructions
        self.context.compactions = compactions
        agent = Agent(
            name="memory-retrieval",
            instructions=instructions if instructions is not None else base_instructions,
            model=self.context.model,
            model_settings=self.context.model_settings,
            tools=self.answer_tools,
            tool_use_behavior=StopAtTools(stop_at_tool_names=["submit_answer"]),
        )
        skills = "\n".join(
            f"- {skill['name']}: {skill['description']}" for skill in self.catalog
        )
        if input_items is None:
            input_items = f"Question: {normalize_text(question)}\n\nAvailable skills:\n{skills}"
        trace_files = []
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
            if result.final_output is not None:
                submission = AnswerSubmission.model_validate_json(str(result.final_output))
                return RolloutSession(
                    answer=submission.answer,
                    status="answered",
                    trace_files=trace_files,
                )
            if not needs_compaction:
                raise InvalidTrace("answer loop ended without submit_answer")
            self.context.compactions += 1
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
            if self.context.compactions == 3:
                return RolloutSession(
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

    async def close(self) -> None:
        await self.context.client.close()
        await self.context.embedding_client.aclose()
