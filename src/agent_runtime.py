import base64
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from agents import Agent, ModelSettings, OpenAIResponsesModel, RunConfig, Runner, ToolOutputImage, ToolOutputText, function_tool
from openai import AsyncOpenAI
from openai.types.shared import Reasoning
from pydantic import BaseModel

from src.harness import VideoSQLHarness


IMAGE_DATA_URL = re.compile(r"data:image/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=]+")


def serialize_usage(usage) -> dict:
    return {
        "requests": usage.requests,
        "input_tokens": usage.input_tokens,
        "input_tokens_details": usage.input_tokens_details.model_dump(mode="json"),
        "output_tokens": usage.output_tokens,
        "output_tokens_details": usage.output_tokens_details.model_dump(mode="json"),
        "total_tokens": usage.total_tokens,
        "request_usage_entries": [{"input_tokens": entry.input_tokens, "input_tokens_details": entry.input_tokens_details.model_dump(mode="json"), "output_tokens": entry.output_tokens, "output_tokens_details": entry.output_tokens_details.model_dump(mode="json"), "total_tokens": entry.total_tokens} for entry in usage.request_usage_entries],
    }


def normalize_trace(value):
    if isinstance(value, dict):
        return {key: normalize_trace(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_trace(item) for item in value]
    if isinstance(value, str):
        return IMAGE_DATA_URL.sub("[image bytes omitted; see recorded media path]", value)
    return value


class ClipAnswer(BaseModel):
    status: Literal["answered", "unknown"]
    answer: str
    evidence_timestamps: list[float]


@dataclass
class ModelEndpoint:
    model: str
    api_key: str
    base_url: str
    reasoning_effort: Literal["medium", "high"]
    prompt_cache_retention: Literal["24h"] | None


@dataclass
class RolloutResult:
    status: Literal["answered", "compaction_limit"]
    answer: str | None
    score: int | None
    reflection: str | None
    compactions: int
    trace_path: str


def data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    media_type = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
    return f"data:{media_type};base64," + base64.b64encode(path.read_bytes()).decode()


class AgentRuntime:
    def __init__(self, harness: VideoSQLHarness, endpoint: ModelEndpoint, client: AsyncOpenAI, context_limit: int, compaction_prompt: Path, reflection_final_prompt: Path):
        self.harness = harness
        self.endpoint = endpoint
        self.context_limit = context_limit
        self.compaction_prompt = compaction_prompt.read_text(encoding="utf-8")
        self.reflection_final_prompt = reflection_final_prompt.read_text(encoding="utf-8")
        self.model = OpenAIResponsesModel(model=endpoint.model, openai_client=client)
        self.settings = ModelSettings(
            reasoning=Reasoning(effort=endpoint.reasoning_effort),
            parallel_tool_calls=True,
            truncation="disabled",
            prompt_cache_retention=endpoint.prompt_cache_retention,
            preserve_raw_usage=True,
        )

    def main_tools(self, nested_tool_traces: list[dict]):
        @function_tool(failure_error_function=None)
        def run_sql(sql: str) -> dict:
            """Run read-only DuckDB SQL. Results are limited to 100 rows and long cells to 500 characters."""
            return self.harness.run_sql(sql)

        @function_tool(failure_error_function=None)
        def semantic_search(query: str, n: int, table: str) -> dict:
            """Search an embedding-enabled table. The table must be one listed in the database schema."""
            return self.harness.semantic_search(query, n, table)

        @function_tool(failure_error_function=None)
        def add_note(text: str) -> dict:
            """Persist a note for this rollout."""
            return self.harness.add_note(text)

        @function_tool(failure_error_function=None)
        def read_notes() -> dict:
            """Read all persisted notes for this rollout."""
            return self.harness.read_notes()

        @function_tool(failure_error_function=None)
        def agent_raise(reason: str):
            """Abort this rollout because a required dependency or state is broken."""
            self.harness.agent_raise(reason)

        @function_tool(failure_error_function=None)
        async def clip_qa(video_id: str, start_sec: float, end_sec: float, question: str):
            """Inspect a video segment with a nested visual agent. Prefer segments under 10 minutes; segments over 20 minutes are rejected."""
            if end_sec - start_sec > 1200:
                return {"status": "rejected", "error": "clip_qa segments must be at most 20 minutes"}

            @function_tool(failure_error_function=None)
            def sample_video(sample_start_sec: float, sample_end_sec: float, fps: float, x1: int, x2: int, y1: int, y2: int, return_caption: bool):
                """Sample this clip into timestamped 4x3 contact sheets. Crop coordinates are integers from 0 to 1000."""
                if not 0 <= x1 < x2 <= 1000 or not 0 <= y1 < y2 <= 1000:
                    return {"status": "rejected", "error": "crop coordinates must satisfy 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000"}
                if sample_start_sec < 0 or sample_end_sec <= sample_start_sec or fps <= 0:
                    return {"status": "rejected", "error": "sample_start_sec, sample_end_sec, and fps define an invalid sampling interval"}
                if sample_start_sec < start_sec or sample_end_sec > end_sec:
                    return {"status": "rejected", "error": f"sample interval must stay inside clip_qa interval {start_sec}-{end_sec}"}
                result = self.harness.sample_video(video_id, sample_start_sec, sample_end_sec, fps, x1, x2, y1, y2, return_caption)
                text = json.dumps({"frame_timestamps": result["frame_timestamps"], "captions": result["captions"], "contact_sheets": [Path(path).name for path in result["contact_sheets"]]}, ensure_ascii=False)
                outputs = [ToolOutputText(text=text)]
                outputs.extend(ToolOutputImage(image_url=data_url(Path(path)), detail="high") for path in result["contact_sheets"])
                return outputs

            @function_tool(failure_error_function=None)
            def add_clip_note(text: str) -> dict:
                """Persist a note for this clip investigation."""
                return self.harness.add_note(f"clip:{video_id}:{start_sec}-{end_sec} {text}")

            @function_tool(failure_error_function=None)
            def read_clip_notes() -> dict:
                """Read notes available to this clip investigation."""
                return self.harness.read_notes()

            clip_agent = Agent(
                name="clip investigator",
                instructions=f"Answer only the supplied question about video {video_id} between {start_sec} and {end_sec} seconds. Inspect frames until evidence is sufficient. Return unknown instead of guessing. Evidence timestamps must be inside this interval.",
                model=self.model,
                model_settings=self.settings,
                tools=[sample_video, add_clip_note, read_clip_notes],
                output_type=ClipAnswer,
            )
            nested_trace = {
                "tool": "clip_qa",
                "input": {"video_id": video_id, "start_sec": start_sec, "end_sec": end_sec, "question": question},
                "transcript": [],
                "responses": [],
            }
            nested_tool_traces.append(nested_trace)
            clip_stream = Runner.run_streamed(clip_agent, question, max_turns=50, run_config=RunConfig(reasoning_item_id_policy="preserve"))
            saved_clip_responses = 0
            try:
                async for event in clip_stream.stream_events():
                    if event.type == "raw_response_event" and event.data.type == "response.completed":
                        nested_trace["responses"].extend({"output": [item.model_dump(mode="json") for item in response.output], "usage": serialize_usage(response.usage), "response_id": response.response_id, "request_id": response.request_id, "raw_usage": response.raw_usage} for response in clip_stream.raw_responses[saved_clip_responses:])
                        saved_clip_responses = len(clip_stream.raw_responses)
            finally:
                nested_trace["responses"].extend({"output": [item.model_dump(mode="json") for item in response.output], "usage": serialize_usage(response.usage), "response_id": response.response_id, "request_id": response.request_id, "raw_usage": response.raw_usage} for response in clip_stream.raw_responses[saved_clip_responses:])
                nested_trace["transcript"] = normalize_trace(clip_stream.to_input_list(mode="normalized"))
            if clip_stream.run_loop_exception:
                raise clip_stream.run_loop_exception
            answer = clip_stream.final_output
            if answer is None:
                raise RuntimeError("clip agent ended without output")
            if answer.status == "unknown":
                return answer.model_dump()
            if not answer.evidence_timestamps:
                raise ValueError("answered clip_qa result requires at least one evidence timestamp")
            if any(timestamp < start_sec or timestamp > end_sec for timestamp in answer.evidence_timestamps):
                raise ValueError(f"evidence timestamps must stay inside clip_qa interval {start_sec}-{end_sec}")
            screenshots = self.harness.evidence_screenshots(video_id, answer.evidence_timestamps)
            answer_payload = answer.model_dump()
            answer_payload["evidence_screenshots"] = [Path(path).name for path in screenshots]
            outputs = [ToolOutputText(text=json.dumps(answer_payload, ensure_ascii=False))]
            outputs.extend(ToolOutputImage(image_url=data_url(Path(path)), detail="high") for path in screenshots)
            return outputs

        return [run_sql, semantic_search, clip_qa, add_note, read_notes, agent_raise]

    def main_agent(self, skill: str, nested_tool_traces: list[dict]) -> Agent:
        schema = (Path(__file__).with_name("schema.sql")).read_text(encoding="utf-8")
        instructions = "Answer the video-memory question using the read-only database and tool evidence. Give the answer and supporting video timestamps.\n\nDatabase schema:\n" + schema + "\n\nSkill:\n" + skill
        return Agent(name="video memory SQL agent", instructions=instructions, model=self.model, model_settings=self.settings, tools=self.main_tools(nested_tool_traces))

    async def run_rollout(self, question: str, skill: str, trace_path: Path, enable_compaction: bool) -> RolloutResult:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        endpoint_trace = asdict(self.endpoint)
        del endpoint_trace["api_key"]
        nested_tool_traces: list[dict] = []
        trace = {"question": question, "model": endpoint_trace, "responses": [], "model_responses": [], "turns": [], "nested_tool_traces": nested_tool_traces, "compactions": []}
        current_input = question
        compactions = 0
        while True:
            agent = self.main_agent(skill, nested_tool_traces)
            stream = Runner.run_streamed(
                agent,
                current_input,
                max_turns=50,
                run_config=RunConfig(reasoning_item_id_policy="preserve"),
                auto_previous_response_id=not enable_compaction,
            )
            saved_stream_responses = 0
            compact_after_turn = False
            try:
                async for event in stream.stream_events():
                    if event.type != "raw_response_event" or event.data.type != "response.completed":
                        continue
                    response = event.data.response
                    serialized = response.model_dump(mode="json")
                    trace["responses"].append(serialized)
                    trace["model_responses"].extend({"output": [item.model_dump(mode="json") for item in model_response.output], "usage": serialize_usage(model_response.usage), "response_id": model_response.response_id, "request_id": model_response.request_id, "raw_usage": model_response.raw_usage} for model_response in stream.raw_responses[saved_stream_responses:])
                    saved_stream_responses = len(stream.raw_responses)
                    trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
                    usage = response.usage
                    remaining = self.context_limit - usage.input_tokens - usage.output_tokens
                    has_tool_call = any(item.type == "function_call" for item in response.output)
                    if enable_compaction and remaining <= 50_000 and has_tool_call:
                        compact_after_turn = True
                        stream.cancel(mode="after_turn")
            finally:
                trace["model_responses"].extend({"output": [item.model_dump(mode="json") for item in response.output], "usage": serialize_usage(response.usage), "response_id": response.response_id, "request_id": response.request_id, "raw_usage": response.raw_usage} for response in stream.raw_responses[saved_stream_responses:])
                trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
            if stream.run_loop_exception:
                raise stream.run_loop_exception
            transcript = normalize_trace(stream.to_input_list(mode="normalized"))
            trace["turns"].append(transcript)
            trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
            if not compact_after_turn:
                if stream.final_output is None:
                    raise RuntimeError("agent run ended without final output")
                trace["status"] = "answered"
                trace["answer"] = stream.final_output
                trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
                return RolloutResult(status="answered", answer=str(stream.final_output), score=None, reflection=None, compactions=compactions, trace_path=str(trace_path))

            compaction_input = transcript + [{"role": "user", "content": f"{self.compaction_prompt}\n\nCompactions already completed for this rollout: {compactions}"}]
            compacted = await Runner.run(agent, compaction_input, max_turns=1, run_config=RunConfig(reasoning_item_id_policy="preserve"))
            if compacted.final_output is None:
                raise RuntimeError("main agent ended compaction without a handoff")
            compactions += 1
            trace["compactions"].append({
                "index": compactions,
                "handoff": compacted.final_output,
                "responses": [{"output": [item.model_dump(mode="json") for item in response.output], "usage": serialize_usage(response.usage), "response_id": response.response_id, "request_id": response.request_id, "raw_usage": response.raw_usage} for response in compacted.raw_responses],
            })
            trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
            if compactions == 3:
                reflection_agent = Agent(name="compaction-limit reflection", instructions=self.reflection_final_prompt, model=self.model, model_settings=self.settings)
                reflection_input = json.dumps({"compactions_completed": compactions, "trusted_handoff": compacted.final_output}, ensure_ascii=False)
                reflection_result = await Runner.run(reflection_agent, reflection_input, max_turns=1, run_config=RunConfig(reasoning_item_id_policy="preserve"))
                if reflection_result.final_output is None:
                    raise RuntimeError("compaction-limit reflection ended without output")
                trace["status"] = "compaction_limit"
                trace["score"] = 0
                trace["reflection"] = reflection_result.final_output
                trace["reflection_responses"] = [{"output": [item.model_dump(mode="json") for item in response.output], "usage": serialize_usage(response.usage), "response_id": response.response_id, "request_id": response.request_id, "raw_usage": response.raw_usage} for response in reflection_result.raw_responses]
                trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
                return RolloutResult(status="compaction_limit", answer=None, score=0, reflection=str(reflection_result.final_output), compactions=compactions, trace_path=str(trace_path))
            current_input = [{"role": "user", "content": f"Trusted compacted handoff from the prior conversation:\n\n{compacted.final_output}"}]
