import json
from dataclasses import dataclass

from agents import ModelSettings, ToolOutputImage, ToolOutputText
from agents.tool_context import ToolContext
from PIL import Image
import pytest

from src.agent_runtime import AgentRuntime, ClipAnswer, ModelEndpoint, normalize_trace


@dataclass
class FakeUsage:
    input_tokens: int = 60_000
    output_tokens: int = 1


class FakeOutput:
    type = "function_call"


class FakeResponse:
    usage = FakeUsage()
    output = [FakeOutput()]

    def model_dump(self, mode):
        return {"type": "response.completed", "usage": {"input_tokens": 60_000, "output_tokens": 1}}


class FakeStream:
    run_loop_exception = None
    final_output = None
    raw_responses = []

    def __init__(self):
        self.cancel_mode = None

    async def stream_events(self):
        event_data = type("EventData", (), {"type": "response.completed", "response": FakeResponse()})()
        yield type("Event", (), {"type": "raw_response_event", "data": event_data})()

    def cancel(self, mode):
        self.cancel_mode = mode

    def to_input_list(self, mode):
        assert mode == "normalized"
        assert self.cancel_mode == "after_turn"
        return [{"type": "function_call_output", "output": [{"image_url": "data:image/jpeg;base64,AAAA"}, {"text": '{"contact_sheets":["/tmp/contact.jpg"]}'}]}]


class FakeAgent:
    created = []

    def __init__(self, name, instructions, model, model_settings, tools=None):
        self.name = name
        self.instructions = instructions
        self.tools = tools
        self.created.append(self)


class FakeRunner:
    compact_inputs = []
    reflection_input = None
    run_count = 0
    streamed_agents = []
    streamed_inputs = []

    @classmethod
    def run_streamed(cls, agent, input, **kwargs):
        cls.streamed_agents.append(agent)
        cls.streamed_inputs.append(input)
        return FakeStream()

    @classmethod
    async def run(cls, agent, input, **kwargs):
        cls.run_count += 1
        if agent.name == "main":
            assert agent is cls.streamed_agents[-1]
            assert "base64" not in json.dumps(input)
            assert "/tmp/contact.jpg" in json.dumps(input)
            assert input[-1]["role"] == "user"
            assert "fixed compaction prompt" in input[-1]["content"]
            cls.compact_inputs.append(input)
            return type("Result", (), {"final_output": f"handoff-{len(cls.compact_inputs)}", "raw_responses": []})()
        assert agent.name == "compaction-limit reflection"
        assert agent.tools is None
        cls.reflection_input = input
        return type("Result", (), {"final_output": "final reflection", "raw_responses": []})()


async def test_third_compaction_scores_zero_and_runs_fixed_reflection(tmp_path, monkeypatch):
    monkeypatch.setattr("src.agent_runtime.Agent", FakeAgent)
    monkeypatch.setattr("src.agent_runtime.Runner", FakeRunner)
    FakeAgent.created = []
    FakeRunner.compact_inputs = []
    FakeRunner.reflection_input = None
    FakeRunner.run_count = 0
    FakeRunner.streamed_agents = []
    FakeRunner.streamed_inputs = []
    runtime = object.__new__(AgentRuntime)
    runtime.endpoint = ModelEndpoint("qwen", "secret", "http://local", "high", None)
    runtime.context_limit = 100_000
    runtime.compaction_prompt = "fixed compaction prompt"
    runtime.reflection_final_prompt = "fixed reflection prompt"
    runtime.model = object()
    runtime.settings = object()
    runtime.main_agent = lambda skill, nested_tool_traces: FakeAgent("main", skill, runtime.model, runtime.settings, [])
    trace_path = tmp_path / "trace.json"

    result = await runtime.run_rollout("question", "skill", trace_path, True)

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert result.status == "compaction_limit"
    assert result.score == 0
    assert result.reflection == "final reflection"
    assert result.compactions == 3
    assert trace["score"] == 0
    assert trace["reflection"] == "final reflection"
    assert len(trace["compactions"]) == 3
    assert "base64" not in trace_path.read_text(encoding="utf-8")
    assert "/tmp/contact.jpg" in trace_path.read_text(encoding="utf-8")
    assert FakeRunner.run_count == 4
    assert FakeRunner.streamed_inputs == ["question", [{"role": "user", "content": "Trusted compacted handoff from the prior conversation:\n\nhandoff-1"}], [{"role": "user", "content": "Trusted compacted handoff from the prior conversation:\n\nhandoff-2"}]]
    assert FakeAgent.created[-1].instructions == "fixed reflection prompt"
    assert json.loads(FakeRunner.reflection_input) == {"compactions_completed": 3, "trusted_handoff": "handoff-3"}


def test_normalize_trace_preserves_paths_and_removes_image_bytes():
    normalized = normalize_trace({"image": "data:image/png;base64,AAAA", "path": "/tmp/evidence.jpg"})
    assert normalized == {"image": "[image bytes omitted; see recorded media path]", "path": "/tmp/evidence.jpg"}


async def test_main_tool_error_propagates_without_sdk_error_text():
    class RaisingHarness:
        def run_sql(self, sql):
            raise RuntimeError(f"database failure for {sql}")

    runtime = object.__new__(AgentRuntime)
    runtime.harness = RaisingHarness()
    run_sql = {tool.name: tool for tool in runtime.main_tools([])}["run_sql"]
    context = ToolContext(None, tool_name="run_sql", tool_call_id="call-1", tool_arguments='{"sql":"SELECT 1"}')
    with pytest.raises(RuntimeError, match="database failure for SELECT 1"):
        await run_sql.on_invoke_tool(context, '{"sql":"SELECT 1"}')
    assert run_sql._use_default_failure_error_function is False
    assert run_sql._failure_error_function is None


async def test_clip_tools_expose_filenames_not_absolute_paths(tmp_path, monkeypatch):
    contact_path = tmp_path / "contact.jpg"
    evidence_path = tmp_path / "evidence.jpg"
    Image.new("RGB", (16, 16), "red").save(contact_path)
    Image.new("RGB", (16, 16), "blue").save(evidence_path)

    class Harness:
        def sample_video(self, *args):
            return {"frame_timestamps": [1.0], "captions": [], "contact_sheets": [str(contact_path)]}

        def evidence_screenshots(self, *args):
            return [str(evidence_path)]

    class ClipStream:
        raw_responses = []
        run_loop_exception = None
        final_output = ClipAnswer(status="answered", answer="person in red", evidence_timestamps=[1.0])

        async def stream_events(self):
            if False:
                yield None

        def to_input_list(self, mode):
            assert mode == "normalized"
            return []

    captured = {}

    def run_streamed(agent, input, **kwargs):
        captured["agent"] = agent
        return ClipStream()

    monkeypatch.setattr("src.agent_runtime.Runner.run_streamed", run_streamed)
    runtime = object.__new__(AgentRuntime)
    runtime.harness = Harness()
    runtime.model = "test-model"
    runtime.settings = ModelSettings()
    clip_qa = {tool.name: tool for tool in runtime.main_tools([])}["clip_qa"]
    context = ToolContext(None, tool_name="clip_qa", tool_call_id="call-1", tool_arguments='{"video_id":"1","start_sec":0,"end_sec":2,"question":"what?"}')
    clip_outputs = await clip_qa.on_invoke_tool(context, '{"video_id":"1","start_sec":0,"end_sec":2,"question":"what?"}')
    clip_payload = json.loads(clip_outputs[0].text)
    assert clip_payload["evidence_screenshots"] == ["evidence.jpg"]
    assert str(tmp_path) not in clip_outputs[0].text
    assert isinstance(clip_outputs[1], ToolOutputImage)

    sample_video = {tool.name: tool for tool in captured["agent"].tools}["sample_video"]
    sample_context = ToolContext(None, tool_name="sample_video", tool_call_id="call-2", tool_arguments='{"sample_start_sec":0,"sample_end_sec":2,"fps":1,"x1":0,"x2":1000,"y1":0,"y2":1000,"return_caption":true}')
    sample_outputs = await sample_video.on_invoke_tool(sample_context, '{"sample_start_sec":0,"sample_end_sec":2,"fps":1,"x1":0,"x2":1000,"y1":0,"y2":1000,"return_caption":true}')
    assert isinstance(sample_outputs[0], ToolOutputText)
    assert json.loads(sample_outputs[0].text)["contact_sheets"] == ["contact.jpg"]
    assert str(tmp_path) not in sample_outputs[0].text
    assert isinstance(sample_outputs[1], ToolOutputImage)
    rejected_context = ToolContext(None, tool_name="sample_video", tool_call_id="call-3", tool_arguments='{"sample_start_sec":-1,"sample_end_sec":2,"fps":1,"x1":0,"x2":1000,"y1":0,"y2":1000,"return_caption":true}')
    rejected_output = await sample_video.on_invoke_tool(rejected_context, '{"sample_start_sec":-1,"sample_end_sec":2,"fps":1,"x1":0,"x2":1000,"y1":0,"y2":1000,"return_caption":true}')
    assert rejected_output == {"status": "rejected", "error": "sample interval must stay inside clip_qa interval 0.0-2.0"}


async def test_clip_stream_error_preserves_completed_responses_and_transcript(monkeypatch):
    class TokenDetails:
        def model_dump(self, mode):
            assert mode == "json"
            return {}

    class Usage:
        requests = 1
        input_tokens = 12
        input_tokens_details = TokenDetails()
        output_tokens = 3
        output_tokens_details = TokenDetails()
        total_tokens = 15
        request_usage_entries = []

    class ModelResponse:
        output = []
        usage = Usage()
        response_id = "response-1"
        request_id = "request-1"
        raw_usage = {"input_tokens": 12, "output_tokens": 3}

    class BrokenClipStream:
        raw_responses = [ModelResponse()]
        run_loop_exception = None
        final_output = None

        async def stream_events(self):
            raise RuntimeError("clip stream broke")
            yield

        def to_input_list(self, mode):
            assert mode == "normalized"
            return [{"role": "assistant", "content": "partial clip investigation"}]

    monkeypatch.setattr("src.agent_runtime.Runner.run_streamed", lambda *args, **kwargs: BrokenClipStream())
    runtime = object.__new__(AgentRuntime)
    runtime.model = "test-model"
    runtime.settings = ModelSettings()
    nested_tool_traces = []
    clip_qa = {tool.name: tool for tool in runtime.main_tools(nested_tool_traces)}["clip_qa"]
    context = ToolContext(None, tool_name="clip_qa", tool_call_id="call-1", tool_arguments='{"video_id":"1","start_sec":0,"end_sec":2,"question":"what?"}')

    with pytest.raises(RuntimeError, match="clip stream broke"):
        await clip_qa.on_invoke_tool(context, '{"video_id":"1","start_sec":0,"end_sec":2,"question":"what?"}')

    assert nested_tool_traces == [{
        "tool": "clip_qa",
        "input": {"video_id": "1", "start_sec": 0.0, "end_sec": 2.0, "question": "what?"},
        "transcript": [{"role": "assistant", "content": "partial clip investigation"}],
        "responses": [{
            "output": [],
            "usage": {
                "requests": 1,
                "input_tokens": 12,
                "input_tokens_details": {},
                "output_tokens": 3,
                "output_tokens_details": {},
                "total_tokens": 15,
                "request_usage_entries": [],
            },
            "response_id": "response-1",
            "request_id": "request-1",
            "raw_usage": {"input_tokens": 12, "output_tokens": 3},
        }],
    }]


async def test_main_stream_error_persists_completed_response(tmp_path, monkeypatch):
    class TokenDetails:
        def model_dump(self, mode):
            assert mode == "json"
            return {}

    class Usage:
        requests = 1
        input_tokens = 20
        input_tokens_details = TokenDetails()
        output_tokens = 5
        output_tokens_details = TokenDetails()
        total_tokens = 25
        request_usage_entries = []

    class ModelResponse:
        output = []
        usage = Usage()
        response_id = "main-response-1"
        request_id = "main-request-1"
        raw_usage = {"input_tokens": 20, "output_tokens": 5}

    class BrokenMainStream:
        raw_responses = [ModelResponse()]
        run_loop_exception = None
        final_output = None

        async def stream_events(self):
            raise RuntimeError("main stream broke")
            yield

    monkeypatch.setattr("src.agent_runtime.Runner.run_streamed", lambda *args, **kwargs: BrokenMainStream())
    runtime = object.__new__(AgentRuntime)
    runtime.endpoint = ModelEndpoint("qwen", "secret", "http://local", "high", None)
    runtime.context_limit = 100_000
    runtime.main_agent = lambda skill, nested_tool_traces: object()
    trace_path = tmp_path / "trace.json"

    with pytest.raises(RuntimeError, match="main stream broke"):
        await runtime.run_rollout("question", "skill", trace_path, False)

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["model_responses"] == [{
        "output": [],
        "usage": {
            "requests": 1,
            "input_tokens": 20,
            "input_tokens_details": {},
            "output_tokens": 5,
            "output_tokens_details": {},
            "total_tokens": 25,
            "request_usage_entries": [],
        },
        "response_id": "main-response-1",
        "request_id": "main-request-1",
        "raw_usage": {"input_tokens": 20, "output_tokens": 5},
    }]
