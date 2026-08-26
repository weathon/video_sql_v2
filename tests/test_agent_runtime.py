import json
from dataclasses import dataclass

from src.agent_runtime import AgentRuntime, ModelEndpoint, normalize_trace


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

    @classmethod
    def run_streamed(cls, *args, **kwargs):
        return FakeStream()

    @classmethod
    async def run(cls, agent, input, **kwargs):
        cls.run_count += 1
        if agent.name == "rollout compactor":
            assert "base64" not in input
            assert "/tmp/contact.jpg" in input
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
    assert FakeAgent.created[-1].instructions == "fixed reflection prompt"
    assert json.loads(FakeRunner.reflection_input) == {"compactions_completed": 3, "trusted_handoff": "handoff-3"}


def test_normalize_trace_preserves_paths_and_removes_image_bytes():
    normalized = normalize_trace({"image": "data:image/png;base64,AAAA", "path": "/tmp/evidence.jpg"})
    assert normalized == {"image": "[image bytes omitted; see recorded media path]", "path": "/tmp/evidence.jpg"}
