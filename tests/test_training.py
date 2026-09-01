import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import duckdb
import httpx2
import pytest
from agents import ModelSettings, RunContextWrapper
from agents.exceptions import ModelBehaviorError
from agents.exceptions import UserError

from src.harness import AgentRaised, ClipAnswer, RolloutContext, RolloutResult, StreamModelBehaviorError, clip_qa
from src.training import JudgeResult, is_retryable_error, rollout_trace, run_evaluation_rollout, run_judge, run_stage


async def test_run_stage_retries_and_reuses_saved_output(tmp_path: Path) -> None:
    calls = 0

    async def first_call(attempt_dir: Path) -> tuple[JudgeResult, dict[str, int]]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ModelBehaviorError("invalid structured output")
        return JudgeResult(score=4), {"requests": 1, "input_tokens": 2, "output_tokens": 3, "total_tokens": 5}

    output, usage = await run_stage(tmp_path / "stage", JudgeResult, 3, "formal", first_call)
    assert output.score == 4
    assert usage["total_tokens"] == 5
    assert calls == 2

    async def forbidden_call(attempt_dir: Path) -> tuple[JudgeResult, dict[str, int]]:
        raise AssertionError(f"Saved output was not reused: {attempt_dir}")

    resumed, resumed_usage = await run_stage(tmp_path / "stage", JudgeResult, 3, "formal", forbidden_call)
    assert resumed.score == 4
    assert resumed_usage == usage


async def test_judge_input_contains_only_reference_and_candidate(tmp_path: Path, monkeypatch) -> None:
    async def saved_stage(stage_dir: Path, output_type, attempts: int, mode: str, call):
        return JudgeResult(score=5), {"requests": 1, "input_tokens": 2, "output_tokens": 3, "total_tokens": 5}

    monkeypatch.setattr("src.training.run_stage", saved_stage)
    result, _ = await run_judge(object(), "gold", "candidate", tmp_path / "judge", 3, "formal", 10)
    assert result.score == 5
    assert json.loads((tmp_path / "judge" / "input.json").read_text()) == {"reference_answer": "gold", "candidate_answer": "candidate"}


async def test_main_rollout_input_excludes_gold_answer(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    class Runtime:
        async def run_rollout(self, question: str, skill: str, attempt_dir: Path, clip_max_turns: int) -> RolloutResult:
            captured.update({"question": question, "skill": skill, "attempt_dir": attempt_dir, "clip_max_turns": clip_max_turns})
            return RolloutResult(status="completed", answer="candidate", turns=2, compactions=0, usage={"requests": 2, "input_tokens": 3, "output_tokens": 4, "total_tokens": 7})

    async def judge(runtime, reference_answer: str, candidate_answer: str, stage_dir: Path, attempts: int, mode: str, clip_max_turns: int):
        assert reference_answer == "secret gold"
        assert candidate_answer == "candidate"
        return JudgeResult(score=5), {"requests": 1, "input_tokens": 2, "output_tokens": 3, "total_tokens": 5}

    monkeypatch.setattr("src.training.run_judge", judge)
    item = {"index": 9, "question": "question only", "answer": "secret gold"}
    result = await run_evaluation_rollout(Runtime(), item, 0, "current skill", tmp_path / "rollout", 3, "formal", 12)
    assert result["score"] == 5
    assert captured["question"] == "question only"
    assert captured["skill"] == "current skill"
    assert captured["clip_max_turns"] == 12
    assert json.loads((tmp_path / "rollout" / "input.json").read_text()) == {"question_index": 9, "question": "question only", "rollout_index": 0, "skill_sha256": hashlib.sha256(b"current skill").hexdigest()}


def test_agent_raise_retries_only_in_formal_mode() -> None:
    error = AgentRaised("stop")
    assert is_retryable_error(error, "formal") is True
    assert is_retryable_error(error, "debug") is False
    wrapped = UserError("wrapped")
    wrapped.__cause__ = error
    assert is_retryable_error(wrapped, "formal") is True
    assert is_retryable_error(wrapped, "debug") is False
    wrapped.__cause__ = httpx2.ReadTimeout("timed out")
    assert is_retryable_error(wrapped, "formal") is True
    assert is_retryable_error(wrapped, "debug") is True


async def test_cancelled_stage_does_not_save_terminal_error(tmp_path: Path) -> None:
    async def cancelled_call(attempt_dir: Path) -> tuple[JudgeResult, dict[str, int]]:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await run_stage(tmp_path / "stage", JudgeResult, 1, "formal", cancelled_call)
    assert not (tmp_path / "stage" / "terminal_error.json").exists()
    assert (tmp_path / "stage" / "attempt-01" / "cancelled.json").exists()

    async def completed_call(attempt_dir: Path) -> tuple[JudgeResult, dict[str, int]]:
        return JudgeResult(score=5), {"requests": 1, "input_tokens": 2, "output_tokens": 3, "total_tokens": 5}

    output, _ = await run_stage(tmp_path / "stage", JudgeResult, 1, "formal", completed_call)
    assert output.score == 5
    assert (tmp_path / "stage" / "attempt-02" / "output.json").exists()


async def test_exhausted_evaluation_rollout_is_zero_scored(tmp_path: Path) -> None:
    rollout_dir = tmp_path / "rollout"
    (rollout_dir / "attempt-01").mkdir(parents=True)
    item = {"index": 9, "question": "question only", "answer": "secret gold"}
    result = await run_evaluation_rollout(object(), item, 0, "current skill", rollout_dir, 1, "formal", 12)
    assert result["status"] == "failed"
    assert result["score"] == 0


async def test_clip_qa_continues_after_structured_output_error(tmp_path: Path, monkeypatch) -> None:
    database_path = tmp_path / "clip.duckdb"
    connection = duckdb.connect(str(database_path))
    connection.execute("CREATE TABLE videos (video_id VARCHAR, duration_s DOUBLE)")
    connection.execute("INSERT INTO videos VALUES ('4', 10.0)")
    connection.close()
    runtime = SimpleNamespace(database_path=database_path, model="test-model", model_settings=ModelSettings(), run_config=object(), context_limit=262144)
    rollout = RolloutContext(runtime=runtime, attempt_dir=tmp_path / "attempt", clip_max_turns=10)
    inputs = []

    class InvalidResult:
        current_turn = 3

        def to_input_list(self):
            return [{"role": "user", "content": "original question"}]

    class ValidResult:
        current_turn = 1

        def final_output_as(self, output_type, raise_if_incorrect_type: bool):
            return ClipAnswer(status="unknown", answer="insufficient evidence", evidence_timestamps_s=[])

    async def fake_stream_agent(agent, input_items, context, max_turns: int, run_config, trace_path: Path, context_limit: int, allow_compaction: bool):
        inputs.append(input_items)
        if len(inputs) == 1:
            failed_output = [{"role": "assistant", "type": "message", "content": [{"type": "output_text", "text": "plain text"}]}]
            raise StreamModelBehaviorError(ModelBehaviorError("Invalid JSON when parsing model output"), InvalidResult(), failed_output)
        return ValidResult(), False

    monkeypatch.setattr("src.harness.stream_agent", fake_stream_agent)
    output = await clip_qa.__wrapped__(RunContextWrapper(rollout), "4", 0.0, 2.0, "What is visible?")
    assert json.loads(output[0].text) == {"status": "unknown", "answer": "insufficient evidence", "evidence_timestamps_s": [], "evidence_files": []}
    assert inputs[1][-2] == {"role": "assistant", "type": "message", "content": [{"type": "output_text", "text": "plain text"}]}
    assert "Invalid JSON when parsing model output" in inputs[1][-1]["content"]


def test_reflection_trace_excludes_duplicate_representations(tmp_path: Path) -> None:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    payload = {"input": "question", "is_complete": True, "current_turn": 1, "responses": [{"output": [{"type": "message", "content": "answer"}]}], "items": [{"type": "message", "content": "answer"}], "events": [{"type": "response.output_text.delta", "delta": "answer"}], "usage": {"total_tokens": 5}, "final_output": "answer"}
    (trace_dir / "main-segment-00.json").write_text(json.dumps(payload))
    reflected = rollout_trace(tmp_path)
    assert reflected == [{"path": "traces/main-segment-00.json", "trace": {"input": "question", "is_complete": True, "current_turn": 1, "items": [{"type": "message", "content": "answer"}]}}]
