import hashlib
import json
from pathlib import Path

from agents.exceptions import ModelBehaviorError

from src.harness import AgentRaised, RolloutResult
from src.training import JudgeResult, is_retryable_error, run_evaluation_rollout, run_judge, run_stage


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
