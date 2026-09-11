import asyncio
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from src import trace2skill_baseline as baseline
from src import val


@pytest.fixture
def eval_case(tmp_path, monkeypatch):
    answer = AsyncMock(return_value=SimpleNamespace(status="answered", answer="package, quiz, auction"))
    harnesses = []

    def make_harness(**kwargs):
        harness = SimpleNamespace(context=SimpleNamespace(run_dir=kwargs["run_dir"], model_settings=SimpleNamespace()), answer=answer, close=AsyncMock())
        harnesses.append((kwargs, harness))
        return harness

    monkeypatch.setattr(val, "VideoSQLHarness", make_harness)
    judge = AsyncMock(return_value=5)
    monkeypatch.setattr(val, "judge_answer", judge)
    slots = asyncio.Semaphore(1)
    kwargs = dict(semaphore=slots, question={"timeline": "day", "index": 7, "question": "Order the events", "answer": "package, quiz, auction"}, rollout=0, skill_dir=tmp_path / "skills", runtime_root=tmp_path / "runtime", run_dir=tmp_path, model_name=baseline.MODEL, api_key="local", embedding_url=baseline.EMBEDDING_URL, embedding_model="qwen3-vl-embedding-8b", attempts=3, base_url=baseline.BASE_URL, clip_qa_video=True)
    return kwargs, answer, judge, harnesses


def test_judge_resume_reuses_answer_and_releases_main_slot(eval_case, tmp_path):
    kwargs, answer, judge, harnesses = eval_case
    root = tmp_path / "val/question-day-7/rollout-00"
    events = []

    async def qa(question):
        assert kwargs["semaphore"].locked()
        events.append({"stage": "qa", "input": question, "output": "package, quiz, auction"})
        return SimpleNamespace(status="answered", answer="package, quiz, auction")

    async def judgment(harness, question, reference, candidate, attempts):
        assert not kwargs["semaphore"].locked()
        assert json.loads((root / "answer.json").read_text())["answer"] == candidate
        events.append({"stage": "judge", "candidate": candidate})
        if len(events) == 2:
            raise RuntimeError("judge service unavailable")
        return 5

    answer.side_effect = qa
    judge.side_effect = judgment
    with pytest.raises(RuntimeError, match="judge service unavailable"):
        asyncio.run(val.run_eval_rollout(**kwargs))
    assert not (root / "result.json").exists()
    assert not (root / "attempt-1/error.json").exists()
    result = asyncio.run(val.run_eval_rollout(**kwargs))
    assert result["score"] == 5
    assert answer.await_count == 1
    assert judge.await_count == 2
    assert len(harnesses) == 3
    for settings, harness in harnesses:
        assert settings["base_url"] == baseline.BASE_URL
        assert settings["clip_qa_video"] is True
        assert harness.context.model_settings.reasoning == {"effort": "xhigh", "summary": "auto"}
        harness.close.assert_awaited_once()
    assert harnesses[0][1].context.model_settings.extra_body == {"prompt_cache_key": f"{tmp_path.name}:val:day:7"}
    assert asyncio.run(val.run_eval_rollout(**kwargs)) == result
    assert len(harnesses) == 3
    (tmp_path / "val-lifecycle.json").write_text(json.dumps(events, indent=2) + "\n")


def test_interrupted_attempt_is_archived(eval_case, tmp_path):
    kwargs, answer, judge, harnesses = eval_case
    root = tmp_path / "val/question-day-7/rollout-00"
    trace = root / "attempt-1/traces/answer-segment-00.json"
    trace.parent.mkdir(parents=True)
    trace.write_text('{"status":"interrupted","input":"Order the events"}')
    result = asyncio.run(val.run_eval_rollout(**kwargs))
    archived = list(root.glob("interrupted-1-*/traces/answer-segment-00.json"))
    assert len(archived) == 1
    assert json.loads(archived[0].read_text())["status"] == "interrupted"
    assert str(archived[0]) in result["trace_files"]
    assert result["attempt"] == 1
    assert result["status"] == "answered"


@pytest.mark.parametrize("failure", ["invalid_status", "retryable", "deterministic", "cancelled"])
def test_qa_failures_are_retried_skipped_or_raised(eval_case, tmp_path, failure):
    kwargs, answer, judge, harnesses = eval_case
    root = tmp_path / "val/question-day-7/rollout-00"
    if failure == "invalid_status":
        answer.return_value = SimpleNamespace(status="compaction_limit", answer=None)
    elif failure == "retryable":
        answer.side_effect = val.InvalidTrace("malformed model answer")
    elif failure == "deterministic":
        answer.side_effect = RuntimeError("missing video data")
    else:
        answer.side_effect = asyncio.CancelledError()
    if failure in ("invalid_status", "retryable"):
        assert asyncio.run(val.run_eval_rollout(**kwargs)) is None
        assert answer.await_count == 3
        assert json.loads((root / "skipped.json").read_text())["reason"] == "qa_attempts_exhausted"
        assert asyncio.run(val.run_eval_rollout(**kwargs)) is None
        assert answer.await_count == 3
    else:
        with pytest.raises(RuntimeError if failure == "deterministic" else asyncio.CancelledError):
            asyncio.run(val.run_eval_rollout(**kwargs))
        assert answer.await_count == 1
        assert not (root / "skipped.json").exists()
    if failure == "cancelled":
        assert not (root / "attempt-1/error.json").exists()
    assert not (root / "result.json").exists()
    assert not kwargs["semaphore"].locked()
    judge.assert_not_awaited()
    for settings, harness in harnesses:
        harness.close.assert_awaited_once()


def test_judge_exhaustion_skips_without_zero_score(eval_case, tmp_path):
    kwargs, answer, judge, harnesses = eval_case
    judge.side_effect = val.JudgeAttemptsExhausted("judge attempts exhausted")
    assert asyncio.run(val.run_eval_rollout(**kwargs)) is None
    root = tmp_path / "val/question-day-7/rollout-00"
    assert (root / "answer.json").exists()
    assert not (root / "result.json").exists()
    assert json.loads((root / "skipped.json").read_text())["reason"] == "judge_attempts_exhausted"
    assert asyncio.run(val.run_eval_rollout(**kwargs)) is None
    assert answer.await_count == judge.await_count == 1


def test_main_restores_completed_count_and_excludes_skips(tmp_path, monkeypatch):
    args = SimpleNamespace(api_key_env=None, run_dir=tmp_path / "run", model_name=baseline.MODEL, base_url=baseline.BASE_URL, embedding_url=baseline.EMBEDDING_URL, embedding_model="qwen3-vl-embedding-8b", attempts=3, rollouts=1, concurrency=15, clip_qa_video=True, skill_dir=tmp_path / "skills", test_file=tmp_path / "questions.json", runtime_root=tmp_path / "runtime")
    args.test_file.write_text(json.dumps([{"timeline": "day", "index": i} for i in range(3)]))
    saved = args.run_dir / "val/question-day-0/rollout-00/result.json"
    saved.parent.mkdir(parents=True)
    saved.write_text(json.dumps({"question_index": 0, "rollout": 0, "status": "answered", "score": 5}))
    skipped = args.run_dir / "val/question-day-1/rollout-00/skipped.json"
    skipped.parent.mkdir(parents=True)
    skipped.write_text('{"reason":"qa_attempts_exhausted"}')
    run = Mock()
    monkeypatch.setattr(val.wandb, "init", Mock(return_value=run))
    monkeypatch.setattr(val, "init_weave", Mock())
    monkeypatch.setattr(val.weave, "finish", Mock())
    monkeypatch.setattr(val, "milestone", Mock())
    monkeypatch.setattr(val.subprocess, "run", Mock(return_value=SimpleNamespace(stdout="unit-test-commit")))
    monkeypatch.setattr(val, "dotenv_values", Mock(side_effect=AssertionError("Local serving must not load a cloud API key")))
    execute = AsyncMock(return_value=None)
    monkeypatch.setattr(val, "run_eval_rollout", execute)
    asyncio.run(val.main_async(args))
    assert run.log.call_args_list[0].args[0] == {"val/completed_rollouts": 1, "val/skipped_rollouts": 1}
    execute.assert_awaited_once()
    assert execute.call_args.args[1]["index"] == 2
    result = json.loads((args.run_dir / "result.json").read_text())
    assert result == {"requested_rollouts": 3, "rollouts": 1, "skipped_rollouts": 2, "score_sum": 5, "accuracy": 1.0}
    config = json.loads((args.run_dir / "config.json").read_text())
    assert config["reasoning"] == {"effort": "xhigh", "summary": "auto"}
    assert config["main_agent_concurrency"] == 15
    assert config["client_concurrency"] is None
    assert config["answer_context_limit"] == 128000
    val.weave.finish.assert_called_once()
    run.finish.assert_called_once()


def test_cli_defaults_match_current_local_rollout(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["val", "--test-file", str(tmp_path / "questions.json"), "--runtime-root", str(tmp_path), "--run-dir", str(tmp_path)])
    execute = AsyncMock()
    monkeypatch.setattr(val, "main_async", execute)
    val.main()
    args = execute.call_args.args[0]
    assert args.model_name == baseline.MODEL
    assert args.base_url == baseline.BASE_URL
    assert args.embedding_url == baseline.EMBEDDING_URL
    assert args.concurrency == baseline.MAIN_AGENT_CONCURRENCY
    assert args.clip_qa_video is True
    assert args.api_key_env is None
