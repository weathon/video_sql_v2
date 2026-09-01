import argparse
import asyncio
import dataclasses
import hashlib
import json
import subprocess
import traceback
from pathlib import Path
from typing import Awaitable, Callable, Literal, TypeVar

import httpx
import wandb
from agents import Agent
from agents.exceptions import MaxTurnsExceeded, ModelBehaviorError, ModelTimeoutError, UserError
from openai import APIConnectionError, APIError, APIResponseValidationError, APITimeoutError, AuthenticationError, BadRequestError, ConflictError, InternalServerError, NotFoundError, PermissionDeniedError, RateLimitError, UnprocessableEntityError
from pydantic import BaseModel, Field

from src.harness import AgentRaised, PilotHarness, RolloutContext, StreamModelBehaviorError, add_note, agent_raise, clip_qa, read_notes, run_sql, semantic_search, stream_agent


class JudgeResult(BaseModel):
    score: int = Field(ge=0, le=5)


class ReflectionResult(BaseModel):
    reflection: str
    meta_reflection: str


class QuestionReflectionSummary(BaseModel):
    reflection_summary: str
    meta_reflection_summary: str


class SkillUpdate(BaseModel):
    skill_md: str
    meta_skill_md: str


class StageAttemptsExhausted(RuntimeError):
    pass


OutputModel = TypeVar("OutputModel", bound=BaseModel)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def lock_stage_input(path: Path, payload: object) -> None:
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise RuntimeError(f"Saved stage input differs from current input: {path}")
        return
    write_json(path, payload)


def usage_payload(result: object) -> dict[str, int]:
    usage = result.context_wrapper.usage
    return {"requests": usage.requests, "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens, "total_tokens": usage.total_tokens}


def is_retryable_error(error: BaseException, mode: Literal["debug", "formal"]) -> bool:
    candidate = error.__cause__ if isinstance(error, UserError) and error.__cause__ is not None else error
    if isinstance(candidate, AgentRaised):
        return mode == "formal"
    if isinstance(candidate, (BadRequestError, AuthenticationError, PermissionDeniedError, NotFoundError, ConflictError, UnprocessableEntityError)):
        return False
    if isinstance(candidate, (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError, APIResponseValidationError, ModelBehaviorError, ModelTimeoutError, MaxTurnsExceeded)):
        return True
    if type(candidate) is APIError:
        return True
    if isinstance(candidate, httpx.HTTPStatusError):
        return candidate.response.status_code == 429 or candidate.response.status_code >= 500
    return isinstance(candidate, httpx.RequestError)


async def run_stage(stage_dir: Path, output_type: type[OutputModel], attempts: int, mode: Literal["debug", "formal"], call: Callable[[Path], Awaitable[tuple[OutputModel, dict[str, int]]]]) -> tuple[OutputModel, dict[str, int]]:
    output_path = stage_dir / "output.json"
    usage_path = stage_dir / "usage.json"
    if output_path.exists():
        return output_type.model_validate_json(output_path.read_text()), json.loads(usage_path.read_text())
    if (stage_dir / "terminal_error.json").exists():
        raise RuntimeError(f"Stage has a saved non-retryable failure: {stage_dir}")
    if (stage_dir / "attempts_exhausted.json").exists():
        raise StageAttemptsExhausted(f"Stage retry attempts are exhausted: {stage_dir}")
    stage_dir.mkdir(parents=True, exist_ok=True)
    all_attempts = sorted(path for path in stage_dir.glob("attempt-*") if path.is_dir())
    for prior_attempt in all_attempts:
        if (prior_attempt / "output.json").exists():
            output = output_type.model_validate_json((prior_attempt / "output.json").read_text())
            usage = json.loads((prior_attempt / "usage.json").read_text())
            write_json(output_path, output.model_dump(mode="json"))
            write_json(usage_path, usage)
            return output, usage
    prior_attempts = [path for path in all_attempts if not (path / "cancelled.json").exists()]
    if len(prior_attempts) >= attempts:
        write_json(stage_dir / "attempts_exhausted.json", {"attempts": len(prior_attempts)})
        raise StageAttemptsExhausted(f"Stage retry attempts are exhausted: {stage_dir}")
    next_attempt_number = max((int(path.name.removeprefix("attempt-")) for path in all_attempts), default=0) + 1
    for attempt_index in range(len(prior_attempts), attempts):
        attempt_number = next_attempt_number + attempt_index - len(prior_attempts)
        attempt_dir = stage_dir / f"attempt-{attempt_number:02d}"
        attempt_dir.mkdir(parents=True)
        write_json(attempt_dir / "started.json", {"attempt": attempt_number})
        try:
            output, usage = await call(attempt_dir)
            write_json(attempt_dir / "output.json", output.model_dump(mode="json"))
            write_json(attempt_dir / "usage.json", usage)
            write_json(output_path, output.model_dump(mode="json"))
            write_json(usage_path, usage)
            return output, usage
        except asyncio.CancelledError:
            write_json(attempt_dir / "cancelled.json", {"attempt": attempt_number})
            raise
        except Exception as error:
            error_payload = {"type": type(error).__name__, "message": str(error), "traceback": "".join(traceback.format_exception(error))}
            write_json(attempt_dir / "error.json", error_payload)
            if not is_retryable_error(error, mode):
                write_json(stage_dir / "terminal_error.json", error_payload)
                raise
            if attempt_index + 1 == attempts:
                write_json(stage_dir / "attempts_exhausted.json", {"attempts": attempts, "last_error": error_payload})
                raise StageAttemptsExhausted(f"Stage retry attempts are exhausted: {stage_dir}") from error
    raise RuntimeError("Unreachable stage retry state")


async def run_one_call_once(runtime: PilotHarness, name: str, instructions: str, input_payload: object, output_type: type[OutputModel], attempt_dir: Path, clip_max_turns: int) -> tuple[OutputModel, dict[str, int]]:
    context = RolloutContext(runtime=runtime, attempt_dir=attempt_dir, clip_max_turns=clip_max_turns)
    agent = Agent(name=name, instructions=instructions, model=runtime.model, model_settings=runtime.model_settings, output_type=output_type)
    result, _ = await stream_agent(agent, json.dumps(input_payload, ensure_ascii=False), context, 1, runtime.run_config, attempt_dir / "trace.json", runtime.context_limit, False)
    return result.final_output_as(output_type, raise_if_incorrect_type=True), usage_payload(result)


async def run_judge(runtime: PilotHarness, reference_answer: str, candidate_answer: str, stage_dir: Path, attempts: int, mode: Literal["debug", "formal"], clip_max_turns: int) -> tuple[JudgeResult, dict[str, int]]:
    input_payload = {"reference_answer": reference_answer, "candidate_answer": candidate_answer}
    lock_stage_input(stage_dir / "input.json", input_payload)
    instructions = "Compare only the reference answer and candidate answer for semantic answer equivalence. Return an integer score from 0 to 5, where 5 is fully correct, 0 is wrong, and intermediate values represent partial correctness. For numeric answers, count a number only when it is asserted as answer content; ignore incidental video IDs, timestamps, row IDs, and other identifiers."

    async def call(attempt_dir: Path) -> tuple[JudgeResult, dict[str, int]]:
        return await run_one_call_once(runtime, "answer_judge", instructions, input_payload, JudgeResult, attempt_dir, clip_max_turns)

    return await run_stage(stage_dir, JudgeResult, attempts, mode, call)


async def run_reflection_once(runtime: PilotHarness, input_payload: object, meta_skill: str, attempt_dir: Path, clip_max_turns: int, reflection_max_turns: int) -> tuple[ReflectionResult, dict[str, int]]:
    context = RolloutContext(runtime=runtime, attempt_dir=attempt_dir, clip_max_turns=clip_max_turns)
    instructions = f"Re-examine the rollout trace, its final answer, the reference answer, and the score. Use the retrieval tools when additional database or visual inspection is needed to establish why the approach worked or failed. Return two reusable abstractions: reflection for future retrieval and meta_reflection for future reflection. Remove dataset-specific names, dates, answer values, and private details from both outputs. Update, replace, or reject prior guidance when appropriate instead of only adding advice.\n\nCurrent reflection skill:\n{meta_skill}"
    agent = Agent(name="rollout_reflection", instructions=instructions, tools=[run_sql, semantic_search, clip_qa, add_note, read_notes, agent_raise], model=runtime.model, model_settings=runtime.model_settings, output_type=ReflectionResult)
    active_agent = agent
    input_items: str | list[dict[str, object]] = json.dumps(input_payload, ensure_ascii=False)
    used_turns = 0
    segment = 0
    tools_enabled = True
    usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    while True:
        if used_turns >= reflection_max_turns:
            raise MaxTurnsExceeded(f"Reflection exceeded {reflection_max_turns} total turns")
        try:
            result, final_required = await stream_agent(active_agent, input_items, context, reflection_max_turns - used_turns, runtime.run_config, attempt_dir / f"reflection-segment-{segment:02d}.json", runtime.context_limit, tools_enabled)
        except StreamModelBehaviorError as error:
            result = error.result
            used_turns += result.current_turn
            segment_usage = usage_payload(result)
            for key in usage:
                usage[key] += segment_usage[key]
            if used_turns >= reflection_max_turns:
                raise MaxTurnsExceeded(f"Reflection exceeded {reflection_max_turns} total turns") from error
            input_items = result.to_input_list() + [{"role": "user", "content": f"Your output could not be parsed as ReflectionResult JSON: {error}. Continue evidence checking if needed. When ready, return exactly the required structured output with reflection and meta_reflection."}]
            segment += 1
            continue
        used_turns += result.current_turn
        segment_usage = usage_payload(result)
        for key in usage:
            usage[key] += segment_usage[key]
        if final_required:
            if used_turns >= reflection_max_turns:
                raise MaxTurnsExceeded(f"Reflection reached {reflection_max_turns} turns before its forced final response")
            active_agent = agent.clone(tools=[])
            tools_enabled = False
            input_items = result.to_input_list() + [{"role": "user", "content": "The context limit is near. Do not call tools. Return the required abstract reflection and meta_reflection now."}]
            segment += 1
            continue
        return result.final_output_as(ReflectionResult, raise_if_incorrect_type=True), usage


async def run_reflection(runtime: PilotHarness, input_payload: object, meta_skill: str, stage_dir: Path, attempts: int, mode: Literal["debug", "formal"], clip_max_turns: int, reflection_max_turns: int) -> tuple[ReflectionResult, dict[str, int]]:
    lock_stage_input(stage_dir / "input.json", input_payload)

    async def call(attempt_dir: Path) -> tuple[ReflectionResult, dict[str, int]]:
        return await run_reflection_once(runtime, input_payload, meta_skill, attempt_dir, clip_max_turns, reflection_max_turns)

    return await run_stage(stage_dir, ReflectionResult, attempts, mode, call)


def rollout_trace(attempt_dir: Path) -> list[dict[str, object]]:
    traces = []
    for trace_path in sorted((attempt_dir / "traces").glob("*.json")):
        traces.append({"path": str(trace_path.relative_to(attempt_dir)), "trace": json.loads(trace_path.read_text())})
    return traces


async def run_training_rollout(runtime: PilotHarness, item: dict[str, object], rollout_index: int, skill: str, meta_skill: str, rollout_dir: Path, attempts: int, mode: Literal["debug", "formal"], clip_max_turns: int, reflection_max_turns: int) -> dict[str, object]:
    result_path = rollout_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    if (rollout_dir / "terminal_error.json").exists():
        raise RuntimeError(f"Rollout has a saved non-retryable failure: {rollout_dir}")
    rollout_dir.mkdir(parents=True, exist_ok=True)
    lock_stage_input(rollout_dir / "input.json", {"question_index": item["index"], "question": item["question"], "rollout_index": rollout_index, "skill_sha256": hashlib.sha256(skill.encode()).hexdigest()})
    all_attempts = sorted(path for path in rollout_dir.glob("attempt-*") if path.is_dir())
    prior_attempts = [path for path in all_attempts if not (path / "cancelled.json").exists()]
    if len(prior_attempts) >= attempts:
        failure = {"status": "failed", "question_index": item["index"], "rollout_index": rollout_index, "score": 0, "attempts": len(prior_attempts), "reason": "rollout_attempts_exhausted"}
        write_json(result_path, failure)
        return failure
    next_attempt_number = max((int(path.name.removeprefix("attempt-")) for path in all_attempts), default=0) + 1
    for attempt_index in range(len(prior_attempts), attempts):
        attempt_number = next_attempt_number + attempt_index - len(prior_attempts)
        attempt_dir = rollout_dir / f"attempt-{attempt_number:02d}"
        attempt_dir.mkdir(parents=True)
        write_json(attempt_dir / "started.json", {"attempt": attempt_number})
        try:
            retrieval = await runtime.run_rollout(str(item["question"]), skill, attempt_dir, clip_max_turns)
            write_json(attempt_dir / "retrieval.json", dataclasses.asdict(retrieval))
            if retrieval.status == "compaction_limit":
                score = 0
                judge_payload = None
            else:
                judge, judge_usage = await run_judge(runtime, str(item["answer"]), retrieval.answer, attempt_dir / "judge", attempts, mode, clip_max_turns)
                score = judge.score
                judge_payload = {"score": judge.score, "usage": judge_usage}
            reflection_input = {"question": item["question"], "candidate_answer": retrieval.answer, "reference_answer": item["answer"], "score": score, "rollout_status": retrieval.status, "rollout_trace": rollout_trace(attempt_dir), "current_skill": skill}
            if retrieval.status == "compaction_limit":
                reflection_input["focus"] = "Identify the reusable causes of repeated or circular retrieval that forced the third compaction."
            reflection, reflection_usage = await run_reflection(runtime, reflection_input, meta_skill, attempt_dir / "reflection", attempts, mode, clip_max_turns, reflection_max_turns)
            payload = {"status": retrieval.status, "question_index": item["index"], "rollout_index": rollout_index, "attempt": attempt_number, "answer": retrieval.answer, "reference_answer": item["answer"], "score": score, "turns": retrieval.turns, "compactions": retrieval.compactions, "retrieval_usage": retrieval.usage, "judge": judge_payload, "reflection": reflection.model_dump(mode="json"), "reflection_usage": reflection_usage}
            write_json(result_path, payload)
            return payload
        except StageAttemptsExhausted as error:
            failure = {"status": "failed", "question_index": item["index"], "rollout_index": rollout_index, "score": 0, "attempt": attempt_number, "reason": str(error)}
            write_json(result_path, failure)
            return failure
        except asyncio.CancelledError:
            write_json(attempt_dir / "cancelled.json", {"attempt": attempt_number})
            raise
        except Exception as error:
            error_payload = {"type": type(error).__name__, "message": str(error), "traceback": "".join(traceback.format_exception(error))}
            write_json(attempt_dir / "error.json", error_payload)
            if not is_retryable_error(error, mode):
                write_json(rollout_dir / "terminal_error.json", error_payload)
                raise
            if attempt_index + 1 == attempts:
                failure = {"status": "failed", "question_index": item["index"], "rollout_index": rollout_index, "score": 0, "attempts": attempts, "reason": error_payload}
                write_json(result_path, failure)
                return failure
    raise RuntimeError("Unreachable rollout retry state")


async def run_evaluation_rollout(runtime: PilotHarness, item: dict[str, object], rollout_index: int, skill: str, rollout_dir: Path, attempts: int, mode: Literal["debug", "formal"], clip_max_turns: int) -> dict[str, object]:
    result_path = rollout_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    if (rollout_dir / "terminal_error.json").exists():
        raise RuntimeError(f"Rollout has a saved non-retryable failure: {rollout_dir}")
    rollout_dir.mkdir(parents=True, exist_ok=True)
    lock_stage_input(rollout_dir / "input.json", {"question_index": item["index"], "question": item["question"], "rollout_index": rollout_index, "skill_sha256": hashlib.sha256(skill.encode()).hexdigest()})
    all_attempts = sorted(path for path in rollout_dir.glob("attempt-*") if path.is_dir())
    prior_attempts = [path for path in all_attempts if not (path / "cancelled.json").exists()]
    if len(prior_attempts) >= attempts:
        failure = {"status": "failed", "question_index": item["index"], "rollout_index": rollout_index, "score": 0, "attempts": len(prior_attempts), "reason": "rollout_attempts_exhausted"}
        write_json(result_path, failure)
        return failure
    next_attempt_number = max((int(path.name.removeprefix("attempt-")) for path in all_attempts), default=0) + 1
    for attempt_index in range(len(prior_attempts), attempts):
        attempt_number = next_attempt_number + attempt_index - len(prior_attempts)
        attempt_dir = rollout_dir / f"attempt-{attempt_number:02d}"
        attempt_dir.mkdir(parents=True)
        write_json(attempt_dir / "started.json", {"attempt": attempt_number})
        try:
            retrieval = await runtime.run_rollout(str(item["question"]), skill, attempt_dir, clip_max_turns)
            write_json(attempt_dir / "retrieval.json", dataclasses.asdict(retrieval))
            if retrieval.status == "compaction_limit":
                score = 0
                judge_payload = None
            else:
                judge, judge_usage = await run_judge(runtime, str(item["answer"]), retrieval.answer, attempt_dir / "judge", attempts, mode, clip_max_turns)
                score = judge.score
                judge_payload = {"score": judge.score, "usage": judge_usage}
            payload = {"status": retrieval.status, "question_index": item["index"], "rollout_index": rollout_index, "attempt": attempt_number, "answer": retrieval.answer, "reference_answer": item["answer"], "score": score, "turns": retrieval.turns, "compactions": retrieval.compactions, "retrieval_usage": retrieval.usage, "judge": judge_payload}
            write_json(result_path, payload)
            return payload
        except StageAttemptsExhausted as error:
            failure = {"status": "failed", "question_index": item["index"], "rollout_index": rollout_index, "score": 0, "attempt": attempt_number, "reason": str(error)}
            write_json(result_path, failure)
            return failure
        except asyncio.CancelledError:
            write_json(attempt_dir / "cancelled.json", {"attempt": attempt_number})
            raise
        except Exception as error:
            error_payload = {"type": type(error).__name__, "message": str(error), "traceback": "".join(traceback.format_exception(error))}
            write_json(attempt_dir / "error.json", error_payload)
            if not is_retryable_error(error, mode):
                write_json(rollout_dir / "terminal_error.json", error_payload)
                raise
            if attempt_index + 1 == attempts:
                failure = {"status": "failed", "question_index": item["index"], "rollout_index": rollout_index, "score": 0, "attempts": attempts, "reason": error_payload}
                write_json(result_path, failure)
                return failure
    raise RuntimeError("Unreachable rollout retry state")


async def summarize_question(runtime: PilotHarness, reflections: list[dict[str, str]], stage_dir: Path, attempts: int, mode: Literal["debug", "formal"], clip_max_turns: int) -> QuestionReflectionSummary:
    input_payload = {"rollout_reflections": reflections}
    lock_stage_input(stage_dir / "input.json", input_payload)
    instructions = "Combine the rollout reflections for one question into one reusable retrieval summary and one reusable reflection-process summary. Remove question-specific names, dates, answers, and private details. Reconcile conflicts and repetition rather than listing every item."

    async def call(attempt_dir: Path) -> tuple[QuestionReflectionSummary, dict[str, int]]:
        return await run_one_call_once(runtime, "question_reflection_summarizer", instructions, input_payload, QuestionReflectionSummary, attempt_dir, clip_max_turns)

    output, _ = await run_stage(stage_dir, QuestionReflectionSummary, attempts, mode, call)
    return output


async def update_skills(runtime: PilotHarness, skill: str, meta_skill: str, summaries: list[dict[str, str]], stage_dir: Path, attempts: int, mode: Literal["debug", "formal"], clip_max_turns: int) -> SkillUpdate:
    input_payload = {"current_skill_md": skill, "current_meta_skill_md": meta_skill, "question_summaries": summaries}
    lock_stage_input(stage_dir / "input.json", input_payload)
    instructions = "Return complete replacement contents for SKILL.md and META-SKILL.md. Incorporate only reusable lessons supported by the summaries. Modify or replace weak guidance and remove redundant, conflicting, obsolete, or overly specific guidance. Do not leak dataset-specific names, dates, answers, or private details."

    async def call(attempt_dir: Path) -> tuple[SkillUpdate, dict[str, int]]:
        return await run_one_call_once(runtime, "skill_updater", instructions, input_payload, SkillUpdate, attempt_dir, clip_max_turns)

    output, _ = await run_stage(stage_dir, SkillUpdate, attempts, mode, call)
    return output


def git_state() -> tuple[str, str]:
    revision = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--short"], check=True, capture_output=True, text=True).stdout
    return revision, status


def initialize_wandb(run_dir: Path, launch_config: dict[str, object], entity: str, project: str, run_name: str):
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    if config_path.exists():
        saved = json.loads(config_path.read_text())
        if saved["launch"] != launch_config:
            raise RuntimeError("Saved run configuration differs from the current launch configuration")
        run = wandb.init(entity=entity, project=project, name=run_name, id=saved["wandb_id"], resume="must", config=launch_config)
        if run is None:
            raise RuntimeError("wandb.init returned no run")
        return run
    wandb_id = wandb.util.generate_id()
    run = wandb.init(entity=entity, project=project, name=run_name, id=wandb_id, resume="never", config=launch_config)
    if run is None:
        raise RuntimeError("wandb.init returned no run")
    write_json(config_path, {"wandb_id": wandb_id, "launch": launch_config})
    return run


def milestone(run_dir: Path, payload: dict[str, object]) -> None:
    line = json.dumps(payload, ensure_ascii=False)
    with (run_dir / "AGENT_STDOUT.txt").open("a") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def validate_positive(values: dict[str, int]) -> None:
    invalid = {name: value for name, value in values.items() if value <= 0}
    if invalid:
        raise ValueError(f"Positive integer arguments required: {invalid}")


async def train(args: argparse.Namespace) -> None:
    validate_positive({"questions_per_batch": args.questions_per_batch, "rollouts_per_question": args.rollouts_per_question, "epochs": args.epochs, "attempts": args.attempts, "clip_max_turns": args.clip_max_turns, "reflection_max_turns": args.reflection_max_turns, "context_limit": args.context_limit})
    items = json.loads(args.train_json.read_text())
    skill = args.skill_path.read_text()
    meta_skill = args.meta_skill_path.read_text()
    revision, status = git_state()
    launch_config = {"command": "training", "git_hash": revision, "git_status": status, "database": str(args.database.resolve()), "index_root": str(args.index_root.resolve()), "clip_root": str(args.clip_root.resolve()), "train_json": str(args.train_json.resolve()), "skill_sha256": hashlib.sha256(skill.encode()).hexdigest(), "meta_skill_sha256": hashlib.sha256(meta_skill.encode()).hexdigest(), "model_base_url": args.model_base_url, "model_name": args.model_name, "embedding_base_url": args.embedding_base_url, "context_limit": args.context_limit, "questions_per_batch": args.questions_per_batch, "rollouts_per_question": args.rollouts_per_question, "epochs": args.epochs, "attempts": args.attempts, "clip_max_turns": args.clip_max_turns, "reflection_max_turns": args.reflection_max_turns, "mode": args.mode}
    wandb_run = initialize_wandb(args.run_dir, launch_config, args.wandb_entity, args.wandb_project, args.wandb_run_name)
    runtime = PilotHarness(args.database, args.index_root, args.clip_root, args.model_base_url, args.model_name, args.embedding_base_url, args.context_limit)
    global_batch = 0
    try:
        for epoch in range(args.epochs):
            for batch_start in range(0, len(items), args.questions_per_batch):
                batch_items = items[batch_start:batch_start + args.questions_per_batch]
                batch_dir = args.run_dir / "train" / f"epoch-{epoch:03d}" / f"batch-{global_batch:06d}"
                completed_path = batch_dir / "completed.json"
                if completed_path.exists():
                    skill = (batch_dir / "SKILL-after.md").read_text()
                    meta_skill = (batch_dir / "META-SKILL-after.md").read_text()
                    global_batch += 1
                    continue
                batch_dir.mkdir(parents=True, exist_ok=True)
                if (batch_dir / "SKILL-before.md").exists():
                    if (batch_dir / "SKILL-before.md").read_text() != skill or (batch_dir / "META-SKILL-before.md").read_text() != meta_skill:
                        raise RuntimeError(f"Saved batch skill input differs from current skill state: {batch_dir}")
                else:
                    (batch_dir / "SKILL-before.md").write_text(skill)
                    (batch_dir / "META-SKILL-before.md").write_text(meta_skill)
                tasks = []
                for item in batch_items:
                    for rollout_index in range(args.rollouts_per_question):
                        rollout_dir = batch_dir / "questions" / f"question-{int(item['index']):06d}" / f"rollout-{rollout_index:02d}"
                        tasks.append(run_training_rollout(runtime, item, rollout_index, skill, meta_skill, rollout_dir, args.attempts, args.mode, args.clip_max_turns, args.reflection_max_turns))
                results = await asyncio.gather(*tasks)
                summary_tasks = []
                for item in batch_items:
                    question_results = [result for result in results if result["question_index"] == item["index"] and result["status"] != "failed"]
                    reflections = [result["reflection"] for result in question_results]
                    if reflections:
                        summary_dir = batch_dir / "question-summaries" / f"question-{int(item['index']):06d}"
                        summary_tasks.append(summarize_question(runtime, reflections, summary_dir, args.attempts, args.mode, args.clip_max_turns))
                summaries = await asyncio.gather(*summary_tasks)
                update = await update_skills(runtime, skill, meta_skill, [summary.model_dump(mode="json") for summary in summaries], batch_dir / "updater", args.attempts, args.mode, args.clip_max_turns)
                skill = update.skill_md
                meta_skill = update.meta_skill_md
                (batch_dir / "SKILL-after.md").write_text(skill)
                (batch_dir / "META-SKILL-after.md").write_text(meta_skill)
                scores = [int(result["score"]) for result in results]
                completed_rollouts = sum(result["status"] != "failed" for result in results)
                failed_rollouts = len(results) - completed_rollouts
                completed = {"epoch": epoch, "global_batch": global_batch, "question_indices": [item["index"] for item in batch_items], "completed_rollouts": completed_rollouts, "failed_rollouts": failed_rollouts, "scores": scores}
                metrics = {"train/epoch": epoch, "train/completed_rollouts": completed_rollouts, "train/failed_rollouts": failed_rollouts, "train/mean_score": sum(scores) / len(scores)}
                wandb_run.log(metrics, step=global_batch)
                write_json(completed_path, completed)
                milestone(args.run_dir, {"event": "batch_completed", **completed, "mean_score": sum(scores) / len(scores)})
                global_batch += 1
        milestone(args.run_dir, {"event": "training_completed", "epochs": args.epochs, "batches": global_batch})
    finally:
        await runtime.close()
        wandb_run.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--clip-root", type=Path, required=True)
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--skill-path", type=Path, required=True)
    parser.add_argument("--meta-skill-path", type=Path, required=True)
    parser.add_argument("--model-base-url", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--embedding-base-url", required=True)
    parser.add_argument("--context-limit", type=int, required=True)
    parser.add_argument("--questions-per-batch", type=int, required=True)
    parser.add_argument("--rollouts-per-question", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--attempts", type=int, required=True)
    parser.add_argument("--clip-max-turns", type=int, required=True)
    parser.add_argument("--reflection-max-turns", type=int, required=True)
    parser.add_argument("--mode", choices=["debug", "formal"], required=True)
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-run-name", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(train(parse_args()))
