import argparse
import asyncio
import dataclasses
import hashlib
import json
import shutil
import subprocess
import traceback
from pathlib import Path
from typing import Awaitable, Callable, Literal, TypeVar

import httpx
import httpx2
import wandb
import yaml
from agents import Agent, RunContextWrapper, StopAtTools, function_tool
from agents.exceptions import MaxTurnsExceeded, ModelBehaviorError, ModelTimeoutError, UserError
from openai import APIConnectionError, APIError, APIResponseValidationError, APITimeoutError, AuthenticationError, BadRequestError, ConflictError, InternalServerError, NotFoundError, PermissionDeniedError, RateLimitError, UnprocessableEntityError
from pydantic import BaseModel, Field
from wandb.sdk.lib.runid import generate_id

from src.harness import MAX_TURNS, META_SKILL_NAME, SUBMIT_REMINDER, AgentRaised, PilotHarness, RolloutContext, add_note, agent_raise, clip_qa, list_skill_files, read_notes, run_sql, semantic_search, skill_frontmatter, stop_when_submitted, stream_agent


class JudgeResult(BaseModel):
    score: int = Field(ge=0, le=5)


class ReflectionResult(BaseModel):
    reflection: str
    meta_reflection: str


class QuestionReflectionSummary(BaseModel):
    reflection_summary: str
    meta_reflection_summary: str


class SkillUpdate(BaseModel):
    commit: str
    changed_files: list[str]


class StageAttemptsExhausted(RuntimeError):
    pass


OutputModel = TypeVar("OutputModel", bound=BaseModel)
GIT_IDENTITY = ["-c", "user.name=codex", "-c", "user.email=codex@localhost"]


def git(skills_dir: Path, *arguments: str) -> str:
    return subprocess.run(["git", *GIT_IDENTITY, *arguments], cwd=skills_dir, check=True, capture_output=True, text=True).stdout


def skills_digest(skills_dir: Path) -> str:
    return hashlib.sha256(json.dumps([[path.name, path.read_text()] for path in sorted(skills_dir.glob("*.md"))]).encode()).hexdigest()


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
    return isinstance(candidate, (httpx.RequestError, httpx2.TimeoutException))


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


async def run_one_call_once(runtime: PilotHarness, name: str, instructions: str, input_payload: object, output_type: type[OutputModel], attempt_dir: Path, skills_dir: Path) -> tuple[OutputModel, dict[str, int]]:
    context = RolloutContext(runtime=runtime, attempt_dir=attempt_dir, skills_dir=skills_dir)
    agent = Agent(name=name, instructions=instructions, model=runtime.model, model_settings=runtime.model_settings, output_type=output_type)
    result, _ = await stream_agent(agent, json.dumps(input_payload, ensure_ascii=False), context, 1, runtime.run_config, attempt_dir / "trace.json", runtime.context_limit, False, [])
    return result.final_output_as(output_type, raise_if_incorrect_type=True), usage_payload(result)


async def run_judge(runtime: PilotHarness, reference_answer: str, candidate_answer: str, stage_dir: Path, attempts: int, mode: Literal["debug", "formal"], skills_dir: Path) -> tuple[JudgeResult, dict[str, int]]:
    input_payload = {"reference_answer": reference_answer, "candidate_answer": candidate_answer}
    lock_stage_input(stage_dir / "input.json", input_payload)
    instructions = "Compare only the reference answer and candidate answer for semantic answer equivalence. Return an integer score from 0 to 5, where 5 is fully correct, 0 is wrong, and intermediate values represent partial correctness. For numeric answers, count a number only when it is asserted as answer content; ignore incidental video IDs, timestamps, row IDs, and other identifiers."

    async def call(attempt_dir: Path) -> tuple[JudgeResult, dict[str, int]]:
        return await run_one_call_once(runtime, "answer_judge", instructions, input_payload, JudgeResult, attempt_dir, skills_dir)

    return await run_stage(stage_dir, JudgeResult, attempts, mode, call)


@function_tool(name_override="submit_answer", failure_error_function=None)
async def submit_reflection(context: RunContextWrapper[RolloutContext], reflection: str, meta_reflection: str) -> str:
    """Submit the final reflection for future retrieval and meta_reflection for future reflection. Calling this ends the reflection."""
    context.context.submission = ReflectionResult(reflection=reflection, meta_reflection=meta_reflection)
    return json.dumps({"submitted": True})


async def run_reflection_once(runtime: PilotHarness, input_payload: object, meta_skill: str, attempt_dir: Path, skills_dir: Path) -> tuple[ReflectionResult, dict[str, int]]:
    context = RolloutContext(runtime=runtime, attempt_dir=attempt_dir, skills_dir=skills_dir)
    instructions = f"Re-examine the rollout trace, its final answer, the reference answer, and the score. Use the retrieval tools when additional database or visual inspection is needed to establish why the approach worked or failed. When done, call submit_answer with two reusable abstractions: reflection for future retrieval and meta_reflection for future reflection. Remove dataset-specific names, dates, answer values, and private details from both outputs. Update, replace, or reject prior guidance when appropriate instead of only adding advice.\n\nCurrent reflection skill:\n{meta_skill}"
    agent = Agent(name="rollout_reflection", instructions=instructions, tools=[run_sql, semantic_search, clip_qa, add_note, read_notes, agent_raise, submit_reflection], model=runtime.model, model_settings=runtime.model_settings, tool_use_behavior=stop_when_submitted)
    active_agent = agent
    input_items: str | list[dict[str, object]] = json.dumps(input_payload, ensure_ascii=False)
    used_turns = 0
    segment = 0
    tools_enabled = True
    completions: list[str] = []
    usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    async with httpx.AsyncClient(base_url=runtime.model_base_url, timeout=None) as tokenizer_client:
        print(json.dumps({"stage": "reflection_tokenize", "input_chars": len(input_items)}))
        token_response = await tokenizer_client.post("/tokenize", json={"model": runtime.model_name, "prompt": input_items})
        token_response.raise_for_status()
        input_tokens = token_response.json()["count"]
    if input_tokens >= runtime.context_limit - 50_000:
        active_agent = agent.clone(tools=[submit_reflection])
        tools_enabled = False
    while True:
        if used_turns >= MAX_TURNS:
            raise MaxTurnsExceeded(f"Reflection exceeded {MAX_TURNS} total turns")
        result, final_required = await stream_agent(active_agent, input_items, context, MAX_TURNS - used_turns, runtime.run_config, attempt_dir / f"reflection-segment-{segment:02d}.json", runtime.context_limit, tools_enabled, completions)
        used_turns += result.current_turn
        segment_usage = usage_payload(result)
        for key in usage:
            usage[key] += segment_usage[key]
        if context.submission is not None:
            return context.submission, usage
        if final_required:
            active_agent = agent.clone(tools=[submit_reflection])
            tools_enabled = False
            input_items = result.to_input_list() + [{"role": "user", "content": "The context limit is near. Do not call other tools. Call submit_answer with the abstract reflection and meta_reflection now."}]
        else:
            input_items = result.to_input_list() + [{"role": "user", "content": SUBMIT_REMINDER}]
        segment += 1


async def run_reflection(runtime: PilotHarness, input_payload: object, meta_skill: str, stage_dir: Path, attempts: int, mode: Literal["debug", "formal"], skills_dir: Path) -> tuple[ReflectionResult, dict[str, int]]:
    lock_stage_input(stage_dir / "input.json", input_payload)

    async def call(attempt_dir: Path) -> tuple[ReflectionResult, dict[str, int]]:
        return await run_reflection_once(runtime, input_payload, meta_skill, attempt_dir, skills_dir)

    return await run_stage(stage_dir, ReflectionResult, attempts, mode, call)


def rollout_trace(attempt_dir: Path) -> list[dict[str, object]]:
    trace_path = sorted((attempt_dir / "traces").glob("main-segment-*.json"))[-1]
    trace = json.loads(trace_path.read_text())
    trace = {key: trace[key] for key in ("input", "is_complete", "current_turn", "items")}
    for item in trace["items"]:
        if item["type"] == "function_call":
            item["arguments"] = json.loads(item["arguments"])
        elif item["type"] == "function_call_output":
            if isinstance(item["output"], str):
                item["output"] = json.loads(item["output"])
            else:
                assert isinstance(item["output"], list)
                for part in item["output"]:
                    if part["type"] == "input_text":
                        part["text"] = json.loads(part["text"])
    return [{"path": str(trace_path.relative_to(attempt_dir)), "trace": trace}]


async def run_training_rollout(runtime: PilotHarness, item: dict[str, object], rollout_index: int, skills_dir: Path, rollout_dir: Path, attempts: int, mode: Literal["debug", "formal"]) -> dict[str, object]:
    result_path = rollout_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    if (rollout_dir / "terminal_error.json").exists():
        raise RuntimeError(f"Rollout has a saved non-retryable failure: {rollout_dir}")
    rollout_dir.mkdir(parents=True, exist_ok=True)
    lock_stage_input(rollout_dir / "input.json", {"question_index": item["index"], "question": item["question"], "rollout_index": rollout_index, "skills_sha256": skills_digest(skills_dir)})
    meta_skill = (skills_dir / f"{META_SKILL_NAME}.md").read_text()
    current_skills = [{"name": path.stem, "content": path.read_text()} for path in list_skill_files(skills_dir)]
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
            retrieval = await runtime.run_rollout(str(item["question"]), skills_dir, attempt_dir)
            write_json(attempt_dir / "retrieval.json", dataclasses.asdict(retrieval))
            if retrieval.status == "compaction_limit":
                score = 0
                judge_payload = None
            else:
                judge, judge_usage = await run_judge(runtime, str(item["answer"]), retrieval.answer, attempt_dir / "judge", attempts, mode, skills_dir)
                score = judge.score
                judge_payload = {"score": judge.score, "usage": judge_usage}
            reflection_input = {"question": item["question"], "candidate_answer": retrieval.answer, "reference_answer": item["answer"], "score": score, "rollout_status": retrieval.status, "rollout_trace": rollout_trace(attempt_dir), "current_skills": current_skills}
            if retrieval.status == "compaction_limit":
                reflection_input["focus"] = "Identify the reusable causes of repeated or circular retrieval that forced the third compaction."
            reflection, reflection_usage = await run_reflection(runtime, reflection_input, meta_skill, attempt_dir / "reflection", attempts, mode, skills_dir)
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


async def run_evaluation_rollout(runtime: PilotHarness, item: dict[str, object], rollout_index: int, skills_dir: Path, rollout_dir: Path, attempts: int, mode: Literal["debug", "formal"]) -> dict[str, object]:
    result_path = rollout_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    if (rollout_dir / "terminal_error.json").exists():
        raise RuntimeError(f"Rollout has a saved non-retryable failure: {rollout_dir}")
    rollout_dir.mkdir(parents=True, exist_ok=True)
    lock_stage_input(rollout_dir / "input.json", {"question_index": item["index"], "question": item["question"], "rollout_index": rollout_index, "skills_sha256": skills_digest(skills_dir)})
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
            retrieval = await runtime.run_rollout(str(item["question"]), skills_dir, attempt_dir)
            write_json(attempt_dir / "retrieval.json", dataclasses.asdict(retrieval))
            if retrieval.status == "compaction_limit":
                score = 0
                judge_payload = None
            else:
                judge, judge_usage = await run_judge(runtime, str(item["answer"]), retrieval.answer, attempt_dir / "judge", attempts, mode, skills_dir)
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


async def summarize_question(runtime: PilotHarness, reflections: list[dict[str, str]], stage_dir: Path, attempts: int, mode: Literal["debug", "formal"], skills_dir: Path) -> QuestionReflectionSummary:
    input_payload = {"rollout_reflections": reflections}
    lock_stage_input(stage_dir / "input.json", input_payload)
    instructions = "Combine the rollout reflections for one question into one reusable retrieval summary and one reusable reflection-process summary. Remove question-specific names, dates, answers, and private details. Reconcile conflicts and repetition rather than listing every item."

    async def call(attempt_dir: Path) -> tuple[QuestionReflectionSummary, dict[str, int]]:
        return await run_one_call_once(runtime, "question_reflection_summarizer", instructions, input_payload, QuestionReflectionSummary, attempt_dir, skills_dir)

    output, _ = await run_stage(stage_dir, QuestionReflectionSummary, attempts, mode, call)
    return output


@function_tool(failure_error_function=None)
async def list_skills(context: RunContextWrapper[RolloutContext]) -> str:
    """List the current skill files with their descriptions, plus META-SKILL."""
    skills_dir = context.context.skills_dir
    entries = [skill_frontmatter(path) for path in list_skill_files(skills_dir)]
    entries.append({"name": META_SKILL_NAME, "description": "How to reflect on a rollout"})
    return json.dumps(entries, ensure_ascii=False)


@function_tool(failure_error_function=None)
async def read_current_skill(context: RunContextWrapper[RolloutContext], name: str) -> str:
    """Read the full current content of one skill file by name (use META-SKILL for the reflection skill)."""
    path = context.context.skills_dir / f"{name}.md"
    if not path.is_file():
        return json.dumps({"error": f"No skill file named {name!r}"})
    return path.read_text()


@function_tool(failure_error_function=None)
async def write_skill(context: RunContextWrapper[RolloutContext], name: str, content: str) -> str:
    """Create or fully overwrite one skill file. Skill files need YAML frontmatter with name equal to the file name and a description saying when to use it; META-SKILL is plain Markdown."""
    skills_dir = context.context.skills_dir
    if "/" in name or name.startswith(".") or not name:
        return json.dumps({"error": "name must be a plain file name without extension"})
    path = skills_dir / f"{name}.md"
    previous = path.read_text() if path.is_file() else None
    path.write_text(content)
    if name != META_SKILL_NAME:
        try:
            skill_frontmatter(path)
        except (ValueError, KeyError, TypeError, yaml.YAMLError) as error:
            if previous is None:
                path.unlink()
            else:
                path.write_text(previous)
            return json.dumps({"error": f"Invalid skill frontmatter, file not written: {error}"})
    return json.dumps({"written": name, "characters": len(content)})


@function_tool(failure_error_function=None)
async def delete_skill(context: RunContextWrapper[RolloutContext], name: str) -> str:
    """Delete one skill file by name. META-SKILL cannot be deleted."""
    path = context.context.skills_dir / f"{name}.md"
    if name == META_SKILL_NAME:
        return json.dumps({"error": "META-SKILL cannot be deleted"})
    if not path.is_file():
        return json.dumps({"error": f"No skill file named {name!r}"})
    path.unlink()
    return json.dumps({"deleted": name})


@function_tool(failure_error_function=None)
async def git_history(context: RunContextWrapper[RolloutContext], name: str | None) -> str:
    """List past commits of the skill repository, newest first, optionally restricted to one skill file name."""
    arguments = ["log", "--format=%H%x09%ad%x09%s", "--date=iso-strict", "--name-status"]
    if name is not None:
        arguments += ["--", f"{name}.md"]
    return git(context.context.skills_dir, *arguments)


@function_tool(failure_error_function=None)
async def read_skill_at_commit(context: RunContextWrapper[RolloutContext], name: str, commit: str) -> str:
    """Read the full content of one skill file as it was at a past commit from git_history."""
    try:
        return git(context.context.skills_dir, "show", f"{commit}:{name}.md")
    except subprocess.CalledProcessError as error:
        return json.dumps({"error": error.stderr})


@function_tool(failure_error_function=None)
async def stop(context: RunContextWrapper[RolloutContext]) -> str:
    """Finish the skill update after all file operations are done."""
    return json.dumps({"stopped": True})


async def update_skills_once(runtime: PilotHarness, summaries: list[dict[str, str]], batch_label: str, expected_commit: str, attempt_dir: Path, skills_dir: Path) -> tuple[SkillUpdate, dict[str, int]]:
    head = git(skills_dir, "rev-parse", "HEAD").strip()
    if head != expected_commit:
        raise RuntimeError(f"Skills repository HEAD {head} differs from the updater input commit {expected_commit}")
    git(skills_dir, "checkout", "--", ".")
    git(skills_dir, "clean", "-fdq")
    context = RolloutContext(runtime=runtime, attempt_dir=attempt_dir, skills_dir=skills_dir)
    instructions = "Update the retrieval skill files and META-SKILL from the question summaries. Incorporate only reusable lessons supported by the summaries. Add a skill file for a new recurring situation, rewrite files whose guidance is weak, and delete files that are redundant, conflicting, obsolete, or overly specific. Each skill file is <name>.md with YAML frontmatter (name equal to the file name, description stating when to use it) followed by Introduction, When to use, and Main content sections; put SQL examples inline in the Markdown. Use git_history and read_skill_at_commit to consult earlier versions and rewrite from them when a change should be reverted. Do not leak dataset-specific names, dates, answers, or private details. Call stop when all file operations are complete."
    agent = Agent(name="skill_updater", instructions=instructions, tools=[list_skills, read_current_skill, write_skill, delete_skill, git_history, read_skill_at_commit, stop], model=runtime.model, model_settings=runtime.model_settings, tool_use_behavior=StopAtTools(stop_at_tool_names=["stop"]))
    active_agent = agent
    input_items: str | list[dict[str, object]] = json.dumps({"question_summaries": summaries}, ensure_ascii=False)
    used_turns = 0
    segment = 0
    tools_enabled = True
    completions: list[str] = []
    usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    while True:
        if used_turns >= MAX_TURNS:
            raise MaxTurnsExceeded(f"Skill updater exceeded {MAX_TURNS} total turns")
        result, final_required = await stream_agent(active_agent, input_items, context, MAX_TURNS - used_turns, runtime.run_config, attempt_dir / f"updater-segment-{segment:02d}.json", runtime.context_limit, tools_enabled, completions)
        used_turns += result.current_turn
        segment_usage = usage_payload(result)
        for key in usage:
            usage[key] += segment_usage[key]
        if any(item.type == "tool_call_item" and item.raw_item.name == "stop" for item in result.new_items):
            break
        if final_required:
            active_agent = agent.clone(tools=[stop])
            tools_enabled = False
            input_items = result.to_input_list() + [{"role": "user", "content": "The context limit is near. Do not call other tools. Call stop now."}]
        else:
            input_items = result.to_input_list() + [{"role": "user", "content": "Your last response did not call stop. Continue the file operations if needed, then call stop."}]
        segment += 1
    git(skills_dir, "add", "-A")
    changed_files = git(skills_dir, "diff", "--cached", "--name-only").split()
    git(skills_dir, "commit", "-q", "--allow-empty", "-m", f"Skill update after {batch_label}")
    commit = git(skills_dir, "rev-parse", "HEAD").strip()
    return SkillUpdate(commit=commit, changed_files=changed_files), usage


async def update_skills(runtime: PilotHarness, summaries: list[dict[str, str]], batch_label: str, stage_dir: Path, attempts: int, mode: Literal["debug", "formal"], skills_dir: Path) -> SkillUpdate:
    input_payload = {"skills_commit": git(skills_dir, "rev-parse", "HEAD").strip(), "question_summaries": summaries}
    lock_stage_input(stage_dir / "input.json", input_payload)

    async def call(attempt_dir: Path) -> tuple[SkillUpdate, dict[str, int]]:
        return await update_skills_once(runtime, summaries, batch_label, input_payload["skills_commit"], attempt_dir, skills_dir)

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
    wandb_id = generate_id()
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
    validate_positive({"questions_per_batch": args.questions_per_batch, "rollouts_per_question": args.rollouts_per_question, "epochs": args.epochs, "attempts": args.attempts, "context_limit": args.context_limit})
    items = json.loads(args.train_json.read_text())
    revision, status = git_state()
    launch_config = {"command": "training", "git_hash": revision, "git_status": status, "database": str(args.database.resolve()), "index_root": str(args.index_root.resolve()), "clip_root": str(args.clip_root.resolve()), "train_json": str(args.train_json.resolve()), "skills_dir": str(args.skills_dir.resolve()), "skills_sha256": skills_digest(args.skills_dir), "model_base_url": args.model_base_url, "model_name": args.model_name, "embedding_base_url": args.embedding_base_url, "context_limit": args.context_limit, "questions_per_batch": args.questions_per_batch, "rollouts_per_question": args.rollouts_per_question, "epochs": args.epochs, "attempts": args.attempts, "max_turns": MAX_TURNS, "mode": args.mode}
    wandb_run = initialize_wandb(args.run_dir, launch_config, args.wandb_entity, args.wandb_project, args.wandb_run_name)
    skills_dir = args.run_dir / "skills"
    if not skills_dir.exists():
        shutil.copytree(args.skills_dir, skills_dir)
        git(skills_dir, "init", "-q")
        git(skills_dir, "add", "-A")
        git(skills_dir, "commit", "-q", "-m", "Initial skills")
    runtime = PilotHarness(args.database, args.index_root, args.clip_root, args.model_base_url, args.model_name, args.embedding_base_url, args.context_limit)
    global_batch = 0
    expected_commit = None
    try:
        for epoch in range(args.epochs):
            for batch_start in range(0, len(items), args.questions_per_batch):
                batch_items = items[batch_start:batch_start + args.questions_per_batch]
                batch_dir = args.run_dir / "train" / f"epoch-{epoch:03d}" / f"batch-{global_batch:06d}"
                completed_path = batch_dir / "completed.json"
                if completed_path.exists():
                    expected_commit = json.loads((batch_dir / "skills-after.json").read_text())["commit"]
                    global_batch += 1
                    continue
                batch_dir.mkdir(parents=True, exist_ok=True)
                head = git(skills_dir, "rev-parse", "HEAD").strip()
                if expected_commit is not None and head != expected_commit:
                    raise RuntimeError(f"Skills repository HEAD {head} differs from the last completed batch commit {expected_commit}")
                lock_stage_input(batch_dir / "skills-before.json", {"commit": head})
                async with asyncio.TaskGroup() as rollout_group:
                    rollout_tasks = []
                    for item in batch_items:
                        for rollout_index in range(args.rollouts_per_question):
                            rollout_dir = batch_dir / "questions" / f"question-{int(item['index']):06d}" / f"rollout-{rollout_index:02d}"
                            rollout_tasks.append(rollout_group.create_task(run_training_rollout(runtime, item, rollout_index, skills_dir, rollout_dir, args.attempts, args.mode)))
                results = [task.result() for task in rollout_tasks]
                async with asyncio.TaskGroup() as summary_group:
                    summary_tasks = []
                    for item in batch_items:
                        question_results = [result for result in results if result["question_index"] == item["index"] and result["status"] != "failed"]
                        reflections = [result["reflection"] for result in question_results]
                        if reflections:
                            summary_dir = batch_dir / "question-summaries" / f"question-{int(item['index']):06d}"
                            summary_tasks.append(summary_group.create_task(summarize_question(runtime, reflections, summary_dir, args.attempts, args.mode, skills_dir)))
                summaries = [task.result() for task in summary_tasks]
                update = await update_skills(runtime, [summary.model_dump(mode="json") for summary in summaries], f"epoch {epoch} batch {global_batch}", batch_dir / "updater", args.attempts, args.mode, skills_dir)
                write_json(batch_dir / "skills-after.json", update.model_dump(mode="json"))
                expected_commit = update.commit
                scores = [int(result["score"]) for result in results]
                completed_rollouts = sum(result["status"] != "failed" for result in results)
                failed_rollouts = len(results) - completed_rollouts
                completed = {"epoch": epoch, "global_batch": global_batch, "question_indices": [item["index"] for item in batch_items], "completed_rollouts": completed_rollouts, "failed_rollouts": failed_rollouts, "scores": scores, "skills_commit": update.commit, "changed_skill_files": update.changed_files}
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
    parser.add_argument("--skills-dir", type=Path, required=True)
    parser.add_argument("--model-base-url", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--embedding-base-url", required=True)
    parser.add_argument("--context-limit", type=int, required=True)
    parser.add_argument("--questions-per-batch", type=int, required=True)
    parser.add_argument("--rollouts-per-question", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--attempts", type=int, required=True)
    parser.add_argument("--mode", choices=["debug", "formal"], required=True)
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-run-name", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(train(parse_args()))
