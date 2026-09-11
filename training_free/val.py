import argparse
import asyncio
import json
import subprocess
import uuid
from pathlib import Path
from typing import Any

import httpx
import httpx2
import openai
import wandb
import weave
from agents import (
    Agent,
    MaxTurnsExceeded,
    ModelBehaviorError,
    StopAtTools,
    UserError,
    function_tool,
)
from dotenv import dotenv_values
from pydantic import BaseModel, ValidationError
from wandb.sdk.lib.runid import generate_id
from training_free.tracing_bridge import init_weave

from training_free.harness import (
    AgentRaised,
    ANSWER_CONTEXT_LIMIT,
    CLIP_IMAGE_LIMIT,
    FRAME_SIZE,
    InvalidTrace,
    VideoSQLHarness,
    exception_chain,
    run_stream,
    tool_error_json,
)


class Judgment(BaseModel):
    score: int


class JudgeAttemptsExhausted(RuntimeError):
    pass


def retryable(error: BaseException) -> bool:
    chain = exception_chain(error)
    for item in chain:
        if isinstance(item, openai.APIError) and (
            item.code == "insufficient_quota" or "no credits remaining" in str(item).lower()
        ):
            return False
    for item in chain:
        if isinstance(
            item,
            (
                AgentRaised,
                InvalidTrace,
                MaxTurnsExceeded,
                ModelBehaviorError,
                ValidationError,
                UserError,
                httpx2.RemoteProtocolError,
                httpx.RemoteProtocolError,
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.RateLimitError,
                openai.InternalServerError,
            ),
        ):
            return True
        if type(item) is openai.APIError:
            return True
    return False


def write_error(path: Path, error: BaseException) -> None:
    if any(isinstance(item, openai.APIError) and (
        item.code == "insufficient_quota" or "no credits remaining" in str(item).lower()
    ) for item in exception_chain(error)):
        # Preserve the pause without marking this QA/judge attempt as exhausted.
        path = path.with_name(f"{path.stem}-quota-pause-{uuid.uuid4().hex}.json")
    elif any(isinstance(item, asyncio.CancelledError) for item in exception_chain(error)):
        # SDK stream cleanup can wrap cancellation; it still did not exhaust an attempt.
        path = path.with_name(f"{path.stem}-cancelled-{uuid.uuid4().hex}.json")
    print(f"error_file={path} " + " <- ".join(
        f"{type(item).__name__}: {item}" for item in exception_chain(error)
    ), flush=True)
    path.write_text(
        json.dumps(
            {
                "type": type(error).__name__,
                "message": str(error),
                "chain": [
                    {"type": type(item).__name__, "message": str(item)}
                    for item in exception_chain(error)
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


def milestone(message: str) -> None:
    print(message, flush=True)
    with Path("AGENT_STDOUT.txt").open("a") as output:
        output.write(message + "\n")


async def judge_answer(
    harness: VideoSQLHarness,
    question: str,
    reference_answer: str,
    candidate_answer: str,
    attempts: int,
) -> int:
    @function_tool(failure_error_function=tool_error_json)
    async def submit_judgment(score: int) -> str:
        """Submit an integer semantic-equivalence score from 0 through 5."""
        if not 0 <= score <= 5:
            raise ValueError("score must be from 0 through 5")
        return Judgment(score=score).model_dump_json()

    prompt = json.dumps(
        {
            "question": question,
            "reference_answer": reference_answer,
            "candidate_answer": candidate_answer,
        },
        ensure_ascii=False,
    )
    instructions = (
        "Judge only whether the candidate final answer is semantically equivalent to the "
        "reference final answer in the context of the original question. Interpret option "
        "numbers using the question when present. Ignore incidental video IDs, timestamps, row IDs, and other "
        "identifiers. Return 5 for fully equivalent and 0 for wrong, with intermediate integers "
        "for partially correct answers. Call submit_judgment."
    )
    for attempt in range(1, attempts + 1):
        error_path = harness.context.run_dir / f"judge-attempt-{attempt}-error.json"
        if error_path.exists():
            if json.loads(error_path.read_text())["type"] != "CancelledError":
                continue
            error_path.rename(error_path.with_name(f"judge-attempt-{attempt}-cancelled-{uuid.uuid4().hex}.json"))
        agent = Agent(
            name="judge",
            instructions=instructions,
            model=harness.context.model,
            model_settings=harness.context.model_settings,
            tools=[submit_judgment],
            tool_use_behavior=StopAtTools(stop_at_tool_names=["submit_judgment"]),
        )
        try:
            result, _, _ = await run_stream(
                harness.context,
                agent,
                prompt,
                f"judge-attempt-{attempt}",
                False,
            )
            return Judgment.model_validate_json(str(result.final_output)).score
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            write_error(
                harness.context.run_dir / f"judge-attempt-{attempt}-error.json",
                error,
            )
            if not retryable(error):
                raise
    raise JudgeAttemptsExhausted("judge attempts exhausted")


async def run_eval_rollout(
    semaphore: asyncio.Semaphore,
    question: dict[str, Any],
    rollout: int,
    skill_dir: Path,
    runtime_root: Path,
    run_dir: Path,
    model_name: str,
    api_key: str,
    embedding_url: str,
    embedding_model: str,
    attempts: int,
    *,
    base_url: str,
    clip_qa_video: bool,
) -> dict[str, Any] | None:
    rollout_root = (
        run_dir
        / "val"
        / f"question-{question['timeline']}-{question['index']}"
        / f"rollout-{rollout:02d}"
    )
    result_path = rollout_root / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    if (rollout_root / "skipped.json").exists():
        print(f"SKIP saved exhausted rollout: {rollout_root}", flush=True)
        return None
    answer_path = rollout_root / "answer.json"
    if answer_path.exists():
        answer = json.loads(answer_path.read_text())
    else:
        async with semaphore:
            for attempt in range(1, attempts + 1):
                attempt_dir = rollout_root / f"attempt-{attempt}"
                if (attempt_dir / "error.json").exists():
                    continue
                if attempt_dir.exists():
                    attempt_dir.rename(rollout_root / f"interrupted-{attempt}-{uuid.uuid4().hex}")
                attempt_dir.mkdir(parents=True)
                harness = VideoSQLHarness(
                    database=runtime_root / question["timeline"] / "index.duckdb",
                    embedding_index=runtime_root / question["timeline"] / "embedding_index",
                    media_map=runtime_root / question["timeline"] / "media-map.json",
                    skill_dir=skill_dir, run_dir=attempt_dir, model_name=model_name, api_key=api_key,
                    base_url=base_url, embedding_url=embedding_url, embedding_model=embedding_model,
                    clip_qa_video=clip_qa_video,
                )
                harness.context.model_settings.reasoning = {"effort": "xhigh", "summary": "auto"}
                harness.context.model_settings.extra_body = {"prompt_cache_key": f"{run_dir.name}:val:{question['timeline']}:{question['index']}"}
                try:
                    session = await harness.answer(question["question"])
                    if session.status != "answered" or session.answer is None:
                        raise InvalidTrace(f"QA ended without an answer: {session.status}")
                    answer = {
                        "question_index": question["index"], "timeline": question["timeline"],
                        "rollout": rollout, "attempt": attempt, "status": session.status,
                        "answer": session.answer, "reference_answer": question["answer"],
                    }
                    answer_path.write_text(json.dumps(answer, ensure_ascii=False, indent=2) + "\n")
                    print(f"ANSWER phase=val timeline={question['timeline']} question={question['index']} rollout={rollout}", flush=True)
                    break
                except asyncio.CancelledError:
                    raise
                except BaseException as error:
                    write_error(attempt_dir / "error.json", error)
                    if not retryable(error):
                        raise
                finally:
                    await harness.close()
            else:
                (rollout_root / "skipped.json").write_text(json.dumps({"reason": "qa_attempts_exhausted"}) + "\n")
                print(f"SKIP QA attempts exhausted: {rollout_root}", flush=True)
                return None

    attempt_dir = rollout_root / f"attempt-{answer['attempt']}"
    harness = VideoSQLHarness(
        database=runtime_root / question["timeline"] / "index.duckdb",
        embedding_index=runtime_root / question["timeline"] / "embedding_index",
        media_map=runtime_root / question["timeline"] / "media-map.json",
        skill_dir=skill_dir, run_dir=attempt_dir, model_name=model_name, api_key=api_key,
        base_url=base_url, embedding_url=embedding_url, embedding_model=embedding_model,
        clip_qa_video=clip_qa_video,
    )
    harness.context.model_settings.reasoning = {"effort": "xhigh", "summary": "auto"}
    try:
        judgment_path = rollout_root / "judgment.json"
        if judgment_path.exists():
            score = json.loads(judgment_path.read_text())["score"]
        else:
            try:
                score = await judge_answer(harness, question["question"], question["answer"], answer["answer"], attempts)
            except JudgeAttemptsExhausted as error:
                (rollout_root / "skipped.json").write_text(json.dumps({"reason": "judge_attempts_exhausted", "error": str(error)}) + "\n")
                print(f"SKIP judge attempts exhausted: {rollout_root}", flush=True)
                return None
            judgment_path.write_text(json.dumps({"score": score}) + "\n")
        result = {**answer, "score": score, "trace_files": sorted(str(path) for path in rollout_root.glob("*/traces/*.json"))}
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(f"SCORE phase=val timeline={question['timeline']} question={question['index']} rollout={rollout} score={score}", flush=True)
        return result
    finally:
        await harness.close()


async def main_async(args: argparse.Namespace) -> None:
    api_key = "local"
    if args.api_key_env is not None:
        api_key = dotenv_values(Path(__file__).resolve().parents[1] / ".env")[args.api_key_env]
        if not api_key:
            raise ValueError(f"{args.api_key_env} must have a value in the project .env")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    main_git_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    config_path = args.run_dir / "config.json"
    if config_path.exists():
        saved_config = json.loads(config_path.read_text())
        wandb_id = saved_config["wandb_id"]
        resume = "must"
    else:
        wandb_id = generate_id()
        resume = "never"
    config = {
        "model": args.model_name, "base_url": args.base_url,
        "reasoning": {"effort": "xhigh", "summary": "auto"}, "reasoning_replay": True,
        "embedding_url": args.embedding_url, "embedding_model": args.embedding_model,
        "attempts": args.attempts, "rollouts": args.rollouts,
        "main_agent_concurrency": args.concurrency, "client_concurrency": None,
        "answer_context_limit": ANSWER_CONTEXT_LIMIT, "forced_answer_after_limit": True,
        "sql_scope_required": True, "replace_tool": True, "clip_qa_video": args.clip_qa_video,
        "frame_size": list(FRAME_SIZE), "clip_image_limit": CLIP_IMAGE_LIMIT,
        "skill_dir": str(args.skill_dir), "main_git_hash": main_git_hash, "wandb_id": wandb_id,
    }
    if resume == "must":
        for key in config:
            if key in ("main_agent_concurrency", "client_concurrency", "main_git_hash", "wandb_id"):
                continue
            if key not in saved_config or saved_config[key] != config[key]:
                raise ValueError(f"Existing validation config differs for {key}; use a new --run-dir")
        config["main_git_hash"] = saved_config["main_git_hash"]
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    init_weave()
    wandb_run = wandb.init(
        entity="3dsmile",
        project="video-sql-v2",
        id=wandb_id,
        name=args.run_dir.name,
        config=config,
        resume=resume,
        dir=args.run_dir,
    )
    try:
        questions = json.loads(args.test_file.read_text())
        semaphore = asyncio.Semaphore(args.concurrency)
        all_results = []
        skipped = 0
        async with asyncio.TaskGroup() as group:
            tasks = []
            for question in questions:
                for rollout in range(args.rollouts):
                    directory = args.run_dir / "val" / f"question-{question['timeline']}-{question['index']}" / f"rollout-{rollout:02d}"
                    if (directory / "result.json").exists():
                        all_results.append(json.loads((directory / "result.json").read_text()))
                    elif (directory / "skipped.json").exists():
                        skipped += 1
                    else:
                        tasks.append(group.create_task(run_eval_rollout(
                            semaphore, question, rollout, args.skill_dir, args.runtime_root,
                            args.run_dir, args.model_name, api_key, args.embedding_url,
                            args.embedding_model, args.attempts, base_url=args.base_url,
                            clip_qa_video=args.clip_qa_video,
                        )))
            wandb_run.log({"val/completed_rollouts": len(all_results), "val/skipped_rollouts": skipped})
            for task in asyncio.as_completed(tasks):
                result = await task
                if result is None:
                    skipped += 1
                else:
                    all_results.append(result)
                    wandb_run.log({"val/rollout_score": result["score"], "val/question_index": result["question_index"], "val/rollout": result["rollout"]})
                wandb_run.log({"val/completed_rollouts": len(all_results), "val/skipped_rollouts": skipped})
                if all_results:
                    accuracy = sum(result["score"] for result in all_results) / (5 * len(all_results))
                    wandb_run.log({"val/accuracy": accuracy})
                    milestone(f"val completed={len(all_results)} skipped={skipped} accuracy={accuracy:.6f}")
        if not all_results:
            raise RuntimeError("No scored rollouts in val")
        final = {
            "requested_rollouts": len(questions) * args.rollouts,
            "rollouts": len(all_results),
            "skipped_rollouts": skipped,
            "score_sum": sum(result["score"] for result in all_results),
            "accuracy": sum(result["score"] for result in all_results)
            / (5 * len(all_results)),
        }
        (args.run_dir / "result.json").write_text(
            json.dumps(final, ensure_ascii=False, indent=2) + "\n"
        )
        milestone(
            f"validation complete rollouts={final['rollouts']} "
            f"accuracy={final['accuracy']:.6f}"
        )
    finally:
        weave.finish()
        wandb_run.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--skill-dir", type=Path, default=Path(__file__).resolve().parents[1] / "skills/initial")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3.8-27B-FP8")
    parser.add_argument("--base-url", default="http://127.0.0.1:8003/v1")
    parser.add_argument("--api-key-env", help="Project .env variable; omit for a local server")
    parser.add_argument("--embedding-url", default="http://127.0.0.1:8014/v1")
    parser.add_argument("--embedding-model", default="qwen3-vl-embedding-8b")
    parser.add_argument("--clip-qa-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=15)
    parser.add_argument("--rollouts", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
