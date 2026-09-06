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

from src.harness import (
    AgentRaised,
    CLIP_IMAGE_LIMIT,
    FRAME_SIZE,
    InvalidTrace,
    OPENAI_BASE_URL,
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
) -> dict[str, Any]:
    rollout_root = (
        run_dir
        / "val"
        / f"question-{question['timeline']}-{question['index']}"
        / f"rollout-{rollout:02d}"
    )
    result_path = rollout_root / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    async with semaphore:
        for attempt in range(1, attempts + 1):
            attempt_dir = rollout_root / f"attempt-{attempt}"
            if (attempt_dir / "error.json").exists():
                continue
            attempt_dir.mkdir(parents=True, exist_ok=True)
            harness = VideoSQLHarness(
                database=runtime_root / question["timeline"] / "index.duckdb",
                embedding_index=runtime_root
                / question["timeline"]
                / "embedding_index",
                media_map=runtime_root / question["timeline"] / "media-map.json",
                skill_dir=skill_dir,
                run_dir=attempt_dir,
                model_name=model_name,
                api_key=api_key,
                embedding_url=embedding_url,
                embedding_model=embedding_model,
            )
            try:
                session = await harness.answer(question["question"])
                score = (
                    0
                    if session.status == "compaction_limit"
                    else await judge_answer(
                        harness,
                        question["question"],
                        question["answer"],
                        session.answer,
                        attempts,
                    )
                )
                result = {
                    "question_index": question["index"],
                    "timeline": question["timeline"],
                    "rollout": rollout,
                    "attempt": attempt,
                    "status": session.status,
                    "answer": session.answer,
                    "reference_answer": question["answer"],
                    "score": score,
                    "trace_files": sorted(str(path) for path in (attempt_dir / "traces").glob("*.json")),
                }
                rollout_root.mkdir(parents=True, exist_ok=True)
                result_path.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2) + "\n"
                )
                return result
            except BaseException as error:
                write_error(attempt_dir / "error.json", error)
                if not retryable(error):
                    raise
            finally:
                await harness.close()
        result = {
            "question_index": question["index"],
            "timeline": question["timeline"],
            "rollout": rollout,
            "status": "failed",
            "answer": None,
            "reference_answer": question["answer"],
            "score": 0,
        }
        rollout_root.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        return result


async def main_async(args: argparse.Namespace) -> None:
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
        config = json.loads(config_path.read_text())
        wandb_id = config["wandb_id"]
        resume = "must"
    else:
        wandb_id = generate_id()
        resume = "never"
        config = {
            "model": args.model_name,
            "base_url": OPENAI_BASE_URL,
            "reasoning": {"effort": "medium", "context": "all_turns"},
            "embedding_url": args.embedding_url,
            "embedding_model": args.embedding_model,
            "attempts": args.attempts,
            "batch_size": args.batch_size,
            "rollouts": args.rollouts,
            "concurrency": args.batch_size * args.rollouts,
            "frame_size": list(FRAME_SIZE),
            "clip_image_limit": CLIP_IMAGE_LIMIT,
            "skill_dir": str(args.skill_dir),
            "main_git_hash": main_git_hash,
            "wandb_id": wandb_id,
        }
        config_path.write_text(json.dumps(config, indent=2) + "\n")
    weave.init("3dsmile/video-sql-v2")
    wandb_run = wandb.init(
        entity="3dsmile",
        project="video-sql-v2",
        id=wandb_id,
        name=args.run_dir.name,
        config=config,
        resume=resume,
        dir=args.run_dir,
    )
    questions = json.loads(args.test_file.read_text())
    semaphore = asyncio.Semaphore(args.batch_size * args.rollouts)
    all_results = []
    try:
        for batch_index, start in enumerate(range(0, len(questions), args.batch_size)):
            batch = questions[start : start + args.batch_size]
            tasks = [
                run_eval_rollout(
                    semaphore,
                    question,
                    rollout,
                    args.skill_dir,
                    args.runtime_root,
                    args.run_dir,
                    args.model_name,
                    api_key,
                    args.embedding_url,
                    args.embedding_model,
                    args.attempts,
                )
                for question in batch
                for rollout in range(args.rollouts)
            ]
            results = await asyncio.gather(*tasks)
            all_results.extend(results)
            for result in results:
                wandb_run.log(
                    {
                        "val/rollout_score": result["score"],
                        "val/question_index": result["question_index"],
                        "val/rollout": result["rollout"],
                    }
                )
            accuracy = sum(result["score"] for result in all_results) / (
                5 * len(all_results)
            )
            wandb_run.log(
                {
                    "val/batch": batch_index,
                    "val/accuracy": accuracy,
                    "val/completed_rollouts": len(all_results),
                }
            )
            milestone(
                f"val batch={batch_index} completed={len(all_results)} "
                f"accuracy={accuracy:.6f}"
            )
        final = {
            "rollouts": len(all_results),
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
        wandb_run.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--skill-dir", type=Path, default=Path(__file__).resolve().parents[1] / "skills/initial")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="gpt-5.6-luna")
    parser.add_argument("--api-key-env", default="PERSONAL_OPENAI_KEY")
    parser.add_argument("--embedding-url", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--rollouts", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
