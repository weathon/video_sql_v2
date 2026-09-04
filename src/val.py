import argparse
import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import wandb
import weave
from dotenv import load_dotenv
from wandb.sdk.lib.runid import generate_id

from src.harness import VideoSQLHarness
from src.training import judge_answer, milestone, retryable, write_error


async def run_eval_rollout(
    semaphore: asyncio.Semaphore,
    question: dict[str, Any],
    rollout: int,
    skill_dir: Path,
    runtime_root: Path,
    run_dir: Path,
    model_name: str,
    api_key: str,
    base_url: str | None,
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
                base_url=base_url,
                embedding_url=embedding_url,
                embedding_model=embedding_model,
                reasoning_effort="medium",
            )
            try:
                session = await harness.answer(question["question"])
                score = (
                    0
                    if session.status == "compaction_limit"
                    else await judge_answer(
                        harness,
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
                    "trace_files": session.trace_files,
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
    load_dotenv()
    api_key = os.environ[args.api_key_env]
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
            "base_url": args.base_url,
            "embedding_url": args.embedding_url,
            "embedding_model": args.embedding_model,
            "attempts": args.attempts,
            "batch_size": args.batch_size,
            "rollouts": args.rollouts,
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
    semaphore = asyncio.Semaphore(24)
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
                    args.base_url,
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
    parser.add_argument("--skill-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--embedding-url", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--rollouts", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
