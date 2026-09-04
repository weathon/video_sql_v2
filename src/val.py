from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import wandb
import weave

from src.harness import (
    CompactionLimit,
    DATASET_SNAPSHOT,
    RolloutContext,
    judge_answer,
    make_model_bundle,
    make_retrieval_agent,
    run_main_with_compaction,
)
from src.training import PROJECT, initialize_wandb, retry_stage, write_json


async def evaluate_rollout(
    question: dict[str, Any],
    rollout_index: int,
    args: argparse.Namespace,
    bundle: Any,
) -> dict[str, Any]:
    question_id = f"{question['timeline']}-{question['source_split'].replace('/', '-')}-{question['index']:06d}"
    rollout_dir = args.run_dir / "val" / question_id / f"rollout-{rollout_index:02d}"
    result_path = rollout_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())

    state: dict[str, Any] = {}

    async def answer_attempt(attempt: int) -> str:
        attempt_dir = rollout_dir / f"attempt-{attempt:02d}"
        context = RolloutContext(
            database_path=args.database,
            embedding_root=args.embedding_root,
            embedding_url=args.embedding_url,
            dataset_root=args.dataset_root,
            skills_dir=args.skills_dir,
            run_dir=attempt_dir,
            model_bundle=bundle,
            video_id=question["timeline"],
        )
        agent = make_retrieval_agent(context)
        state["answer_attempt"] = attempt
        answer, _, compactions = await run_main_with_compaction(
            agent,
            f"Question:\n{question['question']}\n\nLogical timeline: {question['timeline']}",
            context,
            attempt_dir / "traces",
        )
        state["compactions"] = compactions
        return answer.answer

    try:
        candidate = await retry_stage(
            answer_attempt, args.attempts, rollout_dir / "answer-errors.json"
        )
        judged = await retry_stage(
            lambda attempt: judge_answer(
                bundle,
                question["answer"],
                candidate,
                rollout_dir / f"judge-attempt-{attempt:02d}.json",
            ),
            args.attempts,
            rollout_dir / "judge-errors.json",
        )
        result = {
            "status": "complete",
            "timeline": question["timeline"],
            "source_split": question["source_split"],
            "question_index": question["index"],
            "rollout_index": rollout_index,
            "answer_attempt": state["answer_attempt"],
            "candidate_answer": candidate,
            "reference_answer": question["answer"],
            "score": judged.score,
            "compactions": state["compactions"],
        }
    except CompactionLimit:
        result = {
            "status": "compaction_limit",
            "timeline": question["timeline"],
            "source_split": question["source_split"],
            "question_index": question["index"],
            "rollout_index": rollout_index,
            "answer_attempt": state["answer_attempt"],
            "score": 0,
            "compactions": 3,
        }
    except RuntimeError as error:
        result = {
            "status": "failed",
            "timeline": question["timeline"],
            "source_split": question["source_split"],
            "question_index": question["index"],
            "rollout_index": rollout_index,
            "score": 0,
            "error": str(error),
        }
    write_json(result_path, result)
    return result


async def validate(args: argparse.Namespace) -> None:
    args.run_dir.mkdir(parents=True, exist_ok=True)
    questions = json.loads(args.test_file.read_text())
    bundle = make_model_bundle(args.model, args.base_url, args.api_key, "medium")
    weave.init(PROJECT)
    run = initialize_wandb(
        args.run_dir,
        {
            "model": args.model,
            "skills_dir": str(args.skills_dir),
            "test_file": str(args.test_file),
            "questions": 300,
            "rollouts_per_question": 5,
            "attempts": 3,
            "git_hash": subprocess.run(
                ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
            ).stdout.strip(),
        },
    )
    results = []
    for offset in range(0, len(questions), 4):
        batch = questions[offset : offset + 4]
        batch_results = await asyncio.gather(
            *(
                evaluate_rollout(question, rollout, args, bundle)
                for question in batch
                for rollout in range(5)
            )
        )
        results.extend(batch_results)
        for result in batch_results:
            run.log(
                {
                    "rollout_score": result["score"],
                    "rollout_accuracy": result["score"] / 5,
                    "rollout_success": int(result["status"] == "complete"),
                }
            )
        milestone = f"questions={min(offset + 4, len(questions))}/300 rollouts={len(results)}/1500"
        with (args.run_dir / "AGENT_STDOUT.txt").open("a") as handle:
            handle.write(milestone + "\n")
        print(milestone, flush=True)
    accuracy = sum(result["score"] for result in results) / (5 * len(results))
    summary = {
        "accuracy": accuracy,
        "mean_judge_score": sum(result["score"] for result in results) / len(results),
        "questions": len(questions),
        "rollouts": len(results),
        "failed_rollouts": sum(result["status"] == "failed" for result in results),
    }
    write_json(args.run_dir / "val" / "summary.json", summary)
    run.log(summary)
    run.finish()
    print(json.dumps(summary), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--skills-dir", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, default=Path("data/test-seed-0.json"))
    parser.add_argument("--database", type=Path, default=Path("data/pilot/stage2/index.duckdb"))
    parser.add_argument("--embedding-root", type=Path, default=Path("data/pilot/stage2/embedding_index"))
    parser.add_argument("--embedding-url", default="http://127.0.0.1:8013/v1/embeddings")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_SNAPSHOT)
    parser.add_argument("--model", default="Qwen/Qwen3.8-Flash-Next-FP8")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "not-needed"))
    parser.add_argument("--attempts", type=int, default=3, choices=[3])
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(validate(parse_args()))
