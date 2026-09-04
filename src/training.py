from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any, Awaitable, Callable, TypeVar

import wandb
import weave
from wandb.sdk.lib.runid import generate_id

from src.harness import (
    CompactionLimit,
    DATASET_SNAPSHOT,
    RolloutContext,
    continue_reflection,
    judge_answer,
    make_model_bundle,
    make_retrieval_agent,
    run_main_with_compaction,
    summarize_reflections,
    update_skills,
)


T = TypeVar("T")
PROJECT = "3dsmile/video-sql-v2"


async def retry_stage(
    operation: Callable[[int], Awaitable[T]],
    attempts: int,
    errors_path: Path,
) -> T:
    errors = []
    for attempt in range(1, attempts + 1):
        try:
            return await operation(attempt)
        except CompactionLimit:
            raise
        except Exception as error:
            errors.append({"attempt": attempt, "type": type(error).__name__, "error": str(error)})
            errors_path.parent.mkdir(parents=True, exist_ok=True)
            errors_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2))
    raise RuntimeError(f"stage failed after {attempts} attempts: {errors_path}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def initialize_skills(skills_dir: Path, initial_skills: Path) -> None:
    if skills_dir.exists():
        return
    shutil.copytree(initial_skills, skills_dir)
    subprocess.run(["git", "init"], cwd=skills_dir, check=True, capture_output=True)
    subprocess.run(["git", "add", "-A"], cwd=skills_dir, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=codex",
            "-c",
            "user.email=codex@localhost",
            "commit",
            "-m",
            "Initial skills",
        ],
        cwd=skills_dir,
        check=True,
        capture_output=True,
    )


def initialize_wandb(run_dir: Path, config: dict[str, Any]) -> Any:
    state_path = run_dir / "wandb.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        return wandb.init(
            entity="3dsmile",
            project="video-sql-v2",
            name=run_dir.name,
            id=state["id"],
            resume="must",
            config=config,
        )
    run_id = generate_id()
    run = wandb.init(
        entity="3dsmile",
        project="video-sql-v2",
        name=run_dir.name,
        id=run_id,
        resume="never",
        config=config,
    )
    write_json(state_path, {"id": run_id})
    return run


async def run_rollout(
    question: dict[str, Any],
    epoch: int,
    batch_index: int,
    rollout_index: int,
    args: argparse.Namespace,
    bundle: Any,
    skills_dir: Path,
) -> dict[str, Any]:
    rollout_id_prefix = (
        f"epoch-{epoch:03d}/batch-{batch_index:06d}/question-{question['index']:06d}/"
        f"rollout-{rollout_index:02d}"
    )
    rollout_dir = args.run_dir / "train" / rollout_id_prefix
    result_path = rollout_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())

    state: dict[str, Any] = {}

    async def answer_attempt(attempt: int) -> tuple[Any, Any, RolloutContext, Any, int]:
        attempt_dir = rollout_dir / f"attempt-{attempt:02d}"
        context = RolloutContext(
            database_path=args.database,
            embedding_root=args.embedding_root,
            embedding_url=args.embedding_url,
            dataset_root=args.dataset_root,
            skills_dir=skills_dir,
            run_dir=attempt_dir,
            model_bundle=bundle,
            video_id="month",
        )
        agent = make_retrieval_agent(context)
        state["answer_attempt"] = attempt
        state["context"] = context
        state["agent"] = agent
        prompt = f"Question:\n{question['question']}\n\nLogical timeline: month"
        answer, history, compactions = await run_main_with_compaction(
            agent, prompt, context, attempt_dir / "traces"
        )
        return answer, history, context, agent, compactions

    try:
        answer, history, context, agent, compactions = await retry_stage(
            answer_attempt, args.attempts, rollout_dir / "answer-errors.json"
        )
        score = await retry_stage(
            lambda attempt: judge_answer(
                bundle,
                question["answer"],
                answer.answer,
                rollout_dir / f"judge-attempt-{attempt:02d}.json",
            ),
            args.attempts,
            rollout_dir / "judge-errors.json",
        )
        candidate_answer = answer.answer
        score_value = score.score
        compaction_limited = False
    except CompactionLimit as error:
        context = state["context"]
        agent = state["agent"]
        history = error.history
        candidate_answer = ""
        score_value = 0
        compactions = 3
        compaction_limited = True
    except RuntimeError as error:
        result = {"rollout_id": rollout_id_prefix, "status": "failed", "score": 0, "error": str(error)}
        write_json(result_path, result)
        return result

    async def reflection_attempt(attempt: int) -> Any:
        state["reflection_attempt"] = attempt
        return await continue_reflection(
            agent,
            history,
            context,
            question["answer"],
            score_value,
            rollout_dir / f"reflection-attempt-{attempt:02d}.json",
        )

    try:
        reflection = await retry_stage(
            reflection_attempt, args.attempts, rollout_dir / "reflection-errors.json"
        )
    except RuntimeError as error:
        result = {
            "rollout_id": rollout_id_prefix,
            "status": "failed",
            "candidate_answer": candidate_answer,
            "score": score_value,
            "compactions": compactions,
            "error": str(error),
        }
        write_json(result_path, result)
        return result

    result = {
        "rollout_id": rollout_id_prefix,
        "status": "complete",
        "candidate_answer": candidate_answer,
        "reference_answer": question["answer"],
        "score": score_value,
        "answer_attempt": state["answer_attempt"],
        "reflection_attempt": state["reflection_attempt"],
        "compactions": compactions,
        "compaction_limit": compaction_limited,
        "reflection": reflection.reflection,
        "meta_reflection": reflection.meta_reflection,
    }
    write_json(result_path, result)
    return result


async def train(args: argparse.Namespace) -> None:
    args.run_dir.mkdir(parents=True, exist_ok=True)
    skills_dir = args.run_dir / "skills"
    initialize_skills(skills_dir, args.initial_skills)
    questions = json.loads(args.train_file.read_text())
    bundle = make_model_bundle(args.model, args.base_url, args.api_key, "medium")
    weave.init(PROJECT)
    run = initialize_wandb(
        args.run_dir,
        {
            "model": args.model,
            "batch_questions": 4,
            "rollouts_per_question": 5,
            "epochs": 3,
            "attempts": 3,
            "seed": args.seed,
            "database": str(args.database),
            "embedding_root": str(args.embedding_root),
            "git_hash": subprocess.run(
                ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
            ).stdout.strip(),
        },
    )
    update_count = 0
    consecutive_empty_batches = 0
    for epoch in range(3):
        shuffled = list(questions)
        random.Random(args.seed + epoch).shuffle(shuffled)
        for batch_index, offset in enumerate(range(0, len(shuffled), 4)):
            batch_dir = args.run_dir / "train" / f"epoch-{epoch:03d}" / f"batch-{batch_index:06d}"
            done_path = batch_dir / "complete.json"
            if done_path.exists():
                completed_batch = json.loads(done_path.read_text())
                update_count += completed_batch["updated"]
                consecutive_empty_batches = (
                    consecutive_empty_batches + 1 if completed_batch["updated"] == 0 else 0
                )
                if consecutive_empty_batches == 2:
                    raise RuntimeError("two consecutive batches had no successful rollouts")
                continue
            batch = shuffled[offset : offset + 4]
            results = await asyncio.gather(
                *(
                    run_rollout(question, epoch, batch_index, rollout, args, bundle, skills_dir)
                    for question in batch
                    for rollout in range(5)
                )
            )
            complete = [result for result in results if result["status"] == "complete"]
            for result in results:
                run.log(
                    {
                        "epoch": epoch,
                        "batch": batch_index,
                        "rollout_score": result["score"],
                        "rollout_success": int(result["status"] == "complete"),
                    }
                )
            if not complete:
                consecutive_empty_batches += 1
                write_json(done_path, {"updated": 0, "reason": "all rollouts failed"})
                if consecutive_empty_batches == 2:
                    raise RuntimeError("two consecutive batches had no successful rollouts")
                continue
            consecutive_empty_batches = 0

            summaries = []
            for question in batch:
                question_results = [
                    result
                    for result in complete
                    if f"question-{question['index']:06d}/" in result["rollout_id"]
                ]
                if not question_results:
                    continue
                summary_path = batch_dir / "summaries" / f"question-{question['index']:06d}.json"
                if summary_path.exists():
                    summaries.append(json.loads(summary_path.read_text()))
                    continue
                summary = await retry_stage(
                    lambda attempt, qr=question_results, qi=question["index"]: summarize_reflections(
                        bundle,
                        [
                            {
                                "rollout_id": result["rollout_id"],
                                "reflection": result["reflection"],
                                "meta_reflection": result["meta_reflection"],
                            }
                            for result in qr
                        ],
                        batch_dir / "traces" / f"summary-question-{qi:06d}-attempt-{attempt:02d}.json",
                    ),
                    args.attempts,
                    batch_dir / "summary-errors" / f"question-{question['index']:06d}.json",
                )
                summary_record = {"question_index": question["index"], "summary": summary}
                write_json(summary_path, summary_record)
                summaries.append(summary_record)

            await retry_stage(
                lambda attempt: update_skills(
                    bundle,
                    skills_dir,
                    summaries,
                    [
                        result["rollout_id"] + f"/attempt-{result['answer_attempt']:02d}"
                        for result in complete
                    ],
                    batch_dir / "traces" / f"updater-attempt-{attempt:02d}.json",
                ),
                args.attempts,
                batch_dir / "updater-errors.json",
            )
            update_count += 1
            write_json(done_path, {"updated": 1, "rollouts": len(complete)})
            run.log({"skill_updates": update_count, "batch_mean_score": sum(r["score"] for r in results) / len(results)})
            milestone = (
                f"epoch={epoch} batch={batch_index} updates={update_count} "
                f"complete_rollouts={len(complete)}"
            )
            with (args.run_dir / "AGENT_STDOUT.txt").open("a") as handle:
                handle.write(milestone + "\n")
            print(milestone, flush=True)
            if args.max_updates is not None and update_count >= args.max_updates:
                run.finish()
                return
    run.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--database", type=Path, default=Path("data/pilot/stage2/index.duckdb"))
    parser.add_argument("--embedding-root", type=Path, default=Path("data/pilot/stage2/embedding_index"))
    parser.add_argument("--embedding-url", default="http://127.0.0.1:8013/v1/embeddings")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_SNAPSHOT)
    parser.add_argument("--train-file", type=Path, default=DATASET_SNAPSHOT / "month/train.json")
    parser.add_argument("--initial-skills", type=Path, default=Path("skills/initial"))
    parser.add_argument("--model", default="Qwen/Qwen3.8-Flash-Next-FP8")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "not-needed"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attempts", type=int, default=3, choices=[3])
    parser.add_argument("--max-updates", type=int)
    args = parser.parse_args()
    if args.model == "gpt-5.6-luna":
        args.max_updates = min(args.max_updates, 4) if args.max_updates is not None else 4
    return args


if __name__ == "__main__":
    asyncio.run(train(parse_args()))
