import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from src.harness import PilotHarness
from src.training import TURN_SAFETY_LIMIT, git_state, initialize_wandb, milestone, run_evaluation_rollout, validate_positive, write_json


async def evaluate(args: argparse.Namespace) -> None:
    validate_positive({"questions_per_batch": args.questions_per_batch, "rollouts_per_question": args.rollouts_per_question, "attempts": args.attempts, "context_limit": args.context_limit})
    items = json.loads(args.test_json.read_text())
    skill = args.skill_path.read_text()
    revision, status = git_state()
    launch_config = {"command": "validation", "git_hash": revision, "git_status": status, "database": str(args.database.resolve()), "index_root": str(args.index_root.resolve()), "clip_root": str(args.clip_root.resolve()), "test_json": str(args.test_json.resolve()), "skill_path": str(args.skill_path.resolve()), "skill_sha256": hashlib.sha256(skill.encode()).hexdigest(), "model_base_url": args.model_base_url, "model_name": args.model_name, "embedding_base_url": args.embedding_base_url, "context_limit": args.context_limit, "questions_per_batch": args.questions_per_batch, "rollouts_per_question": args.rollouts_per_question, "attempts": args.attempts, "clip_max_turns": TURN_SAFETY_LIMIT, "mode": args.mode}
    wandb_run = initialize_wandb(args.run_dir, launch_config, args.wandb_entity, args.wandb_project, args.wandb_run_name)
    runtime = PilotHarness(args.database, args.index_root, args.clip_root, args.model_base_url, args.model_name, args.embedding_base_url, args.context_limit)
    all_scores = []
    batch_index = 0
    try:
        for batch_start in range(0, len(items), args.questions_per_batch):
            batch_items = items[batch_start:batch_start + args.questions_per_batch]
            batch_dir = args.run_dir / "validation" / f"batch-{batch_index:06d}"
            completed_path = batch_dir / "completed.json"
            if completed_path.exists():
                all_scores.extend(json.loads(completed_path.read_text())["scores"])
                batch_index += 1
                continue
            async with asyncio.TaskGroup() as rollout_group:
                tasks = []
                for item in batch_items:
                    for rollout_index in range(args.rollouts_per_question):
                        rollout_dir = batch_dir / "questions" / f"question-{int(item['index']):06d}" / f"rollout-{rollout_index:02d}"
                        tasks.append(rollout_group.create_task(run_evaluation_rollout(runtime, item, rollout_index, skill, rollout_dir, args.attempts, args.mode, TURN_SAFETY_LIMIT)))
            results = [task.result() for task in tasks]
            scores = [int(result["score"]) for result in results]
            all_scores.extend(scores)
            completed_rollouts = sum(result["status"] != "failed" for result in results)
            failed_rollouts = len(results) - completed_rollouts
            completed = {"batch": batch_index, "question_indices": [item["index"] for item in batch_items], "completed_rollouts": completed_rollouts, "failed_rollouts": failed_rollouts, "scores": scores}
            metrics = {"validation/completed_rollouts": completed_rollouts, "validation/failed_rollouts": failed_rollouts, "validation/mean_score": sum(scores) / len(scores)}
            wandb_run.log(metrics, step=batch_index)
            write_json(completed_path, completed)
            milestone(args.run_dir, {"event": "validation_batch_completed", **completed, "mean_score": sum(scores) / len(scores)})
            batch_index += 1
        final = {"event": "validation_completed", "batches": batch_index, "scored_rollouts": len(all_scores)}
        if all_scores:
            final["mean_score"] = sum(all_scores) / len(all_scores)
        write_json(args.run_dir / "validation_summary.json", final)
        milestone(args.run_dir, final)
    finally:
        await runtime.close()
        wandb_run.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--clip-root", type=Path, required=True)
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument("--skill-path", type=Path, required=True)
    parser.add_argument("--model-base-url", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--embedding-base-url", required=True)
    parser.add_argument("--context-limit", type=int, required=True)
    parser.add_argument("--questions-per-batch", type=int, required=True)
    parser.add_argument("--rollouts-per-question", type=int, required=True)
    parser.add_argument("--attempts", type=int, required=True)
    parser.add_argument("--mode", choices=["debug", "formal"], required=True)
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-run-name", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(evaluate(parse_args()))
