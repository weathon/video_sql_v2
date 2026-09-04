from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from pathlib import Path

import wandb
import weave

from src.harness import CompactionLimit, HarnessConfig, HarnessContext, Question, Stage2Index, answer_question, judge_answer
from src.training import load_questions


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-json", default="data/test-seed-0.json")
    parser.add_argument("--skills-dir", default="skills/initial")
    parser.add_argument("--run-dir", default="runs/val_initial_current")
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--rollouts-per-question", type=int, default=5)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=24)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config = HarnessConfig(repo_root=repo_root, main_model=args.model, main_base_url=args.base_url, reasoning_effort=args.reasoning_effort, attempts=args.attempts)
    run_dir = repo_root / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    questions = load_questions(repo_root / args.test_json)
    index = Stage2Index(config)
    git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    wandb_config = dict(vars(args))
    wandb_config["git_hash"] = git_hash
    wandb.init(entity="3dsmile", project="video-sql-v2", dir=str(run_dir), name=run_dir.name, config=wandb_config)
    weave.init("3dsmile/video-sql-v2")
    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(question: Question, q_idx: int, r_idx: int):
        out_path = run_dir / "answers" / f"question-{q_idx:06d}-rollout-{r_idx:02d}.json"
        if out_path.exists():
            return json.loads(out_path.read_text())["score"]
        async with semaphore:
            ctx = HarnessContext(config=config, index=index, run_dir=run_dir / "answers" / f"question-{q_idx:06d}-rollout-{r_idx:02d}", skills_dir=repo_root / args.skills_dir)
            answer = None
            for answer_attempt in range(args.attempts):
                try:
                    answer, _result, _agent = await answer_question(ctx, question, f"question-{q_idx:06d}-rollout-{r_idx:02d}-answer-attempt-{answer_attempt:02d}")
                    break
                except Exception as exc:
                    if isinstance(exc, CompactionLimit):
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        out_path.write_text(json.dumps({"question": question.model_dump(), "answer_attempt": answer_attempt, "judge_attempt": None, "answer": {"status": "compaction_limit", "answer": "", "evidence_times": []}, "score": 0, "accuracy": 0}, ensure_ascii=False, indent=2))
                        return 0
                    err_path = ctx.run_dir / f"answer-attempt-{answer_attempt:02d}-error.txt"
                    err_path.parent.mkdir(parents=True, exist_ok=True)
                    err_path.write_text(str(exc))
                    if answer_attempt + 1 == args.attempts:
                        raise
            if answer is None:
                raise RuntimeError("answer attempts exhausted")
            for judge_attempt in range(args.attempts):
                try:
                    score = await judge_answer(config, question.answer, answer.answer, ctx.run_dir / "traces" / f"judge-attempt-{judge_attempt:02d}.json")
                    break
                except Exception as exc:
                    err_path = ctx.run_dir / f"judge-attempt-{judge_attempt:02d}-error.txt"
                    err_path.parent.mkdir(parents=True, exist_ok=True)
                    err_path.write_text(str(exc))
                    if judge_attempt + 1 == args.attempts:
                        raise
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps({"question": question.model_dump(), "answer_attempt": answer_attempt, "judge_attempt": judge_attempt, "answer": answer.model_dump(), "score": score, "accuracy": score / 5}, ensure_ascii=False, indent=2))
            wandb.log({"rollout/score": score})
            return score

    scores = await asyncio.gather(*[one(question, q_idx, r_idx) for q_idx, question in enumerate(questions) for r_idx in range(args.rollouts_per_question)])
    mean_score = sum(scores) / len(scores)
    accuracy = sum(scores) / (5 * len(scores))
    (run_dir / "summary.json").write_text(json.dumps({"rollouts": len(scores), "mean_score_out_of_5": mean_score, "accuracy": accuracy}, indent=2))
    wandb.log({"val/mean_score_out_of_5": mean_score, "val/accuracy": accuracy, "val/rollouts": len(scores)})
    wandb.finish()


if __name__ == "__main__":
    asyncio.run(main())
