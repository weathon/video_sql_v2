from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import wandb
import weave

from src.harness import HarnessConfig, HarnessContext, Question, Stage2Index, answer_question, judge_answer
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
    wandb.init(entity="3dsmile", project="video-sql-v2", dir=str(run_dir), name=run_dir.name, config=vars(args))
    weave.init("3dsmile/video-sql-v2")
    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(question: Question, q_idx: int, r_idx: int):
        out_path = run_dir / "answers" / f"question-{q_idx:06d}-rollout-{r_idx:02d}.json"
        if out_path.exists():
            return json.loads(out_path.read_text())["score"]
        async with semaphore:
            ctx = HarnessContext(config=config, index=index, run_dir=run_dir / "answers" / f"question-{q_idx:06d}-rollout-{r_idx:02d}", skills_dir=repo_root / args.skills_dir)
            answer, _result, _agent = await answer_question(ctx, question, f"question-{q_idx:06d}-rollout-{r_idx:02d}")
            score = await judge_answer(config, question.answer, answer.answer)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps({"question": question.model_dump(), "answer": answer.model_dump(), "score": score}, ensure_ascii=False, indent=2))
            wandb.log({"rollout/score": score})
            return score

    scores = await asyncio.gather(*[one(question, q_idx, r_idx) for q_idx, question in enumerate(questions) for r_idx in range(args.rollouts_per_question)])
    aggregate = sum(scores) / len(scores)
    (run_dir / "summary.json").write_text(json.dumps({"rollouts": len(scores), "mean_score_out_of_5": aggregate}, indent=2))
    wandb.log({"val/mean_score_out_of_5": aggregate, "val/rollouts": len(scores)})
    wandb.finish()


if __name__ == "__main__":
    asyncio.run(main())

