from __future__ import annotations

import argparse
import asyncio
import json
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

import wandb
import weave
from agents import Agent, ModelSettings, function_tool

from src.harness import AgentAnswer, AgentRaised, CompactionLimit, HarnessConfig, HarnessContext, Question, Stage2Index, answer_question, judge_answer, reflect_question, responses_model, run_agent_stream, summarize_reflections


DEFAULT_TRAIN_JSON = "/raid/huggingface/hub/datasets--MM-Lifelong--MM-Lifelong/snapshots/248aa82039a574e63a2e524746a7cd8f32330443/month/train.json"


def load_questions(path: Path) -> list[Question]:
    raw = json.loads(path.read_text())
    questions = []
    for item in raw:
        timeline = item.get("timeline") or item.get("video_id") or "month"
        source_split = item.get("source_split", "")
        questions.append(Question(timeline=str(timeline), source_split=source_split, index=item["index"], question=item["question"], answer=item["answer"]))
    return questions


def ensure_skill_repo(repo_root: Path, run_dir: Path) -> Path:
    skills_dir = run_dir / "skills"
    if skills_dir.exists():
        return skills_dir
    shutil.copytree(repo_root / "skills" / "initial", skills_dir)
    subprocess.run(["git", "init"], cwd=skills_dir, check=True)
    subprocess.run(["git", "config", "user.name", "codex"], cwd=skills_dir, check=True)
    subprocess.run(["git", "config", "user.email", "codex@localhost"], cwd=skills_dir, check=True)
    subprocess.run(["git", "add", "."], cwd=skills_dir, check=True)
    subprocess.run(["git", "commit", "-m", "Initial skills"], cwd=skills_dir, check=True)
    return skills_dir


def safe_skill_path(skills_dir: Path, filename: str) -> Path:
    path = (skills_dir / filename).resolve()
    if skills_dir.resolve() not in path.parents and path != skills_dir.resolve():
        raise RuntimeError(f"path escapes skills dir: {filename}")
    if path.name == ".git" or ".git" in path.parts:
        raise RuntimeError("direct .git edits are forbidden")
    return path


def skill_update_commit_exists(skills_dir: Path, batch_id: str) -> bool:
    log = subprocess.check_output(["git", "log", "--format=%s"], cwd=skills_dir, text=True)
    return f"Update skills for {batch_id}" in log.splitlines()


def updater_tools(skills_dir: Path) -> list[Any]:
    def tool_error(_ctx, exc: Exception) -> str:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)

    @function_tool(failure_error_function=tool_error)
    def list_skill_files() -> str:
        """List Markdown files in the skill repository."""
        return json.dumps([path.name for path in sorted(skills_dir.glob("*.md"))], ensure_ascii=False)

    @function_tool(failure_error_function=tool_error)
    def read_skill_file(filename: str) -> str:
        """Read a Markdown skill file from the skill repository."""
        return safe_skill_path(skills_dir, filename).read_text()

    @function_tool(failure_error_function=tool_error)
    def write_skill_file(filename: str, content: str) -> str:
        """Write a Markdown skill file in the skill repository."""
        path = safe_skill_path(skills_dir, filename)
        path.write_text(content)
        return json.dumps({"ok": True, "file": path.name}, ensure_ascii=False)

    @function_tool(failure_error_function=tool_error)
    def delete_skill_file(filename: str) -> str:
        """Delete a Markdown skill file except META-SKILL.md."""
        if filename == "META-SKILL.md":
            raise RuntimeError("META-SKILL.md cannot be deleted")
        safe_skill_path(skills_dir, filename).unlink()
        return json.dumps({"ok": True, "deleted": filename}, ensure_ascii=False)

    @function_tool(failure_error_function=tool_error)
    def git_status() -> str:
        """Read git status for the skill repository."""
        return subprocess.check_output(["git", "status", "--short"], cwd=skills_dir, text=True)

    @function_tool(failure_error_function=tool_error)
    def git_log() -> str:
        """Read recent skill repository commits."""
        return subprocess.check_output(["git", "log", "--oneline", "-10"], cwd=skills_dir, text=True)

    @function_tool(failure_error_function=tool_error)
    def git_show(commit: str, filename: str) -> str:
        """Read one file from a historical skill repository commit."""
        safe_skill_path(skills_dir, filename)
        return subprocess.check_output(["git", "show", f"{commit}:{filename}"], cwd=skills_dir, text=True)

    @function_tool
    def stop(reason: str) -> str:
        """Stop the skill updater after file edits are complete."""
        return json.dumps({"reason": reason}, ensure_ascii=False)

    return [list_skill_files, read_skill_file, write_skill_file, delete_skill_file, git_status, git_log, git_show, stop]


async def run_updater(config: HarnessConfig, skills_dir: Path, run_dir: Path, summary: str, rollout_ids: list[str], batch_id: str) -> None:
    agent = Agent(
        name="skill_updater",
        instructions="Update this Git-backed directory of Markdown skills from the batch summary. You may add, edit, or delete ordinary skill files. Never delete META-SKILL.md. Put brief **Reason:** text after each guidance point or section. Call stop when finished.",
        model=responses_model(config),
        model_settings=ModelSettings(reasoning={"effort": config.reasoning_effort}, preserve_raw_usage=True),
        tools=updater_tools(skills_dir),
        tool_use_behavior={"stop_at_tool_names": ["stop"]},
    )
    ctx = HarnessContext(config=config, index=Stage2Index(config), run_dir=run_dir, skills_dir=skills_dir)
    await run_agent_stream(agent, "Rollout IDs:\n" + "\n".join(rollout_ids) + "\n\nBatch summary:\n" + summary, ctx, run_dir / "traces" / f"{batch_id}-updater.json", force_submit_tool="stop")
    subprocess.run(["git", "add", "-A"], cwd=skills_dir, check=True)
    body = "Rollout IDs:\n" + "\n".join(rollout_ids) + "\n\nReason:\n" + summary
    subprocess.run(["git", "commit", "--allow-empty", "-m", f"Update skills for {batch_id}", "-m", body], cwd=skills_dir, check=True)


async def run_rollout(config: HarnessConfig, index: Stage2Index, skills_dir: Path, run_dir: Path, question: Question, rollout_id: str) -> dict[str, Any]:
    out_path = run_dir / "rollouts" / f"{rollout_id}.json"
    if out_path.exists():
        return json.loads(out_path.read_text())
    rollout_dir = run_dir / "rollouts" / rollout_id
    ctx = HarnessContext(config=config, index=index, run_dir=rollout_dir, skills_dir=skills_dir)
    answer = None
    answer_result = None
    agent = None
    for attempt in range(config.attempts):
        try:
            answer, answer_result, agent = await answer_question(ctx, question, f"{rollout_id}-attempt-{attempt:02d}")
            break
        except AgentRaised as exc:
            err_path = rollout_dir / f"attempt-{attempt:02d}-agent-raise.txt"
            err_path.parent.mkdir(parents=True, exist_ok=True)
            err_path.write_text(str(exc))
            if attempt + 1 == config.attempts:
                raise
        except CompactionLimit as exc:
            candidate = AgentAnswer(status="compaction_limit", answer="", evidence_times=[])
            for reflection_attempt in range(config.attempts):
                try:
                    reflection = await reflect_question(ctx, exc.agent, exc.result, question, candidate, 0, f"{rollout_id}-compaction-reflection-attempt-{reflection_attempt:02d}")
                    payload = {"rollout_id": rollout_id, "answer_attempt": attempt, "judge_attempt": None, "reflection_attempt": reflection_attempt, "question": question.model_dump(), "answer": candidate.model_dump(), "score": 0, "reflection": reflection.model_dump(), "compaction_limit": True}
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
                    return payload
                except Exception as reflection_exc:
                    err_path = rollout_dir / f"compaction-reflection-attempt-{reflection_attempt:02d}-error.txt"
                    err_path.parent.mkdir(parents=True, exist_ok=True)
                    err_path.write_text(str(reflection_exc))
                    if reflection_attempt + 1 == config.attempts:
                        raise
        except Exception as exc:
            err_path = rollout_dir / f"attempt-{attempt:02d}-error.txt"
            err_path.parent.mkdir(parents=True, exist_ok=True)
            err_path.write_text(str(exc))
            if attempt + 1 == config.attempts:
                raise
    if answer is None or answer_result is None or agent is None:
        raise RuntimeError("answer stage exhausted without output")
    for judge_attempt in range(config.attempts):
        try:
            score = await judge_answer(config, question.answer, answer.answer, rollout_dir / "traces" / f"judge-attempt-{judge_attempt:02d}.json")
            break
        except Exception as exc:
            err_path = rollout_dir / f"judge-attempt-{judge_attempt:02d}-error.txt"
            err_path.parent.mkdir(parents=True, exist_ok=True)
            err_path.write_text(str(exc))
            if judge_attempt + 1 == config.attempts:
                raise
    for reflection_attempt in range(config.attempts):
        try:
            reflection = await reflect_question(ctx, agent, answer_result, question, answer, score, f"{rollout_id}-reflection-attempt-{reflection_attempt:02d}")
            payload = {"rollout_id": rollout_id, "answer_attempt": attempt, "judge_attempt": judge_attempt, "reflection_attempt": reflection_attempt, "question": question.model_dump(), "answer": answer.model_dump(), "score": score, "reflection": reflection.model_dump(), "compaction_limit": False}
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            return payload
        except Exception as exc:
            err_path = rollout_dir / f"reflection-attempt-{reflection_attempt:02d}-error.txt"
            err_path.parent.mkdir(parents=True, exist_ok=True)
            err_path.write_text(str(exc))
            if reflection_attempt + 1 == config.attempts:
                raise
    raise RuntimeError("reflection stage exhausted without output")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", default=DEFAULT_TRAIN_JSON)
    parser.add_argument("--run-dir", default="runs/pilot_openai_agents_current")
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--questions-per-batch", type=int, default=4)
    parser.add_argument("--rollouts-per-question", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=24)
    args = parser.parse_args()
    if args.max_updates is None and args.model == "gpt-5.6-luna":
        args.max_updates = 4

    repo_root = Path(__file__).resolve().parents[1]
    config = HarnessConfig(repo_root=repo_root, main_model=args.model, main_base_url=args.base_url, reasoning_effort=args.reasoning_effort, attempts=args.attempts)
    run_dir = repo_root / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    skills_dir = ensure_skill_repo(repo_root, run_dir)
    questions = load_questions(repo_root / args.train_json)
    index = Stage2Index(config)
    git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    wandb_config = dict(vars(args))
    wandb_config["git_hash"] = git_hash
    wandb.init(entity="3dsmile", project="video-sql-v2", dir=str(run_dir), name=run_dir.name, config=wandb_config)
    weave.init("3dsmile/video-sql-v2")
    updates_done = 0
    consecutive_failed_batches = 0
    for epoch in range(args.epochs):
        rng = random.Random(args.seed + epoch)
        ordered = list(questions)
        rng.shuffle(ordered)
        for start in range(0, len(ordered), args.questions_per_batch):
            if args.max_updates is not None and updates_done >= args.max_updates:
                wandb.finish()
                return
            batch = ordered[start:start + args.questions_per_batch]
            batch_id = f"epoch-{epoch:03d}-batch-{start // args.questions_per_batch:06d}"
            batch_dir = run_dir / batch_id
            complete_path = batch_dir / "complete.json"
            if complete_path.exists():
                complete = json.loads(complete_path.read_text())
                updates_done += complete["updated"]
                consecutive_failed_batches = consecutive_failed_batches + 1 if complete["updated"] == 0 else 0
                if consecutive_failed_batches >= 2:
                    raise RuntimeError("two consecutive batches had no successful rollouts")
                continue
            semaphore = asyncio.Semaphore(args.concurrency)
            async def guarded(question: Question, q_idx: int, r_idx: int):
                async with semaphore:
                    return await run_rollout(config, index, skills_dir, batch_dir, question, f"{batch_id}-question-{q_idx:06d}-rollout-{r_idx:02d}")
            tasks = [guarded(question, q_idx, r_idx) for q_idx, question in enumerate(batch) for r_idx in range(args.rollouts_per_question)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            good = [item for item in results if not isinstance(item, Exception)]
            failed = [str(item) for item in results if isinstance(item, Exception)]
            wandb.log({"batch/good_rollouts": len(good), "batch/failed_rollouts": len(failed), "batch/index": updates_done})
            if not good:
                consecutive_failed_batches += 1
                with (run_dir / "AGENT_STDOUT.txt").open("a") as handle:
                    handle.write(f"{batch_id} good_rollouts=0 failed_rollouts={len(failed)}\n")
                print(f"{batch_id} good_rollouts=0 failed_rollouts={len(failed)}", flush=True)
                complete_path.parent.mkdir(parents=True, exist_ok=True)
                complete_path.write_text(json.dumps({"updated": 0, "good_rollouts": 0, "failed_rollouts": len(failed)}, indent=2))
                if consecutive_failed_batches >= 2:
                    raise RuntimeError("two consecutive batches had no successful rollouts")
                continue
            consecutive_failed_batches = 0
            summaries = []
            for q_idx, _question in enumerate(batch):
                question_rollouts = [item for item in good if f"question-{q_idx:06d}-" in item["rollout_id"]]
                if not question_rollouts:
                    continue
                summary_path = batch_dir / "summaries" / f"question-{q_idx:06d}.json"
                if summary_path.exists():
                    summaries.append(json.loads(summary_path.read_text()))
                    continue
                for summary_attempt in range(config.attempts):
                    try:
                        summary_payload = json.dumps(question_rollouts, ensure_ascii=False)
                        summary = await summarize_reflections(config, summary_payload, batch_dir / "traces" / f"summary-question-{q_idx:06d}-attempt-{summary_attempt:02d}.json")
                        record = {"question_index": q_idx, "summary_attempt": summary_attempt, "summary": summary}
                        summary_path.parent.mkdir(parents=True, exist_ok=True)
                        summary_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
                        summaries.append(record)
                        break
                    except Exception as exc:
                        err_path = batch_dir / "summary-errors" / f"question-{q_idx:06d}-attempt-{summary_attempt:02d}.txt"
                        err_path.parent.mkdir(parents=True, exist_ok=True)
                        err_path.write_text(str(exc))
                        if summary_attempt + 1 == config.attempts:
                            raise
            summary = json.dumps(summaries, ensure_ascii=False)
            rollout_ids = [item["rollout_id"] + f"/attempt-{item['answer_attempt']:02d}" for item in good]
            if not skill_update_commit_exists(skills_dir, batch_id):
                for updater_attempt in range(config.attempts):
                    try:
                        await run_updater(config, skills_dir, batch_dir / f"updater-attempt-{updater_attempt:02d}", summary, rollout_ids, batch_id)
                        break
                    except Exception as exc:
                        err_path = batch_dir / "updater-errors" / f"attempt-{updater_attempt:02d}.txt"
                        err_path.parent.mkdir(parents=True, exist_ok=True)
                        err_path.write_text(str(exc))
                        if updater_attempt + 1 == config.attempts:
                            raise
            if not skill_update_commit_exists(skills_dir, batch_id):
                raise RuntimeError(f"missing skill update commit for {batch_id}")
            updates_done += 1
            complete_path.write_text(json.dumps({"updated": 1, "good_rollouts": len(good), "failed_rollouts": len(failed)}, indent=2))
            with (run_dir / "AGENT_STDOUT.txt").open("a") as handle:
                handle.write(f"{batch_id} update={updates_done} good_rollouts={len(good)} failed_rollouts={len(failed)}\n")
            print(f"{batch_id} update={updates_done} good_rollouts={len(good)} failed_rollouts={len(failed)}", flush=True)
    wandb.finish()


if __name__ == "__main__":
    asyncio.run(main())
