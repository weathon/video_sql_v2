import argparse
import asyncio
import json
import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

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
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from src.harness import (
    AgentRaised,
    InvalidTrace,
    VideoSQLHarness,
    run_stream,
    tool_error_json,
)


class Judgment(BaseModel):
    score: int


class ReflectionSummary(BaseModel):
    summary: str


class UpdaterStop(BaseModel):
    reason: str


def run_git(skill_dir: Path, arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=skill_dir,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def initialize_skill_repo(initial_skills: Path, skill_dir: Path) -> None:
    if skill_dir.exists():
        if not (skill_dir / ".git").is_dir():
            raise RuntimeError(f"existing skill directory is not a Git repository: {skill_dir}")
        return
    shutil.copytree(initial_skills, skill_dir)
    run_git(skill_dir, ["init"])
    run_git(skill_dir, ["config", "user.name", "codex"])
    run_git(skill_dir, ["config", "user.email", "codex@localhost"])
    run_git(skill_dir, ["add", "."])
    run_git(skill_dir, ["commit", "-m", "initial skills"])


def exception_chain(error: BaseException) -> list[BaseException]:
    chain = []
    current: BaseException | None = error
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def retryable(error: BaseException) -> bool:
    for item in exception_chain(error):
        if isinstance(
            item,
            (
                AgentRaised,
                InvalidTrace,
                MaxTurnsExceeded,
                ModelBehaviorError,
                ValidationError,
                UserError,
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
    reference_answer: str,
    candidate_answer: str,
    attempts: int,
) -> int:
    @function_tool(failure_error_function=None)
    async def submit_judgment(score: int) -> str:
        """Submit an integer semantic-equivalence score from 0 through 5."""
        if not 0 <= score <= 5:
            raise ValueError("score must be from 0 through 5")
        return Judgment(score=score).model_dump_json()

    prompt = json.dumps(
        {
            "reference_answer": reference_answer,
            "candidate_answer": candidate_answer,
        },
        ensure_ascii=False,
    )
    instructions = (
        "Judge only whether the candidate final answer is semantically equivalent to the "
        "reference final answer. Ignore incidental video IDs, timestamps, row IDs, and other "
        "identifiers. Return 5 for fully equivalent and 0 for wrong, with intermediate integers "
        "for partially correct answers. Call submit_judgment."
    )
    for attempt in range(1, attempts + 1):
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
        except BaseException as error:
            write_error(
                harness.context.run_dir / f"judge-attempt-{attempt}-error.json",
                error,
            )
            if not retryable(error):
                raise
    raise RuntimeError("judge attempts exhausted")


async def reflect_rollout(
    harness: VideoSQLHarness,
    session: Any,
    reference_answer: str,
    score: int,
    attempts: int,
) -> Any:
    for attempt in range(1, attempts + 1):
        try:
            return await harness.reflect(session, reference_answer, score)
        except BaseException as error:
            write_error(
                harness.context.run_dir / f"reflection-attempt-{attempt}-error.json",
                error,
            )
            if not retryable(error):
                raise
    raise RuntimeError("reflection attempts exhausted")


async def run_rollout(
    semaphore: asyncio.Semaphore,
    question: dict[str, Any],
    epoch: int,
    batch: int,
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
        / "train"
        / f"epoch-{epoch:03d}"
        / f"batch-{batch:03d}"
        / f"question-{question['index']}"
        / f"rollout-{rollout:02d}"
    )
    result_path = rollout_root / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    async with semaphore:
        for attempt in range(1, attempts + 1):
            attempt_dir = rollout_root / f"attempt-{attempt}"
            if attempt_dir.exists():
                continue
            attempt_dir.mkdir(parents=True)
            rollout_id = (
                f"{epoch}/{batch}/question-{question['index']}/"
                f"rollout-{rollout}/attempt-{attempt}"
            )
            harness = VideoSQLHarness(
                database=runtime_root / "month" / "index.duckdb",
                embedding_index=runtime_root / "month" / "embedding_index",
                media_map=runtime_root / "month" / "media-map.json",
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
                if session.status == "compaction_limit":
                    score = 0
                else:
                    score = await judge_answer(
                        harness,
                        question["answer"],
                        session.answer,
                        attempts,
                    )
                reflection = await reflect_rollout(
                    harness,
                    session,
                    question["answer"],
                    score,
                    attempts,
                )
                result = {
                    "rollout_id": rollout_id,
                    "question_index": question["index"],
                    "status": session.status,
                    "answer": session.answer,
                    "reference_answer": question["answer"],
                    "score": score,
                    "reflection": reflection.reflection,
                    "meta_reflection": reflection.meta_reflection,
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
        return {
            "question_index": question["index"],
            "rollout": rollout,
            "status": "failed",
        }


async def summarize_question(
    harness: VideoSQLHarness,
    results: list[dict[str, Any]],
    attempts: int,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        return json.loads(output.read_text())

    @function_tool(failure_error_function=None)
    async def submit_summary(summary: str) -> str:
        """Submit an abstract cross-rollout summary."""
        return ReflectionSummary(summary=summary).model_dump_json()

    prompt = json.dumps(
        [
            {
                "rollout_id": result["rollout_id"],
                "score": result["score"],
                "reflection": result["reflection"],
                "meta_reflection": result["meta_reflection"],
            }
            for result in results
        ],
        ensure_ascii=False,
    )
    for attempt in range(1, attempts + 1):
        agent = Agent(
            name="reflection-summarizer",
            instructions=(
                "Combine the same question's rollout reflections into concise, generalizable "
                "retrieval and meta-reflection lessons. Remove example-specific identifying "
                "content and call submit_summary."
            ),
            model=harness.context.model,
            model_settings=harness.context.model_settings,
            tools=[submit_summary],
            tool_use_behavior=StopAtTools(stop_at_tool_names=["submit_summary"]),
        )
        try:
            result, _, _ = await run_stream(
                harness.context,
                agent,
                prompt,
                f"summarizer-attempt-{attempt}",
                False,
            )
            summary = ReflectionSummary.model_validate_json(str(result.final_output))
            payload = {
                "summary": summary.summary,
                "rollout_ids": [item["rollout_id"] for item in results],
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            return payload
        except BaseException as error:
            write_error(output.parent / f"summarizer-attempt-{attempt}-error.json", error)
            if not retryable(error):
                raise
    raise RuntimeError("summarizer attempts exhausted")


def skill_path(skill_dir: Path, filename: str) -> Path:
    if Path(filename).name != filename or not filename.endswith(".md"):
        raise ValueError("skill filename must be a Markdown basename")
    return skill_dir / filename


async def update_skills(
    harness: VideoSQLHarness,
    skill_dir: Path,
    summaries: list[dict[str, Any]],
    rollout_ids: list[str],
    attempts: int,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        return json.loads(output.read_text())

    @function_tool(failure_error_function=tool_error_json)
    async def list_skill_files() -> str:
        """List current skill Markdown files."""
        return json.dumps(sorted(path.name for path in skill_dir.glob("*.md")))

    @function_tool(failure_error_function=tool_error_json)
    async def read_skill_file(filename: str) -> str:
        """Read a current skill file."""
        return skill_path(skill_dir, filename).read_text()

    @function_tool(failure_error_function=tool_error_json)
    async def write_skill_file(filename: str, content: str) -> str:
        """Create or replace a skill Markdown file."""
        path = skill_path(skill_dir, filename)
        path.write_text(content.rstrip() + "\n")
        return json.dumps({"written": filename})

    @function_tool(failure_error_function=tool_error_json)
    async def delete_skill_file(filename: str) -> str:
        """Delete an obsolete ordinary skill file."""
        if filename == "META-SKILL.md":
            raise ValueError("META-SKILL.md cannot be deleted")
        skill_path(skill_dir, filename).unlink()
        return json.dumps({"deleted": filename})

    @function_tool(failure_error_function=tool_error_json)
    async def skill_history() -> str:
        """Read the independent skill repository commit history."""
        return run_git(skill_dir, ["log", "--format=%H%n%B%n---"]).stdout

    @function_tool(failure_error_function=tool_error_json)
    async def read_skill_version(commit: str, filename: str) -> str:
        """Read a skill file from an earlier independent skill commit."""
        skill_path(skill_dir, filename)
        return run_git(skill_dir, ["show", f"{commit}:{filename}"]).stdout

    @function_tool(failure_error_function=tool_error_json)
    async def skill_diff() -> str:
        """Read the current uncommitted diff in the independent skill repository."""
        return run_git(skill_dir, ["diff", "--", "*.md"]).stdout

    @function_tool(failure_error_function=None)
    async def stop(reason: str) -> str:
        """Stop after all skill edits are complete."""
        return UpdaterStop(reason=reason).model_dump_json()

    tools = [
        list_skill_files,
        read_skill_file,
        write_skill_file,
        delete_skill_file,
        skill_history,
        read_skill_version,
        skill_diff,
        stop,
    ]
    prompt = json.dumps(summaries, ensure_ascii=False)
    for attempt in range(1, attempts + 1):
        agent = Agent(
            name="skill-updater",
            instructions=(
                "Update the skill directory from the supplied abstract summaries. Use the file "
                "tools across as many turns as needed, then call stop. You may add, replace, or "
                "delete ordinary skills and may edit META-SKILL.md, but never delete "
                "META-SKILL.md. Ordinary skills require YAML frontmatter with name and "
                "description plus Introduction, When to use, and Main content. Keep the set "
                "concise by replacing duplicated, conflicting, obsolete, or similar guidance. "
                "Every saved guidance point must include a short bold Reason explaining its "
                "evidence; one Reason may cover a whole section."
            ),
            model=harness.context.model,
            model_settings=harness.context.model_settings,
            tools=tools,
            tool_use_behavior=StopAtTools(stop_at_tool_names=["stop"]),
        )
        try:
            result, _, _ = await run_stream(
                harness.context,
                agent,
                prompt,
                f"updater-attempt-{attempt}",
                False,
            )
            stopped = UpdaterStop.model_validate_json(str(result.final_output))
            diff_result = subprocess.run(
                ["git", "diff", "--quiet"],
                cwd=skill_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if diff_result.returncode not in (0, 1):
                raise RuntimeError(diff_result.stderr.decode())
            if diff_result.returncode == 0:
                raise RuntimeError("skill updater made no changes")
            for path in skill_dir.glob("*.md"):
                content = path.read_text()
                if path.name != "META-SKILL.md" and (
                    not content.startswith("---\n")
                    or "name: " not in content
                    or "description: " not in content
                    or "# Introduction" not in content
                    or "# When to use" not in content
                    or "# Main content" not in content
                ):
                    raise RuntimeError(f"invalid skill format: {path.name}")
                if "**Reason:**" not in content:
                    raise RuntimeError(f"missing guidance reason: {path.name}")
            run_git(skill_dir, ["add", "-A"])
            body = (
                "Rollouts:\n"
                + "\n".join(rollout_ids)
                + "\n\nOverall reason:\n"
                + stopped.reason
            )
            run_git(
                skill_dir,
                ["commit", "-m", "batch skill update", "-m", body],
            )
            payload = {
                "reason": stopped.reason,
                "rollout_ids": rollout_ids,
                "skill_commit": run_git(skill_dir, ["rev-parse", "HEAD"]).stdout.strip(),
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            return payload
        except BaseException as error:
            write_error(output.parent / f"updater-attempt-{attempt}-error.json", error)
            if not retryable(error) and not isinstance(error, RuntimeError):
                raise
    raise RuntimeError("updater attempts exhausted")


async def main_async(args: argparse.Namespace) -> None:
    load_dotenv()
    api_key = os.environ[args.api_key_env]
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    skill_dir = run_dir / "skills"
    initialize_skill_repo(args.initial_skills, skill_dir)
    main_git_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    config_path = run_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        wandb_id = config["wandb_id"]
        resume = "must"
    else:
        wandb_id = wandb.util.generate_id()
        resume = "never"
        config = {
            "model": args.model_name,
            "base_url": args.base_url,
            "embedding_url": args.embedding_url,
            "embedding_model": args.embedding_model,
            "attempts": args.attempts,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "rollouts": args.rollouts,
            "seed": args.seed,
            "main_git_hash": main_git_hash,
            "wandb_id": wandb_id,
        }
        config_path.write_text(json.dumps(config, indent=2) + "\n")
    weave.init("3dsmile/video-sql-v2")
    wandb_run = wandb.init(
        entity="3dsmile",
        project="video-sql-v2",
        id=wandb_id,
        name=run_dir.name,
        config=config,
        resume=resume,
        dir=run_dir,
    )
    questions = json.loads((args.dataset_root / "month" / "train.json").read_text())
    semaphore = asyncio.Semaphore(24)
    completed_updates = 0
    consecutive_failed_batches = 0
    try:
        for epoch in range(args.epochs):
            shuffled = list(questions)
            random.Random(args.seed + epoch).shuffle(shuffled)
            batches = [
                shuffled[index : index + args.batch_size]
                for index in range(0, len(shuffled), args.batch_size)
            ]
            for batch_index, batch_questions in enumerate(batches):
                batch_root = (
                    run_dir / "train" / f"epoch-{epoch:03d}" / f"batch-{batch_index:03d}"
                )
                batch_result_path = batch_root / "batch.json"
                if batch_result_path.exists():
                    batch_result = json.loads(batch_result_path.read_text())
                    if batch_result["status"] == "updated":
                        completed_updates += 1
                    consecutive_failed_batches = (
                        consecutive_failed_batches + 1
                        if batch_result["status"] == "all_failed"
                        else 0
                    )
                    if args.max_updates is not None and completed_updates >= args.max_updates:
                        milestone(f"training stopped after {completed_updates} completed updates")
                        return
                    continue
                tasks = [
                    run_rollout(
                        semaphore,
                        question,
                        epoch,
                        batch_index,
                        rollout,
                        skill_dir,
                        args.runtime_root,
                        run_dir,
                        args.model_name,
                        api_key,
                        args.base_url,
                        args.embedding_url,
                        args.embedding_model,
                        args.attempts,
                    )
                    for question in batch_questions
                    for rollout in range(args.rollouts)
                ]
                results = await asyncio.gather(*tasks)
                successful = [result for result in results if result["status"] != "failed"]
                for result in successful:
                    wandb_run.log(
                        {
                            "train/rollout_score": result["score"],
                            "train/question_index": result["question_index"],
                            "train/rollout_id": result["rollout_id"],
                        }
                    )
                if not successful:
                    consecutive_failed_batches += 1
                    batch_root.mkdir(parents=True, exist_ok=True)
                    batch_result_path.write_text(
                        json.dumps({"status": "all_failed"}, indent=2) + "\n"
                    )
                    milestone(
                        f"epoch={epoch} batch={batch_index} all {len(results)} rollouts failed"
                    )
                    if consecutive_failed_batches == 2:
                        raise RuntimeError("two consecutive batches had no successful rollouts")
                    continue
                consecutive_failed_batches = 0
                summaries = []
                for question in batch_questions:
                    question_results = [
                        result
                        for result in successful
                        if result["question_index"] == question["index"]
                    ]
                    if not question_results:
                        continue
                    summary_dir = (
                        batch_root / f"question-{question['index']}" / "summary"
                    )
                    summary_harness = VideoSQLHarness(
                        database=args.runtime_root / "month" / "index.duckdb",
                        embedding_index=args.runtime_root / "month" / "embedding_index",
                        media_map=args.runtime_root / "month" / "media-map.json",
                        skill_dir=skill_dir,
                        run_dir=summary_dir,
                        model_name=args.model_name,
                        api_key=api_key,
                        base_url=args.base_url,
                        embedding_url=args.embedding_url,
                        embedding_model=args.embedding_model,
                        reasoning_effort="medium",
                    )
                    try:
                        summaries.append(
                            await summarize_question(
                                summary_harness,
                                question_results,
                                args.attempts,
                                summary_dir / "result.json",
                            )
                        )
                    finally:
                        await summary_harness.close()
                updater_dir = batch_root / "updater"
                updater_harness = VideoSQLHarness(
                    database=args.runtime_root / "month" / "index.duckdb",
                    embedding_index=args.runtime_root / "month" / "embedding_index",
                    media_map=args.runtime_root / "month" / "media-map.json",
                    skill_dir=skill_dir,
                    run_dir=updater_dir,
                    model_name=args.model_name,
                    api_key=api_key,
                    base_url=args.base_url,
                    embedding_url=args.embedding_url,
                    embedding_model=args.embedding_model,
                    reasoning_effort="medium",
                )
                rollout_ids = [result["rollout_id"] for result in successful]
                try:
                    update = await update_skills(
                        updater_harness,
                        skill_dir,
                        summaries,
                        rollout_ids,
                        args.attempts,
                        updater_dir / "result.json",
                    )
                finally:
                    await updater_harness.close()
                snapshot = (
                    run_dir
                    / "skill-snapshots"
                    / f"epoch-{epoch:03d}-batch-{batch_index:03d}"
                )
                shutil.copytree(
                    skill_dir,
                    snapshot,
                    ignore=shutil.ignore_patterns(".git"),
                )
                scores = [result["score"] for result in successful]
                batch_payload = {
                    "status": "updated",
                    "successful_rollouts": len(successful),
                    "failed_rollouts": len(results) - len(successful),
                    "mean_score": sum(scores) / len(scores),
                    "skill_commit": update["skill_commit"],
                }
                batch_result_path.write_text(
                    json.dumps(batch_payload, ensure_ascii=False, indent=2) + "\n"
                )
                completed_updates += 1
                wandb_run.log(
                    {
                        "train/epoch": epoch,
                        "train/batch": batch_index,
                        "train/mean_score": batch_payload["mean_score"],
                        "train/successful_rollouts": len(successful),
                        "train/failed_rollouts": len(results) - len(successful),
                        "train/completed_updates": completed_updates,
                    }
                )
                milestone(
                    f"epoch={epoch} batch={batch_index} update={completed_updates} "
                    f"successful={len(successful)} mean_score={batch_payload['mean_score']:.4f}"
                )
                if args.max_updates is not None and completed_updates >= args.max_updates:
                    milestone(f"training stopped after {completed_updates} completed updates")
                    return
        milestone("training complete")
    finally:
        wandb_run.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--initial-skills", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--embedding-url", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--rollouts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-updates", type=int)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
