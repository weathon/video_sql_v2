import asyncio
import base64
import copy
import json
import random
import shutil
import subprocess
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import wandb
import weave
from agents import OpenAIResponsesModel
from dotenv import dotenv_values
from wandb.sdk.lib.runid import generate_id

from src.harness import VideoSQLHarness, stripped_json
from src.val import judge_answer, retryable, write_error


ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUN = ROOT / "runs/training_free_luna_seed0_7510e535"
RUN_DIR = ROOT / "runs/luna_fixed_prefix_seed0"
MODEL = "gpt-5.6-luna"
ATTEMPTS = 3
RETRIEVAL_TOOLS = {"semantic_search", "run_sql", "sql_qa", "clip_qa"}


class CollectionResponsesModel(OpenAIResponsesModel):
    def __init__(self, client, state, request_log, branch, phase):
        super().__init__(model=MODEL, openai_client=client)
        self.state = state
        self.request_log = request_log
        self.branch = branch
        self.phase = phase

    async def stream_response(
        self, system_instructions, input, model_settings, tools,
        output_schema, handoffs, tracing, **kwargs,
    ):
        call_id = uuid.uuid4().hex
        if self.phase == "rollout":
            self.state["waiting"] += 1
            try:
                await self.state["semaphore"].acquire()
            finally:
                self.state["waiting"] -= 1
        self.state[self.phase] += 1
        response_id = None
        error_text = None
        try:
            self.request_log.write(json.dumps({
                "event": "start", "time": time.time(), "call_id": call_id,
                "branch": self.branch, "phase": self.phase,
                "active_rollout": self.state["rollout"],
                "active_judge": self.state["judge"],
                "waiting_rollout": self.state["waiting"],
                "model": MODEL, "instructions": system_instructions,
                "input": stripped_json(input),
                "model_settings": stripped_json(model_settings),
            }, ensure_ascii=False) + "\n")
            async for event in super().stream_response(
                system_instructions, input, model_settings, tools,
                output_schema, handoffs, tracing, **kwargs,
            ):
                if event.type == "response.completed":
                    response_id = event.response.id
                yield event
        except BaseException as error:
            error_text = f"{type(error).__name__}: {error}"
            raise
        finally:
            self.state[self.phase] -= 1
            if self.phase == "rollout":
                self.state["semaphore"].release()
            self.request_log.write(json.dumps({
                "event": "end", "time": time.time(), "call_id": call_id,
                "branch": self.branch, "phase": self.phase,
                "active_rollout": self.state["rollout"],
                "active_judge": self.state["judge"],
                "waiting_rollout": self.state["waiting"],
                "response_id": response_id, "error": error_text,
            }, ensure_ascii=False) + "\n")


async def collect_branch(branch, api_key, state, request_log, wandb_run):
    branch_dir = RUN_DIR / "branches" / branch["question_id"] / branch["branch_id"]
    branch_dir.mkdir(parents=True, exist_ok=True)
    result_path = branch_dir / "result.json"
    answer_path = branch_dir / "answer.json"
    source_trace = json.loads(Path(branch["source_trace"]).read_text())
    prefix = source_trace["history"][:branch["history_cut"]]
    prefix_path = branch_dir / "prefix.json"
    if not prefix_path.exists():
        prefix_path.write_text(json.dumps({
            "instructions": source_trace["agent"]["instructions"],
            "model_settings": source_trace["agent"]["model_settings"],
            "tools": source_trace["agent"]["tools"],
            "input": prefix,
            "media_dir": branch["source_media_dir"],
        }, ensure_ascii=False, indent=2) + "\n")
    if result_path.exists():
        result = json.loads(result_path.read_text())
    else:
        answer = json.loads(answer_path.read_text()) if answer_path.exists() else None
        if answer is None:
            for attempt in range(1, ATTEMPTS + 1):
                attempt_dir = branch_dir / f"attempt-{attempt}"
                if (attempt_dir / "error.json").exists():
                    continue
                if attempt_dir.exists():
                    attempt_dir.rename(branch_dir / f"interrupted-attempt-{attempt}-{uuid.uuid4().hex}")
                runtime = ROOT / "data/runtime" / branch["timeline"]
                harness = VideoSQLHarness(
                    database=runtime / "index.duckdb",
                    embedding_index=runtime / "embedding_index",
                    media_map=runtime / "media-map.json",
                    skill_dir=ROOT / "skills/initial",
                    run_dir=attempt_dir,
                    model_name=MODEL, api_key=api_key,
                    embedding_url="http://127.0.0.1:8013/v1",
                    embedding_model="qwen3-vl-embedding-8b",
                )
                harness.context.model = CollectionResponsesModel(
                    harness.context.client, state, request_log,
                    str(attempt_dir), "rollout",
                )
                try:
                    restored = copy.deepcopy(prefix)
                    for item in restored:
                        if item.get("type") != "function_call_output" or not isinstance(item["output"], list):
                            continue
                        for index, block in enumerate(item["output"]):
                            if block["type"] == "input_image":
                                metadata = json.loads(item["output"][index - 1]["text"])
                                image = Path(branch["source_media_dir"]) / metadata["filename"]
                                block["image_url"] = "data:image/jpeg;base64," + base64.b64encode(image.read_bytes()).decode()
                    session = await harness.answer(
                        branch["question"], input_items=restored,
                        instructions=source_trace["agent"]["instructions"],
                    )
                    answer = {
                        **branch, "attempt": attempt, "status": session.status,
                        "answer": session.answer, "trace_files": session.trace_files,
                        "media_dir": str(attempt_dir / "media"),
                    }
                    answer_path.write_text(json.dumps(answer, ensure_ascii=False, indent=2) + "\n")
                    break
                except asyncio.CancelledError:
                    raise
                except BaseException as error:
                    write_error(attempt_dir / "error.json", error)
                    if not retryable(error):
                        raise
                finally:
                    await harness.close()
        if answer is None:
            result = {**branch, "status": "failed", "answer": None, "score": None}
        elif answer["status"] != "answered":
            result = {**answer, "score": None}
        else:
            attempt_dir = branch_dir / f"attempt-{answer['attempt']}"
            runtime = ROOT / "data/runtime" / branch["timeline"]
            harness = VideoSQLHarness(
                database=runtime / "index.duckdb",
                embedding_index=runtime / "embedding_index",
                media_map=runtime / "media-map.json",
                skill_dir=ROOT / "skills/initial", run_dir=attempt_dir,
                model_name=MODEL, api_key=api_key,
                embedding_url="http://127.0.0.1:8013/v1",
                embedding_model="qwen3-vl-embedding-8b",
            )
            harness.context.model = CollectionResponsesModel(
                harness.context.client, state, request_log,
                str(attempt_dir), "judge",
            )
            try:
                score = await judge_answer(
                    harness, branch["question"], branch["reference_answer"],
                    answer["answer"], ATTEMPTS,
                )
                result = {
                    **answer, "score": score,
                    "all_trace_files": sorted(str(p) for p in (attempt_dir / "traces").glob("*.json")),
                }
            finally:
                await harness.close()
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    is_pair = result["status"] == "answered" and result["score"] < 3
    if is_pair:
        suffix = []
        for path in result["trace_files"]:
            trace = json.loads(Path(path).read_text())
            suffix.append({
                "trace_file": path,
                "history_start": len(trace["input"]) if isinstance(trace["input"], list) else 1,
            })
        pair = {
            "timeline": branch["timeline"], "question_index": branch["question_index"],
            "question": branch["question"], "reference_answer": branch["reference_answer"],
            "branch_id": branch["branch_id"], "response_index": branch["response_index"],
            "prefix_file": str(prefix_path),
            "positive": {
                "answer": branch["positive_answer"], "score": branch["positive_score"],
                "source_result": branch["source_result"],
                "suffix": [{"trace_file": branch["source_trace"], "history_start": branch["history_cut"]}],
                "media_dir": branch["source_media_dir"],
            },
            "negative": {
                "answer": result["answer"], "score": result["score"],
                "result_file": str(result_path), "suffix": suffix,
                "trace_files": result["all_trace_files"], "media_dir": result["media_dir"],
            },
        }
        (branch_dir / "pair.json").write_text(json.dumps(pair, ensure_ascii=False, indent=2) + "\n")
    state["completed"] += 1
    state["pairs"] += int(is_pair)
    wandb_run.log({
        "collection/completed": state["completed"], "collection/pairs": state["pairs"],
        "collection/score": result["score"],
    })
    print(
        f"completed={state['completed']} pairs={state['pairs']} "
        f"question={branch['question_id']} branch={branch['branch_id']} "
        f"status={result['status']} score={result['score']}", flush=True,
    )
    return result


async def main_async():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    branches_path = RUN_DIR / "branches.json"
    if branches_path.exists():
        branches = json.loads(branches_path.read_text())
    else:
        questions = {
            (q["timeline"], q["index"]): q
            for q in json.loads((ROOT / "data/test-seed-0.json").read_text())
        }
        successful = []
        grouped = defaultdict(list)
        for path in sorted((SOURCE_RUN / "val").glob("question-*/rollout-*/result.json")):
            result = json.loads(path.read_text())
            if result["score"] < 4:
                continue
            successful.append({"source_result": str(path), "result": result})
            trace_paths = sorted((path.parent / f"attempt-{result['attempt']}" / "traces").glob("answer-segment-*.json"))
            traces = [json.loads(p.read_text()) for p in trace_paths]
            turns = sum(
                any(item["type"] == "function_call" for item in response["output"])
                for trace in traces for response in trace["raw_responses"]
            )
            grouped[(result["timeline"], result["question_index"])].append(
                (turns, result["rollout"], path, result, trace_paths)
            )
        (RUN_DIR / "successful_rollouts.json").write_text(json.dumps(successful, ensure_ascii=False, indent=2) + "\n")
        randomizer = random.Random(0)
        selected = []
        branches = []
        for key in sorted(grouped):
            turns, rollout, path, result, trace_paths = min(grouped[key], key=lambda x: (x[0], x[1]))
            if len(trace_paths) != 1:
                raise ValueError(f"selected source has compaction: {path}")
            trace = json.loads(trace_paths[0].read_text())
            question = questions[key]
            if question["answer"] != result["reference_answer"]:
                raise ValueError(f"reference answer differs from source: {path}")
            question_id = f"question-{key[0]}-{key[1]}"
            archived = RUN_DIR / "sources" / question_id / f"rollout-{rollout:02d}"
            source_attempt = path.parent / f"attempt-{result['attempt']}"
            archived_attempt = archived / source_attempt.name
            shutil.copytree(source_attempt, archived_attempt, dirs_exist_ok=True)
            shutil.copy2(path, archived / "result.json")
            source = {
                "timeline": key[0], "question_index": key[1], "question_id": question_id,
                "question": question["question"], "reference_answer": result["reference_answer"],
                "source_rollout": rollout, "source_turns": turns,
                "original_result": str(path), "source_result": str(archived / "result.json"),
                "source_trace": str(archived_attempt / "traces" / trace_paths[0].name),
                "source_media_dir": str(archived_attempt / "media"),
                "original_attempt_dir": str(source_attempt),
                "positive_answer": result["answer"], "positive_score": result["score"],
            }
            selected.append(source)
            candidates = [
                index for index, response in enumerate(trace["raw_responses"])
                if any(item["type"] == "function_call" and item["name"] in RETRIEVAL_TOOLS for item in response["output"])
            ]
            for response_index in sorted(randomizer.sample(candidates, min(5, len(candidates)))):
                first = trace["raw_responses"][response_index]["output"][0]
                cut = next(i for i, item in enumerate(trace["history"]) if item.get("id") == first["id"])
                branches.append({
                    **source, "branch_id": f"branch-{response_index:03d}",
                    "response_index": response_index, "history_cut": cut,
                })
        (RUN_DIR / "selected_sources.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n")
        branches_path.write_text(json.dumps(branches, ensure_ascii=False, indent=2) + "\n")
    api_key = dotenv_values(ROOT / ".env")["PERSONAL_OPENAI_KEY"]
    if not api_key:
        raise ValueError("PERSONAL_OPENAI_KEY must have a value in the project .env")
    config_path = RUN_DIR / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        resume = "must"
    else:
        config = {
            "source_run": str(SOURCE_RUN), "model": MODEL, "judge_model": MODEL,
            "seed": 0, "branches_per_question": 5, "continuation_concurrency": 50,
            "judge_shares_concurrency": False, "attempts": ATTEMPTS,
            "positive_score_min": 4, "negative_score_below": 3,
            "reasoning": {"effort": "medium", "context": "all_turns"},
            "reasoning_item_id_policy": "preserve", "branches": len(branches),
            "embedding_url": "http://127.0.0.1:8013/v1",
            "embedding_model": "qwen3-vl-embedding-8b",
            "git_hash": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "wandb_id": generate_id(),
        }
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
        resume = "never"
    weave.init("3dsmile/video-sql-v2")
    wandb_run = wandb.init(
        entity="3dsmile", project="video-sql-v2", id=config["wandb_id"],
        name=RUN_DIR.name, config=config, resume=resume, dir=RUN_DIR,
    )
    state = {"semaphore": asyncio.Semaphore(50), "rollout": 0, "judge": 0, "waiting": 0, "completed": 0, "pairs": 0}
    print(f"formal collection start branches={len(branches)} continuation_concurrency=50 judge_separate=true", flush=True)
    try:
        with (RUN_DIR / "model-requests.jsonl").open("a", buffering=1) as request_log:
            async with asyncio.TaskGroup() as group:
                tasks = [group.create_task(collect_branch(branch, api_key, state, request_log, wandb_run)) for branch in branches]
            results = [task.result() for task in tasks]
        pair_files = sorted((RUN_DIR / "branches").glob("question-*/branch-*/pair.json"))
        pairs = [json.loads(path.read_text()) for path in pair_files]
        (RUN_DIR / "pairs.json").write_text(json.dumps(pairs, ensure_ascii=False, indent=2) + "\n")
        summary = {
            "branches": len(branches), "completed": len(results), "pairs": len(pairs),
            "statuses": dict(Counter(result["status"] for result in results)),
            "scores": dict(Counter(result["score"] for result in results)),
        }
        (RUN_DIR / "result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        print("formal collection complete " + json.dumps(summary), flush=True)
    finally:
        wandb_run.finish()


if __name__ == "__main__":
    asyncio.run(main_async())
