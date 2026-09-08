import asyncio
import base64
import copy
import json
import random
import subprocess
import time
import uuid
from collections import Counter
from pathlib import Path

import httpx
import wandb
import weave
from agents import OpenAIResponsesModel
from wandb.sdk.lib.runid import generate_id

from src.harness import VideoSQLHarness, stripped_json
from src.val import JudgeAttemptsExhausted, judge_answer, retryable, write_error


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = Path("/raid/huggingface/hub/datasets--MM-Lifelong--MM-Lifelong/snapshots/248aa82039a574e63a2e524746a7cd8f32330443/month/train.json")
RUN_DIR = ROOT / "runs/qwen27b_train_fixed_prefix_seed0"
MODEL = "Qwen/Qwen3.8-27B-FP8"
BASE_URL = "http://127.0.0.1:8002/v1"
ATTEMPTS = 3
RETRIEVAL_TOOLS = {"semantic_search", "run_sql", "sql_qa", "clip_qa"}


class LocalResponsesModel(OpenAIResponsesModel):
    def __init__(self, context, state, request_log, phase):
        super().__init__(model=MODEL, openai_client=context.client)
        self.context = context
        self.state = state
        self.request_log = request_log
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
            is_main = any(tool.name == "submit_answer" for tool in tools)
            prefix_state = None
            if is_main:
                notes = self.context.run_dir / "agent_notes.jsonl"
                prefix_state = {
                    "base_instructions": self.context.base_instructions,
                    "compactions": self.context.compactions,
                    "notes_text": notes.read_bytes().decode() if notes.exists() else None,
                }
            self.request_log.write(json.dumps({
                "event": "start", "time": time.time(), "call_id": call_id,
                "job": str(self.context.run_dir), "phase": self.phase,
                "active_rollout": self.state["rollout"],
                "active_judge": self.state["judge"],
                "waiting_rollout": self.state["waiting"],
                "model": MODEL, "base_url": str(self.context.client.base_url),
                "instructions": system_instructions, "input": stripped_json(input),
                "model_settings": stripped_json(model_settings),
                "tools": [tool.name for tool in tools], "prefix_state": prefix_state,
            }, ensure_ascii=False) + "\n")
            async for event in super().stream_response(
                system_instructions, input, model_settings, tools,
                output_schema, handoffs, tracing, **kwargs,
            ):
                if event.type == "response.completed":
                    response_id = event.response.id
                    if is_main:
                        self.context.response_states[response_id] = prefix_state
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
                "job": str(self.context.run_dir), "phase": self.phase,
                "active_rollout": self.state["rollout"],
                "active_judge": self.state["judge"],
                "waiting_rollout": self.state["waiting"],
                "response_id": response_id, "error": error_text,
            }, ensure_ascii=False) + "\n")


def make_harness(attempt_dir, state, request_log, phase):
    runtime = ROOT / "data/runtime/month"
    harness = VideoSQLHarness(
        database=runtime / "index.duckdb", embedding_index=runtime / "embedding_index",
        media_map=runtime / "media-map.json", skill_dir=ROOT / "skills/initial",
        run_dir=attempt_dir, model_name=MODEL, api_key="local",
        base_url=BASE_URL, embedding_url="http://127.0.0.1:8013/v1",
        embedding_model="qwen3-vl-embedding-8b", context_limit=262_144,
        compaction_threshold=50_000,
    )
    harness.context.model = LocalResponsesModel(harness.context, state, request_log, phase)
    return harness


async def collect_rollout(job, job_dir, state, request_log, prefix=None):
    job_dir.mkdir(parents=True, exist_ok=True)
    result_path = job_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    answer_path = job_dir / "answer.json"
    answer = json.loads(answer_path.read_text()) if answer_path.exists() else None
    if answer is None:
        for attempt in range(1, ATTEMPTS + 1):
            attempt_dir = job_dir / f"attempt-{attempt}"
            if (attempt_dir / "error.json").exists():
                continue
            if attempt_dir.exists():
                attempt_dir.rename(job_dir / f"interrupted-attempt-{attempt}-{uuid.uuid4().hex}")
            harness = make_harness(attempt_dir, state, request_log, "rollout")
            try:
                if prefix is None:
                    session = await harness.answer(job["question"])
                else:
                    restored = copy.deepcopy(prefix["input"])
                    for item in restored:
                        if item.get("type") != "function_call_output" or not isinstance(item["output"], list):
                            continue
                        for index, block in enumerate(item["output"]):
                            if block["type"] == "input_image":
                                metadata = json.loads(item["output"][index - 1]["text"])
                                image = Path(prefix["media_dir"]) / metadata["filename"]
                                block["image_url"] = "data:image/jpeg;base64," + base64.b64encode(image.read_bytes()).decode()
                    if prefix["notes_text"] is not None:
                        (attempt_dir / "agent_notes.jsonl").write_bytes(prefix["notes_text"].encode())
                    session = await harness.answer(
                        job["question"], input_items=restored,
                        instructions=prefix["instructions"],
                        base_instructions=prefix["base_instructions"],
                        compactions=prefix["compactions"],
                    )
                answer = {
                    **job, "attempt": attempt, "status": session.status,
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
        result = {**job, "status": "failed", "answer": None, "score": None}
    elif answer["status"] != "answered":
        result = {**answer, "score": None}
    else:
        attempt_dir = job_dir / f"attempt-{answer['attempt']}"
        harness = make_harness(attempt_dir, state, request_log, "judge")
        try:
            score = await judge_answer(
                harness, job["question"], job["reference_answer"], answer["answer"], ATTEMPTS,
            )
            result = {**answer, "score": score}
        except JudgeAttemptsExhausted as error:
            result = {**answer, "status": "judge_failed", "score": None, "error": str(error)}
        finally:
            await harness.close()
        result["all_trace_files"] = sorted(str(p) for p in (attempt_dir / "traces").glob("*.json"))
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"job={job_dir.relative_to(RUN_DIR)} status={result['status']} score={result['score']}", flush=True)
    return result


async def collect_question(question, state, request_log, wandb_run):
    question_dir = RUN_DIR / "train" / question["question_id"]
    question_dir.mkdir(parents=True, exist_ok=True)
    result_path = question_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
    else:
        result_files = []
        success = None
        for rollout in range(20):
            rollout_dir = question_dir / f"rollout-{rollout:02d}"
            result = await collect_rollout(
                {**question, "rollout": rollout}, rollout_dir, state, request_log,
            )
            result_files.append(str(rollout_dir / "result.json"))
            if result["status"] == "answered" and result["score"] >= 4:
                success = str(rollout_dir / "result.json")
                break
        result = {
            **question, "status": "successful" if success is not None else "exhausted",
            "source_result": success, "rollout_results": result_files,
        }
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    state["questions_completed"] += 1
    state["questions_successful"] += int(result["status"] == "successful")
    wandb_run.log({
        "source/questions_completed": state["questions_completed"],
        "source/questions_successful": state["questions_successful"],
    })
    print(f"questions_completed={state['questions_completed']} successful={state['questions_successful']} question={question['question_id']} status={result['status']}", flush=True)
    return result


async def collect_branch(branch, state, request_log, wandb_run):
    branch_dir = RUN_DIR / "branches" / branch["question_id"] / branch["branch_id"]
    branch_dir.mkdir(parents=True, exist_ok=True)
    prefix_path = branch_dir / "prefix.json"
    if prefix_path.exists():
        prefix = json.loads(prefix_path.read_text())
    else:
        trace = json.loads(Path(branch["source_trace"]).read_text())
        prefix = {
            **trace["raw_responses"][branch["response_index"]]["prefix_state"],
            "instructions": trace["agent"]["instructions"],
            "model_settings": trace["agent"]["model_settings"], "tools": trace["agent"]["tools"],
            "input": trace["history"][:branch["history_cut"]],
            "media_dir": branch["source_media_dir"],
            "prior_trace_files": branch["source_trace_files"][:branch["trace_index"]],
        }
        prefix_path.write_text(json.dumps(prefix, ensure_ascii=False, indent=2) + "\n")
    result = await collect_rollout(branch, branch_dir, state, request_log, prefix)
    is_pair = result["status"] == "answered" and result["score"] < 3
    if is_pair:
        positive_suffix = [{"trace_file": branch["source_trace"], "history_start": branch["history_cut"]}]
        negative_suffix = []
        for paths, suffix in [
            (branch["source_trace_files"][branch["trace_index"] + 1:], positive_suffix),
            (result["trace_files"], negative_suffix),
        ]:
            for path in paths:
                trace = json.loads(Path(path).read_text())
                suffix.append({
                    "trace_file": path,
                    "history_start": len(trace["input"]) if isinstance(trace["input"], list) else 1,
                })
        pair = {
            "timeline": branch["timeline"], "question_index": branch["question_index"],
            "question": branch["question"], "reference_answer": branch["reference_answer"],
            "branch_id": branch["branch_id"], "trace_index": branch["trace_index"],
            "response_index": branch["response_index"], "prefix_file": str(prefix_path),
            "positive": {
                "answer": branch["positive_answer"], "score": branch["positive_score"],
                "source_result": branch["source_result"], "suffix": positive_suffix,
                "trace_files": branch["source_all_trace_files"], "media_dir": branch["source_media_dir"],
            },
            "negative": {
                "answer": result["answer"], "score": result["score"],
                "result_file": str(branch_dir / "result.json"), "suffix": negative_suffix,
                "trace_files": result["all_trace_files"], "media_dir": result["media_dir"],
            },
        }
        (branch_dir / "pair.json").write_text(json.dumps(pair, ensure_ascii=False, indent=2) + "\n")
    state["branches_completed"] += 1
    state["pairs"] += int(is_pair)
    wandb_run.log({"branch/completed": state["branches_completed"], "branch/pairs": state["pairs"]})
    print(f"branches_completed={state['branches_completed']} pairs={state['pairs']}", flush=True)
    return result


async def main_async():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    questions_path = RUN_DIR / "questions.json"
    if questions_path.exists():
        questions = json.loads(questions_path.read_text())
    else:
        original = json.loads(TRAIN_FILE.read_text())
        random.Random(0).shuffle(original)
        questions = [{
            "timeline": "month", "question_index": q["index"],
            "question_id": f"question-month-{q['index']}",
            "question": q["question"], "reference_answer": q["answer"],
        } for q in original]
        questions_path.write_text(json.dumps(questions, ensure_ascii=False, indent=2) + "\n")
    for source in json.loads((ROOT / "data/runtime/month/media-map.json").read_text())["sources"]:
        Path(source["path"]).stat()
    config_path = RUN_DIR / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        resume = "must"
    else:
        config = {
            "train_file": str(TRAIN_FILE), "questions": len(questions), "seed": 0,
            "model": MODEL, "judge_model": MODEL, "base_url": BASE_URL,
            "max_rollouts_per_question": 20, "attempts": ATTEMPTS,
            "success_score_min": 4, "negative_score_below": 3, "branches_per_question": 5,
            "continuation_concurrency": 50, "judge_shares_concurrency": False,
            "reasoning": {"effort": "medium"}, "context_limit": 262_144,
            "compaction_threshold": 50_000, "reasoning_item_id_policy": "preserve",
            "embedding_url": "http://127.0.0.1:8013/v1", "embedding_model": "qwen3-vl-embedding-8b",
            "skill_dir": str(ROOT / "skills/initial"),
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
    state = {
        "semaphore": asyncio.Semaphore(50), "rollout": 0, "judge": 0, "waiting": 0,
        "questions_completed": 0, "questions_successful": 0, "branches_completed": 0, "pairs": 0,
    }
    try:
        async with httpx.AsyncClient(timeout=100_000) as proxy:
            response = await proxy.post(BASE_URL.removesuffix("/v1") + "/clear_encoder_cache")
            print(f"startup encoder cache cleanup status={response.status_code} body={response.text}", flush=True)
            response.raise_for_status()
            with (RUN_DIR / "model-requests.jsonl").open("a", buffering=1) as request_log:
                print(f"formal source collection start questions={len(questions)} max_rollouts=20 concurrency=50 judge_separate=true", flush=True)
                async with asyncio.TaskGroup() as group:
                    tasks = [group.create_task(collect_question(q, state, request_log, wandb_run)) for q in questions]
                question_results = [task.result() for task in tasks]
                successful = [q for q in question_results if q["status"] == "successful"]
                (RUN_DIR / "successful_rollouts.json").write_text(json.dumps(successful, ensure_ascii=False, indent=2) + "\n")
                print(f"formal source collection complete questions={len(question_results)} successful={len(successful)}", flush=True)
                response = await proxy.post(BASE_URL.removesuffix("/v1") + "/clear_encoder_cache")
                print(f"between stages encoder cache cleanup status={response.status_code} body={response.text}", flush=True)
                response.raise_for_status()
                branches_path = RUN_DIR / "branches.json"
                if branches_path.exists():
                    branches = json.loads(branches_path.read_text())
                else:
                    selected = []
                    branches = []
                    randomizer = random.Random(0)
                    for question in sorted(successful, key=lambda q: (q["timeline"], q["question_index"])):
                        result = json.loads(Path(question["source_result"]).read_text())
                        source = {
                            **{k: question[k] for k in ("timeline", "question_index", "question_id", "question", "reference_answer")},
                            "source_result": question["source_result"], "source_rollout": result["rollout"],
                            "source_trace_files": result["trace_files"], "source_all_trace_files": result["all_trace_files"],
                            "source_media_dir": result["media_dir"],
                            "positive_answer": result["answer"], "positive_score": result["score"],
                        }
                        selected.append(source)
                        candidates = []
                        main_response = 0
                        for trace_index, path in enumerate(result["trace_files"]):
                            trace = json.loads(Path(path).read_text())
                            if not Path(path).name.startswith("answer-segment-"):
                                continue
                            for response_index, response_data in enumerate(trace["raw_responses"]):
                                if any(item["type"] == "function_call" and item["name"] in RETRIEVAL_TOOLS for item in response_data["output"]):
                                    first = response_data["output"][0]
                                    cut = next(i for i, item in enumerate(trace["history"]) if item.get("id") == first["id"])
                                    candidates.append({
                                        **source, "source_trace": path, "trace_index": trace_index,
                                        "response_index": response_index, "history_cut": cut,
                                        "branch_id": f"branch-{main_response:03d}",
                                    })
                                main_response += 1
                        indices = sorted(randomizer.sample(range(len(candidates)), min(5, len(candidates))))
                        branches.extend(candidates[index] for index in indices)
                    (RUN_DIR / "selected_sources.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n")
                    branches_path.write_text(json.dumps(branches, ensure_ascii=False, indent=2) + "\n")
                print(f"formal branch collection start branches={len(branches)}", flush=True)
                async with asyncio.TaskGroup() as group:
                    tasks = [group.create_task(collect_branch(b, state, request_log, wandb_run)) for b in branches]
                branch_results = [task.result() for task in tasks]
            pairs = [json.loads(p.read_text()) for p in sorted((RUN_DIR / "branches").glob("question-*/branch-*/pair.json"))]
            (RUN_DIR / "pairs.json").write_text(json.dumps(pairs, ensure_ascii=False, indent=2) + "\n")
            summary = {
                "questions": len(question_results), "successful_questions": len(successful),
                "source_rollouts": sum(len(q["rollout_results"]) for q in question_results),
                "branches": len(branch_results), "pairs": len(pairs),
                "branch_statuses": dict(Counter(r["status"] for r in branch_results)),
                "branch_scores": dict(Counter(r["score"] for r in branch_results)),
            }
            (RUN_DIR / "result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
            print("formal collection complete " + json.dumps(summary), flush=True)
            response = await proxy.post(BASE_URL.removesuffix("/v1") + "/clear_encoder_cache")
            print(f"final encoder cache cleanup status={response.status_code} body={response.text}", flush=True)
            response.raise_for_status()
    finally:
        wandb_run.finish()


if __name__ == "__main__":
    asyncio.run(main_async())
