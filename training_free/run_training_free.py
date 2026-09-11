"""Training-free harness (static skills/initial) on EgoLifeQA over the MAGIC-graph-as-SQL index, one rollout per question."""

import asyncio
import base64
import copy
import dataclasses
import heapq
import itertools
import json
import shutil
import time
import uuid
from contextvars import ContextVar
from pathlib import Path

import wandb
import weave
from agents import OpenAIResponsesModel, trace

from training_free.harness import ANSWER_CONTEXT_LIMIT, REQUEST_TRACE, InvalidTrace, VideoSQLHarness, frame_jpeg_bytes
from training_free.tracing_bridge import RunMetrics, init_weave, response_cost
from training_free.val import JudgeAttemptsExhausted, judge_answer, retryable, write_error

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "runs/egolife-week-shared"
RUN = ROOT / "runs/training-free-egolife-week-magic-index"
MODEL = "Qwen/Qwen3.8-27B-FP8"
BASE_URL = "http://127.0.0.1:8003/v1"
API_KEY = "local"
EMBEDDING_URL = "http://127.0.0.1:8014/v1"
REQUEST_PRIORITY = ContextVar("request_priority", default=2)
MAIN_AGENT_CONCURRENCY = 10
PHASE = "egolife"


class MainAgentSlots:
    def __init__(self, run):
        self.condition = asyncio.Condition()
        self.waiters = []
        self.order = itertools.count()
        self.run = run
        self.active = 0
        self.peak = 0

    async def __aenter__(self):
        ticket = (REQUEST_PRIORITY.get(), next(self.order))
        async with self.condition:
            heapq.heappush(self.waiters, ticket)
            try:
                await self.condition.wait_for(lambda: self.active < MAIN_AGENT_CONCURRENCY and self.waiters[0] == ticket)
            except asyncio.CancelledError:
                self.waiters.remove(ticket)
                heapq.heapify(self.waiters)
                self.condition.notify_all()
                raise
            heapq.heappop(self.waiters)
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.condition.notify_all()
        self.run.log({"client/active_main_agents": self.active, "client/peak_main_agents": self.peak})

    async def __aexit__(self, *exc):
        async with self.condition:
            self.active -= 1
            self.condition.notify_all()
        self.run.log({"client/active_main_agents": self.active, "client/peak_main_agents": self.peak})


class RecordedResponsesModel(OpenAIResponsesModel):
    def __init__(self, context, metrics):
        super().__init__(model=MODEL, openai_client=context.client)
        self.context = context
        self.metrics = metrics

    async def stream_response(self, *args, **kwargs):
        request = REQUEST_TRACE.get()
        async for event in super().stream_response(*args, **kwargs):
            if event.type in ("response.completed", "response.incomplete", "response.failed"):
                response = event.response
                recorded = {"response_id": response.id, "agent": request["agent"], "model": response.model,
                            "usage": None if response.usage is None else response.usage.model_dump(),
                            "service_tier": response.service_tier,
                            "status": event.type}
                self.context.response_usage[response.id] = recorded
                if request["agent"] != "judge" and event.type == "response.completed":
                    self.context.path_responses.append(recorded)
                if "snapshot" in request:
                    self.context.response_states[response.id] = request["snapshot"]
                if response.usage is None:
                    print(f"UNPRICED response={response.id} status={event.type}; preserve SDK failure retry", flush=True)
                else:
                    self.metrics.record_response(recorded)
            yield event


def make_harness(directory, metrics, cutoff_key):
    harness = VideoSQLHarness(
        database=SHARED / "magic_index/cutoffs" / cutoff_key / "index.duckdb",
        embedding_index=SHARED / "magic_index/embedding_index",
        media_map=SHARED / "cutoffs" / cutoff_key / "media-map.json",
        skill_dir=RUN / "skills/initial", run_dir=directory,
        model_name=MODEL, api_key=API_KEY, base_url=BASE_URL,
        embedding_url=EMBEDDING_URL, embedding_model="qwen3-vl-embedding-8b",
        clip_qa_video=True,
    )
    harness.context.model = RecordedResponsesModel(harness.context, metrics)
    harness.context.model_settings.reasoning = {"effort": "xhigh", "summary": "auto"}
    original_clip_response = harness.context.clip_response

    async def recorded_clip_response(*args, **kwargs):
        raw, recorded = await original_clip_response(*args, **kwargs)
        metrics.record_response(recorded)
        return raw, recorded

    harness.context.clip_response = recorded_clip_response
    original_summary_response = harness.context.summary_response

    async def recorded_summary_response(*args, **kwargs):
        submission, recorded = await original_summary_response(*args, **kwargs)
        metrics.record_response(recorded)
        return submission, recorded

    harness.context.summary_response = recorded_summary_response
    return harness


@weave.op(postprocess_inputs=lambda x: {k: v for k, v in x.items() if k != "metrics"})
async def rollout(question, metrics):
    directory = RUN / "qa" / f"question-{question['ID']}" / "rollout-00"
    directory.mkdir(parents=True, exist_ok=True)
    saved_answer = directory / "answer.json"
    if saved_answer.exists():
        return json.loads(saved_answer.read_text())
    if (directory / "skipped.json").exists():
        print(f"SKIP saved exhausted rollout: {directory}", flush=True)
        return None
    metadata = {"run": RUN.name, "question": question["ID"], "type": question["type"], "rollout": 0,
                "skill_version": "initial", "phase": PHASE, "cutoff_s": question["cutoff_s"], "cutoff_key": question["cutoff_key"]}
    for attempt in range(1, 4):
        attempt_dir = directory / f"attempt-{attempt}"
        if (attempt_dir / "error.json").exists():
            continue
        if attempt_dir.exists():
            attempt_dir.rename(directory / f"interrupted-{attempt}-{uuid.uuid4().hex}")
        harness = make_harness(attempt_dir, metrics, question["cutoff_key"])
        harness.context.model_settings.extra_body = {"prompt_cache_key": f"{RUN.name}:{question['ID']}"}
        started = time.monotonic()
        try:
            with weave.attributes({**metadata, "attempt": attempt}), trace("Training-free EgoLife QA rollout", metadata=metadata):
                session = await harness.answer(question["question"])
                if session.status != "answered" or session.answer is None:
                    raise InvalidTrace(f"QA ended without an answer: {session.status}")
            path_responses = harness.context.path_responses
            trajectory = attempt_dir / "trajectory.json"
            trajectory.write_text(json.dumps({"histories": harness.context.path_histories}, ensure_ascii=False, indent=2) + "\n")
            answer = {
                **metadata, "attempt": attempt, **dataclasses.asdict(session),
                "directory": str(directory), "trajectory_file": str(trajectory),
                "question_text": question["question"], "original_question": question["original_question"],
                "reference_answer": question["answer"],
                "media_files": sorted(str(p) for p in (attempt_dir / "media").glob("frame-*.json")),
                "trace_length": {
                    "generated_tokens": sum(r["usage"]["output_tokens"] for r in path_responses),
                    "model_calls": len(path_responses), "tool_calls": len(harness.context.tool_calls),
                    "input_tokens": sum(r["usage"]["input_tokens"] for r in path_responses) / len(path_responses),
                    "peak_context_tokens": max(r["usage"]["input_tokens"] + r["usage"]["output_tokens"] for r in path_responses),
                },
                "rollout_time_seconds": time.monotonic() - started + sum(
                    json.loads(p.read_text())["seconds"] for p in directory.glob("*/rollout-time.json")),
            }
            saved_answer.write_text(json.dumps(answer, ensure_ascii=False, indent=2) + "\n")
            print(f"ANSWER question={question['ID']} type={question['type']}", flush=True)
            return answer
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            write_error(attempt_dir / "error.json", error)
            if not retryable(error):
                raise
        finally:
            (attempt_dir / "rollout-time.json").write_text(json.dumps({"seconds": time.monotonic() - started}) + "\n")
            await harness.close()
    (directory / "skipped.json").write_text(json.dumps({**metadata, "reason": "qa_attempts_exhausted"}, indent=2) + "\n")
    print(f"SKIP QA attempts exhausted: {directory}", flush=True)
    return None


@weave.op(postprocess_inputs=lambda x: {k: v for k, v in x.items() if k != "metrics"})
async def score_rollout(answer, metrics):
    directory = Path(answer["directory"])
    result_path = directory / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    if (directory / "skipped.json").exists():
        return None
    attempt_dir = directory / f"attempt-{answer['attempt']}"
    harness = make_harness(attempt_dir, metrics, answer["cutoff_key"])
    try:
        judgment_path = directory / "judgment.json"
        if judgment_path.exists():
            score = json.loads(judgment_path.read_text())["score"]
        else:
            try:
                score = await judge_answer(harness, answer["original_question"], answer["reference_answer"], answer["answer"], 3)
            except JudgeAttemptsExhausted as error:
                (directory / "skipped.json").write_text(json.dumps({"reason": "judge_attempts_exhausted", "error": str(error)}) + "\n")
                print(f"SKIP judge attempts exhausted: {directory}", flush=True)
                return None
            judgment_path.write_text(json.dumps({"score": score}) + "\n")
        trace_files = sorted(str(p) for p in directory.rglob("traces/*.json"))
        billable = {}
        replacement_call_ids = set()
        retrieval_call_ids = {"run_sql": set(), "semantic_search": set()}
        qa_call_ids = {"run_sql": set(), "semantic_search": set()}
        for filename in trace_files:
            artifact = json.loads(Path(filename).read_text())
            for response in artifact["api_usage"]:
                billable[response["response_id"]] = response
            if artifact["agent"]["name"] != "judge":
                replacement_call_ids.update(call["call_id"] for call in artifact["tool_calls"] if call["name"] == "replace_tool")
                for call in artifact["tool_calls"]:
                    if call["name"] in retrieval_call_ids:
                        retrieval_call_ids[call["name"]].add(call["call_id"])
                        arguments = json.loads(call["arguments"])
                        if "question" in arguments and arguments["question"] is not None:
                            qa_call_ids[call["name"]].add(call["call_id"])
        total_context_saved = 0
        for path in sorted(directory.glob("*/replace-*.json")):
            measurement = json.loads(path.read_text())
            if measurement["replace_call_id"] not in replacement_call_ids:
                continue
            for side in ("before", "after"):
                if f"{side}_tokens" in measurement:
                    continue
                items = copy.deepcopy(measurement[side])
                output = items[1]["output"]
                if isinstance(output, list):
                    for index, part in enumerate(output):
                        if part["type"] == "input_image":
                            recipe_name = json.loads(output[index - 1]["text"])["filename"]
                            recipe = path.parent / "media" / recipe_name
                            image_bytes = await asyncio.to_thread(frame_jpeg_bytes, json.loads(recipe.read_text()))
                            part["image_url"] = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode()
                print(f"count replacement tokens side={side} artifact={path}", flush=True)
                token_count = await harness.context.client.responses.input_tokens.count(
                    model=MODEL, input=[{"role": "user", "content": answer["question_text"]}, *items],
                    reasoning=measurement["reasoning"],
                )
                measurement[f"{side}_tokens"] = token_count.input_tokens
                path.write_text(json.dumps(measurement, ensure_ascii=False) + "\n")
            total_context_saved += measurement["before_tokens"] - measurement["after_tokens"]
        result = {
            **answer, "score": score, "success": score >= 4, "trace_files": trace_files,
            "cost_usd": sum(response_cost(r) for r in billable.values()),
            "replacement_metrics": {"total_context_saved_tokens": total_context_saved,
                                    "replace_tool_calls": len(replacement_call_ids),
                                    "preemptive_qa_calls": sum(len(ids) for ids in qa_call_ids.values()),
                                    "total_compression_calls": len(replacement_call_ids) + sum(len(ids) for ids in qa_call_ids.values())},
            "retrieval_qa_metrics": {name: {"tool_calls": len(retrieval_call_ids[name]), "qa_calls": len(qa_call_ids[name])} for name in retrieval_call_ids},
        }
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(f"SCORE question={answer['question']} type={answer['type']} score={score}", flush=True)
        return result
    finally:
        await harness.close()


def summarize(questions, results):
    by_id = {r["question"]: r for r in results}
    skipped = [q["ID"] for q in questions if q["ID"] not in by_id]
    groups = {}
    for r in results:
        groups.setdefault(r["type"], []).append(r)
    def stats(rows):
        return {"questions": len(rows), "total_score": sum(r["score"] for r in rows), "mean_score": sum(r["score"] for r in rows) / len(rows),
                "accuracy_score_5": sum(r["score"] == 5 for r in rows) / len(rows), "score_ge_4_fraction": sum(r["score"] >= 4 for r in rows) / len(rows)}
    summary = {"requested_questions": len(questions), "completed_questions": len(results), "skipped_question_ids": skipped,
               "overall": stats(results), "by_type": {t: stats(rows) for t, rows in sorted(groups.items())}}
    (RUN / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


async def main():
    RUN.mkdir(parents=True, exist_ok=True)
    init_weave()
    config = {
        "model": MODEL, "reasoning": "xhigh", "reasoning_summary": "auto", "base_url": BASE_URL, "api": "responses",
        "main_agent_concurrency": MAIN_AGENT_CONCURRENCY, "replace_tool": True, "answer_context_limit": ANSWER_CONTEXT_LIMIT,
        "clip_qa_video": True, "index": "magic-video-artifacts graph as SQL", "questions": "EgoLifeQA A1_JAKE open-ended", "rollouts": 1,
    }
    wandb_run = wandb.init(entity="3dsmile", project="video-sql-v2", id=RUN.name, name=RUN.name, resume="allow", dir=str(RUN), config=config)
    (RUN / "wandb-run.json").write_text(json.dumps({"id": wandb_run.id, "url": wandb_run.url}, indent=2) + "\n")
    try:
        initial = RUN / "skills/initial"
        if not initial.exists():
            shutil.copytree(ROOT / "training_free/skills/initial", initial)
        for source in (ROOT / "training_free/skills/initial").glob("*.md"):
            if (initial / source.name).read_bytes() != source.read_bytes():
                raise RuntimeError(f"Initial skill changed: {source.name}")
        questions = json.loads((SHARED / "questions.json").read_text())
        main_slots = MainAgentSlots(wandb_run)
        metrics = RunMetrics(wandb_run, PHASE, RUN)
        jobs = iter(questions)
        results = []
        lock = asyncio.Lock()

        async def worker():
            for question in jobs:
                async with main_slots:
                    answer = await rollout(question, metrics)
                if answer is None:
                    continue
                result = await score_rollout(answer, metrics)
                if result is not None:
                    async with lock:
                        results.append(result)
                        metrics.record_rollout(result)

        async with asyncio.TaskGroup() as group:
            for _ in range(MAIN_AGENT_CONCURRENCY):
                group.create_task(worker())
        summary = summarize(questions, sorted(results, key=lambda r: int(r["question"])))
        wandb_run.log({f"{PHASE}/final_mean_score": summary["overall"]["mean_score"], f"{PHASE}/final_accuracy_score_5": summary["overall"]["accuracy_score_5"]})
        print("TRAINING-FREE COMPLETE " + json.dumps(summary["overall"]), flush=True)
    finally:
        weave.finish()
        wandb_run.finish()


if __name__ == "__main__":
    asyncio.run(main())
