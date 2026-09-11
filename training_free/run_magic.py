"""MAGIC-video (unified graph + narrative chains) on EgoLifeQA with its own artifacts graph, the local Qwen LLM and the 8014 embedder."""

import os

# Thread caps and no GPU must be set before numpy/torch are imported (baselines_plan.md section 3).
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "10"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import asyncio
import base64
import copy
import io
import json
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import wandb
import weave
from openai import OpenAI
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training_free/MAGIC-video/src"))
sys.path.insert(0, str(ROOT / "training_free/MAGIC-video/src/EgoRAG"))

import worldmm.memory.memory as magic_memory  # noqa: E402
from worldmm.llm import PromptTemplateManager  # noqa: E402
from worldmm.memory import WorldMemory  # noqa: E402

from training_free.harness import VideoSQLHarness  # noqa: E402
from training_free.normalize import normalize_text  # noqa: E402
from training_free.run_training_free import RecordedResponsesModel  # noqa: E402
from training_free.tracing_bridge import RunMetrics, init_weave  # noqa: E402
from training_free.val import JudgeAttemptsExhausted, judge_answer, retryable, write_error  # noqa: E402

SHARED = ROOT / "runs/egolife-week-shared"
RUN = ROOT / "runs/magic-video-egolife-week"
GRAPH_PATH = SHARED / "magic_graph/graph.pkl"
ARTIFACTS = Path("/raid/huggingface/hub/datasets--jiazhengli7--magic-video-artifacts/snapshots/8573f317684ce55c0999f86386666d3d3fe76ae3/egolife/A1_JAKE")
MODEL = "Qwen/Qwen3.8-27B-FP8"
BASE_URL = "http://127.0.0.1:8003/v1"
API_KEY = "local"
EMBEDDING_URL = "http://127.0.0.1:8014/v1"
EMBEDDING_MODEL = "qwen3-vl-embedding-8b"
CONCURRENCY = 10
PHASE = "egolife"
# eval/eval_egolife.py defaults (paper config)
MAX_ROUNDS = 5
MAX_ERRORS = 5
EPISODIC_TOP_K, SEMANTIC_TOP_K, VISUAL_TOP_K = 3, 10, 3
BM25_WEIGHT = 0.3
DAMPING = 0.85
CHAIN_MIN_HITS, CHAIN_MAX_TOPICS, CHAIN_MAX_EVENTS = 1, 3, 1
CHAIN_TOPIC_SIM, CHAIN_EVENT_SIM, CHAIN_EVENT_MIN_HITS = 0.7, 0.7, 1
CHAIN_EVENT_GRANULARITIES = ("30sec", "3min")

CURRENT = threading.local()  # per-worker attempt directory for LLM/embedding call logs


class HttpEmbeddingModel:
    """Replaces MAGIC's in-process Qwen3-Embedding-4B / VLM2Vec: text and visual-query embeddings both come from 8014."""

    def __init__(self):
        self.client = httpx.Client(timeout=600)

    def encode_text(self, texts, **kwargs):
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        response = self.client.post(EMBEDDING_URL.rstrip("/") + "/embeddings", json={"model": EMBEDDING_MODEL, "input": [normalize_text(t) for t in batch]})
        response.raise_for_status()
        data = response.json()["data"]
        if len(data) != len(batch) or any(item["index"] != i for i, item in enumerate(data)):
            raise RuntimeError(f"invalid embedding response indexes: {[d['index'] for d in data]}")
        vectors = np.asarray([item["embedding"] for item in data], dtype=np.float32)
        if vectors.shape != (len(batch), 4096) or not np.isfinite(vectors).all():
            raise RuntimeError(f"invalid embedding vectors: {vectors.shape}")
        with (CURRENT.attempt_dir / "embedding_calls.jsonl").open("a") as log:
            log.write(json.dumps({"n": len(batch), "dimension": int(vectors.shape[1]), "first_text": batch[0][:200]}, ensure_ascii=False) + "\n")
        return vectors

    def encode_vis_query(self, texts, **kwargs):
        return self.encode_text(texts, **kwargs)


class LocalChatModel:
    """Replaces MAGIC's OpenRouterModel: OpenAI-compatible chat completions on the local Qwen proxy, no sampling or reasoning parameters."""

    def __init__(self, metrics, role):
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=3600, max_retries=0)
        self.model_name = MODEL
        self.metrics = metrics
        self.role = role
        self.kwargs = {}

    @staticmethod
    def image_data_url(image):
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=90)
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()

    def generate(self, prompt, **kwargs):
        if kwargs:
            raise ValueError(f"unexpected generate kwargs {sorted(kwargs)}")
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else copy.deepcopy(prompt)
        logged = []
        image_count = 0
        for message in messages:
            content = message["content"]
            if not isinstance(content, list):
                logged.append({"role": message["role"], "content": content})
                continue
            converted = []
            shown = []
            for item in content:
                if item["type"] == "text":
                    converted.append({"type": "text", "text": item["text"]})
                    shown.append(item)
                elif item["type"] == "image" and isinstance(item["image"], Image.Image):
                    converted.append({"type": "image_url", "image_url": {"url": self.image_data_url(item["image"])}})
                    shown.append({"type": "image", "size": item["image"].size})
                    image_count += 1
                else:
                    raise ValueError(f"unsupported content item type {item['type']}")
            message["content"] = converted
            logged.append({"role": message["role"], "content": shown})
        started = time.monotonic()
        response = self.client.chat.completions.create(model=self.model_name, messages=messages)
        choice = response.choices[0]
        if choice.finish_reason != "stop":
            raise RuntimeError(f"chat completion did not finish: {choice.finish_reason}")
        content = choice.message.content
        recorded = {"response_id": response.id, "agent": f"magic-{self.role}", "model": response.model,
                    "usage": {"input_tokens": response.usage.prompt_tokens, "output_tokens": response.usage.completion_tokens},
                    "service_tier": None, "status": "completed"}
        self.metrics.record_response(recorded)
        CURRENT.responses.append(recorded)
        with (CURRENT.attempt_dir / "llm_calls.jsonl").open("a") as log:
            log.write(json.dumps({"role": self.role, "images": image_count, "seconds": time.monotonic() - started, "request": logged,
                                  "response": content, "reasoning": getattr(choice.message, "reasoning_content", None),
                                  "usage": recorded["usage"], "response_id": response.id}, ensure_ascii=False) + "\n")
        return content


def make_world_memory(metrics):
    embedding_model = HttpEmbeddingModel()
    retriever = LocalChatModel(metrics, "retriever")
    responder = LocalChatModel(metrics, "respond")
    memory = WorldMemory(
        embedding_model=embedding_model, retriever_llm_model=retriever, respond_llm_model=responder,
        prompt_template_manager=PromptTemplateManager(), max_rounds=MAX_ROUNDS, max_errors=MAX_ERRORS,
        retrieval_backend="unified_graph", graph_path=str(GRAPH_PATH), cache_dir=str(RUN / "hipporag-cache-unused"),
        chain_mode="facts", topic_chain_facts_path=str(ARTIFACTS / "topic_chains/topic_chains.json"),
        event_chain_path=str(ARTIFACTS / "event_chains/step3_enriched_chains.json"),
    )
    memory.bm25_weight = BM25_WEIGHT
    memory.damping = DAMPING
    memory.chain_min_hits = CHAIN_MIN_HITS
    memory.chain_max_topics = CHAIN_MAX_TOPICS
    memory.chain_max_events = CHAIN_MAX_EVENTS
    memory.chain_topic_sim = CHAIN_TOPIC_SIM
    memory.chain_event_sim = CHAIN_EVENT_SIM
    memory.chain_event_min_hits = CHAIN_EVENT_MIN_HITS
    memory.chain_event_granularities = CHAIN_EVENT_GRANULARITIES
    memory.set_retrieval_top_k(episodic=EPISODIC_TOP_K, semantic=SEMANTIC_TOP_K, visual=VISUAL_TOP_K)
    memory.reasoning_version = "egolife"
    memory.qa_template_name = "qa_mmlifelong"  # open-ended answers (no choices); chain mode appends _chain
    caption_files = {g: str(SHARED / "magic_index/artifacts/captions" / f"A1_JAKE_{g}.json") for g in ("30sec", "3min", "10min", "1h")}
    memory.load_episodic_captions(caption_files=caption_files)
    memory.load_semantic_triples(data=json.loads((ARTIFACTS / "semantic_memory/semantic_consolidation_results_openai_gpt-oss-120b.json").read_text()))
    memory.load_visual_clips(clips_data=json.loads(Path(caption_files["30sec"]).read_text()))
    return memory


def serialize_items(items):
    out = []
    for item in items:
        if isinstance(item.content, str):
            out.append({"memory_type": item.memory_type, "content": item.content, "query": item.query, "round": item.round_num, "label": item.label})
        else:
            out.append({"memory_type": item.memory_type, "frames": len(item.content), "query": item.query, "round": item.round_num,
                        "start_ts": getattr(item, "_start_ts", None)})
    return out


def answer_question(memory, question, metrics):
    directory = RUN / "qa" / f"question-{question['ID']}"
    directory.mkdir(parents=True, exist_ok=True)
    saved = directory / "answer.json"
    if saved.exists():
        return json.loads(saved.read_text())
    if (directory / "skipped.json").exists():
        print(f"SKIP saved exhausted question: {directory}", flush=True)
        return None
    metadata = {"run": RUN.name, "question": question["ID"], "type": question["type"], "rollout": 0, "phase": PHASE,
                "cutoff_s": question["cutoff_s"], "cutoff_key": question["cutoff_key"], "until_time": question["until_time"]}
    for attempt in range(1, 4):
        attempt_dir = directory / f"attempt-{attempt}"
        if (attempt_dir / "error.json").exists():
            continue
        if attempt_dir.exists():
            attempt_dir.rename(directory / f"interrupted-{attempt}-{uuid.uuid4().hex}")
        attempt_dir.mkdir()
        CURRENT.attempt_dir = attempt_dir
        CURRENT.responses = []
        started = time.monotonic()
        try:
            # Force a fresh time-filtered subgraph for every question (WorldMemory.index skips when indexed_time >= until_time).
            memory.indexed_time = 0
            memory.unified_memory._indexed_time = 0
            with weave.attributes({**metadata, "attempt": attempt}):
                result = memory.answer(query=question["question"], choices=None, until_time=question["until_time"])
            if result.answer == "Unable to generate answer":
                raise RuntimeError("MAGIC returned its 'Unable to generate answer' fallback")
            (attempt_dir / "round_history.json").write_text(json.dumps(result.round_history, ensure_ascii=False, indent=2) + "\n")
            (attempt_dir / "retrieved_items.json").write_text(json.dumps(serialize_items(result.retrieved_items), ensure_ascii=False, indent=2) + "\n")
            responses = CURRENT.responses
            answer = {
                **metadata, "attempt": attempt, "answer": result.answer, "num_rounds": result.num_rounds,
                "directory": str(directory), "question_text": question["question"], "original_question": question["original_question"],
                "reference_answer": question["answer"],
                "trace_length": {
                    "generated_tokens": sum(r["usage"]["output_tokens"] for r in responses),
                    "model_calls": len(responses), "tool_calls": sum(1 for r in result.round_history if r["decision"] == "search"),
                    "input_tokens": sum(r["usage"]["input_tokens"] for r in responses) / len(responses),
                    "peak_context_tokens": max(r["usage"]["input_tokens"] + r["usage"]["output_tokens"] for r in responses),
                },
                "rollout_time_seconds": time.monotonic() - started + sum(json.loads(p.read_text())["seconds"] for p in directory.glob("*/rollout-time.json")),
            }
            saved.write_text(json.dumps(answer, ensure_ascii=False, indent=2) + "\n")
            print(f"ANSWER question={question['ID']} type={question['type']} rounds={result.num_rounds}", flush=True)
            return answer
        except BaseException as error:
            write_error(attempt_dir / "error.json", error)
            if not retryable(error):
                raise
        finally:
            (attempt_dir / "rollout-time.json").write_text(json.dumps({"seconds": time.monotonic() - started}) + "\n")
    (directory / "skipped.json").write_text(json.dumps({**metadata, "reason": "qa_attempts_exhausted"}, indent=2) + "\n")
    print(f"SKIP QA attempts exhausted: {directory}", flush=True)
    return None


async def judge(answer, metrics):
    directory = Path(answer["directory"])
    harness = VideoSQLHarness(
        database=SHARED / "magic_index/cutoffs" / answer["cutoff_key"] / "index.duckdb",
        embedding_index=SHARED / "magic_index/embedding_index",
        media_map=SHARED / "cutoffs" / answer["cutoff_key"] / "media-map.json",
        skill_dir=ROOT / "training_free/skills/initial", run_dir=directory / "judge",
        model_name=MODEL, api_key=API_KEY, base_url=BASE_URL,
        embedding_url=EMBEDDING_URL, embedding_model=EMBEDDING_MODEL, clip_qa_video=True,
    )
    harness.context.model = RecordedResponsesModel(harness.context, metrics)
    harness.context.model_settings.reasoning = {"effort": "xhigh", "summary": "auto"}
    try:
        return await judge_answer(harness, answer["original_question"], answer["reference_answer"], answer["answer"], 3)
    finally:
        await harness.close()


def score(answer, metrics):
    directory = Path(answer["directory"])
    result_path = directory / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    judgment_path = directory / "judgment.json"
    if judgment_path.exists():
        value = json.loads(judgment_path.read_text())["score"]
    else:
        try:
            value = asyncio.run(judge(answer, metrics))
        except JudgeAttemptsExhausted as error:
            (directory / "skipped.json").write_text(json.dumps({"reason": "judge_attempts_exhausted", "error": str(error)}) + "\n")
            print(f"SKIP judge attempts exhausted: {directory}", flush=True)
            return None
        judgment_path.write_text(json.dumps({"score": value}) + "\n")
    result = {
        **answer, "score": value, "success": value >= 4, "cost_usd": 0.0,
        "replacement_metrics": {"total_context_saved_tokens": 0, "replace_tool_calls": 0, "preemptive_qa_calls": 0, "total_compression_calls": 0},
        "retrieval_qa_metrics": {},
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"SCORE question={answer['question']} type={answer['type']} score={value}", flush=True)
    return result


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


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    init_weave()
    config = {"model": MODEL, "base_url": BASE_URL, "api": "chat.completions", "sampling_params": "server default", "reasoning": "server default",
              "embedding": EMBEDDING_MODEL, "graph": str(GRAPH_PATH), "retrieval_backend": "unified_graph", "chain_mode": "facts",
              "max_rounds": MAX_ROUNDS, "max_errors": MAX_ERRORS, "top_k": [EPISODIC_TOP_K, SEMANTIC_TOP_K, VISUAL_TOP_K], "bm25_weight": BM25_WEIGHT,
              "damping": DAMPING, "chain": [CHAIN_MIN_HITS, CHAIN_MAX_TOPICS, CHAIN_MAX_EVENTS, CHAIN_TOPIC_SIM, CHAIN_EVENT_SIM, CHAIN_EVENT_MIN_HITS],
              "qa_template": "qa_mmlifelong(_chain)", "reasoning_template": "memory_reasoning_unified_egolife", "concurrency": CONCURRENCY, "device": "cpu"}
    wandb_run = wandb.init(entity="3dsmile", project="video-sql-v2", id=RUN.name, name=RUN.name, resume="allow", dir=str(RUN), config=config)
    (RUN / "wandb-run.json").write_text(json.dumps({"id": wandb_run.id, "url": wandb_run.url}, indent=2) + "\n")
    try:
        questions = json.loads((SHARED / "questions.json").read_text())
        metrics = RunMetrics(wandb_run, PHASE, RUN)
        # MAGIC's hard-coded OpenRouter time-extraction model is replaced by the same local Qwen.
        time_model = LocalChatModel(metrics, "time-extract")
        magic_memory._time_extract_llm = time_model
        pending = [q for q in questions if not (RUN / "qa" / f"question-{q['ID']}" / "result.json").exists()
                   and not (RUN / "qa" / f"question-{q['ID']}" / "skipped.json").exists()]
        print(f"{len(pending)} questions pending of {len(questions)}", flush=True)
        results = [json.loads((RUN / "qa" / f"question-{q['ID']}" / "result.json").read_text()) for q in questions
                   if (RUN / "qa" / f"question-{q['ID']}" / "result.json").exists()]
        for result in results:
            metrics.record_rollout(result)
        lock = threading.Lock()
        jobs = iter(pending)
        abort = threading.Event()  # a non-retryable error in any worker stops the whole run

        def worker():
            memory = make_world_memory(metrics)
            print("worker memory ready", flush=True)
            for question in jobs:
                if abort.is_set():
                    return
                try:
                    answer = answer_question(memory, question, metrics)
                except BaseException:
                    abort.set()
                    raise
                if answer is None:
                    continue
                CURRENT.attempt_dir = Path(answer["directory"]) / f"attempt-{answer['attempt']}"
                CURRENT.responses = []
                result = score(answer, metrics)
                if result is not None:
                    with lock:
                        results.append(result)
                        metrics.record_rollout(result)

        with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            futures = [executor.submit(worker) for _ in range(CONCURRENCY)]
            for future in futures:
                future.result()
        if abort.is_set():
            raise RuntimeError("run aborted by a non-retryable error")
        summary = summarize(questions, sorted(results, key=lambda r: int(r["question"])))
        wandb_run.log({f"{PHASE}/final_mean_score": summary["overall"]["mean_score"], f"{PHASE}/final_accuracy_score_5": summary["overall"]["accuracy_score_5"]})
        print("MAGIC COMPLETE " + json.dumps(summary["overall"]), flush=True)
    finally:
        weave.finish()
        wandb_run.finish()


if __name__ == "__main__":
    main()
