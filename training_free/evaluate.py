"""Resumable training-free evaluation, accounting, and sequential latency runs."""

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import threading
import time
import traceback
from itertools import chain
from pathlib import Path
from typing import Literal
from uuid import uuid4

import httpx
import numpy as np
import openai
import pynvml
import wandb
import weave
from dotenv import load_dotenv
from prometheus_client.parser import text_string_to_metric_families
from pydantic import BaseModel, Field
from wandb.sdk.lib.runid import generate_id

from third_party.trace2skill.tracing_bridge import init_weave
from training_free.harness import INVESTIGATION_CALL_LIMIT, SHARED_ROW_IDS, SHARED_VECTORS, VideoSQLHarness, exception_chain
from training_free.index_schema import SEARCH_TABLES
from training_free.magic import load_magic_artifacts, run_magic
from training_free.normalize import normalize_text
from training_free.prepare_baselines import QA_SOURCE
from training_free.evaluate_mm_official import SYSTEM_PROMPT as MM_SYSTEM_PROMPT, parse_score as parse_mm_score


MODEL = "Qwen/Qwen3.8-27B-FP8"
OPENAI_JUDGE_MODEL = "gpt-5.6-terra"
BASE_URL = "http://127.0.0.1:8103/v1"
PD_URL = "http://127.0.0.1:8124"
EMBEDDING_URL = "http://127.0.0.1:8114/v1"
EMBEDDING_MODEL = "qwen3-vl-embedding-8b"
RUBRIC = "Evaluate whether the candidate correctly answers the question, using the reference answer and authoritative metadata. The reference is a concise correct answer, not an exhaustive list of permissible wording or supporting details. Score 5 when all facts required by the question are correct and the answer has no material contradiction. Accept ordinary synonyms, paraphrases, reversed list order, and recognizable spelling variants of the same name; do not require exact wording. Additional specificity, explanations, citations, episode IDs, timestamps, formatting, verbosity, or cautious wording are not errors by themselves. Do not assume an extra detail is false merely because the short reference omits it. A genuinely wrong additional claim or unresolved alternative that changes the answer can warrant a deduction. Judge what the question asks: a who-question requires the correct people, not every incidental event in the reference. The metadata identifies the questioner: I, me, my, mine, the narrator, and the camera wearer refer to Jake / A1_JAKE / a1_jake. Use the supplied query day and clock time to compare absolute DAYn timestamps with relative expressions such as last night, yesterday, or three hours ago. DAY3 is a day label, not automatically three days ago. Respect the reference's temporal precision: approximate hours and parts of the day such as noon do not require an exact minute match. Return a brief justification and one integer score: 0 = no valid answer or the core answer is wrong; 1 = only a few relevant correct details and the core question is unanswered; 2 = partially correct with major omissions or errors; 3 = the core is mostly correct but still has material omissions or errors; 4 = essentially complete and correct, with an identifiable minor factual error or omission relevant to the question; 5 = fully correct on the required answer and semantically equivalent at the question's intended level of detail, with no material contradiction. For any score below 5, identify the actual factual error or missing required information; stylistic differences or harmless additional context are not sufficient reasons to deduct. The score must agree with the justification."
JUDGE_VERSION = 2
COUNTERS = {"input_tokens": "vllm:prompt_tokens_total", "cached_tokens": "vllm:prompt_tokens_cached_total", "output_tokens": "vllm:generation_tokens_total"}
GAUGES = ("vllm:num_requests_running", "vllm:num_requests_waiting")
LOCAL_TOKEN_RATES = {"uncached_input": 0.15, "output": 0.47, "cached_input": 0.016}
ATTEMPTS = (1, 2, 3, 4)


class Judgment(BaseModel):
    justification: str = Field(min_length=1, description="Brief factual basis for the score; identify any actual error or required omission when deducting points.")
    score: int = Field(strict=True, ge=0, le=5)


class SimilarityJudgment(BaseModel):
    justification: str = Field(min_length=1, description="Brief analysis explaining the semantic similarity between the groundtruth and the candidate answer.")
    score: int = Field(strict=True, ge=0, le=5)


class McqSelection(BaseModel):
    selected: Literal["A", "B", "C", "D", "multiple", "none"] = Field(description="The single option letter the answer text selects; multiple when it selects more than one option; none when it selects no option.")


def collect_usage(trace_dirs, rates=LOCAL_TOKEN_RATES):
    responses = {}
    for directory in trace_dirs:
        for path in sorted(directory.rglob("*")):
            if path.suffix not in (".json", ".jsonl") or "traces" not in path.parts:
                continue
            with path.open() as handle:
                documents = (json.loads(line) for line in handle) if path.suffix == ".jsonl" else [json.load(handle)]
                pending = None
                for line_number, document in enumerate(chain(documents, [{"type": "trace_end"}])):
                    candidates = list(document["api_usage"]) if "api_usage" in document else []
                    if document.get("type") == "run_start":
                        model = document["agent"]["model"]
                    elif document.get("type") == "model_input":
                        pending = {"id": f"unreturned-request:{path}:{line_number}", "model": model, "usage": None, "request_without_response": True}
                    elif document.get("type") == "trace_end" and pending is not None:
                        candidates.append(pending)
                    if document.get("error") is not None and document.get("response") is None and document["error"]["type"] != "InvestigationBudgetExhausted":
                        request = document["request"] if "request" in document else document["input"] if "api" in document else None
                        if isinstance(request, dict) and request.get("model") in (MODEL, OPENAI_JUDGE_MODEL):
                            candidates.append({"id": f"unreturned-request:{path}", "model": request["model"], "usage": None, "request_without_response": True})
                    if isinstance(document.get("response"), dict):
                        payload = document["response"]
                        if "model" in payload and ("choices" in payload or payload.get("status") in ("completed", "incomplete", "failed", "cancelled")):
                            candidates.append(payload)
                    if document.get("type") == "raw_response_event":
                        event = document["data"]
                        if event["type"] == "response.created":
                            pending = None
                        if event["type"] in ("response.created", "response.completed", "response.incomplete", "response.failed"):
                            candidates.append(event["response"])
                    for response in candidates:
                        response_id = response["response_id"] if "response_id" in response else response["id"]
                        usage = response["usage"]
                        record = {"response_id": response_id, "model": response["model"], "usage": usage, "input_tokens": None, "cached_tokens": None, "output_tokens": None, "trace_files": [str(path)]}
                        if response.get("request_without_response"):
                            record["request_key"] = response_id
                            record["response_id"] = None
                        if usage is not None:
                            input_key = "input_tokens" if "input_tokens" in usage else "prompt_tokens"
                            output_key = "output_tokens" if "output_tokens" in usage else "completion_tokens"
                            record["input_tokens"] = usage[input_key]
                            record["output_tokens"] = usage[output_key]
                            detail_key = "input_tokens_details" if input_key == "input_tokens" else "prompt_tokens_details"
                            if isinstance(usage.get(detail_key), dict) and "cached_tokens" in usage[detail_key]:
                                record["cached_tokens"] = usage[detail_key]["cached_tokens"]
                        if response_id in responses:
                            previous = responses[response_id]
                            if previous["usage"] is None and usage is not None:
                                previous["usage"] = usage
                            for key in COUNTERS:
                                if previous[key] is not None and record[key] is not None and previous[key] != record[key]:
                                    raise ValueError(f"Conflicting usage for response {response_id}: {previous} versus {record}")
                                if previous[key] is None:
                                    previous[key] = record[key]
                            if str(path) not in previous["trace_files"]:
                                previous["trace_files"].append(str(path))
                        else:
                            responses[response_id] = record
    for record in responses.values():
        record["fee_usd"] = None
        if rates is not None and all(record[key] is not None for key in COUNTERS):
            ordinary = record["input_tokens"] - record["cached_tokens"]
            if ordinary < 0:
                raise ValueError(f"Invalid response cache accounting: {record}")
            record["fee_usd"] = (ordinary * rates["uncached_input"] + record["output_tokens"] * rates["output"] + record["cached_tokens"] * rates["cached_input"]) / 1_000_000
    return responses


def classify_failure(record):
    chain = record["error"]["chain"]
    types = {item["type"] for item in chain}
    message = "\n".join(item["message"] for item in chain)
    lower = message.lower()
    evidence = {"error_file": record["error_file"], "trace_files": record["trace_files"], "question_input": record["question_input"]}
    category = "unknown"
    cause = "Failure requires manual classification from the saved input, traceback, and traces"
    if "AnswerContextLimit" in types or any(item.get("code") == "context_length_exceeded" for item in chain) or any(text in lower for text in ("maximum context length", "context length exceeded", "context window exceeded")):
        category, cause = "algorithm", "context_overflow"
    elif "InvestigationBudgetExhausted" in types or "MAGIC exhausted max_errors=5" in message:
        category, cause = "algorithm", "budget_exhausted_without_answer"
    elif types & {"MagicOutputError", "LengthFinishReasonError", "ContentFilterFinishReasonError", "ModelBehaviorError"} and record["trace_files"]:
        category, cause = "algorithm", "invalid_model_output_or_refusal"
    elif "InvalidTrace" in types and any(text in message for text in ("answer loop ended without submit_answer", "three consecutive non-reasoning completions were identical", "returned no completed structured answer")):
        category, cause = "algorithm", "invalid_model_answer"
    elif "Method returned an invalid final answer:" in message and "candidate_answer" in record:
        category, cause = "algorithm", "invalid_model_answer"
    elif "ValidationError" in types and any((frame["file"].endswith("training_free/magic.py") and frame["name"] == "complete") or (frame["file"].endswith("training_free/harness.py") and frame["name"] == "answer") for frame in record["error"]["frames"]) and record["trace_files"]:
        category, cause = "algorithm", "invalid_structured_model_output"
    elif types & {"APIConnectionError", "APITimeoutError", "ConnectError", "ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout", "RemoteProtocolError"}:
        category, cause = "maintenance", "server_connection_or_timeout"
    elif "InternalServerError" in types:
        category, cause = "maintenance", "server_http_error"
    elif types & {"FileNotFoundError", "PermissionError", "IsADirectoryError", "NotADirectoryError", "EOFError", "UnpicklingError"}:
        category, cause = "maintenance", "missing_unreadable_or_corrupt_file"
    elif types & {"ModuleNotFoundError", "ImportError", "NameError", "UnboundLocalError", "AttributeError", "TypeError", "KeyError", "IndexError", "AssertionError"}:
        category, cause = "maintenance", "implementation_error"
    elif types & {"MemoryError", "OutOfMemoryError"} or any(text in lower for text in ("cuda out of memory", "cuda error", "device-side assert", "nccl error", "engine is dead")):
        category, cause = "maintenance", "hardware_or_engine_failure"
    elif any(text in message for text in ("Database/embedding IDs differ", "Missing offline", "Graph visual embedding differs", "Invalid video frame metadata", "Frame decode failed", "missing actual reasoning", "completed response did not include usage")):
        category, cause = "maintenance", "invalid_required_artifact_or_service_response"
    return {"category": category, "cause": cause, "evidence": evidence}


class Evaluation:
    def __init__(self, args, questions, run):
        self.args = args
        self.questions = questions
        self.run = run
        self.http = httpx.AsyncClient(timeout=60)
        self.judge_model = "gpt-5" if args.official_open_judge else OPENAI_JUDGE_MODEL if args.openai_judge else args.model
        self.judge_client = openai.AsyncOpenAI(api_key=os.environ["SHARED_OPENAI_KEY"], max_retries=0, timeout=None) if args.openai_judge else openai.AsyncOpenAI(api_key="EMPTY", base_url=args.base_url, max_retries=0, timeout=None)
        self.judgment_type = SimilarityJudgment if args.mm_lifelong_judge else Judgment
        self.answer_client = openai.AsyncOpenAI(api_key="EMPTY", base_url=args.base_url, max_retries=0, timeout=None)
        self.semaphore = asyncio.Semaphore(1 if args.method == "latency" else args.concurrency)
        self.phase_dir = args.run_dir / "phases"
        self.phase_dir.mkdir(parents=True, exist_ok=True)
        self.log_lock = threading.Lock()
        self.monitor_stop = threading.Event()
        self.monitor_ready = asyncio.Event()
        self.progress = {f"progress/{kind}/{state}": 0 for kind in ("answer", "judge") for state in ("started", "active", "completed", "failed")}
        existing_usage = collect_usage([args.run_dir], rates=None) if args.openai_judge and args.vllm_log is None else {}
        incomplete_responses = [record for record in existing_usage.values() if record["response_id"] is not None and any(record[key] is None for key in COUNTERS)]
        if incomplete_responses:
            raise ValueError(f"Completed OpenAI responses have incomplete usage: {incomplete_responses}")
        self.cloud_usage = {key: sum(record[key] for record in existing_usage.values() if record["response_id"] is not None) for key in COUNTERS}
        self.cloud_requests_without_response = sum(record["response_id"] is None for record in existing_usage.values())
        self.magic_scores = {}
        self.completed_scores = {}
        if args.method == "sql":
            reference = args.magic_reference
            reference_config = json.loads((reference / "config.json").read_text())["evaluation"]
            if reference_config["judge_rubric"] != RUBRIC or reference_config["judge_version"] != JUDGE_VERSION:
                raise ValueError("Live MAGIC comparison requires the same judge rubric and version")
            reference_questions = {str(question["id"]): question for question in reference_config["questions"]}
            for question in questions:
                qid = str(question["id"])
                if reference_questions[qid] != question:
                    raise ValueError(f"MAGIC comparison question differs: {qid}")
                if not args.concurrent_magic_reference:
                    result = json.loads((reference / "magic/questions" / qid / "result.json").read_text())
                    if result["category"] != "maintenance":
                        self.magic_scores[qid] = result["score"]
                judgment = args.run_dir / "sql/questions" / qid / "judgment.json"
                if judgment.exists():
                    self.completed_scores[qid] = json.loads(judgment.read_text())["score"]
            self.update_overlap()

    def update_overlap(self):
        if self.args.concurrent_magic_reference:
            for question in self.questions:
                qid = str(question["id"])
                if qid in self.magic_scores:
                    continue
                root = self.args.magic_reference / "magic/questions" / qid
                if (root / "judgment.json").exists():
                    self.magic_scores[qid] = json.loads((root / "judgment.json").read_text())["score"]
                elif (root / "failure.json").exists():
                    self.magic_scores[qid] = 0
        ids = sorted(self.completed_scores.keys() & self.magic_scores.keys(), key=int)
        payload = {"overlap/count": len(ids)}
        if ids:
            ours = [self.completed_scores[qid] for qid in ids]
            magic = [self.magic_scores[qid] for qid in ids]
            payload.update({"overlap/sql_mean_score": sum(ours) / len(ids), "overlap/magic_mean_score": sum(magic) / len(ids), "overlap/sql_accuracy": sum(score == 5 for score in ours) / len(ids), "overlap/magic_accuracy": sum(score == 5 for score in magic) / len(ids)})
            if self.args.mm_lifelong_judge:
                for name, scores in (("sql", ours), ("magic", magic)):
                    payload[f"overlap/{name}_accuracy"] = sum(score >= 4 for score in scores) / len(ids)
                    payload[f"overlap/{name}_mapped_mean_score"] = sum(1 if score >= 4 else 0.5 if score == 3 else 0 for score in scores) / len(ids)
            payload["overlap/mean_score_delta"] = payload["overlap/sql_mean_score"] - payload["overlap/magic_mean_score"]
            payload["overlap/accuracy_delta"] = payload["overlap/sql_accuracy"] - payload["overlap/magic_accuracy"]
        self.progress.update(payload)
        with self.log_lock:
            self.run.log(payload)

    def monitor(self, loop):
        if self.args.vllm_log is not None:
            print(f"LIVE_METRICS start source=vllm_log path={self.args.vllm_log} progress_interval_seconds=1", flush=True)
            with self.args.vllm_log.open() as source, (self.args.run_dir / "vllm-throughput.jsonl").open("a", buffering=1) as history:
                source.seek(0, os.SEEK_END)
                loop.call_soon_threadsafe(self.monitor_ready.set)
                stopped = False
                deadline = time.monotonic()
                while True:
                    if time.monotonic() >= deadline:
                        with self.log_lock:
                            self.run.log({"monitor/timestamp": time.time(), **self.progress.copy()})
                        deadline += 1
                    line = source.readline()
                    if not line:
                        if stopped:
                            return
                        stopped = self.monitor_stop.wait(min(0.2, max(0, deadline - time.monotonic())))
                        continue
                    if "throughput" not in line or not line.endswith("\n"):
                        continue
                    payload = {"monitor/timestamp": time.time(), **self.progress.copy(), "vllm_log/line": line.rstrip("\n")}
                    engine = re.search(r"Engine (\d+):", line)
                    prefix = f"vllm_log/engine_{engine.group(1)}/" if engine is not None else "vllm_log/"
                    body = line[line.index("]") + 1:]
                    if engine is not None:
                        body = body[body.index(":") + 1:]
                    if "metrics:" in body:
                        body = body[body.index("metrics:") + 8:]
                    metrics = {}
                    for chunk in body.strip().split(", "):
                        if ":" in chunk:
                            name, value = chunk.split(":", 1)
                            name = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
                            metrics[name] = []
                        else:
                            value = chunk
                        number = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*(?:tokens/s|tokens|reqs|%)?\s*", value)
                        if number is None:
                            raise ValueError(f"Unparsed vLLM log metric {chunk!r} in line: {line!r}")
                        metrics[name].append(float(number.group(1)))
                    for name, values in metrics.items():
                        if len(values) == 1:
                            payload[f"{prefix}{name}"] = values[0]
                        else:
                            payload.update({f"{prefix}{name}/{index}": value for index, value in enumerate(values)})
                    history.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    with self.log_lock:
                        self.run.log(payload)
        if self.args.openai_judge:
            previous = None
            deadline = time.monotonic()
            stopped = False
            print("LIVE_METRICS start source=openai_response_usage interval_seconds=1", flush=True)
            with (self.args.run_dir / "live-metrics.jsonl").open("a", buffering=1) as history:
                while True:
                    timestamp, monotonic = time.time(), time.monotonic()
                    with self.log_lock:
                        current = self.cloud_usage.copy()
                        payload = {"monitor/timestamp": timestamp, **self.progress.copy(), "tokens/in": current["input_tokens"] - current["cached_tokens"], "tokens/cache": current["cached_tokens"], "tokens/out": current["output_tokens"], "usage/requests_without_response": self.cloud_requests_without_response, "usage/complete": int(self.cloud_requests_without_response == 0)}
                        if previous is not None:
                            interval = monotonic - previous["monotonic"]
                            payload.update({"throughput/in_tokens_per_second": ((current["input_tokens"] - current["cached_tokens"]) - (previous["input_tokens"] - previous["cached_tokens"])) / interval, "throughput/cache_tokens_per_second": (current["cached_tokens"] - previous["cached_tokens"]) / interval, "throughput/out_tokens_per_second": (current["output_tokens"] - previous["output_tokens"]) / interval, "monitor/interval_seconds": interval})
                        history.write(json.dumps({"metrics": payload, "provider_usage": current}) + "\n")
                        self.run.log(payload)
                    loop.call_soon_threadsafe(self.monitor_ready.set)
                    previous = {**current, "monotonic": monotonic}
                    if stopped:
                        break
                    deadline += 1
                    stopped = self.monitor_stop.wait(max(0, deadline - time.monotonic()))
            return
        pynvml.nvmlInit()
        devices = {index: pynvml.nvmlDeviceGetHandleByIndex(index) for index in (4, 5, 6)}
        required = {*COUNTERS.values(), *GAUGES, "process_start_time_seconds"}
        baseline_path = self.args.run_dir / "live-baseline.json"
        baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
        previous = None
        deadline = time.monotonic()
        print(f"LIVE_METRICS start url={PD_URL}/metrics interval_seconds=1", flush=True)
        try:
            with httpx.Client(timeout=10) as client, (self.args.run_dir / "live-metrics.prom").open("a", buffering=1) as raw, (self.args.run_dir / "live-metrics.jsonl").open("a", buffering=1) as history:
                while True:
                    response = client.get(f"{PD_URL}/metrics")
                    timestamp, monotonic = time.time(), time.monotonic()
                    raw.write(f"# scrape_timestamp_seconds {timestamp}\n{response.text}\n")
                    response.raise_for_status()
                    samples = {name: [] for name in required}
                    for family in text_string_to_metric_families(response.text):
                        for sample in family.samples:
                            if sample.name in required:
                                samples[sample.name].append({"labels": sample.labels, "value": sample.value})
                    for name in required:
                        if not samples[name]:
                            raise RuntimeError(f"Required live server metric missing: {name}")
                        samples[name].sort(key=lambda item: sorted(item["labels"].items()))
                    current = {"timestamp": timestamp, "samples": samples}
                    if baseline is None:
                        baseline = current
                        baseline_path.write_text(json.dumps(baseline, indent=2) + "\n")
                    if samples["process_start_time_seconds"] != baseline["samples"]["process_start_time_seconds"]:
                        raise RuntimeError("Server restarted during this run; live token accounting requires reconciliation")
                    payload = {"monitor/timestamp": timestamp, **self.progress.copy()}
                    totals, rates = {}, {}
                    for key, metric in COUNTERS.items():
                        start, now = baseline["samples"][metric], samples[metric]
                        prior = previous["samples"][metric] if previous is not None else start
                        if len(now) != len(start) or len(now) != len(prior) or any(a["labels"] != b["labels"] or a["labels"] != c["labels"] or b["value"] < c["value"] for a, b, c in zip(start, now, prior)):
                            raise RuntimeError(f"Live server counter series changed or reset: {metric}")
                        totals[key] = sum(b["value"] - a["value"] for a, b in zip(start, now))
                        if previous is not None:
                            rates[key] = sum(b["value"] - a["value"] for a, b in zip(prior, now)) / (monotonic - previous["monotonic"])
                    payload.update({"tokens/in": totals["input_tokens"] - totals["cached_tokens"], "tokens/cache": totals["cached_tokens"], "tokens/out": totals["output_tokens"]})
                    payload["tokens/fee_usd"] = (payload["tokens/in"] * 0.15 + payload["tokens/out"] * 0.47 + payload["tokens/cache"] * 0.016) / 1_000_000
                    if previous is not None:
                        payload.update({"throughput/in_tokens_per_second": rates["input_tokens"] - rates["cached_tokens"], "throughput/cache_tokens_per_second": rates["cached_tokens"], "throughput/out_tokens_per_second": rates["output_tokens"], "monitor/interval_seconds": monotonic - previous["monotonic"]})
                    for metric in GAUGES:
                        payload[f"server/{metric.removeprefix('vllm:')}"] = sum(sample["value"] for sample in samples[metric])
                    for index, device in devices.items():
                        payload[f"gpu/{index}/utilization_percent"] = pynvml.nvmlDeviceGetUtilizationRates(device).gpu
                        payload[f"gpu/{index}/memory_used_bytes"] = pynvml.nvmlDeviceGetMemoryInfo(device).used
                    history.write(json.dumps({"metrics": payload, "server": current}) + "\n")
                    with self.log_lock:
                        self.run.log(payload)
                    loop.call_soon_threadsafe(self.monitor_ready.set)
                    previous = {**current, "monotonic": monotonic}
                    deadline += 1
                    if self.monitor_stop.wait(max(0, deadline - time.monotonic())):
                        break
        except Exception as error:
            print(f"LIVE_METRICS failed url={PD_URL}/metrics error={type(error).__name__}: {error}", flush=True)
            raise
        finally:
            pynvml.nvmlShutdown()

    async def execute(self):
        try:
            await self.monitor_ready.wait()
            for path in sorted(self.phase_dir.glob("*.json")):
                if path.name.endswith(".usage.json"):
                    continue
                phase = json.loads(path.read_text())
                if not phase["complete"]:
                    phase["recovered_after_interrupt"] = True
                    await self.finish_phase(path, phase)
            for method in ("magic", "sql") if self.args.method == "latency" else (self.args.method,):
                await self.evaluate_method(method)
        finally:
            self.monitor_stop.set()

    async def snapshot(self, path):
        if self.args.vllm_log is not None:
            snapshot = {"time": time.time(), "file": str(path), "source": str(self.args.vllm_log)}
            path.write_text(json.dumps(snapshot, indent=2) + "\n")
            return snapshot
        if self.args.openai_judge:
            with self.log_lock:
                snapshot = {"time": time.time(), "file": str(path), "tokens": self.cloud_usage.copy()}
            path.write_text(json.dumps(snapshot, indent=2) + "\n")
            return snapshot
        print(f"METRICS request={PD_URL}/metrics artifact={path}", flush=True)
        response = await self.http.get(f"{PD_URL}/metrics")
        path.write_text(response.text)
        response.raise_for_status()
        required = {*COUNTERS.values(), *GAUGES, "process_start_time_seconds"}
        samples = {name: [] for name in required}
        for family in text_string_to_metric_families(response.text):
            for sample in family.samples:
                if sample.name in required:
                    samples[sample.name].append({"labels": sample.labels, "value": sample.value})
        for name in required:
            if not samples[name]:
                raise RuntimeError(f"Required server metric {name} missing: {path}")
            samples[name].sort(key=lambda item: sorted(item["labels"].items()))
        for name in GAUGES:
            if sum(item["value"] for item in samples[name]) != 0:
                raise RuntimeError(f"Server is not idle: {name}={samples[name]}; snapshot={path}")
        return {"time": time.time(), "file": str(path), "samples": samples}

    async def finish_phase(self, path, phase):
        if "after" not in phase:
            phase["after"] = await self.snapshot(path.with_suffix(".after.prom"))
            path.write_text(json.dumps(phase, ensure_ascii=False, indent=2) + "\n")
        if self.args.vllm_log is not None:
            phase["complete"] = True
            path.write_text(json.dumps(phase, ensure_ascii=False, indent=2) + "\n")
            print(f"PHASE done={phase['name']} throughput_source={self.args.vllm_log}", flush=True)
            return
        if self.args.openai_judge:
            delta = {key: phase["after"]["tokens"][key] - phase["before"]["tokens"][key] for key in COUNTERS}
            usage = collect_usage([Path(item) for item in phase["trace_dirs"]], rates=None)
            per_request = {key: sum(item[key] for item in usage.values()) if all(item[key] is not None for item in usage.values()) else None for key in COUNTERS}
            if all(value is not None for value in per_request.values()) and per_request != delta:
                raise RuntimeError(f"OpenAI response usage differs from live counters: phase={path}; response_usage={per_request}; live_delta={delta}")
            phase["response_ids"] = sorted(record["response_id"] for record in usage.values() if record["response_id"] is not None)
            phase["requests_without_response"] = [record["request_key"] for record in usage.values() if record["response_id"] is None]
            phase["usage_complete"] = not phase["requests_without_response"]
            phase["request_tokens"] = per_request
            phase["provider_tokens"] = delta
            phase["provider_minus_requests"] = {key: delta[key] - per_request[key] if per_request[key] is not None else None for key in COUNTERS}
            phase["complete"] = True
            path.write_text(json.dumps(phase, ensure_ascii=False, indent=2) + "\n")
            (path.with_suffix(".usage.json")).write_text(json.dumps(list(usage.values()), ensure_ascii=False, indent=2) + "\n")
            print(f"PHASE done={phase['name']} tokens={delta} provider_minus_requests={phase['provider_minus_requests']}", flush=True)
            with self.log_lock:
                self.run.log({f"{phase['method']}/{phase['kind']}/{key}": value for key, value in delta.items()})
            return
        before, after = phase["before"]["samples"], phase["after"]["samples"]
        if before["process_start_time_seconds"] != after["process_start_time_seconds"]:
            path.write_text(json.dumps(phase, ensure_ascii=False, indent=2) + "\n")
            raise RuntimeError(f"Server restarted during phase; exact accounting requires manual reconciliation: {path}")
        delta = {}
        for key, metric in COUNTERS.items():
            previous, current = before[metric], after[metric]
            if len(previous) != len(current) or any(a["labels"] != b["labels"] or b["value"] < a["value"] for a, b in zip(previous, current)):
                raise RuntimeError(f"Server counter series changed or reset: {metric}; phase={path}")
            delta[key] = sum(int(b["value"] - a["value"]) for a, b in zip(previous, current))
        delta["uncached_input_tokens"] = delta["input_tokens"] - delta["cached_tokens"]
        if delta["uncached_input_tokens"] < 0:
            raise ValueError(f"Server reports more cached than input tokens: {delta}")
        delta["fee_usd"] = (delta["uncached_input_tokens"] * 0.15 + delta["output_tokens"] * 0.47 + delta["cached_tokens"] * 0.016) / 1_000_000
        phase["server_tokens"] = delta
        path.write_text(json.dumps(phase, ensure_ascii=False, indent=2) + "\n")
        usage = collect_usage([Path(item) for item in phase["trace_dirs"]])
        per_request = {key: sum(item[key] for item in usage.values()) if all(item[key] is not None for item in usage.values()) else None for key in COUNTERS}
        phase["response_ids"] = sorted(record["response_id"] for record in usage.values() if record["response_id"] is not None)
        phase["requests_without_response"] = [record["request_key"] for record in usage.values() if record["response_id"] is None]
        phase["request_tokens"] = per_request
        phase["server_minus_requests"] = {key: delta[key] - per_request[key] if per_request[key] is not None else None for key in COUNTERS}
        phase["complete"] = True
        path.write_text(json.dumps(phase, ensure_ascii=False, indent=2) + "\n")
        (path.with_suffix(".usage.json")).write_text(json.dumps(list(usage.values()), ensure_ascii=False, indent=2) + "\n")
        print(f"PHASE done={phase['name']} tokens={delta} server_minus_requests={phase['server_minus_requests']}", flush=True)
        with self.log_lock:
            self.run.log({f"{phase['method']}/{phase['kind']}/{key}": value for key, value in delta.items()})

    async def answer(self, method, question, attempt, directory):
        async with self.semaphore:
            directory.mkdir(parents=True, exist_ok=True)
            question_root = directory.parent.parent
            record = {"id": question["id"], "method": method, "attempt": attempt, "question_input": {"question": question["question"], "query_time": question["query_time"]}, "trace_dir": str(directory)}
            question_prompt = question["question"]
            if self.args.dataset == "mcq":
                question_prompt += "\n\nOptions:\n" + "\n".join(f"{letter}. {question['choices'][letter]}" for letter in "ABCD") + "\n\nSelect exactly one option. Your final answer must end with a line of the form Answer: X where X is the option letter A, B, C, or D."
                record["question_input"]["prompt"] = question_prompt
            harness = None
            if self.args.method == "latency":
                await self.snapshot(directory / "before-question.prom")
            started = time.perf_counter()
            self.progress["progress/answer/started"] += 1
            self.progress["progress/answer/active"] += 1
            try:
                if method == "magic":
                    candidate, answered_at = await asyncio.to_thread(run_magic, question_prompt, question["query_time"], directory, self.args.artifact_root, self.args.base_url, EMBEDDING_URL, self.args.model, self.args.reasoning_effort, self.args.dataset == "mcq")
                    status = "answered"
                else:
                    harness = VideoSQLHarness(database=self.args.artifact_root / "questions" / f"{question['id']}.duckdb", embedding_index=self.args.artifact_root / "embeddings", media_map=self.args.artifact_root / "media-map.json", skill_dir=Path(__file__).resolve().parent / ("skills" if self.args.semantic_search == "ppr" else "skills-vector"), run_dir=directory, model_name=self.args.model, api_key="EMPTY", embedding_url=EMBEDDING_URL, embedding_model=EMBEDDING_MODEL, query_time=question["query_time"], base_url=self.args.base_url, clip_qa_video=True, reasoning_effort=self.args.reasoning_effort, semantic_search_mode=self.args.semantic_search)
                    session = await harness.answer(question_prompt)
                    answered_at = time.perf_counter()
                    candidate, status = session.answer, session.status
                record.update({"candidate_answer": candidate, "status": status, "latency_seconds": answered_at - started})
                if status != "answered" or not isinstance(candidate, str) or not candidate.strip():
                    raise ValueError(f"Method returned an invalid final answer: status={status}, candidate={candidate!r}")
                if self.args.dataset == "mcq":
                    letters = set(re.findall(r"answer:\s*\(?([a-d])(?![a-z0-9])", normalize_text(candidate)))
                    record["answer_line_letters"] = sorted(letters)
                    if len(letters) != 1:
                        messages = [{"role": "system", "content": "Extract which option a multiple-choice answer text selects. Return the single selected option letter; return multiple when the text selects more than one option; return none when it selects no option."}, {"role": "user", "content": json.dumps({"options": question["choices"], "answer_text": candidate}, ensure_ascii=False)}]
                        trace = {"request": {"model": self.args.model, "messages": messages, "response_format": McqSelection.model_json_schema()}, "response": None, "error": None}
                        path = directory / "mcq-parse.json"
                        print(f"MCQ_PARSE method={method} id={question['id']} attempt={attempt} answer_line_letters={sorted(letters)} trace={path}", flush=True)
                        try:
                            raw = await self.answer_client.chat.completions.with_raw_response.parse(model=self.args.model, messages=messages, response_format=McqSelection)
                            trace["request"] = json.loads(raw.http_request.content)
                            trace["response"] = raw.http_response.json()
                            parsed = raw.parse().choices[0].message.parsed
                            if parsed is None:
                                raise ValueError(f"MCQ parse returned no structured selection: {trace['response']}")
                            letters = {parsed.selected}
                        except Exception as error:
                            trace["error"] = {"type": type(error).__name__, "message": str(error)}
                            raise
                        finally:
                            path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n")
                        record["mcq_parse_file"] = str(path)
                    selected = letters.pop().upper()
                    if selected not in ("A", "B", "C", "D"):
                        raise ValueError(f"MCQ answer does not identify exactly one option: {candidate!r}; parsed={selected}")
                    record["selected_choice"] = selected
                (directory / "candidate.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
                (question_root / "answer.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
                record["outcome"] = "answered"
                print(f"ANSWER method={method} id={question['id']} attempt={attempt} seconds={record['latency_seconds']:.3f}", flush=True)
            except Exception as error:
                record["outcome"] = "failed"
                record["latency_seconds"] = time.perf_counter() - started
                chain = []
                for item in exception_chain(error):
                    entry = {"type": type(item).__name__, "message": str(item)}
                    if isinstance(item, openai.APIError):
                        entry["body"] = item.body
                        entry["code"] = item.code
                        entry["request"] = item.request.content.decode()
                    chain.append(entry)
                record["error"] = {"chain": chain, "traceback": traceback.format_exc(), "frames": [{"file": frame.filename, "line": frame.lineno, "name": frame.name, "source": frame.line} for frame in traceback.extract_tb(error.__traceback__)]}
                record["error_file"] = str(directory / "error.json")
                record["trace_files"] = sorted(str(path) for path in directory.rglob("*") if path.suffix in (".json", ".jsonl") and "traces" in path.parts)
                (directory / "error.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
                print(f"ANSWER_FAILED method={method} id={question['id']} attempt={attempt} error={type(error).__name__}: {error} evidence={record['error_file']}", flush=True)
            finally:
                self.progress["progress/answer/active"] -= 1
                if harness is not None:
                    record["model_calls"] = {"investigation": harness.context.investigation_calls, "forced_final": harness.context.final_calls, "investigation_limit": INVESTIGATION_CALL_LIMIT}
                if "outcome" in record:
                    self.progress["progress/answer/completed" if record["outcome"] == "answered" else "progress/answer/failed"] += 1
                    (question_root / f"answer-attempt-{attempt}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
                if harness is not None:
                    await harness.close()

    async def judge(self, method, question, attempt, directory):
        async with self.semaphore:
            directory.mkdir(parents=True, exist_ok=True)
            question_root = directory.parent.parent
            answer = json.loads((question_root / "answer.json").read_text())
            if self.args.dataset == "mcq":
                outcome = {"id": question["id"], "attempt": attempt, "outcome": "scored", "score": 5 if answer["selected_choice"] == question["correct_choice"] else 0, "justification": f"selected={answer['selected_choice']} correct={question['correct_choice']}"}
                (question_root / "judgment.json").write_text(json.dumps(outcome, ensure_ascii=False, indent=2) + "\n")
                (question_root / f"judge-attempt-{attempt}.json").write_text(json.dumps(outcome, ensure_ascii=False, indent=2) + "\n")
                self.progress["progress/judge/started"] += 1
                self.progress["progress/judge/completed"] += 1
                if self.args.method == "sql":
                    self.completed_scores[str(question["id"])] = outcome["score"]
                    self.update_overlap()
                print(f"JUDGE method={method} id={question['id']} attempt={attempt} score={outcome['score']}", flush=True)
                return
            query_time = str(question["query_time"]).zfill(9)
            metadata = {"questioner": "A1_JAKE", "questioner_name": "Jake", "perspective": "first-person narrator and camera wearer", "query_time_raw_DHHMMSSFF": question["query_time"], "query_day": int(query_time[0]), "query_time_of_day": f"{query_time[1:3]}:{query_time[3:5]}:{query_time[5:7]}", "query_frame_index": int(query_time[7:9])}
            if self.args.official_open_judge:
                messages = [{"role": "system", "content": MM_SYSTEM_PROMPT}, {"role": "user", "content": "Question: {}\nMetadata: {}\nGroundtruth answer: {}\nCandidate answer: {}\nYour response: ".format(question["question"], json.dumps(metadata, ensure_ascii=False), question["reference_answer"], answer["candidate_answer"])}]
                traces = directory / "traces"
                traces.mkdir()
                self.progress["progress/judge/started"] += 1
                self.progress["progress/judge/active"] += 1
                request_attempt = 0
                while True:
                    request_attempt += 1
                    timeout = 60 * request_attempt
                    path = traces / f"judge-{request_attempt}.json"
                    trace = {"request": {"model": self.judge_model, "messages": messages, "timeout": timeout}, "response": None, "error": None}
                    path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n")
                    print(f"JUDGE request method={method} id={question['id']} attempt={request_attempt} timeout={timeout} trace={path}", flush=True)
                    try:
                        raw = await self.judge_client.chat.completions.with_raw_response.create(model=self.judge_model, messages=messages, timeout=timeout)
                        trace["request"] = json.loads(raw.http_request.content)
                        trace["request"]["timeout"] = timeout
                        trace["response"] = raw.http_response.json()
                        path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n")
                        content = raw.parse().choices[0].message.content.strip()
                        score = parse_mm_score(content)
                        if score < 0:
                            raise ValueError(f"Invalid official judge output: {content}")
                    except Exception as error:
                        trace["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
                        if isinstance(error, openai.APIError):
                            trace["error"]["body"] = error.body
                        path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n")
                        print(f"JUDGE_RETRY method={method} id={question['id']} attempt={request_attempt} error={type(error).__name__}: {error} trace={path}", flush=True)
                        continue
                    outcome = {"id": question["id"], "attempt": attempt, "request_attempts": request_attempt, "outcome": "scored", "score": score, "justification": content, "trace_file": str(path)}
                    (question_root / "judgment.json").write_text(json.dumps(outcome, ensure_ascii=False, indent=2) + "\n")
                    (question_root / f"judge-attempt-{attempt}.json").write_text(json.dumps(outcome, ensure_ascii=False, indent=2) + "\n")
                    self.progress["progress/judge/active"] -= 1
                    self.progress["progress/judge/completed"] += 1
                    if self.args.method == "sql":
                        self.completed_scores[str(question["id"])] = score
                        self.update_overlap()
                    print(f"JUDGE method={method} id={question['id']} attempt={attempt} request_attempts={request_attempt} score={score}", flush=True)
                    return
            messages = [{"role": "system", "content": RUBRIC}, {"role": "user", "content": json.dumps({"question": question["question"], "metadata": metadata, "reference_answer": question["reference_answer"], "candidate_answer": answer["candidate_answer"]}, ensure_ascii=False)}]
            trace = {"request": {"model": self.judge_model, "messages": messages, "reasoning_effort": "xhigh", "response_format": {"type": "json_schema", "json_schema": {"name": self.judgment_type.__name__, "schema": self.judgment_type.model_json_schema(), "strict": True}}}, "response": None, "error": None}
            traces = directory / "traces"
            traces.mkdir()
            path = traces / "judge.json"
            path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n")
            print(f"JUDGE request method={method} id={question['id']} attempt={attempt} trace={path}", flush=True)
            outcome = {"id": question["id"], "attempt": attempt, "trace_file": str(path)}
            self.progress["progress/judge/started"] += 1
            self.progress["progress/judge/active"] += 1
            try:
                raw = await self.judge_client.chat.completions.with_raw_response.parse(model=self.judge_model, messages=messages, reasoning_effort="xhigh", response_format=self.judgment_type)
                trace["request"] = json.loads(raw.http_request.content)
                trace["response"] = raw.http_response.json()
                path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n")
                if self.args.openai_judge:
                    usage = trace["response"]["usage"]
                    input_key = "input_tokens" if "input_tokens" in usage else "prompt_tokens"
                    output_key = "output_tokens" if "output_tokens" in usage else "completion_tokens"
                    detail_key = "input_tokens_details" if input_key == "input_tokens" else "prompt_tokens_details"
                    tokens = {"input_tokens": usage[input_key], "cached_tokens": usage[detail_key]["cached_tokens"], "output_tokens": usage[output_key]}
                    with self.log_lock:
                        for key in COUNTERS:
                            self.cloud_usage[key] += tokens[key]
                response = raw.parse()
                message = response.choices[0].message
                if response.choices[0].finish_reason != "stop" or message.refusal or message.parsed is None:
                    raise ValueError(f"Judge did not return a completed judgment: {trace['response']}")
                if not self.args.openai_judge:
                    reasoning = trace["response"]["choices"][0]["message"]["reasoning"]
                    if not isinstance(reasoning, str) or not reasoning.strip():
                        raise ValueError("Judge response has no actual reasoning")
                outcome.update({"outcome": "scored", "score": message.parsed.score, "justification": message.parsed.justification})
                (question_root / "judgment.json").write_text(json.dumps(outcome, ensure_ascii=False, indent=2) + "\n")
                if self.args.method == "sql":
                    self.completed_scores[str(question["id"])] = outcome["score"]
                    self.update_overlap()
                print(f"JUDGE method={method} id={question['id']} attempt={attempt} score={outcome['score']}", flush=True)
            except Exception as error:
                trace["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
                if isinstance(error, openai.APIError):
                    trace["error"]["body"] = error.body
                    trace["request"] = json.loads(error.request.content)
                if self.args.openai_judge and trace["response"] is None:
                    with self.log_lock:
                        self.cloud_requests_without_response += 1
                outcome.update({"outcome": "failed", "error": trace["error"]})
                print(f"JUDGE_FAILED method={method} id={question['id']} attempt={attempt} error={type(error).__name__}: {error} trace={path}", flush=True)
            finally:
                self.progress["progress/judge/active"] -= 1
                path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n")
                if "outcome" in outcome:
                    self.progress["progress/judge/completed" if outcome["outcome"] == "scored" else "progress/judge/failed"] += 1
                    (question_root / f"judge-attempt-{attempt}.json").write_text(json.dumps(outcome, ensure_ascii=False, indent=2) + "\n")

    async def batch(self, method, kind, attempt, questions):
        pending = []
        for question in questions:
            root = self.args.run_dir / method / "questions" / str(question["id"])
            root.mkdir(parents=True, exist_ok=True)
            if (root / ("answer.json" if kind == "answer" else "judgment.json")).exists() or (root / f"{kind}-attempt-{attempt}.json").exists():
                continue
            directory = root / f"{kind}-attempt-{attempt}" / uuid4().hex
            pending.append((question, directory))
        if not pending:
            return
        name = f"{method}-{kind}-{attempt}-{uuid4().hex}"
        path = self.phase_dir / f"{name}.json"
        phase_kind = "answer_and_judge" if kind == "answer" and self.args.method != "latency" else kind
        phase = {"name": name, "method": method, "kind": phase_kind, "attempt": attempt, "ids": [question["id"] for question, _ in pending], "trace_dirs": [str(directory) for _, directory in pending], "complete": False}
        phase["before"] = await self.snapshot(path.with_suffix(".before.prom"))
        path.write_text(json.dumps(phase, ensure_ascii=False, indent=2) + "\n")
        print(f"PHASE start={name} questions={len(pending)} concurrency={1 if self.args.method == 'latency' else self.args.concurrency}", flush=True)
        try:
            if self.args.method == "latency":
                for question, directory in pending:
                    await (self.answer(method, question, attempt, directory) if kind == "answer" else self.judge(method, question, attempt, directory))
            elif phase_kind == "answer_and_judge":
                work = iter(pending)

                async def answer_and_judge_worker():
                    for question, directory in work:
                        await self.answer(method, question, attempt, directory)
                        root = self.args.run_dir / method / "questions" / str(question["id"])
                        if (root / "answer.json").exists():
                            judge_directory = root / "judge-attempt-1" / uuid4().hex
                            phase["trace_dirs"].append(str(judge_directory))
                            path.write_text(json.dumps(phase, ensure_ascii=False, indent=2) + "\n")
                            await self.judge(method, question, 1, judge_directory)

                async with asyncio.TaskGroup() as group:
                    for _ in range(min(self.args.concurrency, len(pending))):
                        group.create_task(answer_and_judge_worker())
            else:
                async with asyncio.TaskGroup() as group:
                    for question, directory in pending:
                        group.create_task(self.answer(method, question, attempt, directory) if kind == "answer" else self.judge(method, question, attempt, directory))
        finally:
            await self.finish_phase(path, phase)

    async def evaluate_method(self, method):
        if method == "sql" and self.args.answers_from is None and self.args.semantic_search == "ppr":
            await asyncio.to_thread(load_magic_artifacts, self.args.artifact_root)
        if self.args.method == "latency":
            directory = self.args.run_dir / method / f"cache-clear-{uuid4().hex}"
            directory.mkdir(parents=True)
            await self.snapshot(directory / "before-cache-clear.prom")
            cache_trace = []
            for url in (f"{PD_URL}/reset_prefix_cache?reset_external=true", f"{EMBEDDING_URL.removesuffix('/v1')}/reset_prefix_cache", f"{EMBEDDING_URL.removesuffix('/v1')}/reset_encoder_cache", f"{BASE_URL.removesuffix('/v1')}/clear_encoder_cache"):
                print(f"CACHE_CLEAR method={method} request=POST {url}", flush=True)
                response = await self.http.post(url)
                cache_trace.append({"request": {"method": "POST", "url": url}, "status_code": response.status_code, "response": response.text})
                (directory / "cache-clear.json").write_text(json.dumps(cache_trace, indent=2) + "\n")
                response.raise_for_status()
                if "/reset_prefix_cache" in url and response.json()["success"] is not True:
                    raise RuntimeError(f"Cache reset failed: {url}: {response.text}")
            await self.snapshot(directory / "after-cache-clear.prom")
            if method == "magic":
                await asyncio.to_thread(load_magic_artifacts, self.args.artifact_root)
            else:
                key = str(self.args.artifact_root / "embeddings")
                SHARED_ROW_IDS[key] = {table: np.load(Path(key) / f"{table}.row_ids.npy", mmap_mode="r", allow_pickle=False) for table in SEARCH_TABLES}
                SHARED_VECTORS[key] = {table: np.load(Path(key) / (f"{table}.video_vectors.npy" if table == "visual_clip" else f"{table}.vectors.npy"), mmap_mode="r", allow_pickle=False) for table in SEARCH_TABLES}
        if self.args.answers_from is not None:
            for question in self.questions:
                source = self.args.answers_from / method / "questions" / str(question["id"])
                root = self.args.run_dir / method / "questions" / str(question["id"])
                root.mkdir(parents=True, exist_ok=True)
                names = ("answer.json",) if (source / "answer.json").exists() else tuple(path.name for path in sorted(source.glob("answer-attempt-*.json")))
                original = json.loads((source / names[0]).read_text())
                if original["question_input"] != {"question": question["question"], "query_time": question["query_time"]}:
                    raise ValueError(f"Rejudge source question differs: {source}")
                for name in names:
                    shutil.copyfile(source / name, root / name)
            print(f"REJUDGE reused_answers_from={self.args.answers_from} questions={len(self.questions)}", flush=True)
        else:
            for attempt in ATTEMPTS:
                await self.batch(method, "answer", attempt, self.questions)
        answerable, failures = [], []
        for question in self.questions:
            root = self.args.run_dir / method / "questions" / str(question["id"])
            if (root / "answer.json").exists():
                answerable.append(question)
            else:
                last_path = root / f"answer-attempt-{ATTEMPTS[-1]}.json"
                last = json.loads(last_path.read_text())
                failure = {"id": question["id"], "category": "failed", "stage": "answer", "attempts": len(ATTEMPTS), "last_attempt_file": str(last_path), "error": last["error"]}
                (root / "failure.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n")
                failures.append(failure)
                if self.args.method == "sql":
                    self.completed_scores[str(question["id"])] = 0
        if self.args.method == "sql":
            self.update_overlap()
        for attempt in ATTEMPTS:
            await self.batch(method, "judge", attempt, answerable)
        unresolved_judges = [question["id"] for question in answerable if not (self.args.run_dir / method / "questions" / str(question["id"]) / "judgment.json").exists()]
        for question_id in unresolved_judges:
            root = self.args.run_dir / method / "questions" / str(question_id)
            last_path = root / f"judge-attempt-{ATTEMPTS[-1]}.json"
            last = json.loads(last_path.read_text())
            failure = {"id": question_id, "category": "failed", "stage": "judge", "attempts": len(ATTEMPTS), "last_attempt_file": str(last_path), "error": last["error"]}
            (root / "failure.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n")
            failures.append(failure)
            if self.args.method == "sql":
                self.completed_scores[str(question_id)] = 0
        if self.args.method == "sql" and unresolved_judges:
            self.update_overlap()
        results = []
        for question in self.questions:
            root = self.args.run_dir / method / "questions" / str(question["id"])
            if (root / "answer.json").exists() and (root / "judgment.json").exists():
                answer = json.loads((root / "answer.json").read_text())
                judgment = json.loads((root / "judgment.json").read_text())
                result = {"id": question["id"], "category": "answered", "score": judgment["score"], "candidate_answer": answer["candidate_answer"], "latency_seconds": answer["latency_seconds"], "attempt": answer["attempt"]}
            else:
                failure = json.loads((root / "failure.json").read_text())
                result = {**failure, "score": 0}
            results.append(result)
            (root / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        included = results
        summary = {"method": method, "requested": len(results), "denominator": len(included), "mean_score": sum(item["score"] for item in included) / len(included), "score5_count": sum(item["score"] == 5 for item in included), "score5_fraction": sum(item["score"] == 5 for item in included) / len(included), "failure_count": sum(item["category"] == "failed" for item in results)}
        if self.args.mm_lifelong_judge:
            summary.update({"mapped_mean_score": sum(1 if item["score"] >= 4 else 0.5 if item["score"] == 3 else 0 for item in included) / len(included), "mapped_full_credit_count": sum(item["score"] >= 4 for item in included), "mapped_full_credit_fraction": sum(item["score"] >= 4 for item in included) / len(included), "score_distribution": {str(score): sum(item["score"] == score for item in included) for score in range(6)}})
        if self.args.dataset == "mcq":
            summary["accuracy"] = summary["score5_fraction"]
        if self.args.method == "latency":
            per_question = []
            for result in results:
                root = self.args.run_dir / method / "questions" / str(result["id"])
                attempts = [json.loads(path.read_text()) for path in sorted(root.glob("answer-attempt-*.json"))]
                per_question.append({"id": result["id"], "category": result["category"], "latency_seconds": result["latency_seconds"] if result["category"] == "answered" else None, "attempt_seconds": [item["latency_seconds"] for item in attempts]})
            complete = all(item["latency_seconds"] is not None for item in per_question)
            total = sum(item["latency_seconds"] for item in per_question) if complete else None
            summary["latency"] = {"per_question": per_question, "total_seconds": total, "mean_seconds": total / len(per_question) if complete else None}
        (self.args.run_dir / method / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        if self.args.method == "sql":
            ids = sorted(self.completed_scores.keys() & self.magic_scores.keys(), key=int)
            comparison = {"reference": str(self.args.magic_reference), "accuracy_definition": "score >= 4" if self.args.mm_lifelong_judge else "score == 5", "metrics": {key: value for key, value in self.progress.items() if key.startswith("overlap/")}, "ids": ids}
            (self.args.run_dir / "overlap-comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n")
        with self.log_lock:
            self.run.log({f"{method}/{key}": value for key, value in summary.items() if isinstance(value, (int, float))})
        print(f"METHOD_COMPLETE {json.dumps(summary, ensure_ascii=False)}", flush=True)

    def save_token_log(self):
        phases = []
        for path in sorted(self.phase_dir.glob("*.json")):
            if path.name.endswith(".usage.json"):
                continue
            phase = json.loads(path.read_text())
            phase["phase_file"] = str(path)
            phases.append(phase)
        if self.args.vllm_log is not None:
            payload = {"authoritative_source": "vLLM log throughput lines", "vllm_log": str(self.args.vllm_log), "complete": all(phase["complete"] for phase in phases), "phases": phases}
            (self.args.run_dir / "token-log.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            return
        responses = collect_usage([Path(directory) for phase in phases for directory in phase["trace_dirs"]], rates=None) if self.args.openai_judge else collect_usage([Path(directory) for phase in phases for directory in phase["trace_dirs"]])
        complete = all(phase["complete"] for phase in phases) and (not self.args.openai_judge or all(phase["usage_complete"] for phase in phases))
        totals = {}
        if complete:
            for method in sorted({phase["method"] for phase in phases}):
                totals[method] = {}
                for kind in (*sorted({phase["kind"] for phase in phases if phase["method"] == method}), "all"):
                    selected = [phase for phase in phases if phase["method"] == method and (kind == "all" or phase["kind"] == kind)]
                    fields = tuple(COUNTERS) if self.args.openai_judge else (*COUNTERS, "uncached_input_tokens", "fee_usd")
                    source = "provider_tokens" if self.args.openai_judge else "server_tokens"
                    totals[method][kind] = {key: sum(phase[source][key] for phase in selected) for key in fields}
        payload = {"authoritative_source": "OpenAI response usage" if self.args.openai_judge else f"{PD_URL}/metrics phase counter differences", "input_tokens_include_cached": True, "rates_usd_per_million": None if self.args.openai_judge else LOCAL_TOKEN_RATES, "complete": complete, "totals": totals, "responses": list(responses.values()), "phases": phases}
        (self.args.run_dir / "token-log.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        if complete:
            (self.args.run_dir / "token-metrics.json").write_text(json.dumps(totals, ensure_ascii=False, indent=2) + "\n")
            print(f"TOKEN_TOTALS {json.dumps(totals)}", flush=True)


async def main_async(args):
    global RUBRIC, JUDGE_VERSION
    args.reasoning_effort = None if args.no_reasoning_effort else "xhigh"
    if args.official_open_judge:
        if args.dataset != "open":
            raise ValueError("The official Open judge applies only to the open dataset")
        args.openai_judge = True
        args.mm_lifelong_judge = True
        RUBRIC = MM_SYSTEM_PROMPT
        JUDGE_VERSION = "mm-lifelong-official-openai-gpt-5"
    elif args.mm_lifelong_judge:
        RUBRIC = Path(__file__).with_name("mm_lifelong_rubric.txt").read_text()
        JUDGE_VERSION = "mm-lifelong-openai-gpt-5.6-terra" if args.openai_judge else "mm-lifelong"
    if args.dataset == "mcq":
        RUBRIC = "Exact match between the single selected option letter and the EgoLifeQA answer letter"
        JUDGE_VERSION = "mcq-option-letter"
    args.artifact_root = args.artifact_root.resolve()
    args.run_dir = args.run_dir.resolve()
    args.magic_reference = (args.magic_reference if args.magic_reference is not None else args.artifact_root / "results/quality-magic-judge-v2").resolve()
    if args.answers_from is not None:
        args.answers_from = args.answers_from.resolve()
        if args.method == "latency" or args.answers_from == args.run_dir:
            raise ValueError("Rejudging requires a separate output directory and method magic or sql")
        if (args.answers_from / "INVALIDATED.json").exists():
            raise ValueError(f"Cannot rejudge invalidated results: {args.answers_from}")
    if (args.run_dir / "INVALIDATED.json").exists():
        raise ValueError(f"Cannot resume invalidated results: {args.run_dir}")
    repository = Path(__file__).resolve().parents[1]
    load_dotenv(repository / ".env")
    if args.run_dir.is_relative_to(repository):
        relative = str(args.run_dir.relative_to(repository)) + "/"
        ignored = subprocess.run(["git", "check-ignore", "--quiet", relative], cwd=repository)
        if ignored.returncode != 0:
            raise ValueError(f"Run directory must be ignored before writing experiment artifacts: {args.run_dir}")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset == "mcq":
        questions = []
        for item in json.loads(QA_SOURCE.read_text()):
            choices = {letter: item["choice_" + letter.lower()] for letter in "ABCD"}
            questions.append({"id": item["ID"], "question": normalize_text(item["question"]), "query_time": int(item["query_time"]["date"].removeprefix("DAY") + item["query_time"]["time"].zfill(8)), "reference_answer": normalize_text(choices[item["answer"].upper()]), "choices": choices, "correct_choice": item["answer"].upper()})
    else:
        questions = json.loads((args.artifact_root / "questions.json").read_text())
    by_id = {str(question["id"]): question for question in questions}
    if len(by_id) != len(questions):
        raise ValueError("questions.json contains duplicate IDs")
    if args.method == "latency":
        ids = json.loads((args.artifact_root / "latency-ids.json").read_text())
        questions = [by_id[str(question_id)] for question_id in ids]
        if args.ids is not None:
            requested = set(args.ids)
            if not requested <= {str(question["id"]) for question in questions}:
                raise ValueError("Latency --ids must be drawn from latency-ids.json")
            questions = [question for question in questions if str(question["id"]) in requested]
    elif args.ids is not None:
        questions = [by_id[question_id] for question_id in args.ids]
    if not questions or args.concurrency < 1:
        raise ValueError("Evaluation requires questions and positive concurrency")
    config = {"method": args.method, "dataset": args.dataset, "artifact_root": str(args.artifact_root), "model": args.model, "base_url": args.base_url, "pd_url": PD_URL, "embedding_url": EMBEDDING_URL, "embedding_model": EMBEDDING_MODEL, "judge_model": "gpt-5" if args.official_open_judge else OPENAI_JUDGE_MODEL if args.openai_judge else args.model, "judge_api": "exact_option_letter" if args.dataset == "mcq" else "openai" if args.openai_judge else "local", "reasoning_effort": args.reasoning_effort, "concurrency": 1 if args.method == "latency" else args.concurrency, "answer_attempts": len(ATTEMPTS), "judge_attempts": "unlimited" if args.official_open_judge else len(ATTEMPTS), "questions": questions, "judge_rubric": RUBRIC, "judge_version": JUDGE_VERSION, "answer_source_run_dir": str(args.answers_from) if args.answers_from is not None else None, "investigation_call_limit": INVESTIGATION_CALL_LIMIT if args.answers_from is None else None, "live_metrics_interval_seconds": 1, "live_metrics_source": str(args.vllm_log) if args.vllm_log is not None else "openai_response_usage" if args.openai_judge else "prometheus"}
    if args.method == "sql":
        config.update({"retrieval_method": f"sql_{args.semantic_search}_semantic" if args.answers_from is None else "saved_answers", "live_magic_reference": str(args.magic_reference), "concurrent_magic_reference": args.concurrent_magic_reference, "overlap_accuracy_definition": "score >= 4" if args.mm_lifelong_judge else "score == 5"})
    config_path = args.run_dir / "config.json"
    if config_path.exists():
        saved = json.loads(config_path.read_text())
        if saved["evaluation"] != config:
            raise ValueError(f"Saved evaluation configuration differs: {config_path}")
        wandb_id, resume = saved["wandb_id"], "must"
    else:
        wandb_id, resume = generate_id(), "never"
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository, check=True, capture_output=True, text=True).stdout.strip()
        config_path.write_text(json.dumps({"evaluation": config, "wandb_id": wandb_id, "git_commit": commit}, ensure_ascii=False, indent=2) + "\n")
    init_weave()
    run = wandb.init(entity="3dsmile", project="video-sql-v2", id=wandb_id, name=args.run_dir.name, config=config, resume=resume, dir=str(args.run_dir), settings=wandb.Settings(x_stats_sampling_interval=1.0))
    evaluation = Evaluation(args, questions, run)
    try:
        async with asyncio.TaskGroup() as group:
            group.create_task(asyncio.to_thread(evaluation.monitor, asyncio.get_running_loop()))
            group.create_task(evaluation.execute())
    finally:
        try:
            evaluation.save_token_log()
        finally:
            await evaluation.judge_client.close()
            await evaluation.answer_client.close()
            await evaluation.http.aclose()
            weave.get_client().finish(use_progress_bar=False)
            weave.finish()
            run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("magic", "sql", "latency"), required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("/raid/videosql-video-artifacts"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--dataset", choices=("open", "mcq"), default="open", help="open uses artifact questions.json; mcq uses the original EgoLifeQA choices and exact option-letter scoring")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--no-reasoning-effort", action="store_true", help="Send no reasoning_effort to the answer model")
    parser.add_argument("--semantic-search", choices=("ppr", "vector"), default="ppr", help="vector restores the original per-table cosine semantic_search and its skills")
    parser.add_argument("--vllm-log", type=Path, help="Tail this vLLM log for throughput lines instead of scraping /metrics")
    parser.add_argument("--official-open-judge", action="store_true", help="Judge open answers with the official MM-Lifelong prompt on gpt-5, retrying without limit with growing timeouts")
    parser.add_argument("--concurrent-magic-reference", action="store_true", help="Read overlap scores from a MAGIC run that is still writing judgments")
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--answers-from", type=Path, help="Rejudge saved answers and reuse their final failure classifications without answering again")
    parser.add_argument("--mm-lifelong-judge", action="store_true", help="Use the MM-Lifelong semantic-similarity rubric and its 4/5=1, 3=0.5, 0/1/2=0 mapping")
    parser.add_argument("--openai-judge", action="store_true", help="Judge saved answers with gpt-5.6-terra through the OpenAI API and SHARED_OPENAI_KEY")
    parser.add_argument("--magic-reference", type=Path, help="Completed MAGIC judgments with the same rubric for live overlap")
    asyncio.run(main_async(parser.parse_args()))
