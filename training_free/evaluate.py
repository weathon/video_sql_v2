"""Resumable training-free evaluation, accounting, and sequential latency runs."""

import argparse
import asyncio
import json
import subprocess
import time
import traceback
from pathlib import Path
from uuid import uuid4

import httpx
import numpy as np
import openai
import wandb
import weave
from dotenv import load_dotenv
from prometheus_client.parser import text_string_to_metric_families
from pydantic import BaseModel, Field
from wandb.sdk.lib.runid import generate_id

from third_party.trace2skill.tracing_bridge import init_weave
from training_free.harness import SHARED_ROW_IDS, SHARED_VECTORS, VideoSQLHarness, exception_chain
from training_free.index_schema import SEARCH_TABLES
from training_free.magic import load_magic_artifacts, run_magic


MODEL = "Qwen/Qwen3.8-27B-FP8"
BASE_URL = "http://127.0.0.1:8103/v1"
PD_URL = "http://127.0.0.1:8124"
EMBEDDING_URL = "http://127.0.0.1:8114/v1"
EMBEDDING_MODEL = "qwen3-vl-embedding-8b"
RUBRIC = "Judge the candidate answer against the reference answer in the context of the question. Return an integer score from 0 through 5: 0 = wrong or no answer; 1 = a few correct details but the core question is unanswered; 2 = partially correct with major omissions or errors; 3 = the core is mostly correct but still has material omissions or errors; 4 = complete except for minor issues; 5 = fully correct and semantically equivalent, with no contradiction."
COUNTERS = {"input_tokens": "vllm:prompt_tokens_total", "cached_tokens": "vllm:prompt_tokens_cached_total", "output_tokens": "vllm:generation_tokens_total"}
GAUGES = ("vllm:num_requests_running", "vllm:num_requests_waiting")


class Judgment(BaseModel):
    score: int = Field(strict=True, ge=0, le=5)


def collect_usage(trace_dirs):
    responses = {}
    for directory in trace_dirs:
        for path in sorted(directory.rglob("*")):
            if path.suffix not in (".json", ".jsonl") or "traces" not in path.parts:
                continue
            with path.open() as handle:
                documents = (json.loads(line) for line in handle) if path.suffix == ".jsonl" else [json.load(handle)]
                for document in documents:
                    candidates = list(document["api_usage"]) if "api_usage" in document else []
                    if isinstance(document.get("response"), dict):
                        payload = document["response"]
                        if "model" in payload and ("choices" in payload or payload.get("status") in ("completed", "incomplete", "failed", "cancelled")):
                            candidates.append(payload)
                    if document.get("type") == "raw_response_event":
                        event = document["data"]
                        if event["type"] in ("response.completed", "response.incomplete", "response.failed"):
                            candidates.append(event["response"])
                    for response in candidates:
                        response_id = response["response_id"] if "response_id" in response else response["id"]
                        usage = response["usage"]
                        record = {"response_id": response_id, "model": response["model"], "usage": usage, "input_tokens": None, "cached_tokens": None, "output_tokens": None, "trace_files": [str(path)]}
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
        if all(record[key] is not None for key in COUNTERS):
            ordinary = record["input_tokens"] - record["cached_tokens"]
            if ordinary < 0:
                raise ValueError(f"Invalid response cache accounting: {record}")
            record["fee_usd"] = (ordinary * 0.15 + record["output_tokens"] * 0.47 + record["cached_tokens"] * 0.016) / 1_000_000
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
    elif types & {"FileNotFoundError", "PermissionError", "IsADirectoryError", "NotADirectoryError", "EOFError", "UnpicklingError"}:
        category, cause = "maintenance", "missing_unreadable_or_corrupt_file"
    elif types & {"ModuleNotFoundError", "ImportError", "NameError", "UnboundLocalError", "AttributeError", "TypeError", "KeyError", "IndexError", "AssertionError"}:
        category, cause = "maintenance", "implementation_error"
    elif types & {"MemoryError", "OutOfMemoryError"} or any(text in lower for text in ("cuda out of memory", "cuda error", "device-side assert", "nccl error", "engine is dead")):
        category, cause = "maintenance", "hardware_or_engine_failure"
    elif any(text in message for text in ("Database/embedding IDs differ", "Missing offline", "Graph visual embedding differs", "Invalid video frame metadata", "Frame decode failed", "missing actual reasoning_content", "completed response did not include usage")):
        category, cause = "maintenance", "invalid_required_artifact_or_service_response"
    return {"category": category, "cause": cause, "evidence": evidence}


class Evaluation:
    def __init__(self, args, questions, run):
        self.args = args
        self.questions = questions
        self.run = run
        self.http = httpx.AsyncClient(timeout=60)
        self.judge_client = openai.AsyncOpenAI(api_key="EMPTY", base_url=BASE_URL, max_retries=0)
        self.semaphore = asyncio.Semaphore(1 if args.method == "latency" else args.concurrency)
        self.phase_dir = args.run_dir / "phases"
        self.phase_dir.mkdir(parents=True, exist_ok=True)

    async def snapshot(self, path):
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
        phase["response_ids"] = sorted(usage)
        phase["request_tokens"] = per_request
        phase["server_minus_requests"] = {key: delta[key] - per_request[key] if per_request[key] is not None else None for key in COUNTERS}
        phase["complete"] = True
        path.write_text(json.dumps(phase, ensure_ascii=False, indent=2) + "\n")
        (path.with_suffix(".usage.json")).write_text(json.dumps(list(usage.values()), ensure_ascii=False, indent=2) + "\n")
        print(f"PHASE done={phase['name']} tokens={delta} server_minus_requests={phase['server_minus_requests']}", flush=True)
        self.run.log({f"{phase['method']}/{phase['kind']}/{key}": value for key, value in delta.items()})

    async def answer(self, method, question, attempt, directory):
        async with self.semaphore:
            directory.mkdir(parents=True, exist_ok=True)
            question_root = directory.parent.parent
            record = {"id": question["id"], "method": method, "attempt": attempt, "question_input": {"question": question["question"], "query_time": question["query_time"]}, "trace_dir": str(directory)}
            harness = None
            if self.args.method == "latency":
                await self.snapshot(directory / "before-question.prom")
            started = time.perf_counter()
            try:
                if method == "magic":
                    candidate, answered_at = await asyncio.to_thread(run_magic, question["question"], question["query_time"], directory, self.args.artifact_root, BASE_URL, EMBEDDING_URL)
                    status = "answered"
                else:
                    harness = VideoSQLHarness(database=self.args.artifact_root / "questions" / f"{question['id']}.duckdb", embedding_index=self.args.artifact_root / "embeddings", media_map=self.args.artifact_root / "media-map.json", skill_dir=Path(__file__).resolve().parent / "skills", run_dir=directory, model_name=MODEL, api_key="EMPTY", embedding_url=EMBEDDING_URL, embedding_model=EMBEDDING_MODEL, query_time=question["query_time"], base_url=BASE_URL, clip_qa_video=True)
                    session = await harness.answer(question["question"])
                    answered_at = time.perf_counter()
                    candidate, status = session.answer, session.status
                record.update({"candidate_answer": candidate, "status": status, "latency_seconds": answered_at - started})
                if status != "answered" or not isinstance(candidate, str) or not candidate.strip():
                    raise ValueError(f"Method returned an invalid final answer: status={status}, candidate={candidate!r}")
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
                if "outcome" in record:
                    (question_root / f"answer-attempt-{attempt}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
                if harness is not None:
                    await harness.close()

    async def judge(self, method, question, attempt, directory):
        async with self.semaphore:
            directory.mkdir(parents=True, exist_ok=True)
            question_root = directory.parent.parent
            answer = json.loads((question_root / "answer.json").read_text())
            messages = [{"role": "system", "content": RUBRIC}, {"role": "user", "content": json.dumps({"question": question["question"], "reference_answer": question["reference_answer"], "candidate_answer": answer["candidate_answer"]}, ensure_ascii=False)}]
            trace = {"request": {"model": MODEL, "messages": messages, "reasoning_effort": "xhigh", "response_format": {"type": "json_schema", "json_schema": {"name": "Judgment", "schema": Judgment.model_json_schema(), "strict": True}}}, "response": None, "error": None}
            traces = directory / "traces"
            traces.mkdir()
            path = traces / "judge.json"
            path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n")
            print(f"JUDGE request method={method} id={question['id']} attempt={attempt} trace={path}", flush=True)
            outcome = {"id": question["id"], "attempt": attempt, "trace_file": str(path)}
            try:
                raw = await self.judge_client.chat.completions.with_raw_response.parse(model=MODEL, messages=messages, reasoning_effort="xhigh", response_format=Judgment)
                trace["request"] = json.loads(raw.http_request.content)
                trace["response"] = raw.http_response.json()
                path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n")
                response = raw.parse()
                message = response.choices[0].message
                if response.choices[0].finish_reason != "stop" or message.refusal or message.parsed is None:
                    raise ValueError(f"Judge did not return a completed judgment: {trace['response']}")
                reasoning = trace["response"]["choices"][0]["message"]["reasoning_content"]
                if not isinstance(reasoning, str) or not reasoning.strip():
                    raise ValueError("Judge response has no actual reasoning_content")
                outcome.update({"outcome": "scored", "score": message.parsed.score})
                (question_root / "judgment.json").write_text(json.dumps(outcome, ensure_ascii=False, indent=2) + "\n")
                print(f"JUDGE method={method} id={question['id']} attempt={attempt} score={outcome['score']}", flush=True)
            except Exception as error:
                trace["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
                if isinstance(error, openai.APIError):
                    trace["error"]["body"] = error.body
                    trace["request"] = json.loads(error.request.content)
                outcome.update({"outcome": "failed", "error": trace["error"]})
                print(f"JUDGE_FAILED method={method} id={question['id']} attempt={attempt} error={type(error).__name__}: {error} trace={path}", flush=True)
            finally:
                path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n")
                if "outcome" in outcome:
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
        phase = {"name": name, "method": method, "kind": kind, "attempt": attempt, "ids": [question["id"] for question, _ in pending], "trace_dirs": [str(directory) for _, directory in pending], "complete": False}
        phase["before"] = await self.snapshot(path.with_suffix(".before.prom"))
        path.write_text(json.dumps(phase, ensure_ascii=False, indent=2) + "\n")
        print(f"PHASE start={name} questions={len(pending)} concurrency={1 if self.args.method == 'latency' else self.args.concurrency}", flush=True)
        try:
            if self.args.method == "latency":
                for question, directory in pending:
                    await (self.answer(method, question, attempt, directory) if kind == "answer" else self.judge(method, question, attempt, directory))
            else:
                async with asyncio.TaskGroup() as group:
                    for question, directory in pending:
                        group.create_task(self.answer(method, question, attempt, directory) if kind == "answer" else self.judge(method, question, attempt, directory))
        finally:
            await self.finish_phase(path, phase)

    async def evaluate_method(self, method):
        if self.args.method == "latency":
            directory = self.args.run_dir / method / f"cache-clear-{uuid4().hex}"
            directory.mkdir(parents=True)
            await self.snapshot(directory / "before-cache-clear.prom")
            cache_trace = []
            for url in (f"{PD_URL}/reset_prefix_cache", f"{EMBEDDING_URL.removesuffix('/v1')}/reset_prefix_cache", f"{EMBEDDING_URL.removesuffix('/v1')}/reset_encoder_cache", f"{BASE_URL.removesuffix('/v1')}/clear_encoder_cache"):
                print(f"CACHE_CLEAR method={method} request=POST {url}", flush=True)
                response = await self.http.post(url)
                cache_trace.append({"request": {"method": "POST", "url": url}, "status_code": response.status_code, "response": response.text})
                (directory / "cache-clear.json").write_text(json.dumps(cache_trace, indent=2) + "\n")
                response.raise_for_status()
            await self.snapshot(directory / "after-cache-clear.prom")
            if method == "magic":
                await asyncio.to_thread(load_magic_artifacts, self.args.artifact_root)
            else:
                key = str(self.args.artifact_root / "embeddings")
                SHARED_ROW_IDS[key] = {table: np.load(Path(key) / f"{table}.row_ids.npy", mmap_mode="r", allow_pickle=False) for table in SEARCH_TABLES}
                SHARED_VECTORS[key] = {table: np.load(Path(key) / (f"{table}.video_vectors.npy" if table == "visual_clip" else f"{table}.vectors.npy"), mmap_mode="r", allow_pickle=False) for table in SEARCH_TABLES}
        for attempt in (1, 2):
            await self.batch(method, "answer", attempt, self.questions)
        answerable, failures = [], []
        for question in self.questions:
            root = self.args.run_dir / method / "questions" / str(question["id"])
            if (root / "answer.json").exists():
                answerable.append(question)
            else:
                failure = classify_failure(json.loads((root / "answer-attempt-2.json").read_text()))
                failure["id"] = question["id"]
                (root / "failure.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n")
                failures.append(failure)
        for attempt in (1, 2):
            await self.batch(method, "judge", attempt, answerable)
        unresolved_judges = [question["id"] for question in answerable if not (self.args.run_dir / method / "questions" / str(question["id"]) / "judgment.json").exists()]
        unknown = [failure for failure in failures if failure["category"] == "unknown"]
        if unknown or unresolved_judges:
            blocked = {"unknown_failures": unknown, "unresolved_judges": unresolved_judges}
            (self.args.run_dir / method / "blocked.json").write_text(json.dumps(blocked, ensure_ascii=False, indent=2) + "\n")
            raise RuntimeError(f"Summary blocked by unknown failure causes or unresolved judge errors: {blocked}")
        results = []
        for question in self.questions:
            root = self.args.run_dir / method / "questions" / str(question["id"])
            if (root / "answer.json").exists():
                answer = json.loads((root / "answer.json").read_text())
                judgment = json.loads((root / "judgment.json").read_text())
                result = {"id": question["id"], "category": "answered", "score": judgment["score"], "candidate_answer": answer["candidate_answer"], "latency_seconds": answer["latency_seconds"], "attempt": answer["attempt"]}
            else:
                failure = json.loads((root / "failure.json").read_text())
                result = {**failure, "score": 0 if failure["category"] == "algorithm" else None}
            results.append(result)
            (root / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        included = [result for result in results if result["category"] != "maintenance"]
        if not included:
            raise RuntimeError(f"No questions remain in the denominator for {method}")
        summary = {"method": method, "requested": len(results), "denominator": len(included), "mean_score": sum(item["score"] for item in included) / len(included), "score5_count": sum(item["score"] == 5 for item in included), "score5_fraction": sum(item["score"] == 5 for item in included) / len(included), "algorithm_failure_count": sum(item["category"] == "algorithm" for item in results), "maintenance_failure_count": sum(item["category"] == "maintenance" for item in results)}
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
        responses = collect_usage([Path(directory) for phase in phases for directory in phase["trace_dirs"]])
        complete = all(phase["complete"] for phase in phases)
        totals = {}
        if complete:
            for method in sorted({phase["method"] for phase in phases}):
                totals[method] = {}
                for kind in ("answer", "judge", "all"):
                    selected = [phase for phase in phases if phase["method"] == method and (kind == "all" or phase["kind"] == kind)]
                    totals[method][kind] = {key: sum(phase["server_tokens"][key] for phase in selected) for key in (*COUNTERS, "uncached_input_tokens", "fee_usd")}
        payload = {"authoritative_source": f"{PD_URL}/metrics phase counter differences", "input_tokens_include_cached": True, "rates_usd_per_million": {"uncached_input": 0.15, "output": 0.47, "cached_input": 0.016}, "complete": complete, "totals": totals, "responses": list(responses.values()), "phases": phases}
        (self.args.run_dir / "token-log.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        if complete:
            (self.args.run_dir / "token-metrics.json").write_text(json.dumps(totals, ensure_ascii=False, indent=2) + "\n")
            print(f"TOKEN_TOTALS {json.dumps(totals)}", flush=True)


async def main_async(args):
    args.artifact_root = args.artifact_root.resolve()
    args.run_dir = args.run_dir.resolve()
    repository = Path(__file__).resolve().parents[1]
    load_dotenv(repository / ".env")
    if args.run_dir.is_relative_to(repository):
        relative = str(args.run_dir.relative_to(repository)) + "/"
        ignored = subprocess.run(["git", "check-ignore", "--quiet", relative], cwd=repository)
        if ignored.returncode != 0:
            raise ValueError(f"Run directory must be ignored before writing experiment artifacts: {args.run_dir}")
    args.run_dir.mkdir(parents=True, exist_ok=True)
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
    config = {"method": args.method, "artifact_root": str(args.artifact_root), "model": MODEL, "base_url": BASE_URL, "pd_url": PD_URL, "embedding_url": EMBEDDING_URL, "embedding_model": EMBEDDING_MODEL, "reasoning_effort": "xhigh", "concurrency": 1 if args.method == "latency" else args.concurrency, "answer_attempts": 2, "judge_attempts": 2, "questions": questions, "judge_rubric": RUBRIC}
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
    run = wandb.init(entity="3dsmile", project="video-sql-v2", id=wandb_id, name=args.run_dir.name, config=config, resume=resume, dir=str(args.run_dir))
    evaluation = Evaluation(args, questions, run)
    try:
        for path in sorted(evaluation.phase_dir.glob("*.json")):
            if path.name.endswith(".usage.json"):
                continue
            phase = json.loads(path.read_text())
            if not phase["complete"]:
                phase["recovered_after_interrupt"] = True
                await evaluation.finish_phase(path, phase)
        for method in ("magic", "sql") if args.method == "latency" else (args.method,):
            await evaluation.evaluate_method(method)
    finally:
        try:
            evaluation.save_token_log()
        finally:
            await evaluation.judge_client.close()
            await evaluation.http.aclose()
            weave.finish()
            run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("magic", "sql", "latency"), required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("/raid/videosql-video-artifacts"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--ids", nargs="+")
    asyncio.run(main_async(parser.parse_args()))
