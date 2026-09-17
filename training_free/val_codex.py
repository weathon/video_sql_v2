"""Evaluate Codex VideoSQL on Month validation or EgoLife Open; --ids selects an independent debug run."""

import argparse
import json
import os
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.metadata import version
from pathlib import Path

import wandb
from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI
from openai_codex import Codex, CodexConfig

from training_free.codex_harness import AttemptFailed, EMBEDDING_MODEL, MODEL, QA_MODEL, ROOT, SKILLS, answer_question, instructions_for, trace_json
from training_free.evaluate_mm_official import SYSTEM_PROMPT, build_prompt, parse_score, score_mapping


def judge_messages(dataset, question, answer):
    if dataset == "mmlifelong-month":
        system = SYSTEM_PROMPT.split("\n\nEgoLife metadata rules:", 1)[0]
        content = f"Question: {question['question']}\nGroundtruth answer: {question['reference_answer']}\nCandidate answer: {answer}\nYour response: "
    else:
        time = str(question["query_time"]).zfill(9)
        metadata = {"questioner": "A1_JAKE", "questioner_name": "Jake", "perspective": "first-person narrator and camera wearer", "query_time_raw_DHHMMSSFF": question["query_time"], "query_day": int(time[0]), "query_time_of_day": f"{time[1:3]}:{time[3:5]}:{time[5:7]}", "query_frame_index": int(time[7:9])}
        system = SYSTEM_PROMPT
        content = build_prompt({"question": question["question"], "metadata": metadata, "answer": question["reference_answer"], "pred": {"answer": answer}})
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def usage_records(directory):
    records = []
    terra = directory / "usage.json"
    usage = json.loads(terra.read_text()) if terra.exists() else None
    records.append({"role": "terra", "usage": None if usage is None else {"input_tokens": usage["inputTokens"], "output_tokens": usage["outputTokens"], "cached_tokens": usage["cachedInputTokens"], "total_tokens": usage["totalTokens"]}})
    for path in sorted((directory / "traces").glob("*.json")):
        if not path.name.startswith(("sql-qa-", "semantic-qa-", "clip-")):
            continue
        record = json.loads(path.read_text())
        response = record["response"]
        usage = None if response is None else response["usage"]
        if usage is not None:
            chat = record["api"] == "chat.completions"
            detail_key = "prompt_tokens_details" if chat else "input_tokens_details"
            details = usage.get(detail_key)
            usage = {"input_tokens": usage["prompt_tokens" if chat else "input_tokens"], "output_tokens": usage["completion_tokens" if chat else "output_tokens"], "total_tokens": usage["total_tokens"], "cached_tokens": None if details is None else details["cached_tokens"]}
        records.append({"role": record["agent"], "usage": usage})
    return records


class Evaluation:
    def __init__(self, dataset, artifact_root, run_dir, qa_url, embedding_url, concurrency, baseline_run, ids):
        self.dataset = dataset
        self.artifact_root = artifact_root.resolve(strict=True)
        self.run_dir = run_dir.resolve()
        self.qa_url = qa_url
        self.embedding_url = embedding_url
        self.concurrency = concurrency
        self.baseline_run = baseline_run.resolve(strict=True)
        self.attempt_limit = 2 if dataset == "mmlifelong-month" else 4
        all_questions = json.loads((self.artifact_root / "questions.json").read_text())
        self.full_denominator = 623 if dataset == "mmlifelong-month" else 500
        if len(all_questions) != self.full_denominator:
            raise ValueError(f"Expected the complete {dataset} question set, found {len(all_questions)}")
        selected = None if ids is None else set(ids)
        self.questions = [q for q in all_questions if selected is None or str(q["id"]) in selected]
        if selected is not None and {str(q["id"]) for q in self.questions} != selected:
            raise ValueError("--ids contains unknown question IDs")
        if not 1 <= concurrency <= 5:
            raise ValueError("concurrency must be between 1 and 5")
        self.denominator = len(self.questions) if ids is not None else self.full_denominator
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.baseline = {}
        for question in self.questions:
            record = json.loads((self.baseline_run / "sql/questions" / str(question["id"]) / "judgment.json").read_text())
            if str(record["id"]) != str(question["id"]):
                raise ValueError(f"Baseline ID mismatch: {record['id']}")
            self.baseline[str(record["id"])] = score_mapping(record["score"])
        config = {
            "dataset": dataset, "artifact_root": str(self.artifact_root), "run_dir": str(self.run_dir), "baseline_run": str(self.baseline_run),
            "debug": ids is not None, "ids": [q["id"] for q in self.questions], "denominator": self.denominator, "formal_denominator": self.full_denominator,
            "model": MODEL, "reasoning_effort": "max", "service_tier": "default", "ephemeral": False,
            "qa_model": QA_MODEL, "qa_reasoning_effort": "xhigh" if dataset == "mmlifelong-month" else None, "qa_url": qa_url,
            "embedding_model": EMBEDDING_MODEL, "embedding_url": embedding_url, "judge_model": "gpt-5", "concurrency": concurrency,
            "attempt_limit": self.attempt_limit, "retry_protocol": "batch-end ordinary failures; maintenance stops and does not consume attempts",
            "context_management": "Codex native", "investigation_call_budget": None, "client_token_limits": None,
            "egolife_qa_change": "removed 16384-token cap and truncation retries", "semantic_search": "vector with SQL filtering",
            "skills": str(SKILLS), "month_skills": str(ROOT / "training_free/skills-month") if dataset == "mmlifelong-month" else None,
            "instructions": instructions_for(dataset), "versions": {name: version(name) for name in ("openai-codex", "openai", "mcp")},
        }
        path = self.run_dir / "config.json"
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError(f"Run configuration differs from {path}; use a separate run directory")
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.answers = set()
        self.scores = {}
        self.failures = set()
        self.records = []
        self.retry_count = 0
        for question in self.questions:
            directory = self.run_dir / "questions" / str(question["id"])
            if (directory / "answer.json").exists():
                self.answers.add(str(question["id"]))
            if (directory / "judgment.json").exists():
                record = json.loads((directory / "judgment.json").read_text())
                self.scores[str(record["id"])] = score_mapping(record["score"])
            if (directory / "failure.json").exists():
                self.failures.add(str(question["id"]))
            ordinary = 0
            for attempt in directory.glob("attempts/*"):
                if (attempt / "usage-summary.json").exists():
                    self.records.extend(json.loads((attempt / "usage-summary.json").read_text()))
                if (attempt / "outcome.json").exists():
                    ordinary += json.loads((attempt / "outcome.json").read_text())["status"] == "ordinary_failure"
            self.retry_count += max(0, ordinary - 1) + int(ordinary > 0 and str(question["id"]) in self.answers)
            for path in directory.glob("judge/*/usage.json"):
                self.records.append(json.loads(path.read_text()))
        load_dotenv(ROOT / ".env")
        self.judge_client = OpenAI(api_key=os.environ["SHARED_OPENAI_KEY"], max_retries=0)
        wandb_path = self.run_dir / "wandb-run.json"
        wandb_id = json.loads(wandb_path.read_text())["id"] if wandb_path.exists() else wandb.util.generate_id()
        self.wandb = wandb.init(project="video-sql-codex", id=wandb_id, name=self.run_dir.parent.name + "-" + self.run_dir.name, dir=str(self.run_dir), config=config, resume="must" if wandb_path.exists() else "never")
        wandb_path.write_text(json.dumps({"id": self.wandb.id, "url": self.wandb.url}) + "\n")

    def emit(self, event, question_id=None):
        with self.lock:
            overlap = sorted(self.scores.keys() & self.baseline.keys())
            final = len(self.scores) + len(self.failures) == self.denominator
            metrics = {"completed_answers": len(self.answers), "scored": len(self.scores), "final_failures": len(self.failures), "retries": self.retry_count, "denominator": self.denominator, "complete": final}
            if final:
                metrics["score_1_ratio"] = sum(score == 1 for score in self.scores.values()) / self.denominator
                metrics["average_rounded_score"] = sum(self.scores.values()) / self.denominator
                metrics["baseline_score_1_ratio"] = sum(score == 1 for score in self.baseline.values()) / self.denominator
                metrics["baseline_average_rounded_score"] = sum(self.baseline.values()) / self.denominator
                metrics["delta_score_1_ratio"] = metrics["score_1_ratio"] - metrics["baseline_score_1_ratio"]
                metrics["delta_average_rounded_score"] = metrics["average_rounded_score"] - metrics["baseline_average_rounded_score"]
            metrics["overlap_count"] = len(overlap)
            if overlap:
                metrics["overlap_score_1_ratio"] = sum(self.scores[key] == 1 for key in overlap) / len(overlap)
                metrics["overlap_average_rounded_score"] = sum(self.scores[key] for key in overlap) / len(overlap)
                metrics["overlap_delta_score_1_ratio"] = sum(int(self.scores[key] == 1) - int(self.baseline[key] == 1) for key in overlap) / len(overlap)
                metrics["overlap_delta_average_rounded_score"] = sum(self.scores[key] - self.baseline[key] for key in overlap) / len(overlap)
            tokens = {}
            for role in ("terra", "sql-qa", "semantic-qa", "clip-qa", "judge"):
                records = [r for r in self.records if r["role"] == role]
                tokens[role] = {"requests": len(records), "unknown_usage": sum(r["usage"] is None for r in records)}
                for key in ("input_tokens", "output_tokens", "cached_tokens", "total_tokens"):
                    tokens[role][key] = None if any(r["usage"] is None or r["usage"][key] is None for r in records) else sum(r["usage"][key] for r in records)
            record = {"event": event, "question_id": question_id, "metrics": metrics, "tokens": tokens}
            with (self.run_dir / "events.jsonl").open("a") as output:
                output.write(json.dumps(record) + "\n")
            (self.run_dir / "summary.json").write_text(json.dumps({**record, "failed_ids": sorted(self.failures), "wandb_url": self.wandb.url}, indent=2) + "\n")
            payload = dict(metrics)
            payload.update({f"tokens/{role}/{key}": value for role, counts in tokens.items() for key, value in counts.items() if value is not None})
            self.wandb.log(payload)
            print(f"EVENT {event} id={question_id} answers={len(self.answers)} scored={len(self.scores)} failed={len(self.failures)}", flush=True)

    def judge(self, question, directory):
        answer = json.loads((directory / "answer.json").read_text())["candidate_answer"]
        messages = judge_messages(self.dataset, question, answer)
        request_attempt = len(list((directory / "judge").glob("*")))
        while not self.stop.is_set():
            request_attempt += 1
            trace_dir = directory / "judge" / f"{request_attempt:04d}-{uuid.uuid4().hex}"
            trace_dir.mkdir(parents=True)
            timeout = 60 * request_attempt
            record = {"request": {"model": "gpt-5", "messages": messages, "timeout": timeout}, "response": None, "error": None}
            path = trace_dir / "trace.json"
            path.write_text(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"JUDGE_START id={question['id']} attempt={request_attempt} timeout={timeout}", flush=True)
            usage_record = {"role": "judge", "usage": None}
            try:
                raw = self.judge_client.chat.completions.with_raw_response.create(model="gpt-5", messages=messages, timeout=timeout)
                record["request"] = json.loads(raw.http_request.content)
                record["response"] = raw.http_response.json()
                usage = record["response"]["usage"]
                if usage is not None:
                    details = usage.get("prompt_tokens_details")
                    usage_record["usage"] = {"input_tokens": usage["prompt_tokens"], "output_tokens": usage["completion_tokens"], "total_tokens": usage["total_tokens"], "cached_tokens": None if details is None else details["cached_tokens"]}
                content = raw.parse().choices[0].message.content
                score = -1 if content is None else parse_score(content)
                if score < 0:
                    raise ValueError(f"Invalid official judge output: {content}")
            except (APIConnectionError, APIStatusError, ValueError) as error:
                record["error"] = {"type": type(error).__name__, "message": str(error)}
                print(f"JUDGE_ERROR id={question['id']} error={error} trace={path}", flush=True)
                if isinstance(error, APIStatusError) and error.status_code not in (408, 429) and error.status_code < 500:
                    raise
                retry = True
            else:
                retry = False
            finally:
                path.write_text(json.dumps(record, ensure_ascii=False) + "\n")
                (trace_dir / "usage.json").write_text(json.dumps(usage_record) + "\n")
                with self.lock:
                    self.records.append(usage_record)
            if retry:
                self.emit("judge_retry", question["id"])
                continue
            result = {"id": question["id"], "score": score, "mapped_score": score_mapping(score), "justification": content, "trace": str(path), "model": record["response"]["model"]}
            (directory / "judgment.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            with self.lock:
                self.scores[str(question["id"])] = result["mapped_score"]
            self.emit("scored", question["id"])
            return

    def process(self, question):
        if self.stop.is_set():
            return
        question_id = str(question["id"])
        directory = self.run_dir / "questions" / question_id
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "judgment.json").exists() or (directory / "failure.json").exists():
            return
        for attempt in directory.glob("attempts/*"):
            if (attempt / "thread.json").exists() and not (attempt / "archive.json").exists():
                thread_id = json.loads((attempt / "thread.json").read_text())["id"]
                request = json.loads((attempt / "request.json").read_text())
                with Codex(CodexConfig(cwd=str(attempt / "workspace"), config_overrides=tuple(request["overrides"]))) as codex:
                    archived = codex.thread_archive(thread_id)
                (attempt / "archive.json").write_text(json.dumps({"thread_id": thread_id, "response": trace_json(archived)}) + "\n")
            if (attempt / "submitted-answer.json").exists() and not (directory / "answer.json").exists():
                submitted = json.loads((attempt / "submitted-answer.json").read_text())
                usage = json.loads((attempt / "usage.json").read_text()) if (attempt / "usage.json").exists() else None
                answer = {"id": question["id"], "candidate_answer": submitted["answer"], "thread_id": submitted["thread_id"], "usage": usage, "attempt_dir": str(attempt)}
                (directory / "answer.json").write_text(json.dumps(answer, ensure_ascii=False, indent=2) + "\n")
                (attempt / "outcome.json").write_text(json.dumps({"status": "answered"}) + "\n")
                with self.lock:
                    self.answers.add(question_id)
                self.emit("answer_recovered", question["id"])
        if not (directory / "answer.json").exists():
            outcomes = [json.loads(path.read_text()) for path in directory.glob("attempts/*/outcome.json")]
            failures = sum(r["status"] == "ordinary_failure" for r in outcomes)
            if failures >= self.attempt_limit:
                (directory / "failure.json").write_text(json.dumps({"id": question["id"], "ordinary_failures": failures}) + "\n")
                with self.lock:
                    self.failures.add(question_id)
                self.emit("final_failure", question["id"])
                return
            attempt = directory / "attempts" / uuid.uuid4().hex
            attempt.mkdir(parents=True)
            if failures:
                with self.lock:
                    self.retry_count += 1
                self.emit("answer_retry", question["id"])
            self.emit("answer_started", question["id"])
            try:
                answer = answer_question(self.dataset, self.artifact_root, attempt, {key: question[key] for key in ("id", "question", "query_time")}, self.qa_url, self.embedding_url, self.stop)
            except AttemptFailed as error:
                (attempt / "outcome.json").write_text(json.dumps({"status": "ordinary_failure", "error": str(error)}) + "\n")
                failed = True
            except BaseException:
                self.stop.set()
                (attempt / "outcome.json").write_text(json.dumps({"status": "maintenance", "error": traceback.format_exc()}) + "\n")
                raise
            else:
                (directory / "answer.json").write_text(json.dumps(answer, ensure_ascii=False, indent=2) + "\n")
                (attempt / "outcome.json").write_text(json.dumps({"status": "answered"}) + "\n")
                with self.lock:
                    self.answers.add(question_id)
                failed = False
            finally:
                records = usage_records(attempt)
                (attempt / "usage-summary.json").write_text(json.dumps(records) + "\n")
                with self.lock:
                    self.records.extend(records)
            self.emit("answer_failed" if failed else "answered", question["id"])
            if failed:
                if failures + 1 == self.attempt_limit:
                    (directory / "failure.json").write_text(json.dumps({"id": question["id"], "ordinary_failures": failures + 1}) + "\n")
                    with self.lock:
                        self.failures.add(question_id)
                    self.emit("final_failure", question["id"])
                return
        self.judge(question, directory)

    def run(self):
        try:
            self.emit("resumed")
            while len(self.scores) + len(self.failures) < self.denominator:
                pending = [q for q in self.questions if str(q["id"]) not in self.scores and str(q["id"]) not in self.failures]
                with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                    futures = [pool.submit(self.process, question) for question in pending]
                    try:
                        for future in as_completed(futures):
                            future.result()
                    except BaseException:
                        self.stop.set()
                        for future in futures:
                            future.cancel()
                        raise
            self.emit("complete")
        finally:
            self.judge_client.close()
            self.wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mmlifelong-month", "egolife-open"), required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--qa-url", required=True)
    parser.add_argument("--embedding-url", required=True)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--ids", nargs="+")
    args = parser.parse_args()
    Evaluation(args.dataset, args.artifact_root, args.run_dir, args.qa_url, args.embedding_url, args.concurrency, args.baseline_run, args.ids).run()
