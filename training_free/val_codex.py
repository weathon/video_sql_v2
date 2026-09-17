"""Run Month, then EgoLife Open, with Codex and the official GPT-5 judge."""

import argparse
import json
import os
import subprocess
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import wandb
from dotenv import load_dotenv
from openai import OpenAI
from openai_codex import Codex, CodexConfig
from wandb.sdk.lib.runid import generate_id

from training_free.codex_harness import DATASETS, MODEL, QA_MODEL, RETRYABLE, ROOT, answer_question
from training_free.evaluate_mm_official import SYSTEM_PROMPT, parse_score


def judge_answer(client, dataset, question, answer, directory):
    rubric = SYSTEM_PROMPT.split("\nEgoLife metadata rules:", 1)[0] if dataset == "month" else SYSTEM_PROMPT
    prompt = f"Question: {question['question']}\n"
    if dataset == "egolife":
        query_time = str(question["query_time"]).zfill(9)
        metadata = {"questioner": "A1_JAKE", "questioner_name": "Jake", "perspective": "first-person narrator and camera wearer", "query_time_raw_DHHMMSSFF": question["query_time"], "query_day": int(query_time[0]), "query_time_of_day": f"{query_time[1:3]}:{query_time[3:5]}:{query_time[5:7]}", "query_frame_index": int(query_time[7:9])}
        prompt += f"Metadata: {json.dumps(metadata, ensure_ascii=False)}\n"
    prompt += f"Groundtruth answer: {question['reference_answer']}\nCandidate answer: {answer['candidate_answer']}\nYour response: "
    messages = [{"role": "system", "content": rubric}, {"role": "user", "content": prompt}]
    attempt = len(list(directory.glob("judge-*.json")))
    while True:
        attempt += 1
        path = directory / f"judge-{attempt}.json"
        request = {"model": "gpt-5", "messages": messages, "timeout": 60 * attempt}
        trace = {"request": request, "response": None, "error": None}
        path.write_text(json.dumps(trace, ensure_ascii=False) + "\n")
        print(f"JUDGE_START dataset={dataset} id={question['id']} attempt={attempt} trace={path}", flush=True)
        try:
            raw = client.chat.completions.with_raw_response.create(**request)
            trace["request"] = json.loads(raw.http_request.content)
            trace["response"] = raw.http_response.json()
            content = raw.parse().choices[0].message.content.strip()
            score = parse_score(content)
            if score < 0:
                raise ValueError(f"Invalid official judge output: {content}")
            judgment = {"analysis": content, "score": score}
            judgment["trace_file"] = str(path)
            (directory / "judgment.json").write_text(json.dumps(judgment, ensure_ascii=False) + "\n")
            return judgment
        except Exception as error:
            trace["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
            print(f"JUDGE_ERROR dataset={dataset} id={question['id']} error={error} trace={path}", flush=True)
        finally:
            path.write_text(json.dumps(trace, ensure_ascii=False) + "\n")


def process_question(codex, judge_client, dataset, question, attempt, directory, qa_url, embedding_url):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "question.json").write_text(json.dumps(question, ensure_ascii=False) + "\n")
    result_path = directory / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    answer_path = directory / "answer.json"
    attempt_path = directory / f"answer-attempt-{attempt}.json"
    if not answer_path.exists():
        if attempt_path.exists():
            saved = json.loads(attempt_path.read_text())
            if saved["status"] == "retry":
                return saved
        trace_dir = directory / f"answer-attempt-{attempt}" / uuid.uuid4().hex
        started = time.monotonic()
        try:
            answer = answer_question(codex, dataset, question, trace_dir, qa_url, embedding_url)
            answer.update({"attempt": attempt, "latency_seconds": time.monotonic() - started})
            answer_path.write_text(json.dumps(answer, ensure_ascii=False) + "\n")
        except BaseException as error:
            record = {"id": question["id"], "attempt": attempt, "status": "retry" if isinstance(error, RETRYABLE) else "error", "trace_dir": str(trace_dir), "error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}}
            attempt_path.write_text(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"ANSWER_ERROR dataset={dataset} id={question['id']} attempt={attempt} error={error}", flush=True)
            if not isinstance(error, RETRYABLE):
                raise
            if attempt == DATASETS[dataset]["attempts"]:
                record.update({"status": "failed", "score": 0, "rounded_score": 0})
                result_path.write_text(json.dumps(record, ensure_ascii=False) + "\n")
            return record
    answer = json.loads(answer_path.read_text())
    judgment_path = directory / "judgment.json"
    judgment = json.loads(judgment_path.read_text()) if judgment_path.exists() else judge_answer(judge_client, dataset, question, answer, directory)
    score = judgment["score"]
    result = {**answer, "status": "answered", "score": score, "rounded_score": 1 if score >= 4 else 0.5 if score == 3 else 0, "judgment": judgment}
    result_path.write_text(json.dumps(result, ensure_ascii=False) + "\n")
    return result


def summarize(dataset, results, baseline, selected):
    denominator = DATASETS[dataset]["denominator"]
    scored = [result for result in results.values() if result["status"] in ("answered", "failed")]
    full_credit = sum(result["rounded_score"] == 1 for result in scored)
    rounded_total = sum(result["rounded_score"] for result in scored)
    baseline_ratio = sum(result["score"] >= 4 for result in baseline) / denominator
    baseline_average = sum(1 if result["score"] >= 4 else 0.5 if result["score"] == 3 else 0 for result in baseline) / denominator
    return {
        "dataset": dataset, "selected": selected, "completed": len(scored), "failed": sum(result["status"] == "failed" for result in scored), "denominator": denominator,
        "score_1_count": full_credit, "rounded_score_sum": rounded_total,
        "score_1_ratio": full_credit / denominator, "average_rounded_score": rounded_total / denominator,
        "baseline_score_1_ratio": baseline_ratio, "baseline_average_rounded_score": baseline_average,
        "delta_score_1_ratio": full_credit / denominator - baseline_ratio, "delta_average_rounded_score": rounded_total / denominator - baseline_average,
    }


def main(args):
    load_dotenv(ROOT / ".env")
    run_root = args.run_dir.resolve()
    if run_root.is_relative_to(ROOT):
        subprocess.run(["git", "check-ignore", "--quiet", str(run_root.relative_to(ROOT)) + "/"], cwd=ROOT, check=True)
    run_root.mkdir(parents=True, exist_ok=True)
    if args.ids is not None and args.dataset == "both":
        raise ValueError("--ids requires --dataset month or egolife")
    with Codex(CodexConfig(cwd=str(ROOT))) as codex, OpenAI(api_key=os.environ["SHARED_OPENAI_KEY"], max_retries=0, timeout=None) as judge_client:
        for dataset in (("month", "egolife") if args.dataset == "both" else (args.dataset,)):
            settings = DATASETS[dataset]
            questions = json.loads((settings["artifacts"] / "questions.json").read_text())
            baseline = [json.loads((settings["baseline"] / "sql/questions" / str(question["id"]) / "result.json").read_text()) for question in questions]
            if args.ids is not None:
                by_id = {str(question["id"]): question for question in questions}
                questions = [by_id[qid] for qid in args.ids]
            directory = run_root / dataset
            directory.mkdir(exist_ok=True)
            config = {"dataset": dataset, "model": MODEL, "reasoning_effort": "max", "service_tier": "default", "qa_model": QA_MODEL, "qa_reasoning_effort": settings["reasoning_effort"], "qa_url": args.qa_url, "embedding_url": args.embedding_url, "answer_attempts": settings["attempts"], "judge_model": "gpt-5", "judge_attempts": "unlimited", "concurrency": 2, "denominator": settings["denominator"], "ids": [question["id"] for question in questions], "baseline": str(settings["baseline"])}
            config_path = directory / "config.json"
            if config_path.exists():
                saved = json.loads(config_path.read_text())
                if saved["evaluation"] != config:
                    raise ValueError(f"Saved configuration differs: {config_path}")
                wandb_id, resume = saved["wandb_id"], "must"
            else:
                wandb_id, resume = generate_id(), "never"
                config_path.write_text(json.dumps({"evaluation": config, "wandb_id": wandb_id}, ensure_ascii=False, indent=2) + "\n")
            run = wandb.init(entity="3dsmile", project="video-sql-v2", name=f"{run_root.name}-{dataset}", id=wandb_id, resume=resume, config=config, dir=str(directory))
            results = {}
            try:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    for attempt in range(1, settings["attempts"] + 1):
                        pending = [question for question in questions if str(question["id"]) not in results or results[str(question["id"])]["status"] == "retry"]
                        futures = {pool.submit(process_question, codex, judge_client, dataset, question, attempt, directory / "questions" / str(question["id"]), args.qa_url, args.embedding_url): question for question in pending}
                        try:
                            for future in as_completed(futures):
                                question = futures[future]
                                result = future.result()
                                results[str(question["id"])] = result
                                summary = summarize(dataset, results, baseline, len(questions))
                                (directory / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
                                run.log({key: value for key, value in summary.items() if key != "dataset"})
                                print(f"PROGRESS dataset={dataset} attempt={attempt} id={question['id']} status={result['status']} completed={summary['completed']}/{len(questions)}", flush=True)
                        except BaseException:
                            for future in futures:
                                future.cancel()
                            raise
                run.summary.update(summary)
            except BaseException:
                run.finish(exit_code=1)
                raise
            else:
                run.finish()
            print(f"DATASET_DONE dataset={dataset} metrics={directory / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("both", "month", "egolife"), default="both")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--qa-url", required=True)
    parser.add_argument("--embedding-url", required=True)
    parser.add_argument("--ids", nargs="+")
    main(parser.parse_args())
