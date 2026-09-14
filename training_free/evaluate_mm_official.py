"""Run the official MM-Lifelong GPT judge with EgoLife metadata."""

import argparse
import json
import os
import re
import subprocess
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import wandb
from dotenv import load_dotenv
from openai import OpenAI
from wandb.sdk.lib.runid import generate_id


MODEL_NAME = "gpt-5"
MAX_WORKERS = 128
RETRY_TIMES = 3
TIMEOUT_SEC = 60

SYSTEM_PROMPT = """
As an AI assistant, your task is to evaluate a candidate answer in comparison to a given correct answer.
The question itself, the correct “groundtruth” answer, and the candidate answer will be provided to you.
The following is a comparison table of some proper nouns; matching any one of them is considered correct.

You must FIRST provide a brief analysis explaining the semantic similarity between the groundtruth
and the candidate answer.

THEN, on a new line, output the final score.

Scoring criteria (semantic similarity only):

- 0: No similarity.
  The candidate answer is completely irrelevant, contradictory, or does not address the question at all.

- 1: Very low similarity.
  The candidate answer mentions a related topic or keyword, but fails to answer the question
  and does not convey the main meaning of the groundtruth.

- 2: Low similarity.
  The candidate answer addresses the question in a limited way, capturing some minor aspects,
  but misses or misrepresents the core idea or key facts of the groundtruth.

- 3: Moderate similarity.
  The candidate answer captures the main idea of the groundtruth,
  but omits several important details or includes noticeable inaccuracies.

- 4: High similarity.
  The candidate answer correctly captures the main idea and most key details of the groundtruth,
  with only minor omissions, simplifications, or non-critical inaccuracies.

- 5: Complete similarity.
  The candidate answer is semantically equivalent to the groundtruth,
  covering all essential information with no meaningful omissions or errors.

Special Rules:

- Hallucination-sensitive questions:
Score 5 only if all required items are correct;
if any item is incorrect, missing, or hallucinated, score 0 (no partial credit).

- Time-duration questions:
Allow errors within the range defined by the question; answers outside the range should receive score 0.

Output format (strictly follow):
Analysis:
<your analysis>

Final Score:
<an integer from 0 to 5>
""" + """

EgoLife metadata rules:
The supplied metadata identifies the questioner. I, me, my, mine, the narrator, and the camera wearer refer to Jake / A1_JAKE.
Use the supplied query day and clock time to interpret relative expressions such as last night, yesterday, or three hours ago.
A DAYn value is a day label and must not be interpreted as n days ago.
"""


def build_prompt(item):
    return (
        "Question: {}\n"
        "Metadata: {}\n"
        "Groundtruth answer: {}\n"
        "Candidate answer: {}\n"
        "Your response: "
    ).format(item["question"], json.dumps(item["metadata"], ensure_ascii=False), item["answer"], item["pred"]["answer"])


def parse_score(text):
    match = re.search(r"Final Score:\s*([0-5])", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"\b([0-5])\b", text)
    if match:
        return int(match.group(1))
    return -1


def score_mapping(score):
    if score in (4, 5):
        return 1.0
    if score == 3:
        return 0.5
    return 0


def score_one(item, position, client, run_dir, state, lock):
    prompt = build_prompt(item)
    question_dir = run_dir / "questions" / str(item["index"])
    question_dir.mkdir(parents=True, exist_ok=True)
    with lock:
        state["started"] += 1
        state["active"] += 1
    for attempt in range(RETRY_TIMES):
        attempt_dir = question_dir / f"attempt-{attempt + 1}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        trace_path = attempt_dir / "trace.json"
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        trace = {"request": {"model": MODEL_NAME, "messages": messages, "temperature": 0.0, "timeout": TIMEOUT_SEC}, "response": None, "error": None}
        trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n")
        print(f"JUDGE request method={run_dir.name} id={item['index']} attempt={attempt + 1} trace={trace_path}", flush=True)
        try:
            response = client.chat.completions.create(model=MODEL_NAME, messages=messages, temperature=0.0, timeout=TIMEOUT_SEC)
            trace["response"] = response.model_dump(mode="json")
            trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n")
            usage = trace["response"]["usage"]
            if usage is None or usage["prompt_tokens"] is None or usage["completion_tokens"] is None or usage["prompt_tokens_details"] is None or usage["prompt_tokens_details"]["cached_tokens"] is None:
                with lock:
                    state["responses_without_usage"] += 1
            else:
                with lock:
                    state["input_tokens"] += usage["prompt_tokens"]
                    state["cached_tokens"] += usage["prompt_tokens_details"]["cached_tokens"]
                    state["output_tokens"] += usage["completion_tokens"]
            text = response.choices[0].message.content.strip()
            score = parse_score(text)
            if 0 <= score <= 5:
                result = {**item, "gpt_score": score, "gpt_raw": text, "attempt": attempt + 1, "trace_file": str(trace_path)}
                (question_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
                with lock:
                    state["active"] -= 1
                    state["completed"] += 1
                print(f"JUDGE id={item['index']} attempt={attempt + 1} score={score}", flush=True)
                return position, result
            raise ValueError(f"Invalid score output: {text}")
        except Exception as error:
            traceback.print_exc()
            trace["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
            trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n")
            if trace["response"] is None:
                with lock:
                    state["requests_without_response"] += 1
            if attempt == RETRY_TIMES - 1:
                result = {**item, "gpt_score": -1, "gpt_error": str(error), "attempt": attempt + 1, "trace_file": str(trace_path)}
                (question_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
                with lock:
                    state["active"] -= 1
                    state["failed"] += 1
                return position, result
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("Official retry loop ended without a result")


def monitor(run, run_dir, state, lock, stop):
    previous = None
    deadline = time.monotonic()
    stopped = False
    with (run_dir / "live-metrics.jsonl").open("a", buffering=1) as history:
        while True:
            timestamp, monotonic = time.time(), time.monotonic()
            with lock:
                current = state.copy()
                payload = {"monitor/timestamp": timestamp, "progress/started": current["started"], "progress/active": current["active"], "progress/completed": current["completed"], "progress/failed": current["failed"], "tokens/in": current["input_tokens"] - current["cached_tokens"], "tokens/cache": current["cached_tokens"], "tokens/out": current["output_tokens"], "usage/requests_without_response": current["requests_without_response"], "usage/responses_without_usage": current["responses_without_usage"], "usage/complete": int(current["requests_without_response"] == 0 and current["responses_without_usage"] == 0)}
                if previous is not None:
                    interval = monotonic - previous["monotonic"]
                    payload.update({"throughput/in_tokens_per_second": ((current["input_tokens"] - current["cached_tokens"]) - (previous["input_tokens"] - previous["cached_tokens"])) / interval, "throughput/cache_tokens_per_second": (current["cached_tokens"] - previous["cached_tokens"]) / interval, "throughput/out_tokens_per_second": (current["output_tokens"] - previous["output_tokens"]) / interval, "monitor/interval_seconds": interval})
                history.write(json.dumps({"metrics": payload, "provider_usage": current}) + "\n")
                run.log(payload)
            previous = {**current, "monotonic": monotonic}
            if stopped:
                return
            deadline += 1
            stopped = stop.wait(max(0, deadline - time.monotonic()))


def main(args):
    repository = Path(__file__).resolve().parents[1]
    load_dotenv(repository / ".env")
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    items = [json.loads(line) for line in args.input_jsonl.read_text().splitlines() if line.strip()]
    config = {"model": MODEL_NAME, "max_workers": MAX_WORKERS, "retry_times": RETRY_TIMES, "timeout_seconds": TIMEOUT_SEC, "temperature": 0.0, "input_jsonl": str(args.input_jsonl.resolve()), "method": args.method, "metadata": "EgoLife questioner and query time", "prompt_source": "MM-Lifelong eval_acc.py plus EgoLife metadata rules", "live_metrics_interval_seconds": 1}
    config_path = run_dir / "config.json"
    if config_path.exists():
        saved = json.loads(config_path.read_text())
        if saved["evaluation"] != config:
            raise ValueError(f"Saved evaluation configuration differs: {config_path}")
        wandb_id, resume = saved["wandb_id"], "must"
    else:
        wandb_id, resume = generate_id(), "never"
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository, check=True, capture_output=True, text=True).stdout.strip()
        config_path.write_text(json.dumps({"evaluation": config, "wandb_id": wandb_id, "git_commit": commit}, ensure_ascii=False, indent=2) + "\n")
    run = wandb.init(entity="3dsmile", project="video-sql-v2", id=wandb_id, name=run_dir.name, config=config, resume=resume, dir=str(run_dir), settings=wandb.Settings(x_stats_sampling_interval=1.0))
    client = OpenAI(api_key=os.environ["SHARED_OPENAI_KEY"])
    lock = threading.Lock()
    state = {"started": 0, "active": 0, "completed": 0, "failed": 0, "input_tokens": 0, "cached_tokens": 0, "output_tokens": 0, "requests_without_response": 0, "responses_without_usage": 0}
    results = [None] * len(items)
    for position, item in enumerate(items):
        result_path = run_dir / "questions" / str(item["index"]) / "result.json"
        if result_path.exists():
            results[position] = json.loads(result_path.read_text())
            state["completed" if results[position]["gpt_score"] >= 0 else "failed"] += 1
    response_ids = set()
    for trace_path in run_dir.glob("questions/*/attempt-*/trace.json"):
        trace = json.loads(trace_path.read_text())
        if trace["response"] is None:
            if trace["error"] is not None:
                state["requests_without_response"] += 1
            continue
        response = trace["response"]
        if response["id"] in response_ids:
            continue
        response_ids.add(response["id"])
        usage = response["usage"]
        if usage is None or usage["prompt_tokens"] is None or usage["completion_tokens"] is None or usage["prompt_tokens_details"] is None or usage["prompt_tokens_details"]["cached_tokens"] is None:
            state["responses_without_usage"] += 1
        else:
            state["input_tokens"] += usage["prompt_tokens"]
            state["cached_tokens"] += usage["prompt_tokens_details"]["cached_tokens"]
            state["output_tokens"] += usage["completion_tokens"]
    stop = threading.Event()
    monitor_thread = threading.Thread(target=monitor, args=(run, run_dir, state, lock, stop))
    monitor_thread.start()
    try:
        pending = [(position, item) for position, item in enumerate(items) if results[position] is None]
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(score_one, item, position, client, run_dir, state, lock) for position, item in pending]
            for future in as_completed(futures):
                position, result = future.result()
                results[position] = result
    finally:
        stop.set()
        monitor_thread.join()
        client.close()
    if any(result is None for result in results):
        raise RuntimeError("Official judge did not produce every result")
    (run_dir / "eval_thinking.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    valid = [result for result in results if isinstance(result["gpt_score"], int) and result["gpt_score"] >= 0]
    mapped = [score_mapping(result["gpt_score"]) for result in valid]
    average = sum(mapped) / len(mapped) if mapped else None
    distribution = Counter(mapped)
    metrics = {"valid_samples": len(valid), "average_score": average, "one_score_count": distribution[1.0], "one_score_fraction": distribution[1.0] / len(valid) if valid else None, "half_score_count": distribution[0.5], "zero_score_count": distribution[0], "raw_score_distribution": {str(score): sum(result["gpt_score"] == score for result in valid) for score in range(6)}, "judge_failures": len(results) - len(valid)}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    with lock:
        token_log = {"authoritative_source": "OpenAI response usage", "complete": state["requests_without_response"] == 0 and state["responses_without_usage"] == 0, "input_tokens_include_cached": True, "input_tokens": state["input_tokens"], "cached_tokens": state["cached_tokens"], "output_tokens": state["output_tokens"], "requests_without_response": state["requests_without_response"], "responses_without_usage": state["responses_without_usage"]}
        (run_dir / "token-log.json").write_text(json.dumps(token_log, ensure_ascii=False, indent=2) + "\n")
        run.log({f"{args.method}/{key}": value for key, value in metrics.items() if isinstance(value, (int, float))})
    print("===== GPT Scoring Statistics =====")
    print(f"Valid samples: {len(valid)}")
    print(f"Average score: {average:.4f}")
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--method", choices=("magic", "sql"), required=True)
    main(parser.parse_args())
