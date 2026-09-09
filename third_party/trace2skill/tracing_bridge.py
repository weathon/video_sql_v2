"""Keep raw structured-parse Responses visible in the Weave API call spans."""

import json
import threading
import time
from collections import deque

import weave
from agents import set_trace_processors
from openai._legacy_response import LegacyAPIResponse
from weave.integrations.openai import openai_sdk
from weave.integrations.openai_agents.openai_agents import WeaveTracingProcessor


RESPONSES_OUTPUT_PROCESSOR = openai_sdk.responses_on_finish_post_processor


def response_cost(response):
    usage = response["usage"]
    if usage is None:
        raise RuntimeError(f"Cannot report complete rollout cost: response {response['response_id']} has no usage")
    inputs = usage["input_tokens"]
    cached = usage["input_tokens_details"]["cached_tokens"]
    written = usage["input_tokens_details"]["cache_write_tokens"]
    ordinary = inputs - cached - written
    if ordinary < 0:
        raise ValueError(f"Invalid input token breakdown: {usage}")
    tier = {"default": 1, "priority": 2, "fast": 2, "flex": 0.5}[response["service_tier"]]
    input_rate = 0.40 if inputs > 272000 else 0.20
    output_rate = 1.80 if inputs > 272000 else 1.20
    return tier * (input_rate * (ordinary + 0.1 * cached + 1.25 * written)
                   + output_rate * usage["output_tokens"]) / 1_000_000


class RunMetrics:
    def __init__(self, run, phase, directory):
        self.run = run
        self.phase = phase
        self.path = directory / f"{phase}-trace-metrics.json"
        self.started = time.monotonic()
        self.responses = deque()
        self.results = []
        self.lock = threading.Lock()
        self.prior_seconds = 0
        self.total_cost = 0
        if self.path.exists():
            saved = json.loads(self.path.read_text())
            self.prior_seconds = saved[f"{phase}/active_elapsed_seconds"]
            self.total_cost = saved[f"{phase}/total_cost_usd"]
        self.values = {}

    def record_response(self, response):
        with self.lock:
            now = time.monotonic()
            usage = response["usage"]
            self.responses.append((now, usage["input_tokens"], usage["output_tokens"]))
            while self.responses[0][0] < now - 60:
                self.responses.popleft()
            elapsed = min(now - self.started, 60)
            self.total_cost += response_cost(response)
            self.values.update({
                f"{self.phase}/throughput/input_tokens_per_s": sum(r[1] for r in self.responses) / elapsed,
                f"{self.phase}/throughput/output_tokens_per_s": sum(r[2] for r in self.responses) / elapsed,
                f"{self.phase}/throughput/responses_per_s": len(self.responses) / elapsed,
                f"{self.phase}/request_context_tokens": usage["input_tokens"] + usage["output_tokens"],
                f"{self.phase}/active_elapsed_seconds": self.prior_seconds + now - self.started,
                f"{self.phase}/total_cost_usd": self.total_cost,
            })
            self.run.log(self.values)
            self.path.write_text(json.dumps(self.values, indent=2) + "\n")

    def record_rollout(self, result):
        self.results.append(result)
        count = len(self.results)
        elapsed = self.prior_seconds + time.monotonic() - self.started
        self.values.update({
            f"{self.phase}/completed_rollouts": count,
            f"{self.phase}/active_elapsed_seconds": elapsed,
            f"{self.phase}/total_cost_usd": self.total_cost,
            f"{self.phase}/rollouts_per_minute": 60 * count / elapsed,
            f"{self.phase}/mean_score": sum(r["score"] for r in self.results) / count,
            f"{self.phase}/score_ge_4_fraction": sum(r["score"] >= 4 for r in self.results) / count,
            f"{self.phase}/cost_per_rollout_usd": sum(r["cost_usd"] for r in self.results) / count,
            f"{self.phase}/average_rollout_time_seconds": sum(r["rollout_time_seconds"] for r in self.results) / count,
            f"{self.phase}/mean_total_context_saved_tokens": sum(r["replacement_metrics"]["total_context_saved_tokens"] for r in self.results) / count,
            f"{self.phase}/replace_tool_calls": sum(r["replacement_metrics"]["replace_tool_calls"] for r in self.results),
        })
        for key in result["trace_length"]:
            self.values[f"{self.phase}/trace/mean_{key}"] = sum(r["trace_length"][key] for r in self.results) / count
        self.run.log(self.values)
        self.path.write_text(json.dumps(self.values, indent=2) + "\n")


def response_output(value):
    # Save raw output without triggering schema validation inside Weave. The caller
    # retains the actual response and supplies validation errors to the model.
    if isinstance(value, LegacyAPIResponse):
        return json.loads(value.content)
    return RESPONSES_OUTPUT_PROCESSOR(value)


def init_weave():
    openai_sdk.responses_on_finish_post_processor = response_output
    client = weave.init("3dsmile/video-sql-v2")
    set_trace_processors([WeaveTracingProcessor()])
    return client
