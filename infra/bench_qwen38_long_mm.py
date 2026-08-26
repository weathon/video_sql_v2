import asyncio
import json
import os
import time
from pathlib import Path

import httpx
import wandb


def parse_metrics(body):
    names = (
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:kv_cache_usage_perc",
        "vllm:time_to_first_token_seconds_sum",
        "vllm:time_to_first_token_seconds_count",
        "vllm:request_prefill_time_seconds_sum",
        "vllm:request_prefill_time_seconds_count",
        "vllm:request_decode_time_seconds_sum",
        "vllm:request_decode_time_seconds_count",
        "vllm:request_time_per_output_token_seconds_sum",
        "vllm:request_time_per_output_token_seconds_count",
    )
    values = {}
    selected_lines = []
    for line in body.splitlines():
        if line.startswith("#") or not line:
            continue
        metric_name = line.split("{", 1)[0].split(" ", 1)[0]
        if metric_name in names or "spec_decode" in metric_name:
            value = float(line.rsplit(" ", 1)[1])
            if metric_name in values:
                values[metric_name] += value
            else:
                values[metric_name] = value
            selected_lines.append(line)
    return values, selected_lines


async def main():
    root = Path(__file__).resolve().parents[1] / "runs" / "qwen38_bench"
    run_label = os.environ["RUN_LABEL"]
    run_dir = root / "results" / run_label
    run_dir.mkdir(parents=True)
    request_data = json.loads((root / "request.json").read_text())
    done = asyncio.Event()
    state = {
        "last_progress": time.monotonic(),
        "prompt_tokens": 0.0,
        "generation_tokens": 0.0,
        "first_token_time": None,
        "last_token_time": None,
        "usage": None,
        "finish_reason": None,
    }

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(None, connect=60.0),
    ) as client:
        models_response = await client.get("http://127.0.0.1:8000/v1/models")
        models_response.raise_for_status()
        model = models_response.json()["data"][0]["id"]
        payload = {
            "model": model,
            "messages": request_data["messages"],
            "max_tokens": 32144,
            "temperature": 1.0,
            "top_p": 0.95,
            "stream": True,
            "stream_options": {"include_usage": True},
            "ignore_eos": True,
            "top_k": 20,
        }
        wandb_run = wandb.init(
            project="video-sql-v2",
            name=f"qwen38-long-mm-{run_label}",
            dir=root,
            config={
                "run_label": run_label,
                "model": model,
                "image_count": 1024,
                "prompt_tokens": 230000,
                "visual_tokens": 115000,
                "text_and_template_tokens": 115000,
                "max_tokens": 32144,
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 20,
                "ignore_eos": True,
                "tensor_parallel_size": 2,
                "gpu_indices": [5, 6],
                "max_model_len": 262144,
                "max_num_batched_tokens": 32768,
            },
        )
        (run_dir / "request_summary.json").write_text(
            json.dumps(
                {
                    "run_label": run_label,
                    "model": model,
                    "image_count": 1024,
                    "prompt_tokens": 230000,
                    "visual_tokens": 115000,
                    "text_and_template_tokens": 115000,
                    "max_tokens": 32144,
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "top_k": 20,
                    "ignore_eos": True,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        metrics_response = await client.get("http://127.0.0.1:8000/metrics")
        metrics_response.raise_for_status()
        metrics_before, selected_lines = parse_metrics(metrics_response.text)
        (run_dir / "metrics_before.txt").write_text("\n".join(selected_lines) + "\n")
        state["prompt_tokens"] = metrics_before["vllm:prompt_tokens_total"]
        state["generation_tokens"] = metrics_before["vllm:generation_tokens_total"]
        request_start = time.monotonic()
        state["last_progress"] = request_start
        print(
            json.dumps(
                {
                    "event": "request_start",
                    "run_label": run_label,
                    "model": model,
                    "prompt_tokens": 230000,
                    "image_count": 1024,
                    "max_tokens": 32144,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        async def monitor():
            with (run_dir / "metrics_trace.log").open("x") as trace_file:
                while not done.is_set():
                    await asyncio.sleep(10)
                    response = await client.get("http://127.0.0.1:8000/metrics")
                    response.raise_for_status()
                    metrics, lines = parse_metrics(response.text)
                    now = time.monotonic()
                    prompt_tokens = metrics["vllm:prompt_tokens_total"]
                    generation_tokens = metrics["vllm:generation_tokens_total"]
                    if (
                        prompt_tokens != state["prompt_tokens"]
                        or generation_tokens != state["generation_tokens"]
                    ):
                        state["last_progress"] = now
                        state["prompt_tokens"] = prompt_tokens
                        state["generation_tokens"] = generation_tokens
                    sample = {
                        "elapsed_seconds": now - request_start,
                        "seconds_without_progress": now - state["last_progress"],
                        "metrics": metrics,
                        "raw": lines,
                    }
                    trace_file.write(json.dumps(sample) + "\n")
                    trace_file.flush()
                    print(
                        json.dumps(
                            {
                                "event": "metrics",
                                "elapsed_seconds": round(now - request_start, 3),
                                "seconds_without_progress": round(
                                    now - state["last_progress"], 3
                                ),
                                "prompt_tokens_total": prompt_tokens,
                                "generation_tokens_total": generation_tokens,
                                "running": metrics["vllm:num_requests_running"],
                                "waiting": metrics["vllm:num_requests_waiting"],
                            }
                        ),
                        flush=True,
                    )
                    wandb.log(
                        {
                            "runtime/elapsed_seconds": now - request_start,
                            "runtime/seconds_without_progress": now
                            - state["last_progress"],
                            "runtime/prompt_tokens_total": prompt_tokens,
                            "runtime/generation_tokens_total": generation_tokens,
                            "runtime/running_requests": metrics[
                                "vllm:num_requests_running"
                            ],
                            "runtime/waiting_requests": metrics[
                                "vllm:num_requests_waiting"
                            ],
                        }
                    )
                    if now - state["last_progress"] >= 1200:
                        health = await client.get("http://127.0.0.1:8000/health")
                        probe_payload = {
                            "model": model,
                            "messages": [
                                {"role": "user", "content": "只回答一个字：好"}
                            ],
                            "max_tokens": 1,
                            "temperature": 1.0,
                            "top_p": 0.95,
                            "ignore_eos": True,
                            "top_k": 20,
                        }
                        probe_result = {
                            "health_status": health.status_code,
                            "elapsed_seconds": now - request_start,
                            "seconds_without_progress": now
                            - state["last_progress"],
                        }
                        try:
                            probe = await client.post(
                                "http://127.0.0.1:8000/v1/chat/completions",
                                json=probe_payload,
                                timeout=60.0,
                            )
                            probe_result["probe_status"] = probe.status_code
                            probe_result["probe_body"] = probe.text
                        except httpx.ReadTimeout as error:
                            probe_result["probe_error"] = repr(error)
                        (run_dir / "wedge_probe.json").write_text(
                            json.dumps(probe_result, ensure_ascii=False, indent=2)
                        )
                        wandb.log(
                            {
                                "wedge/detected": 1,
                                "wedge/elapsed_seconds": now - request_start,
                                "wedge/seconds_without_progress": now
                                - state["last_progress"],
                                "wedge/health_status": health.status_code,
                            }
                        )
                        wandb_run.finish(exit_code=1)
                        raise RuntimeError(
                            "No prompt, generation, or stream progress for 1200 seconds"
                        )

        async def run_request():
            with (
                (run_dir / "stream.sse").open("x") as stream_file,
                (run_dir / "reasoning.txt").open("x") as reasoning_file,
                (run_dir / "output.txt").open("x") as output_file,
            ):
                async with client.stream(
                    "POST",
                    "http://127.0.0.1:8000/v1/chat/completions",
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        stream_file.write(line + "\n")
                        stream_file.flush()
                        state["last_progress"] = time.monotonic()
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        event = json.loads(data)
                        if "usage" in event and event["usage"] is not None:
                            state["usage"] = event["usage"]
                        if event["choices"]:
                            choice = event["choices"][0]
                            if choice["finish_reason"] is not None:
                                state["finish_reason"] = choice["finish_reason"]
                            delta = choice["delta"]
                            token_arrived = False
                            if (
                                "reasoning_content" in delta
                                and delta["reasoning_content"] is not None
                            ):
                                reasoning_file.write(delta["reasoning_content"])
                                reasoning_file.flush()
                                token_arrived = bool(delta["reasoning_content"])
                            if "content" in delta and delta["content"] is not None:
                                output_file.write(delta["content"])
                                output_file.flush()
                                token_arrived = token_arrived or bool(delta["content"])
                            if token_arrived:
                                now = time.monotonic()
                                if state["first_token_time"] is None:
                                    state["first_token_time"] = now
                                    print(
                                        json.dumps(
                                            {
                                                "event": "first_token",
                                                "ttft_seconds": now - request_start,
                                            }
                                        ),
                                        flush=True,
                                    )
                                state["last_token_time"] = now
            done.set()

        async with asyncio.TaskGroup() as group:
            group.create_task(monitor())
            group.create_task(run_request())

        request_end = time.monotonic()
        metrics_response = await client.get("http://127.0.0.1:8000/metrics")
        metrics_response.raise_for_status()
        metrics_after, selected_lines = parse_metrics(metrics_response.text)
        (run_dir / "metrics_after.txt").write_text("\n".join(selected_lines) + "\n")
        prompt_tokens_delta = (
            metrics_after["vllm:prompt_tokens_total"]
            - metrics_before["vllm:prompt_tokens_total"]
        )
        generation_tokens_delta = (
            metrics_after["vllm:generation_tokens_total"]
            - metrics_before["vllm:generation_tokens_total"]
        )
        prefill_seconds = (
            metrics_after["vllm:request_prefill_time_seconds_sum"]
            - metrics_before["vllm:request_prefill_time_seconds_sum"]
        )
        decode_seconds = (
            metrics_after["vllm:request_decode_time_seconds_sum"]
            - metrics_before["vllm:request_decode_time_seconds_sum"]
        )
        usage = state["usage"]
        result = {
            "run_label": run_label,
            "model": model,
            "finish_reason": state["finish_reason"],
            "usage": usage,
            "wall_seconds": request_end - request_start,
            "wall_ttft_seconds": state["first_token_time"] - request_start,
            "wall_decode_seconds": state["last_token_time"]
            - state["first_token_time"],
            "wall_decode_tokens_per_second": (usage["completion_tokens"] - 1)
            / (state["last_token_time"] - state["first_token_time"]),
            "engine_prompt_tokens": prompt_tokens_delta,
            "engine_generation_tokens": generation_tokens_delta,
            "engine_prefill_seconds": prefill_seconds,
            "engine_decode_seconds": decode_seconds,
            "engine_prefill_tokens_per_second": prompt_tokens_delta
            / prefill_seconds,
            "engine_decode_tokens_per_second": generation_tokens_delta
            / decode_seconds,
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
        }
        (run_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2)
        )
        assert usage["prompt_tokens"] == 230000
        assert usage["completion_tokens"] == 32144
        assert state["finish_reason"] == "length"
        assert prompt_tokens_delta == 230000
        assert generation_tokens_delta == 32144
        wandb.log(
            {
                "result/wall_seconds": result["wall_seconds"],
                "result/wall_ttft_seconds": result["wall_ttft_seconds"],
                "result/wall_decode_tokens_per_second": result[
                    "wall_decode_tokens_per_second"
                ],
                "result/engine_prefill_seconds": prefill_seconds,
                "result/engine_decode_seconds": decode_seconds,
                "result/engine_prefill_tokens_per_second": result[
                    "engine_prefill_tokens_per_second"
                ],
                "result/engine_decode_tokens_per_second": result[
                    "engine_decode_tokens_per_second"
                ],
                "result/prompt_tokens": usage["prompt_tokens"],
                "result/completion_tokens": usage["completion_tokens"],
                "wedge/detected": 0,
            }
        )
        wandb_run.finish()
        print(json.dumps(result, ensure_ascii=False), flush=True)


asyncio.run(main())
