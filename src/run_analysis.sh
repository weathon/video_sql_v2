#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
exec .venv_baselines/bin/python -u - <<'PY'
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx


def service_ready(url, model):
    try:
        response = httpx.get(url + "/models", timeout=10, trust_env=False)
    except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError):
        return False
    response.raise_for_status()
    models = [item["id"] for item in response.json()["data"]]
    if model not in models:
        raise RuntimeError(f"Unexpected models at {url}: {models}; expected {model}")
    print(f"SERVICE_READY url={url} model={model}", flush=True)
    return True


directory = Path("runs/analysis/services")
directory.mkdir(parents=True, exist_ok=True)
image = "video-sql-vllm-epd:egolife-baselines"
services = [
    (
        "embedding",
        8214,
        "qwen3-vl-embedding-8b",
        1,
        "8g",
        ["-v", "/raid/videosql-video-artifacts:/raid/videosql-video-artifacts:ro"],
        [
            "Qwen/Qwen3-VL-Embedding-8B",
            "--served-model-name",
            "qwen3-vl-embedding-8b",
            "--runner",
            "pooling",
            "--gpu-memory-utilization",
            "0.45",
            "--max-num-seqs",
            "30",
            "--mm-processor-cache-gb",
            "0",
            "--trust-remote-code",
            "--allowed-local-media-path",
            "/raid/videosql-video-artifacts/videos",
            "--renderer-num-workers",
            "4",
            "--max-num-batched-tokens",
            "32768",
            "--media-io-kwargs",
            json.dumps(
                {
                    "video": {
                        "video_backend": "qwen3_vl_uniform",
                        "backend": "torchcodec",
                        "num_frames": -1,
                        "seek_mode": "exact",
                        "num_ffmpeg_threads": 4,
                    }
                }
            ),
        ],
    ),
    (
        "qwen",
        8310,
        "Qwen/Qwen3.8-27B-FP8",
        2,
        "32g",
        ["-e", "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1"],
        [
            "Qwen/Qwen3.8-27B-FP8",
            "--max-model-len",
            "314573",
            "--hf-overrides",
            json.dumps(
                {
                    "text_config": {
                        "rope_parameters": {
                            "mrope_interleaved": True,
                            "mrope_section": [11, 11, 10],
                            "rope_type": "yarn",
                            "rope_theta": 10000000,
                            "partial_rotary_factor": 0.25,
                            "factor": 1.2,
                            "original_max_position_embeddings": 262144,
                        }
                    }
                }
            ),
            "--max-num-seqs",
            "50",
            "--gpu-memory-utilization",
            "0.90",
            "--enable-prefix-caching",
            "--no-enable-flashinfer-autotune",
            "--tensor-parallel-size",
            "2",
            "--mm-processor-cache-gb",
            "4",
            "--limit-mm-per-prompt",
            json.dumps({"image": 650, "video": 1}),
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "qwen3_coder",
            "--reasoning-parser",
            "qwen3",
            "--enable-prompt-tokens-details",
            "--speculative-config",
            json.dumps(
                {"method": "mtp", "num_speculative_tokens": 3, "max_model_len": 314573}
            ),
            "--kv-transfer-config",
            json.dumps(
                {
                    "kv_connector": "SimpleCPUOffloadConnector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {
                        "cpu_bytes_to_use_per_rank": 137438953472,
                        "lazy_offload": False,
                    },
                }
            ),
            "--default-chat-template-kwargs",
            json.dumps({"enable_thinking": True, "preserve_thinking": True}),
        ],
    ),
]
started = []
try:
    for role, port, model, gpu_count, shm_size, docker_args, model_args in services:
        url = f"http://127.0.0.1:{port}/v1"
        print(f"SERVICE_CHECK role={role} url={url}", flush=True)
        if service_ready(url, model):
            continue
        containers = subprocess.check_output(
            ["docker", "ps", "--filter", f"publish={port}", "--format", "{{.ID}}"],
            text=True,
        ).split()
        process = None
        if containers:
            if len(containers) != 1:
                raise RuntimeError(
                    f"Multiple containers publish port {port}: {containers}"
                )
            name = containers[0]
            print(f"SERVICE_WAIT role={role} container={name}", flush=True)
        else:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", port))
            inventory = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            active = set(
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=gpu_uuid",
                        "--format=csv,noheader",
                    ],
                    text=True,
                ).split()
            )
            free = []
            for row in inventory.splitlines():
                index, gpu_uuid, used, utilization = (
                    value.strip() for value in row.split(",")
                )
                if int(used) == 0 and int(utilization) == 0 and gpu_uuid not in active:
                    free.append(index)
            print(f"GPU_INVENTORY\n{inventory}", flush=True)
            if len(free) < gpu_count:
                raise RuntimeError(
                    f"{role} needs {gpu_count} idle GPUs; available: {free}"
                )
            memory = dict(
                line.split(":", 1)
                for line in Path("/proc/meminfo").read_text().splitlines()
            )
            available = int(memory["MemAvailable"].split()[0]) * 1024
            required = 2 * 137438953472 if role == "qwen" else 8 * 1024**3
            print(
                f"HOST_MEMORY available_bytes={available} required_bytes={required}",
                flush=True,
            )
            if available < required:
                raise RuntimeError(
                    f"Insufficient host memory for {role}: {available} < {required}"
                )
            devices = ",".join(free[:gpu_count])
            name = f"video-sql-analysis-{role}-{os.getpid()}"
            command = [
                "docker",
                "run",
                "--init",
                "--pull=never",
                "--name",
                name,
                "--gpus",
                f'"device={devices}"',
                "--shm-size",
                shm_size,
                "-p",
                f"127.0.0.1:{port}:8000",
                "-e",
                "HF_HOME=/hf_cache",
                "-e",
                "HF_HUB_CACHE=/hf_cache",
                "-e",
                "HF_HUB_OFFLINE=1",
                "-e",
                "VLLM_SERVER_DEV_MODE=1",
                "-v",
                "/raid/huggingface:/hf_cache",
                *docker_args,
                image,
                *model_args,
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
            ]
            log_path = directory / f"{name}.log"
            print(
                f"SERVICE_START log={log_path} command={json.dumps(command)}",
                flush=True,
            )
            log = log_path.open("w")
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            started.append((name, process, log))
        while not service_ready(url, model):
            if process is not None:
                status = process.poll()
                if status is not None:
                    raise RuntimeError(
                        f"{name} exited with status {status}; see {log_path}"
                    )
            else:
                running = subprocess.check_output(
                    ["docker", "inspect", "--format", "{{.State.Running}}", name],
                    text=True,
                ).strip()
                if running != "true":
                    raise RuntimeError(
                        f"Existing service {name} stopped before {url} was ready"
                    )
            time.sleep(5)
    command = [
        sys.executable,
        "-u",
        "-m",
        "src.analysis",
        *[
            str(path)
            for path in sorted(
                Path("runs/mmlifelong-week/collection/questions").glob("*/batch.json")
            )
        ],
        "--output",
        "runs/analysis",
    ]
    print(f"ANALYSIS_START command={json.dumps(command)}", flush=True)
    subprocess.run(command, check=True)
finally:
    for name, process, log in reversed(started):
        print(f"SERVICE_CLEANUP container={name}", flush=True)
        if process.poll() is None:
            subprocess.run(["docker", "stop", name], check=True)
        process.wait()
        subprocess.run(["docker", "rm", name], check=True)
        log.close()
PY
