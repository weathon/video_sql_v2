"""Start the online-training Qwen TP=4 server in a foreground Docker session."""

import argparse
import csv
import io
import json
import os
import socket
import subprocess
from pathlib import Path

import httpx


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--gpus", required=True, help="Four idle GPU indices, comma separated"
)
parser.add_argument("--port", type=int, default=8310)
parser.add_argument("--run-dir", type=Path, default=Path("runs/online-training"))
args = parser.parse_args()
gpu_ids = [int(value) for value in args.gpus.split(",")]
if len(gpu_ids) != 4 or len(set(gpu_ids)) != 4:
    raise ValueError("The training service requires four distinct GPUs")
gpu_text = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.used",
        "--format=csv,noheader,nounits",
    ],
    text=True,
)
process_text = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits",
    ],
    text=True,
)
occupied = {row[0].strip() for row in csv.reader(io.StringIO(process_text))}
gpu_rows = {int(row[0]): row for row in csv.reader(io.StringIO(gpu_text))}
for gpu in gpu_ids:
    row = gpu_rows[gpu]
    if row[1].strip() in occupied or int(row[2]) >= 1024:
        raise RuntimeError(
            f"Requested GPU is occupied: {row}; active processes: {process_text}"
        )
with socket.socket() as probe:
    probe.bind(("127.0.0.1", args.port))
memory = {
    line.split(":")[0]: int(line.split()[1]) * 1024
    for line in Path("/proc/meminfo").read_text().splitlines()
}
cpu_offload_per_rank = 137438953472
required_memory = cpu_offload_per_rank * 4 + 32 * 1024**3
if memory["MemAvailable"] < required_memory:
    raise RuntimeError(
        f"Insufficient host memory: available={memory['MemAvailable']} required={required_memory}"
    )
response = httpx.get("http://127.0.0.1:8214/v1/models", timeout=10, trust_env=False)
response.raise_for_status()
if "qwen3-vl-embedding-8b" not in [item["id"] for item in response.json()["data"]]:
    raise RuntimeError(f"Unexpected shared embedding service: {response.text}")
directory = args.run_dir.resolve() / "services"
directory.mkdir(parents=True, exist_ok=True)
command = [
    "docker",
    "run",
    "--init",
    "--name",
    "online-training-qwen",
    "--gpus",
    f'"device={args.gpus}"',
    "--shm-size=32g",
    "-p",
    f"127.0.0.1:{args.port}:8000",
    "-e",
    "HF_HOME=/hf_cache",
    "-e",
    "HF_HUB_CACHE=/hf_cache",
    "-e",
    "HF_HUB_OFFLINE=1",
    "-e",
    "VLLM_SERVER_DEV_MODE=1",
    "-e",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1",
    "-v",
    "/raid/huggingface:/hf_cache",
    "video-sql-vllm-epd:egolife-baselines",
    "Qwen/Qwen3.8-27B-FP8",
    "--host",
    "0.0.0.0",
    "--port",
    "8000",
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
    "4",
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
    json.dumps({"method": "mtp", "num_speculative_tokens": 3, "max_model_len": 314573}),
    "--kv-transfer-config",
    json.dumps(
        {
            "kv_connector": "SimpleCPUOffloadConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "cpu_bytes_to_use_per_rank": cpu_offload_per_rank,
                "lazy_offload": False,
            },
        }
    ),
    "--default-chat-template-kwargs",
    json.dumps({"enable_thinking": True, "preserve_thinking": True}),
]
(directory / "qwen-command.json").write_text(json.dumps(command, indent=2) + "\n")
print(
    json.dumps(
        {
            "gpus": gpu_ids,
            "host_memory_available": memory["MemAvailable"],
            "command": command,
        }
    ),
    flush=True,
)
os.execvp(command[0], command)
