"""Launch one fair-comparison service (Qwen3.5 vLLM or the embedding server) in the foreground."""

import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/fair-qwen35/services"
MODEL = "Qwen/Qwen3.5-35B-A3B-FP8"
IMAGE = "video-sql-vllm-epd:egolife-baselines"
role = sys.argv[1]
RUN.mkdir(parents=True, exist_ok=True)
command = ["docker", "run", "--init", "--name", "fair-qwen35-" + role]
shared = ["-e", "HF_HOME=/hf_cache", "-e", "HF_HUB_CACHE=/hf_cache", "-e", "HF_HUB_OFFLINE=1", "-v", "/raid/huggingface:/hf_cache"]
if role == "vllm":
    command += ["--gpus", '"device=0,1,2,3"', "--shm-size=32g", "-p", "127.0.0.1:8130:8000"] + shared + ["-e", "HF_HUB_CACHE=/hf_cache/hub"]
    command += [IMAGE, MODEL, "--host", "0.0.0.0", "--port", "8000", "--trust-remote-code", "--enable-expert-parallel", "--tensor-parallel-size", "2", "--data-parallel-size", "2",
                "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder", "--reasoning-parser", "qwen3", "--mm-encoder-tp-mode", "data",
                "--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": 2}),
                "--kv-transfer-config", json.dumps({"kv_connector": "SimpleCPUOffloadConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"cpu_bytes_to_use_per_rank": 128000000000, "lazy_offload": False}})]
elif role == "embedding":
    command += ["--gpus", '"device=4"', "--shm-size=8g", "-p", "127.0.0.1:8114:8000"] + shared
    command += ["-v", "/raid/videosql-video-artifacts:/raid/videosql-video-artifacts:ro", IMAGE, "Qwen/Qwen3-VL-Embedding-8B", "--served-model-name", "qwen3-vl-embedding-8b", "--runner", "pooling",
                "--host", "0.0.0.0", "--port", "8000", "--gpu-memory-utilization", "0.45", "--max-num-seqs", "30", "--mm-processor-cache-gb", "0", "--trust-remote-code",
                "--allowed-local-media-path", "/raid/videosql-video-artifacts/videos", "--renderer-num-workers", "4", "--max-num-batched-tokens", "32768",
                "--media-io-kwargs", json.dumps({"video": {"video_backend": "qwen3_vl_uniform", "backend": "torchcodec", "num_frames": -1, "seek_mode": "exact", "num_ffmpeg_threads": 4}})]
else:
    raise ValueError(f"Unknown service: {role}")
(RUN / f"{role}-command.json").write_text(json.dumps(command, indent=2) + "\n")
print(json.dumps(command), flush=True)
os.execvp(command[0], command)
