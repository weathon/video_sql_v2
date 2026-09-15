"""Launch one Qwen3.8-27B fair-comparison service (vLLM for MAGIC, vLLM for VideoSQL, or the embedding server)."""

import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/fair-qwen38/services"
MODEL = "Qwen/Qwen3.8-27B-FP8"
IMAGE = "video-sql-vllm-epd:egolife-baselines"
role = sys.argv[1]
RUN.mkdir(parents=True, exist_ok=True)
command = ["docker", "run", "--init", "--name", "fair-qwen38-" + role]
shared = ["-e", "HF_HOME=/hf_cache", "-e", "HF_HUB_CACHE=/hf_cache", "-e", "HF_HUB_OFFLINE=1", "-e", "VLLM_SERVER_DEV_MODE=1", "-v", "/raid/huggingface:/hf_cache"]
if role in ("vllm-magic", "vllm-sql"):
    command += ["--gpus", '"device=0,1"' if role == "vllm-magic" else '"device=2,3"', "--shm-size=32g",
                "-p", "127.0.0.1:8130:8000" if role == "vllm-magic" else "127.0.0.1:8131:8000",
                "-e", "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1"] + shared
    command += [IMAGE, MODEL, "--host", "0.0.0.0", "--port", "8000", "--max-model-len", "314573",
                "--hf-overrides", json.dumps({"text_config": {"rope_parameters": {"mrope_interleaved": True, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 1.2, "original_max_position_embeddings": 262144}}}),
                "--max-num-seqs", "50", "--gpu-memory-utilization", "0.90", "--enable-prefix-caching", "--no-enable-flashinfer-autotune", "--tensor-parallel-size", "2",
                "--mm-processor-cache-gb", "4", "--limit-mm-per-prompt", json.dumps({"image": 650, "video": 1}),
                "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder", "--reasoning-parser", "qwen3", "--enable-prompt-tokens-details",
                "--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": 3, "max_model_len": 314573}),
                "--kv-transfer-config", json.dumps({"kv_connector": "SimpleCPUOffloadConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"cpu_bytes_to_use_per_rank": 137438953472, "lazy_offload": False}}),
                "--default-chat-template-kwargs", json.dumps({"enable_thinking": True, "preserve_thinking": True})]
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
