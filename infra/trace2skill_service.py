"""Foreground, isolated services for the Trace2Skill baseline."""

import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/trace2skill-baseline"
MODEL = "Qwen/Qwen3.8-27B-FP8"
IMAGE = "video-sql-trace2skill:fp8"
NETWORK = "video-sql-trace2skill"
VOLUME = "video-sql-trace2skill-ec"
role = sys.argv[1]
name = "trace2skill-" + role
command = ["docker", "run", "--init", "--name", name, "--network", NETWORK]
shared = ["-e", "HF_HOME=/hf_cache", "-v", "/raid/huggingface:/hf_cache", "-v", VOLUME + ":/epd_shared/cache"]
if role in ("profile", "pd"):
    command += ["--gpus", '"device=0,1,2,3"', "--shm-size=32g", "-p", "127.0.0.1:8024:8000",
                "-e", "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1", "-e", "VLLM_SERVER_DEV_MODE=1",
                "-e", "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1"] + shared
    command += [IMAGE, MODEL, "--host", "0.0.0.0", "--port", "8000", "--max-model-len", "314573",
                "--hf-overrides", json.dumps({"text_config": {"rope_parameters": {"mrope_interleaved": True, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 1.2, "original_max_position_embeddings": 262144}}}),
                "--max-num-batched-tokens", "32768", "--gpu-memory-utilization", "0.90",
                "--kv-cache-dtype", "auto", "--enable-prefix-caching", "--enable-chunked-prefill",
                "--mamba-cache-mode", "align", "--async-scheduling", "--attention-backend", "FLASHINFER",
                "--enable-flashinfer-autotune", "--mm-processor-cache-gb", "4",
                "--mm-processor-kwargs", json.dumps({"min_pixels": 65536, "max_pixels": 414720}),
                "--limit-mm-per-prompt", json.dumps({"image": 650, "video": 0}),
                "--tensor-parallel-size", "4", "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
                "--reasoning-parser", "qwen3", "--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": 1}),
                "--kv-transfer-config", json.dumps({"kv_connector": "SimpleCPUOffloadConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"cpu_bytes_to_use_per_rank": 236223201280, "lazy_offload": False}}),
                "--ec-transfer-config", json.dumps({"ec_connector": "ECExampleConnector", "ec_role": "ec_consumer", "ec_connector_extra_config": {"shared_storage_path": "/epd_shared/cache"}}),
                "--default-chat-template-kwargs", json.dumps({"enable_thinking": True, "preserve_thinking": True})]
    if role == "profile":
        command += ["--max-num-seqs", "128"]
    else:
        command += ["--max-num-seqs", "50"]
elif role == "encoder":
    command += ["--gpus", '"device=4"', "--shm-size=32g", "-e", "VLLM_SERVER_DEV_MODE=1"] + shared
    command += [IMAGE, MODEL, "--host", "0.0.0.0", "--port", "8000", "--mm-encoder-only", "--enforce-eager",
                "--no-enable-prefix-caching", "--gpu-memory-utilization", "0.45", "--max-num-batched-tokens", "114688",
                "--max-num-seqs", "24", "--mm-processor-cache-gb", "4", "--limit-mm-per-prompt", json.dumps({"image": 650, "video": 0}),
                "--mm-processor-kwargs", json.dumps({"min_pixels": 65536, "max_pixels": 414720}),
                "--enable-request-id-headers", "--ec-transfer-config", json.dumps({"ec_connector": "ECExampleConnector", "ec_role": "ec_producer", "ec_connector_extra_config": {"shared_storage_path": "/epd_shared/cache"}})]
elif role == "embedding":
    command += ["--gpus", '"device=4"', "--shm-size=8g", "-p", "127.0.0.1:8014:8000"] + shared
    command += ["-v", "/raid/wg25r/video_sql_v2:/workspace/video_sql_v2:ro", IMAGE,
                "Qwen/Qwen3-VL-Embedding-8B", "--served-model-name", "qwen3-vl-embedding-8b", "--runner", "pooling",
                "--host", "0.0.0.0", "--port", "8000", "--gpu-memory-utilization", "0.45",
                "--mm-processor-cache-gb", "0", "--trust-remote-code", "--allowed-local-media-path", "/workspace/video_sql_v2/data/pilot/clips",
                "--renderer-num-workers", "4", "--max-num-batched-tokens", "32768",
                "--media-io-kwargs", json.dumps({"video": {"video_backend": "qwen3_vl", "backend": "torchcodec", "fps": 2, "num_frames": -1, "seek_mode": "approximate", "num_ffmpeg_threads": 4}})]
elif role == "proxy":
    command += ["-p", "127.0.0.1:8003:8000", "-v", str(ROOT / "infra") + ":/video_sql_infra:ro",
                "-v", VOLUME + ":/epd_shared/cache", "--entrypoint", "python3", IMAGE,
                "/video_sql_infra/responses_epd_proxy.py", "--host", "0.0.0.0", "--port", "8000",
                "--encoder-url", "http://trace2skill-encoder:8000", "--pd-url", "http://trace2skill-pd:8000", "--cache-dir", "/epd_shared/cache"]
else:
    raise ValueError(f"Unknown service: {role}")
RUN.mkdir(parents=True, exist_ok=True)
(RUN / f"{role}-command.json").write_text(json.dumps(command, indent=2) + "\n")
print(json.dumps(command), flush=True)
os.execvp(command[0], command)
