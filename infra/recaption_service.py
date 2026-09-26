"""Launch one MM-Lifelong re-caption service in the foreground."""

import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/mmlifelong-recaption/services'
ARTIFACTS = ROOT / 'runs/mmlifelong-recaption/artifacts'
MODEL = 'Qwen/Qwen3.8-27B-FP8'
IMAGE = 'video-sql-vllm-epd:egolife-baselines'
NETWORK = 'mmlifelong-recaption'
VOLUME = 'mmlifelong-recaption-ec-sql'
role = sys.argv[1]
RUN.mkdir(parents=True, exist_ok=True)
command = ['docker', 'run', '--init', '--name', 'mmlifelong-recaption-' + role, '--network', NETWORK]
shared = ['-e', 'HF_HOME=/hf_cache', '-e', 'HF_HUB_CACHE=/hf_cache', '-e', 'HF_HUB_OFFLINE=1', '-e', 'VLLM_SERVER_DEV_MODE=1', '-v', '/raid/huggingface:/hf_cache']
if role == 'caption':
    command += ['--gpus', '"device=0,1,2,3"', '--shm-size=64g', '--ipc=host', '-p', '127.0.0.1:8244:8000'] + shared
    command += [IMAGE, MODEL, '--host', '0.0.0.0', '--port', '8000', '--data-parallel-size', '4', '--max-model-len', '131072', '--max-num-seqs', '32', '--max-num-batched-tokens', '32768', '--enable-chunked-prefill', '--gpu-memory-utilization', '0.90', '--enable-prefix-caching', '--no-enable-flashinfer-autotune', '--mm-processor-cache-gb', '4', '--limit-mm-per-prompt', json.dumps({'image': 650, 'video': 0}), '--reasoning-parser', 'qwen3', '--enable-prompt-tokens-details', '--speculative-config', json.dumps({'method': 'mtp', 'num_speculative_tokens': 3}), '--default-chat-template-kwargs', json.dumps({'enable_thinking': True, 'preserve_thinking': True})]
elif role == 'pd-sql':
    command += ['--gpus', '"device=5,6"', '--shm-size=32g', '-p', '127.0.0.1:8234:8000', '-e', 'VLLM_ALLOW_LONG_MAX_MODEL_LEN=1', '-v', VOLUME + ':/epd_shared/cache'] + shared
    command += [IMAGE, MODEL, '--host', '0.0.0.0', '--port', '8000', '--max-model-len', '314573', '--hf-overrides', json.dumps({'text_config': {'rope_parameters': {'mrope_interleaved': True, 'mrope_section': [11, 11, 10], 'rope_type': 'yarn', 'rope_theta': 10000000, 'partial_rotary_factor': 0.25, 'factor': 1.2, 'original_max_position_embeddings': 262144}}}), '--max-num-seqs', '50', '--gpu-memory-utilization', '0.90', '--enable-prefix-caching', '--no-enable-flashinfer-autotune', '--tensor-parallel-size', '2', '--mm-processor-cache-gb', '4', '--limit-mm-per-prompt', json.dumps({'image': 650, 'video': 1}), '--enable-auto-tool-choice', '--tool-call-parser', 'qwen3_coder', '--reasoning-parser', 'qwen3', '--enable-prompt-tokens-details', '--speculative-config', json.dumps({'method': 'mtp', 'num_speculative_tokens': 3, 'max_model_len': 314573}), '--kv-transfer-config', json.dumps({'kv_connector': 'SimpleCPUOffloadConnector', 'kv_role': 'kv_both', 'kv_connector_extra_config': {'cpu_bytes_to_use_per_rank': 137438953472, 'lazy_offload': False}}), '--ec-transfer-config', json.dumps({'ec_connector': 'ECExampleConnector', 'ec_role': 'ec_consumer', 'ec_connector_extra_config': {'shared_storage_path': '/epd_shared/cache'}}), '--default-chat-template-kwargs', json.dumps({'enable_thinking': True, 'preserve_thinking': True})]
elif role == 'encoder-sql':
    command += ['--gpus', '"device=4"', '--shm-size=32g', '-p', '127.0.0.1:8235:8000', '-v', VOLUME + ':/epd_shared/cache'] + shared
    command += [IMAGE, MODEL, '--host', '0.0.0.0', '--port', '8000', '--mm-encoder-only', '--enforce-eager', '--no-enable-prefix-caching', '--gpu-memory-utilization', '0.45', '--max-num-batched-tokens', '114688', '--max-num-seqs', '30', '--mm-processor-cache-gb', '4', '--limit-mm-per-prompt', json.dumps({'image': 650, 'video': 1}), '--enable-request-id-headers', '--ec-transfer-config', json.dumps({'ec_connector': 'ECExampleConnector', 'ec_role': 'ec_producer', 'ec_connector_extra_config': {'shared_storage_path': '/epd_shared/cache'}})]
elif role == 'embedding':
    command += ['--gpus', '"device=4"', '--shm-size=8g', '-p', '127.0.0.1:8214:8000', '-v', f'{ARTIFACTS}:{ARTIFACTS}:ro'] + shared
    command += [IMAGE, 'Qwen/Qwen3-VL-Embedding-8B', '--served-model-name', 'qwen3-vl-embedding-8b', '--runner', 'pooling', '--host', '0.0.0.0', '--port', '8000', '--gpu-memory-utilization', '0.45', '--max-num-seqs', '30', '--mm-processor-cache-gb', '0', '--trust-remote-code', '--allowed-local-media-path', str(ARTIFACTS / 'videos'), '--renderer-num-workers', '4', '--max-num-batched-tokens', '32768', '--media-io-kwargs', json.dumps({'video': {'video_backend': 'qwen3_vl_uniform', 'backend': 'torchcodec', 'num_frames': -1, 'seek_mode': 'exact', 'num_ffmpeg_threads': 4}})]
elif role == 'proxy-sql':
    command += ['-p', '127.0.0.1:8204:8000', '-v', str(ROOT / 'infra') + ':/video_sql_infra:ro', '-v', VOLUME + ':/epd_shared/cache', '--entrypoint', 'python3', IMAGE, '/video_sql_infra/responses_epd_proxy.py', '--host', '0.0.0.0', '--port', '8000', '--encoder-url', 'http://mmlifelong-recaption-encoder-sql:8000', '--pd-url', 'http://mmlifelong-recaption-pd-sql:8000', '--cache-dir', '/epd_shared/cache']
else:
    raise ValueError(f'Unknown service: {role}')
(RUN / f'{role}-command.json').write_text(json.dumps(command, indent=2) + '\n')
print(json.dumps(command), flush=True)
os.execvp(command[0], command)
