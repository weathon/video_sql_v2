"""Run one dedicated Week TP=2 server in a foreground Docker session."""

import argparse
import json
import os
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('lane', type=int, choices=(0, 1))
parser.add_argument('--gpus', required=True)
parser.add_argument('--port', type=int, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
directory = root / 'runs/mmlifelong-week/services'
directory.mkdir(parents=True, exist_ok=True)
command = ['docker', 'run', '--init', '--name', f'week-rollout-{args.lane}', '--gpus', f'"device={args.gpus}"', '--shm-size=32g', '-p', f'127.0.0.1:{args.port}:8000', '-e', 'HF_HOME=/hf_cache', '-e', 'HF_HUB_CACHE=/hf_cache', '-e', 'HF_HUB_OFFLINE=1', '-e', 'VLLM_SERVER_DEV_MODE=1', '-e', 'VLLM_ALLOW_LONG_MAX_MODEL_LEN=1', '-v', '/raid/huggingface:/hf_cache']
command += ['video-sql-vllm-epd:egolife-baselines', 'Qwen/Qwen3.8-27B-FP8', '--host', '0.0.0.0', '--port', '8000', '--max-model-len', '314573', '--hf-overrides', json.dumps({'text_config': {'rope_parameters': {'mrope_interleaved': True, 'mrope_section': [11, 11, 10], 'rope_type': 'yarn', 'rope_theta': 10000000, 'partial_rotary_factor': 0.25, 'factor': 1.2, 'original_max_position_embeddings': 262144}}}), '--max-num-seqs', '50', '--gpu-memory-utilization', '0.90', '--enable-prefix-caching', '--no-enable-flashinfer-autotune', '--tensor-parallel-size', '2', '--mm-processor-cache-gb', '4', '--limit-mm-per-prompt', json.dumps({'image': 650, 'video': 1}), '--enable-auto-tool-choice', '--tool-call-parser', 'qwen3_coder', '--reasoning-parser', 'qwen3', '--enable-prompt-tokens-details', '--speculative-config', json.dumps({'method': 'mtp', 'num_speculative_tokens': 3, 'max_model_len': 314573}), '--kv-transfer-config', json.dumps({'kv_connector': 'SimpleCPUOffloadConnector', 'kv_role': 'kv_both', 'kv_connector_extra_config': {'cpu_bytes_to_use_per_rank': 137438953472, 'lazy_offload': False}}), '--default-chat-template-kwargs', json.dumps({'enable_thinking': True, 'preserve_thinking': True})]
(directory / f'lane-{args.lane}-command.json').write_text(json.dumps(command, indent=2) + '\n')
print(json.dumps(command), flush=True)
os.execvp(command[0], command)
