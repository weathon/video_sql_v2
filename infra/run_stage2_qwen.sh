#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: $0 GPU_ID HOST_PORT" >&2
    exit 2
fi

GPU_ID="$1"
HOST_PORT="$2"

exec docker run --rm --name stage2-qwen38-index --gpus "device=${GPU_ID}" --ipc=host -p "${HOST_PORT}:8000" -e HF_HOME=/hf_cache -v /raid/wg25r/video_sql_v2/runs/qwen38_bench/hf_cache:/hf_cache -v /raid/wg25r/video_sql_v2:/raid/wg25r/video_sql_v2:ro vllm/vllm-openai:qwen38 Inferact/Qwen3.8-27B-NVFP4 --host 0.0.0.0 --port 8000 --tensor-parallel-size 1 --max-model-len 262144 --max-num-batched-tokens 32768 --max-num-seqs 32 --limit-mm-per-prompt.image 0 --limit-mm-per-prompt.video 1 --allowed-local-media-path /raid/wg25r/video_sql_v2/data/pilot/clips/stage2 --media-io-kwargs '{"video":{"video_backend":"qwen3_vl","backend":"torchcodec","fps":2,"num_frames":-1,"seek_mode":"approximate","num_ffmpeg_threads":4}}' --mm-encoder-tp-mode data --mm-processor-cache-type shm --mm-processor-cache-gb 8 --kv-cache-dtype fp8 --gpu-memory-utilization 0.75 --gdn-prefill-backend flashinfer --enable-chunked-prefill --async-scheduling --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --speculative-config '{"method":"mtp","num_speculative_tokens":3}' --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true}'
