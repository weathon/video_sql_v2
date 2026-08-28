# Qwen reasoning settings for video_sql_v2

- For `Inferact/Qwen3.8-Flash-Next-NVFP4`, use `reasoning_effort=low` for indexing and other descriptive preprocessing, and `reasoning_effort=medium` for the actual agent loop. Never use `xhigh` for this project.
- Keep `enable_thinking=true` and `preserve_thinking=true` in the vLLM server default chat-template kwargs. Keep reasoning items preserved in the Responses API harness. Before the formal run, verify both the actual server startup arguments and a real multi-turn server/harness trace; do not infer this from Responses API defaults alone.
