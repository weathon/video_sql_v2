# QA harness + Trace2Skill baseline

This worktree implements the approved baseline, independently of the project's skill trainer.
The upstream source is pinned to Qwen-Applications/Trace2Skill commit
`3d0b52a140f002a512930252b613c49048f7d5ac`. Its mandatory skill validator comes from
anthropics/skills blob `ed8e1dddce77b16af13c6f36b3fe86c4ac7c590c`.

Run `uv run python -m src.trace2skill_baseline` from this worktree with
`DEEPINFRA_API_KEY` in its ignored `.env` and the existing embedding service at
`http://127.0.0.1:8014/v1`. It collects all 266 month training questions with five
independent rollouts each, analyzes score 4–5 as success and 0–3 as failure, performs
one upstream Combined evolution, then collects the same 1,330 rollouts with the frozen
evolved skill. Results, complete traces/media, intermediate patches and comparison
are written under `runs/trace2skill-baseline-deepinfra-glm-5.3-flash-image-replace`. Completed rollout, analysis and evolution
LLM artifacts are reused on resume; failures exhaust their applicable correction
opportunities and raise. No held-out evaluation is started.

The seven initial Markdown files are copied byte-for-byte. Only the directory bridge
adds frontmatter and reference links to upstream SKILL.md. The six subskills retain
their contents and on-demand read_skill(name) catalog. Evolution retains upstream
parallel proposals against one frozen snapshot, hierarchical merge, translation,
programmatic application and three-round verification. The bridge preserves those
stages while requiring SDK Chat structured-parse output and rejecting silent fallback.
The upstream spreadsheet failure-analysis prompt is adapted in error_analysis_qa.txt
to evidence search, minimal answer correction and the original 0–5 QA judge.

All generative model calls use DeepInfra `zai-org/GLM-5.3-Flash` through
`https://api.deepinfra.com/v1/openai` Chat Completions with medium reasoning.
Agents SDK explicitly replays same-model `reasoning_content`; direct analysis and
evolution calls preserve it in assistant messages during correction and continuation.
The shared request
semaphore is 256, including nested agents, judges and embeddings. HTTP connect timeout
is 60 seconds; read/write/pool timeout is 100000 seconds; pools permit 1000 connections
and 256 keepalive connections. Cloud calls omit sampling parameters and token limits.
Weave uses the Agents SDK processor and rollout roots, with local traces retained.
Streaming requests include usage and preserve the provider's raw usage payload.
Multimodal tool results are carried in user messages after the complete batch of tool
replies, retaining tool IDs and ordered text/images. This avoids the Chat adapter's
text-only tool-output conversion; canonical histories retain the original tool results.
Each tool result exposes an integer tool_id. replace_tool(tool_id, replacement) changes
the model's context while preserving original outputs and replacement calls in the trace
for Trace2Skill analysis. Its description recommends replacing the latest result for KV
cache reuse and explains removing wrong-call output and summarizing redundant evidence.
This run starts from scratch with the DeepInfra model. W&B records completed-rollout mean generated
tokens (including reasoning), model/tool calls, mean input tokens and peak context tokens,
including retries and nested agents but excluding judge from trace length. Throughput is
completion-accounted input/output tokens and responses per second over 60 seconds,
including judge; input-token throughput includes cached tokens.
W&B also records `initial/average_rollout_time_seconds` and
`evolved/average_rollout_time_seconds`: mean elapsed QA time per completed rollout,
including request queueing, tools, nested agents, compaction and timed retry attempts,
excluding judge and process downtime. Per-attempt durations are saved for resume.

Generative calls no longer use the local PD, encoder, EPD proxy or cache-clearing
endpoints. Embeddings retain `qwen3-vl-embedding-8b` to match the existing index.
The harness keeps its 314573 context budget, 178000 remaining-token compaction
threshold, 480p frames and cumulative 650-image allowance per clip. The old Qwen
image-token calculation is not used as a claim about GLM; accounting uses API usage.
Compaction remains the harness's text handoff, without Responses compaction items.

The embedding service runs in a foreground exec session via infra/trace2skill_service.py. Full
repository review precedes the formal run. Formal traces provide correctness evidence;
no syntax, smoke or preflight tests are used. Stop and manually remove this run's
containers after inspecting the final artifacts and logs.
