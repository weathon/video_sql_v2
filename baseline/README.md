# QA harness + Trace2Skill baseline

The current baseline starts fresh with NVFP4 and nontraining day/week subsets.
It does not reuse Luna rollouts. The new formal run has not started; disk space and
review authorization remain unresolved. See [the current handoff](../notes/TRACE2SKILL_NVFP4_HANDOFF.md)
for verified model traces, configuration, service state and remaining work.

This worktree implements the approved baseline, independently of the project's skill trainer.
The upstream source is pinned to Qwen-Applications/Trace2Skill commit
`3d0b52a140f002a512930252b613c49048f7d5ac`. Its mandatory skill validator comes from
anthropics/skills blob `ed8e1dddce77b16af13c6f36b3fe86c4ac7c590c`.

When the user asks to resume, run `.venv/bin/python -u -m src.trace2skill_baseline`
from `/raid/wg25r/video_sql_2` with the local EPD Responses endpoint at
`http://127.0.0.1:8003/v1` and embedding service at
`http://127.0.0.1:8014/v1`. It samples 100 questions each from day/test and week/test with seed 0 and five
independent rollouts each, analyzes score 4–5 as success and 0–3 as failure, performs
one upstream Combined evolution, then collects the same 1,000 rollouts with the frozen
evolved skill. Results, complete traces/media, intermediate patches and comparison
are written under `runs/trace2skill-baseline-nvfp4-day-week-seed0`. Keep this directory
and use a new foreground log filename when resuming. Completed rollout, analysis and evolution
LLM artifacts are reused on resume; saved answers continue through judging without
repeating QA, while interrupted attempts without an answer are archived and rerun.
Exhausted QA/judge retries skip the whole rollout, preserving traces/media and a
resumable skipped.json record; skipped rollouts are excluded from scored means and
trajectory analysis, with skipped_rollouts counted in the phase summary. Per-trajectory
analysis skips also retain records and are counted separately; no empty report enters
evolution. Quota exhaustion and non-retryable errors still stop the run. Upstream
patch evolution retains its error behavior. No held-out evaluation is started.

The seven initial Markdown files are copied byte-for-byte. Only the directory bridge
adds frontmatter and reference links to upstream SKILL.md. The six subskills retain
their contents and on-demand read_skill(name) catalog. Evolution retains upstream
parallel proposals against one frozen snapshot, hierarchical merge, translation,
programmatic application and three-round verification. The bridge preserves those
stages while requiring SDK Responses structured-parse output and rejecting silent fallback.
The upstream spreadsheet failure-analysis prompt is adapted in error_analysis_qa.txt
to evidence search, minimal answer correction and the original 0–5 QA judge.

All generative model calls use `Inferact/Qwen3.8-Flash-Next-NVFP4` through
`http://127.0.0.1:8003/v1` Responses with `reasoning={"effort":"medium"}`.
Agents SDK replays complete Responses items. Direct analysis and evolution calls
use `responses.with_raw_response.parse` and preserve every output item during
correction and continuation. Requests use `store=false`; native reasoning content
remains in local traces and is replayed on subsequent requests.
The shared request
semaphore is 256, including nested agents, judges, embeddings and replacement token counting. HTTP connect timeout
is 60 seconds; read/write/pool timeout is 100000 seconds; pools permit 1000 connections
and 256 keepalive connections. Cloud calls omit sampling parameters and token limits.
Weave uses the Agents SDK processor and rollout roots, with local traces retained.
Streaming requests include usage and preserve the provider's raw usage payload.
Multimodal tool results remain in native Responses function_call_output items,
retaining tool IDs and ordered text/images; canonical histories retain the original results.
JPEGs are generated in memory from saved frame JSON recipes and original PNGs;
the harness no longer writes JPG files. Analysis and replacement token counting
regenerate images from those recipes, with PNG paths relative to the recipe directory.
Each tool result exposes an integer tool_id. replace_tool(tool_id, replacement) changes
the model's context while preserving original outputs and replacement calls in the trace
for Trace2Skill analysis. Its description recommends replacing the latest result for KV
cache reuse and explains removing wrong-call output and summarizing redundant evidence.
Resume preserves this new run's rollouts; no Luna trajectories are imported.
W&B records completed-rollout mean generated
tokens (including reasoning), model/tool calls, mean input tokens and peak context tokens,
including retries and nested agents but excluding judge from trace length. Throughput is
completion-accounted input/output tokens and responses per second over 60 seconds,
including judge; input-token throughput includes cached tokens.
W&B also records `initial/average_rollout_time_seconds` and
`evolved/average_rollout_time_seconds`: mean elapsed QA time per completed rollout,
including request queueing, tools, nested agents, compaction and timed retry attempts,
excluding judge and process downtime. Per-attempt durations are saved for resume.
W&B records `initial/mean_total_context_saved_tokens` and the corresponding evolved
metric: sum each rollout's replacement token differences, then average across completed
rollouts. The proxy adapts SDK input-token counting to vLLM's native Responses conversion
and multimodal tokenizer, including text and images; repeated replacements
compare against the previous replacement, and negative savings remain negative.
`initial/replace_tool_calls` and its evolved counterpart count all completed replace_tool
invocations across completed rollouts, including returned errors and retry attempts,
excluding judge. Counting runs after QA timing and saves each before/after count for resume.

Generative calls use local PD, separate encoder and EPD proxy.
Embeddings retain `qwen3-vl-embedding-8b` to match the existing index.
The harness keeps its 314573 context budget, 178000 remaining-token compaction
threshold, 480p frames and cumulative 650-image allowance per clip. The model processor's
407-token image upper bound leaves at least 50000 tokens for text and reasoning;
actual context accounting uses API usage.
Compaction remains the harness's text handoff, without Responses compaction items.

Start the encoder, embedding, PD and proxy roles in foreground exec sessions via
`infra/trace2skill_service.py`; the current allocation is PD=0,1,2,3, encoder=4,
embedding=5, with embedding port 8014. Check resource availability at that time,
without waiting for or stopping other tasks. Full
repository review precedes the formal run. Formal traces provide correctness evidence;
no syntax, smoke or preflight tests are used. Stop and manually remove this run's
containers after inspecting the final artifacts and logs.
