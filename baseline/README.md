# QA harness + Trace2Skill baseline

This worktree implements the approved baseline, independently of the project's skill trainer.
The upstream source is pinned to Qwen-Applications/Trace2Skill commit
`3d0b52a140f002a512930252b613c49048f7d5ac`. Its mandatory skill validator comes from
anthropics/skills blob `ed8e1dddce77b16af13c6f36b3fe86c4ac7c590c`.

Run `uv run python -m src.trace2skill_baseline` from this worktree after starting its
dedicated infrastructure. It collects all 266 month training questions with five
independent rollouts each, analyzes score 4–5 as success and 0–3 as failure, performs
one upstream Combined evolution, then collects the same 1,330 rollouts with the frozen
evolved skill. Results, complete traces/media, intermediate patches and comparison
are written under `runs/trace2skill-baseline`. Completed rollout, analysis and evolution
LLM artifacts are reused on resume; failures exhaust their applicable correction
opportunities and raise. No held-out evaluation is started.

The seven initial Markdown files are copied byte-for-byte. Only the directory bridge
adds frontmatter and reference links to upstream SKILL.md. The six subskills retain
their contents and on-demand read_skill(name) catalog. Evolution retains upstream
parallel proposals against one frozen snapshot, hierarchical merge, translation,
programmatic application and three-round verification. The bridge preserves those
stages while requiring valid structured Responses output and rejecting silent fallback.
The upstream spreadsheet failure-analysis prompt is adapted in error_analysis_qa.txt
to evidence search, minimal answer correction and the original 0–5 QA judge.

All model calls use Qwen/Qwen3.8-27B-FP8 and medium reasoning. The shared request
semaphore is 128, including nested agents, judges and embeddings. HTTP connect timeout
is 60 seconds; read/write/pool timeout is 100000 seconds; pools permit 1000 connections
and 256 keepalive connections. The EPD proxy has no short total request deadline.
Weave uses the Agents SDK processor and rollout roots, with local traces retained.

Infrastructure uses TP4 on PD GPUs 0,1,2,3, encoder GPU 4 and embedding GPU 4, with separate
trace2skill containers, network, ports and EC tmpfs volume. Context is 314573 with
YaRN factor 1.2, thinking and preserved reasoning. The 480p image budget is 650:
Qwen's patch size 16 and merge size 2 round 854×480 to 864×480, yielding 405 visual
tokens plus two delimiters; 50000 tokens are reserved for text/reasoning.

The researched configuration uses 32768 batched tokens, MTP1 and FlashInfer autotune;
these are deployment candidates, not a measured throughput claim. CPU eager KV offload,
BF16 KV, prefix cache, async scheduling and 4GiB MM processor cache remain enabled.
The user selected max-num-seqs 50 for TP4. The PD service performs native memory
profiling for this configuration and lets vLLM schedule requests when KV is constrained.
Its profiler records the exact available KV budget for each rank after non-KV and
CUDA-graph deductions; server_info reports the exact resident concurrency capacity.
The earlier TP2 budget is not reused. CPU offload space is not included in resident
capacity. No inference preflight is sent.

Services run in foreground exec sessions via infra/trace2skill_service.py. Full
repository review precedes the formal run. Formal traces provide correctness evidence;
no syntax, smoke or preflight tests are used. Stop and manually remove this run's
containers after inspecting the final artifacts and logs.
