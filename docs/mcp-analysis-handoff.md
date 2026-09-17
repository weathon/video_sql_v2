# MCP server and analysis harness handoff

## Entry points and current state

Repository: `/raid/wg25r/video_sql_trace2skill`. Use its `.venv_baselines/bin/python` environment; the installed Python Codex SDK is `openai-codex` 0.154.0.

| File | Responsibility |
| --- | --- |
| `src/mcp.py` | Stateful stdio MCP server exposing the existing retrieval tools; no main agent loop. |
| `src/prompt.md` | MCP server instructions, complete database schema, time conventions, and original skill routing. |
| `src/harness.py` | Shared tool implementations, QA requests, source-frame reconstruction, and the older OpenAI Agents SDK answering loop. The MCP server imports the tools and context, not that loop. |
| `src/analysis.py` | Codex SDK agents analyze saved main/branch trajectories, then a separate agent synthesizes their experiences into skills. It does not answer the original questions as a new rollout. |
| `src/run_analysis.py` | Starts or reuses model services, waits for readiness, then runs all Week batches through analysis with four workers. |

Analysis and synthesis now use `gpt-5.6-terra`, reasoning effort `high`, service tier `default` (normal speed). The tool-side QA model remains `Qwen/Qwen3.8-27B-FP8`; changing the analysis model does not change that service.

The stopped Sol run is preserved at `/raid/wg25r/video_sql_trace2skill/runs/analysis-sol-backup-20260917T044046Z`: 129 nonempty experiences and 542 files including manifests, events, prompts, and service/run logs. It was moved intact from `runs/analysis`; absolute paths embedded in old records still name that original location. Its MCP artifacts remain under `runs/mcp/`. Terra starts with a fresh `runs/analysis`, so it does not reuse Sol experiences. Original Week rollouts are unchanged under `runs/mmlifelong-week/collection/questions/` (200 `batch.json` files).

## Starting and connecting to MCP

The MCP transport is stdio, not an HTTP endpoint. A Codex client launches the configured Python process; that process connects to the two HTTP model services. `src/mcp.py` checks both `/models` endpoints before completing initialization. Missing services therefore prevent the required MCP server from initializing.

The existing registration in `/home/wg25r/.codex/config.toml` is:

```toml
[mcp_servers.video_sql]
command = "/raid/wg25r/video_sql_trace2skill/.venv_baselines/bin/python"
args = ["-m", "src.mcp"]
tool_timeout_sec = 100000

[mcp_servers.video_sql.env]
PYTHONPATH = "/raid/wg25r/video_sql_trace2skill"
VIDEO_SQL_QA_URL = "http://127.0.0.1:8310/v1"
VIDEO_SQL_EMBEDDING_URL = "http://127.0.0.1:8214/v1"
HF_HOME = "/raid/huggingface"
OMP_NUM_THREADS = "10"
MKL_NUM_THREADS = "10"
OPENBLAS_NUM_THREADS = "10"
NUMEXPR_NUM_THREADS = "10"
```

Expected served models are `Qwen/Qwen3.8-27B-FP8` on 8310 and `qwen3-vl-embedding-8b` on 8214. This registration already exists; another agent does not need to reinstall it.

The analysis client applies these per-client overrides in `CodexConfig`, without changing global configuration:

```python
config_overrides=(
    "mcp_servers.video_sql.required=true",
    'mcp_servers.video_sql.default_tools_approval_mode="approve"',
    "features.use_legacy_landlock=true",
)
```

Its `thread_start` uses `Sandbox.read_only` and `ApprovalMode.deny_all`. Read-only permits Python to read JSON and perform in-memory analysis. The explicit MCP approval authorizes the requested tools while shell/file escalation stays disabled. Without it, actual calls failed with `MCP tool call requires approval, but approval policy is never`. Legacy Landlock is a verified workaround for this host's failing bwrap setup in the pinned Codex runtime; a real read succeeded and a real filesystem write was denied. Keep these permissions when reusing this setup. The MCP process itself writes its private traces and notes; the Codex shell's read-only sandbox does not prevent those server-side effects.

Each MCP process has one `MCPHarnessContext`, its own notes, counters, and UUID artifact directory. Separate investigations need separate MCP processes/contexts if their notes must be isolated. There is no tool for resetting a context. Its QA semaphore is per process, not a shared limit across all MCP processes.

## Data and time conventions

The server currently hardcodes:

- Database: `/raid/videosql-video-artifacts/index.duckdb`, opened read-only.
- Embedding index: `/raid/videosql-video-artifacts/embeddings`.
- Media map: `/raid/videosql-video-artifacts/media-map.json`; source paths resolve relative to its parent, including `videos/`.
- Original skills: `training_free/skills-vector`.
- Seven-day EgoLife mode, `query_time=None`, ordinary vector retrieval, and video-mode `clip_qa`.

There is no question-time cutoff. Database timestamps use EgoLife `DHHMMSSFF`, not seconds; `topic_facts.time` and `event_steps` retain their source strings. `clip_qa` uses seconds since DAY1 00:00:00. Vector search returns `start_s` and `end_s` for this conversion. This server is not configured for Month/per-video timestamp mode.

The schema and tool descriptions are the live contract. SQL natural-language text is casefolded; identifiers and source time strings retain their case. Scope SQL to the relevant time/entity/IDs or a finite result. `ORDER BY` alone does not satisfy the current `run_sql` scope check.

## Exposed tools

Descriptions and JSON schemas come directly from `make_tools(context)` in `src/harness.py`; retrieve them with MCP `list_tools` instead of maintaining another tool schema copy.

| Tool | Parameters and behavior |
| --- | --- |
| `run_sql` | `sql`, optional `question`. One scoped read-only DuckDB query. `question=null` returns up to 100 raw rows with long text cells shortened; a question invokes Qwen QA on the complete scoped result. |
| `semantic_search` | `query`, `table`, `sql`, optional `question`. Scores all indexed rows in the chosen table by cosine similarity, then executes SQL over `semantic_hits`. Tables: `episode`, `entity`, `semantic`, `topic_facts`, `event_steps`, `visual_clip`. Use SQL for filtering, ranking, or aggregation. Raw/QA behavior matches `run_sql`. This server does not expose the older PPR signature. |
| `clip_qa` | `start_s`, `end_s`, `fps`, `question`, optional `video_id`. Targeted verification of an already located interval, at most 20 minutes and 650 sampled frames per call; video mode requires at least two frames. Returns answer/unknown, actual evidence times, frame recipes, and evidence images. |
| `add_note` | `note`. Appends a private investigation note. |
| `read_notes` | No parameters. Returns that context's notes. |
| `read_skill` | `name`. Returns the full original skill Markdown. |
| `agent_raise` | `reason`. Returns an MCP error representing an investigation that must stop. |
| `submit_answer` | `answer`. Returns JSON containing the final answer. The MCP server itself keeps running. |

`replace_tool` is deliberately absent: MCP cannot remove or rewrite the client's already received history. Tool descriptions remain inline with their implementations; the main instructions are in `src/prompt.md` and exposed through MCP initialization.

Every call's content starts with a JSON `tool_id` header. Most outputs then contain JSON text; skills contain Markdown, and `clip_qa` has multiple text/image content blocks. Preserve these blocks and inspect returned errors rather than assuming every text response is successful. Some recoverable argument/query errors are JSON error text; `agent_raise` sets MCP `isError`; dependency/code errors are logged and re-raised.

The MCP override of `reserve_model_call` only counts and logs QA calls. It does **not** enforce the older answering loop's shared 60-call investigation budget or the subsequent forced final answer. SQL/semantic QA retain the existing 16,384-token cap and truncation retries; that exception is not a Codex analysis input/output cap.

## Integrating a new question-answering harness

The next agent's work is the client-side answering loop over this MCP server. Reuse the connection, schemas, instructions, and logging approach above; `src/analysis.py` is a Codex SDK usage example, not the answering prompt or rollout protocol.

1. Start/reuse the required model endpoints, connect a fresh MCP context for the investigation, and consume its initialization instructions plus `list_tools` schemas. Do not embed analysis or synthesis instructions into an answering agent.
2. Feed the original question, options where applicable, and ID. Saved reference answers, judge results, branches, and experience-generation inputs are for analysis/evaluation, not the question-answering agent's evidence input.
3. Drive the selected agent with the MCP tool results, preserving reasoning/history and multimodal content. Read the relevant skills through `read_skill`.
4. Have the client finish on `submit_answer` and extract its answer; stop and record the reason on `agent_raise`. The server only returns these tool results; it has no main loop to stop for the client.
5. Implement any requested rollout budget, forced final-answer step, branch snapshots, retry policy, scheduling, scoring, and result persistence in that answering harness. None of those is provided by the MCP transport wrapper. This handoff does not select a new answering-agent model or alter that experiment's protocol.

Generated skills are currently **not** loaded by the MCP server: it still points to `training_free/skills-vector`, and `src/prompt.md` embeds that catalog/root routing. A future experiment using learned skills must explicitly wire both the skill source and routing; merely producing `runs/analysis/skills` does not activate them.

## Running and resuming analysis

From the repository root:

```bash
.venv_baselines/bin/python -u src/run_analysis.py
```

The launcher validates model IDs, reuses ready endpoints, otherwise checks ports, actual idle GPUs, and host memory before starting task-owned foreground Docker services. It uses the existing fair Qwen serving settings, including TP=2, MTP, prefix caching, 4 GiB multimodal cache, and CPU KV offload. It does not use `--rm`. Normal cleanup stops and removes only containers it started; a reused embedding service is retained. A forced interruption can leave a container behind, so verify the exact task container before manual removal. Do not stop a shared service to free resources.

After both services are ready, the launcher passes every `*/batch.json` below `runs/mmlifelong-week/collection/questions/` to `src.analysis`, using `runs/analysis` as output. To select batches when services are already ready:

```bash
.venv_baselines/bin/python -u -m src.analysis /absolute/path/to/questions/0/batch.json --output runs/analysis-debug
```

There is one shared `Codex` client and four analysis workers. Each batch gets its own Codex thread. All JSON paths beneath that batch directory are written to `analyses/<batch>/json_paths.json`; the prompt includes only the batch and manifest paths. The agent loads the manifest with Python and reads necessary JSON fields. This avoids the SDK's 1,048,576-character input rejection without truncating the file list or trace evidence.

Each analysis compares the main trajectory and branches from their shared prefix, examines real tool inputs/outputs, and may use MCP to check a hypothesis. It returns one short English paragraph with dos/don'ts, conditions, limitations, and supporting trace references. The updated prompt requires activity-independent lessons: for example, “doing repeated events across days” instead of “shopping across days.” Concrete references support the lesson without limiting it to a question's scene, objects, or people.

An existing `experience.md` is reused as complete. A sample exception is logged to `errors.log` with its traceback and printed as `ANALYSIS_SKIP`; other samples continue. There is no analysis-level retry loop. A later invocation retries samples without an experience file. This resume behavior does not detect model/prompt changes, which is why the Sol output was moved before starting Terra.

After analysis futures finish, a separate Codex thread reads the successful experience files and original skills, then returns a structured `LearnedSkills` bundle. The caller writes individual Markdown files with YAML `name`/`description` and a `ROOT-SKILL.md`. Synthesis errors still fail the run; they are not sample skips. No experiences means synthesis is not started. A cached synthesis bundle is reused when its recorded input path list matches; content changes at the same paths are not detected by that comparison.

Both stages use Terra/high/normal, retain persistent sessions (`ephemeral=False`), and call `thread_archive` in cleanup. This preserves Codex session records while archiving finished tasks; events also remain on disk. Abrupt process termination can prevent the archive call from completing.

## Artifacts and verification

| Path | Contents |
| --- | --- |
| `runs/analysis/analyses/<batch>/experience.md` | Completed per-batch experience used for resume and synthesis. |
| `runs/analysis/analyses/<batch>/json_paths.json`, `prompt.txt`, `events-*.jsonl` | Full path manifest, compact prompt, and request/thread/tool/response/error/archive events. |
| `runs/analysis/errors.log` | Whole-sample failures and tracebacks. |
| `runs/analysis/synthesis/` | Synthesis events, input paths, and structured `skills.json`. |
| `runs/analysis/skills/` | Learned small Markdown skills and root routing guide. |
| `runs/analysis/services/` | Logs for task-started model containers. |
| `runs/mcp/<uuid>/traces/` | Actual MCP arguments/results/errors, QA/wire requests, and embedding requests/responses. Server stderr prints this directory at readiness. |
| `runs/mcp/<uuid>/media/` | JSON frame recipes and sampling diagnostics. |

Tool results carry images to the client, but harness artifacts store source video path, PTS, time base, crop, render settings, and timestamps instead of persisted JPEG/PNG/base64 payloads. `frame_jpeg_bytes` in `src/harness.py` rebuilds an image from a recipe. Analysis events sanitize image/video data. This describes the harness artifacts, not an independently audited storage guarantee for Codex's own internal session format.

The previous Sol run verified Python manifest reads and successful MCP `read_skill`, `run_sql`, `semantic_search`, and `clip_qa` calls after the approval fix. Terra execution must be checked in its own new events: inspect actual tool arguments and results, experience contents, final synthesis, and archive responses. A progress count or output-file existence alone is insufficient. Keep long jobs in controlled foreground sessions and reduce monitoring to the repository's required interval after stable startup.

`runs/` is ignored. Preserve raw records there; commit source and handoff documents, and only compact final results/deliverables when applicable. Do not commit per-sample traces, bulk outputs, videos, embeddings, or databases.
