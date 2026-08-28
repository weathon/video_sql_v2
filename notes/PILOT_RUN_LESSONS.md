# Pilot Run Lessons and Formal-Run Handoff

## Current checkpoint

- The current valid Pilot workspace is `runs/pilot_luna` and the clean database is `data/pilot.duckdb`.
- The run has 99 completed training rollout `result.json` files, 19 question summaries, and 3 batch skill snapshots.
- W&B run: `https://wandb.ai/3dsmile/video-sql-v2/runs/c1sjjmam`.
- The only incomplete rollout is `train/question-527/rollout-1`.
- Its three attempts are already consumed: attempt 1 ended in a nested streamed base `openai.APIError`, attempt 2 ended in `MaxTurnsExceeded(50)`, and attempt 3 ended in another nested streamed base `openai.APIError`.
- Resuming without a protocol decision will stop because the configured three-attempt limit is exhausted. Do not delete, rename, or ignore attempt directories to manufacture another attempt. Continuing this rollout requires an explicit decision to keep the failure or authorize a new retry budget.
- Commit `4bfa42e` recognizes an exact base `openai.APIError`, including one wrapped as the direct cause of an Agents SDK `UserError`, as retryable while excluding deterministic 4xx subclasses. This fixes future classification but does not restore the consumed attempts for question 527.

## Runs and data that must never be resumed

- `runs/pilot_luna_contaminated_answer_leak_20260827`: invalid because the retrieval agent could access gold answers.
- `runs/pilot_luna_invalid_numeric_judge_20260827`: invalid because the judge could confuse incidental numeric identifiers with numeric answer content.
- `data/pilot_answer_leakage.duckdb`: retained only as incident evidence and must not be passed to the harness.

## Gold-answer isolation

- Gold answers, questions, question types, clue intervals, and temporal certificates do not belong in the agent-facing DuckDB. Keep them in host-side JSON used only by orchestration, judging, reflection, and summaries.
- `read_only=True` is not sufficient isolation. DuckDB can otherwise read external files or attach another database through SQL.
- Every harness DuckDB connection, including `run_sql`, `semantic_search`, `sample_video`, and evidence screenshot paths, must use the shared locked configuration that disables external access.
- All concurrent DuckDB connections must use identical configuration. Mixing configured and unconfigured readers caused DuckDB's different-configuration connection error.
- Full incident details are in `notes/INCIDENT_PILOT_ANSWER_LEAKAGE.md`.

## Judge protocol

- The judge receives only `reference_answer` and `candidate_answer`. It does not receive the question, trace, tool results, evidence, or reflection.
- Judge semantic equivalence of final answers only. The trace is for the rollout's own reflection, never for scoring.
- Numeric gold answers require explicit disambiguation. Before the prompt fix, gold answer `13` could be incorrectly matched to an incidental string such as `Video 13`.
- The shared judge instructions require comparing answer content and ignoring incidental video IDs, timestamps, row IDs, and other identifiers.
- Protocol checks are saved under `runs/judge_protocol_check/`: the incidental-ID candidate scored 0 and the genuinely correct numeric candidate scored 5.

## Tool error boundaries

- Model-correctable input errors must return the exact error to the same agent as a rejected tool result so it can correct the call. Covered cases include an invalid semantic-search table, invalid crop coordinates, non-positive FPS, negative or non-increasing intervals, sampling outside the clip interval, and an `answered` clip result with missing or invalid evidence timestamps.
- Database, index, media, missing-clip, and corrupted-cache failures are system or data errors and must propagate instead of being converted into a generic tool rejection.
- Keep `failure_error_function=None`; a generic SDK error message hides the actual failure and prevents correct classification.
- A transient boundary failure is retried with the same method and settings. A deterministic code or data failure terminates the run.

## Nested and streamed model failures

- Failures raised inside the nested clip agent may be wrapped by the Agents SDK as `UserError`; inspect the direct `__cause__` when classifying them.
- A base `openai.APIError` emitted by a streamed server-sent error event is transient and retryable.
- `BadRequestError` and other deterministic request errors remain non-retryable even though they inherit from `APIError`.
- Existing retryable classes include connection, timeout, rate-limit, internal-server, malformed model output, and max-turn failures, subject to the configured rollout attempt limit.

## Trace persistence

- Both main-agent and nested-agent streaming loops save completed responses, usage, and the partial transcript in `finally` before re-raising an error.
- Persist each rollout immediately. A run directory or partial trace alone is not completion; the expected terminal result and status must exist.
- Preserve prior completed rollout results, question summaries, patches, and batch skill snapshots during resume.
- Normalize embedded image bytes out of JSON traces and retain media as separate files so traces stay readable and resumable.

## Concurrent image artifact collision

- Concurrent calls using the same interval previously wrote the same image filename while another call was reading it, producing empty base64 tool images.
- Generated contact sheets and evidence screenshots now include the full sampling parameters plus a per-call UUID.
- Model-facing tool output exposes basenames rather than absolute filesystem paths.
- Full incident details are in `notes/INCIDENT_EMPTY_BASE64_TOOL_IMAGE.md`.

## Compaction behavior

- Compaction belongs to the current main retrieval agent. Do not start a separate compactor agent.
- When the remaining context reaches the threshold, tell the current agent to produce the fixed project-specific handoff. Its next response becomes the only prior context in a fresh conversation.
- The first and second compactions continue the same rollout. After the third handoff is produced, terminate immediately with score 0, skip the judge, and run the fixed loop-focused reflection.
- Reflection does not compact. Near its context limit, stop tool use and force the final reflection and meta-reflection.

## Resume and monitoring

- Resume from actual completed artifacts; do not restart completed questions or batches and do not resume either archived invalid run.
- Do not erase failed attempt evidence to bypass the three-attempt ceiling.
- Run the process directly through the Codex foreground execution session, not tmux. Append resumed output to `runs/pilot_luna/training.log` rather than overwriting it.
- Keep terminal output bounded and inspect only short log tails. Poll after 2 minutes, then 4 minutes once stable, then use the 30-minute foreground trigger.

## Formal-run preflight

- Read `CODEX_PLAN.md`, this note, the two incident notes, and `notes/SKIP_OR_CRASH.md` before launch.
- Resolve the exhausted retry budget for question 527 explicitly before attempting to resume this Pilot checkpoint.
- Run the repository tests and inspect the actual test output; the last verified suite contained 20 passing tests.
- Perform the required full-repository Claude CLI review immediately before the formal run because small debugging commits followed the previous full review.
- Confirm the embedding service on `127.0.0.1:8012` serves `qwen3-vl-embedding-8b` and verify a real 4096-dimensional response.
- Load `LAB_OPENAI_KEY` from `.env` and export it as `OPENAI_API_KEY` for the Pilot process.
- After launch, manually inspect the first newly completed rollout: every tool input and returned value, partial and final responses, usage, judge input and score, contact sheet, and evidence screenshots.
