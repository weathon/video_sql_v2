# Incident: Pilot answer leakage through `run_sql`

## What happened

The Pilot main retrieval agent could execute arbitrary read-only SQL against the same DuckDB that stored evaluation gold answers. The `questions` table exposed its `answer` column through `run_sql`. Read-only database access prevented writes but did not isolate gold labels from the agent.

The first question in the fixed order was question 277. Rollout 0 executed SQL that selected `questions.answer`, received `13`, answered `13 push-ups`, and was scored 5. Its seven `clip_qa` calls returned conflicting counts: `unknown`, `0`, `0`, `8`, `0`, `6`, and `1`. None independently supported 13. The saved reflection explicitly said the final response followed the exact question row despite unresolved visual evidence.

## Impact

The Pilot was stopped with exit code 130 after 139 result files and 27 question summaries had been written. A direct inspection of all 140 saved main-agent traces found SQL reading an `answer` column in 139 traces. The generated Pilot scores, reflections, summaries, updater outputs, and learned skill snapshots are contaminated and must not be used as experimental results.

## Why it was missed

- Gold answers and retrieval data were stored in the same database exposed to the main agent.
- `run_sql` allowed arbitrary `SELECT` statements without a gold-column boundary.
- Read-only access was incorrectly treated as sufficient isolation.
- The first real rollout trace was not manually inspected before the larger Pilot run continued.
- Existing tests checked read-only behavior and SQL limits but did not test that the main agent could not retrieve gold answers.

## Required prevention

- Keep reference answers outside every database, index, schema, prompt, note, and tool available to the main retrieval agent.
- Give judge, reflection, and training orchestration access to gold answers through a separate host-side data path that is never exposed through agent tools.
- Remove gold columns from the retrieval database rather than relying on prompt instructions or SQL text filtering.
- Add an integration check that searches the agent-visible schema and real tool outputs for gold fields before any Pilot or evaluation run.
- Manually inspect the first complete rollout, including every tool input and returned value, before allowing the larger run to continue.

## Implemented remediation

- Removed `questions` and `question_clues` from the agent-visible DuckDB schema and database builder.
- Kept questions, reference answers, question types, temporal certificates, and clue intervals only in dataset-style JSON files used by host-side orchestration; only the judge and training reflection path receive the reference answer.
- Replaced `videos.video_path` with a filename-only field so SQL cannot reveal host absolute paths.
- Added schema and real Pilot database checks that reject any gold table or gold column.
- The contaminated Pilot run, learned skill, summaries, scores, resume artifacts, and W&B run must not be resumed; the Pilot restarts from the initial skills in a new run directory and W&B run.
