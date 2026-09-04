# INCIDENT: timeline isolation leak

## What happened

During the local Qwen rollout, the agent generated a `sql_qa` subquestion containing the phrase `non-month videos` while answering a Month training question.

This showed that the model-visible world was not isolated to the current timeline. The model should not merely be told to search inside Month; it should be unable to know that other timelines, videos, or splits exist.

## Why it happened

The invalid implementation exposed cross-timeline concepts through the agent-facing surface:

- the prompt included `timeline: month`, implying timeline is a variable dimension;
- the schema exposed `video_id`, letting the model reason about non-current videos;
- tool arguments and results exposed `video_id` values;
- skill text mentioned day/week/month separation;
- `run_sql` and `sql_qa` accepted arbitrary SQL against the full database view.

Because those concepts were visible, the model could invent a check like `non-month videos` even when the task was supposed to be Month-only.

## Correct interpretation

This is not primarily a missing `WHERE video_id = 'month'` bug. A hard SQL filter would prevent wrong rows from being returned, but it would still leak the existence of other timelines if the schema, prompt, tool API, or skill text mentions them.

The required fix is complete model-facing isolation: create three separate databases for day/week/month, and make each run connect only to its own database. Prompt, schema, tool parameters, tool returns, and skill content must not reveal other timelines/videos/splits by name, count, absence, or queryability.

## Required remediation

Rebuild the harness and skills from the deleted state with separate databases from the start:

1. Do not include `timeline: month/day/week` in the question prompt.
2. Do not expose `video_id` or split identifiers in agent-facing schema or tool returns.
3. Make `run_sql` and `sql_qa` execute only against the current run's physical database, not a shared full database or filtered view.
4. Make `semantic_search`, `sample_video`, and `clip_qa` operate on opaque current-world clip handles or hidden internal IDs, not user-visible cross-timeline video IDs.
5. Rewrite initial skills so they never mention day/week/month, other timelines, cross-timeline comparison, or filtering by video identity.
6. Treat any generated trace mentioning non-current timelines/videos/splits as evidence that isolation failed and the run is invalid.

## Lesson

Isolation is a visibility property, not only an execution constraint. If the model can name the hidden world, the experiment has already leaked it.
