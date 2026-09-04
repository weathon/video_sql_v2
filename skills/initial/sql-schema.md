---
name: sql-schema
description: Use when writing DuckDB queries or joining indexed facts back to source evidence.
---

# Introduction

The schema separates timelines, summaries, facts, chains, state observations, and physical video mappings.

# When to use

Read this skill before nontrivial SQL, especially joins across chains, evidence tables, or source videos.

# Main content

- `episodes` contains `30sec`, `3min`, `1h`, `6h`, `video`, and `scale` text. Join by `row_id`; filter by `video_id` and `granularity`. **Reason:** Granularity determines whether a row is evidence or context.
- `topic_facts` joins `topic_chains` through `chain_row_id`; `topic_fact_evidence` maps facts to actual 30-second episode IDs. `event_steps`, `event_chains`, and `event_step_evidence` follow the same pattern. **Reason:** Chain summaries must not be treated as direct evidence.
- `object_states.episode_3min_row_id` supplies outer context and `caption_30sec_row_id` supplies the exact observation. **Reason:** Keeping both keys prevents imprecise state attribution.
- `semantic_triples.caption_30sec_row_id` points directly to a 30-second episode. **Reason:** Subject-predicate-object facts remain traceable to their source interval.
- `source_videos` maps logical seconds to a physical `filename` with `timeline_start_s` and `timeline_end_s`; `visual_clips` already stores both logical and source offsets. **Reason:** Physical filenames cannot be inferred from row IDs.
- Use explicit join keys and inspect a few real rows before building aggregations. **Reason:** Similar column names across tables do not imply interchangeable identifiers.

```sql
SELECT f.time_s, f.fact, e.start_s, e.end_s, e.text
FROM topic_facts AS f
JOIN topic_fact_evidence AS x ON x.fact_row_id = f.row_id
JOIN episodes AS e ON e.row_id = x.caption_30sec_row_id
WHERE f.video_id = 'month'
ORDER BY f.time_s;
```
