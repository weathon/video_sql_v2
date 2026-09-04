---
name: sql-schema
description: Use this whenever writing SQL, joining evidence tables, tracing rows to videos, or checking which Stage 2 table contains a fact.
---

# SQL Schema

## Introduction

The system prompt contains the live DuckDB schema. Use this skill to choose tables and joins.

## When to use

Use it before nontrivial SQL, when a query returns unexpected rows, or when you need 30 second evidence for a chain or object state.

## Main content

1. Use `episodes` for text at `30sec`, `3min`, `30min`, `1h`, `6h`, `video`, and `scale` granularities. Filter `granularity` explicitly when the level matters. **Reason:** summary levels can answer context questions but only 30 second rows are direct evidence.
2. Use `visual_clips` for actual clip filenames and source offsets. Do not invent paths from row IDs. **Reason:** regenerated Week rows and existing video embeddings are tied by stored filenames.
3. For object state questions, join `object_states.caption_30sec_row_id` to `episodes.row_id` for the exact evidence and `episode_3min_row_id` for surrounding context. **Reason:** the 3 minute row is context, not the exact state occurrence.
4. For topic or event chain evidence, search `topic_facts` or `event_steps`, then follow `topic_fact_evidence` or `event_step_evidence` back to 30 second episodes. **Reason:** chain summaries can hide which moment actually supports the answer.
5. Keep SQL read-only. Prefer explicit selected columns: row IDs, video IDs, start/end seconds, granularity, and text. **Reason:** those fields make later visual verification possible.

```sql
SELECT e.row_id, e.video_id, e.start_s, e.end_s, e.granularity, e.text
FROM episodes AS e
WHERE e.video_id = 'month'
  AND e.granularity = '30sec'
  AND e.text LIKE '%phone%'
ORDER BY e.start_s
LIMIT 20;
```

