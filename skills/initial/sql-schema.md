---
name: sql-schema
description: Tables and columns available to run_sql, with example queries for exact lookup, joins and time-range filtering. Use whenever you are about to write SQL or are unsure which table holds the information you need.
---

# SQL Schema

## Introduction

`run_sql` executes one read-only DuckDB query, returns at most 100 rows and truncates each cell to 500 characters while reporting the original length. Every content table has `video_id` plus a time position, so results from different tables can be joined on `video_id` and overlapping time.

**Reason:** Knowing the query limits and shared location columns prevents false assumptions about returned evidence.

## When to use

Before writing any SQL, and whenever a query returns nothing and you need to check whether you picked the right table or column.

**Reason:** Schema consultation avoids invalid queries and searches against the wrong evidence source.

## Main content

Tables:

- `videos(video_id, filename, duration_s)`
- `episodes(row_id, video_id, start_s, end_s, granularity, text)` — narrative summaries at granularity `30sec`, `3min`, `10min`, `1h`.
- `visual_clips(row_id, video_id, start_s, end_s, text)` — descriptions of roughly 30 second visual clips.
- `entities(row_id, video_id, name, normalized_name, first_seen_s, last_seen_s, frequency)`
- `semantic_triples(row_id, video_id, snapshot_s, subject, predicate, object, text)`
- `openie_triples(row_id, episode_row_id, video_id, subject, predicate, object, text)` — extracted from episode text.
- `topic_chains(row_id, video_id, entity, has_lifecycle)` and `topic_facts(row_id, chain_row_id, video_id, time_s, fact)` — facts about one entity over time.
- `event_chains(row_id, video_id, name, chain_type)`, `event_chain_entities(row_id, chain_row_id, video_id, entity)` and `event_steps(row_id, chain_row_id, video_id, step_index, start_s, end_s, description)` — multi-step events.

Text columns are normalized: lower case, single spaces, Latin accents folded to ASCII. Write literals in lower case and match with `LIKE '%...%'` or `ILIKE`.

Examples:

```sql
-- Which videos exist and how long they are
SELECT video_id, filename, duration_s FROM videos ORDER BY video_id;
```

```sql
-- Exact keyword lookup in short summaries, ordered by time
SELECT video_id, start_s, end_s, text
FROM episodes
WHERE granularity = '30sec' AND text LIKE '%target activity%'
ORDER BY video_id, start_s;
```

```sql
-- Everything known about one entity in one video
SELECT e.name, e.first_seen_s, e.last_seen_s, e.frequency
FROM entities e
WHERE e.normalized_name LIKE '%target entity%';
```

```sql
-- Join a text hit to the visual clips that overlap it
SELECT v.video_id, v.start_s, v.end_s, v.text
FROM visual_clips v
WHERE v.video_id = 'candidate_video_id' AND v.end_s >= 0 AND v.start_s <= 60
ORDER BY v.start_s;
```

```sql
-- Steps of every event chain whose name mentions a keyword
SELECT c.video_id, c.name, s.step_index, s.start_s, s.end_s, s.description
FROM event_chains c JOIN event_steps s ON s.chain_row_id = c.row_id
WHERE c.name LIKE '%target event%'
ORDER BY c.video_id, c.name, s.step_index;
```

**Reason:** The table inventory and executable examples show how to locate, filter, join, and order the available evidence correctly.
