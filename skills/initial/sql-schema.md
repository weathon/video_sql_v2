---
name: sql-schema
description: Query the DuckDB schema and follow its evidence relationships. Use whenever a question needs exact filtering, joins, timelines, chains, source-video mapping, or retrieval of rows returned by semantic search.
---

# SQL Schema

## Introduction

The database has three logical timelines, `day`, `week`, and `month`. Logical `video_id` values identify those timelines; `source_video_id` identifies physical source files.

**Reason:** Separating logical and physical identifiers prevents joins across unrelated clocks or files.

## When to use

Use this skill before writing SQL that crosses tables, follows evidence, or maps logical time back to a playable clip.

**Reason:** The schema contains both summary rows and direct 30-second evidence, so the join path determines what a result actually proves.

## Main content

- `videos(video_id, scale, duration_s, source_count)` describes the three logical timelines.
- `source_groups(source_group_id, video_id, name, timeline_start_s, timeline_end_s)` and `source_videos(source_video_id, video_id, source_group_id, filename, duration_s, timeline_start_s, timeline_end_s)` map physical files onto a logical timeline.
- `episodes(row_id, video_id, source_video_id, start_s, end_s, granularity, text)` stores `30sec`, `3min`, `1h`, `6h`, `video`, and `scale` text levels.
- `visual_clips(row_id, video_id, source_video_id, start_s, end_s, source_start_s, source_end_s, filename, text)` maps logical intervals to physical 30-second media.
- `entities(row_id, video_id, name, normalized_name, first_seen_s, last_seen_s, frequency)` stores entity summaries.
- `semantic_triples(row_id, video_id, caption_30sec_row_id, snapshot_s, subject, predicate, object, text)` points directly to 30-second caption evidence.
- `topic_chains` and `topic_facts` hold topic summaries and facts. Join `topic_facts.chain_row_id = topic_chains.row_id`; use `topic_fact_evidence(fact_row_id, caption_30sec_row_id)` for direct evidence.
- `event_chains`, `event_chain_entities`, and `event_steps` hold event sequences. Join `event_steps.chain_row_id = event_chains.row_id`; use `event_step_evidence(step_row_id, caption_30sec_row_id)` for direct evidence.
- `object_states(row_id, video_id, source_video_id, episode_3min_row_id, caption_30sec_row_id, start_s, end_s, object, status, text)` links broader 3-minute context and the exact 30-second state evidence.
- `graph_nodes` and `graph_edges` preserve typed relations. A `co_clip` edge can connect a 30-second evidence row to its `visual_clips` row.

**Reason:** These are the actual tables and foreign-key-like columns in the current full index.

## SQL patterns

Fetch a semantic hit by its real primary key:

```sql
SELECT row_id, video_id, start_s, end_s, granularity, text
FROM episodes
WHERE row_id = ?;
```

Follow an object state to its exact caption and video clip:

```sql
SELECT s.object, s.status, e30.text, v.filename, v.source_start_s, v.source_end_s
FROM object_states AS s
JOIN episodes AS e30 ON e30.row_id = s.caption_30sec_row_id
LEFT JOIN graph_edges AS g
  ON g.src_row_id = s.caption_30sec_row_id AND g.edge_type = 'co_clip'
LEFT JOIN visual_clips AS v ON v.row_id = g.dst_row_id
WHERE s.row_id = ?;
```

**Reason:** Primary-key lookup and explicit evidence joins avoid relying on database row order or summary text as if it were direct evidence.
