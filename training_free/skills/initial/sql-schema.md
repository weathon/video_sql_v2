---
name: sql-schema
description: Use when writing SQL or joining memory tables and evidence links.
---

# Introduction

The database contains multi-granularity episode captions, named entities, knowledge-snapshot triples, visual-clip descriptions, graph edges, and topic/event chains.

# When to use

Read this skill before complex joins, aggregations, or evidence tracing.

# Main content

With question=null, run_sql returns at most 100 raw rows and shortens long text cells. Supplying question optionally requests a separate QA answer from the complete query result with supporting row IDs and times.

Every query, including QA calls, must restrict its result with a relevant outer WHERE, a finite LIMIT, HAVING/QUALIFY, sampling, or a compact global COUNT/MIN/MAX/SUM/AVG aggregation. Filter to the requested day/time, entity, or row IDs before retrieving detailed text. ORDER BY alone and the QA question are not restrictions. Unrestricted queries are rejected before execution and must be rewritten.

Available tables and columns:

- episode(id, granularity, date, start_time, end_time, start_ts, end_ts, start_s, end_s, text, video_path); granularities include 30sec, 3min, 10min, and 1h.
- entity(id, name, normalized_name, frequency, start_ts, end_ts, start_s, end_s).
- semantic(id, snapshot_ts, snapshot_s, index_in_snapshot, subject, predicate, object, text, text_id); each knowledge snapshot repeats all triples known at that time.
- semantic_texts(text_id, text, subject, predicate, object, first_snapshot_ts, last_snapshot_ts, first_snapshot_s, last_snapshot_s, snapshot_count); one row per distinct triple.
- visual_clip(id, date, start_time, end_time, start_ts, end_ts, start_s, end_s, video_path, text).
- graph_edges(src_id, dst_id, edge_type, weight); edge types co_clip, contains, temporal_next, mentioned_in, appeared_in, has_property.
- topic_chains(entity, has_lifecycle, fact_count, first_time, last_time, start_s, end_s).
- topic_facts(fact_id, entity, time, time_ts, time_s, fact).
- event_chains(chain_id, name, key_entities, step_count).
- event_steps(step_id, chain_id, name, step_index, day, start_time, end_time, start_ts, end_ts, start_s, end_s, description).

Join facts to their chains through entity and steps to their chains through chain_id. Join a distinct triple to its snapshots through text_id. Join entities to the 30-second episodes that mention them through graph_edges with edge_type mentioned_in, and 30-second episodes to their visual clips through co_clip. Columns ending in _ts and the date/start_time/end_time/day/time columns are wall-clock values (DHHMMSSFF: day digit, hour, minute, second, frame; 111094300 is DAY1 11:09:43). Columns ending in _s are seconds on one continuous logical clock and are the times used by clip_qa.
