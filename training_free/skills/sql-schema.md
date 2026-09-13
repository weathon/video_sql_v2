---
name: sql-schema
description: Use when writing SQL or joining memory tables and evidence links.
---

# Introduction

The database contains normalized MAGIC nodes, topic/event chains, and explicit graph edges.

# When to use

Read this skill before complex joins, aggregations, or evidence tracing.

# Main content

With question=null, run_sql returns at most 100 raw rows and shortens long text cells. Supplying question optionally requests a separate QA answer from the complete query result with supporting row IDs and times.

Every run_sql query, including QA calls, must restrict its result with a relevant outer WHERE, a finite LIMIT, HAVING/QUALIFY, sampling, or a compact global COUNT/MIN/MAX/SUM/AVG aggregation. Filter to the requested day/time, entity, or row IDs before retrieving detailed text. ORDER BY alone and the QA question are not restrictions. Unrestricted queries are rejected before execution and must be rewritten.

For semantic_search only, pre_sql returns exactly one id column from the question database; its temporary memory_nodes view unions entity, episode, semantic, and visual_clip. post_sql queries only ppr_hits(id, node_type, text, start_ts, end_ts, start_s, end_s, metadata JSON, rank, score) after PPR. Usually choose pre to constrain search scope or post to refine retrieved evidence; neither stage is required. These temporary tables do not change the stored database.

Available tables and columns:

- entity, episode, semantic, visual_clip each have (id, node_type, text, start_ts, end_ts, metadata).
- metadata is native JSON: episode has granularity (30sec, 3min, 10min, 1h), video_path, date, start_time, end_time; entity has normalized_name and frequency; semantic has subject, predicate, object; visual_clip has video_path, date, start_time, end_time. Use metadata->>'granularity' for a text value.
- topic_chains(id, entity, has_lifecycle); topic_facts(id, chain_id, time, fact, location).
- event_chains(id, name, key_entities); event_steps(id, chain_id, step_index, day, start_time, end_time, description).
- topic_entities(chain_id, entity_id); event_entities(chain_id, entity_id).
- mentioned_in: entity to episode; appeared_in: entity to visual_clip; has_property: entity to semantic; temporal_next and contains: episode to episode; co_clip_visual_episode and co_clip_episode_visual preserve both original directions.
- Every edge table has (src_id, dst_id, edge_type, weight), with foreign keys to the stated endpoint tables.

Join facts and steps through chain_id. Use only the explicit edge tables for evidence relationships; there are no inferred caption links or object-state rows. Node times are DHHMMSSFF, where D is the day and FF is the source frame number within a second. The database is already filtered to the question's query time. Semantic search additionally returns start_s/end_s in seconds since DAY1 00:00:00 for clip_qa; do not pass native integer timestamps as seconds.
