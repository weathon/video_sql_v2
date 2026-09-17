---
name: sql-schema
description: Use when writing SQL, tracing evidence links, joining memory tables, or computing scoped aggregates.
---

---
name: sql-schema
description: Use when writing SQL or joining memory tables and evidence links.
---

# Introduction

The database contains normalized MAGIC nodes, topic/event chains, and explicit graph edges.

# When to use

Read this skill before complex joins, aggregations, evidence tracing, or exact time-scoped record retrieval.

# Main content

Use SQL to recover a small, relevant evidence slice after candidate discovery. Filter by the requested day, time, entity, chain, or row IDs before retrieving detailed text. Query neighboring fine-grained records when a summary, topical fact, or semantic result may omit the start, completion, role, or outcome. Bind a value to its exact actor, item, and completed event before aggregating.

Every query, including QA calls, must have a relevant outer `WHERE`, finite `LIMIT`, `HAVING`/`QUALIFY`, sampling clause, or compact global `COUNT`/`MIN`/`MAX`/`SUM`/`AVG` aggregation. `ORDER BY` alone and a QA question are not restrictions. With `question=null`, `run_sql` returns at most 100 raw rows and shortens long text cells; supplying `question` optionally requests a separate QA answer.

Every question can access all 14 indexed videos. `video_id` identifies a broadcast; timestamps are seconds within that video, including fractional seconds. There is no query-time cutoff. Never compare local timestamps across videos without retaining `video_id`. Numeric video IDs give broadcast ordering.

Available tables and columns:

- `entity`, `episode`, `semantic`, and `visual_clip` each have `(id, node_type, text, start_ts, end_ts, metadata, video_id)`.
- `episode.metadata` has `granularity` (`30sec`, `3min`, `10min`, `1h`), `video_path`, `start_sec`, and `end_sec`; `entity.metadata` has `normalized_name` and `frequency`; `semantic.metadata` has `subject`, `predicate`, and `object`; `visual_clip` also identifies the exact cached interval.
- `topic_chains(id, entity, has_lifecycle, video_id)` and `topic_facts(id, chain_id, time, fact, location, video_id, start_ts, end_ts)`.
- `event_chains(id, name, key_entities, video_id)` and `event_steps(id, chain_id, step_index, day, start_time, end_time, description, video_id, start_ts, end_ts)`. `day` is NULL; source time strings use HH:MM:SS.
- `topic_entities(chain_id, entity_id)` and `event_entities(chain_id, entity_id)`.
- `mentioned_in`, `appeared_in`, `has_property`, `temporal_next`, `contains`, `co_clip_visual_episode`, and `co_clip_episode_visual` are explicit edge tables with `(src_id, dst_id, edge_type, weight)`.

Join facts and steps through `chain_id`. Use explicit edge tables for evidence relationships; there are no inferred caption links or object-state rows. IDs include video namespaces. `semantic_hits.start_s` and `end_s` are seconds local to `video_id`; pass the same `video_id` and interval to `clip_qa`.
