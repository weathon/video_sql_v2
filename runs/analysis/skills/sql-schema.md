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

Available tables and columns:

- `entity`, `episode`, `semantic`, and `visual_clip` each have `(id, node_type, text, start_ts, end_ts, metadata)`.
- `episode.metadata` has `granularity` (`30sec`, `3min`, `10min`, `1h`), `video_path`, `date`, `start_time`, and `end_time`; `entity.metadata` has `normalized_name` and `frequency`; `semantic.metadata` has `subject`, `predicate`, and `object`; `visual_clip.metadata` has `video_path`, `date`, `start_time`, and `end_time`.
- `topic_chains(id, entity, has_lifecycle)` and `topic_facts(id, chain_id, time, fact, location)`.
- `event_chains(id, name, key_entities)` and `event_steps(id, chain_id, step_index, day, start_time, end_time, description)`.
- `topic_entities(chain_id, entity_id)` and `event_entities(chain_id, entity_id)`.
- `mentioned_in`, `appeared_in`, `has_property`, `temporal_next`, `contains`, `co_clip_visual_episode`, and `co_clip_episode_visual` are explicit edge tables with `(src_id, dst_id, edge_type, weight)`.

Join facts and steps through `chain_id`. Use explicit edge tables for evidence relationships; there are no inferred caption links or object-state rows. Node times are DHHMMSSFF, where `D` is the day and `FF` is the source frame number within a second. The database is already filtered to the question's query time. `semantic_hits.start_s` and `end_s` are the values for `clip_qa`; do not pass native integer timestamps as seconds.
