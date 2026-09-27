---
name: sql-schema
description: Use when writing SQL or joining memory tables and evidence links.
---

# Main content

Keyword matching here is the first choice for finding evidence; reach for semantic_search only when it returns nothing. Every question can access all 14 indexed videos. video_id identifies a broadcast; timestamps are seconds within that video, including fractional seconds. There is no query-time cutoff. Never compare local timestamps across videos without retaining video_id. Numeric video IDs give broadcast ordering.

- entity, episode, semantic, visual_clip: id, node_type, text, start_ts, end_ts, metadata, video_id.
- episode metadata includes granularity (30sec, 3min, 10min, 1h), video_path, start_sec, end_sec. visual_clip also identifies the exact cached interval. entity metadata includes normalized_name and frequency; semantic metadata includes subject, predicate, object.
- topic_chains: id, entity, has_lifecycle, video_id. topic_facts: id, chain_id, time, fact, location, video_id, start_ts, end_ts.
- event_chains: id, name, key_entities, video_id. event_steps: id, chain_id, step_index, day, start_time, end_time, description, video_id, start_ts, end_ts. day is NULL; source time strings use HH:MM:SS.
- topic_entities and event_entities have chain_id, entity_id. They preserve explicit source associations.
- mentioned_in links entity to episode; appeared_in links entity to visual_clip; has_property links entity to semantic. temporal_next and contains link episodes. co_clip_visual_episode and co_clip_episode_visual retain both original directions. Edge columns are src_id, dst_id, edge_type, weight.

Use chain_id and the explicit edge tables for joins. IDs include video namespaces. Caption text holds a scene description, an object inventory written as name: state pairs after `objects:`, and verbatim screen strings after `on-screen text:`; there are no separate object-state rows and no inferred caption links. Use metadata->>'granularity' for text access. Natural-language text is casefolded, so write LIKE patterns in lowercase.

run_sql requires a relevant outer WHERE, finite LIMIT, HAVING/QUALIFY, sampling, or compact COUNT/MIN/MAX/SUM/AVG aggregation. ORDER BY alone is not a restriction. With question=null it returns at most 100 raw rows with long text shortened; supplying question requests a separate QA answer over the complete scoped result.

semantic_search SQL queries semantic_hits: all columns of the chosen table plus cosine_sim, start_s and end_s. These seconds are local to video_id. Pass the same video_id and interval to clip_qa.
