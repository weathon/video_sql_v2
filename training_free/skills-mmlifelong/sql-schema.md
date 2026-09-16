---
name: sql-schema
description: Use when writing SQL or joining memory tables and evidence links.
---

The database contains the complete MM-Lifelong corpus for every question. All node and chain IDs are namespaced by video. video_id identifies the broadcast. start_ts/end_ts are seconds within that video, including zero; timestamps from different videos are separate timelines.

Node tables entity, episode, semantic and visual_clip contain id, video_id, node_type, text, start_ts, end_ts, metadata. metadata preserves native fields: episode granularity and video_path; entity normalized_name and frequency; semantic subject/predicate/object; visual_clip video_path/start_sec/end_sec. Read metadata->>'granularity' as text.

topic_chains has id, video_id, entity, has_lifecycle. topic_facts has id, video_id, chain_id, time, fact, location, start_ts, end_ts. event_chains has id, video_id, name, key_entities. event_steps has id, video_id, chain_id, step_index, start_time, end_time, description, start_ts, end_ts. Fact and step time strings are HH:MM:SS within their video. Join them through chain_id.

topic_entities and event_entities link chain_id to entity_id. Graph edge tables contain src_id, dst_id, edge_type, weight: mentioned_in links entity to episode; appeared_in links entity to visual_clip; has_property links entity to semantic; temporal_next and contains link episodes; co_clip_visual_episode and co_clip_episode_visual preserve both source directions. Use these explicit edges rather than inventing evidence relationships.

Every run_sql query must restrict its result with a relevant outer WHERE, finite LIMIT, HAVING/QUALIFY, sampling, or compact global aggregation. ORDER BY alone and an optional QA question are not restrictions. With question=null, at most 100 raw rows are returned and long text cells are shortened. Supplying question requests QA over the complete scoped result and consumes one shared model call.

semantic_search builds semantic_hits with every column of the chosen table plus cosine_sim, start_s and end_s. These seconds equal start_ts/end_ts. Pass the returned video_id, start_s and end_s to clip_qa for targeted visual verification.
