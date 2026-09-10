---
name: sql-schema
description: Use when writing SQL or joining memory tables and evidence links.
---

# Introduction

The database contains normalized textual memory, hierarchical chains, visual-clip descriptions, and evidence links.

# When to use

Read this skill before complex joins, aggregations, or evidence tracing.

# Main content

With question=null, run_sql returns at most 100 raw rows and shortens long text cells. Supplying question optionally requests a separate QA answer from the complete query result with supporting row IDs and times.

Every query, including QA calls, must restrict its result with a relevant outer WHERE, a finite LIMIT, HAVING/QUALIFY, sampling, or a compact global COUNT/MIN/MAX/SUM/AVG aggregation. Filter to the requested day/time, entity, or row IDs before retrieving detailed text. ORDER BY alone and the QA question are not restrictions. Unrestricted queries are rejected before execution and must be rewritten.

Available tables and columns:

- episodes(row_id, start_s, end_s, granularity, text); granularities include 30sec, 3min, 1h, 6h, video, and scale.
- entities(row_id, name, normalized_name, first_seen_s, last_seen_s, frequency).
- semantic_triples(row_id, caption_30sec_row_id, snapshot_s, subject, predicate, object, text).
- topic_chains(row_id, entity, has_lifecycle, level, scope_id, start_s, end_s, summary).
- topic_facts(row_id, chain_row_id, caption_30sec_row_id, time_s, fact).
- event_chains(row_id, name, chain_type, level, scope_id, start_s, end_s, summary).
- event_steps(row_id, chain_row_id, caption_start_row_id, caption_end_row_id, step_index, start_s, end_s, description).
- visual_clips(row_id, start_s, end_s, text).
- object_states(row_id, episode_3min_row_id, caption_30sec_row_id, start_s, end_s, object, status, text).
- graph_edges(row_id, src_row_id, dst_row_id, edge_type, weight).
- topic_fact_evidence(row_id, fact_row_id, caption_30sec_row_id).
- event_step_evidence(row_id, step_row_id, caption_30sec_row_id).

Join facts and steps to their chains through chain_row_id. Join precise evidence to episodes.row_id through caption row IDs. For object states, episode_3min_row_id gives surrounding context and caption_30sec_row_id gives the precise evidence interval. Times are seconds on one continuous logical clock.
