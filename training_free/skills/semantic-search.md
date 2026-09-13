---
name: semantic-search
description: Use when exact SQL terms are unknown or a concept must be located from meaning.
---

# Introduction

Semantic search retrieves rows whose embedded meaning resembles a query.

# When to use

Use it for paraphrases, uncertain names, visual descriptions, actions, and broad candidate discovery.

# Main content

Use semantic_search(query, n, pre_sql, post_sql) when a clue must be matched by meaning. It uses MAGIC PPR with text, BM25, and video seeds; it is not a default first step. Use run_sql for direct SQL filters, joins, and aggregates. n is the maximum number of PPR hits before post SQL; multiscale quotas may return fewer.

pre_sql selects exactly one column named id from the question database, including memory_nodes (the union of entity, episode, semantic, and visual_clip). Selected IDs limit both seeds and graph propagation. Use it for a known hard scope such as yesterday. post_sql instead queries only ppr_hits after retrieval to select, order, project, or aggregate useful semantic matches. Usually choose one SQL stage according to purpose; both are available when necessary. Use JSON null for an unused stage. Load sql-schema before writing either SQL stage.

The returned node fields are id, node_type, text, start_ts, end_ts, start_s, end_s, metadata, rank, and score unless post_sql selects different columns. start_s/end_s are source seconds for clip_qa. Only nodes permitted by this question's database participate. Topic facts and event steps remain available through run_sql. Inspect actual evidence and times; a PPR score ranks candidates and is not proof of correctness. This tool returns raw evidence without an extra LLM QA call.
