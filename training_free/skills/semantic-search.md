---
name: semantic-search
description: Use when exact SQL terms are unknown or a concept must be located from meaning.
---

# Introduction

Semantic search retrieves rows whose embedded meaning resembles a query.

# When to use

Use it for paraphrases, uncertain names, visual descriptions, actions, and broad candidate discovery.

# Main content

Prefer sql_ppr_search(query, n, pre_sql, post_sql) when a natural-language clue benefits from connected graph evidence. MAGIC combines text, BM25, and video seeds, runs PPR, reranks, and applies its multiscale quotas; n is the maximum hit count before post SQL. Use pre_sql to restrict candidate nodes and propagation to a known scope, or post_sql to refine the returned evidence. Usually one SQL stage is sufficient. No implicit LLM QA call is made. The original semantic_search below remains useful for searching a specific table, including topic_facts and event_steps that are stored separately from graph nodes.

Choose the table whose searchable content best matches the question: episode.text, entity.text, semantic.text, topic_facts.fact, event_steps.description, or visual_clip video embeddings. Search with a compact description of one clue at a time in query. With question=null, return raw hits; supplying question optionally requests a separate QA answer with supporting IDs and times and consumes one shared investigation model call. Only IDs present in this question's filtered database participate in ranking. Inspect returned evidence and time, then use SQL to recover nearby context and explicit relationships. A cosine score ranks candidates but is not evidence of correctness. If the top range is unhelpful, request a later rank range or reformulate the semantic clue; do not repeatedly request an overlapping range without a reason.
