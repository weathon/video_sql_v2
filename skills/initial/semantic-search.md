---
name: semantic-search
description: Use vector search to discover relevant indexed rows when exact wording is unknown. Read for paraphrases, fuzzy concepts, entity aliases, activities, states, and candidate video intervals.
---

# Semantic Search

## Introduction

Semantic search is candidate discovery, not final proof. It embeds the query with `Qwen/Qwen3-VL-Embedding-8B`, scores the selected table, and returns real row IDs for SQL lookup.

**Reason:** Similarity can locate paraphrases but cannot establish exact counts, ordering, or whether a summary is grounded in the required evidence.

## When to use

Use it when exact SQL terms are unknown or when the same event may be described with different words. Search only the indexed tables: `episodes`, `entities`, `semantic_triples`, `topic_facts`, `event_steps`, `object_states`, and `visual_clips`.

**Reason:** Restricting search to tables with aligned row-ID and vector files preserves the index-to-database mapping.

## Main content

1. Choose the table whose row meaning matches the question: episodes for time-bounded descriptions, entities for names, triples for relations, topic facts for recurring facts, event steps for sequences, object states for changing state, and visual clips for visual candidates.
2. Issue several concise query variants when the question contains aliases, indirect descriptions, or multiple sub-events.
3. Treat the returned score as ranking only. Use the returned `row_id` to retrieve the actual row and its evidence links with SQL.
4. Apply exact logical timeline, time-window, and granularity filters before comparing or aggregating candidates.

**Reason:** Matching row semantics first and verifying the underlying rows reduces false matches from broad natural-language similarity.
