---
name: semantic-search
description: Use when exact SQL terms are unknown and semantic candidates must be found before relational verification.
---

# Introduction

Semantic search retrieves candidate row IDs from one embedded table; SQL then exposes full rows and relations.

# When to use

Use for paraphrases, visually described objects, activities with uncertain wording, or broad event discovery.

# Main content

1. Choose the narrowest relevant table: `episodes` for narrative context, `entities` for names, `semantic_triples` for relations, `topic_facts` for persistent topics, `event_steps` for ordered actions, `object_states` for state changes, and `visual_clips` for visual similarity. **Reason:** Each index represents a different evidence unit.
2. Form queries around concrete observable semantics rather than copying the entire question. **Reason:** Dense retrieval works better when the target event is not diluted by answer-format language.
3. Inspect successive rank ranges when the first results are near-duplicates; do not repeatedly request an overlapping top-N. **Reason:** Rank pagination expands coverage without wasting context.
4. Treat scores as ranking signals, not truth. Fetch returned row IDs from DuckDB and verify time, timeline, granularity, and evidence links. **Reason:** Similar text can describe the wrong occurrence.
5. After finding distinctive vocabulary or entities, switch to SQL to enumerate, join, count, or order all relevant occurrences. **Reason:** Vector ranking cannot prove completeness or relational constraints.
