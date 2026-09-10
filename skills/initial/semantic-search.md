---
name: semantic-search
description: Use when exact SQL terms are unknown or a concept must be located from meaning.
---

# Introduction

Semantic search retrieves rows whose embedded meaning resembles a query.

# When to use

Use it for paraphrases, uncertain names, visual descriptions, actions, and broad candidate discovery.

# Main content

Choose the table whose searchable field best matches the question: episodes.text, entities.name, semantic_triples.text, topic_facts.fact, event_steps.description, object_states.text, or visual_clips.text. Search with a compact description of one clue at a time in query. With question=null, return raw hits; supplying question optionally requests a separate QA answer with supporting row IDs and times. Inspect the returned evidence and time, then use SQL to recover nearby context and exact relationships. A cosine score ranks candidates but is not evidence of correctness. If the top range is unhelpful, request a later rank range or reformulate the semantic clue; do not repeatedly request an overlapping range without a reason.

