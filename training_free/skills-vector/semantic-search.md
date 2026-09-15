---
name: semantic-search
description: Use when exact SQL terms are unknown or a concept must be located from meaning.
---

# Introduction

Semantic search scores every row of one embedded table by cosine similarity to a query, then lets you filter and rank those rows with SQL.

# When to use

Use it for paraphrases, uncertain names, visual descriptions, actions, and broad candidate discovery.

# Main content

Call semantic_search(query, table, sql, question). Choose the table whose searchable content best matches the question: episode.text, entity.text, semantic.text, topic_facts.fact, event_steps.description, or visual_clip video embeddings. Write a compact description of one clue at a time in query. Only IDs present in this question's filtered database participate.

After retrieval the tool builds a temporary in-memory table named semantic_hits with all columns of the chosen table plus cosine_sim (higher means more similar to query), start_s and end_s (seconds since DAY1 00:00:00, ready for clip_qa). Your sql is one read-only SELECT or WITH query over semantic_hits; it is how you rank, filter, order, or aggregate the candidates. Examples: SELECT id, text, start_ts, cosine_sim FROM semantic_hits ORDER BY cosine_sim DESC LIMIT 10 ranks by similarity; SELECT id, text, start_s, end_s, cosine_sim FROM semantic_hits WHERE start_ts >= 200000000 AND end_ts < 300000000 ORDER BY cosine_sim DESC LIMIT 10 restricts to DAY2; SELECT id, text, start_ts FROM semantic_hits WHERE cosine_sim >= 0.4 ORDER BY start_ts LIMIT 20 orders good matches by time; SELECT id, text, cosine_sim FROM semantic_hits WHERE text LIKE '%alice%' ORDER BY text, cosine_sim DESC LIMIT 20 filters and sorts by a name; SELECT COUNT(*) FROM semantic_hits WHERE cosine_sim >= 0.5 counts strong matches. Text columns are casefolded, so use lowercase in LIKE patterns.

With question=null, the tool returns up to 100 result rows with long text cells shortened; supplying question requests a separate QA answer from the complete result with supporting IDs and times and consumes one shared investigation model call. Inspect returned evidence and time, then use run_sql to recover nearby context and explicit relationships. A cosine score ranks candidates but is not evidence of correctness. If the top range is unhelpful, adjust the sql or reformulate the semantic clue; do not repeat the same query and sql without a reason.
