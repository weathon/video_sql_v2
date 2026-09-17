---
name: semantic-search
description: Use when exact SQL terms are unknown or a concept must be located from meaning.
---

---
name: semantic-search
description: Use when exact SQL terms are unknown or a concept must be located from meaning.
---

# Introduction

Semantic search scores every row of one embedded table by cosine similarity to a query, then lets you filter and rank those rows with SQL.

# When to use

Use it for paraphrases, uncertain names, visual descriptions, actions, and broad candidate discovery.

# Main content

Choose the table whose searchable content best matches the clue: `episode.text`, `entity.text`, `semantic.text`, `topic_facts.fact`, `event_steps.description`, or `visual_clip` video embeddings. Query one compact clue at a time. Only IDs present in this question's filtered database participate.

After retrieval, treat the result as candidate discovery. Inspect the returned text and time, then use `run_sql` to recover local context, direct wording, and explicit relationships. Reformulate a weak query or change the filter when the result range is unhelpful; do not repeat the same query and SQL without a reason. An empty exact-term search does not establish that a visually depicted activity or unindexed descriptor is absent.

The tool builds a temporary in-memory `semantic_hits` table with all columns of the chosen table plus `cosine_sim` (higher is more similar), `start_ts` and `end_ts` (DHHMMSSFF; added for `topic_facts` and `event_steps`), and `start_s` and `end_s` (seconds since DAY1 00:00:00, ready for `clip_qa`). Use one read-only `SELECT` or `WITH` query over `semantic_hits` to rank, filter, order, or aggregate candidates.

Examples: `SELECT id, text, start_ts, cosine_sim FROM semantic_hits ORDER BY cosine_sim DESC LIMIT 10`; `SELECT id, text, start_s, end_s, cosine_sim FROM semantic_hits WHERE start_ts >= 200000000 AND end_ts < 300000000 ORDER BY cosine_sim DESC LIMIT 10`; `SELECT id, text, start_ts FROM semantic_hits WHERE cosine_sim >= 0.4 ORDER BY start_ts LIMIT 20`; `SELECT id, text, cosine_sim FROM semantic_hits WHERE text LIKE '%alice%' ORDER BY text, cosine_sim DESC LIMIT 20`.

Text columns are casefolded, so use lowercase in `LIKE` patterns. With `question=null`, the tool returns up to 100 raw rows with long text shortened; supplying `question` requests a separate QA answer from the complete result. Inspect returned evidence before relying on it.
