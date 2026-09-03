---
name: semantic-search
description: How to use semantic_search to find rows when exact keywords are unknown or SQL LIKE returns nothing, and how to turn a hit into exact SQL follow-up. Use whenever the question describes a scene or event in words that may not appear verbatim in the index.
---

# Semantic Search

## Introduction

`semantic_search(query, n, table)` embeds the query with the same model used to build the index and returns the top `n` rows by exact cosine similarity, each with `row_id`, `video_id`, `start_s`, `end_s`, a 500 character text preview and the score. Available tables: `episodes`, `entities`, `semantic_triples`, `openie_triples`, `topic_facts`, `event_steps`, `visual_clips`. The `visual_clips` index was built from the video frames themselves, so a text query against it finds visually similar moments even when the clip description does not mention the words.

**Reason:** This distinguishes semantic ranking from exact evidence and explains which index can locate visual scenes.

## When to use

- The question paraphrases an event ("became displeased with fans", "opened blind boxes") and keyword SQL finds too little.
- You need candidate videos and time ranges before narrowing with SQL.
- You want visual matches for a described scene without knowing where it happens.

**Reason:** These are the cases where approximate semantic retrieval adds information that exact SQL matching may miss.

## Main content

1. Write the query as a short description of the scene or fact, not as a question.
2. Start with `episodes` or `visual_clips` for scenes and `topic_facts` or `openie_triples` for facts about a named entity.
3. Use `n` around 20 to 50; the score is only a ranking signal, so inspect the previews rather than thresholding.
4. Take the returned `video_id` and time range and expand around it with SQL to read neighbouring rows. `run_sql` still limits every returned text cell to a 500-character preview, so do not treat a long cell as untruncated evidence:

```sql
-- Read full context around a semantic hit at video 21, 19881-19980 s
SELECT granularity, start_s, end_s, text
FROM episodes
WHERE video_id = '21' AND end_s >= 19881 - 120 AND start_s <= 19980 + 120
ORDER BY granularity, start_s;
```

5. Repeat with reworded queries when the first results are off topic; different phrasings retrieve different rows.

**Reason:** Ranking candidates first and verifying them with exact surrounding rows prevents similarity scores from being treated as answers.
