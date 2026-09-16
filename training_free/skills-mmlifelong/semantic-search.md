---
name: semantic-search
description: Use when exact SQL terms are unknown or a concept must be located from meaning.
---

Call semantic_search(query, table, sql, question). It scores every row of one embedded table by cosine similarity, then applies sql to semantic_hits. Choose episode.text, entity.text, semantic.text, topic_facts.fact, event_steps.description, or visual_clip video embeddings according to the clue. Every question can retrieve from every video; evidence discovered during retrieval can guide subsequent video or time filters.

semantic_hits contains the source table columns plus cosine_sim, start_s and end_s. video_id identifies a broadcast; start_ts/end_ts and start_s/end_s are seconds within that video. Rank, filter, order or aggregate with one SELECT or WITH query. Examples: SELECT id, video_id, text, start_s, end_s, cosine_sim FROM semantic_hits ORDER BY cosine_sim DESC LIMIT 10; SELECT id, video_id, text, start_s, end_s FROM semantic_hits WHERE video_id = '14' AND start_s >= 600 AND end_s <= 1200 ORDER BY cosine_sim DESC LIMIT 10; SELECT video_id, COUNT(*) FROM semantic_hits WHERE cosine_sim >= 0.5 GROUP BY video_id. Natural-language columns are casefolded; use lowercase LIKE patterns.

With question=null, the tool returns up to 100 rows and shortens long text cells. Supplying question requests QA from the complete SQL result and consumes one shared investigation model call. A similarity score ranks candidates but is not evidence of correctness. Inspect content and provenance, use run_sql for nearby context or explicit relationships, and call clip_qa with video_id and local seconds only when a located detail needs visual verification.
