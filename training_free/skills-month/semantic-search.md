---
name: semantic-search
description: Use when exact SQL terms are unknown or a concept must be located from meaning.
---

# Main content

Call semantic_search(query, table, sql, question). The query is embedded and compared by cosine similarity with every row of the chosen table across all indexed videos; SQL then filters, ranks, orders or aggregates those rows.

Choose episode for caption text, visual_clip for video embeddings, entity for names, semantic for triples, topic_facts for facts, or event_steps for descriptions. semantic_hits contains the chosen table's columns plus cosine_sim, start_s and end_s. video_id identifies the source video; all numeric times are seconds within that video.

Examples: SELECT id, video_id, text, start_s, end_s, cosine_sim FROM semantic_hits ORDER BY cosine_sim DESC LIMIT 10; SELECT id, video_id, text, start_s, end_s FROM semantic_hits WHERE video_id = '14' AND start_s >= 300 AND end_s <= 600 ORDER BY cosine_sim DESC LIMIT 10; SELECT video_id, COUNT(*) FROM semantic_hits WHERE cosine_sim >= 0.5 GROUP BY video_id.

With question=null, return up to 100 raw rows with long text shortened. Supplying question requests a separate QA answer from the complete SQL result and consumes a shared investigation model call. Preserve evidence IDs, video identity, times and uncertainty. Similarity ranks candidates; it does not establish correctness. Use run_sql for neighboring context and explicit links, and clip_qa(video_id=..., start_s=..., end_s=..., fps=..., question=...) for an unresolved visual detail in a located interval.
