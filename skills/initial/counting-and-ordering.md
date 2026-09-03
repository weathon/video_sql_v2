---
name: counting-and-ordering
description: Answering "how many times", "in which places" and "what is the correct order" questions across one or many videos. Use whenever the answer is a count, a list of distinct occurrences, or a sequence of events.
---

# Counting and Ordering

## Introduction

Counting and ordering questions fail when duplicates are counted twice or when occurrences in different videos are ordered by time within the video instead of by the video order. Collect every candidate occurrence with its `video_id` and time, deduplicate, then count or sort.

## When to use

- "How many times did X happen?"
- "In which N places did X happen?"
- "What is the correct order of these scenes?"

## Main content

1. Gather candidates from several sources (30sec episodes, visual clips, topic facts, event steps) using both keyword and semantic search; a single source usually misses some occurrences.
2. Merge candidates that are close in time in the same video, because the same occurrence spans several 30 second rows.

```sql
-- Group nearby 30sec hits into occurrences (gap larger than 300 s starts a new occurrence)
WITH hits AS (
  SELECT video_id, start_s, end_s
  FROM episodes
  WHERE granularity = '30sec' AND text LIKE '%blind box%'
),
marked AS (
  SELECT *, CASE WHEN start_s - LAG(end_s) OVER (PARTITION BY video_id ORDER BY start_s) > 300 THEN 1 ELSE 0 END AS new_group
  FROM hits
),
grouped AS (
  SELECT *, SUM(new_group) OVER (PARTITION BY video_id ORDER BY start_s) AS occurrence
  FROM marked
)
SELECT video_id, occurrence, MIN(start_s) AS start_s, MAX(end_s) AS end_s, COUNT(*) AS rows_in_group
FROM grouped
GROUP BY video_id, occurrence
ORDER BY video_id, start_s;
```

3. Verify each merged occurrence by reading its full text (and `clip_qa` when the text is unclear) before including it in the count; drop occurrences that only mention the topic without the event actually happening.
4. For ordering, rank by `video_id` first and time inside the video second, and confirm the video order from the `videos` table and surrounding episode text when the videos are parts of one trip.

```sql
SELECT video_id, MIN(start_s) AS first_time
FROM visual_clips
WHERE text LIKE '%sunshine rainbow%'
GROUP BY video_id
ORDER BY video_id, first_time;
```
