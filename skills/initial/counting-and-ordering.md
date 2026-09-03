---
name: counting-and-ordering
description: Answering "how many times", "in which places" and "what is the correct order" questions across one or many videos. Use whenever the answer is a count, a list of distinct occurrences, or a sequence of events.
---

# Counting and Ordering

## Introduction

Counting and ordering questions fail when duplicates are counted twice or when occurrences in different videos are ordered by time within the video instead of by the video order. Collect every candidate occurrence with its `video_id` and time, deduplicate, then count or sort.

**Reason:** Counts and sequences are only reliable when occurrence identity and cross-video chronology are explicit.

## When to use

- "How many times did X happen?"
- "In which N places did X happen?"
- "What is the correct order of these scenes?"

**Reason:** Each question form requires reasoning across multiple candidate occurrences rather than returning one retrieved row.

## Main content

1. Gather candidates from several sources (30sec episodes, visual clips, topic facts, event steps) using both keyword and semantic search; a single source usually misses some occurrences.
2. Sort candidate rows by numeric video ID and time so adjacent rows from the same possible occurrence can be inspected together. Do not assume a fixed time-gap threshold: nearby rows may be distinct events, while one event may span several summaries.

```sql
SELECT video_id, start_s, end_s, text
FROM episodes
WHERE granularity = '30sec' AND text LIKE '%target event%'
ORDER BY TRY_CAST(video_id AS INTEGER), video_id, start_s;
```

3. Merge rows only after their text and visual evidence show that they describe the same occurrence. Verify each resulting occurrence with `clip_qa` when the text is unclear before including it in the count; drop rows that only mention the topic without the event actually happening.
4. For ordering, rank the pilot's numeric `video_id` values numerically first and time inside the video second, and confirm the video order from the `videos` table and surrounding episode text when the videos are parts of one trip.

```sql
SELECT video_id, MIN(start_s) AS first_time
FROM visual_clips
WHERE text LIKE '%target scene%'
GROUP BY video_id
ORDER BY TRY_CAST(video_id AS INTEGER), video_id, first_time;
```

**Reason:** Evidence-based deduplication avoids arbitrary merge thresholds, while numeric video ordering matches the pilot's actual IDs.
