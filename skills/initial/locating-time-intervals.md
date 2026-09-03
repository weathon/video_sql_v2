---
name: locating-time-intervals
description: Finding the exact video and time interval where a specific activity or moment happens, and reading what happened right before or after it. Use for questions that ask where or when something occurred, what happened at a particular place or moment, or before handing an interval to clip_qa.
---

# Locating Time Intervals

## Introduction

Every table stores positions in seconds within one `video_id`. Locating a moment means narrowing from a whole video to a range of a few minutes that can be read in full or inspected visually.

## When to use

- The question names a place, activity or object and asks what happened there.
- You have a rough hit and need precise start and end times.
- You need to look at the surroundings of a moment (what led to it, what followed).

## Main content

1. Find candidate rows by keyword or semantic search; note `video_id`, `start_s`, `end_s`.
2. Zoom in using the episode granularity ladder: `1h` and `10min` rows for orientation, `3min` for context, `30sec` for exact positions.

```sql
-- Coarse-to-fine: all 3min summaries covering a candidate hour in video 14
SELECT start_s, end_s, text
FROM episodes
WHERE video_id = '14' AND granularity = '3min' AND start_s >= 2400 AND end_s <= 6000
ORDER BY start_s;
```

```sql
-- Exact position: 30sec summaries and visual clip descriptions around 2685 s
SELECT 'episode' AS source, start_s, end_s, text FROM episodes
WHERE video_id = '14' AND granularity = '30sec' AND end_s >= 2685 - 180 AND start_s <= 2685 + 180
UNION ALL
SELECT 'clip', start_s, end_s, text FROM visual_clips
WHERE video_id = '14' AND end_s >= 2685 - 180 AND start_s <= 2685 + 180
ORDER BY start_s, source;
```

3. Use entities to bound an interval when a person or object is the anchor:

```sql
SELECT name, first_seen_s, last_seen_s, frequency
FROM entities
WHERE video_id = '14' AND normalized_name LIKE '%basketball%'
ORDER BY first_seen_s;
```

4. Once the interval is under about 10 minutes, hand it to `clip_qa` with a focused question if visual confirmation is needed. Intervals over 20 minutes are rejected, so split long ranges.
