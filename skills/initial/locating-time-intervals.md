---
name: locating-time-intervals
description: Use this for before/after questions, last/first occurrence, temporal order, repeated events, or when expanding from a clue moment to a wider interval.
---

# Locating Time Intervals

## Introduction

Times are seconds within one logical timeline. Keep day, week, and month separate.

## When to use

Use this when the answer depends on temporal position, ordering, or a window around an event.

## Main content

1. Anchor the clearest clue first, then expand outward with SQL windows. **Reason:** broad time searches often return many similar events from other days or videos.
2. For order questions, retrieve candidate rows with `start_s` and sort in SQL. **Reason:** final ordering should come from timeline seconds, not result order or semantic rank.
3. For last/first questions, filter the candidate set first and then use `ORDER BY start_s DESC/ASC LIMIT 1`. **Reason:** ordering all rows before filtering answers the wrong question.
4. When evidence is sparse, check neighboring 30 second rows and 3 minute context. **Reason:** the decisive visual or spoken detail can appear just outside the hit row.

```sql
SELECT row_id, video_id, start_s, end_s, text
FROM episodes
WHERE video_id = 'week'
  AND granularity = '30sec'
  AND start_s BETWEEN 3600 AND 5400
ORDER BY start_s;
```

