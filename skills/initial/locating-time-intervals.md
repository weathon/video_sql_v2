---
name: locating-time-intervals
description: Use for questions with relative time, ordered events, first/last occurrences, or uncertain temporal windows.
---

# Introduction

Temporal retrieval alternates between coarse localization and bounded expansion until the decisive interval is found.

# When to use

Use for before/after, earlier/later, day references, first/last, duration, or cross-episode questions.

# Main content

1. Locate anchor events independently before applying relative constraints. **Reason:** Searching a compound temporal relation directly can hide which anchor was missed.
2. Start with `video`/`6h`/`1h` summaries to identify regions, then inspect `3min` and `30sec` rows around each candidate. **Reason:** Coarse-to-fine search limits context while retaining precise evidence.
3. Expand a failed window symmetrically or toward the direction implied by the question, recording boundaries already checked. **Reason:** Fixed narrow windows create false negatives near summary boundaries.
4. Use numerical `start_s`/`end_s` comparisons in SQL for ordering and adjacency. **Reason:** Narrative ordering in text is less reliable than timeline fields.
5. For Week day labels, resolve the label through `source_groups` or `source_videos` rather than assuming a fixed day length. **Reason:** Week consists of many physical clips with explicit offsets.

```sql
SELECT start_s, end_s, text
FROM episodes
WHERE video_id = 'week' AND granularity = '30sec'
  AND start_s BETWEEN 100000 AND 102000
ORDER BY start_s;
```
