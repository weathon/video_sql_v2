---
name: counting-and-ordering
description: Use this for counts, totals, durations, most/least, comparisons, top-k, order lists, or questions requiring deduplication before answering.
---

# Counting And Ordering

## Introduction

Use SQL aggregation for counts and ordering, then inspect representative evidence.

## When to use

Use this when the answer is numeric, ranked, ordered, or comparative.

## Main content

1. Define the unit before counting: 30 second clip, 3 minute episode, event step, object state, or chain fact. **Reason:** counting different units gives different numbers.
2. Deduplicate by the row type that represents one occurrence. For repeated 30 second hits from the same longer event, group by an event or time bucket when appropriate. **Reason:** semantic and keyword searches can return adjacent rows for one real occurrence.
3. Use SQL `GROUP BY`, `COUNT`, `SUM`, `MIN`, `MAX`, and `ORDER BY` for aggregate claims. **Reason:** the agent should not manually count long result lists in context.
4. For ordered answers, return the final order after checking the actual `start_s` values. **Reason:** result presentation order is not always chronological.

```sql
SELECT object, COUNT(*) AS n
FROM object_states
WHERE video_id = 'day'
GROUP BY object
ORDER BY n DESC
LIMIT 10;
```

