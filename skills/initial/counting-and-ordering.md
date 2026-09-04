---
name: counting-and-ordering
description: Use for counts, totals, frequencies, rankings, durations, and ordered sequences.
---

# Introduction

Counting and ordering require explicit definitions of the unit, identity, scope, and deduplication key.

# When to use

Use whenever the answer depends on how many, how often, how long, most/least, first/last, or sequence order.

# Main content

1. State the counted unit before querying: distinct object, event occurrence, interval, person, or state transition. **Reason:** Row counts often overcount repeated descriptions of one event.
2. Inspect candidate rows and select a deduplication key such as a 30-second evidence row, chain step, object identity, or non-overlapping interval. **Reason:** Multiple tables and granularities can describe the same occurrence.
3. Restrict the temporal and logical scope before aggregation. **Reason:** Aggregating first can mix independent timelines or unrelated episodes.
4. For duration, merge overlapping intervals before summing; do not sum caption or fact-row durations blindly. **Reason:** Overlapping evidence creates duplicate time.
5. For order, sort by numeric evidence time and retain ties until visual or textual evidence distinguishes them. **Reason:** SQL row order and chain wording are not chronological guarantees.

```sql
SELECT caption_30sec_row_id, MIN(time_s) AS first_time, COUNT(*) AS fact_rows
FROM topic_facts
WHERE video_id = 'month' AND fact ILIKE '%target%'
GROUP BY caption_30sec_row_id
ORDER BY first_time;
```
