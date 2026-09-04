---
name: counting-and-ordering
description: Compute exact counts, frequencies, rankings, and event order with SQL. Use for how-many, most/least, first/last, repeated-event, comparison, and ordered-list questions.
---

# Counting and Ordering

## Introduction

Define the counted unit before aggregating. A single real event can appear in several summaries, chains, facts, triples, and clips.

**Reason:** Counting index rows without defining an event identity systematically overcounts duplicated representations.

## When to use

Use this skill when the requested answer depends on `COUNT`, `SUM`, `MIN`, `MAX`, grouping, deduplication, ranking, or chronological ordering.

**Reason:** These operations require explicit relational logic rather than similarity ranking.

## Main content

1. State whether one unit is a distinct 30-second evidence row, a chain step, a physical clip, a source video, or a deduplicated real-world occurrence.
2. Resolve candidate rows to direct 30-second evidence IDs before deduplicating repeated summaries of the same occurrence.
3. Filter the correct logical timeline and time interval before grouping.
4. Use `COUNT(DISTINCT ...)` only on the identifier that matches the chosen unit; inspect grouped rows when near-duplicate events may remain.
5. For ordered answers, select the supporting timestamp for each item and sort numerically by logical time before formatting the response.

**Reason:** Explicit units, evidence-level deduplication, and numeric ordering make the computation reproducible from returned rows.

## SQL patterns

```sql
SELECT caption_30sec_row_id, MIN(snapshot_s) AS first_time
FROM semantic_triples
WHERE video_id = ? AND predicate = ?
GROUP BY caption_30sec_row_id
ORDER BY first_time;
```

**Reason:** Grouping by direct evidence ID collapses multiple relational descriptions attached to the same 30-second occurrence.
