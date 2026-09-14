---
name: counting-and-ordering
description: Use for counts, distinct occurrences, rankings, comparisons, and ordered sequences.
---

# Introduction

Repeated descriptions at several granularities can duplicate one underlying occurrence.

# When to use

Use it whenever the answer is a count, frequency, rank, maximum, minimum, or sequence.

# Main content

Define the unit being counted before aggregating. Prefer precise 30-second evidence or event steps and deduplicate overlapping descriptions of the same occurrence. Do not count a chain summary and each of its evidence rows as separate events. Use stable event boundaries, caption IDs, or non-overlapping intervals for deduplication. For rankings, compute the comparison value explicitly and inspect ties. For order questions, retrieve all candidates, sort by evidence time or the requested value, and verify missing candidates before submitting.

