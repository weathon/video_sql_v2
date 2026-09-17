---
name: counting-and-ordering
description: Use for counts, totals, distinct occurrences, rankings, extrema, ordinals, comparisons, and ordered sequences.
---

---
name: counting-and-ordering
description: Use for counts, distinct occurrences, rankings, comparisons, and ordered sequences.
---

# Introduction

Repeated descriptions at several granularities can duplicate one underlying occurrence. Similar objects, bids, and intermediate states can also be mistaken for completed outcomes.

# When to use

Use it whenever the answer is a count, frequency, total, rank, maximum, minimum, ordinal, comparison, or sequence.

# Main content

Define the unit before retrieving or aggregating. State whether it is an action, completed transfer, sale, visit, participant, item, source, display object, or other question-specific unit. Define inclusion rules for actor, time window, outcome, completion, and deduplication before counting.

Build a chronological ledger from precise, non-overlapping evidence. For each candidate, record its interval, entity, action, final outcome or value, and whether it is confirmed, duplicate, or unresolved. Deduplicate overlapping summaries, repeated captions, re-handlings, and multiple observations of the same event. Do not count a chain summary and its evidence rows as separate events, a plan as an occurrence, a bid as a sale, an opening value as a closing value, or temporary possession as a completed outcome.

For totals and rankings, bind every number to the exact item, actor, event, and final state. Compare only values inside the requested phase or interval. Explicit in-window totals or direct final-outcome statements can answer the question when they match the defined unit; otherwise enumerate the ledger before summing. Preserve uncertainty when a value, ownership link, or final result is not established.

For order questions, retrieve every candidate, verify that each is an actual occurrence, sort by its relevant event time or requested value, and inspect ties. Include all supported multi-select items rather than returning only the clearest one. Do not use visual identity, appearance, or a loose semantic match as proof of order unless that mapping is required.
