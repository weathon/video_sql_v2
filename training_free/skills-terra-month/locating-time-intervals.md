---
name: locating-time-intervals
description: Use for before/after, first/last, duration, recurrence, day boundaries, and relative event windows.
---

---
name: locating-time-intervals
description: Use when the question depends on before, after, first, last, duration, recurrence, or a relative event window.
---

# Introduction

Temporal questions require locating the intended event boundaries before measuring, sorting, or inspecting evidence.

# When to use

Use it for relative chronology, duration, transitions, first or nth occurrences, recurring events, and sparse clues separated in time.

# Main content

Locate each anchor independently and record candidate times. Confirm anchors from precise rows before applying `MIN`, `MAX`, ordering, interval arithmetic, or visual inspection. Retain `video_id` with every timestamp: numeric times are seconds within that video, and numeric video IDs give broadcast ordering. Use the located `video_id` and local `start_s`/`end_s` for `clip_qa`.

Match the boundary to the question's event definition. Distinguish a plan, first mention, setup, arrival, payment, or intermediate state from an actual start, completion, exit, transfer, or participation transition. For durations, use the event-level start and its relevant completion or departure, rather than a coarse summary label or a later topical mention. For a named segment, establish both segment boundaries before including candidates.

For first, last, nth, and after-event questions, inspect the complete intervening timeline, including both endpoints. For repeated events, bind each candidate to the same day, activity, actor, and requested predicate before ordering. Do not replace an earlier occurrence with a more salient later recurrence, use a sub-action time for a named activity, or fill a chronological gap by inference.

Search a narrow neighborhood first and expand only when evidence indicates that the boundary lies outside it. Use visual verification only when the temporal record leaves a material boundary or visible transition unresolved.
