---
name: locating-time-intervals
description: Locate events on the day, week, or month logical timeline and map them to physical video intervals. Use for before/after, duration, recurrence, cross-day, and vague-time questions.
---

# Locating Time Intervals

## Introduction

Search time in two layers: locate candidate events on a logical timeline, then map the selected interval through `source_videos` or `visual_clips` to physical media time.

**Reason:** Logical timeline seconds and source-file seconds are different coordinate systems.

## When to use

Use this skill whenever the question contains a date, part of day, relative order, elapsed time, repeated occurrence, or a request to inspect a particular moment.

**Reason:** Time assumptions made before identifying the correct timeline can shift every later query and video sample.

## Main content

1. Determine whether the question belongs to `day`, `week`, or `month` before filtering by `video_id`.
2. Search coarse `scale`, `video`, `6h`, or `1h` episodes to find the region, then narrow through `3min` and `30sec` rows.
3. Expand the search window around a candidate when the event boundary or prerequisite action may fall outside the matching row.
4. Order by logical `start_s` for cross-file chronology. Use `source_start_s` and `source_end_s` only after selecting the corresponding physical clip.
5. For elapsed-time questions, identify the two boundary events independently before subtracting their logical timestamps.

**Reason:** Coarse-to-fine localization limits search while preserving the exact 30-second evidence and correct chronology.
