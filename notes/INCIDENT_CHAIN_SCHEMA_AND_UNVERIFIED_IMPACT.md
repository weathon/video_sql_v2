# Incident: invented chain schema and unverified impact claim

Date: 2026-08-28

## What happened

The implementation introduced the terms and tables `event_chains`, `event_steps`, `topic_chains`, and `topic_facts`, then required every parent chain row to contain one non-null `video_id`. These names, abstractions, and parent-level constraints were not first verified against the MAGIC-video source and artifacts.

When asked whether this affected Pilot correctness, I immediately claimed that it did because a chain might cross videos. I had not inspected the actual Pilot videos or database relationships before making that claim.

## Verified evidence

- `data/pilot.duckdb` contains 14 rows in `videos`, with filenames `month/4.mp4` and `month/11.mp4` through `month/23.mp4`.
- All 14 referenced files exist under the known Pilot dataset root, and `ffprobe` reports exactly one video stream in every file.
- The Pilot database contains 173 `event_chains` rows across those 14 video IDs and 1,680 `topic_chains` rows across those 14 video IDs.
- Zero parent chain rows reference a missing `videos.video_id`.
- Zero `event_steps` rows have a `video_id` different from their parent `event_chains` row.
- Zero `topic_facts` rows have a `video_id` different from their parent `topic_chains` row.

The inspected data therefore shows that the parent `video_id` assignment did not create an incorrect video association in the Pilot database. My earlier statement that this specific constraint affected Pilot correctness was unsupported and wrong.

## Separate schema error

The absence of an observed Pilot video-association error does not justify the custom schema. The `chain` terminology, tables, fields, and constraints were introduced without a verified mapping to MAGIC-video. Formal preprocessing must not carry them forward.

## Required prevention

- Before adding any table, field, relation, or constraint, inspect the complete relevant MAGIC-video source and a real generated artifact, then record the exact upstream mapping.
- Do not introduce `chain` terminology or the custom `event_chains`, `event_steps`, `topic_chains`, or `topic_facts` tables into the new formal schema.
- Do not require a parent or grouped record to belong to one `video_id` unless the upstream record itself is explicitly scoped to one video.
- Verify every result-impact claim against the underlying data before stating it. If the evidence has not been inspected, state that the impact is unknown rather than inferring it from plausibility.
