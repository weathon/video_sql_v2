# Incident: conflating logical video IDs with video files and making an unverified impact claim

Date: 2026-08-28

## What happened

I saw `video_id` on topic and event chain records and incorrectly interpreted it as a constraint tying every chain to one physical video file. I then claimed that the chain terminology had been invented locally and that the constraint affected Pilot correctness. I had not inspected the fixed MAGIC-video commit, the actual Pilot files, or the database relationships before making those claims.

## Verified evidence

- MAGIC-video commit `0f2273f7f07360cc58588399553b5a6ea758b084` contains `preprocess/mmlifelong/extract_topic_chains.py` and `preprocess/mmlifelong/extract_event_chains.py`.
- Both upstream extractors process and store chains by logical `video_id`; the chain terminology was not invented by this project.
- In the formal index, `video_id` denotes a complete logical video timeline. Month, day, and week are separate timelines, and multiple source files may contribute to one timeline.
- Physical clip lookup is separate from the logical timeline: `source_video_id` and the relative `filename` identify the source file, while chain, caption, ASR, topic, and event times are mapped into the logical `video_id` timeline.
- `data/pilot.duckdb` contains 14 logical video rows. All 14 referenced Pilot files exist, and `ffprobe` reports exactly one video stream in every file.
- The Pilot database contains 173 event-chain rows and 1,680 topic-chain rows. Zero parent rows reference a missing logical `video_id`, zero event-step rows disagree with their parent event row, and zero topic-fact rows disagree with their parent topic row.

The inspected upstream source and Pilot data show that retaining `video_id` on a chain is correct. The error was treating `video_id` as a physical-file identifier and claiming an effect on correctness without verification.

## Required prevention

- Before changing any table, field, relation, or constraint, inspect the complete relevant MAGIC-video source and a real generated artifact, then record the exact upstream mapping.
- Keep logical `video_id`, physical `source_video_id`, and relative `filename` distinct in schema, preprocessing, retrieval, and explanations.
- Preserve upstream topic and event chains, with their timestamps mapped to the logical video timeline.
- Verify every result-impact claim against the underlying data before stating it. If the evidence has not been inspected, state that the impact is unknown rather than inferring it from plausibility.
