# Incident: stored durable transcripts under the disposable runs directory

## Date
2026-08-28

## Summary
The full MM-Lifelong transcription runner wrote every JSON and SRT transcript under `/raid/wg25r/video_sql_v2/runs/magic_whisper_full`. I treated the transcripts as disposable run artifacts instead of durable data required by later pipeline stages. The directory is now absent, and no `*_whisper.json` or `*_whisper.srt` files are present in the repository.

## What went wrong
1. I hardcoded `output_root` in `src/preprocess/transcribe_mmlifelong_magic.py` to `/raid/wg25r/video_sql_v2/runs/magic_whisper_full`.
2. I followed the plan convention that run artifacts belong under `runs/` without separating execution logs from the transcript dataset produced by the run.
3. I did not ask the user to decide the persistent storage location before launching the full 181.195519-hour transcription.
4. I documented the `runs/magic_whisper_full` path in `notes/ASR.txt` and later recorded it in `CODEX_PLAN.md` as the completed transcript location.
5. I reported the transcripts as available without rechecking that the actual files still existed at the reported path.

## Impact
The completed run had reported 6,290 JSON files, 6,290 SRT files, and 175,191 speech segments, but none of those transcript files is currently available in the repository. The usable transcript dataset is therefore 0 of 181.195519 source hours, and downstream stages cannot consume the reported ASR output.

## Root cause
I conflated two storage classes: temporary execution artifacts and persistent derived data. The transcript files are inputs to later indexing and retrieval work, but I placed them beside disposable logs under `runs/`. I also made that storage decision without user confirmation.

## Corrective actions
1. Store transcript JSON and SRT files under `data/`, not `runs/`.
2. Keep execution logs under `runs/` and keep derived datasets required by later stages under `data/`.
3. Ask the user before choosing the storage root for any large generated dataset.
4. Before reporting a generated dataset as available, inspect the actual destination and open real output files from it.
5. Do not mark ASR data as available in CODEX_PLAN.md or notes until the documented persistent path exists and contains the verified artifacts.

## Files touched
This report adds `notes/INCIDENT_TRANSCRIPTS_STORED_IN_RUNS.md`. It does not modify the transcription runner, CODEX_PLAN.md, notes/ASR.txt, or any data path.
