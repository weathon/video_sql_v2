# Incident: Week training questions reused for validation

## What happened

The new Terra skills were learned from analyses of all 200 MM-Lifelong Week questions and their saved main/branch rollouts. When asked to validate those skills, I launched the same 200 Week questions again, with one main trajectory per question and no branches. This was a training-set retest, not held-out validation, even though the dataset file is named `week/test.json`.

The user identified the split error. The run was stopped after 62 question batches had saved answers and GPT-5 scores. Its two dedicated Qwen containers were stopped and deleted; the shared embedding and the other task's QA service were retained. No Month validation was run by this mistaken invocation.

## Cause

I followed the original rollout's dataset path without checking it against the questions used to learn the skills. I failed to distinguish the dataset's published split name from its role in this experiment. The requested validation should use the held-out Month split.

## Records and correction

- Training analyses: `runs/analysis/analyses/`; generated skills: `runs/analysis/skills/`.
- Stopped training-set retest: `runs/mmlifelong-week-validation-terra/collection/`, W&B run `hqo3rzls`. Preserve these records and label their results as training-set retest results; do not report them as held-out validation.
- Run the requested validation on Month, keeping its data, results, and W&B run separate from the stopped Week run.
- Before starting validation, check the actual questions used for skill learning against the proposed evaluation questions. For the requested trained/untrained comparison, align Month records by original ID and question content and compute both methods' metrics on exactly the same completed overlap.
