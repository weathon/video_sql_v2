---
name: visual-verification
description: Use this when the answer depends on visible appearance, identity, object state, spatial relation, text on screen, or when SQL/search evidence is plausible but not visually decisive.
---

# Visual Verification

## Introduction

`clip_qa` runs an inner visual agent over a bounded interval and returns an answer plus evidence times.

## When to use

Use visual verification for visible facts or when text evidence might be a summary, ASR artifact, or paraphrase.

## Main content

1. Keep clips under 10 minutes when possible. Never ask `clip_qa` for more than 20 minutes. **Reason:** long clips dilute visual attention and the tool rejects intervals over 20 minutes.
2. Pass the exact `video_id`, `start_s`, and `end_s` from SQL or semantic hits. **Reason:** clip filenames and source offsets are resolved by the database, not by guessed paths.
3. Ask a focused question about the visual fact needed for the final answer. **Reason:** broad clip questions produce vague summaries instead of decisive evidence.
4. If `clip_qa` returns `unknown`, search adjacent rows or use another evidence path; do not turn unknown into a positive answer. **Reason:** absence of visual confirmation is not proof of the opposite.

