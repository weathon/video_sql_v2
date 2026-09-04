---
name: META-SKILL
description: Use only during reflection and skill updating. It explains how to turn a scored rollout into reusable retrieval guidance without leaking sample-specific details.
---

# Meta Skill

## Introduction

Reflection explains why the rollout succeeded or failed and extracts reusable guidance for future questions.

## When to use

Use this after receiving the reference answer and judge score. Skill updater should also use it when deciding whether to add, replace, or delete guidance.

## Main content

1. Revisit the trace, including tool inputs and returned values, before explaining the score. **Reason:** final answers can look plausible while a tool path silently failed or answered a different question.
2. Separate sample facts from reusable method. Mention the sample only as an anonymized pattern, not with identifying details. **Reason:** skills should improve future retrieval without memorizing private dataset content.
3. If the answer was wrong, identify the earliest decision that made the failure likely. **Reason:** later symptoms are often consequences of one bad search, join, or verification choice.
4. If the answer was right, identify which evidence path made it reliable and what would have broken it. **Reason:** successful traces teach when a method is dependable.
5. Propose edits that replace or delete redundant, conflicting, stale, or overly narrow guidance. **Reason:** skill files should stay small enough to trigger and read usefully.

