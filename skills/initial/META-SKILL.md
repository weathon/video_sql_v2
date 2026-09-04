---
name: meta-reflection
description: Guide the original answering agent through post-judge reflection. Inject only after that rollout receives its reference answer and score, while continuing the same conversation and reasoning history.
---

# Reflection

## Introduction

Continue as the same agent that answered the question. Compare the reference answer and score with your own reasoning, tool calls, tool returns, notes, and submitted answer; do not reconstruct the experience from a trace given to another agent.

**Reason:** The original conversation contains the model's own decision context, which is the evidence needed to explain why its approach succeeded or failed.

## When to use

Use only after the judge has scored a completed or compaction-limited rollout and the harness has appended the reference answer and score to that rollout's conversation.

**Reason:** Loading reflection guidance during initial answering would leak feedback and change the retrieval policy being evaluated.

## Main content

1. Identify the earliest decision that materially helped or harmed the answer, then verify that diagnosis against the actual tool inputs and returned values.
2. Re-query the database or video when the trace does not establish why the answer differs from the reference.
3. Produce `reflection` as reusable retrieval guidance and `meta_reflection` as reusable guidance for conducting future reflections.
4. Remove or abstract question-specific names, dates, answers, and private details. A sanitized example may illustrate a general rule but must not reveal the training item.
5. State whether the new lesson should add to, modify, replace, merge with, or delete existing guidance; do not default to indefinite accumulation.

**Reason:** Evidence-backed diagnosis, abstraction, and explicit revision intent turn one rollout into reusable guidance without leaking its answer or growing contradictory skills.
