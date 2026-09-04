---
name: visual-verification
description: Verify candidate facts against video with clip_qa. Use when text is ambiguous, when appearance or action matters, or when exact visual state and sequence must be confirmed.
---

# Visual Verification

## Introduction

Use indexed text to find a narrow candidate interval, then ask `clip_qa` a concrete visual question about that interval. Preserve the returned evidence timestamps.

**Reason:** Video inspection is strongest as targeted verification and expensive as unguided timeline search.

## When to use

Use this skill for colors, objects, gestures, interactions, on-screen state, identity cues, visual order, or conflicts between textual sources. Prefer intervals under 10 minutes; intervals over 20 minutes are rejected.

**Reason:** Focused intervals produce inspectable contact sheets and keep the visual agent centered on one claim.

## Main content

1. Resolve the logical interval to the matching `visual_clips` or physical source-video interval before calling `clip_qa`.
2. Ask one falsifiable question that distinguishes the candidate answer from alternatives.
3. If the returned status is `unknown`, narrow or shift the interval using database evidence instead of treating uncertainty as a negative fact.
4. Use the independent screenshots produced for returned evidence timestamps to support the final claim.

**Reason:** Explicit mapping, falsifiable questions, and timestamped screenshots make visual conclusions traceable to actual frames.
