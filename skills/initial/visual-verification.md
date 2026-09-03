---
name: visual-verification
description: Using clip_qa to inspect a video interval when the answer depends on what is visible (appearance, attributes, differences between scenes, whether something actually happened). Use whenever text evidence is ambiguous, contradictory, or missing a visual detail the question asks about.
---

# Visual Verification

## Introduction

`clip_qa(video_id, start_s, end_s, question)` runs a nested visual agent that samples frames from the interval and answers one focused question, returning `answered` or `unknown` with evidence timestamps and screenshots. It shares notes with the main agent through `add_note` and `read_notes`.

**Reason:** This explains what visual evidence the tool returns and how it connects to the surrounding retrieval trace.

## When to use

- Attribute questions (colours, clothing, haircut result, what someone is holding).
- Checking whether a described event really happened versus was only talked about.
- Comparing two scenes (differences between the people or objects in each).
- Hallucination checks: confirming a statement before answering.

**Reason:** These cases cannot be resolved reliably from text indexes alone.

## Main content

1. Locate the interval first with SQL or semantic search; do not scan whole videos visually.
2. Keep intervals under about 10 minutes and ask one concrete question per call, for example "What does the person's haircut look like at the end of the visit?" rather than "What happens here?".
3. Before calling, write what you already know into `add_note` so the clip agent can focus on the missing detail.
4. If the result is `unknown`, narrow the interval using the 30sec episodes and visual clip descriptions, or try the neighbouring interval, rather than repeating the same call.
5. Combine the visual answer with the text evidence; when they disagree, prefer what was actually seen in the frames and note the disagreement.

**Reason:** Focused intervals and timestamped screenshots make visual conclusions inspectable instead of relying on unverified summaries.
