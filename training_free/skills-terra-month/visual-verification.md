---
name: visual-verification
description: Use when source frames must establish a visible identity, attribute, state, text, spatial relation, action, or count.
---

---
name: visual-verification
description: Use when stored text cannot establish a visible detail, screen value, identity, state, count, or action precisely.
---

# Introduction

Visual verification examines source frames around a textually or structurally located candidate interval.

# When to use

Use it after locating the relevant interval and identifying a specific unresolved visual distinction. Do not use `clip_qa` for broad speculative search or to scan unbounded timelines.

# Main content

Use the located `video_id` and the tool's video-local `start_s` and `end_s` values. Call `clip_qa` with the smallest interval containing the uncertain evidence, explicit fps, and one concrete visual question. Separate nearby distractors and ask an open physical distinction when object type or label is uncertain. Choose fps for the action's speed; use dense adjacent frames for a literal transition, release, handoff, or first/last boundary.

Returned images are at most 854 by 480 pixels. Sampling must fit within the image budget, so narrow the interval or adjust fps explicitly. Evidence images reuse the supplied frames with their actual source times. Inspect the returned screenshots and timestamps before relying on the answer.

For identity or appearance questions, verify the person and attribute in the same moment. Do not map a visible description to a name from unrelated text, another day, generic properties, or repeated activity facts. For spatial relations, distinguish on, near, held beside, and inside. For counts, define the item and instant first, inspect stable clear frames, enumerate items or actions without mixing moments, and reconcile disagreement before finalizing.

Treat `unknown`, errors, unreadable text, missing frames, and contradictory outputs as unresolved rather than negative or confirming evidence. Divide long candidate windows into meaningful subintervals rather than passing more than 20 minutes, and do not aggregate broad clip-level counts without an event ledger.
