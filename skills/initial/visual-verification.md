---
name: visual-verification
description: Use when stored text cannot establish a visible detail, screen value, identity, state, count, or action precisely.
---

# Introduction

Visual verification examines source frames around a candidate interval.

# When to use

Use it after locating a plausible interval when the answer depends on what is actually visible.

# Main content

Call clip_qa with the smallest interval that contains the uncertain evidence and a concrete visual question. Ask about one distinction at a time. The nested visual agent can sample individual frames at different rates, crop saved original-resolution frames, inspect captions, and keep notes. Returned images are at most 854 by 480 pixels. Sampling and crops share a cumulative image budget; an over-budget request returns an error, so narrow the interval or adjust the requested sampling rate explicitly. Crop coordinates refer to the original full frame in 0-1000 units; the full frame is left=0, top=0, right=1000, bottom=1000, with left < right and top < bottom. The visual agent cites returned frame IDs, and evidence images reuse those observed frames with their actual source times. Treat an unknown result as unresolved rather than negative evidence. For an answered result, inspect the returned evidence screenshots and timestamps in relation to the textual candidate before relying on it. For long candidate windows, divide the investigation into meaningful subintervals rather than passing more than 20 minutes.
