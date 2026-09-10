---
name: visual-verification
description: Use when stored text cannot establish a visible detail, screen value, identity, state, count, or action precisely.
---

# Introduction

Visual verification examines source frames around a candidate interval.

# When to use

Use it after locating a plausible interval when the answer depends on what is actually visible.

# Main content

Call clip_qa with the smallest interval that contains the uncertain evidence, an explicit fps, and a concrete visual question. Ask about one distinction at a time. The tool samples the entire interval at that fps and sends all frames in time order in a single model call. Choose fps to capture the detail or action being checked. Returned images are at most 854 by 480 pixels. Sampling must fit within the image budget; an over-budget request returns an error, so narrow the interval or adjust fps explicitly. Evidence images reuse the supplied frames with their actual source times. Treat an unknown result as unresolved rather than negative evidence. For an answered result, inspect the returned evidence screenshots and timestamps in relation to the textual candidate before relying on it. For long candidate windows, divide the investigation into meaningful subintervals rather than passing more than 20 minutes.
