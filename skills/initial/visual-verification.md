---
name: visual-verification
description: Use when stored text cannot establish a visible detail, screen value, identity, state, count, or action precisely.
---

# Introduction

Visual verification examines source frames around a candidate interval.

# When to use

Use it after locating a plausible interval when the answer depends on what is actually visible.

# Main content

Call clip_qa with the smallest interval that contains the uncertain evidence and a concrete visual question. Ask about one distinction at a time. The nested visual agent can sample at different rates, crop a region, inspect captions, and keep notes. Treat an unknown result as unresolved rather than negative evidence. For an answered result, inspect the returned evidence screenshots and timestamps in relation to the textual candidate before relying on it. For long candidate windows, divide the investigation into meaningful subintervals rather than passing more than 20 minutes.

