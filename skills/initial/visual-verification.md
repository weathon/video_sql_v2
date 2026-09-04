---
name: visual-verification
description: Use when the answer depends on appearance, spatial details, on-screen text, identity, or ambiguous text evidence.
---

# Introduction

`clip_qa` delegates bounded visual inspection and returns an answer plus evidence screenshots.

# When to use

Use after locating candidate intervals when captions cannot establish a visual fact, or when important text evidence may be noisy.

# Main content

1. Narrow intervals with SQL or semantic search before calling `clip_qa`; keep each request below ten minutes when possible. **Reason:** Focused intervals allow useful frame density and precise evidence.
2. Ask one concrete visual question per call and include the distinction that must be resolved. **Reason:** Vague requests produce observations without a decision criterion.
3. In the internal clip agent, sample coarsely to locate a moment, then resample a smaller interval at higher fps or crop a relevant region. **Reason:** Adaptive sampling preserves both coverage and fine detail.
4. Request captions only when they help connect visible evidence to indexed narration. **Reason:** Captions consume context and can bias inspection toward text claims.
5. Require returned evidence timestamps for an `answered` result and inspect the screenshots supplied to the main agent. **Reason:** The visual conclusion must trace to actual frames.
6. Treat `unknown` as unresolved evidence and continue retrieval; do not convert it into a guessed answer. **Reason:** Absence in sampled frames is not proof of absence in the video.
