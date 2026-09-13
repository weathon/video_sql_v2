# Skill loading guide

This guide is included in your instructions automatically. Other skills are not: call `read_skill(name)` to read their full content before applying them.

- At the start of an investigation, load `general-retrieval` to plan the evidence search.
- Before semantic retrieval, load `semantic-search`; before writing SQL or joining evidence, load `sql-schema`.
- For before/after, first/last, duration, or interval boundaries, load `locating-time-intervals`.
- For counts, distinct occurrences, rankings, or ordered events, load `counting-and-ordering`.
- When visible identity, state, text, actions, or counts need checking in source media, load `visual-verification` before using `clip_qa`.
- Check the available skill catalog for additional question-specific skills. Load the matching skills when their conditions apply, including when a new branch changes the investigation.

**Reason:** A skill title or description is only a routing hint, not the skill's actual guidance. Load the relevant content before choosing the detailed method.
