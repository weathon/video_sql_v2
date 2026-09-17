# Skill loading guide

This guide is included in your instructions automatically. Other skills are not: read the routed skill before applying its detailed method.

- At the start of an investigation, load `general-retrieval`.
- Before semantic retrieval, load `semantic-search`; before SQL, joins, or evidence tracing, load `sql-schema`.
- For before/after, first/last, duration, recurrence, day boundaries, or relative windows, load `locating-time-intervals`.
- For counts, totals, extrema, rankings, ordinals, or ordered event sequences, load `counting-and-ordering`.
- When visible identity, appearance, state, text, spatial relation, action, or count is decisive, load `visual-verification` before using `clip_qa`.
- Load every skill whose condition applies when a new branch changes the investigation.

A skill title or description is a routing hint; use the full skill content before choosing a method.
