# Trace / Lab

Framework-free local explorer for `runs/qwen38_27b_fp8_epd_seed0_continuation`.

```bash
python3 vis/build_board.py
```

Open `vis/index.html` in a browser. It also works through a static HTTP server rooted at the repository, with `/vis/` as the URL. No package installation or frontend build is required.

- Search and filter rollouts by batch, score, or tool; switch between same-question attempts.
- Expand individual trace segments to inspect preserved reasoning, tool arguments and returned values. Event search applies to expanded segments. Original JSON links include `raw_responses`.
- Inspect full-size local media, reflections, question summaries and updater traces.
- Compare successive post-update skill snapshots. The first saved snapshot has no earlier snapshot baseline; current skill files are compared with the last saved snapshot.

The board is an exported snapshot, not a live monitor. Run the exporter again and reload to update it. Generated `vis/data/` is ignored by Git. Media and original JSON links read the run directory in place; keep the repository layout when opening the page. Existing `vis/dist/` belongs to the older viewer.
