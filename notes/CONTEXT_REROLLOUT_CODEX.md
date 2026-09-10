# day/196 context rerollout

## Review and run

- Reviewed the one-shot clip QA diff with an independent subagent and the full repository with Claude CLI before the run.
- Review fixes committed as `230d52a`: remove reasoning associated only with discarded replacement calls; remove implementation details from the clip QA tool description and visual-verification skill.
- Reran day question 196, rollout 00, from the root with the current initial skills and one-shot clip QA. Both runs used `gpt-5.6-luna`.
- The new run used the frame-sequence path. Native Qwen video input was not exercised.

## Successful QA path

Includes main-agent compaction and clip QA. Excludes failed/interrupted attempts and judge calls. Input is cumulative across calls; peak context is the maximum input plus output tokens of one call. Reasoning tokens come from API usage and are already included in output tokens.

| Metric | Previous | One-shot clip QA |
| --- | ---: | ---: |
| Model calls | 94 | 38 |
| Main model calls, including compaction | 17 | 20 |
| Clip QA model calls | 77 | 18 |
| Input tokens | 3,732,625 | 3,064,672 |
| Output tokens | 48,658 | 152,371 |
| Reasoning tokens | 33,068 | 139,933 |
| Main-agent peak context tokens | 165,420 | 153,713 |
| Peak context tokens across all QA calls | 183,571 | 185,627 |
| Returned service tier | default | flex |
| QA cost, USD | 0.418345400 | 0.379039605 |
| QA cost with both priced at flex, USD | 0.209172700 | 0.379039605 |
| Main-agent cost, USD | 0.130005680 | 0.063163655 |
| Clip QA cost, USD | 0.288339720 | 0.315875950 |
| replace_tool calls | 0 | 0 |
| Judge score | 0 | 0 |

## Total billed calls and trace inspection

- New run: successful QA $0.379039605 + failed attempt $0.018394900 + judge $0.000128000 = **$0.397562505**.
- Previous run: successful QA $0.418345400 + interrupted attempts $0.145778690 + judge $0.000276000 = **$0.564400090**.
- New QA rollout time, including the failed attempt: **800.820 seconds**. Attempt 1 received an API overload error; the existing retry completed attempt 2.
- Inspected all 18 new clip requests and structured answers: one API response per clip, no internal tool calls, 3,803 sampled frames in total, 16 `answered` and 2 `unknown` results. Frame counts, recorded source times, and evidence indices matched the saved frame recipes. Opened two returned evidence images and inspected their actual content.
- Successful main-agent trace: 71 tool calls, comprising 11 read_skill, 12 semantic_search, 27 run_sql, 18 clip_qa, 2 add_note, and 1 submit_answer. Both old and new final answers received score 0.

## Artifacts

- [Comparison JSON](../runs/context-rerollout-day196-one-shot/comparison.json)
- [New final trace](../runs/context-rerollout-day196-one-shot/collection/day/question-196/rollout-00/attempt-2/traces/answer-segment-02.json)
- [Previous final trace](../runs/luna-day-week-tree-month-val-seed0/collection/day/question-196/rollout-00/attempt-1/traces/answer-segment-17.json)
- [W&B run](https://wandb.ai/3dsmile/video-sql-v2/runs/context-rerollout-day196-one-shot)
