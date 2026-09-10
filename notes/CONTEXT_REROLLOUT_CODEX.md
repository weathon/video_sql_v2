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

## Main-prompt replacement test

Commit `15fccd7` adds the replacement workflow to the main prompt: summarize large results with replace_tool before continuing, including useful results; replace invalid results entirely with a reason; add_note does not remove the original result. Reran the same day/196 question once with the existing model, high reasoning, flex tier, and initial skills. The API response echoed the new instructions.

| Metric | Tool-description guidance only | Main-prompt guidance |
| --- | ---: | ---: |
| replace_tool calls | 0 | 3 |
| Main model calls | 20 | 17 |
| Clip QA calls | 18 | 0 |
| Cumulative QA input tokens | 3,064,672 | 1,106,150 |
| QA output tokens | 152,371 | 7,663 |
| Reasoning tokens, included in output | 139,933 | 4,544 |
| Main-agent peak context tokens | 153,713 | 134,388 |
| QA cost, USD | 0.379039605 | 0.046294830 |
| Total cost including attempts and judge, USD | 0.397562505 | 0.046418430 |
| QA time including attempts, seconds | 800.820 | 127.852 |
| Judge score | 0 | 0 |

All three replacements targeted SQL results. API token counts for the individual before/after payloads were 17,369 → 312, 17,237 → 209, and 16,337 → 175: a summed local reduction of **50,247 tokens**, not a measurement of cumulative billed-token savings. Every subsequent request snapshot contained the rewritten target result with `Replaced with:` and omitted the corresponding replace call and acknowledgement.

The replacement text did not follow every instruction: replacement call 24 omitted its reason; call 33 replaced a SQL result covering 14,400–17,400 seconds with a note about 20,880–21,120 seconds from another query. Call 19 also retained anchors outside its target result's returned interval. The new run made no clip QA calls, so its overall cost difference includes a different investigation path.

The run completed on attempt 1 with 41 main-agent tool calls: 5 read_skill, 16 semantic_search, 16 run_sql, 3 replace_tool, and 1 submit_answer. Inspected the actual tool inputs and returned content, all replacement summaries and subsequent request snapshots, and the final answer and judge output.

- [Main-prompt comparison JSON](../runs/context-rerollout-day196-replace-main-prompt/comparison.json)
- [Main-prompt test trace](../runs/context-rerollout-day196-replace-main-prompt/collection/day/question-196/rollout-00/attempt-1/traces/answer-segment-00.json)
- [Main-prompt W&B run](https://wandb.ai/3dsmile/video-sql-v2/runs/context-rerollout-day196-replace-main-prompt)
