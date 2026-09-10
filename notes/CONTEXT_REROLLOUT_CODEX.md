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

## Ten additional rollouts

Ran the user-requested ten independent root rollouts of day/196 with the same main prompt, initial skills, model, high reasoning, flex tier, and existing retry/concurrency settings. The full-repository Claude CLI review reported no blocking issues. No repository source was changed for this batch.

The initial run name produced a 66-character prompt_cache_key; the API rejected it with a 64-character maximum. All ten initial traces contained zero completed model responses and zero tool calls. The failed launcher was stopped after its Weave flush stalled. Shortening the run name to `day196-replace-main-10` produced a 41-character key; that batch completed all ten rollouts.

Input/output and clip API counts below cover each successful QA path; cost includes failed attempts and judge. Rollout IDs are zero-based.

| Rollout | replace_tool | Clip API calls | Input tokens | Output tokens | Judge score | Total cost, USD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 00 | 0 | 4 | 2,100,978 | 40,412 | 0 | 0.205997330 |
| 01 | 4 | 9 | 2,629,101 | 70,115 | 0 | 0.250767115 |
| 02 | 15 | 14 | 2,031,448 | 101,152 | 0 | 0.242872300 |
| 03 | 6 | 1 | 1,241,926 | 13,231 | 0 | 0.070282825 |
| 04 | 0 | 7 | 1,465,172 | 51,336 | 0 | 0.740979500 |
| 05 | 6 | 16 | 2,431,845 | 121,213 | 5 | 0.491117900 |
| 06 | 3 | 9 | 2,225,546 | 70,799 | 0 | 0.208540040 |
| 07 | 9 | 9 | 1,947,104 | 86,568 | 2 | 0.192567280 |
| 08 | 0 | 9 | 1,648,616 | 78,047 | 0 | 0.159205150 |
| 09 | 4 | 6 | 950,083 | 31,892 | 0 | 0.069363425 |
| Total | 47 | 84 | 18,671,819 | 664,765 | 1 score 5, 1 score 2, 8 score 0 | 2.631692865 |

### Replacement checks

- Seven of ten rollouts used replace_tool. All 47 calls completed: 26 replaced SQL results and 21 replaced clip QA results. Failed attempts contributed no additional replacements in this batch.
- Checked every replacement argument and acknowledgement against the saved target result and all 378 subsequent request snapshots in which the target remained present. All contained the expected `Replaced with:` result and omitted the replace call and acknowledgement. No mismatch was found in those checks.
- Every replacement reduced its individual payload token count. Summed before/after reduction: **331,740 tokens**. This is a local payload measurement, not cumulative billed-token savings.
- Rollout 03 replaced a duplicate-column SQL error with an invalid-result reason. Rollout 07 replaced four unsupported clip answers; inspected representative returned images for all four, which showed a shrine, an equipment overview, a root menu, and a shrine menu rather than the claimed readable treasure transitions.

### Observed problems

- **Replacement is still inconsistent.** Rollouts 00, 04, and 08 made zero replacement calls. Their largest SQL results contained 62,640, 64,801, and 39,249 characters, respectively. Rollouts 00 and 08 also left failed SQL/tool results unreplaced.
- **Some replacement summaries mix evidence from other results.** In rollout 03, replace call 43 targeted tool 41: its 100 returned event-step rows covered 60,060–62,910 seconds and contained no 芭蕉扇, but the summary added 芭蕉扇 dialogue at 66,570–66,600 seconds. Call 42 similarly added 定风珠 at 16,920–16,950 to rows ending at 16,410 that contained no 定风珠. In rollout 07, call 37 added 绣花针 at 61,740 to a result ending at 15,810 with no 绣花针. These were checked against the actual returned row text, not only the time-range metadata.
- **Two rollouts required a second attempt.** Rollout 04 directly emitted prose instead of invoking submit_answer; AnswerSubmission JSON validation rejected it. Rollout 05 received an API overload error. Both completed on the existing retry path.
- **Tool errors occurred and were returned to the model.** There were five over-budget clip requests and five SQL errors across all attempts. SQL errors included nonexistent columns, duplicate column names, treating plain text as JSON, and an invalid alias. No clip trace reported missing video. All 117 completed clip API artifacts across both successful and failed QA attempts contained one completed response, no internal tool calls, and matching frame/image counts with in-range evidence indices.
- **Shutdown was not clean despite exit 0.** After BATCH COMPLETE, the log recorded two asynchronous-generator closing errors ending in `RuntimeError: generator didn't stop after athrow()`. All ten answer/result/judge artifacts were already present; the shutdown error was not counted as a successful check.

Read all ten final answers and judge outputs. The sole score-5 answer identified 凤翅将军 and the actual change 辟火罩 → 芭蕉扇; inspected its boss HUD and equipped-item evidence images. The score-2 answer included additional menu-browsing selections in its replacement sequence.

- [Batch summary](../runs/day196-replace-main-10/summary.json)
- [Metrics and per-rollout errors](../runs/day196-replace-main-10/audit-metrics.json)
- [Replacement inputs, outputs, and replay checks](../runs/day196-replace-main-10/replacement-audit.json)
- [Clip request/answer audit](../runs/day196-replace-main-10/clip-audit.json)
- [Batch W&B run](https://wandb.ai/3dsmile/video-sql-v2/runs/day196-replace-main-10)
