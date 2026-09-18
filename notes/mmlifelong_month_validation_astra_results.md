# MM-Lifelong Month validation with Astra skills

The held-out Month validation completed all 623 questions. The 29 skills were synthesized with `gpt-6-astra` from the existing 200 Week training experiences, then adapted to Month video IDs and local timestamps. The skill taxonomy was unrestricted. The answers used `Qwen/Qwen3.8-27B-FP8`, `xhigh`, 60 shared investigation calls plus one forced final answer, concurrency 20, and at most two algorithm attempts. Each question has one final answer and no branches. Retrieval covered the complete Month corpus without a query-time cutoff.

GPT-5 judged the original question, groundtruth, and candidate with the MM-Lifelong official rubric. The actual judge request omitted optional sampling parameters and reasoning effort. Each saved judgment has exactly one valid `Final Score` matching its saved score.

The comparison uses the same 623 question IDs in both runs. Accuracy means score 4 or 5. The mapped score is 1 for 4/5, 0.5 for 3, and 0 for 0/1/2. Deltas are Astra skills minus untrained.

| Metric | Untrained Month baseline | Astra skills | Delta |
| --- | ---: | ---: | ---: |
| Questions | 623 | 623 | 0 |
| Correct answers | 239 | 235 | -4 |
| Accuracy | 38.362761% | 37.720706% | -0.642055 percentage points |
| Mapped mean score | 0.402889245586 | 0.394863563403 | -0.008025682183 |
| Raw mean score (0–5) | 2.356340288925 | 2.290529695024 | -0.065810593900 |
| Terminal failures | 0 | 0 | 0 |

Astra score counts for 0 through 5: `215, 80, 71, 22, 36, 199`. Its raw score sum is 1427 and mapped score sum is 246. The baseline sums are 1468 and 251.

## Artifacts and verification

- Validation W&B: https://wandb.ai/3dsmile/video-sql-v2/runs/syogjm1k (finished).
- Untrained reference W&B: https://wandb.ai/3dsmile/video-sql-v2/runs/s1akzlg4.
- Validation results: `runs/mmlifelong-month-validation-astra/validation-month-astra/`.
- Baseline results: `runs/mmlifelong-month/results/month-sql-gpt5/`.
- Final skills: `training_free/skills-astra-month/`; original synthesis: `runs/analysis-astra/skills/`.
- Independent recomputation from all per-question files matched `overlap-comparison.json` and W&B. All 623 answer, judgment, and result files agree; there are no terminal failure files.
- Actual traces were inspected for SQL, vector search and filtering, skill loading, video QA, and tool-result replacement. A saved video-frame recipe from question 1 was reconstructed with the harness and visually checked. All completed answer records stayed within the 60-call investigation and one-call final budgets; 91 used the forced final call.

## Recovery and cleanup

Two vLLM streaming failures, on question IDs 694 and 790, reported a missing tool name (`None`) in `ResponseFunctionToolCallItem`. Service health checks remained successful. The same run resumed with unchanged settings and reused completed answers and judgments. Question 257 exceeded the context window on its first algorithm attempt and completed on attempt 2. The other 622 final answers used algorithm attempt 1.

The run retains `usage/complete=0` and `usage/requests_without_response=43` for incomplete request-usage records from interruptions; token totals are not a complete cost comparison. All question scores are complete.

The user explicitly waived the otherwise required Claude review and the additional review for this skill restart. The three dedicated `month-validation-astra-{proxy,encoder,pd}-sql` containers were stopped and manually removed after verification. The shared `mmlifelong-month-embedding` service remains available with model `qwen3-vl-embedding-8b`.

Raw requests, responses, traces, frame recipes, and logs remain in ignored run directories. This document is the compact final result.
