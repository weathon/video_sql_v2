# Luna rollout score audit

Audited run: `runs/luna-day-week-tree-month-val-seed0`. Historical comparison:
`/raid/wg25r/video_sql_v2/runs/training_free_luna_seed0_7510e535`.
The collection stopped with 195 scored trajectories: 64 roots and 131 branches,
covering 64 of the intended 400 questions. Five further answers lack judgments.

| Population | Trajectories | Mean score (0–5) | Mean score / 5 | Score >= 4 |
| --- | ---: | ---: | ---: | ---: |
| Current completed roots and branches | 195 | 3.800 | 76.00% | 70.26% |
| Current completed roots only | 64 | 3.563 | 71.25% | 65.63% |
| Historical day/week, all sampled questions | 1,000 | 1.895 | 37.90% | 31.10% |
| Current roots on 33 matched questions | 33 | 3.182 | 63.64% | 60.61% |
| Historical five rollouts on those same 33 questions | 165 | 3.170 | 63.39% | 60.00% |

Matching checked the actual judge question and reference text, not only IDs.
The matched set contains 14 day questions and 19 week questions. Each historical
question has five results, so the historical mean gives each question equal
weight. On matched questions the mean difference is +0.012/5 and the difference
in score >= 4 is +0.61 percentage points. The historical run's `accuracy` field
means score divided by five; it is not the fraction scoring at least four.

The large unmatched difference is consistent with completion selection: these
same 33 questions already scored substantially higher in the historical run
than its full day/week sample. Branches further change question weights and are
correlated with their parents. The 195 completed trajectories cannot establish
an improvement over the historical evaluation. Both runs use gpt-5.6-luna and
the same judge rubric, but historical traces use reasoning medium, whereas the
current traces use high; context and video image budgets also differ.

## Leakage and score checks

- Checked 1,025 saved main-agent request snapshots associated with completed
  results. No reference-answer fields, candidate-answer judge payloads, judge
  calls, or dataset test/val file paths were found in those model inputs.
- Checked the first recorded request in 429 root segments belonging to all 400
  root questions, plus the completed results' 201 memory-retrieval, 28 clip-qa,
  and two sql-qa trace histories. The same explicit leakage-marker checks found
  no matches. Actual SQL calls did not use external-file reader expressions.
- Checked all 131 completed branches against their selected parent snapshots:
  saved prefix, first request, instructions, tools, notes, tool state, prior
  histories, and path response accounting matched. No parent tool calls from
  the branch point onward appeared in the restored prefix. One checked branch
  followed compaction; none exercised a saved note or replaced tool result.
- Checked all 195 current judgments against actual question/reference/candidate
  payloads and submitted scores. Checked the 1,000 historical day/week results
  against their actual judge payloads and submitted scores. No mismatches.
- Manually read the 64 completed root answers against their references and
  inspected selected main-agent, video-agent, and judge tool inputs/returns.
  Viewed the saved source-image evidence for day/72, day/117, and week/102.
  This is not exhaustive independent re-annotation of every video.

The reference answer in `result.json` and outer tracing metadata is expected;
the answering agent's actual input is the relevant leakage boundary. The
reference enters the judge only after an answer is saved. The checks above found
no evidence of answer or future-parent leakage in the inspected trajectories.
They do not establish that every retrieved caption or model inference is true.

## Inspected examples

- **day/72:** Current answer 舌尝思, score 5; the source crop at 63108.021 s
  visibly reads 舌尝思. Historical scores are `[0, 0, 0, 0, 5]`; the scored
  historical rollout 0 instead answered 鼻嗅爱 using footage around 44330 s.
  Its result points to attempt 2, which was inspected; an unscored attempt 1
  must not be substituted for that result's trace.
- **day/117:** Current answer red fungi covering the character, score 5.
  Saved images at 71116.008 and 71152.008 s visibly show red fungal growths on
  the body and weapon. Historical scores are `[0, 5, 0, 0, 5]`; the zero-score
  answers describe green poison or black residue instead.
- **week/102:** Current answer three drawers, score 5. Two inspected images at
  94235.0 and 94236.2 s show three drawer fronts. The video tool first returned
  EOF errors, then successfully returned frames from a narrower interval;
  the answer is based on those returned images. Historical scores are
  `[0, 0, 5, 5, 5]`; the two zero-score answers claim four drawers.
- **day/168:** Current answer attack +39 and critical rate +3 percentage points
  against reference +30 and +10%, score 0. All five historical results also
  score 0. The current judge is not accepting every detailed-looking answer.

The judge evaluates final-answer semantic equivalence. A score of five does
not certify all supporting explanations, timestamps, or intermediate tool
inferences; partial scores also retain subjective judgment.

Machine-readable checks and all 33 matched result-path pairs are in
`runs/luna-day-week-tree-month-val-seed0/score-audit.json`.

## Run interruption

The collection hit `IsADirectoryError` restoring day/165 rollout 02: a prefix's
media list included a temporary `sample-*` directory and restoration attempted
to read it as a file. The saved error is
`collection/day/question-165/rollout-02/attempt-1/error.json` under the run.
This is a collection failure, not evidence of answer leakage.
