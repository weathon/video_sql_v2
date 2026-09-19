# Online Week skills training

`src/train_online.py` trains skills from Week's 200 questions in 100 sequential batches. The saved random order samples without replacement. Each batch contains two questions with five independent pre rollouts and five independent post rollouts per question. Month is held out.

Training starts from a copy of `training_free/skills-vector`. A batch reads one unchanged skills version throughout both rollout phases. The pre phase produces one general experience per question; only that question's experience is appended to its post prompt. After the post phase, fresh teachers revise the two experiences, and a separate editor updates a candidate skills copy for the next batch. There is no third rollout phase. The post scores measure the injected pre experience, not the subsequently revised experience or edited skills.

Qwen uses the existing Week investigation budget, thinking/history, retry policy, and tool implementations. The teacher and editor use `gpt-5.6-sol`, high effort, and normal service tier through the Codex SDK. The two per-question teachers run concurrently with read-only access and separate MCP contexts. The editor has a candidate-directory workspace and automatic approval for the requested outside-sandbox `apply_patch`. Completed Codex sessions are archived while their records remain available.

## Credit assignment and information flow

GPT-5 judges each candidate with the official rubric and a strict `Final Score` parser. Scores 4/5 map to 1, 3 maps to 0.5, and 0/1/2 map to 0. Each trajectory advantage is its mapped reward minus the mean of its question's five rollouts in the same phase. The teacher assigns evidence-supported local contributions in `[-1, 1]` to meaningful actual tool calls; these need not sum to the trajectory advantage.

Credit assignment also records error categories and whether the answer is grounded in the actual evidence. A visual misidentification does not make preceding correct retrieval or SQL negative. A lucky correct answer does not make the whole trace positive. The private assessment locates the originating error and distinguishes it from downstream propagation.

Within each phase, the teacher contrasts steps and strategies across successful and unsuccessful trajectories, lists the actual rollout IDs supporting each pattern, and records counterexamples and applicability limits. Official success means mapped reward 1. These associations and local contribution scores remain heuristic judgments; they are not labeled as replay- or ablation-verified effects.

Teachers receive the complete original question metadata, answers, judgments, actual requests/results, and clue intervals. Clue times are mapped through source video plus local time into EgoLife seconds. The student's normal input contains the original question and skills; post adds only the public procedural experience. Private attribution records retain specific evidence references. Public experiences contain applicability, dos, don'ts, and steps, with an explicit teacher leakage audit. The editor receives the two final public experiences and current skills.

MCP uses `VIDEO_SQL_SKILL_DIR` for its skill files, ROOT guidance, and catalog. `VIDEO_SQL_RUN_DIR` places its UUID trace directories under the calling teacher's attempt. The existing MCP defaults continue to use the original skills and `runs/mcp` outside online training.

## Foreground execution

Use two controlled foreground sessions. Before launch, check current GPUs, ports, memory, and the shared embedding service. The service launcher rejects occupied GPUs, insufficient memory, a busy port, or the wrong embedding model. The GPU IDs below are candidates and must be checked at execution time.

```bash
mkdir -p runs/online-training/services
.venv_baselines/bin/python -u infra/online_service.py --gpus 0,1 --run-dir runs/online-training > runs/online-training/services/qwen.log 2>&1
```

After the Qwen health/models endpoint at port 8310 is ready:

```bash
.venv_baselines/bin/python -u -m src.train_online --run-dir runs/online-training > runs/online-training/training.log 2>&1
```

The service is one TP=2 `Qwen/Qwen3.8-27B-FP8` container named `online-training-qwen`. It reuses the existing Week serving settings and the shared embedding endpoint at port 8214, using a separate GPU with a 32 GiB VRAM budget. A separate small debugging run can select a pair with `--ids 0 1 --run-dir runs/online-training-debug`; it still uses the complete five-pre/five-post protocol. Debug artifacts and skills are separate from the formal training run.

After actual final-artifact verification, stop and manually remove only the task's Qwen container. Preserve the shared embedding service.

## Artifacts, metrics, and recovery

The run directory stores the configuration and fixed question order, initial skills copy, per-batch metadata, each question's original metadata and mapped clues, pre/post rollouts, judgments, analysis manifests and private credit assignments, public experiences, and editor attempts. Only a normally completed editor result advances the skills chain. Interrupted attempts are retained; completed answers, judgments, analyses, and edits are reused on resume. Teacher retries use fresh sessions and exact error feedback; candidate edits start from the unchanged input skills.

Run the same command to resume. Retryable student failures exhaust at four attempts and receive score zero. Teacher retries exhaust at four attempts and stop the batch. Deterministic failures and missing required services stop execution. Resolve the saved failure explicitly before resuming an exhausted or non-retryable stage; restarting does not silently reset its retry allowance.

W&B records phase metrics as soon as each round is graded. Batch accuracy is the fraction of its ten rollouts with mapped reward 1. It also records mean mapped and raw scores, the proportion of the two questions with at least one correct rollout, and post-minus-pre deltas. Batch completion records cumulative paired metrics, completion counts, failures/retries, investigation/final/tool calls, elapsed time, token/cache counts, and throughput. Missing provider usage remains explicitly incomplete.

`runs/` is ignored. Intermediate skills, requests, responses, logs, and per-question outputs stay there. The final skills copy is `final-skills/`, with compact completion metrics in `summary.json`. Commit only the final skills deliverable and compact final report after verification.

For this run, the user explicitly waived the Claude repository review. The fresh diff review must finish normally with an explicit result before formal execution. Live debugging must inspect actual SQL/vector/video responses, teacher evidence reads, public experience leakage, prompt injection, skills edits, and resume behavior. A successful process exit alone does not establish those paths worked.
