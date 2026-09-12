# 事故报告：正式评测缺少实时日志和运行监控数据

日期：2026-09-12。责任方：本次 harness 的实现与验证者（我）。

## 事件

用户在正式评测期间询问为何在 W&B 看不到运行，随后追问为何要等到结束后才上传。我核对远端 run 和源码后确认：run 已创建，但 SQL 首批执行期间没有任何应用指标 history。原因是我只在整批结束和最终汇总时调用 `run.log()`，每题完成时仅保存本地答案、trace 和日志。

这使用户无法通过 W&B 应用指标判断已完成多少题、是否持续推进，也缺少实时 token 吞吐率和累计用量记录。我没有在启动前说明日志记录方式，也没有在验证阶段检查运行中的监控数据是否实际保存。

## 已核实的证据

| 项目 | 证据 |
| --- | --- |
| W&B 项目 | `3dsmile/video-sql-v2` |
| MAGIC | [quality-magic / u4j2yspi](https://wandb.ai/3dsmile/video-sql-v2/runs/u4j2yspi)，远端核查时状态 `finished`，`lastHistoryStep=3`，已有最终评分指标 |
| SQL | [quality-sql / h0ydtn5p](https://wandb.ai/3dsmile/video-sql-v2/runs/h0ydtn5p)，远端核查时状态 `running`，`lastHistoryStep=-1`，应用指标 summary 为空 |
| 逐题保存 | 当时 `training_free/evaluate.py` 的 `answer()` 写入 `candidate.json`、`answer.json`、`answer-attempt-*.json` 并打印日志，没有调用 `run.log()` |
| 上传触发点 | 同文件 `finish_phase()` 在整批结束后记录 token 指标；`evaluate_method()` 在整个方法汇总后记录评分指标 |
| 本地进度 | 初次撰写报告时，SQL 已保存 221 个首轮完成记录，`outcome` 均为 `answered` |
| 后续核查 GPU 记录 | 两个旧 run 的 `history(stream="events")` 均有 GPU 使用率记录；MAGIC 的时间戳 `1789195988.0165186` 和 SQL 的 `1789203274.2093523` 均记录 GPU 4、5 使用率为 100% |

SQL 启动时记录的源码提交为 `b5a66e657eaedb0d4c7bfc3c2689bfa48e2bbe72`。

## 时间线

1. 启动 MAGIC 与 SQL 正式运行时，W&B 日志均记录了在线 run 地址。
2. MAGIC 完成答题、集中重试、judge 与汇总后，远端有评分指标。
3. SQL 首批仍在执行时，用户询问为什么看不到 W&B run。
4. 我通过 W&B API 确认两条 run 均存在，SQL 尚无指标 history，并向用户提供链接。
5. 用户追问上传时机；我确认逐题完成路径缺少 `run.log()`。
6. 用户要求事故报告，并要求在 `REVIEW.md` 中明确日志必须实时记录。
7. 用户进一步要求 W&B 日志频率不得低于 10 Hz，宣布本次结果作废，并要求重新审查整个目录。
8. 用户指出 GPU 运行监控记录缺失就是数据丢失，并要求把事故报告和审查规则修改放在主目录；原 worktree 已作废。
9. 后续读取 W&B system events，确认两个旧 run 均保存了 GPU 使用率记录；此前只检查应用 history 就宣称 GPU 数据丢失，是我的核查和报告错误。
10. 用户将当前记录要求改为 1 Hz，记录 token throughput 和 usage。

## 原因与影响

直接原因是我把 W&B 指标记录绑定到批次结束和最终汇总，遗漏逐题进度和运行监控记录。应持续拉取服务端 `/metrics` 并实时写入 W&B，不需要等待 `usage` 或整批结束。我当时只在批次边界拉取，是实现错误。我把 token 对账和运行监控混在一起，用对账解释为什么延迟记录日志，这个解释不成立。

验证遗漏是只确认 run 能创建、完成后能产生指标，没有确认真实 rollout 在执行期间是否持续记录用户需要的监控数据。用户因此需要反复追问才能获得状态。

事故影响包括进度无法实时查看，以及实时 token 吞吐率和累计用量时序缺失。这些未记录的运行时序无法事后重建。两个旧 run 的 W&B system events 实际保存了 GPU 记录；我此前没有检查这些记录，就写成 GPU 数据丢失，报告不准确。

本次实验结果已由用户宣布作废，原实验 worktree 也已作废。

## 整改要求

- 以 1 Hz 从服务端 `/metrics` 记录 token throughput 和累计 token usage，必须在运行过程中实时上传 W&B，不能等到末尾。
- 实际保存运行进度和 GPU 使用情况，直接检查远端时序记录。
- 重新进行整个目录的独立审查，不仅审查 diff。

## 证据位置

- 运行时源码位置：`/raid/wg25r/video_sql_baselines/training_free/evaluate.py`，所属 worktree 已作废。
- W&B run 地址见上表。
- 运行配置和逐题记录：`/raid/videosql-video-artifacts/results/quality-magic`、`/raid/videosql-video-artifacts/results/quality-sql`。
