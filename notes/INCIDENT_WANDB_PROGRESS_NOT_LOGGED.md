# 事故报告：正式评测未向 W&B 记录实时进度

日期：2026-09-12。责任方：本次 harness 的实现与验证者（我）。

## 事件

用户在正式评测期间询问为何在 W&B 看不到运行，随后追问为何要等到结束后才上传。我核对远端 run 和本地源码后确认：run 已创建，但 SQL 首批执行期间没有任何应用指标 history。原因是我只在整批结束和最终汇总时调用 `run.log()`，每题完成时仅保存本地答案、trace 和日志。

这使用户无法通过 W&B 指标判断已完成多少题、是否持续推进。我没有在启动前说明这一限制，也没有在验证阶段检查首批尚未结束时的远端进度可见性。

## 已核实的证据

| 项目 | 证据 |
| --- | --- |
| W&B 项目 | `3dsmile/video-sql-v2` |
| MAGIC | [quality-magic / u4j2yspi](https://wandb.ai/3dsmile/video-sql-v2/runs/u4j2yspi)，远端状态 `finished`，`lastHistoryStep=3`，已有最终评分指标 |
| SQL | [quality-sql / h0ydtn5p](https://wandb.ai/3dsmile/video-sql-v2/runs/h0ydtn5p)，远端核查时状态 `running`，`lastHistoryStep=-1`，应用指标 summary 为空 |
| 逐题保存 | `training_free/evaluate.py` 的 `answer()` 写入 `candidate.json`、`answer.json`、`answer-attempt-*.json` 并打印日志，没有调用 `run.log()` |
| 上传触发点 | 同文件 `finish_phase()` 在整批结束后记录 token 指标；`evaluate_method()` 在整个方法汇总后记录评分指标 |
| 本地进度 | 撰写报告时，SQL 已保存 221 个首轮完成记录；这些记录的 `outcome` 均为 `answered` |

SQL 启动时记录的源码提交为 `b5a66e657eaedb0d4c7bfc3c2689bfa48e2bbe72`。远端核查与本地 221 题快照发生在不同时点，不能当成同一瞬间的计数对照。

用户最初描述的是“看不到 run”。已经证实的问题是 SQL 没有实时指标 history；我没有查看用户页面，不能把这一证据进一步解释为 run 在用户列表中不可见的原因。

## 时间线

1. 启动 MAGIC 与 SQL 正式运行时，W&B 日志均记录了在线 run 地址。
2. MAGIC 完成答题、集中重试、judge 与汇总后，远端有评分指标。
3. SQL 首批仍在执行时，用户询问为什么看不到 W&B run。
4. 我通过 W&B API 确认两条 run 均存在，SQL 尚无指标 history，并向用户提供链接。
5. 用户追问上传时机；我检查并确认逐题完成路径缺少 `run.log()`，承认这是实现遗漏。
6. 用户要求撰写本报告。

## 原因与影响

直接原因是我把 W&B 指标记录绑定到批次结束和最终汇总，遗漏逐题进度记录。最终 token 需要整批服务端计数对账，并不要求把已完成题数等进度信息也延迟到批次结束。

验证遗漏是只确认 run 能创建、完成后能产生指标，没有确认真实 rollout 在执行期间是否持续向远端记录进度。用户因此需要中断工作并追问才能获得状态。

本次核查已读取实际逐题完成记录，并确认 MAGIC 最终指标已在远端。已确认的影响是实时进度不可见；现有证据没有显示该遗漏造成答案或 trace 丢失。

## 处理状态与整改要求

已完成：核实远端 run、提供实际链接、定位源码中的记录时机，并保存本报告。

尚未实施：在每题尝试完成时向 W&B 记录已完成题数和尝试状态。最终评分与正式 token 对账继续使用既定口径。验收应直接检查真实运行首批尚未结束时的远端 history，确认已有进度记录。

本次事故报告未修改 harness，也未重启或重跑实验。

## 证据位置

- 源码：`/raid/wg25r/video_sql_baselines/training_free/evaluate.py`
- 运行日志：`/raid/wg25r/video_sql_baselines/runs/egolife-baselines/quality-magic.log`、`quality-magic-summary.log`、`quality-sql.log`
- 运行配置和逐题记录：`/raid/videosql-video-artifacts/results/quality-magic`、`/raid/videosql-video-artifacts/results/quality-sql`
