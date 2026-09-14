# EgoLifeQA A1_JAKE 无训练基线结果

本次有效运行使用 Qwen/Qwen3.8-27B-FP8、xhigh，质量评测每种方法并发 20、500 题，每批失败题集中重试一次。视频在 caption 明确区间内均匀采样。旧 quality-magic、quality-sql 已作废。

## 2026-09-13：补齐 500 题与 MM-Lifelong 重评

本节为最终 500 题结果，下方保留补跑前的历史记录。原 SQL60 的维护失败 ID 131、SQL60+PPR 的维护失败 ID 473 各追加一次答题后均获得有效答案，按原 judge v2 分别得 1、2 分。此前的算法失败仍计 0 分：MAGIC 20 题、原 SQL60 的 ID 346、SQL60+PPR 的 ID 74；三组最终维护失败均为 0。

| 方法，原 judge v2 | 平均分 / 5 | 5 分题数 / 分母 | 5 分率 | 算法失败 |
|---|---:|---:|---:|---:|
| MAGIC | 2.2860 | 188/500 | 37.60% | 20 |
| 原 SQL60 | 2.8100 | 227/500 | 45.40% | 1 |
| SQL60 + PPR | 2.6600 | 222/500 | 44.40% | 1 |

ID 131 使用 `783d3101c1be134deb53e5f8a7be9a1aa9fbc5ee` 的原 SQL60 源码和静态 skills；追加尝试用了 60 次调查加一次强答，答题耗时 2703.94 秒。ID 473 保持原 SQL60+PPR 的答题实现，使用 25 次调查，没有强答，耗时 134.06 秒。旧答案、评分、失败记录及 traces 均保留，补齐结果单独保存。

随后仅对 MAGIC 的 480 份有效答案与 SQL60+PPR 的 499 份有效答案重新 judge，其余算法失败计 0，分母均为 500。采用 MM-Lifelong 官方 `eval_acc.py`（dataset revision `248aa82039a574e63a2e524746a7cd8f32330443`）的语义相似度 rubric，保留其 hallucination-sensitive 和 time-duration 特殊规则；输出格式适配 SDK structured parse。继续提供 EgoLife 的提问者 Jake / A1_JAKE 和 query day/time metadata，judge 使用 Qwen/Qwen3.8-27B-FP8、xhigh、并发 20。实际提示见 [mm_lifelong_rubric.txt](../training_free/mm_lifelong_rubric.txt)。

MM-Lifelong 官方映射为：原始 4、5 分计 1，3 分计 0.5，0、1、2 分计 0；映射平均分与 4–5 分比例分别报告。

| 方法，MM-Lifelong rubric | 原始平均分 / 5 | 原始 5 分题数 | 4–5 分题数 / 分母 | 4–5 分比例 | 映射平均分 / 1 |
|---|---:|---:|---:|---:|---:|
| MAGIC | 2.2120 | 136/500 | 187/500 | 37.40% | 0.3960 |
| SQL60 + PPR | 2.6320 | 179/500 | 229/500 | 45.80% | 0.4700 |

双方按同一 500 题比较，SQL60+PPR 的映射平均分高 0.074，4–5 分比例高 8.40 个百分点。MAGIC 原始 0–5 分分布为 `[172, 82, 37, 22, 51, 136]`，SQL60+PPR 为 `[123, 87, 49, 12, 50, 179]`。ID 473 在本次 MM-Lifelong rubric 下得 0 分，包含在结果中。

本次新增调用与费用如下，token 以独立服务端 `/metrics` 的阶段差值为准，输入列已扣除缓存。1,067 次新增 LLM 请求均返回完整 usage，逐请求合计与服务端三类计数的差均为 0。

| 阶段 | LLM 请求 | in | cache | out | 等效 USD |
|---|---:|---:|---:|---:|---:|
| 原 SQL60，ID 131 补跑及 v2 judge | 62 | 421,749 | 1,297,600 | 586,453 | 0.359657 |
| SQL60+PPR，ID 473 补跑及 v2 judge | 26 | 116,930 | 1,115,200 | 27,058 | 0.048100 |
| MAGIC，MM-Lifelong rejudge | 480 | 285,945 | 0 | 339,412 | 0.202415 |
| SQL60+PPR，MM-Lifelong rejudge | 499 | 374,390 | 0 | 534,141 | 0.307205 |
| 本次新增合计 | 1,067 | 1,199,014 | 2,412,800 | 1,487,064 | 0.917377 |

计入原运行、维护补跑及已经执行过的各版 judge 后，累计实际等效费用为 MAGIC **13.667742 USD**、原 SQL60 **21.783350 USD**、SQL60+PPR **25.719855 USD**。原运行缺失 usage 的记录仍保持未知，其服务端费用已包含在累计值内。

四个阶段使用同一服务串行执行，各阶段内部保留并发 20。运行时以 1 Hz 记录 token、吞吐、GPU 和已完成题的实时 overlap；已核查远端历史。最后一次 1 Hz 采样之后产生的少量 token，以阶段结束的服务端快照更新 W&B summary，原始时序保留。最终 overlap 为 500 题。

- [原 SQL60 补跑 W&B](https://wandb.ai/3dsmile/video-sql-v2/runs/331t5e66)：`results/repair-sql60-q131`。
- [SQL60+PPR 补跑 W&B](https://wandb.ai/3dsmile/video-sql-v2/runs/jlafpvz2)：`results/repair-sql60-ppr-q473`。
- [MAGIC MM-Lifelong 重评 W&B](https://wandb.ai/3dsmile/video-sql-v2/runs/8d08zt0l)：`results/quality-magic-mm-lifelong-judge`。
- [SQL60+PPR MM-Lifelong 重评 W&B](https://wandb.ai/3dsmile/video-sql-v2/runs/gk5yaejt)：`results/quality-sql60-ppr-mm-lifelong-judge`。
- 补齐后的 v2 数据：`results/quality-sql60-completed-judge-v2`、`results/quality-sql60-ppr-completed-judge-v2`。所有路径均相对于 `/raid/videosql-video-artifacts`；config 记录来源，完整 trace 继续位于对应原运行或补跑目录。

已核对补跑中的实际 SQL/PPR 输入与返回、SQL QA 响应、强答时关闭工具、实际视频证据帧，以及重评请求中的 rubric、metadata、答案和 reasoning。逐题重算确认两个重评目录分别复用 480、499 份原答案，最终指标均与逐题分数一致。独立子代理审查和 Claude 全仓库审查已完成，记录在 `runs/egolife-completion/claude-full-review.log`；本任务四个服务容器已停止并删除。下一阶段计划已明确：维护错误需要继续重试，EgoLife 最终分母 500，MM-Lifelong Month validation 最终分母 623。

## SQL60 与新版 judge

原 SQL 60 次调查预算运行已完成，源码为 `783d3101c1be134deb53e5f8a7be9a1aa9fbc5ee`，没有 PPR 工具。以下三组均使用包含提问者和 query time 的新版 judge；MAGIC 和 SQL10 复用原答题结果重新评分，SQL60 从头答完 500 题并执行一次集中失败重试。

| 方法 | 平均分 / 5 | 满分题数 / 有效分母 | 满分率 | 算法失败 | 维护失败 |
|---|---:|---:|---:|---:|---:|
| MAGIC，judge v2 | 2.2860 | 188/500 | 37.60% | 20 | 0 |
| 原 SQL10，judge v2 | 2.1948 | 171/498 | 34.34% | 0 | 2 |
| 原 SQL60，judge v2 | 2.8136 | 227/499 | 45.49% | 1 | 1 |

SQL60 与 MAGIC 的共同 499 题上，平均分分别为 2.8136 和 2.2906，满分率分别为 45.49% 和 37.68%，相差 7.82 个百分点。SQL60 的 ID 131 因实际视频 QA 请求返回 HTTP 500 而归为维护失败并排除；ID 346 因模型选择了无源视频覆盖的时间区间，最终未能回答，归为算法失败并记 0 分。两题均已用完集中重试，分类依据保存在各题 `failure-review.json`。

498 个最终答案中，14 个来自 60 次调查后的额外强答，占 2.81%；旧 SQL10 为 351/498，即 70.48%。本次首轮 500 次尝试触发强答 15 次，含集中重试共 515 次尝试、触发 15 次。已查看实际强答 trace，工具列表为空。

SQL60 的全部答题、子调用、judge 和失败重试共 11,665 次实际 LLM 请求。服务端计数为未缓存输入 46,371,267、缓存输入 229,426,400、输出 22,972,725 token，等效费用 **21.423693 USD**。8 个请求没有返回 usage，逐请求记录仍为未知；已返回 usage 的三类 token 合计与服务端计数差均为 0。答题和 judge 交错执行，以 `answer_and_judge` 记录混合阶段，不把全部 token 计作答题。

对原答案重新评分的新增费用分别为 MAGIC 0.175762 USD、SQL10 0.273161 USD；其原答题和原 judge 账目保留在下方历史结果中。

[SQL60 W&B 运行](https://wandb.ai/3dsmile/video-sql-v2/runs/way68ll5) 的 1 Hz 在线 token、GPU 和已完成题的 MAGIC 同题对比已在实际运行中核查。结果目录为 `results/quality-sql60-judge-v2`，完整账目与预算统计为其中的 `token-log.json`、`final-audit.json`。新版 judge 的旧答案重评分别为 [MAGIC](https://wandb.ai/3dsmile/video-sql-v2/runs/mu0lvo1y) 和 [SQL10](https://wandb.ai/3dsmile/video-sql-v2/runs/s10wp9nx)。

SQL30 和 SQL+PPR15 均已按用户要求停止并标记作废，部分 artifacts 保留。下一阶段在记录本节结果后，以支持 pre/post SQL 的 PPR 替换语义搜索入口，仅作为语义搜索工具，不默认或优先推荐。

## SQL60：PPR 仅替换语义搜索

源码 `99078ceef2dda6f1956d2121bd22137e07111140` 保留原 SQL 工具，以支持 pre/post SQL 的 PPR 替换 `semantic_search`，不默认或优先推荐。全部 500 题从头答题，并发 20、60 次共享调查调用加一次强答，使用同一新版 judge；首轮 19 题失败，集中重试一次，共 519 次尝试。

| 方法 | 平均分 / 5 | 满分题数 / 有效分母 | 满分率 | 算法失败 | 维护失败 |
|---|---:|---:|---:|---:|---:|
| SQL60 + PPR 语义搜索 | 2.6613 | 222/499 | 44.49% | 1 | 1 |

与 MAGIC 的共同 499 题上，满分为 222/499 对 188/499（44.49% 对 37.68%），平均分为 2.6613 对 2.2906。与原 SQL60 的共同 498 题上，满分为 222/498 对 227/498（44.58% 对 45.58%），平均分为 2.6667 对 2.8193；该比较排除了两轮各自的维护失败题。

最终 ID 74 请求了没有原视频覆盖的区间，归为算法失败并记 0 分；ID 473 的 SQL QA 模型请求被服务端断开且未返回响应，归为维护失败并排除。已核对实际工具输入、返回值、源视频覆盖和异常链，依据保存在各题 `failure-review.json`。498 个最终答案中 9 个来自额外强答，占 1.81%；首轮触发 8 次，含重试共 9 次。实际强答请求的工具列表为空。

本轮全部答题、子调用、judge 和重试共 11,800 次 LLM 请求。服务端计数为未缓存输入 55,395,126、缓存输入 210,875,200、输出 29,109,102 token，等效费用 **25.364550 USD**。6 个请求的 usage 不完整，其中 5 个没有响应，保持未知。服务端比已知 usage 合计多 8,752 个输入（含 2,400 个缓存输入）和 2,252 个输出 token，均已纳入费用，未将缺失 usage 记零。

[本轮 W&B](https://wandb.ai/3dsmile/video-sql-v2/runs/ue7d0mwh) 在运行过程中以 1 Hz 记录 token、吞吐、GPU 和 MAGIC 同题对比，已从远端历史核查。结果目录为 `results/quality-sql60-ppr-semantic-judge-v2`，含原始 trace、`token-log.json`、`final-audit.json`、实际帧检查记录和服务日志。完成后的汇总曾因人工审阅记录的字段类型错误退出；修正该记录后复用全部答案与 judge 完成汇总，没有增加答题或评分请求。

以下为原 judge、SQL10 及其串行计时的历史结果。

## 原 judge 的质量

| 方法 | 平均分 / 5 | 满分题数 / 有效分母 | 算法失败 | 维护失败 |
|---|---:|---:|---:|---:|
| MAGIC | 1.9700 | 91/500 | 20 | 0 |
| SQL | 2.0904 | 94/498 | 0 | 2 |

MAGIC 的 20 道最终算法失败包括 17 道 context 溢出和 3 道空答案，均记 0 分。SQL 的两道最终维护失败为 ID 84、468：服务断开连接且没有返回回答，已从分母排除。每题分类均保存实际 trace 审阅证据。

旧 SQL 10 次调查预算下，498 个最终有效答案中有 351 个来自额外的强制回答调用，占 70.48%。首轮 500 次尝试中触发 348 次；含集中重试的 511 次尝试中共触发 354 次。统计逐一核对实际模型请求与最终采用的尝试，明细保存在 `results/turn-budget-audit/sql10.json`。

## Token 与等效费用

以下质量账目含主调用、子调用、judge 和失败重试，以独立服务端计数为准。输入列已经扣除缓存输入。

| 方法 | 未缓存输入 in | 缓存输入 cache | 输出 out | 等效 USD |
|---|---:|---:|---:|---:|
| MAGIC | 77,974,821 | 7,996,000 | 3,117,884 | 13.289565 |
| SQL | 27,291,489 | 40,292,800 | 8,482,436 | 8.725153 |

费用公式为 `(in × 0.15 + out × 0.47 + cache × 0.016) / 1,000,000`。MAGIC 有 60 个请求未返回 usage，SQL 有 11 个；这些请求保持未知状态。MAGIC 已返回 usage 的合计与服务端计数一致。SQL 服务端比已返回 usage 合计多 32,193 个输入和 27,187 个输出 token，这部分已计入上表，无法逐请求分配。

计时和小范围调试的独立账目见同目录 JSON；与质量账目分开保存。

## 串行 20 题计时

seed 0，固定同一顺序、并发 1，先 MAGIC 后 SQL。每种方法开始前清理 GPU/CPU prefix KV、encoder 与运行时查询缓存。时间不含 judge。

| 方法 | 获得最终答案 | 20 题最终答案总时间 | 平均时间 | 首轮执行总时间（含失败） |
|---|---:|---:|---:|---:|
| MAGIC | 19/20 | 不可用 | 不可用 | 1343.97 s |
| SQL | 20/20 | 1118.24 s | 55.91 s | 1118.24 s |

MAGIC 第 216 题首轮 89.67 s、集中重试 45.93 s，均为 context 溢出，未获得最终答案。MAGIC 包含该次重试的全部答题尝试时间为 1389.89 s。没有补跑第三次或替换题目。

| ID（执行顺序） | MAGIC 最终答案耗时 / s | SQL 最终答案耗时 / s |
|---|---:|---:|
| 433 | 114.69 | 68.19 |
| 198 | 28.44 | 82.97 |
| 389 | 20.21 | 66.07 |
| 456 | 53.17 | 64.19 |
| 216 | 失败（89.67 + 45.93） | 88.74 |
| 21 | 75.30 | 34.44 |
| 133 | 105.47 | 17.89 |
| 495 | 30.59 | 19.37 |
| 262 | 58.34 | 103.13 |
| 249 | 91.72 | 29.57 |
| 208 | 103.83 | 45.66 |
| 471 | 53.99 | 32.24 |
| 402 | 152.58 | 47.10 |
| 425 | 19.13 | 132.25 |
| 156 | 26.01 | 70.87 |
| 496 | 49.31 | 38.44 |
| 245 | 91.43 | 33.81 |
| 184 | 59.43 | 58.50 |
| 299 | 49.10 | 46.43 |
| 457 | 71.57 | 38.38 |

## 日志与复现

运行时以 1 Hz 从 `/metrics` 拉取 token throughput 和累计 usage，并记录 GPU 使用率；W&B 在线记录已在真实 rollout 执行期间核查。原始抓取为 `live-metrics.prom`，数值时序为 `live-metrics.jsonl`。逐题答案、原始请求 usage、完整 traces、缓存清理响应和失败审阅保存在 artifacts 的对应运行目录。

- [MAGIC 质量运行](https://wandb.ai/3dsmile/video-sql-v2/runs/0bwm7q7z)：`results/quality-magic-rerun`
- [SQL 质量运行](https://wandb.ai/3dsmile/video-sql-v2/runs/rkh09j0e)：`results/quality-sql-rerun`
- [串行计时](https://wandb.ai/3dsmile/video-sql-v2/runs/qgyz10hz)：`results/latency-rerun`
- 私有 HF artifacts：`weathon/egolife-a1-jake-magic-videosql-qwen3vl8b`，上传核验记录为根目录 `upload-result.json`。
- 实际依赖版本保存在 `training_free/requirements.txt`；入口及操作步骤见项目 README。

SQL 收尾时 Weave 的进度等待停在 3 条未配对开始记录；其正常退出流程随后将这 3 条记录发送到服务端。收尾代码已改为先执行 SDK flush，再结束日志。答题和评分不重跑，原有请求和服务端 token 账目保留。
