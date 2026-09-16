# MM-Lifelong Month 基线结果

完成日期：2026-09-16。评测保留 Month validation 全部 623 个原始 ID、问题和答案；每题检索全部 14 个视频，不使用标注区间或逐题截止时间。两边复用同一批 9,168 条已有视觉向量，视觉查询沿用旧版 messages 与检索指令；视频证据直接从原视频读取。

回答模型为 `Qwen/Qwen3.8-27B-FP8`、`xhigh`，每边并发 20。MAGIC 最多五轮检索；SQL 使用 vector semantic search 与 SQL 过滤，共享 60 次调查调用后允许一次强制回答。保留原 Qwen judge 结果；按用户追加要求，用 fair 的 `gpt-5` 官方 judge 对保存答案重评分，返回模型为 `gpt-5-2025-08-07`。GPT-5 请求省略 reasoning effort、采样参数和 token 限制，提示移除 EgoLife 身份元数据，评分输出沿用 fair 的 Analysis / Final Score 格式。

## 最终评分

原始 4/5 分映射为 1，3 分映射为 0.5，0/1/2 分映射为 0。所有指标分母固定为 623。

| Judge | 方法 | 映射为 1 的题数 | 映射分数总和 | Rounded Score score-1 ratio | Average rounded score |
|---|---|---:|---:|---:|---:|
| Qwen3.8 | MAGIC | 135 | 142.5 | 0.21669342 | 0.22873194 |
| Qwen3.8 | SQL60 | 227 | 240.5 | 0.36436597 | 0.38603531 |
| GPT-5 | MAGIC | 146 | 154.5 | 0.23434992 | 0.24799358 |
| GPT-5 | SQL60 | 239 | 251 | 0.38362761 | 0.40288925 |

MAGIC 最终有 621 个答案；ID 30、526 在普通失败重试后仍返回空内容，计 0 分并保留在分母。19 题在第二次尝试获得答案。SQL 最终 623 题全部获得答案，ID 257、657 在第二次尝试成功。维护中断均在原方法下续跑，未缩小题集。GPT-5 对 621 个 MAGIC 答案和 623 个 SQL 答案分别评分，没有 judge 重试或失败；全部复制答案与 Qwen 阶段逐项一致。两套 judge 的最终 overlap 均为 623 题。

## Token 对账

下表为正式运行，包含回答、子调用、judge、普通重试和维护中断；输入 token 包含缓存输入。调试调用保存在独立目录，不计入下表。

| 统计来源 | 方法 | 输入 | 其中缓存输入 | 输出 |
|---|---|---:|---:|---:|
| Qwen 独立 PD 计数 | MAGIC | 63,007,468 | 18,286,400 | 9,378,009 |
| Qwen 独立 PD 计数 | SQL60 | 439,297,557 | 334,565,600 | 65,397,547 |
| GPT-5 重评分响应 usage | MAGIC | 311,819 | 0 | 201,818 |
| GPT-5 重评分响应 usage | SQL60 | 416,045 | 0 | 274,610 |

两个 PD 的进程启动时间在各阶段保持一致，首尾计数差与分阶段总计完全相同。中断阶段的响应 usage 未全部保留：MAGIC 的 PD 总计比已保存响应多 7,972 输入、0 缓存输入、2,516,719 输出；SQL 多 2,043,178 输入、1,104,000 缓存输入、3,478,085 输出。差额保留在 PD 总计中，未恢复逐请求归属。SQL 有 88 条请求记录缺少完整 usage；MAGIC 被终止时尚未落盘的调用未进入响应清单。未中断的收尾及重试阶段对账差额为 0，两组 GPT-5 重评分对账差额也均为 0。

按实际响应区分主调用、SQL QA、clip QA、judge 的统计保存在 [Qwen 对账记录](../runs/mmlifelong-month/qwen-accounting-audit.json)，GPT-5 记录见 [重评分核验](../runs/mmlifelong-month/gpt5-final-audit.json)。完整逐请求 usage、阶段快照及维护记录保留在各运行目录的 `token-log.json` 和 `phases/`。

## 核验与留存

核验四组结果的 623 个 ID 与原运行配置完全一致，独立重算映射指标及最终 overlap。检查全部答题输入记录均仅含原问题和空 query_time；SQL 保存的调用计数最大为 60 次调查、1 次强答。手工检查实际检索输入和返回、跨视频身份、reasoning 历史及 judge 请求；包括 MAGIC ID 697 的蒙古犬种证据、SQL ID 0 的 SQL/语义检索和四次视频调用，并按正式 trace 的 PTS 读取视频 21 的 0 秒及 18380.02994661328 秒源帧。前期调试还覆盖小数秒、同视频不同片段、旧 EgoLife 时间模式和已有视觉向量复用。四组运行的监控日志实际间隔中位数均约为 1 秒。

原运行与 GPT-5 重评分的逐题结果、完整 traces、日志和 artifacts 均保存在已忽略的 [runs/mmlifelong-month](../runs/mmlifelong-month)，未上传 HF。原计划只回填四个结果字段。

| 运行 | W&B | 本地目录 |
|---|---|---|
| MAGIC / Qwen judge | [0martfc5](https://wandb.ai/3dsmile/video-sql-v2/runs/0martfc5) | `runs/mmlifelong-month/results/month-magic` |
| SQL60 / Qwen judge | [q0yhc5qg](https://wandb.ai/3dsmile/video-sql-v2/runs/q0yhc5qg) | `runs/mmlifelong-month/results/month-sql` |
| MAGIC / GPT-5 judge | [zbrd2sbo](https://wandb.ai/3dsmile/video-sql-v2/runs/zbrd2sbo) | `runs/mmlifelong-month/results/month-magic-gpt5` |
| SQL60 / GPT-5 judge | [s1akzlg4](https://wandb.ai/3dsmile/video-sql-v2/runs/s1akzlg4) | `runs/mmlifelong-month/results/month-sql-gpt5` |
