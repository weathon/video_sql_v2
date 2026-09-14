# Fair comparsion against baseline

# 前情提要
刚才另一个agent完成了刚才另一个plans/baselines_plan.md, 目录在./training_free/, 他eval的是MAGIC Video和我们的方法在EgoLife上面的eval, 部分代码偏离了plan, 这是后期修正并且是intended. 

文件说明
| 文件 | 用途 |
|---|---|
| [harness.py](/raid/wg25r/video_sql_trace2skill/training_free/harness.py) | 答题 agent、工具、共享调用预算 |
| [evaluate.py](/raid/wg25r/video_sql_trace2skill/training_free/evaluate.py) | 批量运行、judge、重试、W&B、token 对账 |
| [evaluate_mm_official.py](/raid/wg25r/video_sql_trace2skill/training_free/evaluate_mm_official.py) | 本次 Open 评分的 GPT judge prompt、EgoLife metadata、分数解析与映射 |
| [prepare_baselines.py](/raid/wg25r/video_sql_trace2skill/training_free/prepare_baselines.py) | 数据转换、建库、embedding、逐题时间过滤 |
| [index_schema.py](/raid/wg25r/video_sql_trace2skill/training_free/index_schema.py) | DuckDB schema、时间转换 |
| [ppr.py](/raid/wg25r/video_sql_trace2skill/training_free/ppr.py) | 支持 pre/post SQL 的 PPR |
| [magic.py](/raid/wg25r/video_sql_trace2skill/training_free/magic.py) | MAGIC 基线适配 |
| [skills/](/raid/wg25r/video_sql_trace2skill/training_free/skills) | 静态工具使用说明 |
| [trace2skill_service.py](/raid/wg25r/video_sql_trace2skill/infra/trace2skill_service.py) | 四个本地服务的启动配置 |

# 本次任务
为了与 MAGIC Video 基线进行公平比较，我们需要使用他们的模型。MAGIC Video 使用了 qwen 3.5 flash（即 Qwen/Qwen3.5-35B-A3B 的host名称）。因此，请在 4 块 GPU 上为 Qwen/Qwen3.5-35B-A3B-FP8 搭建 vLLM 服务，并采用最简单的配置方式（无需遵循 infra.txt 中的复杂设置, 见最下面如何使用, 可以稍微调整）。embedding model依旧使用qwen 3 vl embedding. 替换 vlm 模型并将所有 semantic search 改为非 PPR，在这个模型上面运行:

Setting 1:
Harness: MAGIC Video，非 PPR semantic search
Dataset: Open-EgoLifeQA
Model: Qwen/Qwen3.5-35B-A3B-FP8

Setting 2:
Harness: MAGIC Video，非 PPR semantic search
Dataset: MCQ EgoLifeQA
Model: Qwen/Qwen3.5-35B-A3B-FP8

Setting 3:
Harness: VideoSQL 60，非 PPR semantic search
Dataset: Open EgoLifeQA
Model: Qwen/Qwen3.5-35B-A3B-FP8

Setting 4:
Harness: VideoSQL 60，非 PPR semantic search
Dataset: MCQ EgoLifeQA
Model: Qwen/Qwen3.5-35B-A3B-FP8

4个setting同时运行, 打到同一个vllm server和embedding server. 每个并发设置成8192. 这是客户端并发上限，不是 vLLM 的 max-num-seqs；每组只有 500 题，不复制题目来填满并发。
最终报告一个简单的table, 包含4个setting每个setting的metric. MCQ EgoLife 报告正确选项数 / 500；Open EgoLife 报告映射后 1 分题数 / 500 和 Average score（0–1）。
全部使用现有的artifacts. 

# 数据与非 PPR semantic search

- 四组均使用 A1_JAKE 同一组 500 题及相同题目 ID。Open 复用已处理开放题与正确选项文本，答题模型不接收选项或参考答案；MCQ 使用原题及原顺序选项，正确选项仅用于评分。
- 保留逐题 query time 截止、现有数据库和共享向量；视频继续使用按 caption 明确区间裁剪的缓存片段与均匀采样。缺少必需数据或向量应报错，不能跳过检索分支继续答题。
- Setting 1–4 中 MAGIC 与 VideoSQL 的所有 semantic search 均使用非 PPR 的普通向量相似度检索，不新增实验组。先限定当前逐题数据库允许的节点 ID，再计算向量相似度和排名；VideoSQL 按模型所选表/模态检索并分页。文本使用文本向量，视频使用已有视频向量，查询 embedding 继续使用 Qwen3-VL-Embedding-8B 和既有文本规范化。
- 所有普通 semantic search 均不调用 PPR 图传播、BM25 或 chain 检索。VideoSQL 不保留 PPR 的 pre_sql/post_sql 参数，保留独立 run_sql；semantic_search 仅在需要语义匹配时使用，不默认推荐。工具说明和静态 skills 必须与实际普通向量接口及 entity、episode、semantic、visual_clip schema 一致，不能直接复制 src/ 的旧表名或 row_id。
- VideoSQL 保留原非 PPR 语义工具的排名、分数筛选和可选结果 QA 行为；额外结果 QA 属于调查模型调用。每次尝试共享 60 次调查调用，再额外一次关闭工具的强制回答；主 agent、clip_qa、SQL QA、检索结果 QA 均在实际发出模型请求前占用预算，reasoning 历史继续回传。
- MAGIC 的 semantic search 同样替换为普通向量相似度检索，不使用 PPR；保留独立的 BM25、chain、时间检索分支、原检索轮数与现有时间截止语义。
- 所有答题主调用、MAGIC retriever/responder、时间解析、clip_qa 和结果 QA 都使用同一个 Qwen/Qwen3.5-35B-A3B-FP8 服务；保留既有 xhigh 和 reasoning 历史，不设置客户端采样参数或输入/输出 token 上限。Open judge 独立使用下述 OpenAI 模型。

# Open GPT judge：如何评分

采用刚才实际运行的 `gpt-5` judge，通过项目 `.env` 的 `SHARED_OPENAI_KEY` 调用 OpenAI API；不再使用 `gpt-5.6-terra`。刚才运行返回的模型标识为 `gpt-5-2025-08-07`。参考实现为 `training_free/evaluate_mm_official.py`，正式配置采用刚才运行的 judge 并发 128，独立于答题并发 8192。初始请求超时 60 秒，不设置 temperature、top_p、其他采样参数、reasoning_effort 或 token 上限；重试采用下节的新规则。

System prompt 复用该脚本中的 MM-Lifelong 官方 `eval_acc.py` prompt，来源版本为 `MM-Lifelong/MM-Lifelong@248aa82039a574e63a2e524746a7cd8f32330443`，并保留其 EgoLife metadata 附加规则。语义评分标准如下：

| 原始分数 | 标准 |
|---|---|
| 0 | 无语义相似性，完全无关、矛盾或未回答问题 |
| 1 | 只有相关话题或关键词，没有回答参考答案的核心内容 |
| 2 | 捕捉到少量次要内容，但遗漏或误述核心事实 |
| 3 | 捕捉主要意思，但遗漏重要细节或有明显不准确 |
| 4 | 主要意思和大部分关键细节正确，仅有次要遗漏或不重要的不准确 |
| 5 | 与参考答案语义等价，包含全部必要信息，无实质遗漏或错误 |

保留官方特殊规则：hallucination-sensitive 问题只有所有必要项均正确才给 5，否则给 0；时间/时长问题按题目规定的容差判断，超出容差给 0。

User prompt 按以下顺序构造，不传检索 trace 或独立选项：

```text
Question: <开放题>
Metadata: <JSON metadata>
Groundtruth answer: <正确选项对应的答案文本>
Candidate answer: <本组模型最终答案>
Your response:
```

Metadata 包含 `questioner=A1_JAKE`、`questioner_name=Jake`、第一人称 narrator/camera wearer、原始 `query_time_raw_DHHMMSSFF`、`query_day`、`query_time_of_day=HH:MM:SS` 和 `query_frame_index`。I/me/my、narrator、camera wearer 均指 Jake；用 query day/time 对齐 yesterday、last night 等相对时间；DAYn 是日期标签，不是 n 天前。

Judge 输出 `Analysis:` 和 `Final Score:` 后的 0–5 整数；沿用刚才脚本的文本解析，优先匹配 `Final Score:\s*([0-5])`，未匹配时按官方规则查找首个独立 0–5 数字，没有有效分数即评分失败并重试。保存完整请求、响应、分析、原始分数、原始 usage 和每次尝试。

原始 4/5 分映射为 1，3 分映射为 0.5，0/1/2 分映射为 0。Open 准确率（perfect score portion）是映射后 1 分题数 / 500，Average score 是映射分数总和 / 500，不以原始 5 分率或原始 0–5 平均分替代。MCQ 不调用 GPT judge，使用 SDK structured parse 得到一个合法选项标识，与原始正确选项直接比较。

# 错误与重试

- 不再按“算法错误”与“维护错误”分流或排除题目。所有答题执行错误都进入重试，包括 API/服务错误、context 溢出、空答案、结构化输出错误、缺失证据或检索失败；每题初始尝试加最多三次整题重试，共最多四次，批次结束后集中重试剩余失败题。
- 模型能在当前 rollout 内纠正的工具错误原样回传，后续调查调用继续计入 60 次预算。整题重试隔离可变状态，每次重试重新获得同样的 60+1 调用预算。合法但答错的答案或低 judge 分数不是执行错误，不能据此重新答题择优。
- 四次答题尝试耗尽后仍无有效答案，记录最终失败和全部错误 trace；该题保留在 500 分母中，Open 记 0，MCQ 记错误。只报告失败数量与原因，不计算两类错误指标。
- Judge 不设重试次数上限，每次重试延长请求超时；超时、API 失败、空响应或无有效分数均重试。保留已生成答案，仅重试 judge，不重新答题；有效分数（包括 0）立即保存并结束该题评分。保存各次实际超时设置，恢复时保留尝试历史。
- Judge 尚未成功的题目属于评分未完成，不得从分母剔除或伪造 0 分。最终 Open 结果必须覆盖 500 题（含答题耗尽已记 0 的题目），不能直接采用旧脚本排除 `gpt_score=-1` 后的 `valid_samples` 分母。
- 答案、judge 和每次尝试完成即落盘，恢复复用已完成阶段，不覆盖原始 traces。

# 实时 W&B 与服务统计

- 继续以 1 Hz 实时记录 W&B，不能等结束再上传；四个 setting 分开保存结果和进度。Open overlap 为 Setting 1 与 3 已落盘且评分完成的共同 ID，MCQ overlap 为 Setting 2 与 4 已落盘且评分完成的共同 ID；分别记录实际 overlap 数、两侧准确率，Open 另记两侧 Average score。没有 overlap 时只记录数量，不编造准确率。
- Throughput 使用共享 vLLM 服务日志中的 prompt/generation throughput，设置并核验实际服务日志周期为 1 秒。保存包含 throughput 的完整行及同一统计周期的 MTP/speculative decoding、KV cache、prefix cache 等独立统计行；仅筛选含 throughput 的行会漏掉 MTP。按 engine/rank 去重；服务已给聚合值时不能再叠加各 rank，只有分 rank 值时才在同一统计周期求和。保留各 rank 的命中率/占用率，不能相加百分比。实现依据：[vLLM 日志源码](https://docs.vllm.ai/en/latest/api/vllm/v1/metrics/loggers/)。
- 四组共用服务的 throughput 是服务总量，不作为四个 setting 各自独立 throughput；不能用客户端已完成请求的 token 差值替代服务日志 throughput。
- 累计 token usage 继续每秒直接抓取本地服务 `/metrics`，以共享服务总量核对四组全部请求的原始 usage 总和，不能把共享服务计数重复归入每组。OpenAI judge 的 usage 单独记录，来源为实际 API 响应，不混入本地 vLLM 计数。未返回 usage 保持未知，不按零处理。

# Review 后的实施核对点

- 当前 `training_free/harness.py` 的 semantic_search 实际调用 PPR；只修改 setting 名称不会切换算法，实施时必须修改工具、skills 和 evaluator 中的检索标识，并覆盖 MAGIC 适配及检索路径中的全部 semantic search，检查两种方法的真实检索 trace，确认均未调用 PPR。
- 当前 GPT judge 源码并发为 32、总尝试数为 3、固定超时 60 秒，且聚合时排除评分失败；刚才的实际运行配置为并发 128。实施时需要按本计划修改为并发 128、judge 无上限重试并逐次延长超时、最终分母 500。
- MCQ 需要恢复选项输入并验证合法选项输出，不能直接复用删除选项后的 Open 输入。四组的新答案都必须由本次 Qwen3.5 服务生成，现有答案只用作历史对照。
- [MAGIC 上游 README](../third_party/magic_video/README.md) 分别列出 retriever 为 `openai/gpt-oss-120b`、responder 为 `qwen/qwen3.5-flash-02-23`。本计划共享 Qwen3.5 服务的比较是统一模型后的 harness 比较，不能称为完整复现上游全部模型配置。Qwen3.5-Flash 对应 35B-A3B 的托管版本，但上下文和托管功能另有设置：[Qwen 模型卡](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/README.md)。
- 正式启动时核对四块主服务 GPU、embedding 服务资源、媒体挂载和可用内存；资源不足立即报告。固定实际 vLLM 版本并验证 DP4、EP、MTP2、CPU KV offload、工具/思考解析和真实视频请求可同时工作；不因命令能启动就认定兼容。CPU offload 依赖 prefix caching，命令中显式启用，并查看实际 offload 日志确认：[vLLM connector 源码](https://docs.vllm.ai/en/latest/api/vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector/)。

# vLLM Commands

CPU KV cache 四个 rank 合计 512 GB；按十进制 GB 计算，每个 rank 128,000,000,000 bytes，合计 512,000,000,000 bytes。`cpu_bytes_to_use_per_rank` 是整数，不是列表。该预算属于共享 vLLM 服务，不能给每个 setting 重复分配。

```bash
vllm serve Qwen/Qwen3.5-35B-A3B-FP8 \
  --trust-remote-code \
  --enable-expert-parallel \
  --data-parallel-size 4 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --mm-encoder-tp-mode data \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --kv-transfer-config '{"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use_per_rank":128000000000,"lazy_offload":false}}'
```
