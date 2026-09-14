# Fair comparsion against baseline

# 前情提要

另一个 agent 已完成 [baselines_plan.md](baselines_plan.md)，实现位于 `training_free/`。该实现比较 MAGIC-Video 和 VideoSQL 在 EgoLifeQA A1_JAKE 500 题上的表现；后期修正是当前 intended 行为。本阶段只替换答题模型，保留已经确认的 artifacts、检索算法、视频均匀采样、reasoning 回传、60+1 调查预算和 trace 语义。

| 文件 | 用途 |
|---|---|
| [harness.py](/raid/wg25r/video_sql_trace2skill/training_free/harness.py) | 答题 agent、工具、共享调用预算 |
| [evaluate.py](/raid/wg25r/video_sql_trace2skill/training_free/evaluate.py) | 批量运行、失败重试、W&B、token 对账 |
| [evaluate_mm_official.py](/raid/wg25r/video_sql_trace2skill/training_free/evaluate_mm_official.py) | OpenAI `gpt-5` 开放题 judge |
| [prepare_baselines.py](/raid/wg25r/video_sql_trace2skill/training_free/prepare_baselines.py) | 数据转换、建库、embedding、逐题时间过滤 |
| [index_schema.py](/raid/wg25r/video_sql_trace2skill/training_free/index_schema.py) | DuckDB schema、时间转换 |
| [ppr.py](/raid/wg25r/video_sql_trace2skill/training_free/ppr.py) | 支持 pre/post SQL 的 PPR |
| [magic.py](/raid/wg25r/video_sql_trace2skill/training_free/magic.py) | MAGIC 基线适配 |
| [skills/](/raid/wg25r/video_sql_trace2skill/training_free/skills) | 静态工具使用说明 |

# 比较设置

MAGIC-Video 使用 qwen 3.5 flash；本阶段加载 `Qwen/Qwen3.5-35B-A3B-FP8`，对客户端公布的 served model name 为 `Qwen/Qwen3.5-35B-A3B`。主 agent、MAGIC retriever/responder、`clip_qa`、SQL result QA 和其他答题子调用全部改用该 served model。reasoning 保持 `xhigh` 并保留历史 reasoning；客户端不设置采样参数或输入、输出 token 上限。Embedding 继续使用现有 `Qwen3-VL-Embedding-8B` 向量和服务，不重新生成图、数据库、文本向量或视频向量。

四个 setting 同时运行，共享同一主模型服务和 embedding 服务，每个 setting 并发 10：

| Setting | Harness | Dataset | Model |
|---|---|---|---|
| 1 | MAGIC-Video | Open EgoLifeQA | Qwen/Qwen3.5-35B-A3B |
| 2 | MAGIC-Video | MCQ EgoLifeQA | Qwen/Qwen3.5-35B-A3B |
| 3 | VideoSQL 60 + PPR semantic search | Open EgoLifeQA | Qwen/Qwen3.5-35B-A3B |
| 4 | VideoSQL 60 + PPR semantic search | MCQ EgoLifeQA | Qwen/Qwen3.5-35B-A3B |

Open 与 MCQ 都使用 A1_JAKE 的同一 500 个 ID 和 query time。Open 使用现有 `questions.json`。MCQ 从现有 `/raid/wg25r/egolife/EgoLifeQA/EgoLifeQA_A1_JAKE.json` 读取原题、A–D 选项和答案字母，生成独立的紧凑 question manifest；复用同一批逐题 DuckDB、MAGIC 索引、向量和视频 artifacts。MCQ 向两种方法提供题干及带 A–D 标签的四个选项，要求结构化返回单个选项字母；MAGIC 使用其上游 MCQ responder prompt，Open 继续使用已适配的开放题 prompt。MCQ 只按预测字母与正确字母精确匹配计算准确率，不调用 LLM judge。

# 错误重试与固定分母

不再区分算法错误、维护错误或其他错误，也不因错误类型排除题目。答题阶段的任何失败，包括 API/服务错误、context overflow、工具错误、缺失最终答案、结构化输出无效和 MCQ 选项无效，都在该 setting 的整批结束后集中重试；每题执行一次初始尝试和最多三次相同设置的重试。每次尝试使用隔离的可变检索状态并完整保存请求、响应、工具输入输出、错误和 usage。

三次重试后仍无有效答案的题保留在固定的 500 题分母中。Open 将其候选答案记为空并最终记 0；MCQ 将其记为错误答案。错误链只用于 trace 和问题诊断，不产生影响分母或评分的 failure category。

# Open EgoLifeQA judge

开放题在四个答题 setting 完成全部重试后，对 MAGIC 和 VideoSQL 的保存答案分别运行同一个 OpenAI judge。使用 `.env` 中的 `SHARED_OPENAI_KEY`，模型为 `gpt-5`，每个方法依次运行，方法内并发 128、每题最多三次 judge 尝试、单次超时 60 秒。请求省略 `temperature`、`top_p`、其他采样参数、reasoning 参数和输入/输出 token 上限。

Judge 使用 MM-Lifelong 官方 `eval_acc.py` 的 system prompt、user prompt 结构、文本输出和解析规则，并追加 EgoLife metadata：提问者为 Jake/A1_JAKE，第一人称 `I/me/my/mine`、narrator 和 camera wearer 都指 Jake；提供 query time 原始 DHHMMSSFF、query day、时分秒和 frame index；相对时间按该 query time 解释，`DAYn` 是日期标签，不表示 n 天前。User prompt 依次包含 `Question`、`Metadata`、`Groundtruth answer`、`Candidate answer` 和 `Your response`。模型先输出简短 `Analysis`，再输出 `Final Score` 的 0–5 整数；优先解析 `Final Score`，否则采用官方的首个独立 0–5 数字回退规则。

原始分数规则如下：

| 原始分数 | 规则 |
|---|---|
| 0 | 无相似性；答案无关、矛盾、为空或完全未回答问题 |
| 1 | 仅有相关主题或关键词，没有表达参考答案的主要含义 |
| 2 | 回答了少量次要方面，但遗漏或误述核心事实 |
| 3 | 捕捉主要含义，但有若干重要遗漏或明显不准确 |
| 4 | 主要含义和大部分关键细节正确，仅有轻微遗漏、简化或非关键错误 |
| 5 | 与参考答案语义等价，覆盖全部必要信息且无实质错误 |

保留官方特殊规则：hallucination-sensitive 问题只有全部要求项正确才得 5，任一项错误、缺失或幻觉即得 0；时长问题只接受题目规定范围内的误差，超出范围得 0。最终映射为原始 4/5 → 1，原始 3 → 0.5，原始 0/1/2 → 0。`Average score` 是 500 题映射分数的平均值；open accuracy/perfect score portion 是映射为 1 的题数除以 500。Judge 的任何错误都按官方三次尝试预算重试；三次均失败的题保留在分母中并记 0。

# W&B、overlap 与 token

四个 setting 从启动开始在线同步到 W&B，运行过程中完成一题就立即落盘。每秒记录各 setting 的 started/active/completed/failed、GPU utilization/memory/power、累计 token usage 和已落盘结果的 overlap。Open 的 overlap 是 setting 1 与 3 已落盘 judge 结果的 ID 交集；MCQ 的 overlap 是 setting 2 与 4 已落盘答案结果的 ID 交集。每次从当前交集重新计算双方准确率；Open 还计算双方 `Average score`。不使用最终结果反填早期 overlap。

主 vLLM 的累计 prompt、cached prompt 和 generation token 以每秒直接读取 `/metrics` 的 counter 为准，正式总账使用运行前后 counter 差值；不以逐响应 usage 求和替代服务端总数。逐响应原始 usage 仍保存在 trace 中，用于核对缺失响应。四个 setting 共享服务，因此 vLLM token 与 throughput 是四组任务的服务级总量，不拆分归因。

Throughput 不通过完成 token 的相邻差值计算。vLLM 日志每次产生 throughput/status 行时，原样保存整行并将其中全部指标同步到 W&B，包括 prompt/generation throughput、running/waiting、GPU KV cache、prefix cache hit rate 和 MTP stats；报告为共享服务总 throughput。W&B 在运行过程中持续同步，不在任务结束时才上传。

# 服务

启动前检查 GPU、CPU 内存、端口和现有容器；若不能满足两块 GPU 和合计 512 GiB CPU KV cache，立即报告，不等待或降低规格。主模型使用两块 GPU，因此 `data-parallel-size` 为 2；CPU KV cache 合计 512 GiB，每个 rank 为 256 GiB，即 274877906944 bytes。主服务端口为 8103，embedding 服务端口为 8114。使用独立容器名，不使用 Docker `--rm`；结束后停止并手动删除容器。

```bash
vllm serve Qwen/Qwen3.5-35B-A3B-FP8 \
  --served-model-name Qwen/Qwen3.5-35B-A3B \
  --host 0.0.0.0 \
  --port 8103 \
  --trust-remote-code \
  --enable-expert-parallel \
  --data-parallel-size 2 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --mm-encoder-tp-mode data \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --kv-transfer-config '{"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use_per_rank":274877906944,"lazy_offload":false}}'
```

# 验证与报告

正式运行前分别对四个 setting 做小范围真实调用，人工检查主 agent、MAGIC 检索与 responder、PPR、SQL、SQL QA、视频工具、reasoning 回传、MCQ 选项输入与预测解析、60+1 强答、所有错误重试、Open judge metadata、原始/映射分数、每秒 `/metrics` usage、vLLM 完整 throughput 行和实时 overlap。不能只检查完成数量或最终文件。

中等源码修改后由 fresh subagent 按 `REVIEW.md` 审查 diff；正式运行前由 Claude CLI 按 `REVIEW.md` 和本计划审查整个仓库，不只审查 diff。最终报告一个四行表：两个 MCQ setting 报告 `correct / 500` 和准确率；两个 Open setting 报告映射 1 分题数 `/ 500`、perfect score portion 和 `Average score`。另外分别报告四个 setting 耗尽三次重试后的题数，以及共享服务总 token 和 throughput；原始 traces、逐题结果与服务日志留在忽略目录，Git 只提交紧凑最终报告。
