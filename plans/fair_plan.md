# Fair comparsion against baseline

# 前情提要
刚才另一个agent完成了刚才另一个plans/baselines_plan.md, 目录在./training_free/, 他eval的是MAGIC Video和我们的方法在EgoLife上面的eval, 部分代码偏离了plan, 这是后期修正并且是intended. 

文件说明
| 文件 | 用途 |
|---|---|
| [harness.py](/raid/wg25r/video_sql_trace2skill/training_free/harness.py) | 答题 agent、工具、共享调用预算 |
| [evaluate.py](/raid/wg25r/video_sql_trace2skill/training_free/evaluate.py) | 批量运行、judge、重试、W&B、token 对账 |
| [prepare_baselines.py](/raid/wg25r/video_sql_trace2skill/training_free/prepare_baselines.py) | 数据转换、建库、embedding、逐题时间过滤 |
| [index_schema.py](/raid/wg25r/video_sql_trace2skill/training_free/index_schema.py) | DuckDB schema、时间转换 |
| [ppr.py](/raid/wg25r/video_sql_trace2skill/training_free/ppr.py) | 支持 pre/post SQL 的 PPR |
| [magic.py](/raid/wg25r/video_sql_trace2skill/training_free/magic.py) | MAGIC 基线适配 |
| [skills/](/raid/wg25r/video_sql_trace2skill/training_free/skills) | 静态工具使用说明 |
| [trace2skill_service.py](/raid/wg25r/video_sql_trace2skill/infra/trace2skill_service.py) | 四个本地服务的启动配置 |

# 本次任务
为了与 MAGIC Video 基线进行公平比较，我们需要使用他们的模型。MAGIC Video 使用了 qwen 3.5 flash（即 Qwen/Qwen3.5-35B-A3B 的host名称）。因此，请在 4 块 GPU 上为 Qwen/Qwen3.5-35B-A3B-FP8 搭建 vLLM 服务，并采用最简单的配置方式（无需遵循 infra.txt 中的复杂设置, 见最下面如何使用, 可以稍微调整）。embedding model依旧使用qwen 3 vl embedding. 替换vlm模型, 在这个模型上面运行:

Setting 1:
Harness: MAGIC Video
Dataset: Open-EgoLifeQA
Model: Qwen/Qwen3.5-35B-A3B

Setting 2:
Harness: MAGIC Video
Dataset: MCQ EgoLifeQA
Model: Qwen/Qwen3.5-35B-A3B

Setting 3:
Harness: VideoSQL 60
Dataset: Open EgoLifeQA
Model: Qwen/Qwen3.5-35B-A3B

Setting 4:
Harness: VideoSQL 60
Dataset: MCQ EgoLifeQA
Model: Qwen/Qwen3.5-35B-A3B

4个setting同时运行, 打到同一个vllm server和embedding server. 每个并发设置成20. log到wanbd用同样的计划,overlap部分计算已经落盘的overlap. wandb记录throughput记录总throughput, 通过vllm的log获得(直接截取包含throughput的line, 同时包括那一行的所有metric, 包括MTP stats, kv cache usage, prefix cache hit rate, etc), 而不是通过完成token差值获得. 直接记录日志整行及其中指标，不额外跨 rank 汇总。 不允许设置reasoning effort. 
最终报告一个简单的table, 包含4个setting每个setting的metric. MCQ Egolife报告准确度. open egolife报告准确度(perfect score portion)和平均分. 
全部使用现有的artifacts. 

仅将 VideoSQL 的 semantic_search 替换为非 PPR 的普通向量检索；MAGIC 保留原检索。

MCQ 从最开始提供选项；输出无法判定或包含多个答案时，整题重新开始，沿用初始尝试后最多重试 3 次的规则。

对于 open 问题，judge 使用 OpenAI API 的 `gpt-5`，用项目 `.env` 里的 `SHARED_OPENAI_KEY`；省略采样参数和 reasoning effort。prompt 与 rubric 使用 MM-Lifelong 官方 `eval_acc.py`，沿用 [evaluate_mm_official.py](../training_free/evaluate_mm_official.py) 中的 EgoLife metadata 规则。

Judge 输入为问题、metadata（提问者 Jake/A1_JAKE 与 query time）、正确答案文本和模型答案；按 metadata 解释第一人称和相对时间。输出 Analysis 和 Final Score（0–5），按该脚本解析；4/5 分映射为 1，3 分映射为 0.5，0/1/2 分映射为 0。Open 报告 1 分题数 / 500 和 Average score（映射分数总和 / 500）。

所有答题错误均重试，不再区分算法错误与维护错误；初始尝试后最多重试 3 次，共 4 次。耗尽后仍失败的题目记 0，最终分母必须为 500。Judge 没有重试次数上限，每次重试延长超时，复用已生成答案。

# 执行中的决定
- VideoSQL 的 semantic_search 改为非 PPR 的普通向量检索：恢复原 SQL60 的实现（commit 783d3101 的按表余弦检索及其 skills）。
- MCQ：题干后附 Options 和作答要求，走现有 normalize 路径；MAGIC 最终回答用上游原版 MCQ 模板；正确=5 分、错误=0 分，本地判定不调用 judge。
- vLLM：不加 limit mm per prompt。
- overlap：对比并行运行中的 MAGIC run 已写盘的逐题 judgment。
- 阻塞项一（DP=2 会关闭 vLLM stats logging，日志里没有 throughput 行）：改 TP=2 两个容器，一个给 MAGIC 用一个给 SQL 用，分开 log。
- 阻塞项二（MCQ 字母抽取规则）：要求最后输出 Answer: 看结果，如果提取失败就单独 call 一次 llm 来 parse。
- VideoSQL 在 MCQ 上多次以纯文本消息结束而没有调用 submit_answer：接受最终纯文本消息作为答案。

# vLLM Commands
CPU KV cache 四个 rank 合计 512 GB，每个 rank 128,000,000,000 bytes。

```bash
vllm serve Qwen/Qwen3.5-35B-A3B-FP8 \
  --trust-remote-code \
  --enable-expert-parallel \
  --data-parallel-size 4 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --mm-encoder-tp-mode data \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --kv-transfer-config '{"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use_per_rank":128000000000,"lazy_offload":false}}'
```

# 结束程序
所有程序正常结束时必须运行5/0代表正确完成
