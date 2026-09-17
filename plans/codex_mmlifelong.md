# Codex VideoSQL：MM-Lifelong Month 与 EgoLife Open 评测

## 1. 目标、上下文与已确认设置

使用 Codex 作为 VideoSQL 的主 agent harness，先完成 MM-Lifelong Month validation，再运行 EgoLife Open。两阶段均复用现有数据库、视频和 embedding，与各自已有的 Qwen3.8 VideoSQL、GPT-5 judge 结果比较。

仓库：`/raid/wg25r/video_sql_trace2skill`。使用 `.venv_baselines/bin/python`；已安装 Python Codex SDK `openai-codex 0.154.0`。HF cache 固定为 `/raid/huggingface`。

实现前阅读：

- [上一阶段计划](/raid/wg25r/video_sql_trace2skill/plans/baseline_mmlifelong.md)：Month 数据协议、embedding 复用和已完成结果。
- [MCP 与 analysis 交接说明](/raid/wg25r/video_sql_trace2skill/docs/mcp-analysis-handoff.md)：Codex 接入、权限、事件记录和 thread 归档。
- [src/mcp.py](/raid/wg25r/video_sql_trace2skill/src/mcp.py)：stdio MCP 工具封装。
- [src/analysis.py](/raid/wg25r/video_sql_trace2skill/src/analysis.py)：Codex SDK 生命周期；这里只借鉴客户端实现，不使用分析或技能生成提示。
- [training_free/harness.py](/raid/wg25r/video_sql_trace2skill/training_free/harness.py)：现有 SQL、向量检索、QA、视频取证及数据集时间处理。
- [fair 计划](/raid/wg25r/video_sql_trace2skill/plans/fair_plan.md)：EgoLife Open 对照协议。

两阶段设置如下：

| 项目 | MM-Lifelong Month | EgoLife Open |
|---|---|---|
| 题集 | 原始 validation 全部 623 题 | 已转换好的 A1_JAKE Open 全部 500 题 |
| 主 agent | `gpt-5.6-terra` | 同左 |
| Codex 设置 | `max`、`service_tier="default"` | 同左 |
| QA 模型 | `Qwen/Qwen3.8-27B-FP8` | 同左 |
| QA reasoning effort | `xhigh` | 按 EgoLife fair，省略此参数 |
| 并发 | 5 道题 | 5 道题 |
| 答题尝试 | 初始尝试＋批末一次普通失败重试 | 按 EgoLife fair，最多 4 次尝试 |
| Judge | OpenAI API `gpt-5` | 同左 |
| 检索 | vector semantic search＋SQL 过滤，无 PPR | 同左 |
| 时间范围 | 全部 14 个视频，无逐题截止 | 使用原 query time 和逐题截止数据库 |

其他已确认决定：

- Terra 使用现有 analysis 的设置(把high改成max)，不将不同模型的 reasoning 档位视为可直接比较的量。
- 使用 Codex 原生循环和上下文管理，不设置 SQL60 共享调用预算，不提供 `replace_tool`，不增加额外强制回答步骤。
- 使用原静态检索 skills；本轮不加载 analysis 生成的 skills。
- 客户端不设置输入、输出 token 上限或采样参数。Qwen 保留 thinking；Codex 保留原生会话历史及可获取的 reasoning 事件。
- 参考答案、旧答案、judge 结果仅由 evaluator 使用。
- 本轮只运行 VideoSQL；不重跑 MAGIC、MCQ、embedding 或技能学习。

## 2. 现有 artifacts、embedding 来源与复现信息

### 2.1 MM-Lifelong Month

完整 artifact 根目录：

`/raid/wg25r/video_sql_trace2skill/runs/mmlifelong-month/artifacts`

| 相对路径 | 内容 |
|---|---|
| `questions.json` | 623 个原始 ID、问题、参考答案和空 query time |
| `index.duckdb` | 两种基线共用的完整 Month 数据库 |
| `embeddings/visual_clip.video_vectors.npy` | 实际视觉向量，`9168 × 4096`、float32 |
| `embeddings/visual_clip.row_ids.npy` | 视觉向量对应的节点 ID |
| `embeddings/*.vectors.npy` | 文本向量；`visual_clip.vectors.npy` 是 caption 文本向量 |
| `embeddings/*.row_ids.npy` | 各类向量行与数据库节点的对应关系 |
| `embedding-shards/text/` | 本次生成的文本向量 shards |
| `embedding-shards/video-grouped/` | 从旧数据库导入的视觉向量 shards |
| `magic/graph.pkl`、`magic/visual_embeddings.pkl` | MAGIC 图和视觉向量映射 |
| `magic/captions/` | 合并后的各粒度 captions |
| `videos/month/`、`media-map.json` | 14 个完整视频及视频身份、时间映射 |
| `video-metadata/` | 视频帧率、时长等元数据 |

视频 ID 为 `4、11–23`。所有问题都允许检索这 14 个视频，不使用 `clue_interval`、`total_intervals` 等标注，不构造 query time。时间使用视频内秒数并保留小数，证据和片段通过视频 ID 与起止时间区分。

视觉向量直接复用了：

`/raid/wg25r/video_sql/data/mm_lifelong_stage1/mm_lifelong.sqlite`

从 `visual_clips` 与 `visual_clips_emb` 按 `broadcast_id/start_sec/end_sec` 对应，保留已有向量，不重新生成。

原生成方式：

- 模型：`Qwen/Qwen3-VL-Embedding-8B`。
- 每段均匀采样 16 帧，长边缩放至 420，JPEG 编码。
- 一条 user 消息先放入以下检索指令，再放入 16 张图，生成一个片段向量：

```text
Retrieve video segments that help answer questions about a long-form livestream.
```

- 当前视觉查询采用同样的 `messages` 格式，内容为该指令、换行和查询文本；文本查询沿用普通 `input` 格式。
- 原生成脚本：[/raid/wg25r/video_sql/build/embed_mm_lifelong_stage1.py](/raid/wg25r/video_sql/build/embed_mm_lifelong_stage1.py)。
- 原记录：[/raid/wg25r/video_sql/docs/HANDOFF_MM_LIFELONG_STAGE1.md](/raid/wg25r/video_sql/docs/HANDOFF_MM_LIFELONG_STAGE1.md)。
- 已完成四段重算验证，记录位于 [comparison.json](/raid/wg25r/video_sql_trace2skill/runs/mmlifelong-month/debug/existing-embeddings/comparison.json)。本轮使用保存的原向量，不以重算结果替换。

Month captions、OpenIE、semantic、topic/event chains 来源：

```text
/raid/huggingface/hub/datasets--jiazhengli7--magic-video-artifacts/snapshots/8573f317684ce55c0999f86386666d3d3fe76ae3/mmlifelong
```

原始 MM-Lifelong 数据集快照：

```text
/raid/huggingface/hub/datasets--MM-Lifelong--MM-Lifelong/snapshots/248aa82039a574e63a2e524746a7cd8f32330443
```

原题为其中的 `month/val.json`。

上一轮运行配置：

- [MAGIC config.json](/raid/wg25r/video_sql_trace2skill/runs/mmlifelong-month/results/month-magic/config.json)
- [SQL config.json](/raid/wg25r/video_sql_trace2skill/runs/mmlifelong-month/results/month-sql/config.json)

重建入口为 [prepare_baselines.py](/raid/wg25r/video_sql_trace2skill/training_free/prepare_baselines.py)，`--time-format mmlifelong` 的阶段顺序为：

```text
prepare → media → embed-text → embed-video → assemble → databases
```

其中 `embed-video` 导入旧 SQLite 向量。此处保留重建信息供复现，本轮跳过上述全部阶段，直接使用现有 artifacts。

### 2.2 EgoLife Open

完整 artifact 根目录：

`/raid/videosql-video-artifacts`

| 相对路径 | 内容 |
|---|---|
| `questions.json` | 已转换好的 500 道开放题、原 ID、query time、正确答案文本 |
| `questions/<ID>.duckdb` | 对应题目的截止前数据库 |
| `index.duckdb` | 完整数据库，供核验使用 |
| `embeddings/*.row_ids.npy`、`*.vectors.npy` | 现有文本向量及 ID 映射 |
| `embeddings/visual_clip.video_vectors.npy` | 现有视觉向量，`6223 × 4096`、float32 |
| `media-map.json`、`videos/A1_JAKE/` | 原视频与逻辑时间映射 |
| `clip-map.json`、`videos/caption-clips/A1_JAKE/` | 旧 embedding 流程留下的 caption 片段及映射 |
| `qa-rule-audit.json` | 既有开放题转换核验记录 |

直接使用保存的开放题，不重新去选项、不改写问题或答案、不加载 MCQ 选项。

EgoLife 时间协议：

- 数据库使用 `DHHMMSSFF`，帧号按对应源视频实际 fps 转换。
- `clip_qa` 使用自 DAY1 00:00:00 起的秒数。
- 每题打开 `questions/<ID>.duckdb`，向量检索只使用该数据库允许的 ID。
- 视频取证不得越过原 query time。
- 提问者为 Jake/A1_JAKE；回答提示及 judge 保留相应身份与时间信息。

EgoLife 向量同样由 `Qwen/Qwen3-VL-Embedding-8B` 生成。原视觉向量通过 caption 片段的 `video_url` 请求和均匀采样生成；本轮只复用保存的向量。EgoLife 文本、视觉查询均沿用现有 `input` 请求格式，不加入 Month 的 livestream 检索指令。

相关原始路径：

```text
原视频：/raid/wg25r/egolife/A1_JAKE
原 QA：/raid/wg25r/egolife/EgoLifeQA/EgoLifeQA_A1_JAKE.json
原 merge artifacts：
/raid/huggingface/hub/datasets--jiazhengli7--magic-video-artifacts/snapshots/8573f317684ce55c0999f86386666d3d3fe76ae3/egolife/A1_JAKE
```

索引说明见 [/raid/videosql-video-artifacts/INDEX.md](/raid/videosql-video-artifacts/INDEX.md)。

两套 artifact 都只读使用；不新建裁剪视频缓存。视频 QA 从对应源视频按实际区间读取。这些数据库、向量、视频和原始 traces 不随源码提交，需要保留本机完整目录。

## 3. 实现：Codex harness 与 evaluator

### Codex harness

新增 `training_free/codex_harness.py`，同时承载 Codex 客户端适配和本实验 stdio MCP 入口。

- 参考 `src/mcp.py` 的封装方式，复用 `training_free/harness.py` 的工具实现；不直接使用目前固定 EgoLife 全库、无逐题截止的 `src/mcp.py` 配置。
- 暴露现有 `run_sql`、vector `semantic_search`、`clip_qa`、`add_note`、`read_notes`、`read_skill`、`agent_raise`、`submit_answer`。
- 工具 schema 从实际实现生成；保留 SQL scope、检索后 SQL 过滤、结果格式、视频采样及证据返回规则。视频 QA 保持现有 video-mode 请求。
- 按数据集注入 schema、时间说明、视频身份和原静态 skills 路由。Month 使用已有 Month 专用 SQL/semantic 说明；EgoLife 使用原 `skills-vector` 说明。
- Codex 专用 context 只记录调用次数，不执行 SQL60 调用限制。两种数据集的 QA 请求均不带客户端 token 上限；特别清除 EgoLife 旧 QA 路径中的 16,384 上限及其截断重试逻辑。
- 每题每次尝试创建独立 Codex client、thread、MCP context 和输出目录，隔离 notes、调用计数及证据。
- 固定 Terra/max/default，设置 `ephemeral=False`；正常结束或失败后调用 `thread_archive`，保存 thread ID、事件及归档返回。
- 成功的 `submit_answer` 工具输出作为答案，客户端随后中断并收尾 turn；该预期中断不记失败。`agent_raise` 终止本次尝试；没有成功提交答案的普通终止记失败。
- 通过客户端局部配置覆盖 MCP 注册，不修改全局 Codex 配置。沿用只读 sandbox、禁止提权和已验证的 legacy Landlock 配置。
- 只开放本实验 MCP，关闭 shell、外部检索、其他 MCP 和额外数据访问能力，避免读取参考答案、历史结果或其他题目的输出。
- 保存实际 MCP 参数、返回文本/图片块、QA 请求响应、embedding 请求响应及错误。媒体日志保存来源、PTS、帧重建信息，供手工复核。

### Evaluator

新增 `training_free/val_codex.py`，支持：

```text
--dataset {mmlifelong-month,egolife-open}
--artifact-root
--run-dir
--qa-url
--embedding-url
--concurrency
--baseline-run
--ids
```

`--dataset` 决定时间格式、逐题数据库选择、QA reasoning 参数、答题尝试次数和 judge metadata；`--ids` 仅用于独立调试运行。正式运行使用完整题集。

- 最多同时处理 5 道题，逐题保存答案后安排 judge。
- 已完成答案不重跑；有答案但缺评分时只补 judge。重试创建新 thread，保留之前 attempts。
- Month 普通答题失败批末集中重试一次；维护错误修复后继续同方法运行，不计入普通重试次数。
- EgoLife 按 fair 协议最多 4 次答题尝试。遇到确定的代码、配置或缺失服务问题立即停止并修复，不用重复请求掩盖故障。
- 完成、失败、重试、答案、评分和 usage 持续落盘；最终失败保留原 ID。
- 保留其他 agent 对 `src/` 及现有文件的改动，只修改本次实现必需的代码。

## 4. 服务与执行阶段

### Stage 0：资源检查与服务准备

使用现有镜像 `video-sql-vllm-epd:egolife-baselines`，新建本任务专用服务：

| 服务 | 计划 GPU | 端口 |
|---|---:|---:|
| Qwen3.8-27B-FP8，TP=1 | 4 | 8410 |
| Qwen3-VL-Embedding-8B | 5 | 8414 |

启动前重新检查 GPU、端口及主机可用内存；上述分配不代表资源已预留。资源不足立即报告，不等待、不停止其他任务服务。

- Qwen 使用现有单体服务配置，保留 max-model-len 314573、既有 YaRN overrides、MTP 3、prefix caching、4 GiB 多模态缓存、image 650/video 1、max-num-seqs 50、GPU utilization 0.90、每 rank 128 GiB CPU KV offload，以及 thinking/preserve-thinking。
- 使用单卡 TP=1；不启动独立 encoder 或 EPD proxy。
- embedding served model name 保持 `qwen3-vl-embedding-8b`，服务参数沿用既有配置。本轮仅请求在线查询向量。
- 保存实际命令、容器 ID、镜像 ID、模型和 SDK 版本。
- 服务与评测均通过受控前台会话运行，不使用 `--rm`、后台启动或独立 watcher。
- 使用项目 `.env` 中既有 `SHARED_OPENAI_KEY` 调用 judge；W&B 使用系统已有登录，不查找其他目录的密钥。

### Stage 1：实现、小范围调试与审查

一次实现两种数据模式，完成下面的验证和审查后开始正式 Month 运行。

调试输出和正式结果分开；调试题不自动算作正式完成题。

本次执行范围：仅实现和小测试，不启动正式评测。已新增 `training_free/codex_harness.py` 与 `training_free/val_codex.py`；独立改动审查指出的 embedding 临时错误分类和已提交答案恢复问题已修复。

已完成真实 SQL、文本/视觉向量检索、SQL/semantic/video QA、Month 跨视频与零秒/小数秒、EgoLife 原题截止、五个 context 的 notes 隔离，以及两种 GPT-5 judge metadata、评分映射、ID 对齐和 W&B 事件测试。已人工读取实际请求/响应、reasoning 和证据源帧。一次视频 QA 返回越界证据下标，错误被拒绝；反馈原错误后，同模型请求返回有效证据。最初五 context 测试中 EgoLife 的 ID/query time 配对不一致，其截止测试不计入验证；随后使用原题 1 和 3 的正确配对重新检查。测试记录位于 `runs/mmlifelong-month-codex/debug/` 与 `runs/egolife-open-codex/debug/`，均保持 Git 忽略。

尚未验证：真实 Codex/MCP turn、`submit_answer` 中断、`agent_raise`、thread 归档、答题重试和答案恢复的端到端路径，以及 Terra usage 事件。自动审批拒绝了 MCP 入口启动，理由是将其判定为超出“仅实现和小测试”的授权；已请求允许临时 MCP 小测试，尚待答复。未启动新模型服务、正式评测或正式运行前全仓审查。

### Stage 2：MM-Lifelong Month 正式评测

```bash
.venv_baselines/bin/python -u -m training_free.val_codex \
  --dataset mmlifelong-month \
  --artifact-root /raid/wg25r/video_sql_trace2skill/runs/mmlifelong-month/artifacts \
  --run-dir /raid/wg25r/video_sql_trace2skill/runs/mmlifelong-month-codex/formal \
  --qa-url http://127.0.0.1:8410/v1 \
  --embedding-url http://127.0.0.1:8414/v1 \
  --concurrency 5 \
  --baseline-run /raid/wg25r/video_sql_trace2skill/runs/mmlifelong-month/results/month-sql-gpt5
```

完成全部 623 题的答题、重试、judge 和阶段汇总后，再开始 EgoLife。服务继续保留供下一阶段使用。

### Stage 3：EgoLife Open 正式评测

```bash
.venv_baselines/bin/python -u -m training_free.val_codex \
  --dataset egolife-open \
  --artifact-root /raid/videosql-video-artifacts \
  --run-dir /raid/wg25r/video_sql_trace2skill/runs/egolife-open-codex/formal \
  --qa-url http://127.0.0.1:8410/v1 \
  --embedding-url http://127.0.0.1:8414/v1 \
  --concurrency 5 \
  --baseline-run /raid/wg25r/video_sql_trace2skill/runs/fair-qwen38/formal/fair-qwen38-sql-open
```

使用独立 W&B run、输出目录和全新的 contexts，不带入 Month 的 notes、答案或 thread 历史。

### Stage 4：汇总与服务收尾

两阶段结果核验、保存及提交后，停止并手动删除本任务创建的容器；按记录的容器 ID 核对归属，不清理其他任务资源。

运行根目录：

- Month：`runs/mmlifelong-month-codex/`
- EgoLife：`runs/egolife-open-codex/`
- 服务日志及启动记录：`runs/mmlifelong-month-codex/services/`

确认这些目录已被 Git 忽略后再写入输出。不上传 HF。

## 5. Judge、指标、W&B 和 delta

两阶段使用 fair/MM-Lifelong 官方 `gpt-5` judge：

- 请求使用 OpenAI API，省略 reasoning effort、采样参数和 token 上限。
- 复用 `training_free/evaluate_mm_official.py` 的官方 rubric、Analysis / Final Score 格式及解析规则。
- Month judge 输入问题、参考答案、候选答案，移除 EgoLife metadata。
- EgoLife judge 增加 Jake/A1_JAKE 身份、原 query time、DAY、时分秒及帧号，沿用 fair 对第一人称和相对时间的解释规则。
- 临时 API 或评分格式失败沿用不限次数、逐次增长 timeout 的重试机制，复用已生成答案。确定的代码或配置错误停止执行并修复。
- 保存实际请求、返回模型名、评分分析和 usage。

官方映射为 `4/5→1、3→0.5、0/1/2→0`：

| 数据集 | 固定分母 | score-1 ratio | Average rounded score |
|---|---:|---|---|
| Month | 623 | 映射为 1 的题数 / 623 | 映射分数总和 / 623 |
| EgoLife Open | 500 | 映射为 1 的题数 / 500 | 映射分数总和 / 500 |

最终失败题计 0，不删题或缩小分母。未完成题不提前当作最终失败。

固定对照：

| 数据集 | 对照运行 | score-1 ratio | Average rounded score |
|---|---|---:|---:|
| Month | `runs/mmlifelong-month/results/month-sql-gpt5` | `239/623 = 0.3836276083` | `251/623 = 0.4028892456` |
| EgoLife Open | `runs/fair-qwen38/formal/fair-qwen38-sql-open` | `241/500 = 0.482` | `249.5/500 = 0.499` |

- `delta = 本次 Codex 指标 − 对应 Qwen VideoSQL 指标`，保存绝对差值。
- 按原 ID 对齐；运行中比较共同完成题目，最终使用完整固定分母。
- W&B 使用完成、失败、重试及评分事件触发更新，记录进度、两项指标、overlap 和 delta。
- 去掉 1 Hz 监控、vLLM metrics、GPU 和服务吞吐采集，不再进行 PD counter 对账。
- 分别记录 Terra 主调用、Qwen SQL/semantic/video QA、GPT-5 judge 及重试 token。
- Terra 使用 SDK usage 事件的累计值，避免重复累加；Qwen 和 judge 使用实际响应 usage。缺失 usage 明确记录为未知，不能填 0。
- 记录模型、提示、skills 来源、Codex 原生上下文管理、无 SQL60 预算、并发、服务配置和重试协议，解释 delta 所对应的完整实验设置。
- EgoLife 新调用路径移除旧 QA token 上限，此差异也写入配置和最终报告。

## 6. 验证、审查与最终交付

小范围调试必须覆盖：

- Month 跨视频检索、同视频不同片段、零秒及小数秒、视频身份和视觉查询指令。
- EgoLife `DHHMMSSFF` 与媒体秒数转换、逐题数据库选择、截止后证据不可见、视频读取不越过 query time。
- 两种模式的 SQL 过滤、文本/视觉向量映射、QA 实际请求、返回证据帧。
- 5 并发下 notes、MCP context 和输出目录隔离。
- `submit_answer` 终止、`agent_raise`、普通失败重试、断点恢复及 thread 归档。
- Month QA 实际带 `xhigh`，EgoLife QA 实际省略 effort，两边均无客户端 token 上限。
- 两种 judge 的 metadata、score 映射、逐题 ID 对齐及 delta。
- Terra usage 累计方式、Qwen/judge usage 和 W&B 事件日志。

手工检查真实工具输入、返回、reasoning 事件及源帧，不能只以完成数量或退出码判断成功。不新增测试框架，不运行语法检查。

审查流程：

- 中等代码改动交给新的独立 subagent，直接要求其完整读取当前 `REVIEW.md` 并审查指定改动。
- 沿用已授权的 Claude 替代安排，正式运行前由新的独立 subagent 完成全仓审查。
- 审查必须正常结束并给出明确结论；未完成时不启动正式运行。
- 首次全仓审查覆盖两种数据模式；进入 EgoLife 阶段时，未改变已审查实现则不重复全仓审查。

最终交付：

- 提交本次源码、更新后的本计划和紧凑最终结果报告。
- 报告分别列出两阶段题数、完成数、最终失败、两项正式指标、对应 baseline、delta、token 和 W&B 链接。
- 保存完整逐题答案、judge、attempts、Codex/MCP/QA traces、归档记录、配置与服务日志到各自忽略目录。
- 核对 Month 623 个原始 ID 与 EgoLife 500 个原始 ID 均有最终处理结果。
- 保留既有工作区修改、删除和其他 agent 的成果；不提交逐题输出、原始 traces、数据库、视频或向量。
- 只有两阶段核验与提交、thread 归档及本次服务清理完成后，才报告整个任务完成。
