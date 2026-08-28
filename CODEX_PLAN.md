# 基于 PLANS.md 的完整实施计划

## 总结

- 保留基础设施研究、索引与 harness、pilot、正式训练与评测四个阶段；原 Phase 4 删除，因为 Phase 3 已进行完整评测。
- Pilot 使用 Luna；Pilot 之后的主检索、judge、reflection、skill updater、caption merge 与结构化处理使用 `Inferact/Qwen3.8-Flash-Next-NVFP4`；正式 ASR 按 MAGIC-video 使用 faster-whisper `large-v3-turbo` 与 Silero VAD。
- Pilot 训练每题 5 个 rollout；正式训练每题 10 个并发 rollout；所有验证与测试每题 1 个 rollout。
- 仅主检索 agent 启用 context compaction。完成第三次 compaction 后立即终止该 rollout，记 0 分并进入复盘。
- 固定 compaction prompt，不允许训练过程修改。

## 阶段与实现

### 0. 完成本地 Qwen 基础设施研究

- 继续完成 `notes/INFRA.txt`，不能接受当前失败的 vLLM 日志作为结论。
- 使用真实约 100k-token 多模态对话测试 25–50 个并发请求，检查实际回答和完整调用轨迹。
- 记录最终服务命令、服务框架和版本、GPU 数量、context limit、并发能力、官方 generation config、图像限制、吞吐量、首 token 延迟及整体延迟。
- 移除现有 `temperature=0`；客户端不覆盖 temperature、top-p、top-k，也不设置输入或输出 token 上限。
- 后续所有本地 Qwen 运行只使用最终 `notes/INFRA.txt` 中确认的配置。
- 正式本地模型固定为 `Inferact/Qwen3.8-Flash-Next-NVFP4`，使用四张完全空闲的 GPU 做 tensor parallel 4。除 GPU 部署和 tensor parallel 外，使用 `notes/INFRA.txt` 最上方 Current best choice 的全部参数。
- 正式 Qwen 服务保留 `--reasoning-parser qwen3`，并设置 `--default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true}'`。启动后检查实际 server 参数，并用真实 `/v1/responses` 连续工具调用确认 reasoning item 在下一轮仍被保留；不能只依赖 Responses API 默认行为。
- Qwen3-VL-Embedding-8B embedding server 放在显存已经占用约一半但 compute 空闲的 shared GPU；四张完全空闲的 GPU 只用于正式本地模型。

### 1. 仓库、索引和 harness

- 初始化 Git，身份设为 `codex <codex@localhost>`；使用 `uv` 和仓库内 `/raid` 缓存，提交锁定后的 `uv.lock`。
- 从 [MAGIC-video](https://github.com/lijiazheng0917/MAGIC-video) 只移植当前项目需要的预处理逻辑，并记录上游 commit；不引入其截断、替代数据或静默恢复路径。
- 使用 [MM-Lifelong](https://huggingface.co/datasets/MM-Lifelong/MM-Lifelong) 的官方顺序、时长和区间信息。所有时间统一保存为视频内秒数和全局相对 `INTERVAL`；只有来源提供绝对时间时才保存 `TIMESTAMP`。
- 正式 DuckDB schema 只包含逐项核对 MAGIC-video 完整相关源码和真实生成 artifact 后确认存在的表、字段、关系与约束，并保持其原始名称和语义。保留 MAGIC-video 的 topic chain 和 event chain；chain 的 `video_id` 表示完整逻辑视频时间线，不表示单个视频文件。多个视频文件构成同一时间线时共享一个 `video_id`，单个文件通过 `source_video_id` 和相对 `filename` 定位。问题、答案、question type、temporal certificate 和 clue intervals 全部保存在数据集 JSON 中，不写入 DuckDB；judge 从 JSON 单独接收 reference answer，主检索 agent 无法访问这些字段。embedding 不写入 DuckDB。
- 文本入库前统一执行：拉丁字符转 ASCII 并小写、保留非拉丁字符、去除首尾空白、折叠连续空白。数据库只保留规范化文本。
- 每条可检索文本记录和每个 30 秒视频片段均生成稳定 row ID 和向量。文本及视频使用 [Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B)；每个可检索表分别保存连续 NumPy 向量矩阵和同顺序 row ID 数组，采用 NumPy 精确 cosine scan，不通过 SQL 查询 embedding。先用真实并发查询测试该方案；只有并发性能不足时才改用功能和精度相同的轻量向量数据库。
- 构建期间使用可恢复的磁盘分片，再合并为各表独立 NumPy 矩阵；正式 harness 的 DuckDB 连接只读取 top row ID 对应的元数据，使用 `read_only`、禁用 external access 并锁定配置，且每个并发调用使用完全相同的配置创建独立连接，符合 [DuckDB 并发约束](https://duckdb.org/docs/current/connect/concurrency)。
- 建议目录为 `src/preprocess/`、`src/indexing.py`、`src/harness.py`、`src/training.py`、`src/val.py`、`skills/initial/`、`prompts/COMPACTION.md`、`tests/`、`notes/`；运行产物写入忽略的 `runs/`。

### 2. Pilot

- 将当前 47 个 train 和 623 个 val 问题按共享 clue video 建图，使用已确认的两个连通分量作为固定划分：646 train、24 val，禁止跨集合共享视频。
- Pilot 主 agent、judge、reflection、summarizer 和 updater 使用 `gpt-5.6-luna`，reasoning effort 为 high；使用 Responses API 与 Agents SDK 的流式运行接口。[Agents SDK 运行方式](https://developers.openai.com/api/docs/guides/agents/running-agents)和 [Luna 规格与定价](https://developers.openai.com/api/docs/models/gpt-5.6-luna)作为实现依据。
- Pilot 训练 1 个 epoch，每批 5 个问题，每题 5 个 rollout；顺序以 seed 2026 固定打乱，保存完整 question ID 顺序，最后不足 5 题的批次正常执行。
- 每个 rollout 独立由 judge 评为 0–5 分。judge 只接收该 rollout 的最终 candidate answer 和数据集 JSON 中的 reference answer，只判断两者语义是否正确、完整，并把含义相同的改写视为正确；judge 不接收 question、rollout trace、工具输入输出、证据或检索过程，也不对轨迹质量和努力程度评分。随后 reflection 接收完整 rollout trace、judge 结果和 reference answer，对检索轨迹进行复盘。每题生成一次汇总，每批由 updater 输出该次运行新的完整 `SKILL.md` 和 `META-SKILL.md`；只修改汇总证据要求改动的内容，保留其余原文，不整体重写。
- 初始 skill 提供简短检索指导；初始 meta-skill 提供复盘指导。评测基线使用原始 `SKILL.md`，训练组使用更新后的 skill；测试 prompt 不包含 meta-skill。
- 24 个 val 问题分别执行 trained-skill 和 initial-skill 两组，每题各 1 个 rollout。
- 启用 Luna prompt cache，记录 cache write、cached input、普通 input 和 output token。Pilot API 累计费用超过 100 美元时，仅在当前顶层 rollout 完成后保存恢复状态并暂停。
- 完成 Pilot 后报告每题费用及正式规模外推。

### 3. 正式本地模型训练与完整评测
参考文件:
notes/INFRA.txt 主llm搭建
notes/VIDEO_EMBEDDING_PIPELINE.md embedding搭建
MAGIC-video 生成index流程
- 使用 datasets/Hugging Face 下载机制获取正式数据；重新生成全部正式数据，不复用 Pilot 产物。
- 正式索引预处理必须以 MAGIC-video commit `0f2273f7f07360cc58588399553b5a6ea758b084` 的预处理脚本、阶段顺序和产物语义为基线。保留已经确认的离散 clip cache、TorchCodec 解码、下述 ASR 配置、sample-local skip 分类、`Inferact/Qwen3.8-Flash-Next-NVFP4`、Qwen3-VL-Embedding-8B server、DuckDB 元数据和每表独立 NumPy embedding 矩阵改动；除此之外不得用自行简化的预处理流程替代 MAGIC-video。全部正式数据重新生成，不复用 Pilot 产物。
- 正式索引的 Qwen 请求使用顶层 `reasoning_effort=low`，并显式传入 `enable_thinking=true` 与 `preserve_thinking=true`；正式 agent loop 使用 `reasoning_effort=medium`，不得使用 `xhigh`。harness 使用 Responses API，并保持 `RunConfig(reasoning_item_id_policy="preserve")`；正式运行前同时核对 harness trace 与 server trace 中的 reasoning item 保留情况。
- 索引中的 video 按实际视频时间线定义，不按单个视频文件定义；多个视频文件构成同一时间线时作为一个 video，彼此独立的时间线分别作为不同 video。
- 正式索引使用 `month`、`day`、`week` 三条视频时间线；clip 保留现有稳定 row ID，并由 row ID 中的来源 video ID 定位离散 clip 文件。ASR、caption、topic 和 event 的秒数统一映射到所属时间线。
- month 使用 MAGIC-video 的 MM-Lifelong 提示，week 使用 MAGIC-video 的 EgoLife 提示，day 使用保持相同阶段结构的游戏提示。
- month 与 week 的 caption、topic 和 event 提示使用 MAGIC-video commit `0f2273f7f07360cc58588399553b5a6ea758b084` 中对应原文；week 的 activity `location` 与 event `chain_type` 按原输出写入 NULL。
- 多级 caption 聚合保持 `month`、`day`、`week` 三个逻辑 video；week 聚合不跨 DAY，同一 DAY 内按实际视频时间线连续处理。week topic 输入保留秒级本地时刻，topic/event 的 DAY 与本地时刻映射回 week 逻辑时间线秒数，event 使用完整七日 activities 并执行 EgoLife 原第二轮漏项提取。
- week 的本地时刻使用每个实际视频文件名中的 DAY 和时钟分段映射到逻辑时间线；模型时刻落入没有实际视频的空档时，该响应校验失败并在相同设置内计入最多 10 次重试，不映射到相邻边界。
- OpenIE 与 semantic 使用 MAGIC-video 原完整提示和 one-shot；week topic 不执行 MM-Lifelong entity filter，并排除 EgoLife 参与者姓名；topic/activity/event 的 `HH:MM:SS` 校验与数据库入库一致，小时和分钟解析为整数、秒解析为浮点数。
- 正式索引的每个本地模型请求最多重试 10 次；全部尝试失败后跳过整个样本并记录，完成当前批次后统一补跑缺失样本。正式并发量在运行前使用真实视频请求逐级测试后确定。
- 每次本地模型重试保存该次实际发送的消息及此前校错消息，包括 VLM 请求中的实际 base64 video data URL；确定性 4xx 请求、凭据、模型或服务配置错误直接终止，不按样本跳过。补跑后仍缺失任何样本时终止当前索引运行，不装配不完整 artifact。
- 正式索引运行记录到 W&B project `video-sql-v2`，run name 为 `magic-index-mmlifelong-full`，保存配置和各阶段完成、总数、补跑轮次及缺失数。
- 正式索引不提供截断每条时间线 clip 数量的运行开关。
- 正式 ASR 重新转录使用上述 MAGIC-video commit 的 faster-whisper `large-v3-turbo` 与 Silero VAD 方法处理官方 MM-Lifelong revision `248aa82039a574e63a2e524746a7cd8f32330443`；脚本在导入 CUDA 前固定暴露物理 GPU 1、4、5、6，并将四个 worker 映射到逻辑设备 0、1、2、3；每卡 1,024 并发的首次运行在完成并验证 4,052 个样本后发生 CUDA OOM，恢复运行降为每卡 512；保留模型返回的空文本段，连续空行按一个 SRT block 分隔符解析，并让两行 SRT block 与 JSON 空字符串严格对应；每个新 JSON/SRT 写盘后立即重读并核对段数、时间戳和文本，再记录该视频完成。配置、产物路径和 transcript 使用方法见 `notes/ASR.txt`。caption merge、OpenIE、semantic、topic、event 等后续步骤继续按照 MAGIC-video 的对应预处理脚本执行，模型使用 `Inferact/Qwen3.8-Flash-Next-NVFP4` 和 `notes/INFRA.txt` 指定的配置。
- 从与 Pilot 相同的初始 `SKILL.md` 和 `META-SKILL.md` 重新开始，不继承 Pilot 训练结果。
- 使用 month train 的 266 个问题训练 1 个 epoch，每批 5 题，每题 10 个并发 rollout；seed 2026 固定顺序，最后一批正常处理。
- 每个 rollout 由 `Inferact/Qwen3.8-Flash-Next-NVFP4` judge 独立评分；judge 与 Pilot 一样只比较最终 candidate answer 和 reference answer，不读取 trace。reflection 单独读取完整 trace 和 judge 结果进行复盘；summarizer 和 skill updater 也使用该模型。
- 对 month val 623、day test 200、week test 200 共 1023 题进行完整评测；trained-skill 和 initial-skill 各跑一次，每题每组 1 个 rollout。
- 报告 overall、day/week/month、question_type 以及每题 trained-vs-baseline 配对差值，并保留全部逐题原始结果。不存在额外 Phase 4。

## Agent 接口与 compaction

### Harness 工具

- `run_sql(sql)`：只读执行，30 秒硬超时；最多向模型返回 100 行并说明是否存在更多结果；每个单元格最多返回前 500 字符及原长度。SQL 语法或数据错误返回给模型，数据库系统错误直接终止。
- `semantic_search(query, n, table)`：只允许查询已生成 embedding 的表，返回 table、row_id、video_id、start/end、规范化文本预览及 cosine score。
- `clip_qa(video_id, start_s, end_s, question)`：建议片段不超过 10 分钟，超过 20 分钟直接拒绝。内部 agent 可调用 `sample_video`、`add_note`、`read_notes`。
- `sample_video`：接受时间、fps 和 0–1000 整数裁剪坐标；返回覆盖请求区间的 caption 和带时间标记的 4×3 contact sheet，每页 12 帧，所有帧完整分页。contact sheet 和证据截图只向模型暴露文件名，不暴露宿主机绝对路径。
- `clip_qa` 返回 `answered|unknown`、答案和至少一个证据时间点；harness 将证据时间点转换为独立截图交给主 agent。
- 主 agent 提供 `add_note`、`read_notes` 和 `agent_raise(reason)`；正式运行中同一 rollout 最多按原设置重试 3 次，缺少必要数据或确定性代码错误直接终止。
- 主 agent 最大 50 turns；达到上限按整个 rollout 重试，不截断上下文或返回半成品。

### Compaction 状态机

- 仅主检索 agent 使用 compaction；`clip_qa`、judge、reflection、summarizer 和 updater 不使用。
- 每次主 agent 响应完成后读取服务端 usage，计算 `remaining = notes/INFRA.txt.context_limit - input_tokens - output_tokens`。
- 当 `remaining <= 50,000` 时，通过流式 runner 执行 `cancel(mode="after_turn")` 并排空事件，确保当前工具轮完整结束。
- 在当前主检索 agent 的同一对话中追加固定 compaction 指令，要求它停止工具调用并总结当前对话。该 agent 的下一次输出作为新对话中的 trusted handoff，旧对话不再传入；不创建单独的 compactor agent。
- 第一次和第二次 compaction 后继续 rollout。第三次 compaction 输出生成后立即终止，状态设为 `compaction_limit`，分数设为 0，不调用 judge、不重试，直接进入重点分析绕圈原因的 reflection。
- Reflection 永不 compaction；当其 remaining 小于等于 50k 时，禁止继续调用工具并强制输出最终 reflection 和 meta-reflection。

### 固定项目专用 compaction prompt

```text
Your context window is nearly exhausted. Stop tool use and summarize your current in-progress video-memory SQL investigation. Produce only a self-contained handoff for yourself in a fresh conversation. Do not answer the original question, call tools, propose a new retrieval strategy, or add conclusions that are not already supported by the conversation.

Preserve all information needed to continue the exact rollout:
1. The original question, required answer format, and success condition.
2. The current stage of the investigation and the next concrete action.
3. Every verified fact, including exact table names, row IDs, video IDs, timestamps, entities, quantities, SQL predicates, joins, ordering clauses, and aggregation logic needed to reproduce it.
4. Decisive SQL and semantic-search results, compressed into the smallest set of rows or aggregates that preserves their meaning.
5. Clip observations, screenshot evidence, contact-sheet findings, caption evidence, and exact evidence timestamps.
6. Agent notes and any constraints imposed by the read-only database, tool interfaces, evidence requirements, or the user.
7. Every attempted path, its tool inputs at a reproducible level, its result, and whether it failed, was rejected, or remains unresolved.
8. Errors, contradictory evidence, hypotheses already ruled out, and the evidence that ruled them out.
9. Open questions, unresolved conflicts, and missing evidence.
10. The number of compactions already completed for this rollout.

Use exactly these sections:
Original task
Verified evidence
Working interpretations
Failed or rejected attempts
Unresolved items
Current state
Next action

Compress repetition and long tool output, but do not drop identifiers, timestamps, SQL details, contradictory evidence, failed paths, or current progress. Distinguish verified evidence from interpretation. Never invent missing information or convert an interpretation into a fact. The output will replace the entire prior conversation, so it must be sufficient to continue without access to that conversation.
```

- `prompts/COMPACTION.md` 为只读训练资产。updater 只输出该次运行的完整 `SKILL.md` 和 `META-SKILL.md`，因此训练不能改动 compaction prompt。
- 第三次 compaction 后的固定复盘指令为：`Stop tool use now. Produce the final reflection and meta-reflection from the existing trace. Focus on repeated paths, why the rollout failed to make progress, which evidence was repeatedly revisited, and what skill change would prevent the loop. Do not continue solving the question.`

## 持久化、测试与阶段门

- 每个 rollout 完成后立即保存独立 JSON，包括原始 responses、reasoning items、tool inputs/outputs、usage、截图路径、评分和状态；reflection、question summary、updater 完整输出和每批 `SKILL.md`/`META-SKILL.md` snapshot 也分别落盘。
- 恢复运行时依据实际已有产物继续，不增加完成标记、哈希清单或替代结果。
- 预处理失败按照 `notes/SKIP_OR_CRASH.md` 分类：只影响单个独立样本的已确认错误跳过整个样本并记录稳定 row ID、来源和原始错误，不生成替代结果；可能影响多个样本或运行配置的错误立即终止。
- W&B project 固定为 `video-sql-v2`，记录每个 rollout 分数、每题均值、batch 均值、模型 usage、compaction 次数及最终状态。
- 测试覆盖规范化、时间映射、正式 schema 的每项定义均有 MAGIC-video 源码与真实 artifact 映射、topic/event chain 使用逻辑时间线 `video_id`、多个文件共享同一 `video_id` 时仍由 `source_video_id` 和相对 `filename` 定位正确文件、数据库不含问题、答案和 clue 信息、视频只暴露文件名、DuckDB external access 禁用与锁定配置、只读并发、SQL 超时和返回限制、semantic ranking、notes、`agent_raise`、clip 时长与裁剪、contact-sheet 分页、caption 覆盖、证据截图、judge 只接收两份答案及 numeric-ID 反例、updater 完整输出保留未要求改动的原文、646/24 划分、第三次 compaction 终止、reflection 强制结束、prompt 不可修改、固定顺序、尾批和恢复。
- 集成测试必须读取真实 Pilot 数据库行、查看实际 contact sheet 和截图，并人工检查完整 agent trace、每次工具的真实输入输出和 usage。
- 阶段 1 完成并提交后暂停，交由用户审查代码；获得确认后才进入 Pilot。
- 小修改无需 review。中等修改先提交，再由 Codex subagent 按 `REVIEW.md` 审查相对上一提交的 diff，修复所有明确问题并复查。每次 Pilot 或正式大型运行前使用 Claude CLI 对完整仓库进行跨模型审查，不限于 diff，修复所有明确问题并复查。
- 所有长期任务直接在 Codex harness 的统一前台 exec session 中运行，不使用 tmux、后台命令或独立 watcher；完整日志写文件，stdout 和 `AGENT_STDOUT.txt` 只记录有界里程碑。启动后使用空 `write_stdin` 先等待 2 分钟，稳定后等待 4 分钟，再使用 Codex 的 30 分钟前台 trigger；任务提前结束时由同一 session 立即唤醒。

## 当前进度（2026-08-28）

1. Phase 1 代码、Pilot DuckDB、离散视频 clip cache、TorchCodec harness、embedding 调度和每表独立 NumPy semantic-search 索引已经实现，并已用于有效 Pilot 运行。
2. Pilot 视频 clip cache 已经完成：DuckDB 和磁盘各 9,167 个 clip，所有视频的最终边界与视频流时长一致。文本 embedding 已经生成 813 个 shard，共 416,029 行。
3. Pilot 视频 embedding shard 生成已经结束：9,164 个成功，`14:video_clips:557`、`17:video_clips:557`、`21:video_clips:557` 因已确认的 TorchCodec 视频末尾错误按整 clip 跳过。
4. 原 DuckDB 逐行 embedding 导入已停止，不作为最终存储；Pilot NumPy 索引包含 11 个 embedding 表的 row-ID 矩阵和向量矩阵。
5. 正式本地模型已经改为 `Inferact/Qwen3.8-Flash-Next-NVFP4`。本次正式索引使用物理 GPU 1、4、5、6 做 tensor parallel 4；embedding server 放在显存已经占用约一半但 compute 空闲的 shared GPU。除 GPU 部署和 tensor parallel 外，使用 `notes/INFRA.txt` 最上方 Current best choice 的配置。
6. Pilot 前两次运行分别因 agent 可查询 gold answer 和 judge 将 incidental numeric ID 当作答案而失效，已归档为 `runs/pilot_luna_contaminated_answer_leak_20260827` 和 `runs/pilot_luna_invalid_numeric_judge_20260827`，不得恢复或计入结果。
7. Pilot 已完成 gold 数据与 agent 数据库的物理隔离、DuckDB external access 禁用和连接配置统一；judge 现在只接收 candidate answer 与 reference answer，只判断最终答案的语义等价性，不读取问题、trace 或证据。
8. Pilot 已完成工具输入拒绝与系统错误传播的分类、主 agent 与 nested clip agent 的失败 trace 保存、并发图片文件名隔离、流式 API 错误分类，以及由当前主 agent 生成 handoff 后重开对话的 compaction 实现；相关经验和断点见 `notes/PILOT_RUN_LESSONS.md`。
9. 当前有效 Pilot 位于 `runs/pilot_luna`：已保存 99 个完成的训练 rollout、19 个 question summary 和 3 个 batch 的 skill/meta-skill/updater snapshot；W&B run 为 `https://wandb.ai/3dsmile/video-sql-v2/runs/c1sjjmam`。Pilot 尚未完成。
10. 当前已启动但未完成的 rollout 是 `train/question-527/rollout-1`：三个允许的尝试已经分别因 streamed `openai.APIError`、`MaxTurnsExceeded(50)` 和 streamed `openai.APIError` 结束。继续前必须明确决定保留该失败还是重新授权 retry budget，不能删除或改名 attempt 产物来绕过上限；其余 Pilot 训练问题和完整 24 题双组验证仍未运行完成。
11. 正式 MM-Lifelong video-only DuckDB 和 clip cache 已完成：31 个视频、21,781 个 clip；`videos` 使用相对 `filename` schema。
12. 正式视频 embedding 和 NumPy 索引已完成：21,775 个 4,096 维单位向量；`14:video_clips:557`、`17:video_clips:557`、`21:video_clips:557`、`0:video_clips:589`、`0:video_clips:1138`、`0:video_clips:2209` 因 TorchCodec 视频末尾错误跳过，用户确认不再处理。
13. 正式 MM-Lifelong 全量 ASR 已在 GPU 1、4、5、6 完成；每卡 1,024 并发在完成并验证 4,052 个样本后发生 CUDA OOM，恢复运行降为每卡 512，最终完成并验证 6,290 对 JSON/SRT、171,281 个 segments；W&B run 为 `https://wandb.ai/3dsmile/video-sql-v2/runs/wszu766s`。
14. 正式数据目前只完成 video-only DuckDB、离散 clip cache、视频 embedding/NumPy 索引和 ASR；MAGIC-video 的 caption merge、OpenIE、semantic、topic、event 等后续预处理及相应文本 embedding 尚未完成，完成后才能开始正式训练。
15. 正式索引 Qwen 服务已在物理 GPU 1、4、5、6 以 tensor parallel 4 启动验证：实际 server 参数包含 `reasoning_parser=qwen3`、`enable_thinking=true` 与 `preserve_thinking=true`，render 结果确认 `reasoning_effort=low` 已进入模型 prompt；真实单视频 caption 请求仍在正常生成时被当前 agent 错误取消并停止服务，长输出本身不是失败，也不要求新增固定输出预算。正式索引尚未启动，多轮 `preserve_thinking` trace 尚未验证通过；事故报告见 `notes/INCIDENT_MISCLASSIFIED_LONG_OUTPUT_AS_ERROR.md`。
