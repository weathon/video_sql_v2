# Luna + Trace2Skill baseline：暂停与恢复交接

## 当前状态

用户充值后已使用同一 `PERSONAL_OPENAI_KEY` 恢复过一次。用户现已明确：耗尽重试次数时跳过当前样本。代码已按此修改，继续沿用原有次数，余额耗尽仍暂停整个 run。

- 工作目录：`/raid/wg25r/video_sql_2`；分支：`codex/trace2skill-baseline`。原仓库 `/raid/wg25r/video_sql_v2` 的数据和索引供读取，新输出写入本 worktree。
- 入口：[src/trace2skill_baseline.py](../src/trace2skill_baseline.py)。这是问答 harness + 上游 Trace2Skill baseline，技能学习不使用本项目原来的 reflection、branching 或 updater 方法。
- 当前输出目录：[runs/trace2skill-baseline-openai-gpt-5.6-luna-high](../runs/trace2skill-baseline-openai-gpt-5.6-luna-high/)。停止后核对到 **121 条已完成、已评分的 rollout，覆盖 37 道训练题**；本次恢复新增 41 条，这 37 题并非都完成了 5 条。
- 首轮日志 `formal-001.log` 保留；最新日志 `formal-002.log` 的退出码为 1。直接退出原因是 `initial/question-17/rollout-03` 的 3 次 attempt 均为 `APIConnectionError`，最后一次异常链为 TLS `SSLEOFError`。没有 `insufficient_quota` 或 quota-pause 文件；不能把本次停止归因为余额耗尽。
- Weave 记录部分 call start 上传超时并丢弃，远端调用树可能不完整；本地 trace 和媒体保留。
- `initial-summary.json`、`combined-records.json`、`skills/evolved/` 和 `comparison.json` 均未生成。尚未完成初始采集，也未进入 Trace2Skill 分析、evolution 或复评。
- 本次 `trace2skill-embedding` 已停止并手动删除。旧的 `trace2skill-embedding-gpu4`、PD、encoder、proxy 等历史容器保留为停止状态。

## 旧 Luna 轨迹核对

已在原仓库和本 worktree 的运行产物中寻找；未找到完整的 **266 道 month 训练题 × 5 条 = 1,330 条**旧 Luna 数据。没有导入旧数据，也没有改成只做旧轨迹分析的入口。

| 数据 | 实际范围 | 与本次训练的关系 |
| --- | --- | --- |
| [training_free_luna_seed0_7510e535](/raid/wg25r/video_sql_v2/runs/training_free_luna_seed0_7510e535/) | day/week/month 各 100 题，每题 5 条，共 1,500 条；逐条答案和列出的 trace 文件存在 | seed-0 评估集；其中 month 的 100 题与 266 题训练集的题目 ID、问题文本均无交集 |
| [pilot_luna_fresh2_personal](/raid/wg25r/video_sql_v2/runs/pilot_luna_fresh2_personal/) | 16 道训练题，每题 5 条，共 80 条；逐条 reference 与训练集一致，列出的 trace 文件存在 | 不完整，且含原项目的 reflection/技能更新流程 |
| 当前 baseline run | 37 题中已完成 121 条 | 保留在当前目录，恢复时复用；与上一行的旧 pilot 数据独立 |

旧数据的文件存在性核对不等于所有工具、图像和分析路径都已人工验收。旧 trace 格式也不同：部分没有当前的 `tool_calls` 字段或整数 tool ID，不能直接交给当前分析入口。

## 保留的模型与 harness 配置

| 配置 | 当前值 |
| --- | --- |
| 生成模型 | 官方 OpenAI `gpt-5.6-luna`，`https://api.openai.com/v1`，Responses API |
| Reasoning | `effort=high`、`summary=auto`，用于答题、子 agent、compaction、judge、分析与 evolution |
| 历史保留 | `store=false`，请求 `reasoning.encrypted_content`，重放完整 Responses items；本地保留 summaries 和 encrypted reasoning |
| 凭据 | 本 worktree 的 `.env` 中的 `PERSONAL_OPENAI_KEY`；文件已保留并被 Git 忽略，不在文档或日志中复制 key |
| 并发 | 实际请求共享 semaphore 256，覆盖嵌套 agent、judge、embedding 和 replacement token counting；rollout worker 数也为 256 |
| HTTP | connect 60 秒；read/write/pool 100000 秒；最大连接数 1000，keep-alive 256；实际使用的 `httpx2` 和 `httpx` 均显式配置 |
| 上下文 | harness 预算 314573；剩余 ≤178000 时触发主 agent compaction；每次 run 最多 100 轮 |
| 图像 | 最大 854×480，每个 clip 的采样和裁剪累计额度 650；没有 30 张限制 |
| Embedding | `qwen3-vl-embedding-8b`，`http://127.0.0.1:8014/v1`，与已有索引匹配 |
| 生成参数 | 不设置 temperature、top_p 或输入/输出 token 上限 |

314573 和 650 是当前保留的 harness 设置，不是对 Luna 官方容量的声明。上下文计量使用实际 API usage；不沿用 Qwen processor 的图片 token 估算。Compaction 仍为原会话中的文本 handoff，不使用 Responses compaction items。

生成请求不再经过 DeepInfra、OpenRouter 或本地 vLLM PD/encoder/proxy。历史 FP8、TP、max-num-seqs 和 KV offload 配置不控制当前云端 Luna 请求。

### `replace_tool`

所有有返回的工具调用带整数 `tool_id`，在同一 rollout attempt 内从 1 编号，与嵌套 agent 共用。`replace_tool(tool_id, replacement)` 可以替换任意已完成工具的返回内容，包括图像和之前的 replacement 返回；只改变后续模型输入，不撤销工具副作用。

工具说明要求保留有用证据，建议优先替换最新工具结果以保留 KV cache 前缀，用途是删除读错文件、执行错 SQL 等产生的无用输出，以及把冗余长结果压缩成有用摘要。原始结果和 replacement 调用继续保存在 canonical trace 中，供 Trace2Skill 查看。

每次成功替换另存 `attempt-N/replace-<call_id>.json`，包含原生工具调用及 before/after 输出；token counting 后回写两侧 token 数。重复替换比较上一次 replacement 与本次 replacement，不重复按最初原文计算。替换图像不会返还累计采样额度。

## 训练协议与恢复行为

题目为 `/raid/huggingface/hub/datasets--MM-Lifelong--MM-Lifelong/snapshots/248aa82039a574e63a2e524746a7cd8f32330443/month/train.json` 的全部 266 题。数据库、索引和媒体映射来自 `/raid/wg25r/video_sql_v2/data/runtime/month/`。

入口保留的流程为：每题 5 条独立初始 rollout → 0–5 judge 评分 → 4–5 成功、0–3 失败的 Combined 分析 → 一次完整 Trace2Skill evolution → 冻结技能后，在相同训练集每题再做 5 条复评。没有 held-out 测试。

`skills/initial/` 的 ROOT-SKILL 和六份 subskill 已逐字节复制到 run 内快照，停止后再次核对一致。Catalog、按需加载和 `read_skill(name)` 保留。桥接层映射上游目录格式；所有 patch 基于同一初始快照，保留并行提议、分层合并、translation、应用和验证。上游固定版本及详细协议见 [baseline/README.md](../baseline/README.md)。JSON 分析使用 SDK structured-parse，保留既定纠错机会；单条轨迹分析耗尽可重试错误的机会后记录并跳过该分析样本，不把空报告送入 evolution。

恢复仍使用当前 run 目录和初始技能快照：

- 已有 `result.json` 的 rollout 直接复用，包括已有评分和 replacement 指标。
- 已有 `attempt-N/answer.json` 但没有最终结果时复用答案，继续 judge 和后续统计。
- 只有中断 trace、没有答案的 attempt 会移到 `interrupted-*` 目录，再执行该 attempt；不会从最后一个工具调用续接会话。
- 已有 `error.json` 的 attempt 按入口逻辑跳过，继续其余已配置的机会。SDK 重试和最多 3 次 rollout/judge attempt 设置保留。
- QA 或 judge 重试耗尽后保存 `skipped.json`，保留 trace/media，跳过整条 rollout 并继续其他样本；恢复时复用跳过记录。跳过项不生成评分结果，也不进入 rollout 均值或 Trace2Skill 分析；阶段汇总单列 `skipped_rollouts`。单条分析重试耗尽也保存跳过记录，汇总单列 `analysis_skipped_rollouts`。
- `write_error` 对 `insufficient_quota` / `no credits remaining` 单独写 `*-quota-pause-*.json` 并抛出停止，不生成会消耗 QA/judge attempt 的错误标记。额度耗尽后的恢复不会因此减少重试机会；本次网络错误不属于此分支。
- W&B 使用相同 run ID 和 `resume="allow"`；恢复时重新汇总已完成 rollout 的均值，不把中断期间的停机时间加到 QA 时长中。

恢复时，第 17 题 rollout 3 的既有 3 次失败会触发跳过，不增加重试次数或重置错误记录。在两个前台 exec 会话中分别启动 embedding 和 baseline。先核对实际 GPU 资源；embedding 脚本当前指定 GPU 1，资源不足立即报告，不等待或停止其他任务。下面是下一次恢复的命令模板，日志使用新文件名，不能覆盖旧日志。

```bash
cd /raid/wg25r/video_sql_2
.venv/bin/python -u infra/trace2skill_service.py embedding \
  > runs/trace2skill-baseline-openai-gpt-5.6-luna-high/embedding-resume-002.log 2>&1
```

从该前台会话的实际启动日志确认服务 ready 后，在另一个前台 exec 执行：

```bash
cd /raid/wg25r/video_sql_2
.venv/bin/python -u -m src.trace2skill_baseline \
  > runs/trace2skill-baseline-openai-gpt-5.6-luna-high/formal-003.log 2>&1
```

开始正式运行前遵循仓库审查流程：Claude CLI 全仓库审查；若明确因 usage limit 无法完成，改用新的子 agent 阅读 `REVIEW.md` 并审查全量。此前审查产物 `claude-full-review.json` 及本次只读全仓库审查 `claude-resume-002-review.json` 已保留，本次结论为 no blocking issues。不要运行 syntax、smoke 或 preflight tests；从正式运行的真实 trace 检查行为。稳定后按一小时频率检查前台任务，不启动后台 watcher；运行结束核验真实产物后手动清理本次容器。

## W&B、Weave 和本地证据

[W&B run](https://wandb.ai/3dsmile/video-sql-v2/runs/trace2skill-baseline-openai-gpt-5.6-luna-high) 的 ID 与 run 目录名相同。下表中的 `<phase>` 为 `initial` 或 `evolved`；分析阶段也记录请求吞吐。

| Metric | 口径 |
| --- | --- |
| `<phase>/trace/mean_generated_tokens`、`mean_model_calls`、`mean_tool_calls`、`mean_input_tokens`、`mean_peak_context_tokens` | 按完成 rollout 求均值；包括重试和嵌套 agent，排除 judge。`mean_input_tokens` 先对每条 rollout 的模型请求求输入 token 均值，再跨 rollout 平均 |
| `<phase>/throughput/input_tokens_per_s`、`output_tokens_per_s`、`responses_per_s` | 最近 60 秒按完成的模型请求计量，包含 judge；输入吞吐包含 cache 命中的 token |
| `<phase>/average_rollout_time_seconds` | 完成 rollout 的平均 QA 时长，含队列、工具、嵌套调用、compaction 和已计时的重试；排除 judge、replacement counting 和停机时间 |
| `<phase>/mean_total_context_saved_tokens` | 每条完成 rollout 内加总 replacement 的 before−after token 数，再跨 rollout 平均；包括图像，保留负值，不按后续重用轮数乘算 |
| `<phase>/replace_tool_calls` | 完成 rollout 中已返回的 replacement 调用累计次数，含返回错误和重试 attempt，排除 judge |

Replacement 计量调用 OpenAI `responses.input_tokens.count`，在 QA 计时之后执行；计量文件供恢复复用。本地 `initial-trace-metrics.json` 和逐 rollout 的 `replacement_metrics`、`trace_length`、`rollout_time_seconds` 保留统计依据。

Weave 使用 Agents SDK tracing processor。每条 rollout 作为完整调用树，带 run/question/rollout/attempt/skill_version，包含主 agent、工具、SQL/视觉子 agent、compaction、API calls 和 judge；分析与 evolution 保留各逻辑操作的 spans 和本地产物。

正式运行期间已查看实际语义检索、SQL、视觉工具输入输出和对应视频帧，确认样本中有可阅读 reasoning summary、encrypted reasoning，以及 Weave 中 QA 与 judge 的父子关系。尚未完成全量验收；非零 replacement 节省路径、Trace2Skill 分析和 evolution 仍需从后续正式产物核验，不能把已配置指标视为已运行证据。
