# Training-free Video Understanding Harness：完整实施与交接计划

## 1. 目标、已确认决策与当前状态

将现有项目改为只使用初始静态 skills 的 training-free 视频问答 harness，删除训练链路，修复历史 traces 中影响推理的已确认问题，完成独立 diff 审查。

用户已经确认：

- 主检索 agent、视觉 agent、`sql_qa`、judge 均使用 **OpenAI 官方 API**。
- 密钥读取项目 `.env` 的 **`PERSONAL_OPENAI_KEY`**；模型为 **`gpt-5.6-luna`**，reasoning 为 **`medium`**。
- 使用 **Responses API**，保留多轮 reasoning。之前的 Cerebras / Chat Completions 方案撤销。
- 每批 **30 questions × 5 rollouts = 150 个并发 rollout**，优先整体吞吐。
- 保留答题、judge、逐 rollout 落盘、恢复机制、W&B 和 Weave。
- 用户补充：training-free 也必须保存完整 trace。保存初始指令、模型设置、工具 schemas、问题、reasoning、工具输入输出、judge、compaction、失败记录和逐事件日志；图片独立保存，记录原始帧、裁剪和证据之间的映射。
- 保留 `skills/initial/` 的静态检索技能，删除 META-SKILL 和自动技能更新；不使用训练后的 skills。
- Judge 接收**原始问题、reference answer、candidate answer**，维持开放式回答和原有 0–5 分语义等价评分。
- `sql_qa` 返回的 SQL 结果完整交给子模型，**不截断**；API 溢出明确返回错误。
- 视觉 agent **不做 compaction**。通过累计图片上限控制增长，上限由实施方计算。
- 每张视频帧使用约 **480p**，指单帧而非整张 contact sheet；允许从原始分辨率帧裁剪细节。
- 保留现有 **Qwen3-VL-Embedding-8B API 和 embedding index**，不重建索引。
- 本次交付为代码修改、提交和最终 diff 审查；尚未获得启动新正式评测的指令。

**实施开始前的工作区状态（历史快照）：**

- 工作目录：`/raid/wg25r/video_sql_v2`。
- 已从 `fresh2` 创建分支：`codex/training-free-harness`。
- 分支起点：`b4799309eb060f47ccd8db4e77cb22d94c2c85ed`。
- 迄今只创建了分支，**没有实施代码修改、没有新提交、没有启动运行**。
- 本计划已保存为 `TRAINING_FREE_PLAN.md` 并在 `c1015eef` 提交。其余本节条目保留实施开始前的快照；当前实现状态见 `notes/DOCS_AND_HANDOFF.md`。
- 工作区存在用户已有修改、删除和未跟踪文件，必须保留，不能顺手提交或恢复。其中包括已修改的 `AGENT_STDOUT.txt`、`REVIEW.md`，以及未跟踪的 `OTHERS.md`、`notes/CLAUDE_PROBLEMS.md`、`notes/INCIDENT_redundant_post_patch_read.md`、`notes/index.md`。
- 用户已有的源码、测试和基础设施文件删除同样不能恢复。

历史说明在 [DOCS_AND_HANDOFF.md](/raid/wg25r/video_sql_v2/notes/DOCS_AND_HANDOFF.md)。其中的训练、本地 Qwen、旧并发和旧 judge 设置是历史状态，以本计划和用户的新指令为准。

## 2. 实施方案

### 2.1 删除训练，保留评估所需代码

在 [harness.py](/raid/wg25r/video_sql_v2/src/harness.py) 中删除 reflection 方法、反思类型、专用指令和仅为反思保存的会话字段；移除不再使用的 imports。保留答题过程中的同 agent compaction。

将 `src/training.py` 中评估仍依赖的代码迁入 [val.py](/raid/wg25r/video_sql_v2/src/val.py)：

- `Judgment` 和 `judge_answer`。
- `exception_chain`、`retryable`、`write_error`。
- `milestone`。

随后删除 `src/training.py`，删除 `skills/initial/META-SKILL.md`。保留 ROOT-SKILL 和六个初始检索技能。删除 catalog 中仅为 META-SKILL 服务的特殊处理。

将 `docker/training/Dockerfile` 移至 `docker/harness/Dockerfile`，继续安装现有依赖和 ffmpeg，不增加本地主模型服务依赖。

历史 run、训练结果、技能快照、事故记录和 trace 查看工具保留用于追溯。现有本地 EPD 基础设施不进入运行链路，本任务不重新部署或重启它们。

### 2.2 OpenAI API、reasoning 和上下文

继续使用已安装的 OpenAI Agents SDK 与 `OpenAIResponsesModel`，不自行实现 agent loop，不增加 Chat Completions adapter。

明确设置：

```text
base_url = https://api.openai.com/v1
model = gpt-5.6-luna
api_key = 项目 .env 中的 PERSONAL_OPENAI_KEY
reasoning.effort = medium
reasoning.context = all_turns
store = false
response_include = ["reasoning.encrypted_content"]
RunConfig.reasoning_item_id_policy = preserve
```

沿用 `result.to_input_list()` 保留每个输出项，包括 encrypted reasoning items。正常工具轮次不删除、重排 reasoning。主 agent 执行原有 compaction 时，按既定流程用 handoff 开始新的上下文。

不设置 temperature、top_p、max_tokens、max_output_tokens 或其他采样参数；不降模型或 reasoning。保留已有 SDK `max_retries=3` 和阶段 `attempts=3`。

Luna 官方上下文为 **1,050,000 tokens**，最大输出为 **128,000 tokens**。将主 agent 的容量常量更新为 1,050,000，compaction 触发余量设为 **178,000**：保留官方最大输出空间，再留原有 50,000 的工具与 handoff 余量。这是 compaction 触发条件，不向 API 设置输入或输出 token 限制。

主 agent 仍最多执行三次 compaction；第三次后返回 `compaction_limit`、得分 0、不调用 judge。优先处理成功的答案提交，避免它被同轮 compaction 标记覆盖。

官方依据：

- [Luna 模型容量](https://developers.openai.com/api/docs/models/gpt-5.6-luna)
- [Reasoning 历史与 encrypted content](https://developers.openai.com/api/docs/guides/reasoning)

本地已确认：项目 `.venv` 使用 Python 3.13、Agents SDK 0.22.0；当前 SDK 的 reasoning 类型支持 `context="all_turns"`。

### 2.3 150 并发与评估

沿用 `src.val` 入口：

- batch size 默认改为 30，rollouts 保持 5。
- semaphore 使用 `batch_size × rollouts`，指定配置下为 150。
- 保留每批现有调度方式，不重排问题、不增加提前停止或额外指标。
- 每个 rollout 完成答题与 judge 后立即写入自己的 `result.json`。
- 恢复时读取已有结果，补跑缺失项；沿用同一 W&B run 的恢复方式。
- 准确度继续为全部 rollout 的平均 `score / 5`；失败结果按现有协议计 0。
- 保存配置中的模型、官方 endpoint、reasoning、并发参数、skills 路径和图像设置，便于识别实际实验配置；不保存密钥。

入口使用官方 API 和初始 skills 作为默认配置。保留现有运行目录、测试文件、runtime root、embedding endpoint 等必要参数，不增加不相关 CLI 功能。

实际吞吐受账号 API 限流影响；不通过降低模型、reasoning 或 rollout 数绕过限制。

### 2.4 480p 采样、单帧裁剪与证据一致性

替换现有 contact sheet 生成路径，改为独立帧图片。

**采样和时间：**

- 继续由 agent 决定采样区间、fps、是否读取 captions 和 crop。
- 使用 ffmpeg 的保留时间戳采样方式，记录被选中源帧的真实 PTS，并转换为逻辑时间轴时间。
- 不再用 `start + frame_index / fps` 为被重采样的画面生成时间标签。
- 不使用统一时间偏移修复；历史错误取决于实际帧选择。
- 保存选中帧的原始分辨率图像，供后续裁剪；发送模型的版本保持比例，最大约 `854 × 480`。
- 每个发送的图像项配有 `frame_id`、实际时间、文件名和 crop 信息。模型可见内容不包含宿主机路径或其他 timeline 信息。

**工具接口：**

- `sample_video` 保留现有时间、fps、caption 和 crop 参数，返回独立帧及其元数据。
- 新增 `crop_frame(frame_id, crop_left, crop_top, crop_right, crop_bottom)`。
- 所有 crop 坐标都相对于原始完整帧，单位为 `0–1000`；完整画面为 `0,0,1000,1000`。
- 合法边界为 `0 ≤ left < right ≤ 1000` 和 `0 ≤ top < bottom ≤ 1000`。
- 裁剪始终读取该 frame record 对应的原始分辨率帧，返回新的图像记录和相同源时间。

**提交和证据：**

- 内部 `submit_clip_answer` 改为引用 `evidence_frame_ids`，避免模型自行编写一个没有对应画面的时间。
- `answered` 至少引用一张实际返回给视觉 agent 的图像；所有引用必须属于当前 `clip_qa`。
- 外层 `clip_qa` 继续返回 `status`、`answer`、`evidence_times_s` 和证据图；时间从帧记录派生。
- 主 agent 收到的证据直接复用被引用图像文件，不再按时间重新 seek 视频。
- 使用 SDK 的 `ToolsToFinalOutputResult` callback，仅在 `submit_clip_answer` 成功时结束视觉 loop；校验错误作为 tool result 交回模型纠正。

### 2.5 累计图片上限

采用 **每个 `clip_qa` 累计 1,024 张图片**：

```text
单张上界：854 × 480
patch 数：ceil(854 / 32) × ceil(480 / 32) = 405
Luna 图片 token 估算：ceil(405 × 1.2) = 486
1,024 张：497,664 tokens
上下文剩余：1,050,000 − 497,664 = 552,336 tokens
```

计数规则：

- `sample_video` 和 `crop_frame` 共用累计额度。
- 同一图片通过新的工具返回再次发送时重新计数；SDK 重放已有历史不视作新增图像。
- 每个 `clip_qa` 的图片工具共用锁，防止并行调用各自通过额度检查后累计超额。
- 超额请求整体返回错误及剩余额度；不截断结果、不自动降低 fps、不自动缩小区间。
- 额度只在图片成功进入工具返回时消耗。
- 输出中保留累计数量和剩余额度，便于模型决策及 trace 检查。
- 视觉 agent 不做 compaction，也不增加强制猜答案分支。

该限制控制图片导致的上下文增长，不能保证任意长的文本或 reasoning 永不溢出。其余溢出继续显式返回原错误，不截断上下文。

[图片计数依据](https://developers.openai.com/api/docs/guides/images-vision)

## 3. 已完成审计、接受的修复与证据

审计流程已完成，无需重做：

- `trace_audit_early`：独立检查早期 batches。
- `trace_audit_late`：独立检查后期 batches。
- 两者均未读取 `notes/CLAUDE_PROBLEMS.md`。
- `adversarial_filter`：对独立发现与 Claude 问题进行筛选，并自行读取了 `REVIEW.md`。
- 三个 agent 已结束；后续审查必须创建新 agent，不能复用。

历史证据共同根目录：

```text
/raid/wg25r/video_sql_v2/runs/qwen38_27b_fp8_epd_fresh_3x10_seed0/train/epoch-000/
```

| 接受的问题 | 已检查证据 | 实施行为 |
|---|---|---|
| 成功答案被 compaction 忽略 | `batch-014/question-214/rollout-01/attempt-1`：成功提交后仍进入 compaction；`question-298/rollout-05`：提交后继续调查并改写答案 | 先接受有效 final submission，再处理 compaction |
| 无效视觉提交提前结束 | `batch-000/question-303/rollout-00/attempt-1`：空参数产生校验错误，却成为最终输出；`batch-015/question-140/rollout-09`：证据时间越界后同样结束 | 成功提交才终止，错误返回模型 |
| SQL 同名列静默覆盖 | 实际 JOIN 查询返回两个 `row_id`；转为字典后只剩后者 | 检测重复列名并报错，要求不同 `AS` 别名 |
| 采样标签与证据帧不一致 | `batch-008/question-880/rollout-08/attempt-1`：标签 52067 的画面与独立 evidence-52067 不同；ffmpeg 检查选中帧实际时间约 52067.9667。另有 `batch-015/question-888/rollout-06` | 保留真实 PTS，证据直接引用已看过的帧 |
| crop 单位不清楚 | 历史采样存在坐标全 0 的错误调用 | 明确单位、完整画面参数和边界 |
| SQL 结果截断影响证据 | `batch-000/question-303/rollout-08/attempt-1`：50k 截断发生在后续相关行之前；另有 `batch-013/question-553/rollout-08` | 删除截断，完整交给子模型 |
| judge 缺少问题语境 | `batch-015/question-888/rollout-06/attempt-1`：候选答案为国家名、reference 为选项编号，judge 缺少题目无法对应 | prompt 加入原始问题，维持原评分标准 |
| 图片累计导致视觉上下文溢出 | `batch-009/question-95/rollout-06/attempt-1`：已累计 161 张 contact sheets；后续请求达到 326493 tokens，超过当时服务端 314573 | 480p 单帧加每个 clip 累计额度 |

SQL 重复列的实际复现查询：

```sql
SELECT c.row_id, f.row_id
FROM topic_chains c
JOIN topic_facts f ON f.chain_row_id = c.row_id
LIMIT 1;
```

修复要求是显式报错，不能自动改列名或悄悄改变返回 schema。

`sql_qa` 删除以下行为：

```text
serialized[:50000] + 截断说明
```

保留 `run_sql` 原有 100 行和单元格 500 字符的预览规则；该规则与 `sql_qa` 全量结果不同。

**筛掉的问题不进入此次实现：**

- 仅训练动态技能才会触发的 catalog 丢失、技能重名和 updater 问题。
- 没有实际证据支持的 reasoning 类型、序列化 fallback、missing usage 等推测。
- 连续三次相同 completion 检测的假阳性推测。
- 工具调用频率低、工具未被使用等模型行为问题。
- 已有文档说明的 20 分钟 clip 限制。
- 修改 judge 模型、重试协议、评估集或新增指标。
- 所有 reflection、summarizer、updater 的训练错误。
- 历史前端 `[object Object]` 问题。

## 4. 数据、文件和执行纪律

运行时数据直接使用：

```text
data/runtime/{day,week,month}/index.duckdb
data/runtime/{day,week,month}/embedding_index/
data/runtime/{day,week,month}/media-map.json
```

已有评估文件为 `data/test-seed-0.json`，包含 300 题。主 agent 只获得原始问题和当前隔离数据库，不获得 reference、question_type、clue_interval 或 temporal_certificate。

数据集已存在：

```text
/raid/huggingface/hub/datasets--MM-Lifelong--MM-Lifelong/snapshots/248aa82039a574e63a2e524746a7cd8f32330443/
```

不下载、不重建索引、不覆盖旧 run。媒体路径按实际 `media-map.json` 读取。HF cache 为 `/raid/huggingface`。

现有 embedding 配置：

```text
endpoint: http://127.0.0.1:8013/v1
model: qwen3-vl-embedding-8b
```

其当前可用性尚未在本轮确认，不将历史容器状态作为健康证明。

实施顺序：

1. 将本计划写入 `TRAINING_FREE_PLAN.md`，单独提交。
2. 删除训练链路，迁移评估依赖，提交。
3. 修改官方 API、reasoning、judge、并发和答案停止逻辑，提交。
4. 修改 SQL、视频采样、帧引用、裁剪和图片额度，提交。
5. 更新相关技能说明、Dockerfile、`PLANS.md` 与交接文档，提交。
6. 执行新的独立 diff review，修复成立的问题并提交。

遵守仓库现有规则：

- 主 agent 不读取 `REVIEW.md`；review agent 自行完整读取。
- 每次编辑后提交，包括文档；Git 作者使用 `codex <codex@localhost>`。
- commit/diff 前用 `git ls-files --others --exclude-standard` 列出未跟踪路径。
- 只暂存明确属于本任务的文件；不执行包含用户修改的仓库级自动内容 diff。
- 不读取或输出 `.env` 密钥值，不搜索其他目录的 API keys。
- shell、读取和编辑按现有要求在 sandbox 外执行；遇到真实拒绝不能换机制绕过。
- 不新增测试、syntax checks、preflight 或 smoke tests。
- 不引入 fallback、自动截断、降低设置、通用重试框架或无关重构。
- 使用现有依赖；ffmpeg/ffprobe 已存在。当前环境没有 PyAV/OpenCV，不以它们作为既有依赖。
- 不新增后台 watcher，不使用 tmux/nohup；长任务使用 harness 前台 session。
- 不启动带 `--rm` 的 Docker 容器。

## 5. 审查、验收与交付

完成实现后，创建一个全新的通用 review subagent，指定本分支起点与最终提交，以及本任务明确修改的文件。委派形式：

```text
Review the diff between b4799309eb060f47ccd8db4e77cb22d94c2c85ed and <最终提交>
against REVIEW.md for the purpose of the training-free OpenAI video-understanding
harness for files [本任务修改文件列表].
```

要求 reviewer 自行完整读取 `REVIEW.md`。主 agent 根据真实源码和证据核实每个发现，修复成立的 plain-yes 项。修复较小时不重复 review；修复较大时另建 reviewer 审查修复后的 diff。

代码审查覆盖这些具体场景：

- 成功 `submit_answer` 与 compaction 同轮触发，答案仍被接受。
- 无效 `submit_clip_answer` 返回错误，视觉 agent 可以继续纠正。
- SQL 重复列明确报错，`sql_qa` 输入包含完整 SQL 结果。
- Judge 实际输入包含 question、reference、candidate。
- 帧时间来自源 PTS；sample、crop、最终 evidence 指向同一源帧。
- 多次采样、裁剪和并行工具调用共享累计图片额度；超额请求没有部分成功。
- 所有 LLM 角色使用官方 endpoint、指定模型和 medium reasoning，历史 reasoning 被保留。
- 30×5 调度没有被旧 semaphore=24 限制。
- 已完成 rollout 能逐个落盘并恢复。
- 不存在训练入口、reflection 或自动技能更新调用。

本轮不将代码审查或历史 traces 当作新实现的运行验证。

后续收到正式评测启动指令时：

- 使用新的 run 目录，避免复用已作废历史结果。
- 正式运行前按仓库规则调用 Claude CLI 做一次全仓库 review，并让 Claude 自行完整读取 `REVIEW.md`。
- 直接开始可恢复的正式评测，不另跑 smoke/preflight。
- 从正式产物中人工核对模型请求、reasoning 历史、工具输入输出、实际图片、judge 输入和逐 rollout 结果。
- 外部资源不可用时立即报告具体状态，不自行等待或更换方法。
- 监控遵守仓库前台 session 和稳定后低频轮询规则。

最终交付报告包含分支、提交、实际修复项、review 结果，以及尚未进行正式运行验证的事实。

## 6. 实施交付记录

本计划的代码和文档修改已提交在 `codex/training-free-harness`。新的独立 review subagent 已自行读取 REVIEW.md 并审查 `b4799309..cb5ad4e7` 的指定文件 diff，结论为 `no blocking issues`。用户补充的完整 trace 要求已实现。当前入口和 trace 文件布局见 PLANS.md；历史审计证据保留在本计划第 3 节。

## 7. 正式运行交接（2026-09-06 UTC）

用户要求继续完成并授权 embedding 使用任意空闲 GPU 后，已运行正式评测。运行代码为 `7510e5356e290023b65358541fb1b4f2f36f65a1`。启动前 Claude CLI 使用 claude-opus-4-8 完成全仓审查，自行完整读取 REVIEW.md，结论为 `no blocking issues`，无权限拒绝；原始报告保存为 `runs/training_free_luna_seed0_7510e535/claude-full-review.json`。没有执行 syntax checks、preflight 或 smoke tests。

实际启动命令：

```bash
PYTHONUNBUFFERED=1 HF_HOME=/raid/huggingface .venv/bin/python -m src.val \
  --test-file /raid/wg25r/video_sql_v2/data/test-seed-0.json \
  --runtime-root /raid/wg25r/video_sql_v2/data/runtime \
  --skill-dir /raid/wg25r/video_sql_v2/skills/initial \
  --run-dir /raid/wg25r/video_sql_v2/runs/training_free_luna_seed0_7510e535 \
  --model-name gpt-5.6-luna --api-key-env PERSONAL_OPENAI_KEY \
  --embedding-url http://127.0.0.1:8013/v1 \
  --embedding-model qwen3-vl-embedding-8b \
  --attempts 3 --batch-size 30 --rollouts 5 \
  > runs/training_free_luna_seed0_7510e535/eval.log 2>&1
```

该命令是已经执行的历史命令。不要直接重跑并覆盖 eval.log；恢复条件见下文。

- 新 run 目录：`runs/training_free_luna_seed0_7510e535`。配置文件已核对：官方 endpoint、Luna、medium/all_turns、150 并发、静态初始 skills、854×480 和每 clip 1,024 张图片。
- W&B：`https://wandb.ai/3dsmile/video-sql-v2/runs/tityz8z2`，ID 为 `tityz8z2`。日志记录 W&B 文件同步完成。
- 进程最终退出码为 0，所有 1,500 个 result.json 均已写入；其中 1,447 个为 answered，53 个为 failed。汇总 score_sum=2928，accuracy=0.3904，包含失败计零，不能当作完整有效的最终评测结果。
- 53 个失败的最后一次 error.json 均包含 OpenAI APIError：`You have no credits remaining.`。共 158 条 attempt 错误的异常链为外层 TypeError `object NoneType can't be used in 'await' expression`、内层余额不足 APIError；另有 37 条 AnswerSubmission 格式校验错误。余额不足已向用户报告；没有换 key、模型、reasoning 或启动新 API 调用。
- Weave 写出了实际 call 链接，但退出前出现单条 calls_complete 超过 32,505,856 bytes 的警告；未逐条核实远端上传完整性。完整本地 traces 和媒体均保留，不以进程退出码证明 Weave 上传完整。

正式产物核对范围：人工检查了实际工具输入输出、多轮 encrypted reasoning、judge 的 question/reference/candidate，以及打开的证据图片。`question-month-742/rollout-04` 的采样帧与最终 evidence 引用相同图像文件；画面中的肖像脸部为蓝色。`question-month-40/rollout-02` 的无效 frame ID 提交返回错误，之后成功纠正。`question-month-528/rollout-02` 的超额采样整体被拒绝，后续请求累计从 510 增至 750、990。`question-week-185/rollout-04` 的 sql-qa 输入为 110,153 字符，其中 25 行与同一 SQL 查询的完整数据库结果逐行相同。未逐张人工检查全部图片，也未从本次运行验证未触发的 compaction 场景。

运行后修复 `5f4eb63f`：余额不足或 insufficient_quota 不再归入可重试错误；工具边界向上抛出该 APIError，错误日志打印完整异常链。此为局部修复，未重复 review，尚未经过新 API 运行验证；瞬态错误仍保留原有重试次数。

Embedding 恢复及清理：原 `egolife-qwen3-vl-embedding-8b` 容器保持停止。曾在 GPU 3 启动 `video-sql-training-free-embedding-7510e535`，使用原镜像 `sha256:bd995759b5b8ac51062e04c9e4d7c91c382d1ba377bb787e24dca2ccb39925e9` 和 Qwen/Qwen3-VL-Embedding-8B，显存预算从旧 0.45 改为 0.40，保留其他原设置；正式 semantic_search 返回了实际检索结果。评测后该临时容器已停止并删除，其前台 session 退出码为 0，日志保存在 run 的 embedding.log。当前没有本任务仍在运行的前台任务。

恢复前必须先解决 PERSONAL_OPENAI_KEY 所属账号的 API credits。当前恢复逻辑会直接读取任何已有 result.json，包括 failed；因此直接重新执行入口不会补跑这 53 项。收到恢复指令后，应保留原始失败目录和汇总记录，把确认由余额不足影响的 53 个 rollout 目录移到 run 内独立归档路径，再沿用同一 W&B run 和原评估设置补跑缺失项；不能删除旧 traces 或覆盖已有日志。还需恢复同模型 embedding API，按当时空闲 GPU 状态选择设备。不要在余额尚未恢复时继续重试。
