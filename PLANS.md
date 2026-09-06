# Training-free Video Understanding Harness

当前实施与接手计划见 [TRAINING_FREE_PLAN.md](TRAINING_FREE_PLAN.md)。旧训练部署、结果和事故记录见 [notes/DOCS_AND_HANDOFF.md](notes/DOCS_AND_HANDOFF.md) 的历史部分。既有实验结果及训练后 skills 只保留追溯，不作为新评测的恢复依据。

## 方法与运行数据

Agent 在隔离的 SQL 运行库中检索视频记忆，必要时检查源视频帧，然后提交答案。只使用 `skills/initial/` 的静态 ROOT-SKILL 和六个检索技能；不运行 reflection、summarizer 或 updater。

使用现有 `data/runtime/{day,week,month}/`，每个目录包含 `index.duckdb`、`embedding_index/` 和 `media-map.json`。媒体路径按实际映射读取。数据集快照在 `/raid/huggingface/hub/datasets--MM-Lifelong--MM-Lifelong/snapshots/248aa82039a574e63a2e524746a7cd8f32330443/`；不下载或重新索引，HF cache 使用 `/raid/huggingface`。

评估文件为 `data/test-seed-0.json`，共 300 题。只把问题文本、当前库 schema、ROOT-SKILL 和技能目录交给答题 agent，不暴露 reference、question_type、clue_interval、temporal_certificate 或其他 timeline/video/split。普通技能通过 `read_skill` 读取全文。

文本归一化保持现有实现：NFKD、移除组合标记、标点映射为 ASCII、转小写、折叠空白；保留非英文字符。

## 模型与评估

- 所有 LLM 角色使用 OpenAI 官方 `https://api.openai.com/v1` Responses API，经 OpenAI Agents SDK 调用。
- 使用项目 `.env` 的 `PERSONAL_OPENAI_KEY`，模型 `gpt-5.6-luna`，reasoning effort 为 `medium`。
- `reasoning.context=all_turns`、`store=false`，保留 `reasoning.encrypted_content` 和 reasoning item IDs；正常多轮对话保留所有 reasoning 输出项。
- 不设置 temperature、top_p 或输入/输出 token 上限，不降低 reasoning 或切换模型绕过失败。
- Embedding 保持 `http://127.0.0.1:8013/v1` 的 `qwen3-vl-embedding-8b`，使用现有索引。
- 每批 30 题、每题 5 个 rollout；并发为 batch size × rollouts，即 150。保留问题顺序和逐 rollout 落盘、恢复。
- Judge 接收原始问题、reference answer、candidate answer。按最终答案语义等价给 0–5 整数分，利用问题解释选项编号，忽略附带标识符。
- 评估指标保持全部 rollout 的平均 score/5，失败按既有协议计 0。
- W&B 和 Weave 均使用 `3dsmile/video-sql-v2`，保留原有恢复方式。
- SDK max_retries 和 rollout/judge attempts 均保持既有设置，不增加重试框架。

## 工具与视觉证据

- `run_sql` 只读查询、30 秒超时，返回最多 100 行；单元格保留 500 字符和原长度。重复 SQL 列名报错，要求使用不同 AS 别名。
- `sql_qa` 将完整 SQL 结果交给同模型回答，不限制行数或截断字符；API 溢出明确返回错误。
- `semantic_search` 使用已有七张表的向量，返回 row_id、时间、preview 和分数；rank_start 从 1 开始，分数边界使用 JSON null。
- 主 agent 保留 `add_note`、`read_notes`、`read_skill`、`agent_raise` 和 `submit_answer`。
- `clip_qa` 最多接受 20 分钟。内部工具为 `sample_video`、`crop_frame`、视觉 notes 和 `submit_clip_answer`。
- `sample_video` 返回独立图片、frame_id 和真实源帧时间，保留 fps、区间、caption 和 crop 参数。ffmpeg 使用保留源 PTS 的 select，图像同步为 passthrough，不以 fps 输出序号生成时间。
- 保存原始分辨率帧；送模型的图像保持比例，最大 854×480。`crop_frame` 从保存的原始帧裁剪，而不是从已缩小的图像裁剪。
- Crop 坐标始终相对于原始完整帧，单位 0–1000。完整画面为 0,0,1000,1000；要求 left < right、top < bottom。
- 采样和裁剪共用每个 clip_qa 累计 1,024 张图片额度；并行调用通过同一锁检查。超额请求整体报错，返回剩余额度，不部分返回或自动降低 fps。
- 内部提交引用当前 clip 已返回的 evidence_frame_ids。answered 至少需要一张证据，校验失败交回模型；只有成功提交才结束 loop。
- 外部结果保持 status、answer、evidence_times_s 和证据图。时间从帧记录派生，证据直接复用视觉 agent 看过的图像。
- 工具内容只暴露图像文件名，不暴露宿主机媒体路径；本地调试记录保留完整来源。

480p 图像的预算计算与局限见 TRAINING_FREE_PLAN.md：图片额度控制图片增长，不保证任意长文本或 reasoning 永不溢出。视觉 agent 不做 compaction。

## 停止、compaction 与完整 trace

所有 agent loop 保持最多 100 轮。连续三次排除 reasoning 后的 completion 完全一致，按既有 InvalidTrace 协议处理。普通工具错误返回 error JSON，agent_raise 仍主动结束当前 rollout；不引入静默 fallback。

主 agent 容量为 1,050,000 tokens，剩余空间小于等于 178,000 时 after_turn 触发 compaction。成功 submit_answer 优先于同轮 compaction 标记。由同一 Agent 在原会话中生成 handoff，前两次后继续，第三次后返回 compaction_limit、得分 0、不调用 judge。视觉 agent、sql_qa 和 judge 不做 compaction。

Training-free 也保存完整 trace：

- `traces/*-segment-*.json`：指令、模型设置、工具 schemas、输入、完整历史、raw responses、usage、request/response IDs、最终输出和错误。
- `traces/*-events-*.jsonl`：从 run_start 开始逐事件落盘，包括 provider 流事件、工具调用和返回；异常发生前的已收到事件也保留。
- 主检索、clip、sql_qa、judge 和 compaction 都通过同一记录路径。
- JSON 中去除重复 base64，图片独立保存；`media/frame-*.json` 关联原始帧、源 PTS、时间基准、裁剪和实际发出的 JPEG。
- ffmpeg 命令、stderr 和原始分辨率 PNG 保存在各 sample 目录，供人工检查。
- 每个完成 rollout 的 trace_files 列出该 attempt 的各阶段 JSON traces。

## 入口与交付约束

`src/val.py` 是独立评估入口，运行 `python -m src.val`。必需参数为 `--test-file`、`--runtime-root`、`--run-dir`、`--embedding-url`、`--embedding-model`。模型、key 名称、静态 skills、batch size 和 rollouts 使用本计划中的配置。官方 endpoint 不接受本地 base-url 覆盖。

Dockerfile 位于 `docker/harness/Dockerfile`；容器中读取项目 `.env`，因此运行时需要将该文件挂载到工作目录，不把密钥烘焙进镜像。本轮不构建、部署或启动正式评测。

每次编辑后提交，只提交本任务文件，保留用户原有修改和删除。主 agent 不读 REVIEW.md，新的 review subagent 自行完整读取。正式评测启动前还需 Claude CLI 全仓 review。不得新增测试、syntax checks、preflight 或 smoke tests；运行正确性只从后续正式结果、实际工具 traces 和图像验证。

长任务使用 harness 前台 session，稳定后按 AGENTS.md 低频监控，不使用 tmux/nohup/watcher。外部资源不可用时立即报告，不等待或替换方法。不启动带 --rm 的容器。
