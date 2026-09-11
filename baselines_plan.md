# Baselines 计划（EgoLife week：training-free harness 与 MAGIC-video 对比）

本文件在执行时整体替换 `baselines_plan.md`，不是它的附件。所有约束、步骤、已定参数和待确认项都在这里。执行中用户改动过的决定已经合并进来，原文不再保留。

## 0. 背景与目标

- 同一目录、同一分支（`codex/trace2skill-baseline`）上另有一个 agent 在跑 Trace2Skill rollout（入口 `src/trace2skill_baseline.py`，run 目录 `runs/qwen38-fp8-context128k-day-week-tree-month-val-seed0`）。不碰 `src/`、`skills/`、`third_party/`、它的 run 目录，不改动、不重启任何 vLLM 容器。
- 用户最新指令（执行中改）：不使用我们自己的 week 索引，只用 MAGIC 的 artifacts，跑两套方法。
- 目标一：克隆 MAGIC-video（https://github.com/lijiazheng0917/MAGIC-video.git），改为使用同一个 Qwen LLM 与同一个 embedding 服务，索引用它自己发布的 magic-video-artifacts（EgoLife A1_JAKE），在 EgoLifeQA 上跑 baseline；视频用 MM-Lifelong week 文件。
- 目标二：把 MAGIC 的 artifacts 转换成我们的索引格式，用我们 harness 的 training-free 形态（静态 `skills/initial`，不做 reflection/evolution）跑一遍。
- 目标三：两个 run 按总分、均分、准确率（judge 满分比例）做 overall 与按 EgoLife 题型的对比，写成一份简报和一份详细背景。
- 已存 embedding 的核对：`/raid/wg25r/video_sql_v2/data/pilot/stage2/` 的原始构建曾用 MAGIC 官方 30 秒 caption（`run/dense/week__*` 标 `kind=official`）走我们的 Qwen 结构化流程；其 duckdb 已被重生成版覆盖，但旧 embedding shards（`embedding_shards/text/*`、`embedding_shards/video/visual_clips`）与每条请求的原文 trace（`run/traces/embeddings/`）仍在。用户要求在空闲 GPU 上新起一个 vLLM embedder 核对这些向量，核对通过后 MAGIC 图的 30 秒 caption 节点与 visual_clips 节点直接复用它们。

## 1. 边界与纪律

1. 计划阶段只读了当前目录，没有访问 git 历史、目录以外路径或本机已有的 MAGIC 副本。执行阶段继续成立：不运行任何读历史的 git 命令（`log`、`show`、`diff <commit>` 等），不查找、不打开本机已有的 MAGIC-video 副本（用户批准克隆的新鲜副本除外）。
2. 目录以外的资源，用户已逐项批准如下；未列出的一律不碰：
   - 只读 `/raid/wg25r/video_sql_v2/data/runtime/week/`（`index.duckdb`、`embedding_index/`、`media-map.json`）。
   - 只读 `/raid/huggingface` 下 MM-Lifelong 快照中的 week 源视频（media-map 里列出的文件）。
   - EgoLife 不下载，用本地已有的 `/raid/huggingface/datasets--lmms-lab--EgoLife/snapshots/143fb319be7aa5ae210c936bf4f0f3a86092afb0/`（只读 QA；视频一律用 MM-Lifelong week 文件，本地 EgoLife 视频副本不完整、不使用）。
   - MAGIC artifacts 用本地已有的 `/raid/huggingface/hub/datasets--jiazhengli7--magic-video-artifacts/snapshots/8573f317684ce55c0999f86386666d3d3fe76ae3/egolife/A1_JAKE/`（只读）。
   - 访问 `127.0.0.1:8003`（Responses/Chat 代理）和 `127.0.0.1:8014`（embedding），以及 `docker ps` / `docker inspect` 只读核对。
   - `git clone` GitHub 上的 MAGIC-video。
   - `systemd-run --user --scope` 给 MAGIC 进程设内存上限。
   - 删除操作用 `mv` 到 `/raid/wg25r/trash/` 代替 `rm`（`rm -rf .git` 被拒后用户指定）。
   - 只读 `/raid/wg25r/video_sql_v2/`（查找已存的 MAGIC 相关 embedding；实际用到 `data/pilot/stage2/embedding_shards/`、`data/pilot/stage2/run/traces/embeddings/`、`data/pilot/stage2/run/dense/`、`INFRA.txt`、`notes/`）。
   - 启动一个新的 vLLM embedding 容器做核对：`egolife-embedding-verify`，GPU 7，宿主端口 8015，镜像 `vllm/vllm-openai:qwen38-flash-next`，`Qwen/Qwen3-VL-Embedding-8B`，served name `qwen3-vl-embedding-8b`，`--runner pooling`，`--gpu-memory-utilization 0.45`，其余参数同 `INFRA.txt` 的 embedding 配方，不加 `--rm`；核对完先不停，等用户指示。
3. 服务不在运行、数据缺失、映射对不上、字段与预期不符：停下来问用户，不换模型、不换索引、不造替代数据、不跳过阶段。
4. git：在当前分支直接提交，每次编辑后提交；只 `git add` `training_free/**`、`baselines_plan.md`、以及本计划新建的 markdown。不 add 别人的未跟踪/已修改文件（`AGENTS.md`、`src/analysis.py`、`src/check_status.py`、`notes/INCIDENT_*` 等）。不用 `git stash`。
5. venv：两个，执行时发现同名 venv 已存在（非本会话所建），用户批准 `uv venv --clear` 重建：
   - `.venv_codex_training_free`：`uv venv --clear .venv_codex_training_free`，然后 `UV_PROJECT_ENVIRONMENT=.venv_codex_training_free uv sync`（按 `uv.lock`）。已完成。
   - `.venv_codex_magic`：`uv venv --clear .venv_codex_magic`，在 `training_free/MAGIC-video/` 下 `UV_PROJECT_ENVIRONMENT=<绝对路径>/.venv_codex_magic uv sync`（按它的 `uv.lock`，含 torch 2.7.1+cu128、hipporag/egorag editable），再 `uv pip install` 补 `wandb`、`weave`、`duckdb`、`numpy`、`httpx`、`openai`、`openai-agents`（judge 走 `training_free/val.py`）。已完成，`worldmm` 可 import。
6. 代码风格按仓库规则：研究代码，不加防御、不加 fallback、不设 token 上限、不降 reasoning。大运行前按 `REVIEW.md` 由新的 review subagent 审 `git diff`，主 agent 不读 `REVIEW.md`。
7. 长任务用 harness 前台方式直接运行（不 `nohup … &`），日志写到 run 目录；用定时 Monitor 低频查看，不 sleep 轮询。每个阶段完成后先看真实产物与 trace，再报告。

## 2. 目录布局

```
training_free/
  harness.py          # 复制 src/harness.py，仅改 semantic_search 的 row_id 掩码（见 4.3）
  val.py              # 复制 src/val.py（judge_answer、retryable、write_error、milestone）
  normalize.py        # 复制 src/normalize.py
  tracing_bridge.py   # 复制 third_party/trace2skill/tracing_bridge.py（RunMetrics、init_weave、response_cost）
  skills/initial/     # 复制 skills/initial/ 七个 md，运行时逐字节核对未变
  prepare_egolife.py  # 核对、出题、生成按截止时刻过滤的 duckdb 与 media-map（对转换后的索引做）
  verify_embeddings.py   # 用 8015 新 embedder 重算旧 shards 抽样行（每表 50 行，视频 20 个），比较余弦，写 report
  convert_magic_index.py # 把 MAGIC artifacts 转成我们的 duckdb + npy 格式（细节见第 5 节待定项）
  build_magic_graph.py   # 从 MAGIC artifacts 建 MAGIC 自己的 graph.pkl：30sec/visual 节点向量复用旧 shards，其余文本向量走 8014
  run_training_free.py# 我们方法的 EgoLife 评测入口（只有 magic-index 一种索引）
  MAGIC-video/        # 新鲜克隆，commit 0f2273f7f07360cc58588399553b5a6ea758b084，内嵌 .git 已移到 /raid/wg25r/trash/，由本仓库跟踪
  run_magic.py        # MAGIC 的评测入口（HTTP embedder、LLM 端点、并发 10、judge、W&B）
  compare.py          # 汇总两个 run，写 RESULTS.md / RESULTS_DETAILS.md
  RESULTS.md          # 简报：只有对比数字
  RESULTS_DETAILS.md  # 详细背景：设置、commit、核对结果、W&B 链接、运行统计、偏差说明
runs/egolife-week-shared/          # 两个 run 共用的产物（runs/ 已被 gitignore）
  egolife_raw/                     # QA 原始 json 副本
  verification/                    # 视频时长、映射、抽帧图与 report.json；embedding 核对 report
  questions.json                   # 最终题目（开放式）
  cutoffs/<cutoff_s>/media-map.json  # 截断到截止时刻的 media-map
  magic_index/index.duckdb         # MAGIC artifacts 转换成我们格式的全量库
  magic_index/embedding_index/     # 其 npy
  magic_index/cutoffs/<cutoff_s>/index.duckdb  # 转换库的按截止过滤版本
  magic_graph/graph.pkl            # MAGIC 自己用的图（时间用 DHHMMSSFF，它自己按提问时刻过滤）
runs/training-free-egolife-week-magic-index/ # 我们方法 + MAGIC 索引转换
runs/magic-video-egolife-week/               # MAGIC + MAGIC 索引
```

两个 run 目录名同时是 W&B run 名，用户已批准（原计划里的 `runs/training-free-egolife-week` 按最新指令不再跑）。

## 3. 已确定的参数

| 项 | 值 |
| --- | --- |
| 复制进 training_free 的文件 | harness.py、val.py、normalize.py、skills/initial、tracing_bridge.py（已完成并提交） |
| 题目 | lmms-lab/EgoLife 本地快照的全部 EgoLifeQA（A1_JAKE，500 题），不抽样 |
| 开放式改写 | 规则去选项：题干原样，删除四个选项，正确选项文本作 reference answer，不经 LLM |
| 题目文本格式 | `Asked on DAY3 at 11:22:02 (timeline 52340.5 s). <原题干>`：天 + HH:MM:SS（帧号丢弃）+ 按下一行映射得到的 week 时间轴秒数 |
| query_time → cutoff_s | week 时间轴由 EgoLife 同名 30 秒片段按录制顺序拼接，录制空档被去掉，天与天之间另有约 15 秒间隔，"该天起点 + 时钟差"不成立。映射：找到文件名时钟范围覆盖提问时刻的片段，cutoff = 该片段 timeline_start_s + (提问时刻 − 片段文件名时钟)；提问时刻落在空档里的题（ID 34、289、341、346）用前一个片段的 timeline_end_s。帧号按 20 fps 换算 |
| 硬限制实现 | 每个不同截止时刻生成一份过滤后的 duckdb 与截断的 media-map；不重新生成 npy，semantic_search 按过滤库中存在的 row_id 掩码 |
| 过滤条件 | 区间表保留 `start_s <= 截止`（即使 end_s 在截止后）；点时间表 `snapshot_s`/`time_s <= 截止`；entities 保留 `first_seen_s <= 截止` 且 `last_seen_s` 截断到截止；chains 保留 `start_s <= 截止`，summary 原样 |
| clip_qa | 也限制：media-map 只保留截止前的源视频，最后一个的 `timeline_end_s` 截到截止时刻 |
| harness 设置 | 全部沿用 `src/trace2skill_baseline.py` 的 `make_harness`/`rollout`/`score_rollout`：模型 `Qwen/Qwen3.8-27B-FP8`、base_url `http://127.0.0.1:8003/v1`、reasoning xhigh + summary auto、clip_qa_video 开、650 图额度、prompt_cache_key、attempts 3、耗尽后 skip 整题、judge 同模型 0–5 分 |
| embedding | `http://127.0.0.1:8014/v1`，served name `qwen3-vl-embedding-8b`，4096 维；目前只接纯文本（图像/视频请求被服务器拒绝，见第 5 节） |
| rollout | 每题 1 条 |
| 并发 | 两个 run 都是 10 个题同时进行（我们方法用 `MainAgentSlots`，上限改 10） |
| W&B / Weave | entity `3dsmile`，project `video-sql-v2`，run id 与 name 等于 run 目录名，Weave 开启；都用 `RunMetrics`；MAGIC 用不到的 replacement 指标记 0 |
| MAGIC 克隆 | `training_free/MAGIC-video/`，main 分支 commit `0f2273f7f07360cc58588399553b5a6ea758b084`；内嵌 `.git` 已 mv 到 `/raid/wg25r/trash/MAGIC-video-dot-git-0f2273f7`；已提交 |
| MAGIC 索引 | 用 artifacts 自己的 captions（30sec/3min/10min/1h）、openie、semantic consolidation、topic_chains.json、step3_enriched_chains.json；转换一份全量 graph.pkl（时间 DHHMMSSFF），until_time 传提问时刻，由它自己的 filter_by_time 过滤；不为每个截止时刻单独建图 |
| MAGIC embedding | 唯一不用其原始索引的地方：不装进程内 Qwen3-Embedding-4B / VLM2Vec。30 秒 caption 节点的文本向量与 visual_clips 节点的视频向量复用 `video_sql_v2/data/pilot/stage2/embedding_shards/` 里官方 caption 版 week 的旧向量（核对通过后）；3min/10min/1h 摘要、NER 实体、consolidation 三元组的文本向量和查询向量走现有 8014 `qwen3-vl-embedding-8b` |
| 旧向量核对 | `verify_embeddings.py`：从旧 shards 随机抽 week 行（文本 6 表各 50 行、visual_clips 20 个），用 trace 里保存的原文 / clip 文件在 8015 新 embedder 重算，比较余弦，写 `verification/embedding_check.json` |
| MAGIC LLM | retriever 与 respond 两个角色都用 8003 的 `Qwen/Qwen3.8-27B-FP8`；不传 reasoning 参数；原代码 respond 角色的采样参数（temperature 1.0、top_p 0.95、top_k 20、min_p 0、presence_penalty 1.5、repetition_penalty 1.0）全部删掉，用服务器默认；`parse_time_reference` 里写死走 OpenRouter 的 `qwen/qwen3.5-flash-02-23` 也改到 8003 同一 Qwen |
| MAGIC 提问方式 | 开放式：choices 不传，`question` 字段与我们完全相同；推理模板仍用 `memory_reasoning_unified_egolife`；回答模板换成它自己的 `qa_mmlifelong` / `qa_mmlifelong_chain`；until_time 传提问时刻 DHHMMSSFF；答案用同一个 judge 打 0–5 分 |
| MAGIC 链 | `chain_mode=facts`，用 artifacts 的 topic_chains.json（257 实体，236 有 lifecycle，5267 条 facts）与 step3_enriched_chains.json（18 条链，75 步） |
| MAGIC 其余超参 | `eval_egolife.py` 默认：max_rounds 5、max_errors 5、episodic/semantic/visual top-k 3/10/3、bm25 0.3、damping 0.85、chain_min_hits 1、chain_max_topics 3、chain_max_events 1、chain_topic_sim 0.7、chain_event_sim 0.7、MAX_QA_FRAMES 64、frames_per_clip 16 |
| MAGIC 计算设备 | 不给 GPU：`CUDA_VISIBLE_DEVICES=` 空，torch 在 CPU 上做相似度与 PPR |
| MAGIC 内存 | `systemd-run --user --scope -p MemoryMax=1T` |
| MAGIC 线程 | 启动前设 `OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 NUMEXPR_NUM_THREADS=10 VECLIB_MAXIMUM_THREADS=10`（在 import numpy/torch 之前） |
| MAGIC 失败 | 与 harness 同规则：每题最多 3 次，错误落盘，耗尽写 `skipped.json` 跳过整题；非可重试错误中止整个 run |
| MAGIC 帧 | 回答阶段它用 decord 从 `video_path` 抽帧（每片段最多 16 帧，总 64 帧）；`video_path` 指向 MM-Lifelong week 同名文件 |
| 转换索引的文本嵌入 | 用 8014 对原文直接逐行计算（不做 normalize_text）：episodes.text、entities.name、semantic_triples.text、topic_facts.fact、event_steps.description、visual_clips.text；object_states 空表写空 npy；MAGIC 图的节点文本向量用同一批向量 |
| 指标 | 总分 = 所有题 judge 分之和，同时给均分；准确率 = judge 分等于 5 的比例；分组用 EgoLife 的 `type`；skip 的题单独列出不计入分母 |
| 视频核对 | 只用 MM-Lifelong week 文件：ffprobe 实际时长对 media-map `duration_s`、文件名时钟与时间轴的一致性、每天抽帧看图；不再与 EgoLife 视频副本比对 |
| 输出 | `training_free/RESULTS.md`（只有三列对比数字）、`training_free/RESULTS_DETAILS.md`（背景、运行统计） |

## 4. 步骤

### 4.1 环境与服务核对（已完成）

1. 两个 venv 已按第 1 节第 5 条建好。
2. `docker ps`、`curl 127.0.0.1:8003/v1/models`、`curl 127.0.0.1:8014/v1/models` 已核对：8003 返回 `Qwen/Qwen3.8-27B-FP8`（max_model_len 314573，容器 `trace2skill-turn12-proxy`），8014 返回 `qwen3-vl-embedding-8b`（root `Qwen/Qwen3-VL-Embedding-8B`，容器 `trace2skill-turn12-embedding`，启动参数含 `--allowed-local-media-path /workspace/video_sql_v2/data/pilot/clips`，该路径不存在，导致所有图像/视频嵌入请求被拒）。
3. `src/harness.py`、`src/val.py`、`src/normalize.py`、`third_party/trace2skill/tracing_bridge.py`、`skills/initial/` 已复制到 `training_free/`，import 路径改成 `training_free.*`，已提交。

### 4.2 EgoLife 数据：核对、出题、过滤库（`prepare_egolife.py`）

1. QA 用本地 `EgoLifeQA/EgoLifeQA_A1_JAKE.json`（500 题，字段 ID、query_time{date,time}、type、question、choice_a–d、answer、target_time、keywords、reason 等，已核对与预期一致；题型 EventRecall 126、EntityLog 125、RelationMap 125、TaskMaster 63、HabitInsight 61）。原始 QA 复制一份到 `runs/egolife-week-shared/egolife_raw/`。
2. 视频一致性核对（只对 MM-Lifelong week）：
   - `runtime/week/media-map.json` 的 6266 个 sources 就是 EgoLife A1_JAKE 同名 30 秒片段（20 fps），按天顺序拼接，各天起点 DAY1 0、DAY2 24725.47、DAY3 50195.12、DAY4 77337.42、DAY5 105826.63、DAY6 134069.62、DAY7 166034.65，总长 187175.02 秒；天内有 26 处录制空档（时钟前进、时间轴不前进）。这些观察与逐片段的"文件名时钟 → 时间轴"表写进 `verification/report.json`。
   - 抽样 ffprobe 实际时长对 media-map `duration_s`（已抽 8 个一致，执行时对全部 6266 个跑一遍并记录不一致项）。
   - 每天至少取 3 个时刻（首个片段起点附近、中间、末尾）用 ffmpeg 抽帧保存 jpg 并人工看图，确认画面里叠印的 DAY/时刻与文件名时钟一致。
   - 时长不一致、叠印时刻与文件名不符：停下问用户。
3. 出题写 `questions.json`，每题保留原始 ID、`type`、原题干、四个选项、正确字母、原始 query_time，另加 `question`（第 3 节格式）、`answer`（正确选项文本）、`cutoff_s`（第 3 节映射）、`until_time`（DHHMMSSFF 整数，给 MAGIC）。答题 agent 只看到 `question`；reference 只进 judge。
4. 对每个不同的 `cutoff_s`，从 `magic_index/index.duckdb` 只读复制出 `magic_index/cutoffs/<cutoff_s>/index.duckdb`（`ATTACH … (READ_ONLY)` + `CREATE TABLE … AS SELECT … WHERE …`，参照 `src/prepare_runtime.py` 的写法），过滤条件见第 3 节；`graph_edges`、`topic_fact_evidence`、`event_step_evidence` 只保留两端都还存在的行。同时写截断的 `cutoffs/<cutoff_s>/media-map.json`。不生成 npy。
5. 抽查若干截止时刻的库：行数随截止时刻单调增加、没有 `start_s` 超过截止的行、media-map 末尾被截断。

### 4.3 我们方法：harness 改动与评测入口

1. `training_free/harness.py` 唯一功能改动：`HarnessContext.__post_init__` 里对每个 `SEARCH_TABLES` 表从（过滤后的）duckdb 读出全部 `row_id`，对共享 npy 的 `row_ids` 做成布尔掩码；`semantic_search` 里排序前把掩码外的分数置为 `-inf`（或只在允许位置上 argsort），其余逻辑不变。`visual_clips` 表同样处理。
2. `training_free/run_training_free.py`：以 `src/trace2skill_baseline.py` 的 `RecordedResponsesModel`、`make_harness`、`rollout`、`score_rollout`、`evaluate_month`、`MainAgentSlots`、`RunMetrics` 为蓝本改写：
   - 索引只有一种：database `magic_index/cutoffs/<cutoff_s>/index.duckdb`、embedding_index `magic_index/embedding_index`、media_map `cutoffs/<cutoff_s>/media-map.json`，run 目录 `runs/training-free-egolife-week-magic-index`。
   - `MAIN_AGENT_CONCURRENCY = 10`，每题 1 条 rollout，目录 `<run>/qa/question-<ID>/rollout-00/`。
   - `wandb.init(entity="3dsmile", project="video-sql-v2", id=RUN.name, name=RUN.name, resume="allow", dir=RUN)`，`init_weave()`，`RunMetrics(wandb_run, "egolife", RUN)`；`wandb-run.json` 记 id 与 url。
   - `skills/initial` 复制到 run 目录并逐字节核对（沿用现有做法）。
   - 每题结果 `result.json` 字段与 `score_rollout` 一致（score、success、cost_usd、trace_length、replacement_metrics、retrieval_qa_metrics、trace_files），另带 `type`、`cutoff_s`。全部完成后写 `summary.json`（完成数、skip 列表、总分、均分、准确率、按 type 分组）。
   - 可恢复：已有 `answer.json`/`result.json`/`skipped.json` 直接复用，沿用现有 attempt 归档逻辑。
3. 提交，按 `REVIEW.md` 由新 subagent 审 diff，修掉 plain-yes 项，再提交。
4. 启动：`.venv_codex_training_free/bin/python -u -m training_free.run_training_free --index own`，stdout/stderr 写 `runs/training-free-egolife-week/formal-01.log`；magic-index 同理写到其 run 目录。
5. 运行中与结束后核对 trace（不是只看数量）：随机打开若干题的 answer segment，确认 semantic_search 命中行时间都不超过该题截止、run_sql 结果时间不超过截止、clip_qa 采样区间在截断 media-map 内、judge 输入是原题干/reference/candidate、W&B 有新数据点。

### 4.4 MAGIC-video：克隆、阅读、确认（已完成阅读，部分待定）

1. 克隆、记录 commit、移走 `.git`、提交：已完成。
2. 已完整阅读：README、`eval/eval_egolife.py`、`eval/_common.py`、`src/worldmm/memory/memory.py`、`unified/{memory,graph,nodes,retrieval}.py`、`visual/memory.py`、`episodic/memory.py`、`semantic/memory.py`（加载部分）、`memory/utils.py`、`llm/*`、`templates/{memory_reasoning_unified_egolife,memory_reasoning_unified_mmlifelong,qa_egolife,qa_egolife_chain,qa_mmlifelong,qa_mmlifelong_chain}.py`、`preprocess/build_unified_graph.py`、`preprocess/egolife/extract_topic_chains.py`、`extract_event_chains.py`，以及 artifacts 的 README 与全部 EgoLife json 的结构。
3. 用户已定的项见第 3 节；仍待定的项见第 5 节。

### 4.5 MAGIC-video：改造与运行

1. `build_magic_graph.py`：把 artifacts 的 captions 复制/软链到它期望的 `data/EgoLife/EgoLifeCap/A1_JAKE/`、memory 文件到 `output/metadata/…`（或直接改路径参数），30sec caption 的 `video_path`（`data/EgoLife/A1_JAKE/DAYx/<name>.mp4`）改指向 MM-Lifelong week 同名文件；用 `MultimodalGraph` 的 `add_episode_nodes`/`add_visual_nodes`/`add_entity_nodes`/`add_semantic_nodes` 建图；`compute_text_embeddings` 改为调 8014（4096 维，代码从第一个节点探测维度，不写死 2560）；视觉向量按第 5 节决定；保存 `runs/egolife-week-shared/magic_graph/graph.pkl`。
2. embedder：`EmbeddingModel.encode_text` 与 `encode_vis_query` 改为 HTTP 调 `127.0.0.1:8014/v1/embeddings`，模型 `qwen3-vl-embedding-8b`，校验返回 4096 维有限向量；删除进程内模型加载。
3. LLM：`OpenRouterModel` 改为 OpenAI 兼容客户端指向 `http://127.0.0.1:8003/v1`，模型 `Qwen/Qwen3.8-27B-FP8`，不传 reasoning 与采样参数；`_get_time_extract_llm` 同样；记录每次响应的 usage 给 `RunMetrics.record_response`（chat completions 的 prompt/completion tokens 映射为 input/output tokens，`response_cost` 对该模型名返回 0）。
4. `run_magic.py`：`WorldMemory(retrieval_backend="unified_graph", graph_path=…, chain_mode="facts", topic_chain_facts_path=…, event_chain_path=…)`，`qa_template_name="qa_mmlifelong"`，`reasoning_version="egolife"`，`answer(query=question, choices=None, until_time=<DHHMMSSFF>)`；`load_episodic_captions`/`load_semantic_triples`/`load_visual_clips` 与 `eval_egolife.py` 一致（visual embeddings 按第 5 节）。并发 10 个题（每 worker 一个 `WorldMemory`，与原代码相同）；每题目录 `runs/magic-video-egolife-week/qa/question-<ID>/`，保存 round_history、检索结果、prompt、原始响应、最终答案；失败规则与 harness 相同。
5. judge：用 `training_free/val.py` 的 `judge_answer`，harness 与我们方法完全相同的构造（同模型、同 reasoning、同 prompt），run_dir 指向该题目录。
6. 启动前在 Python 入口顶部设置线程环境变量与 `CUDA_VISIBLE_DEVICES=`；用 `systemd-run --user --scope -p MemoryMax=1T .venv_codex_magic/bin/python -u -m training_free.run_magic` 启动（`PYTHONPATH` 含 `training_free/MAGIC-video/src` 与 `src/EgoRAG`），启动后用 `systemctl --user show` 核对 scope 的 MemoryMax 生效；日志写 `runs/magic-video-egolife-week/formal-01.log`。
7. 提交并按 `REVIEW.md` 审 diff 后再启动。运行中核对：embedding HTTP 调用确实返回向量、检索结果时间不超过提问时刻、LLM 响应正常、W&B 有新数据点、线程数（`ls /proc/<pid>/task | wc -l`）没有失控。

### 4.6 对比与报告（`compare.py`）

1. 读两个 run 的 `result.json`/`skipped.json`，按 `questions.json` 对齐题目。
2. `training_free/RESULTS.md`：只放对比表：overall 与按 `type` 的题数、总分、均分、准确率（分等于 5 的比例），两个 run 各一列，外加 skip 题目 ID 列表。不放运行统计。
3. `training_free/RESULTS_DETAILS.md`：两个服务的实际参数（`/v1/models` 与 `docker ps` 所见）、MAGIC 的 commit 号与改动清单、artifacts 快照号、EgoLife 本地快照号与 QA 字段、视频核对结果（每天时长、映射公式、空档列表、图片路径）、题目文本格式、截止过滤规则与截止时刻数量、索引转换规则、W&B 链接、两个 run 的运行统计（耗时、吞吐、token）、skip 原因、所有与本计划的偏差。
4. 提交。

### 4.7 收尾

- 用本计划内容整体覆盖 `baselines_plan.md`（已做，并随每次决定更新）。
- 不关闭任何 vLLM 服务。

## 5. 仍需向用户确认的点（全部与索引转换有关）

- 旧 shards 的文本向量是对小写化（normalize_text）后的原文算的（trace 里保存的输入全是小写）；而用户已定转换索引的文本向量"不做 normalize_text"。MAGIC 图 30 秒节点复用旧向量与这条决定的关系待用户裁定（复用即等于用了小写化文本的向量）。
- MAGIC artifacts 转成我们格式时 semantic_triples 的转法（consolidation 首次出现 / extraction 文件 / 全量展开），用户表示无偏好。
- 其余 MAGIC → 我们格式的字段映射尚未确认：entities 来源（openie ner_results）与 first/last_seen、topic_chains/event_chains 的 level、scope_id、chain_type 取值、graph_edges 的构造（按 MAGIC 建图规则生成 co_clip/contains/temporal_next/mentioned_in/appeared_in/has_property）、episodes 的 10min 层保留为 granularity 字符串 `10min`、object_states 空表、转换索引 visual_clips 向量是否也复用旧 shards 的视频向量。

## 6. MAGIC-video 已知 bug（保留自原计划）

1. 它使用进程内 embedder，并发下会死锁。解决：改用共享 vLLM embedding 服务，通过 HTTP 调用。
2. 并发时 egolife 会开很多线程，每个并发里 OMP/MKL/BLAS/NumExpr 又各自用满所有核心，线程数暴涨会把机器搞崩（有时还会系统 OOM，连带 NVIDIA 驱动崩掉导致 NCCL 不可用）。解决：用线程环境变量把 OMP/MKL/OPENBLAS/NUMEXPR/VECLIB 各限制到 10 线程，并用 systemd 限制最大内存（本次 1T）。

## 7. 验证方式

- 数据：`verification/report.json` 里每个源文件 ffprobe 时长与 media-map 一致、逐片段的时钟→时间轴表、空档列表、每题 cutoff_s 与 until_time，并且人工看过抽帧 jpg。
- 过滤库：抽查的截止时刻库里没有超过截止的行，media-map 末尾被截断；转换库同样抽查。
- 我们方法：打开真实 trace 核对 semantic_search/run_sql/clip_qa 的时间边界与 judge 输入；`summary.json` 的题数等于 `questions.json` 题数减 skip 数；W&B run 页面有 `egolife/*` 与 `client/*` 曲线。
- MAGIC：每题目录里有 round_history、检索结果、prompt、原始响应；embedding 请求日志有 4096 维返回；检索结果时间不超过提问时刻；scope 的 MemoryMax 已核对；W&B 同上。
- 报告：`RESULTS.md` 的每个数字能从两个 run 目录的 `result.json` 重新算出。
