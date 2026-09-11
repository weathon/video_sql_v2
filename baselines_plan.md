# Baselines 计划（EgoLife week：MAGIC-video 与 training-free harness，两者都只用 MAGIC-video artifacts）

本文件是当前最新规格，整体替换旧版；所有约束、步骤、已定参数和待确认项都在这里。

## 0. 背景与目标

- 同一目录、同一分支（`codex/trace2skill-baseline`）上另有一个 agent 在跑 Trace2Skill rollout（入口 `src/trace2skill_baseline.py`，run 目录 `runs/qwen38-fp8-context128k-day-week-tree-month-val-seed0`）。不碰 `src/`、`skills/`、`third_party/`、它的 run 目录，不改动、不重启任何现有 vLLM 容器。
- 索引只有一个来源：MAGIC-video 发布的 magic-video-artifacts（EgoLife A1_JAKE）。不使用我们自己的 week 索引。
- 目标一：MAGIC-video 用它自己从 artifacts 建出的图，在 EgoLifeQA 上跑 baseline。
- 目标二：把同一份图逐字标成 SQL 表（schema 与 MAGIC 图完全一样，不转换内容），用我们 harness 的 training-free 形态（静态 `skills/initial`，不做 reflection/evolution）跑一遍。
- 目标三：两个 run 按总分、均分、准确率（judge 满分比例）做 overall 与按 EgoLife 题型的对比，写成一份简报和一份详细背景。
- 题目用 EgoLifeQA（A1_JAKE，500 题），视频一律用 MM-Lifelong week 的源文件。LLM、embedding 服务、judge 两个 run 完全相同。

## 1. 边界与纪律

1. 计划阶段只读了当前目录，没有访问 git 历史、目录以外路径或本机已有的 MAGIC 副本。这一点在执行阶段继续成立：不运行任何读历史的 git 命令（`log`、`show`、`diff <commit>` 等），不查找、不打开本机已有的 MAGIC-video 副本。
2. 目录以外的资源，用户已批准如下；未列出的一律不碰：
   - 只读 `/raid/wg25r/video_sql_v2/data/runtime/week/media-map.json`（week 时间轴：源文件、时长、timeline_start_s/end_s）。
   - 只读 `/raid/huggingface` 下 MM-Lifelong 快照中的 week 源视频（media-map 列出的 6266 个文件，1408×1408、20 fps、h264）。
   - EgoLife 不下载，只读本地 `/raid/huggingface/datasets--lmms-lab--EgoLife/snapshots/143fb319be7aa5ae210c936bf4f0f3a86092afb0/EgoLifeQA/EgoLifeQA_A1_JAKE.json`。本地 A1_JAKE 视频副本不完整，不使用。
   - MAGIC artifacts 只读本地 `/raid/huggingface/hub/datasets--jiazhengli7--magic-video-artifacts/snapshots/8573f317684ce55c0999f86386666d3d3fe76ae3/egolife/A1_JAKE/`。
   - 只读 `/raid/wg25r/video_sql_v2/`（用到 `data/pilot/stage2/embedding_shards/`、`data/pilot/stage2/run/chunks.jsonl`、`data/pilot/stage2/run/traces/embeddings/`、`INFRA.txt`）。
   - 访问 `127.0.0.1:8003`（Responses/Chat 代理）和 `127.0.0.1:8014`（embedding，只接纯文本），`docker ps`/`docker inspect` 只读核对。
   - `git clone` GitHub 上的 MAGIC-video。
   - 启动一个新的 vLLM embedding 容器用于视频嵌入（第 3 节"验证/视频 embedder"行）。
   - `systemd-run --user --scope` 给 MAGIC 进程设内存上限。
   - 任何删除都用 `mv` 到 `/raid/wg25r/trash/` 代替 `rm`。
3. 服务不在运行、数据缺失、映射对不上、字段与预期不符：停下来问用户，不换模型、不换索引、不造替代数据、不跳过阶段。
4. git：在当前分支直接提交，每次编辑后提交；只 `git add` `training_free/**`、`baselines_plan.md`、以及本计划新建的 markdown。不 add 别人的未跟踪/已修改文件（`AGENTS.md`、`src/analysis.py`、`src/check_status.py`、`notes/INCIDENT_*` 等）。不用 `git stash`。
5. venv：两个，全部新建（当前目录下没有 `.venv_codex_*`），都被 `.gitignore` 的 `.venv_codex_*/` 忽略：
   - `.venv_codex_training_free`：`uv venv .venv_codex_training_free`，然后 `UV_PROJECT_ENVIRONMENT=.venv_codex_training_free uv sync`（按 `uv.lock`）。
   - `.venv_codex_magic`：`uv venv .venv_codex_magic`，然后在 `training_free/MAGIC-video/` 下 `UV_PROJECT_ENVIRONMENT=<绝对路径>/.venv_codex_magic uv sync`（按它的 `uv.lock`：torch 2.7.1+cu128、hipporag/egorag editable 等），再 `uv pip install` 补 `wandb`、`weave`、`duckdb`、`numpy`、`httpx`、`openai`、`openai-agents`、`sqlglot>=30.18.0`（judge 与 harness 走 `training_free/val.py`、`training_free/harness.py`）。
6. 代码风格按仓库规则：研究代码，不加防御、不加 fallback、不设 token 上限、不降 reasoning。大运行前按 `REVIEW.md` 由新的 review subagent 审 `git diff`，主 agent 不读 `REVIEW.md`。
7. 长任务用 harness 前台方式直接运行（不 `nohup … &`），日志写到 run 目录；用定时 Monitor 低频查看，不 sleep 轮询。每个阶段完成后先看真实产物与 trace，再报告。

## 2. 目录布局

```
training_free/
  harness.py          # 复制 src/harness.py；改动只有两处：截止掩码（4.3.1）和 SEARCH_TABLES/KEY_COLUMNS/SCHEMA 换成 MAGIC 图的 SQL 表（4.3.2）
  val.py              # 复制 src/val.py（judge_answer、retryable、write_error、milestone）
  normalize.py        # 复制 src/normalize.py
  tracing_bridge.py   # 复制 third_party/trace2skill/tracing_bridge.py（RunMetrics、init_weave、response_cost）
  skills/initial/     # 复制 skills/initial/ 七个 md，只替换表名、字段名和 SQL 示例，其余文字一字不改；运行时逐字节核对未变
  prepare_egolife.py  # 视频核对报告、出题、截断 media-map、按截止过滤的 SQL 库
  verify_embeddings.py   # 旧 shards 文本向量抽样核对（已做过一次，结果见第 6 节；重做时同样方法）
  embed_week_videos.py   # 用新 embedder 对 6266 个 week 源文件算视频向量（并发 8，逐文件 npz，可恢复）
  convert_magic_index.py # pretext（预热 8014 文本向量缓存）、graph（建 MAGIC 图 graph.pkl）、sql（图 → SQL 表 + npy）
  run_training_free.py# 我们方法的评测入口
  MAGIC-video/        # 新鲜克隆，commit 0f2273f7f07360cc58588399553b5a6ea758b084，内嵌 .git 移到 trash，由本仓库跟踪
  run_magic.py        # MAGIC 的评测入口
  compare.py          # 汇总两个 run，写 RESULTS.md 与 results_stats.json
  RESULTS.md          # 简报：只有对比数字
  RESULTS_DETAILS.md  # 详细背景（手写）：设置、commit、核对结果、W&B 链接、运行统计、偏差说明
runs/egolife-week-shared/          # 两个 run 共用的产物（runs/ 已被 gitignore）
  egolife_raw/                     # QA 原始 json 副本
  verification/                    # report.json（时长、映射、空档、抽帧）、frames/、embedding_check.json
  questions.json                   # 最终题目（开放式）
  cutoffs/<cutoff_key>/media-map.json         # 截断到截止时刻的 media-map，cutoff_key = f"{cutoff_s:.3f}"
  magic_index/artifacts/captions/  # 30sec caption 的 video_path 改指 MM-Lifelong 文件、去掉 2 条空 caption 后的副本；其余三层原样副本
  magic_index/text_vectors/        # 8014 文本向量缓存（按 normalize_text 后文本的 sha256 命名）
  magic_index/visual_vectors/      # 6266 个源文件的视频向量（新 embedder）
  magic_index/index.duckdb         # MAGIC 图标成的 SQL 全量库
  magic_index/embedding_index/     # 可搜索表的 row_ids/vectors npy
  magic_index/cutoffs/<cutoff_key>/index.duckdb  # 按截止过滤的库
  magic_graph/graph.pkl            # MAGIC 自己用的图（时间 DHHMMSSFF，节点带文本/视觉向量）
runs/training-free-egolife-week-magic-index/ # 我们方法 + MAGIC 图的 SQL
runs/magic-video-egolife-week/               # MAGIC + MAGIC 图
```

两个 run 目录名同时是 W&B run 名。

## 3. 已确定的参数

| 项 | 值 |
| --- | --- |
| 题目 | EgoLifeQA A1_JAKE 全部 500 题（EventRecall 126、EntityLog 125、RelationMap 125、TaskMaster 63、HabitInsight 61），不抽样 |
| 开放式改写 | 规则去选项：题干原样，删除四个选项，正确选项文本作 reference answer，不经 LLM |
| 题目文本格式 | `Asked on DAY3 at 11:22:02 (timeline 52340.5 s). <原题干>`：天 + HH:MM:SS（帧号丢弃）+ week 时间轴秒数（一位小数） |
| query_time → cutoff_s | week 时间轴由 EgoLife 同名 30 秒片段按录制顺序拼接，天内 26 处录制空档被去掉，天与天之间另有约 15 秒间隔。映射：找文件名时钟范围覆盖提问时刻的片段，cutoff = 片段 timeline_start_s + (提问时刻 − 片段文件名时钟)；提问时刻在空档里的题（ID 34、289、341、346）用前一个片段的 timeline_end_s。帧号按 20 fps 换算。共 432 个不同截止 |
| until_time（给 MAGIC） | 整数 DHHMMSSFF，即 `int(day + query_time.time)` |
| 硬限制实现 | 每个不同截止时刻一份过滤后的 duckdb 与截断 media-map；npy 不分截止，semantic_search 按过滤库中存在的主键做掩码 |
| clip_qa | media-map 只保留截止前的源视频，最后一个的 `timeline_end_s` 截到截止时刻 |
| harness 设置 | 沿用 `src/trace2skill_baseline.py`：模型 `Qwen/Qwen3.8-27B-FP8`、base_url `http://127.0.0.1:8003/v1`、reasoning xhigh + summary auto、clip_qa_video 开、650 图额度、prompt_cache_key、attempts 3、耗尽后 skip 整题、replace_tool 开、judge 同模型 0–5 分；judge 输入是原题干（不带时刻前缀）、reference、candidate |
| embedding 服务 | `http://127.0.0.1:8014/v1`，served name `qwen3-vl-embedding-8b`，4096 维；它拒绝一切图像/视频请求（`--allowed-local-media-path` 指向不存在的目录），只用于文本 |
| 文本向量 | 一律 `normalize_text` 后嵌入（与查询侧一致）。30 秒 caption 的向量复用 `video_sql_v2/data/pilot/stage2/embedding_shards/text/episodes` 里官方 caption 版 week 行（`chunks.jsonl` 给出 row_id ↔ 源文件 ↔ official_text，trace 里的输入必须等于 `normalize_text(official_text)`，否则报错）；其余文本（3min/10min/1h caption、NER 实体名、去重后的三元组文本、topic facts、event step 描述）用 8014 新算并按文本缓存 |
| 视频向量 | 旧 shards 的视频向量核对不通过（第 6 节），不用。用新 embedder 对 6266 个 MM-Lifelong 源文件逐个整段嵌入（chat 格式 `video_url` 指向 `file:///hf_cache/...`），并发 8，逐文件 npz，可恢复 |
| 验证/视频 embedder | 新容器：镜像 `vllm/vllm-openai:qwen38-flash-next`，`Qwen/Qwen3-VL-Embedding-8B`，served name `qwen3-vl-embedding-8b`，`--runner pooling`，`--gpu-memory-utilization 0.45`，`--allowed-local-media-path /hf_cache`，其余同 `INFRA.txt` 的 embedding 配方，挂载 `/raid/huggingface:/hf_cache`，宿主端口 8015，不加 `--rm`；GPU 用启动时空闲的那张（上次是 GPU 7，重启前重新 `nvidia-smi` 核对）。用完后停止，是否删除等用户指示 |
| 30 秒空 caption | artifacts 的 30sec caption 有 2 条 text 为空（DAY1_A1_JAKE_19100000、DAY6_A1_JAKE_16430000），从图和 SQL 里去掉（对应 episode 与 visual_clip 节点都不建） |
| rollout | 每题 1 条 |
| 并发 | 两个 run 都是 10 个题同时进行（我们方法用 `MainAgentSlots` 上限 10；MAGIC 用 10 个线程，每线程一个 `WorldMemory`） |
| W&B / Weave | entity `3dsmile`，project `video-sql-v2`，run id 与 name 等于 run 目录名，`resume="allow"`，Weave 开启；两边都用 `RunMetrics(wandb_run, "egolife", RUN)`；MAGIC 用不到的 replacement 指标记 0，cost 记 0 |
| MAGIC 克隆 | `training_free/MAGIC-video/`，main 分支 commit `0f2273f7f07360cc58588399553b5a6ea758b084`，内嵌 `.git` mv 到 trash 后由本仓库跟踪 |
| MAGIC 图 | `MultimodalGraph`：`add_episode_nodes`（四层 caption 副本）、`add_visual_nodes`（30sec caption，视觉向量 = 新算视频向量）、`add_entity_nodes`（openie_results，person=A1_JAKE）、`add_semantic_nodes`（consolidation 623 个快照全部入图，约 133 万节点）。节点文本向量按上一行规则填好，包括语义节点（MAGIC 原本查询时现算，数值相同）和 visual 节点（填 30 秒 caption 向量，避免每题重算；build_igraph 会把它置零）。保存 `magic_graph/graph.pkl` |
| MAGIC 运行 | `retrieval_backend="unified_graph"`，`chain_mode="facts"`（artifacts 的 topic_chains.json + step3_enriched_chains.json），`eval_egolife.py` 默认超参（max_rounds 5、max_errors 5、top-k 3/10/3、bm25 0.3、damping 0.85、chain_min_hits 1、max_topics 3、max_events 1、topic_sim 0.7、event_sim 0.7、event_min_hits 1、granularities 30sec,3min、MAX_QA_FRAMES 64、16 帧/片段）；提问开放式：`choices=None`，题目文本与我们的 `question` 字段完全相同，推理模板 `memory_reasoning_unified_egolife`，回答模板 `qa_mmlifelong`（chain 模式自动用 `_chain` 版）；`until_time` 传 DHHMMSSFF；每题答题前把 `indexed_time` 与 `unified_memory._indexed_time` 置 0 强制重建子图（MAGIC 的 index 只增不减，否则同一 worker 先答晚题再答早题会用到截止后的图） |
| MAGIC LLM | retriever、respond、以及写死走 OpenRouter 的时间提取模型（`parse_time_reference`）都改为 8003 的 `Qwen/Qwen3.8-27B-FP8` chat completions；不传 reasoning 参数；原代码 respond 的采样参数（temperature 1.0、top_p 0.95、top_k 20、min_p 0、presence_penalty 1.5、repetition_penalty 1.0）全部删掉；图片按 MAGIC 原逻辑转 JPEG data URL；每次响应 usage 记入 `RunMetrics.record_response`，每次请求/响应（图片记尺寸）写 `llm_calls.jsonl` |
| MAGIC embedder | `EmbeddingModel` 换成 HTTP 类：`encode_text`、`encode_vis_query` 都调 8014 文本嵌入（`normalize_text` 后），校验 4096 维有限向量；不装 Qwen3-Embedding-4B / VLM2Vec |
| MAGIC 帧 | 回答阶段按 MAGIC 原逻辑用 decord 从 `video_path`（MM-Lifelong 文件）抽帧 |
| MAGIC 计算设备 | `CUDA_VISIBLE_DEVICES=` 空，torch 在 CPU 上做相似度与 PPR |
| MAGIC 内存/线程 | `systemd-run --user --scope -p MemoryMax=1T`；入口顶部先设 `OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 NUMEXPR_NUM_THREADS=10 VECLIB_MAXIMUM_THREADS=10`（在 import numpy/torch 之前） |
| MAGIC 失败 | 每题最多 3 次尝试，错误落盘，耗尽写 `skipped.json` 跳过整题；答案等于 MAGIC 的兜底串 "Unable to generate answer" 视为失败；非可重试错误让全部 worker 停下并中止整个 run |
| 指标 | 总分 = 所有题 judge 分之和，同时给均分；准确率 = judge 分等于 5 的比例；分组用 EgoLife 的 `type`；skip 的题单独列出不计入分母 |
| 视频核对 | 只用 MM-Lifelong week 文件：全部 6266 个 ffprobe 时长对 media-map、文件名时钟与时间轴一致性（空档列表）、每天 3 个时刻抽帧人工看叠印时刻 |
| 输出 | `training_free/RESULTS.md`（只有两列对比数字）、`training_free/RESULTS_DETAILS.md`（背景、运行统计、所有偏差） |

## 4. MAGIC 图 → SQL 的规则

用户要求：用和 MAGIC 完全一样的 schema，不转换内容，唯一需要转化的是把 graph 标成 SQL。每张表只包含 MAGIC 图节点/边或链文件里本来就有的字段，加上用户批准的两项：时间轴秒列 `start_s`/`end_s`（由 DHHMMSSFF 经片段映射算出，供截止过滤与 clip_qa 用）和去重文本表 `semantic_texts`（只为三元组文本做一次向量）。

- `episode(id, granularity, date, start_time, end_time, start_ts, end_ts, start_s, end_s, text, video_path)`：四层 caption 全部入表；`id` 是 MAGIC 节点 id `ep_<start_ts>_<end_ts>_<gran>`；30sec 的 `video_path` 指 MM-Lifelong 文件，其余层为 NULL（MAGIC 元数据里就是 None）。
- `entity(id, name, normalized_name, frequency, start_ts, end_ts, start_s, end_s)`：MAGIC 实体节点自带 `frequency`、`start_ts`（首次提及）、`end_ts`（末次提及）。
- `semantic(id, snapshot_ts, snapshot_s, index_in_snapshot, subject, predicate, object, text, text_id)`：623 个快照的全部三元组按行入表（约 133 万行），`id` 为 `sem_<ts>_<idx>`。
- `semantic_texts(text_id, text, subject, predicate, object, first_snapshot_ts, first_snapshot_s)`：去重文本 6329 行，是 semantic 的可搜索表；`first_*` 只用于 semantic_search 报命中时间（永远 ≤ 截止）。
- `visual_clip(id, date, start_time, end_time, start_ts, end_ts, start_s, end_s, video_path, text)`：来自 30sec caption，`id` 为 `vis_<date>_<start_time>`。
- `graph_edges(src_id, dst_id, edge_type, weight)`：MAGIC 建图规则生成的 co_clip、contains、temporal_next、mentioned_in、appeared_in、has_property。
- `topic_chains(entity, has_lifecycle)`、`topic_facts(fact_id, entity, time, time_ts, time_s, fact)`：来自 topic_chains.json；`time` 原样保留，`time_ts`/`time_s` 用 MAGIC 同样的正则 `DAY(\d+)\s+(\d{1,2}):(\d{2})` 取开头的 DAYx HH:MM，取不到的 13 条（如 'Various'、'GENERAL'、'DAY6 13:02-13:22'）置 NULL。
- `event_chains(chain_id, name, key_entities)`、`event_steps(step_id, chain_id, name, step_index, day, start_time, end_time, start_ts, end_ts, start_s, end_s, description)`：来自 step3_enriched_chains.json（18 条链、75 步，`key_entities` 存 JSON 字符串）。
- 没有 object_states、topic_fact_evidence、event_step_evidence（MAGIC 没有）。
- 时钟 → 时间轴秒的映射与题目 cutoff 相同：落在片段内取片段 timeline_start + 偏移，落在空档取前一片段 timeline_end；早于当天第一个片段的时刻直接报错。
- 可搜索表（SEARCH_TABLES → 嵌入列）：`episode.text`、`entity.name`、`semantic_texts.text`、`visual_clip.text`（向量用视频向量）、`topic_facts.fact`、`event_steps.description`；主键列：`episode.id`、`entity.id`、`semantic_texts.text_id`、`visual_clip.id`、`topic_facts.fact_id`、`event_steps.step_id`。
- 截止库过滤：`episode`/`visual_clip`/`event_steps` 保留 `start_s <= c`；`semantic` 保留全部 `snapshot_s <= c`；`semantic_texts` 保留 `first_snapshot_s <= c`；`topic_facts` 保留 `time_s <= c OR time_s IS NULL`；`topic_chains` 保留还有 fact 的实体，`event_chains` 保留还有 step 的链；`graph_edges` 只保留两端都在的边；`entity` 保留 `start_s <= c`，其 `end_ts`/`end_s`/`frequency` 怎么处理见第 7 节待定项。

## 5. 步骤

### 5.1 环境与服务核对

1. 建两个 venv（第 1 节第 5 条）。
2. `docker ps`、`curl 127.0.0.1:8003/v1/models`、`curl 127.0.0.1:8014/v1/models` 核对：8003 返回 `Qwen/Qwen3.8-27B-FP8`（容器 `trace2skill-turn12-proxy`），8014 返回 `qwen3-vl-embedding-8b`（容器 `trace2skill-turn12-embedding`）。不在跑就停下报告。
3. 复制 `src/harness.py`、`src/val.py`、`src/normalize.py`、`third_party/trace2skill/tracing_bridge.py`、`skills/initial/` 到 `training_free/`，import 改成 `training_free.*`，新 venv 里 import 通过，七个 skill 逐字节一致。提交。
4. 克隆 MAGIC-video 到 `training_free/MAGIC-video/`，`git -C … rev-parse HEAD` 记录 commit（预期 `0f2273f7…`），内嵌 `.git` mv 到 trash，提交。

### 5.2 EgoLife 数据（`prepare_egolife.py`）

1. `verify` 阶段：读 media-map，写 `verification/report.json`（各天起点、6266 个文件 ffprobe 时长对比、26 处空档、每天 3 个时刻的抽帧 jpg 与期望时钟），人工看抽帧。上次结果：时长全部一致，抽帧叠印时刻与文件名时钟加偏移一致。
2. `questions` 阶段：复制原始 QA 到 `egolife_raw/`；写 `questions.json`，每题保留 ID、type、原题干、四个选项、正确字母、query_time、target_time，另加 `question`（第 3 节格式）、`answer`（正确选项文本）、`cutoff_s`、`cutoff_key`、`cutoff_mapping`（inside_clip / recording_gap_after_clip）、`cutoff_clip`、`until_time`。写 432 个 `cutoffs/<cutoff_key>/media-map.json`。
3. `cutoff-dbs` 阶段（在 5.4 之后）：从 `magic_index/index.duckdb` 只读 ATTACH，按第 4 节过滤规则生成 `magic_index/cutoffs/<cutoff_key>/index.duckdb`。抽查：行数随截止单调增加，没有 `start_s` 超过截止的行，media-map 末尾被截断。

### 5.3 向量

1. 启动视频 embedder 容器（第 3 节配方），150 秒后查 `/v1/models`。
2. `embed_week_videos.py`：6266 个源文件的视频向量写到 `magic_index/visual_vectors/`（上次跑到 1592/6266 后随回滚停止，产物已移到 trash，需重跑）。
3. `convert_magic_index.py pretext`：用 8014 预热文本向量缓存（上次 15651 条不同文本，约 15 分钟）。
4. 旧 shards 30 秒 caption 向量在 graph 阶段直接复用，trace 文本与 `normalize_text(official_text)` 不一致就报错。

### 5.4 建图与转 SQL（`convert_magic_index.py`，在 `.venv_codex_magic` 里跑，`PYTHONPATH=training_free/MAGIC-video/src:training_free/MAGIC-video/src/EgoRAG`）

1. `graph`：写 caption 副本（30sec 改 video_path、去掉 2 条空 caption），建图，填向量，保存 `magic_graph/graph.pkl`；打印 `graph.stats()`。
2. `sql`：按第 4 节写 `magic_index/index.duckdb` 与 `embedding_index/*.npy`（可搜索表的 row_ids 与 vectors 一一对应）。
3. 核对：各表行数与图节点数一致（episode = 四层 caption 数 − 2、visual_clip = 6221、entity = 图实体数、semantic = 图语义节点数、semantic_texts = 6329、topic_facts = 5267、event_steps = 75）；抽几行看 start_s/end_s 与 DHHMMSSFF 对应；npy 形状 (行数, 4096)。

### 5.5 我们方法（`harness.py` 改动与 `run_training_free.py`）

1. `harness.py` 改动一：`HarnessContext.__post_init__` 用过滤库的主键对每张可搜索表的 npy `row_ids` 做布尔掩码（库里行数必须等于掩码命中数，否则报错）；`semantic_search` 排序前把掩码外分数置 `-inf` 并只保留有限分数的位置；命中行按 `KEY_COLUMNS` 查询；`semantic_texts` 的命中时间用 `first_snapshot_s`，`topic_facts` 用 `time_s`，其余用 `start_s`/`end_s`。
2. `harness.py` 改动二：`SEARCH_TABLES`、`KEY_COLUMNS`、`SCHEMA` 换成第 4 节的表；`SCHEMA` 只列表名与列名，加两句说明：`*_ts`/`date`/`start_time`/`end_time`/`day`/`time` 是墙上时钟 DHHMMSSFF，`*_s` 是 week 时间轴秒（clip_qa 用）。不写别的解释。
3. `skills/initial`：`sql-schema.md` 的表列表和 `semantic-search.md` 的可搜索表句子替换成新表名与列名，其余文字一字不改；修改前后 diff 写进 RESULTS_DETAILS.md。
4. `run_training_free.py`：以 `src/trace2skill_baseline.py` 的 `RecordedResponsesModel`、`make_harness`、`rollout`、`score_rollout`、`MainAgentSlots` 为蓝本；每题 database `magic_index/cutoffs/<cutoff_key>/index.duckdb`、embedding_index `magic_index/embedding_index`、media_map `cutoffs/<cutoff_key>/media-map.json`；目录 `runs/training-free-egolife-week-magic-index/qa/question-<ID>/rollout-00/`；10 个 worker 从题目迭代器取题；`result.json` 与 `score_rollout` 字段一致并带 `type`、`cutoff_s`；全部完成写 `summary.json`（完成数、skip 列表、总分、均分、准确率、按 type）。可恢复。
5. 提交，review subagent 审，修 plain-yes 项，再提交。
6. 先跑 1 题核对 trace，再启动全量：`.venv_codex_training_free/bin/python -u -m training_free.run_training_free`，日志 `runs/training-free-egolife-week-magic-index/formal-01.log`。
7. 运行中与结束后核对 trace：semantic_search 命中行时间不超过截止、run_sql 结果时间不超过截止、clip_qa 采样区间在截断 media-map 内、judge 输入是原题干/reference/candidate、W&B 有新数据点。

### 5.6 MAGIC（`run_magic.py`）

1. 按第 3 节的 MAGIC 各行实现；每题目录 `runs/magic-video-egolife-week/qa/question-<ID>/attempt-N/` 保存 `llm_calls.jsonl`、`embedding_calls.jsonl`、`round_history.json`、`retrieved_items.json`、`answer.json`；题目目录下 `judgment.json`、`result.json` 或 `skipped.json`。judge 用 `training_free/val.py` 的 `judge_answer`，harness 构造与我们方法相同，run_dir 指向题目目录下 `judge/`。
2. 提交，review subagent 审，修 plain-yes 项，再提交。
3. 先跑 1 题核对 trace，再全量：`systemd-run --user --scope -p MemoryMax=1T env PYTHONPATH=… .venv_codex_magic/bin/python -u -m training_free.run_magic`，`systemctl --user show` 核对 MemoryMax，日志 `runs/magic-video-egolife-week/formal-01.log`。
4. 运行中核对：embedding 调用返回 4096 维、检索结果时间不超过提问时刻、LLM 响应正常、W&B 有新数据点、线程数（`ls /proc/<pid>/task | wc -l`）没有失控。

### 5.7 对比与报告

1. `compare.py`：读两个 run 的 `result.json`/`skipped.json`，按 `questions.json` 对齐，写 `training_free/RESULTS.md`（overall 与按 type 的题数、总分、均分、准确率，两列，外加 skip 题目 ID）和 `results_stats.json`。
2. `training_free/RESULTS_DETAILS.md`（手写）：两个服务的实际参数、MAGIC commit 与改动清单、artifacts 快照号、EgoLife 快照号与 QA 字段、视频核对结果、题目文本格式、截止映射与 432 个截止、图 → SQL 规则、embedding 来源与核对结果、W&B 链接、两个 run 的运行统计、skip 原因、所有与本计划的偏差。
3. 提交。

### 5.8 收尾

- 停止视频 embedder 容器（是否删除等用户指示）。不关闭任何现有 vLLM 服务。

## 8. MAGIC-video 已知 bug（保留自原计划）

1. 它使用进程内 embedder，并发下会死锁。解决：改用共享 vLLM embedding 服务，通过 HTTP 调用。
2. 并发时 egolife 会开很多线程，每个并发里 OMP/MKL/BLAS/NumExpr 又各自用满所有核心，线程数暴涨会把机器搞崩（有时还会系统 OOM，连带 NVIDIA 驱动崩掉导致 NCCL 不可用）。解决：用线程环境变量把 OMP/MKL/OPENBLAS/NUMEXPR/VECLIB 各限制到 10 线程，并用 systemd 限制最大内存（本次 1T）。

## 9. 验证方式

- 数据：`verification/report.json` 里每个源文件 ffprobe 时长与 media-map 一致、空档列表、每题 cutoff_s 与 until_time，并且人工看过抽帧 jpg。
- 向量：`embedding_check.json` 的文本核对数字；视频向量 6266 个 npz 全部存在且 norm 为 1。
- 图与 SQL：`graph.stats()` 与各表行数一一对应；抽查的截止库里没有超过截止的行，media-map 末尾被截断。
- 我们方法：打开真实 trace 核对 semantic_search/run_sql/clip_qa 的时间边界与 judge 输入；`summary.json` 的题数等于 500 减 skip 数；W&B run 页面有 `egolife/*` 与 `client/*` 曲线。
- MAGIC：每题目录里有 `llm_calls.jsonl`、`round_history.json`、`retrieved_items.json`；embedding 日志有 4096 维返回；检索结果时间不超过提问时刻；scope 的 MemoryMax 已核对；W&B 同上。
- 报告：`RESULTS.md` 的每个数字能从两个 run 目录的 `result.json` 重新算出。
