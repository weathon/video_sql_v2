# Baselines 计划（EgoLife week：training-free harness 与 MAGIC-video 对比）

本文件在执行时整体替换 `baselines_plan.md`，不是它的附件。所有约束、步骤、已定参数和待确认项都在这里。

## 0. 背景与目标

- 同一目录、同一分支（`codex/trace2skill-baseline`）上另有一个 agent 在跑 Trace2Skill rollout（入口 `src/trace2skill_baseline.py`，run 目录 `runs/qwen38-fp8-context128k-day-week-tree-month-val-seed0`）。不碰 `src/`、`skills/`、`third_party/`、它的 run 目录，不改动、不重启任何 vLLM 容器。
- 目标一：用我们 harness 的 training-free 形态（静态 `skills/initial`，不做 reflection/evolution）在 EgoLife 题目上跑一遍 baseline，题目用 EgoLifeQA，视频与索引复用 MM-Lifelong week。
- 目标二：克隆 MAGIC-video（https://github.com/lijiazheng0917/MAGIC-video.git），改为使用同一个 Qwen LLM、同一个 embedding 服务和同一份索引，在同一批题上跑 baseline。
- 目标三：两种方法按总分、均分、准确率（judge 满分比例）做 overall 与按 EgoLife 题型的对比，写成一份简报和一份详细背景。

## 1. 边界与纪律

1. 计划阶段只读了当前目录，没有访问 git 历史、目录以外路径或本机已有的 MAGIC 副本。这一点在执行阶段继续成立：不运行任何读历史的 git 命令（`log`、`show`、`diff <commit>` 等），不查找、不打开本机已有的 MAGIC-video 副本。
2. 执行阶段需要用到以下目录以外的资源。用户在计划阶段明确"全部不允许"，因此执行开始时逐项向用户申请，未获批准前一律不碰；申请时只列必要项：
   - 只读 `/raid/wg25r/video_sql_v2/data/runtime/week/`（`index.duckdb`、`embedding_index/`、`media-map.json`；`src/trace2skill_baseline.py` 当前就从这里读）。
   - 只读 `/raid/huggingface` 下 MM-Lifelong 快照中的 week 源视频（media-map 里列出的文件），用于一致性核对与 clip_qa 取帧。
   - 用 `HF_HOME=/raid/huggingface` 下载 `lmms-lab/EgoLife` 的 QA 与 A1_JAKE 全部视频。
   - 访问 `127.0.0.1:8003`（Responses/Chat 代理）和 `127.0.0.1:8014`（embedding），以及 `docker ps` 只读核对。
   - `git clone` GitHub 上的 MAGIC-video。
   - `systemd-run --user --scope` 给 MAGIC 进程设内存上限。
3. 服务不在运行、数据缺失、映射对不上、字段与预期不符：停下来问用户，不换模型、不换索引、不造替代数据、不跳过阶段。
4. git：在当前分支直接提交，每次编辑后提交；只 `git add` `training_free/**`、`baselines_plan.md`、以及本计划新建的 markdown。不 add 别人的未跟踪/已修改文件（`AGENTS.md`、`src/analysis.py`、`src/check_status.py`、`notes/INCIDENT_*` 等）。不用 `git stash`。
5. venv：两个，全部新建，都被 `.gitignore` 的 `.venv_codex_*/` 忽略：
   - `.venv_codex_training_free`：`uv venv .venv_codex_training_free`，然后 `UV_PROJECT_ENVIRONMENT=.venv_codex_training_free uv sync`（按 `uv.lock`）。
   - `.venv_codex_magic`：`uv venv .venv_codex_magic`，按 MAGIC 克隆后的 requirements 安装，再补 `wandb`、`weave`、`duckdb`、`numpy`、`httpx`、`openai`、`openai-agents`（judge 走 `training_free/val.py`）。
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
  prepare_egolife.py  # 下载、核对、出题、生成按截止时刻过滤的 duckdb 与 media-map
  run_training_free.py# 我们方法的 EgoLife 评测入口
  MAGIC-video/        # 新鲜克隆，删除内嵌 .git，由本仓库跟踪
  run_magic.py        # MAGIC 的评测入口（索引转换、HTTP embedder、LLM 端点、并发 10、judge、W&B）
  compare.py          # 汇总两个 run，写 RESULTS.md / RESULTS_DETAILS.md
  RESULTS.md          # 简报：只有对比数字
  RESULTS_DETAILS.md  # 详细背景：设置、commit、核对结果、W&B 链接、运行统计、偏差说明
runs/egolife-week-shared/          # 两种方法共用的产物（runs/ 已被 gitignore）
  egolife_raw/                     # 下载的原始 QA 快照副本（json）
  verification/                    # 时长、映射、抽帧比对图与 report.json
  questions.json                   # 最终题目（开放式）
  cutoffs/<cutoff_s>/index.duckdb  # 每个不同截止时刻一份过滤后的库
  cutoffs/<cutoff_s>/media-map.json
  cutoffs/<cutoff_s>/magic/        # 该截止时刻转换给 MAGIC 的索引（格式待克隆后确定）
runs/training-free-egolife-week/   # 我们方法的 run 目录，W&B run 名同名
runs/magic-video-egolife-week/     # MAGIC 的 run 目录，W&B run 名同名
```

`runs/egolife-week-shared/` 是本计划新增的名字（用户批准的是两个 run 目录名），执行前和其他待确认项一起确认。

## 3. 已确定的参数

| 项 | 值 |
| --- | --- |
| 复制进 training_free 的文件 | harness.py、val.py、normalize.py、skills/initial、tracing_bridge.py（最小集合） |
| 题目 | lmms-lab/EgoLife 的全部 EgoLifeQA（A1_JAKE），不抽样 |
| 开放式改写 | 规则去选项：题干原样，删除四个选项，正确选项文本作 reference answer，不经 LLM |
| query_time | 同时做两件事：题目文本里写明提问时刻；检索硬限制到提问时刻之前 |
| 硬限制实现 | 每个不同截止时刻生成一份过滤后的 duckdb 与截断的 media-map；不重新生成 npy，semantic_search 按过滤库中存在的 row_id 掩码 |
| 过滤条件 | 区间表保留 `start_s <= 截止`（即使 end_s 在截止后）；点时间表 `snapshot_s`/`time_s <= 截止`；entities 保留 `first_seen_s <= 截止` 且 `last_seen_s` 截断到截止；chains 保留 `start_s <= 截止`，summary 原样 |
| clip_qa | 也限制：media-map 只保留截止前的源视频，最后一个的 `timeline_end_s` 截到截止时刻 |
| harness 设置 | 全部沿用 `src/trace2skill_baseline.py` 的 `make_harness`/`rollout`/`score_rollout`：模型 `Qwen/Qwen3.8-27B-FP8`、base_url `http://127.0.0.1:8003/v1`、reasoning xhigh + summary auto、clip_qa_video 开、650 图额度、prompt_cache_key、attempts 3、耗尽后 skip 整题、judge 同模型 0–5 分 |
| embedding | `http://127.0.0.1:8014/v1`，served name `qwen3-vl-embedding-8b`，4096 维，与现有索引一致 |
| rollout | 每题 1 条 |
| 并发 | 两种方法都是 10 个题同时进行（我们方法用 `MainAgentSlots`，上限改 10） |
| W&B / Weave | entity `3dsmile`，project `video-sql-v2`，run id 与 name 等于 run 目录名，Weave 开启；两种方法都用 `RunMetrics`（每次响应的吞吐、上下文、成本；每题完成后 mean_score、score_ge_4、rollouts_per_minute、平均耗时、trace 长度、replacement 指标），MAGIC 用不到的 replacement 指标记 0 |
| MAGIC 克隆 | `training_free/MAGIC-video/`，远端默认分支最新 commit，记录实际 commit 号；删除内嵌 `.git`，由本仓库跟踪 |
| MAGIC LLM | 同一个 8003 代理、同一模型名；不传任何 reasoning 参数，由服务器默认 thinking 设置决定 |
| MAGIC 内存 | `systemd-run --user --scope -p MemoryMax=1T` |
| MAGIC 线程 | 启动前设 `OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 NUMEXPR_NUM_THREADS=10 VECLIB_MAXIMUM_THREADS=10`（在 import numpy/torch 之前） |
| MAGIC 失败 | 与 harness 同规则：每题最多 3 次，错误落盘，耗尽写 `skipped.json` 跳过整题；非可重试错误中止整个 run |
| 指标 | 总分 = 所有题 judge 分之和，同时给均分；准确率 = judge 分等于 5 的比例；分组用 EgoLife 的 `type`；skip 的题单独列出不计入分母 |
| 视频核对 | 逐天总时长 + EgoLife 片段文件名时刻到 week 时间轴的映射 + 对齐时刻两边抽帧像素比对并人工看图；参与者按 A1_JAKE |
| EgoLife 下载 | QA + A1_JAKE 全部视频，`HF_HOME=/raid/huggingface` |
| 输出 | `training_free/RESULTS.md`（只有对比数字）、`training_free/RESULTS_DETAILS.md`（背景、运行统计） |

## 4. 步骤

### 4.1 环境与服务核对

1. 建两个 venv（第 1 节第 5 条）。
2. 获批后：`docker ps` 与 `curl 127.0.0.1:8003/v1/models`、`curl 127.0.0.1:8014/v1/models` 核对两个服务在跑且模型名正确。不在跑就停下报告，不自行启动。
3. 把 `src/harness.py`、`src/val.py`、`src/normalize.py`、`third_party/trace2skill/tracing_bridge.py`、`skills/initial/` 复制到 `training_free/`，import 路径改成 `training_free.*`。提交。

### 4.2 EgoLife 数据：下载、核对、出题、过滤库（`prepare_egolife.py`）

1. 先用 `huggingface_hub` 列出 `lmms-lab/EgoLife` 的文件结构。QA 若是 dataset 配置就用 `datasets.load_dataset`，若是普通 json 就下载该文件；A1_JAKE 视频是普通 mp4 文件，用 `snapshot_download(allow_patterns=…)` 只取 A1_JAKE。原始 QA 复制一份到 `runs/egolife-week-shared/egolife_raw/`。
2. 打开真实 QA 内容核对字段：提问时刻（预期 `query_time` 含 day 与 HHMMSSFF）、题型 `type`、题干、四个选项、正确答案字母。与预期不符就停下问用户；QA 若包含 A1_JAKE 以外参与者的题也停下问。
3. 视频一致性核对（A1_JAKE 对 MM-Lifelong week）：
   - 读 `runtime/week/media-map.json` 的 sources（路径、时长、timeline_start_s/end_s），先弄清 week 时间轴由哪些物理文件按什么顺序拼成，并把观察写进 `verification/report.json`。
   - 逐天比较 MM-Lifelong 对应文件的总时长与 EgoLife 该天全部片段的时长之和（ffprobe 读实际时长，不只看文件名）。
   - 建立映射：EgoLife 片段文件名里的 DAY 与时刻 → week 时间轴秒数。预期是"该天在 week 时间轴上的起点 + （片段时刻 − 该天第一个片段时刻）"；映射公式与每天的起点写进 report。
   - 每天至少取 3 个对齐时刻（首个片段起点附近、中间、末尾），两边用 ffmpeg 取帧，缩放到同一分辨率后算平均绝对像素差，保存两边 jpg 并人工看图。
   - 任一天时长不一致、映射不是简单偏移、抽帧明显不同：停下问用户，不做任何替代。
4. 出题写 `questions.json`，每题保留原始 ID、`type`、原题干、四个选项、正确字母、原始 query_time，另加：
   - `question`：开放式文本，形如 `Asked on DAY3 at 11:22:02. <原题干>`（具体格式执行时定稿后写进 RESULTS_DETAILS.md）。
   - `answer`：正确选项文本。
   - `cutoff_s`：按第 3 步映射得到的 week 时间轴秒数。
   - 答题 agent 只看到 `question`；reference 只进 judge。
5. 对每个不同的 `cutoff_s`，从 `runtime/week/index.duckdb` 只读复制出 `cutoffs/<cutoff_s>/index.duckdb`（`ATTACH … (READ_ONLY)` + `CREATE TABLE … AS SELECT … WHERE …`，参照 `src/prepare_runtime.py` 的写法），过滤条件见第 3 节；`graph_edges`、`topic_fact_evidence`、`event_step_evidence` 只保留两端都还存在的行。同时写截断的 `media-map.json`。不生成 npy。
6. 抽查若干截止时刻的库：行数随截止时刻单调增加、没有 `start_s` 超过截止的行、media-map 末尾被截断。

### 4.3 我们方法：harness 改动与评测入口

1. `training_free/harness.py` 唯一功能改动：`HarnessContext.__post_init__` 里对每个 `SEARCH_TABLES` 表从（过滤后的）duckdb 读出全部 `row_id`，对共享 npy 的 `row_ids` 做成布尔掩码；`semantic_search` 里排序前把掩码外的分数置为 `-inf`（或只在允许位置上 argsort），其余逻辑不变。`visual_clips` 表同样处理。
2. `training_free/run_training_free.py`：以 `src/trace2skill_baseline.py` 的 `RecordedResponsesModel`、`make_harness`、`rollout`、`score_rollout`、`evaluate_month`、`MainAgentSlots`、`RunMetrics` 为蓝本改写：
   - 每题的 database/embedding_index/media_map 分别指向 `cutoffs/<cutoff_s>/index.duckdb`、`runtime/week/embedding_index`（共享）、`cutoffs/<cutoff_s>/media-map.json`。
   - `MAIN_AGENT_CONCURRENCY = 10`，每题 1 条 rollout，目录 `runs/training-free-egolife-week/qa/question-<ID>/rollout-00/`。
   - `wandb.init(entity="3dsmile", project="video-sql-v2", id=RUN.name, name=RUN.name, resume="allow", dir=RUN)`，`init_weave()`，`RunMetrics(wandb_run, "egolife", RUN)`；`wandb-run.json` 记 id 与 url。
   - `skills/initial` 复制到 run 目录并逐字节核对（沿用现有做法）。
   - 每题结果 `result.json` 字段与 `score_rollout` 一致（score、success、cost_usd、trace_length、replacement_metrics、retrieval_qa_metrics、trace_files），另带 `type`、`cutoff_s`。全部完成后写 `summary.json`（完成数、skip 列表、总分、均分、准确率、按 type 分组）。
   - 可恢复：已有 `answer.json`/`result.json`/`skipped.json` 直接复用，沿用现有 attempt 归档逻辑。
3. 提交，按 `REVIEW.md` 由新 subagent 审 diff，修掉 plain-yes 项，再提交。
4. 启动：`.venv_codex_training_free/bin/python -u -m training_free.run_training_free`，stdout/stderr 写 `runs/training-free-egolife-week/formal-01.log`。
5. 运行中与结束后核对 trace（不是只看数量）：随机打开若干题的 answer segment，确认 semantic_search 命中行时间都不超过该题截止、run_sql 结果时间不超过截止、clip_qa 采样区间在截断 media-map 内、judge 输入是原题干/reference/candidate、W&B 有新数据点。

### 4.4 MAGIC-video：克隆、阅读、确认

1. `git clone https://github.com/lijiazheng0917/MAGIC-video.git training_free/MAGIC-video`，记录 `git rev-parse HEAD`，然后删除 `training_free/MAGIC-video/.git`，`git add` 到本仓库，提交。
2. 完整阅读其 README 与代码（不是 grep）：索引/记忆的存储格式与加载方式、检索流程（图、PPR、截断等）、QA prompt 的组装（是否假定 MCQ）、LLM 客户端接口与采样参数、embedder 的实现、并发方式、是否在回答时需要原始视频或帧。
3. 阅读后停下来向用户确认（不自行决定）：
   - 我们 week 库各表到 MAGIC 索引格式的字段映射，以及双方都没有对应物的字段怎么处理。
   - 是否每个截止时刻转换一份 MAGIC 索引（`cutoffs/<cutoff_s>/magic/`），还是转换一份全量索引并在其检索里按截止过滤。
   - MAGIC 若在回答阶段需要视频或帧，用 MM-Lifelong week 文件与截断 media-map 供给的方式。
   - MAGIC 代码里原有的 temperature/top_p/max_tokens 是删掉还是保留（用户已定不传 reasoning 参数、用服务器默认；采样参数未单独定）。
   - 其 query 文本是否先过我们的 `normalize_text` 再送 embedding（我们索引的查询侧是这样做的）。

### 4.5 MAGIC-video：改造与运行（`run_magic.py` 与对克隆的最小修改）

1. embedder 改为 HTTP 调 `127.0.0.1:8014/v1/embeddings`，模型 `qwen3-vl-embedding-8b`，校验返回 4096 维有限向量；删除进程内 embedder 的加载。
2. LLM 改为 OpenAI 兼容客户端指向 `http://127.0.0.1:8003/v1`，模型 `Qwen/Qwen3.8-27B-FP8`，不传 reasoning 参数；记录每次响应的 usage 给 `RunMetrics.record_response`（chat completions 的 prompt/completion tokens 映射为 input/output tokens，`response_cost` 对该模型名返回 0）。
3. 索引转换脚本：按 4.4 确认的映射，从过滤后的 duckdb 生成 MAGIC 格式；转换后抽查行数、时间范围与原库一致。
4. 并发 10 个题同时进行；每题目录 `runs/magic-video-egolife-week/qa/question-<ID>/`，保存 MAGIC 的检索结果、prompt、原始响应、最终答案；失败规则与 harness 相同。
5. judge：用 `training_free/val.py` 的 `judge_answer`，harness 与我们方法完全相同的构造（同模型、同 reasoning、同 prompt），run_dir 指向该题目录。
6. 启动前在 Python 入口顶部设置线程环境变量；用 `systemd-run --user --scope -p MemoryMax=1T .venv_codex_magic/bin/python -u -m training_free.run_magic` 启动，启动后用 `systemctl --user show` 核对 scope 的 MemoryMax 生效；日志写 `runs/magic-video-egolife-week/formal-01.log`。
7. 提交并按 `REVIEW.md` 审 diff 后再启动。运行中核对：embedding HTTP 调用确实返回向量、检索结果时间不超过截止、LLM 响应正常、W&B 有新数据点、线程数（`ls /proc/<pid>/task | wc -l`）没有失控。

### 4.6 对比与报告（`compare.py`）

1. 读两个 run 的 `result.json`/`skipped.json`，按 `questions.json` 对齐题目。
2. `training_free/RESULTS.md`：只放对比表：overall 与按 `type` 的题数、总分、均分、准确率（分等于 5 的比例），两种方法各一列，外加 skip 题目 ID 列表。不放运行统计。
3. `training_free/RESULTS_DETAILS.md`：两个服务的实际参数（`/v1/models` 与 `docker ps` 所见）、MAGIC 的 commit 号与改动清单、EgoLife 下载范围与 QA 字段、视频核对结果（每天时长、映射公式、抽帧差异、图片路径）、题目文本格式、截止过滤规则与截止时刻数量、W&B 链接、两种方法的运行统计（耗时、吞吐、token）、skip 原因、所有与本计划的偏差。
4. 提交。

### 4.7 收尾

- 用本计划内容整体覆盖 `baselines_plan.md`（执行的第一步就做，并提交）。
- 不关闭任何 vLLM 服务。

## 5. 执行时仍需向用户确认的点

- 第 1 节第 2 条列出的每一项外部访问。
- `runs/egolife-week-shared/` 这个共享产物目录名。
- 题目文本里提问时刻的具体格式。
- 4.4 第 3 条 MAGIC 相关的全部问题（只能在读完新鲜克隆之后问）。
- EgoLife 真实字段或 MM-Lifelong week 的物理文件结构与预期不同时的处理。

## 6. MAGIC-video 已知 bug（保留自原计划）

1. 它使用进程内 embedder，并发下会死锁。解决：改用共享 vLLM embedding 服务，通过 HTTP 调用。
2. 并发时 egolife 会开很多线程，每个并发里 OMP/MKL/BLAS/NumExpr 又各自用满所有核心，线程数暴涨会把机器搞崩（有时还会系统 OOM，连带 NVIDIA 驱动崩掉导致 NCCL 不可用）。解决：用线程环境变量把 OMP/MKL/OPENBLAS/NUMEXPR/VECLIB 各限制到 10 线程，并用 systemd 限制最大内存（本次 1T）。

## 7. 验证方式

- 数据：`verification/report.json` 里每天时长一致、映射公式、抽帧差异值，并且人工看过抽帧 jpg。
- 过滤库：抽查的截止时刻库里没有超过截止的行，media-map 末尾被截断。
- 我们方法：打开真实 trace 核对 semantic_search/run_sql/clip_qa 的时间边界与 judge 输入；`summary.json` 的题数等于 `questions.json` 题数减 skip 数；W&B run 页面有 `egolife/*` 与 `client/*` 曲线。
- MAGIC：每题目录里有检索结果、prompt、原始响应；embedding 请求日志有 4096 维返回；scope 的 MemoryMax 已核对；W&B 同上。
- 报告：`RESULTS.md` 的每个数字能从两个 run 目录的 `result.json` 重新算出。
