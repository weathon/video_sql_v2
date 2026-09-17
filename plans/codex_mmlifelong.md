# Codex VideoSQL：MM-Lifelong Month 与 EgoLife Open

## 简介

用 Codex 作为 harness、`gpt-5.6-terra` 作为主模型测试 VideoSQL。Codex 沿用 analysis 设置，将 reasoning 从 high 改为 max，service tier 保持 default。
上一个任务：[baseline_mmlifelong.md](/raid/wg25r/video_sql_trace2skill/plans/baseline_mmlifelong.md)。主要说明：[docs/mcp-analysis-handoff.md](/raid/wg25r/video_sql_trace2skill/docs/mcp-analysis-handoff.md)。

## 步骤

0. 启动 Qwen3.8-27B-FP8 和 Qwen3-VL-Embedding-8B 的 Docker；Qwen 使用单卡。
1. 查看 `src/mcp.py` 的工具架构、`src/analysis.py` 的 Codex SDK 用法和 chat 归档方式。
2. 创建 `training_free/codex_harness.py`，用 Codex 复刻原 harness；创建 `training_free/val_codex.py` 答题。使用 Codex 原生循环与上下文管理，去掉共享 60 次预算和 `replace_tool`。
3. 跳过 embedding 创建，直接使用下述 artifacts，不缓存裁剪视频。
4. 先跑 Month validation 全部 623 题，保留原题、原答案和 ID；每题可检索全部 14 个视频，不使用标注线索或逐题截止。Qwen QA 使用 xhigh，普通失败批末重试一次。
5. Month 完成后跑 EgoLife Open 全部 500 题，使用已有开放题和逐题截止数据库。按已确认的 fair 设置，Qwen QA 省略 reasoning effort，答题最多尝试 4 次。
6. 两阶段均并发 5，所有工具 QA 使用 `Qwen/Qwen3.8-27B-FP8`，记录答案并用 API `gpt-5` judge。沿用各自官方评分和 metadata；客户端不设采样参数或 token 上限。
7. 保留 W&B，去掉 1 Hz 和 vLLM 相关 metrics；记录与下述 Qwen3.8 VideoSQL 结果的 delta。评分 4/5→1、3→0.5、0/1/2→0，报告 score-1 ratio 和 average rounded score，分母分别固定为 623、500。

## Month artifacts 与 embedding

根目录：`/raid/wg25r/video_sql_trace2skill/runs/mmlifelong-month/artifacts`

| 相对路径 | 内容 |
|---|---|
| `embeddings/visual_clip.video_vectors.npy` | 视觉 embedding，9168 × 4096，float32 |
| `embeddings/visual_clip.row_ids.npy` | 对应节点 ID |
| `embeddings/*.vectors.npy`、`*.row_ids.npy` | 文本 embedding 和 ID；`visual_clip.vectors.npy` 是 caption 文本向量 |
| `index.duckdb` | 完整 SQL 数据库 |
| `magic/graph.pkl`、`magic/visual_embeddings.pkl` | MAGIC 图与视觉向量 |
| `magic/captions/` | 各粒度 captions |
| `videos/month/`、`media-map.json` | 14 个完整视频及身份、时间映射 |
| `questions.json` | 623 道题 |

视觉 embedding 来源：`/raid/wg25r/video_sql/data/mm_lifelong_stage1/mm_lifelong.sqlite`，通过 `visual_clips`、`visual_clips_emb` 的视频 ID 和起止秒数匹配。
原生成方式：Qwen3-VL-Embedding-8B，每段均匀取 16 帧、长边 420、JPEG；一条 user 消息先放 `Retrieve video segments that help answer questions about a long-form livestream.`，再放 16 张图。视觉查询沿用该指令＋换行＋查询文本的 messages 格式，文本查询保持 input 格式。

- 原脚本：[embed_mm_lifelong_stage1.py](/raid/wg25r/video_sql/build/embed_mm_lifelong_stage1.py)
- 原记录：[HANDOFF_MM_LIFELONG_STAGE1.md](/raid/wg25r/video_sql/docs/HANDOFF_MM_LIFELONG_STAGE1.md)
- 重算抽查：[comparison.json](/raid/wg25r/video_sql_trace2skill/runs/mmlifelong-month/debug/existing-embeddings/comparison.json)
- 原运行配置：[MAGIC](/raid/wg25r/video_sql_trace2skill/runs/mmlifelong-month/results/month-magic/config.json)、[SQL](/raid/wg25r/video_sql_trace2skill/runs/mmlifelong-month/results/month-sql/config.json)

上游 captions、OpenIE、semantic、topic/event chains：`/raid/huggingface/hub/datasets--jiazhengli7--magic-video-artifacts/snapshots/8573f317684ce55c0999f86386666d3d3fe76ae3/mmlifelong`。
重建入口：[prepare_baselines.py](/raid/wg25r/video_sql_trace2skill/training_free/prepare_baselines.py)，指定 `--time-format mmlifelong`，顺序为 `prepare → media → embed-text → embed-video → assemble → databases`；`embed-video` 导入旧 SQLite 向量。本轮直接复用完整 artifacts。

## EgoLife Open artifacts 与对照结果

EgoLife 根目录：`/raid/videosql-video-artifacts`。使用 `questions.json`、`questions/<ID>.duckdb`、`embeddings/`、`videos/A1_JAKE/` 和 `media-map.json`；说明见 [INDEX.md](/raid/videosql-video-artifacts/INDEX.md)。

| 数据集 | Qwen3.8 VideoSQL / GPT-5 judge 对照目录 |
|---|---|
| Month | `/raid/wg25r/video_sql_trace2skill/runs/mmlifelong-month/results/month-sql-gpt5` |
| EgoLife Open | `/raid/wg25r/video_sql_trace2skill/runs/fair-qwen38/formal/fair-qwen38-sql-open` |

数据库、向量和视频保存在本机的 Git 忽略目录，复现需保留完整 artifacts。
