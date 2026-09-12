# EgoLife A1_JAKE 索引

根目录为 `/raid/videosql-video-artifacts`。本索引来自指定 MAGIC artifacts，MAGIC 源码版本记录在 `third_party/magic_video/UPSTREAM`。直接使用步骤见 [README](../README.md)。

| 路径 | 内容 |
|---|---|
| `questions.json` | 500 道开放题、原 ID、query time、正确选项的答案文本；参考答案只给 judge |
| `qa-rule-audit.json`、`latency-ids.json` | seed 0 的 100 题转换检查与 20 题计时 ID |
| `index.duckdb` | 全部原生节点、metadata、边、topic/event chains 及事实/步骤 |
| `questions/<ID>.duckdb` | 只包含该题截止前完整记录的独立数据库 |
| `embeddings/<table>.row_ids.npy` | 向量位置对应的原生节点或 fact/step ID |
| `embeddings/<table>.vectors.npy` | 按 ID 保存的 4096 维 float32 文本向量 |
| `embeddings/visual_clip.video_vectors.npy` | 相同 visual_clip ID 顺序的视频向量 |
| `magic/graph.pkl`、`magic/visual_embeddings.pkl` | MAGIC 格式完整图及视频向量映射 |
| `magic/captions`、`magic/semantic_memory`、`magic/topic_chains`、`magic/event_chains` | 规范化后的现成内容；原始关联在规范化前完成 |
| `videos/A1_JAKE`、`video-metadata`、`media-map.json` | 原视频缓存片段、逐片帧率/帧数、媒体与逻辑时间映射 |
| `results` | 两种方法的质量、串行计时、全部 attempts、逐题答案、judge、usage 与完整 traces |

四类节点分别为 `entity`、`episode`、`semantic`、`visual_clip`，公共字段为 `id, node_type, text, start_ts, end_ts, metadata`。边分表为 `mentioned_in`、`appeared_in`、`has_property`、`temporal_next`、`contains`、`co_clip_visual_episode` 和 `co_clip_episode_visual`，保留有向端点、类型和权重，端点均为真正的外键。

`topic_facts.chain_id` 与 `event_steps.chain_id` 分别关联所属 chain。`topic_entities`、`event_entities` 只保存原数据中明确给出的实体关联；不从描述或时间重叠推断额外关联。完整 DDL 位于 `training_free/index_schema.py`。

自然语言统一做 NFKC、casefold、连续空白合并与 strip；保留标点和重音。ID、外键、路径、URL、日期时间和类型标识保持原值。原生时间戳为 `DHHMMSSFF`，其中 `FF` 为帧号；转换媒体秒数时使用实际源视频 fps。

逐题 SQL 数据库删除未来、跨截止点和无明确时间的记录，仅保留仍存在端点的边。Chain 按 fact/step 过滤，留下仍含记录的 chain。SQL 保留全部截止前 semantic 快照；MAGIC 继续使用其原生截止及最新快照规则。文本和视频检索均先取当前题库允许的 ID，再对相应向量计算排名。

```python
from pathlib import Path
import duckdb
import numpy as np

root = Path("/raid/videosql-video-artifacts")
db = duckdb.connect(str(root / "questions/1.duckdb"), read_only=True)
ids = np.load(root / "embeddings/episode.row_ids.npy", mmap_mode="r", allow_pickle=False)
vectors = np.load(root / "embeddings/episode.vectors.npy", mmap_mode="r", allow_pickle=False)
allowed = np.asarray([row[0] for row in db.execute("SELECT id FROM episode ORDER BY id").fetchall()])
positions = np.searchsorted(ids, allowed)
rows = db.execute("SELECT id, start_ts, end_ts, text FROM episode ORDER BY start_ts LIMIT 3").fetchall()
```

查询文本使用 `training_free.normalize.normalize_text` 后请求 `Qwen/Qwen3-VL-Embedding-8B`，served model name 为 `qwen3-vl-embedding-8b`。`episode` 检索 caption 文本，`visual_clip` 检索视频向量；空 caption 的文本向量为全零，视频仍正常生成向量。文本每批 512 条，视频每请求一个缓存片段、最多 64 个在途请求，使用 Qwen loader 均匀采样并保留帧顺序，禁止二次采样。
