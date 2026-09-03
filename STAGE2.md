前提提要: PLANS.md

# Step 1: 全量index构建
有可能有另一个agent在执行PLANS.md中的试点部分, 请不要干扰他. 全量构建使用独立的代码路径、运行目录和产物目录, 不修改或覆盖试点数据库、clip、embedding shard、index或运行记录.

## 数据范围和已有产物

1. 索引覆盖MM-Lifelong的全部181.1小时视频, 包括Day、Week和Month, 不只覆盖Month验证集使用的14个视频.
2. Day、Week和Month是三个独立的逻辑时间轴. `video_id`表示逻辑时间轴, `source_video_id`和相对文件名表示物理视频文件, 所有caption、ASR、topic、event和状态时间都映射到对应逻辑时间轴.
3. 复用已经存在的MAGIC-video官方dense text. Month训练集与验证集共引用22个视频; 官方产物覆盖14个, 缺少的8个是`1, 3, 5, 6, 7, 8, 9, 10`. Week已有A1_JAKE的官方30sec dense captions. Day以及任何其他没有官方dense text覆盖的视频都按下面的统一方法生成.
4. 正式运行前逐个核对源视频文件、音频流、时间轴映射和dense text实际内容. 缺少源视频或时间映射时直接停止, 不用其他视频、空文本或已有index代替.

## 缺失dense text生成

1. 沿用MAGIC-video的30秒切分和caption/transcription输入格式. ASR使用Whisper `large-v3-turbo`; 一个Whisper模型实例使用32个worker.
2. 视觉caption和ASR合并使用独立启动的本地`Inferact/Qwen3.8-27B-NVFP4`. 不共享现有LLM服务; 从执行时满足显存条件的空闲B200中任选单卡启动, `tensor-parallel-size=1`, 并遵循`INFRA.txt`的其他服务设置.
3. 新LLM服务允许每个请求输入一个30秒视频. 视频由TorchCodec按2 fps解码并保留所有采样帧. 每个请求同时提供对应Whisper文本, 一次生成最终30sec dense caption.
4. 索引预处理请求使用`reasoning_effort=low`; 服务保持`enable_thinking=true`和`preserve_thinking=true`. 不设置`temperature`、`top_p`或其他可选生成参数, 也不设置completion或输入token上限.
5. Whisper、视频准备、Qwen请求、层级汇总和embedding客户端都使用32个worker或最多32个在途任务. 每个完成的30sec结果立即写入忽略目录中的稳定checkpoint, 以便按batch恢复.

## 简化index生成

1. 收齐30sec dense captions之后跳过OpenIE、NER和官方semantic consolidation pipeline, 不生成`openie_triples`或对应中间产物.
2. 继续保留30sec和3min episode. 每个3min episode由时间连续的6个30sec dense captions按顺序直接拼接, 不增加一次摘要调用.
3. Qwen以30分钟为正常调用边界. 一次结构化调用生成该区间需要的实体、直接semantic triples、topic facts、局部event steps、重要物品状态和向上汇总所需的摘要. 30分钟结果只保存在checkpoint和完整调用trace中, 不作为episode写入最终DuckDB.
4. 每两个连续30分钟结果合成一个1h episode. 后续层级固定为`30min -> 1h -> 6h -> 逻辑视频 -> Day/Week/Month尺度`, 逐层生成和合并topic chains与event chains.
5. 每次模型或外部边界调用最多尝试5次, 每次保持同一方法和设置. 30分钟输入连续5次发生context超限时拆成两个15分钟输入; 15分钟仍连续5次超限时继续二分, 最小到30秒. 30秒输入连续5次仍失败则保留trace并终止整次构建, 不跳过该区间.
6. 所有结构化Qwen输出使用SDK的结构化parse接口和明确schema. 非context错误按项目的sample级或run级失败边界处理; 不解析出有效结果时不得写完成checkpoint或进入上层汇总.

## DuckDB和object states

1. 最终DuckDB保持试点的`episodes`、`entities`、`semantic_triples`、`topic_chains`、`topic_facts`、`event_chains`、`event_steps`和`visual_clips`查询接口, 同时增加物理源文件到逻辑时间轴的映射. semantic triples由30分钟调用直接生成, 不依赖OpenIE.
2. 新增`object_states`表, 至少包含`row_id`、`video_id`、`source_video_id`、`episode_3min_row_id`、`caption_30sec_row_id`、`start_s`、`end_s`、`object`、`status`和规范化`text`.
3. `episode_3min_row_id`是object states的外层join key; `caption_30sec_row_id`指出状态实际出现的30秒区间. 例如血量条为`3/10`时, `object`记录血量条, `status`记录`3/10`, 并关联对应3min和30sec episode.
4. 保留MAGIC-video的topic和event字段含义, 但增加逻辑视频级和Day/Week/Month尺度级chain. 所有外键和时间必须能回到真实30sec caption与物理视频片段.

## Embedding和产物

1. 文本和视频都使用`Qwen/Qwen3-VL-Embedding-8B`. 文本embedding覆盖所有可搜索文本表, 包括`object_states`; 视频embedding覆盖30sec visual clips.
2. 允许共享当前embedding服务, 客户端最多32个请求在途. 为满足服务现有的允许媒体路径, 全量clip写入`data/pilot/clips/stage2`, 不覆盖现有试点clip.
3. 每个embedding立即写入稳定shard, 恢复时核对row ID、4096维向量和有限数值后再跳过. 所有shard完成后才合并为每表的row ID数组和向量矩阵.
4. 全量数据库、checkpoint、trace、clip、embedding shard、合并index、日志和W&B运行记录都放在git忽略路径中; 开始写大量产物前先确认忽略规则生效.

## 审查、执行和验收

1. 代码完成后按`REVIEW.md`执行中等改动审查并修复所有plain-yes项. 正式全量运行前再执行一次完整仓库Claude审查, 然后按项目规则处理审查后的局部diff.
2. 正式运行前验证独立Qwen服务的真实启动参数、video输入、2 fps解码、thinking保留和结构化输出; 验证共享embedding服务返回真实的4096维文本与视频向量.
3. 全量任务通过Codex前台session启动并监控, 详细日志写文件, stdout只保留有界里程碑. 所有run记录到W&B, 并遵循项目规定的逐步降低监控频率.
4. 对Day、Week和Month各人工检查至少一个完整trace: 源30sec视频与音频、Whisper输入输出、Qwen视频和文本输入、结构化返回、3min拼接、30min checkpoint、1h/6h/视频/尺度汇总、DuckDB真实行、object state关联以及最终embedding检索结果.
5. 完成条件是三个逻辑时间轴全部覆盖, 所有上层汇总都能追溯到真实30sec证据, DuckDB与embedding index完整写出, 正式日志到达真实结束行且进程正常退出. 完成并核查后停止, 不进入Step 2.

## Handoff: 使用当前全量index

### 唯一运行时入口

1. 另一个agent应直接使用现用数据库`data/pilot/stage2/index.duckdb`和现用索引目录`data/pilot/stage2/embedding_index`. 这两个位置已经包含重新生成并替换后的Week结果.
2. `data/pilot/stage2/week_regen`保存本次Week重生成的checkpoint、trace、组合数据库和替换前验证产物, 只用于审计, 不是下游运行入口. 不要把它与现用路径混用.
3. DuckDB必须以只读方式打开. 不要重新生成、覆盖或重排现有NPY; `*.row_ids.npy`的位置与同表`*.vectors.npy`第一维严格一一对应.
4. 所有文本和视频向量都是`Qwen/Qwen3-VL-Embedding-8B`生成的4096维`float32`单位向量. 文本查询必须调用同一模型, served model name为`qwen3-vl-embedding-8b`.

### 可搜索表与文件

| DuckDB表 | 检索文本 | row ID文件 | 向量文件 | 当前行数 |
|---|---|---|---|---:|
| `episodes` | `text` | `episodes.row_ids.npy` | `episodes.vectors.npy` | 25700 |
| `entities` | `name` | `entities.row_ids.npy` | `entities.vectors.npy` | 4530 |
| `semantic_triples` | `text` | `semantic_triples.row_ids.npy` | `semantic_triples.vectors.npy` | 10159 |
| `topic_facts` | `fact` | `topic_facts.row_ids.npy` | `topic_facts.vectors.npy` | 30662 |
| `event_steps` | `description` | `event_steps.row_ids.npy` | `event_steps.vectors.npy` | 33327 |
| `object_states` | `text` | `object_states.row_ids.npy` | `object_states.vectors.npy` | 6234 |
| `visual_clips` | 30sec视频 | `visual_clips.row_ids.npy` | `visual_clips.vectors.npy` | 21782 |

`topic_chains`和`event_chains`本身没有单独的向量文件. 先检索`topic_facts`或`event_steps`, 再通过`chain_row_id`回到对应chain. `visual_clips`是复用的原视频embedding; 其row ID和向量文件在Week数据库替换时没有修改.

### 加载索引

```python
from pathlib import Path

import duckdb
import numpy as np

stage2_root = Path("data/pilot/stage2")
connection = duckdb.connect(str(stage2_root / "index.duckdb"), read_only=True)

table = "episodes"
row_ids = np.load(
    stage2_root / "embedding_index" / f"{table}.row_ids.npy",
    mmap_mode="r",
    allow_pickle=False,
)
vectors = np.load(
    stage2_root / "embedding_index" / f"{table}.vectors.npy",
    mmap_mode="r",
    allow_pickle=False,
)
assert vectors.shape == (len(row_ids), 4096)
assert vectors.dtype == np.float32
```

不要根据DuckDB返回顺序猜测向量位置. 命中位置`i`的数据库主键必须取自`row_ids[i]`, 再用该主键查询DuckDB.

### 生成查询向量并检索

当前共享embedding容器为`egolife-qwen3-vl-embedding-8b`, 宿主端口为8013, 使用GPU 4. 运行前先核对真实服务:

```bash
docker ps --filter name=egolife-qwen3-vl-embedding-8b
curl -sS http://127.0.0.1:8013/v1/models
```

如果该服务不在运行, 按`INFRA.txt`启动同一个模型和served model name; 不要换模型或复用其他LLM服务. 从宿主或`--network host`的Docker客户端调用:

```python
import httpx
import numpy as np

query = "拿着紫色屏幕的手机走遍卧室并扫描房间物体"
payload = {"model": "qwen3-vl-embedding-8b", "input": [query]}
response = httpx.post(
    "http://127.0.0.1:8013/v1/embeddings",
    json=payload,
    timeout=None,
)
response.raise_for_status()
data = response.json()["data"]
assert len(data) == 1 and data[0]["index"] == 0
query_vector = np.asarray(data[0]["embedding"], dtype=np.float32)
assert query_vector.shape == (4096,) and np.isfinite(query_vector).all()

scores = vectors @ query_vector
top_positions = np.argsort(-scores)[:20]
hits = [(str(row_ids[i]), float(scores[i])) for i in top_positions]
```

需要限定逻辑时间轴或episode粒度时, 先从DuckDB取得允许的row ID, 再只对这些NPY位置计算分数. 定位原始片段时使用`episodes.granularity='30sec'`或直接检索`visual_clips`; 获取外层时间上下文时使用`3min`、`1h`、`6h`、`video`或`scale`.

命中后按主键读取实际内容, 例如:

```python
row_id, score = hits[0]
row = connection.execute(
    "SELECT row_id, video_id, start_s, end_s, granularity, text "
    "FROM episodes WHERE row_id = ?",
    [row_id],
).fetchone()
```

### 3min外层join和30sec精确证据

`object_states.episode_3min_row_id`是外层上下文join key, `object_states.caption_30sec_row_id`是状态真实出现位置. 不得只保留3min关联而丢掉30sec证据. 通过`co_clip`边可以继续回到实际视频文件:

```sql
SELECT
    s.row_id,
    s.object,
    s.status,
    s.text,
    e3.row_id AS episode_3min_row_id,
    e3.text AS context_3min,
    e30.row_id AS caption_30sec_row_id,
    e30.text AS evidence_30sec,
    v.row_id AS visual_clip_row_id,
    v.filename
FROM object_states AS s
JOIN episodes AS e3
  ON e3.row_id = s.episode_3min_row_id
JOIN episodes AS e30
  ON e30.row_id = s.caption_30sec_row_id
LEFT JOIN graph_edges AS g
  ON g.src_row_id = s.caption_30sec_row_id
 AND g.edge_type = 'co_clip'
LEFT JOIN visual_clips AS v
  ON v.row_id = g.dst_row_id
WHERE s.row_id = ?;
```

`topic_fact_evidence`和`event_step_evidence`同样保存到30sec episode的真实证据关联. 下游agent应沿这些关联读取视频和文本, 不从chain摘要反推不存在的证据.

# Step 2: 全量训练
和试点大部分地方一样, 但是有一部分改进, 以下面为准:
## 修改
1. 有些时候模型会"抽风"进入循环输出, 我们需要在harness层面做两件事情, 1. 添加一个submit_answer的tool, 这个作为最终答案提交, call这个tool后退出agent loop. 如果连续3次AGENT的输出完全一致, 本次trace作废, 进入重试. 
2. 本次使用更大的batch, 5个问题一起, 每个问题10个rollout. 
3. 模型改用**qwen 3.8 next flash**(不是qwen 3.8 27B, 见INFRA.txt如何运行)
4. SKILL部分不再是一个文件, 而是多个, 如何构建SKILL参考antrophic官方的SKILL(https://github.com/anthropics/skills/tree/main/skills/skill-creator), 注意, 虽然这个也可以被理解为META-SKILL但是和我们的不一样, 我们的META SKILL是如何反思, 这个是如何写SKILL, 是在反思完成后更新skill文件用的. 这意味着, 我们的skill不再是一个单独的md文件, 而是一个目录, 这个目录包括多个markdown文件, 每个文件对应一个特殊情况(比如: 查找特定时间任务), 包含介绍, 何时使用, 以及主要内容等. 内容可以包含SQL脚本示例 (类似于SKILL中的脚本文件, 但是我们不单独存储脚本而是放在markdown里面).
6. 为了实现5, 我们需要对harness做两点更改
6.1 rollout时, 模型不读取所有skill文件, 而是一个list, 列出可用skill和何时使用, 然后多了一个工具就是读取SKILL, 是具体SKILL名字, 然后工具返回SKILL内容
6.2 反思更新SKILL的时候, 模型可以增加文件, 修改文件, 和删除文件. 这个SKILL文件夹被单独的git仓库管理, 每次upadte之后(多有update, 不是单个patch), 都会commit一次, 然后反思的时候agent还可以看历史, 回滚修改 (安全问题, 不允许直接git回滚, 而是参考历史重新update). 这个是一个多轮的agent loop, 类似于直接受控操作文件系统. 
