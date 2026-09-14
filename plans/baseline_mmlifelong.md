# Training Free Eval On MM-Lifelong

# 前情提要
刚才另一个agent完成了刚才另一个plans/baselines_plan.md, 目录在./training_free/, 他eval的是MAGIC Video和我们的方法在EgoLife上面的eval, 部分代码偏离了plan, 这是后期修正并且是intended. 

文件说明
| 文件 | 用途 |
|---|---|
| [harness.py](/raid/wg25r/video_sql_trace2skill/training_free/harness.py) | 答题 agent、工具、共享调用预算 |
| [evaluate.py](/raid/wg25r/video_sql_trace2skill/training_free/evaluate.py) | 批量运行、judge、重试、W&B、token 对账 |
| [prepare_baselines.py](/raid/wg25r/video_sql_trace2skill/training_free/prepare_baselines.py) | 数据转换、建库、embedding、逐题时间过滤 |
| [index_schema.py](/raid/wg25r/video_sql_trace2skill/training_free/index_schema.py) | DuckDB schema、时间转换 |
| [ppr.py](/raid/wg25r/video_sql_trace2skill/training_free/ppr.py) | 支持 pre/post SQL 的 PPR |
| [magic.py](/raid/wg25r/video_sql_trace2skill/training_free/magic.py) | MAGIC 基线适配 |
| [skills/](/raid/wg25r/video_sql_trace2skill/training_free/skills) | 静态工具使用说明 |
| [trace2skill_service.py](/raid/wg25r/video_sql_trace2skill/infra/trace2skill_service.py) | 四个本地服务的启动配置 |

换数据集时，主要对接这些地方：
准备数据：准备脚本目前绑定 EgoLife/A1_JAKE 和现成 MAGIC artifacts，需要修改来源路径、输入字段及人物 ID。
时间和身份：当前使用 DHHMMSSFF，judge 的提示和 metadata 写死了 Jake；需随新数据集修改。
产出接口：questions.json 包含 id/question/query_time/reference_answer；另需逐题 DuckDB、embedding、MAGIC 图和视频时间映射。具体见[索引说明](/raid/wg25r/video_sql_trace2skill/notes/index.md)。
对比数据：SQL evaluator 当前读取同一 artifact 根目录下的 results/quality-magic-judge-v2 做实时 overlap，必须对应新数据集。
服务配置：修改实际媒体挂载路径，启动前重新核对 GPU；当前本任务容器已经删除。

```bash
.venv_baselines/bin/python -m training_free.evaluate \
  --method sql --artifact-root <新数据集artifacts目录> \
  --run-dir <新结果目录> --concurrency 20
```

Embedding 的生成入口是 [prepare_baselines.py](/raid/wg25r/video_sql_trace2skill/training_free/prepare_baselines.py)，流程为：

```text
prepare → media → embed-text → embed-video → assemble → databases
```

- `prepare/media`：准备数据库、未嵌入的图和视频时间映射；新数据集需先完成上一条中的适配。
- `embed-text`：去重后的文本每批 512 条，生成 4096 维向量；空文本在组装时使用零向量。
- `embed-video`：按 caption 明确时间区间裁剪缓存片段，再均匀采样，禁止二次采样；最多 64 个请求在途，完成即保存。
- `assemble`：将向量写回 MAGIC 图，并生成 SQL 检索用矩阵。
- `databases`：生成逐题截止过滤数据库；只重算向量、未改数据时不需要重新生成。

模型为 **Qwen3-VL-Embedding-8B**。先在独立前台会话启动 embedding 服务：

```bash
.venv_baselines/bin/python -u infra/trace2skill_service.py embedding
```

新数据集需要同步修改该文件中的媒体挂载路径和 `--allowed-local-media-path`。生成 embedding 只需要 embedding 服务。

数据库和媒体已准备好后，在另一个前台会话运行：

```bash
export HF_HOME=/raid/huggingface
export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 NUMEXPR_NUM_THREADS=10 VECLIB_MAXIMUM_THREADS=10
dataset_artifacts=/raid/new-dataset-artifacts
embedding_memory_bytes=$(awk '/^MemAvailable:/ {printf "%.0f", $2 * 1024 / 5}' /proc/meminfo)

mkdir -p runs/new-dataset-embedding
for stage in embed-text embed-video assemble; do
  systemd-run --user --scope -p MemoryMax="$embedding_memory_bytes" \
    .venv_baselines/bin/python -u -m training_free.prepare_baselines \
    "$stage" --root "$dataset_artifacts" \
    --embedding-url http://127.0.0.1:8114/v1 \
    > "runs/new-dataset-embedding/$stage.log" 2>&1 || exit 1
done
```

**现有脚本会跳过已经存在的 shard，没有 `--force`。** 同一个 artifact 根目录下强制重算时，先把以下旧目录移到备份位置：

```text
embedding-shards/text/
embedding-shards/video-grouped/
```

若 caption 区间和原视频没有变化，可继续复用视频切片缓存。重算完成后必须运行 `assemble`，它会更新：

```text
embeddings/*.row_ids.npy
embeddings/*.vectors.npy
embeddings/visual_clip.video_vectors.npy
magic/graph.pkl
magic/visual_embeddings.pkl
```

见INFRA.txt如何运行所有的vllm server
# 材料
Magic-video的mmlifelong artifacts: /raid/huggingface/hub/datasets--jiazhengli7--magic-video-artifacts/snapshots/8573f317684ce55c0999f86386666d3d3fe76ae3/mmlifelong
Embedding模型: https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B
QA问题和视频: HF Dataset MM-Lifelong/MM-Lifelong


# 当前任务
1. 仅评测 MM-Lifelong Month validation 的 623 题。选择有 MAGIC Video artifacts 的视频，采用 merge artifacts，按照 preparing 方法准备并保存所需 artifacts（包括 embedding）。保留原题和原答案，不进行选项处理或开放题改写。视频片段的缓存与节点标识使用视频身份和起止时间作为 key，区分同一长视频中的不同片段。
2. 现在GPU极度空余, 选择出5个GPU(下俩标记为GPU0-4, 不代表真实GPU id), GPU 0: embedding server和2个encoder. GPU1,2: 一个qwen 3.8 27B server GPU3,4: 另一个 3.8 27B server. 两个encoder分别链接一个PD server. 共享一个embedding server. 两个VLLM server暴露不同的接口
3. 我们的方法使用 SQL60 + PPR：保留原 SQL 工具，以支持 pre/post SQL 的 PPR 作为 semantic_search，不默认或优先推荐。两种方法均进行全局检索，没有 query_time，不沿用 EgoLife 的逐题时间截止过滤。
4. 同时启动 MAGIC Video 和 SQL60 + PPR，分别使用一个 vLLM 接口，并分别记录 token 消耗。overlap 根据运行中双方实时完成的共同题目计算，不要求 MAGIC 全部运行完成后再启动 SQL。
5. 使用 MM-Lifelong 官方 rubric 和评分口径。其余运行设置继承上一轮：所有 LLM 调用（包括 judge）使用 Qwen/Qwen3.8-27B-FP8、xhigh，保留 reasoning 历史；客户端不设置采样参数和 token 上限；每边并发 20，SQL 每次尝试最多 60 次共享调查调用再加一次强答，MAGIC 保留上游检索轮数；失败题在本批结束后集中重试一次；W&B 以 1 Hz 记录 token、吞吐、GPU 和实时 overlap。
6. 两套 harness 的内存上限合计为服务启动后可用内存的 20%，CPU KV cache 合计 512GB。
7. 只比较结果，artifacts 和结果本地保存，不上传 HF。
