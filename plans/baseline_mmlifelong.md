# Training Free Eval On MM-Lifelong

# 前情提要
刚才另一个agent完成了刚才另一个plans/baselines_plan.mdh和plans/fair_plan.md, 目录在./training_free/, 他eval的是MAGIC Video和我们的方法在EgoLife上面的eval, 部分代码偏离了plan, 这是后期修正并且是intended. 

文件说明
| 文件 | 用途 |
|---|---|
| [harness.py](/raid/wg25r/video_sql_trace2skill/training_free/harness.py) | 答题 agent、工具、共享调用预算 |
| [evaluate.py](/raid/wg25r/video_sql_trace2skill/training_free/evaluate.py) | 批量运行、judge、重试、W&B、token 对账 |
| [prepare_baselines.py](/raid/wg25r/video_sql_trace2skill/training_free/prepare_baselines.py) | 数据转换、建库、embedding、逐题时间过滤 |
| [index_schema.py](/raid/wg25r/video_sql_trace2skill/training_free/index_schema.py) | DuckDB schema、时间转换 |
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
- `embed-video`：本次 MM-Lifelong 直接导入已有视觉向量，不重算、不缓存裁剪视频；旧 EgoLife 流程仍按 caption 区间裁剪并均匀采样。
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

本次 MM-Lifelong 不保存视频切片缓存，视频证据直接从原视频按区间读取。导入已有视觉向量、生成文本向量后运行 `assemble`，它会更新：

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
0. 目前的MAGIC Video harness结果修改, 输入的是wall clock time而不是视频时间, 导致无法适用于MM Lifelong. 加上一个backward兼容的flag, 允许运行mm-lifelong格式的时间
1. 仅评测 MM-Lifelong Month validation 的 623 题。选择有 MAGIC Video artifacts 的视频，采用 merge artifacts，按照 preparing 方法准备并保存所需 artifacts（包括 embedding）。保留原题和原答案，不进行选项处理或开放题改写。不缓存裁剪视频；节点标识与视觉 embedding 映射使用视频身份和起止时间作为 key，区分同一长视频中的不同片段。视觉 embedding 直接复用下述已保存向量。
2. 现在GPU极度空余, 选择出5个GPU(下俩标记为GPU0-4, 不代表真实GPU id), GPU 0: embedding server和2个encoder. GPU1,2: 一个qwen 3.8 27B server GPU3,4: 另一个 3.8 27B server. 两个encoder分别链接一个PD server. 共享一个embedding server. 两个VLLM server暴露不同的接口
3. 我们的方法使用 SQL60 不包含PPR的方法, 但是包含semantic search有sql过滤
4. 同时启动 MAGIC Video 和 SQL60，分别使用一个 vLLM 接口，并分别记录 token 消耗。overlap 根据运行中双方实时完成的共同题目计算，不要求 MAGIC 全部运行完成后再启动 SQL。
5. 使用 MM-Lifelong 官方 rubric 和评分口径。其余运行设置继承上一轮：所有 LLM 调用（包括 judge）使用 Qwen/Qwen3.8-27B-FP8、xhigh，保留 reasoning 历史；客户端不设置采样参数和 token 上限；每边并发 20，SQL 每次尝试最多 60 次共享调查调用再加一次强答，MAGIC 保留上游检索轮数；失败题在本批结束后集中重试一次，维护错误继续重试，不能通过排除题目缩小分母；最终报告使用完整评测集分母，EgoLife 为 500，MM-Lifelong Month validation 为 623；W&B 以 1 Hz 记录 token、吞吐、GPU 和实时 overlap。
6. 两套 harness 的内存上限合计为服务启动后可用内存的 20%，CPU KV cache 合计 512GB。
7. 只比较结果，artifacts 和结果本地保存，不上传 HF。
8. 结果写回这个文件下面

MAGIC Video:
Rounded Score score-1 ratio:
Average rounded score: 

VideoSQL:
Rounded Score score-1 ratio:
Average rounded score: 


All code should compute "5/0" at successful task complete. 
Keep the code leanest and simplest possible. 


# Refind Plan
# 执行 MM-Lifelong Month 基线评测

## 目标与已确认决定

完成 [baseline_mmlifelong.md](/raid/wg25r/video_sql_trace2skill/plans/baseline_mmlifelong.md) 的全部准备、评测和结果回填。

- 评测原始 Month validation 全部 623 题，保留原题、原答案和原 ID。
- 每题均可检索全部 14 个有 merge artifacts 的视频；不向模型提供 `clue_interval`、`total_intervals` 等标注，也不据此缩小检索范围。
- QA 没有 `query_time`，本次使用完整视频内容，不构造逐题截止时间。
- 所有结果保存、核验、回填与收尾完成后，仅在整个任务末尾执行一次 `5/0`。

## 数据与接口改动

- 为准备脚本、两套 harness 和 evaluator 增加兼容旧行为的 `--time-format {egolife,mmlifelong}`，默认保持 EgoLife。MM-Lifelong 使用视频内秒数，保留小数精度。
- 使用已缓存的 14 个视频及对应 merge captions、OpenIE、semantic、topic/event chains；按既有阶段完成 `prepare → media → embed-text → embed-video → assemble → databases`。
- 节点、边端点和 chain ID 加视频命名空间；视觉节点及 embedding 映射统一使用“视频身份＋起止时间”，避免同一长视频不同片段互相覆盖。不缓存裁剪视频，视频证据直接从原视频按区间读取。
- 本次 `embed-video` 直接导入 `/raid/wg25r/video_sql/data/mm_lifelong_stage1/mm_lifelong.sqlite` 的 `visual_clips_emb`，通过 `visual_clips` 的 `broadcast_id/start_sec/end_sec` 对应本次片段。已保存全部 14 个视频的 9,168 条、4096 维视觉向量；MAGIC 与 SQL 使用同一份已有向量，不重新生成全量 video embedding。
- 已有视觉向量的生成方式：`Qwen/Qwen3-VL-Embedding-8B`；每段均匀取 16 帧，缩放至长边 420、JPEG 编码；同一条 user 消息先放入 `Retrieve video segments that help answer questions about a long-form livestream.`，再放入 16 张图片，输出一个片段向量。实现及记录分别见 `/raid/wg25r/video_sql/build/embed_mm_lifelong_stage1.py` 和 `/raid/wg25r/video_sql/docs/HANDOFF_MM_LIFELONG_STAGE1.md`。
- 两套 harness 的视觉查询沿用旧版格式：通过 `messages` 传入同一句检索指令、换行和查询文本；文本检索保持当前 `input` 格式。
- 已按上述旧脚本输入重新 embedding 四段并检查实际请求、返回向量和输入帧。与已有向量的余弦相似度分别为：视频 4 的 0–30 秒 `0.989932`、30–60 秒 `0.985090`，视频 14 的 2700–2730 秒 `0.990925`，视频 11 的 11760–11776.81 秒 `0.983610`。验证结果保存在 `runs/mmlifelong-month/debug/existing-embeddings/comparison.json`。本次采用这些已有视觉向量的生成方式；文本 embedding 仍按原准备流程生成。
- SQL 检索行和 MAGIC 证据明确携带视频身份与视频内时间；视频工具在 MM-Lifelong 模式下按视频身份定位。所有题使用同一份完整只读语料。
- MAGIC 切换到上游 MM-Lifelong 提示与时间格式，保留既有检索参数和五轮检索预算；合并全部视频时保留原生关联，不新增推断关系。
- SQL 使用现有 vector semantic search 及 SQL 过滤，不启用 PPR。主调用与调查子调用共享每次尝试 60 次预算，之后允许一次强制回答。
- 为 evaluator 增加独立 PD metrics URL 和 embedding URL 参数，消除固定端点；清除本次调用路径中现有 QA 子调用的客户端 token 上限。

## 服务与运行设置

启动前重新检查资源；目前可用的物理 GPU 分配如下：

| 服务 | GPU | 主机端口 |
|---|---|---|
| 共享 embedding、两个独立 encoder | 2 | 8214、8225、8235 |
| MAGIC PD，TP=2 | 3、5 | 8224 |
| SQL PD，TP=2 | 6、7 | 8234 |
| MAGIC / SQL proxy | 无 | 8203 / 8204 |

- 两个 encoder 分别连接各自 PD，使用独立 encoder cache；先启动 encoder，再启动共享 embedding。
- 所有 LLM 调用包括 judge 均使用 `Qwen/Qwen3.8-27B-FP8`、`xhigh`，保留 reasoning 历史；客户端不设置采样参数或 token 上限。
- 继承已配置的服务参数，包括 PD/encoder 的非零多模态缓存。CPU KV cache 沿用每 rank 128 GiB，四个 rank 合计 512 GiB。
- 全部服务启动后读取可用内存，两套 harness 各设置其 10% 的内存上限，合计 20%。
- 两边各并发 20；MAGIC 初始化运行配置后立即启动 SQL，不等待 MAGIC 完成评测。
- artifacts、日志、缓存与完整 traces 保存到已忽略的 [运行目录](/raid/wg25r/video_sql_trace2skill/runs/mmlifelong-month)。不上传 HF。
- 服务和任务全部以前台会话运行；资源不足立即报告，不自行等待。完成后停止并手动删除本次容器。

## 评分、重试与统计

- 使用本地官方 `eval_acc.py` 的评分准则，采用 SDK structured parse 获取分析与整数评分；移除 EgoLife 的 Jake 元数据。参考答案仅供 judge 使用。
- 普通失败在本批结束后集中重试一次；维护错误修复后继续同方法重试，不计入普通失败次数，不删题缩小分母。
- 两边独立记录回答、子调用、judge 和重试的 token 消耗，通过各自 PD 计数与实际响应 usage 对账。
- W&B 每秒记录 token、吞吐、GPU 和共同完成题目的 overlap。SQL 先完成时继续更新 overlap，双方结束后再次汇总。
- 官方映射为：原始分数 4/5 → 1，3 → 0.5，0/1/2 → 0。最终两项指标固定使用 623 为分母：
  - `Rounded Score score-1 ratio = 映射为 1 的题数 / 623`
  - `Average rounded score = 映射分数总和 / 623`

## 验证、审查与交付

- 用小范围调试验证跨视频检索、同视频不同片段、零秒起点、小数秒、SQL 过滤、视频读取及旧 EgoLife 时间模式；不新增测试框架，不运行语法检查。
- 手动检查实际请求与返回、检索记录、embedding 对应片段及返回帧，确认视频路径可用、证据身份正确、reasoning 被保留，且回答路径未读到答案线索。
- 中等代码改动由新的 review subagent 按 `REVIEW.md` 审查；本次按用户决定跳过不可用的 Claude CLI，由独立 subagent 完成全仓审查。审查未正常完成并给出明确结论时，不启动正式运行。
- 保留已有工作区修改和删除，只提交本次源码、文档改动及紧凑最终结果；逐题输出、原始 traces 和二进制 artifacts 保持忽略。
- 核验双方完整结果、失败处理、执行 traces 和 token 对账后，只回填原计划文件中两种方法的四个结果字段。完成提交与服务收尾后，执行约定的 `5/0`。
