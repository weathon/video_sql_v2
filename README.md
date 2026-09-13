# EgoLifeQA：MAGIC 与 SQL 无训练基线

本目录的 `codex/egolife-baselines` 分支比较 A1_JAKE 全部 500 道开放题。修复实时日志后，两种方法已重新完成质量评测和串行计时；完整结果见 [结果报告](notes/egolife-baselines-results.md) 和 [机器可读结果](notes/egolife-baselines-results.json)。源分支起点为 `f947154c0dc1a6825bdc56f85d7ba4bf294411cb`；MAGIC 上游版本见 [UPSTREAM](third_party/magic_video/UPSTREAM)。

两边所有 LLM 请求均使用 `Qwen/Qwen3.8-27B-FP8`、`xhigh` 并保留 reasoning，不设置客户端 token 上限或采样参数。SQL 当前每次尝试最多 60 次共享调查调用，再允许一次关闭工具的最终回答；原结果使用 10 次预算，SQL30 和 SQL+PPR15 已由用户停止。MAGIC 保留上游 5 轮检索及截止逻辑，时间检索修正日期和秒数单位并应用 query time 截止，chain 相似度检索保留上游前 2000 字符规则。两边使用同一个 0–5 分 judge。每批失败题只集中重试一次。

原 SQL60 结果已记录，复现该方法使用源码 `783d3101c1be134deb53e5f8a7be9a1aa9fbc5ee`。当前下一阶段保留 `run_sql`、`clip_qa`、`replace_tool` 和静态 skills，将 `semantic_search` 替换为支持 pre/post SQL 的 MAGIC PPR。它仅在需要按语义匹配时使用，不设为默认或优先工具；没有单独的 `sql_ppr_search` 工具。pre SQL 从 `memory_nodes` 选择候选 ID，post SQL 查询 PPR 返回的 `ppr_hits`，通常按目的选择其中一步。

新版 judge 接收提问者 Jake/A1_JAKE 和 query time，按问题要求的事实判分，接受同义表达及对应的绝对/相对时间，不因正确补充、引用或篇幅扣分；扣分须在结构化 justification 中指出实际事实错误或必要信息遗漏。详细规则见 `baselines_plan.md` 和 `training_free/evaluate.py` 的 `RUBRIC`。

## 环境与 artifacts

本机目录为 `/raid/wg25r/video_sql_trace2skill`，独立环境为 `.venv_baselines`，使用本目录的 `.env`。W&B/Weave 继续使用项目配置。HF cache 位于 `/raid/huggingface`。

重建本次使用的 CPU 检索环境（模型推理由独立容器提供）：

```bash
uv venv .venv_baselines --python 3.13
uv pip install --python .venv_baselines/bin/python --index https://download.pytorch.org/whl/cpu --index https://pypi.org/simple --index-strategy unsafe-best-match -r training_free/requirements.txt
```

所有数据位于 `/raid/videosql-video-artifacts`。目标 private dataset 为 [weathon/egolife-a1-jake-magic-videosql-qwen3vl8b](https://huggingface.co/datasets/weathon/egolife-a1-jake-magic-videosql-qwen3vl8b)；上传完成状态记录在 artifacts 根目录的 `upload-result.json`，不能仅凭此链接判断已上传。目录、字段和查询示例见 [索引说明](notes/index.md)。

在另一个已安装本项目依赖的环境中，使用获准访问该 private repo 的项目 `HF_TOKEN` 下载普通 artifact 文件：

```python
from pathlib import Path
from dotenv import dotenv_values
from huggingface_hub import snapshot_download

token = dotenv_values(Path.cwd() / ".env")["HF_TOKEN"]
snapshot_download(repo_id="weathon/egolife-a1-jake-magic-videosql-qwen3vl8b", repo_type="dataset", token=token, cache_dir="/raid/huggingface", local_dir="/raid/videosql-video-artifacts")
```

从指定现成 artifacts 重建时，按顺序运行下面的 stage。`prepare` 保留原始键完成 caption/OpenIE 关联后再规范化；不生成新 caption、三元组或 chain。`embed-text` 和 `embed-video` 需要先启动 embedding 服务。

```bash
export HF_HOME=/raid/huggingface
export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 NUMEXPR_NUM_THREADS=10 VECLIB_MAXIMUM_THREADS=10
.venv_baselines/bin/python -m training_free.prepare_baselines prepare
.venv_baselines/bin/python -m training_free.prepare_baselines media
.venv_baselines/bin/python -m training_free.prepare_baselines embed-text
.venv_baselines/bin/python -m training_free.prepare_baselines embed-video
.venv_baselines/bin/python -m training_free.prepare_baselines assemble
.venv_baselines/bin/python -m training_free.prepare_baselines databases
```

## 独立本地服务

启动前核对 GPU 4、5、6、端口和主机内存。资源不足时停止，不等待、不降低实验规格。PD 使用 GPU 4、5，TP=2、最多 50 条并行序列，CPU KV cache 合计 256 GiB、每 rank 128 GiB；encoder 和 embedding 共用 GPU 6。具体启动参数集中在 `infra/trace2skill_service.py`。

```bash
nvidia-smi
free -h
ss -ltn
docker build -f infra/Dockerfile.baselines -t video-sql-vllm-epd:egolife-baselines infra
docker network create egolife-baselines
docker volume create --driver local --opt type=tmpfs --opt device=tmpfs egolife-baselines-ec
mkdir -p runs/egolife-baselines/services
```

以下四条命令分别放在四个受控前台会话中运行，保留会话直到退出。服务 readiness 通过 `/health` 和真实请求检查；日志保存在各自文件。

```bash
.venv_baselines/bin/python infra/trace2skill_service.py pd > runs/egolife-baselines/services/pd.log 2>&1
.venv_baselines/bin/python infra/trace2skill_service.py encoder > runs/egolife-baselines/services/encoder.log 2>&1
.venv_baselines/bin/python infra/trace2skill_service.py embedding > runs/egolife-baselines/services/embedding.log 2>&1
.venv_baselines/bin/python infra/trace2skill_service.py proxy > runs/egolife-baselines/services/proxy.log 2>&1
```

端口依次为 8124、8125、8114、8103。`embed-video` 按 caption 的明确时间区间生成缓存片段，通过本地文件 URL 请求 Qwen loader 均匀采样，保留原始顺序并禁用再次采样。两边共享同一视频向量；MAGIC 的视频证据也读取对应缓存片段。原视频留给按实际时间取证的 `clip_qa`。embedding 服务日志中的 `UNIFORM_DECODE` 记录每段视频的实际抽帧序号。

## 正式评测

正式评测前完成仓库要求的独立 subagent 审查和 `claude -p` 全目录审查。服务全部就绪后测量可用内存，以下前台 scope 将 harness 限制为该值的 20%；质量评测及重新 judge 的并发均为 20。

```bash
export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 NUMEXPR_NUM_THREADS=10 VECLIB_MAXIMUM_THREADS=10
baseline_memory_bytes=$(awk '/^MemAvailable:/ {printf "%.0f", $2 * 1024 / 5}' /proc/meminfo)
systemd-run --user --scope -p MemoryMax="$baseline_memory_bytes" .venv_baselines/bin/python -m training_free.evaluate --method magic --answers-from /raid/videosql-video-artifacts/results/quality-magic-rerun --run-dir /raid/videosql-video-artifacts/results/quality-magic-judge-v2 > runs/egolife-baselines/quality-magic-judge-v2.log 2>&1
systemd-run --user --scope -p MemoryMax="$baseline_memory_bytes" .venv_baselines/bin/python -m training_free.evaluate --method sql --answers-from /raid/videosql-video-artifacts/results/quality-sql-rerun --run-dir /raid/videosql-video-artifacts/results/quality-sql10-judge-v2 > runs/egolife-baselines/quality-sql10-judge-v2.log 2>&1
systemd-run --user --scope -p MemoryMax="$baseline_memory_bytes" .venv_baselines/bin/python -m training_free.evaluate --method sql --run-dir /raid/videosql-video-artifacts/results/quality-sql60-ppr-semantic-judge-v2 > runs/egolife-baselines/quality-sql60-ppr-semantic-judge-v2.log 2>&1
```

恢复时使用原命令和原 `--run-dir`；带有 `INVALIDATED.json` 的旧结果目录禁止恢复。小范围真实调试使用独立目录并附加 `--ids 1 2`。正式时延读取 seed 0 的同一组 20 个 ID，按固定顺序、并发 1，先 MAGIC 后 SQL；每种方法开始前清 prefix/KV 和 encoder cache，每题创建独立在线查询状态，保留离线索引。时延截止于最终答案，不包含 judge 或关闭客户端。

每题答案、judge、失败证据和全部 attempts 均即时落盘；原始 usage 与独立 PD 服务计数对账。`metrics.json` 保存均分、5 分题数、有效分母和两类失败数，`token-log.json` 保存全部实际请求及服务计数。缓存输入从总输入中扣除后，等效费用为 `(in × 0.15 + out × 0.47 + cache × 0.016) / 1,000,000 USD`。缺 usage 的请求保留未知状态，不能当作零。

运行开始即以 1 Hz 拉取 PD `/metrics`，将累计 `tokens/in`、`tokens/cache`、`tokens/out`、相应每秒吞吐率、GPU 使用率和进度上传 W&B，同时写入 `live-metrics.jsonl`；原始抓取保存在 `live-metrics.prom`。实时记录不等待单次请求的 usage 或整批对账完成。进度计数表示本次进程的尝试次数，包含重试；累计 token 从本 run 的首次采集计起。

SQL 每题成功回答后立即 judge。W&B 的 `overlap/count`、`overlap/sql_accuracy`、`overlap/magic_accuracy`、`overlap/accuracy_delta` 和两边均分，比较当前已完成且可评分的共同题目；准确率定义为 5 分比例，MAGIC 参考来自 `results/quality-magic-judge-v2`。未完成或未分类失败不记作 0 分，维护失败排除。最终同题 ID 和指标保存在 `overlap-comparison.json`。交错答题和 judge 的服务端总 token 记为 `answer_and_judge` 混合阶段，原始请求 usage 另行完整保留。

最终失败题先保存自动建议分类，人工阅读其真实请求、工具输入输出及错误 trace 后，在该题目录写入 `failure-review.json`，包含 `category`（`algorithm` 或 `maintenance`）、`cause` 和 `trace_evidence`（实际检查的路径、调用及结论）。未审阅的失败会阻止最终汇总；审阅后用原命令恢复，已完成的答题和 judge 不重跑。

完成后上传 artifacts，并停止及手动删除本任务容器：

```bash
.venv_baselines/bin/python -m training_free.publish_artifacts > runs/egolife-baselines/upload.log 2>&1
docker stop egolife-baselines-proxy egolife-baselines-embedding egolife-baselines-encoder egolife-baselines-pd
docker rm egolife-baselines-proxy egolife-baselines-embedding egolife-baselines-encoder egolife-baselines-pd
```

Git 只保存源码、说明和紧凑最终结果；视频、数据库、向量、逐题结果及 traces 保持忽略。
