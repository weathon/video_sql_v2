# EgoLifeQA：MAGIC 与 SQL 无训练基线

本 worktree 比较 A1_JAKE 全部 500 道开放题。实现尚在验证中，正式结果尚未生成。源分支起点为 `f947154c0dc1a6825bdc56f85d7ba4bf294411cb`；MAGIC 上游版本见 [UPSTREAM](third_party/magic_video/UPSTREAM)。

两边所有 LLM 请求均使用 `Qwen/Qwen3.8-27B-FP8`、`xhigh` 并保留 reasoning，不设置客户端 token 上限或采样参数。SQL 每次尝试最多 10 次共享调查调用，再允许一次关闭工具的最终回答；MAGIC 保留上游 5 轮检索及截止逻辑。两边使用同一个 0–5 分 judge。每批失败题只集中重试一次。

## 环境与 artifacts

本机目录为 `/raid/wg25r/video_sql_baselines`，独立环境为 `.venv_baselines`，`.env` 链接原项目配置。W&B/Weave 继续使用项目配置。HF cache 位于 `/raid/huggingface`。

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

按仓库流程完成独立 diff review 和正式运行前的 Claude 全仓库 review。服务全部就绪后测量可用内存，以下前台 scope 将 harness 限制为该值的 20%；两种质量评测并发均为 20。

```bash
export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 NUMEXPR_NUM_THREADS=10 VECLIB_MAXIMUM_THREADS=10
baseline_memory_bytes=$(awk '/^MemAvailable:/ {printf "%.0f", $2 * 1024 / 5}' /proc/meminfo)
systemd-run --user --scope -p MemoryMax="$baseline_memory_bytes" .venv_baselines/bin/python -m training_free.evaluate --method magic --run-dir /raid/videosql-video-artifacts/results/quality-magic > runs/egolife-baselines/quality-magic.log 2>&1
systemd-run --user --scope -p MemoryMax="$baseline_memory_bytes" .venv_baselines/bin/python -m training_free.evaluate --method sql --run-dir /raid/videosql-video-artifacts/results/quality-sql > runs/egolife-baselines/quality-sql.log 2>&1
systemd-run --user --scope -p MemoryMax="$baseline_memory_bytes" .venv_baselines/bin/python -m training_free.evaluate --method latency --run-dir /raid/videosql-video-artifacts/results/latency > runs/egolife-baselines/latency.log 2>&1
```

恢复时使用原命令和原 `--run-dir`。小范围真实调试可附加 `--ids 1 2`。正式时延读取 seed 0 的同一组 20 个 ID，按固定顺序、并发 1，先 MAGIC 后 SQL；每种方法开始前清 prefix/KV 和 encoder cache，每题创建独立在线查询状态，保留离线索引。时延截止于最终答案，不包含 judge 或关闭客户端。

每题答案、judge、失败证据和全部 attempts 均即时落盘；原始 usage 与独立 PD 服务计数对账。`metrics.json` 保存均分、5 分题数、有效分母和两类失败数，`token-log.json` 保存全部实际请求及服务计数。缓存输入从总输入中扣除后，等效费用为 `(in × 0.15 + out × 0.47 + cache × 0.016) / 1,000,000 USD`。缺 usage 的请求保留未知状态，不能当作零。

完成后上传 artifacts，并停止及手动删除本任务容器：

```bash
.venv_baselines/bin/python -m training_free.publish_artifacts > runs/egolife-baselines/upload.log 2>&1
docker stop egolife-baselines-proxy egolife-baselines-embedding egolife-baselines-encoder egolife-baselines-pd
docker rm egolife-baselines-proxy egolife-baselines-embedding egolife-baselines-encoder egolife-baselines-pd
```

Git 只保存源码、说明和紧凑最终结果；视频、数据库、向量、逐题结果及 traces 保持忽略。
