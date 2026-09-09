# Trace2Skill API 接入交接

工作目录 `/raid/wg25r/video_sql_2`，分支 `codex/trace2skill-baseline`。用户要求先提交保留本地支持的快照，再清理本地兼容；之后由另一位 agent 根据用户给出的 API 说明继续。

## 当前状态

- 本地支持快照：`60c84b62`（`Snapshot: Trace2Skill baseline with local vLLM NVFP4 support`）。包含最后的 max-num-seqs=20、MTP 2 配置；该配置没有恢复正式 rollout。
- 本次 rollout 和 PD、encoder、embedding、proxy 均已停止，四个容器已手动删除。
- 本地 vLLM/EPD Dockerfile、启动器、显存分析、NVFP4 padding 补丁和 Responses/token-count proxy 已移除。历史 INFRA、研究材料和 NVFP4 记录保留用于追溯，不是当前启动说明。
- 原方法的本地专用 `src/collect_local_pairs.py` 也已移除；历史 Luna API 采集入口 `src/collect_fixed_prefix.py` 保留，但 embedding URL 同样改为 `.env` 的 `EMBEDDING_URL`，不再硬编码本地端口。Trace2Skill 不调用该历史采集入口。
- API 接入尚待用户说明。本次没有发送模型请求，也没有启动新 run。

## 保留的 API 路径

`src/harness.py` 保留已有官方 OpenAI Responses 客户端。`src/trace2skill_baseline.py` 与 `third_party/trace2skill/baseline_evolve.py` 恢复此前的 `gpt-5.6-luna`、`https://api.openai.com/v1`、reasoning high、summary auto、`store=false` 和 encrypted reasoning 返回。这是已有 API 默认设置，不代表用户已指定下一次实验模型。

密钥仍从本 worktree 的 `.env` 读取 `PERSONAL_OPENAI_KEY`，未修改或输出密钥。baseline 的 embedding endpoint 改为同一 `.env` 中的必填 `EMBEDDING_URL`，不再默认 localhost。embedding 仍通过 `src/harness.py` 的独立 HTTP API 请求，当前请求未实现提供商专用鉴权；下一位 agent 按用户 API 文档接入。现有索引为 Qwen3-VL-Embedding-8B、4096 维，不能直接改用不匹配的向量模型。

接入时需覆盖问答/子 agent/judge、成功/失败分析，以及单独进程中的 evolution。涉及入口主要为：

- `src/harness.py`：Responses 客户端、reasoning、embedding 请求、工具和 trace。
- `src/trace2skill_baseline.py`：模型、endpoint、密钥、embedding URL、run 目录、成功分析 structured parse、replacement token counting。
- `third_party/trace2skill/baseline_evolve.py`：evolution 的同步 Responses structured parse 客户端及模型。
- `src/val.py`：独立评估 CLI，保留此前 Luna 默认值，embedding URL/model 为必填参数。

Replacement 计量仍直接使用 SDK `responses.input_tokens.count`，包含真实文本与按 recipe 重建的图像。没有本地代理或近似计数替代；下一位 agent 需要按用户选定 API 核对该能力。

## 保留的方法与数据

- day/test、week/test 各抽 100 题，seed 0，每题 5 条 rollout；一次 Trace2Skill Combined evolution 后冻结技能，在同一批题上复评。没有启动新实验。
- ROOT-SKILL、六份初始 subskill、catalog、`read_skill(name)`、0–5 judge、4–5 成功/0–3 失败、上游 map/merge/translation/apply/validation 保留。
- `clip_qa` 对 analysis 是固定工具，只暴露调用参数、返回答案和证据图；其内部 reasoning、工具轨迹和未返回帧不进入成功或失败分析。完整原 trace 仍保存。
- `replace_tool`、从视频动态重建帧的 JSON recipe、完整本地 trace、Weave 调用树、W&B 原有指标与恢复/重试规则保留。
- 保留客户端并发 256、HTTP connect 60 秒、其余 timeout 100000 秒、连接数 1000、keepalive 256，以及已有 harness context/compaction/650 张采样额度；这些不是下一提供商的容量声明，接入时按用户指示处理。
- 数据和索引仍读取原仓库 `data/runtime/day`、`data/runtime/week`；源视频位于 `/raid/huggingface`。

API 输出目录改为 `runs/trace2skill-baseline-api-day-week-seed0`，避免自动复用 NVFP4 结果；该目录尚未运行。旧 `runs/trace2skill-baseline-nvfp4-day-week-seed0` 保留 11 条已评分结果、共 13 条已保存答案及未完成 trace，日志为 `formal-01.log`。是否利用旧轨迹由用户决定，不自动混入新的 API run。

当前只做代码清理和独立 diff review，不运行 syntax、smoke、preflight 或正式实验。下一位 agent 先接收用户的 API 说明，再实施接入和执行仓库要求的正式运行前审查。
