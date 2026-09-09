# NVFP4 + Trace2Skill：day/week 新实验

工作目录为 `/raid/wg25r/video_sql_2`，分支 `codex/trace2skill-baseline`。本次从头生成轨迹，不复用 Luna rollout，也不使用 training split。代码已切换；新实验尚未正式启动。

## 数据与方法

- 数据集 revision：`248aa82039a574e63a2e524746a7cd8f32330443`，从 `day/test.json` 和 `week/test.json` 各抽 100 题。用户指定数量 100；seed 暂沿用 0。
- 每题 5 条独立 rollout，共 1,000 条初始轨迹；一次 Trace2Skill Combined evolution 后冻结技能，在同一批题上各做 5 条复评。
- 数据库、embedding 索引和 media-map 分别读取原仓库 `data/runtime/day/`、`data/runtime/week/`。输出路径与 trace metadata 包含 timeline，避免两种数据的题号冲突。
- ROOT-SKILL 和六份初始 subskill 保持原内容、catalog 和 `read_skill(name)`。0–5 judge、4–5 成功/0–3 失败、并行 patch、分层合并、translation、应用和验证保持不变；不调用本项目的技能训练方法。
- 上游固定为 `3d0b52a140f002a512930252b613c49048f7d5ac`。JSON 使用 SDK structured parse，保留纠错及重试耗尽后跳过整条样本的规则。

## 模型与服务

模型为 `Inferact/Qwen3.8-Flash-Next-NVFP4`，镜像为 `video-sql-trace2skill:nvfp4-padding-fix`。修复、实际权重检查和原始响应见 [NVFP4_TP4_RECHECK.md](NVFP4_TP4_RECHECK.md)。四条原始提示、结构化输出、计算工具往返、真实帧及图像工具往返已人工检查；这些短请求未出现循环，长 rollout 尚未验证。

答题、子 agent、compaction、judge、分析和 evolution 均通过 Responses API 使用 medium reasoning，保留返回的 reasoning content 并在后续请求重放。服务默认 `enable_thinking=true`、`preserve_thinking=true`，不请求 Luna 的 encrypted reasoning。

| 配置 | 当前启动脚本 |
| --- | --- |
| PD | GPU 0、1、2、3，TP4，max-num-seqs=50，MTP 3 |
| Encoder | GPU 4，分离编码，ECExampleConnector |
| Embedding | GPU 5，Qwen3-VL-Embedding-8B，与已有索引相同 |
| 端口 | EPD proxy 8003、PD 8024、embedding 8014，均绑定 localhost |
| 上下文 | 314573，YaRN factor 1.2 |
| KV | prefix cache、SimpleCPUOffloadConnector；启动时 MemAvailable 的 50% 均分给四个 TP rank |
| 通信 | custom all-reduce 开启，没有 `--disable-custom-all-reduce` |
| 图像 | 480p，max_pixels=414720，MM processor cache 4 GB，image limit 650 |
| 客户端 | 总请求并发 256，连接数 1000、keepalive 256；connect 60 秒，其余 100000 秒 |

模型 processor 的 patch size 16、merge size 2，因此单图最多 405 个视觉 token，加两个边界 token 为 407；`floor((314573−50000)/407)=650`，现有累计图像额度无需改变。

新服务使用独立的 `video-sql-trace2skill-nvfp4` 网络、`video-sql-trace2skill-nvfp4-ec` encoder-cache volume 和 `trace2skill-nvfp4-*` 容器。正式运行前需要创建该网络与 tmpfs volume，并检查实际 GPU 资源。各角色通过 `.venv/bin/python -u infra/trace2skill_service.py <encoder|embedding|pd|proxy>` 在独立前台 exec 启动，不能使用 Docker `--rm`。新的 EPD 组合尚未启动验证。

## 输出、指标和恢复

入口为 `.venv/bin/python -u -m src.trace2skill_baseline`；新目录为 `runs/trace2skill-baseline-nvfp4-day-week-seed0/`，题目抽样保存到 `questions.json`。恢复只复用本次新目录的完成结果、答案、跳过记录和 evolution 中间产物。旧 Luna 目录保留独立。

W&B 保留平均 trace 长度、输入/输出 token 与请求吞吐、average rollout time、mean total context saved、replace_tool 调用次数；统计口径见 [Luna 历史指标说明](TRACE2SKILL_LUNA_HANDOFF.md#wandbweave-和本地证据)。Weave 和本地 trace metadata 增加 timeline。

Replacement 计量仍走 SDK `responses.input_tokens.count`，由本地 proxy 使用 vLLM 原生 Responses 转换函数及 `/tokenize` 完成。before/after 均加上相同的实际题目文本，以满足 Qwen chat template；公共部分在差值中抵消。真实文本工具往返的计数为 403，图像工具往返为 778，均与同一请求实际生成返回的 input_tokens 一致；检查记录在 `/tmp/trace2skill-nvfp4-token-count-check.json`。尚未验证完整 baseline 内的 replacement 路径。

## 当前阻塞

`/raid` 已接近满盘，最近检查只剩约 39 MB；正式轨迹和媒体不能开始写入。等待用户指定有足够空间的输出位置，或明确指定可清理的旧目录；没有删除旧产物，也没有迁移正式输出。

本次接入改动的独立 diff review 尚未完成：自动审批将审查子 agent 误判为主 agent，拒绝其读取 `REVIEW.md`；已请求明确授权。正式运行前仍需完成 diff review 和仓库要求的 Claude CLI 全量审查。没有运行 syntax checks。之后从正式 trace 检查真实工具输入输出、图像、多轮 reasoning、offload、Weave 父子关系，以及是否出现循环。
