# NVFP4 + Trace2Skill：day/week 新实验

工作目录为 `/raid/wg25r/video_sql_2`，分支 `codex/trace2skill-baseline`。本次从头生成轨迹，不复用 Luna rollout，也不使用 training split。新实验已启动，正在生成 initial rollout；前台日志为新 run 目录中的 `formal-01.log`。

## 数据与方法

- 数据集 revision：`248aa82039a574e63a2e524746a7cd8f32330443`，从 `day/test.json` 和 `week/test.json` 各抽 100 题。用户指定数量 100；seed 暂沿用 0。
- 每题 5 条独立 rollout，共 1,000 条初始轨迹；一次 Trace2Skill Combined evolution 后冻结技能，在同一批题上各做 5 条复评。
- 数据库、embedding 索引和 media-map 分别读取原仓库 `data/runtime/day/`、`data/runtime/week/`。输出路径与 trace metadata 包含 timeline，避免两种数据的题号冲突。
- ROOT-SKILL 和六份初始 subskill 保持原内容、catalog 和 `read_skill(name)`。0–5 judge、4–5 成功/0–3 失败、并行 patch、分层合并、translation、应用和验证保持不变；不调用本项目的技能训练方法。
- 上游固定为 `3d0b52a140f002a512930252b613c49048f7d5ac`。JSON 使用 SDK structured parse，保留纠错及重试耗尽后跳过整条样本的规则。

## 模型与服务

模型为 `Inferact/Qwen3.8-Flash-Next-NVFP4`，镜像为 `video-sql-trace2skill:nvfp4-padding-fix`。修复、实际权重检查和原始响应见 [NVFP4_TP4_RECHECK.md](NVFP4_TP4_RECHECK.md)。四条原始提示、结构化输出、计算工具往返、真实帧及图像工具往返已人工检查；这些短请求未出现循环，长 rollout 尚未验证。诊断容器已停止（退出码 0）并手动删除；完整服务日志保留在 `/tmp/trace2skill-nvfp4-tp4-padding-fixed-server.log`，停止状态保留在 `/tmp/trace2skill-nvfp4-padding-fixed-stop.json`，镜像和请求产物保留。

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

新服务使用独立的 `video-sql-trace2skill-nvfp4` 网络、`video-sql-trace2skill-nvfp4-ec` encoder-cache volume 和 `trace2skill-nvfp4-*` 容器。各角色通过 `.venv/bin/python -u infra/trace2skill_service.py <encoder|embedding|pd|proxy>` 在独立前台 exec 启动，不能使用 Docker `--rm`。等待分析边界决定时，第一轮服务已停止并手动删除容器，记录保存在 `service-stop.json`。边界修改审查完成后，encoder、embedding、TP4 PD 和 proxy 已重新启动，日志分别为 `encoder-launch2.log`、`embedding-launch2.log`、`pd-launch2.log`、`proxy-launch2.log`。PD 实际参数为 max-num-seqs=50、thinking/preserve_thinking、MTP 3 和 CPU offload，每 rank 分配 256674663936 字节 CPU KV 预算；原生启动报告 GPU KV 容量为 7,407,246 tokens、满长并发 23.55。未发送预检请求，正式 rollout 的真实图像请求已完成 EPD 往返，日志已有 external prefix cache 命中。

## 输出、指标和恢复

入口为 `.venv/bin/python -u -m src.trace2skill_baseline`；新目录为 `runs/trace2skill-baseline-nvfp4-day-week-seed0/`，题目抽样保存到 `questions.json`。恢复只复用本次新目录的完成结果、答案、跳过记录和 evolution 中间产物。旧 Luna 目录保留独立。

W&B 保留平均 trace 长度、输入/输出 token 与请求吞吐、average rollout time、mean total context saved、replace_tool 调用次数；统计口径见 [Luna 历史指标说明](TRACE2SKILL_LUNA_HANDOFF.md#wandbweave-和本地证据)。Weave 和本地 trace metadata 增加 timeline。

图像已改为从源视频动态生成：`media/frame-*.json` 保存视频路径、整数 PTS、time base 和 crop；采样通过内存中的 PNG 流解码，主 agent 证据、成功分析、失败分析 `read_frame` 和 replacement 计量均通过取图记录生成内存中的 JPEG，不再保存 PNG/JPG 文件。分析仅重建调用方收到的证据帧，不读取 video QA 内部未返回的采样或裁剪帧。PyAV 依赖已加入并锁定。一次真实 `sample_frames` 返回的 240 帧与旧 PNG 生成的 JPEG 全部逐字节一致，首、中、末帧动态重取也一致，输出仅包含取图记录和 ffmpeg 命令/日志；记录在 `/tmp/video-sql-source-frame-check/verification.json`。

两个 runs 中的 889,065 份旧帧 JSON 已补齐源视频路径，保留原有 PTS、time base、crop 和其他字段。已逐份核对迁移后的路径；day、week、month、裁剪及归档轨迹的五份真实记录重建结果均与原图处理结果逐字节一致，PNG 删除后又重取了裁剪帧并确认一致。记录在 `/tmp/video-sql-png-migration-verified.json`、`/tmp/video-sql-migrated-frame-replay.json` 和 `/tmp/video-sql-post-cleanup-frame-replay.json`。独立 diff review 已完成。

Replacement 计量仍走 SDK `responses.input_tokens.count`，由本地 proxy 使用 vLLM 原生 Responses 转换函数及 `/tokenize` 完成。before/after 均加上相同的实际题目文本，以满足 Qwen chat template；公共部分在差值中抵消。真实文本工具往返的计数为 403，图像工具往返为 778，均与同一请求实际生成返回的 input_tokens 一致；检查记录在 `/tmp/trace2skill-nvfp4-token-count-check.json`。尚未验证完整 baseline 内的 replacement 路径。

## 清理与审查状态

用户明确授权清理 `/raid/wg25r/video_sql_2/runs` 与 `/raid/wg25r/video_sql_v2/runs` 下全部 JPEG，随后同意 PNG 改为从源视频动态重建并清理。JPEG 已清理；此次再删除 926,127 个 PNG，释放约 835.6 GiB 已分配空间。两个目录复查均剩余 0 个 `.png/.jpg/.jpeg`，889,065 份帧 JSON、ffmpeg 命令/日志、trace 和源视频保留。未生成独立帧 JSON 的旧 PNG 也已核对其 ffmpeg 命令和源视频，仍可按保存的采样命令重建。检查时 `/raid` 可用约 1.16 TiB。

Docker root 创建的文件通过仅挂载这两个 runs 目录、无网络的容器处理；迁移与清理容器均退出 0 并已手动删除。记录在 `/tmp/video-sql-jpeg-cleanup-verified.json`、`/tmp/video-sql-png-reconstruction-audit.json`、`/tmp/video-sql-png-cleanup-docker.log` 和 `/tmp/video-sql-png-cleanup-verified.json`。原仓库源代码未修改。

用户已明确允许审查者读取 `REVIEW.md`。新的独立子 agent 已完成 `3f2f0730..f3ef7bb6` 的相关源码、配置、依赖和文档 diff review；Claude CLI 也完成全仓库审查，未触及 usage limit，报告位于 `/tmp/trace2skill-nvfp4-full-claude-review.json`。

两轮审查指出原成功分析将整条 rollout 的全部图片一次发送，可能超过本地服务单请求 650 张上限。用户随后明确：analysis 不进入 video QA 内部，只看 video QA 的结果，将其作为静态工具。代码已据此修改：成功和失败分析均排除 `clip-qa` 内部 trace，保留调用参数、返回答案和证据帧；分析图片列表从调用方工具结果提取，失败分析的 `read_trace`/`read_frame` 也只允许读取这一范围。失败分析仍可调用固定的 `clip_qa` 验证修正，只使用其返回结果。完整原始轨迹与帧记录继续留存，W&B 指标仍计入重试和子 agent。

已人工核对旧 Luna 成功样本 `question-86/rollout-02`：共 995 份帧记录，但 `clip_qa` 实际只返回 8 张证据图；分析保留这 8 张、工具参数和完整返回答案，排除三个视觉内部 trace。记录在 `/tmp/trace2skill-opaque-videoqa-evidence-audit.json`。本次边界修改 `71d791bb` 的独立 diff review 已完成，无阻塞项；随后 Claude CLI 已完成新边界下的全仓库审查，报告在 `/tmp/trace2skill-opaque-videoqa-full-claude-review.json`。

Claude 提出的失败分析可见 reference answer 的意见未采用：已核对上游 `analysis/error_analysis_system.txt`，其失败分析明确读取 ground-truth solution，并用 `evaluate_output` 验证最小修正；禁止在学习报告中提及 gold 的要求也同时存在。当前 QA 桥接保留这一分析方式，答题 agent 仍不接收 reference answer。

没有运行 syntax checks。审查已完成，继续正式运行；从正式 trace 检查真实工具输入输出、图像、多轮 reasoning、offload、Weave 父子关系，以及是否出现循环。
