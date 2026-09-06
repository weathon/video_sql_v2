# video_sql_v2：项目说明与接手记录

## Training-free 实施状态（2026-09-06）

当前分支为 `codex/training-free-harness`，实施计划见 [TRAINING_FREE_PLAN.md](../TRAINING_FREE_PLAN.md)，当前协议见 [PLANS.md](../PLANS.md)。

已删除训练入口、reflection 和 META-SKILL；答题、judge、静态初始 skills、W&B/Weave、逐 rollout 结果和恢复保留。所有 LLM 角色使用 OpenAI 官方 Responses API、项目 .env 的 PERSONAL_OPENAI_KEY、gpt-5.6-luna、medium reasoning，并保留多轮 reasoning。每批 30 题 × 5 rollouts，合计 150 并发。Embedding 继续使用现有 Qwen3-VL-Embedding-8B API 和索引。

已实现成功提交优先于 compaction、视觉提交校验错误继续 loop、SQL 同名列报错、sql_qa 全量结果、judge 原始问题输入。视觉输入改为最大 854×480 的独立帧，保存真实源 PTS 和原始分辨率帧，支持单帧裁剪；每次 clip_qa 累计 1,024 张图，证据复用已观察帧。

用户补充要求完整 trace：各阶段保存指令、模型设置、工具 schemas、输入输出、reasoning、usage、错误和逐事件 JSONL；图像独立保存并记录源帧与裁剪映射。完整 trace 实现提交为 `b0097c66`。

新的独立 subagent 已完成 `b4799309..cb5ad4e7` 的指定文件 diff review，结论为 `no blocking issues`。正式运行前 Claude CLI 全仓 review 同样无阻塞项。用户已有修改、删除、历史 runs 和数据均保留。

正式 run 为 `runs/training_free_luna_seed0_7510e535`，W&B ID 为 `tityz8z2`。首次运行的 53 个余额不足失败项已在用户充值后补跑完成。最终 1,500 个 rollout 全部 answered、failed=0，平均分 1.98/5（39.60%），score>4 为 442/1500（29.47%）。已从真实 traces 核对检索、采样画面、帧引用、无效视觉提交纠正、图片累计额度、sql_qa 全量输入和 judge 问题语境。完整本地 traces 与媒体保留；Weave 出现单条上传过大的警告，远端完整性未逐条核实。

运行后局部修复 `5f4eb63f` 使余额不足直接终止并显示原始异常链，该代码已用于本次补跑，但未再次触发余额不足分支。补跑代码版本为 `5cfcb519`，日志为 eval-resume-1.log；旧失败目录和汇总保留在 run 的 credit-exhaustion-archive-7510e535 目录。补跑期间在 GPU 5 创建的临时 embedding 容器已停止并删除，正式进程已退出，没有待补跑项或仍在运行的前台任务。实际命令、审查报告、具体 trace 证据和补跑完成记录见 [TRAINING_FREE_PLAN.md](../TRAINING_FREE_PLAN.md) 第 7、8 节。

以下内容是停止训练时的历史交接，不作为当前模型、judge、并发或启动指令。

记录日期：2026-09-05（America/Los_Angeles）。停止训练的 UTC 时间为 2026-09-06 00:05:15。

## 1. 接手时最重要的状态

**训练已经按用户要求停止。不要自行重新开始训练。** 用户说“先停止训练，我发现一些bug”，随后要求写本文档并重开会话。用户尚未描述这些新 bug；下一会话先处理用户提供的问题。此前“继续监控”的授权不能覆盖最新停止指令。

- 工作目录：`/raid/wg25r/video_sql_v2`。
- 写本文档前主仓库 HEAD：`1cb440ee`。本文档会单独提交。
- 当前训练容器：`video-sql-qwen38-27b-fp8-fresh-3x10-full-client`。
- 实际停止结果：`status=exited running=false exit_code=137`。这是本次 `docker stop --time 30` 后的状态，不是一次新发现的训练自然崩溃。
- encoder、prefill/decode、proxy 和 embedding 服务仍运行；本次只停止训练容器。
- 正式 run：`runs/qwen38_27b_fp8_epd_fresh_3x10_seed0`。
- 已落盘 batch 结果到 `train/epoch-000/batch-020/batch.json`；`batch-021` 已有部分 rollout 结果，尚无 batch 完成文件。
- 当前没有需要继续执行的监控目标。旧会话的工具 session ID 不可作为新会话的进程状态证据。

## 2. 项目做什么

用 MM-Lifelong 的视频、字幕/描述及检索数据库回答问题，并通过答题后的 reflection、同题汇总和 updater 迭代检索技能文档。这里的“训练”是技能文档更新循环，不是模型权重反向传播。

主要流程：问题 → 检索/看视频 → 提交答案 → judge 评分 → 在原 Agent 和原会话中 reflection → 同题多个 rollout 汇总 → updater 修改技能并提交 → 保存 batch 和技能快照。

| 文件或目录 | 用途 |
| --- | --- |
| `PLANS.md` | 实验约定；之后的用户指令优先 |
| `INFRA.txt` | 当前 EPD、embedding 部署命令 |
| `src/harness.py` | 工具、视频采样、trace、Agent loop、compaction、reflection |
| `src/training.py` | rollout/judge/reflection 重试、汇总、updater、恢复、W&B/Weave |
| `src/val.py` | 验证接口 |
| `src/prepare_runtime.py`, `src/normalize.py` | 运行时数据准备与规范化；不要因接手而重建数据 |
| `infra/responses_epd_proxy.py` | Responses 请求先预编码图像，再转发 PD；缓存清理接口 |
| `infra/Dockerfile.vllm-epd` | ExampleConnector 在 TP worker 的当前 CUDA device 上加载 tensor 的补丁 |
| `skills/initial/` | 初始技能，包括 `ROOT-SKILL.md`、`META-SKILL.md` |
| `<run>/skills/` | 正在迭代的独立技能 Git 仓库 |
| `<run>/skill-snapshots/` | 各批更新后的技能快照 |
| `<run>/train/epoch-XXX/batch-XXX/` | rollout、trace、summary、updater 和 batch 结果 |
| `notes/` | 事故报告与历史记录；不能把旧记录当作当前状态 |
| `vis/` | 历史 trace 网页与导出器 |

## 3. 数据路径：绝对不能重新下载

数据集与视频已存在于：

```text
/raid/huggingface/hub/datasets--MM-Lifelong--MM-Lifelong/snapshots/248aa82039a574e63a2e524746a7cd8f32330443/
```

媒体路径是相对数据集根目录的路径；不要自行假定所有视频在一个额外的 `videos/` 目录中。按实际 media-map 和文件内容解析。HF cache 是 `/raid/huggingface`，不是 home。

当前运行时数据根目录：`data/runtime`。训练源码读取：

- `month/index.duckdb`
- `month/embedding_index`
- `month/media-map.json`

训练问题来自数据集根目录下 `month/train.json`。不要重下载、重新索引、删除数据或覆盖旧 run。

## 4. 当前实验协议

- 模型：`Qwen/Qwen3.8-27B-FP8`。不要退回 NVFP4 或 Flash；之前用户明确改为 FP8。
- batch：**3 questions × 10 rollouts each**；3 epochs；seed 0；first-epoch-skip 0。
- TP=2，MTP speculative length=3。
- agent reasoning effort=`medium`；保留 `enable_thinking=true`、`preserve_thinking=true` 与 Responses reasoning items。
- harness context limit=262144；服务端 max-model-len=314573，官方 YaRN factor=1.2。
- 训练并发 semaphore=24。不要把它误改为 batch 大小。
- `--attempts 3` 分别用于 rollout、judge、reflection、summarizer、updater；SDK 内部 max_retries=3。
- judge 只接收 reference answer 和 candidate answer；正式 judge 同样使用当前 Qwen。
- `train/fully_correct_portion`：**分数严格 >4 的 rollout 数 / batch 全部 rollout 数**。不是 mean score/5，也不是 >=4。失败 rollout 不进入分子。
- batch 的 `mean_score` 当前按 successful 结果计算，分母与 fully_correct_portion 不同。
- 当前正式启动命令没有 `--max-updates`，不是只跑几个 batch 的 pilot。
- 计划中的初始/训练后测试是各 300 题 × 5 rollout，只答题和 judge；本轮没有执行完成这项评估。

已经实现的用户要求，不能在修 bug 时退回旧方案：

1. Reflection 在答题的同一个 Agent instance、同一会话继续，不能 clone 或序列化 trace 给独立 reflection agent。
2. 答题和 reflection 各允许 3 次 compaction；reflection 第三次 compact 后仅允许提交反思。Handoff/instructions 按现有原 Agent 会话实现维护。
3. Reflection 至少完成 2 次工具调用才可提交；不计答题阶段调用和 submit_reflection 自身。
4. Updater 采用建议性的小范围更新语言，不是硬性 patch 长度门槛，也不是数值优化器。
5. `ROOT-SKILL.md` 内容直接注入，指导何时加载其他技能，并由 updater 维护。
6. Reflection 必须通过实际检索证据找到正确答案才可写入已证实经验；仅获得 reference 不算验证。未找到时只提极小、明确不确定的可能错误原因 patch，由 merger/updater 决定处理。

第 6 条目前是 prompt/meta-skill/汇总与 updater 指令约束，不是额外的“成功证据”自动裁判。不能仅因 run 完成就宣称模型遵守。

## 5. 当前部署与镜像/host 的关系

| 容器 | 状态（本次交接检查） | 资源/地址 |
| --- | --- | --- |
| `video-sql-qwen38-27b-fp8-fresh-3x10-full-client` | 已停止，exit 137 | 正式训练 client |
| `qwen38-27b-fp8-epd-encoder` | 运行中 | GPU 5，内部 8000 |
| `qwen38-27b-fp8-epd-pd` | 运行中 | GPU 0,2，内部 8000 |
| `qwen38-27b-fp8-epd-proxy` | 运行中 | 主机 8002 → 内部 8000 |
| `egolife-qwen3-vl-embedding-8b` | 运行中 | GPU 4，主机 8013 |

网络：`video-sql-v2`。其余 GPU/容器属于共享主机环境，不能广泛 stop/kill。

训练镜像：`video-sql-training:cache-cleanup-a0be03dc`，构建得到短 ID `f00b514d45d0`。训练源码烘焙在镜像中，**host 的 src 修改不会自动进入已有训练容器**。最近构建方式如下（仅记录，接手不要自动执行）：

```bash
docker build -f - -t video-sql-training:cache-cleanup-a0be03dc src <<'DOCKERFILE'
FROM video-sql-training:e3e4ec70
COPY . /workspace/video_sql_v2/src
DOCKERFILE
```

Proxy 的 `infra/` 是 host 只读 bind mount；Python 已加载的代码仍需进程重启才生效。EPD 镜像是 `video-sql-vllm-epd:ec-tp-device`。完整服务命令在 `INFRA.txt`，不要再次无条件创建已有 network/volume/container。

重要继承配置：PD 有 SimpleCPUOffloadConnector，每 rank `cpu_bytes_to_use_per_rank=236223201280`、`lazy_offload=false`；max-num-seqs=24、GPU utilization=0.90；encoder utilization=0.45。不要仅查新增参数而忽略这些继承设置。embedding 的 `--mm-processor-cache-gb 0` 不应被误认为主模型设置。

旧失败容器和旧服务保留了 `-before-cache-cleanup` 后缀，便于读取事故日志；未删除。不要用 `--rm` 启动容器。

## 6. 本轮缓存事故、修复及证据

### 原始事故

自动清理前，ExampleConnector 将 encoder tensor 写入共享 tmpfs，但没有 TTL/淘汰/删除机制。实际检查 tmpfs：总量约 1008G，使用 1008G，仅余 7.3M，100%。内存文件系统不会自己淘汰这些文件。

Encoder 的真实异常链到 `safetensors.torch.save_file`：

```text
SafetensorError: Error while serializing: I/O error: No space left on device (os error 28)
```

encoder 容器退出码是 0、OOMKilled=false，但 engine 确实崩溃。Proxy 随后不能连接 encoder，reflection 重试收到 HTTP 500，训练最终 `RuntimeError: reflection attempts exhausted`，原 client exit 1。不能把容器 exit 0 当作服务健康。

### 实现

- `27bb0868`：proxy 增加 `/clear_encoder_cache`、活动请求计数和 lock；流式请求到实际 stream 结束才释放计数；图像 priming 用 TaskGroup 等待子请求收束。
- `a0be03dc`：训练启动/恢复时、每个 batch 的全部 rollout（含 reflection）结束后调用清理；更新 `INFRA.txt`、`PLANS.md`。
- `1cb440ee`：展开 TaskGroup 中的 HTTPException，保留 encoder 原始 HTTP 错误，不让它一律变成泛化 500。

清理流程：持锁禁止新请求进入 → 有 active request 则返回 409 → 依次调用 encoder 和 PD 的 `/reset_encoder_cache` → 删除共享 volume 中的缓存目录 → 返回删除数量与前后占用。

两端都启用 `VLLM_SERVER_DEV_MODE=1`，因为已安装 vLLM 的 reset 路由在 dev router 内。已读取实际安装源码确认 core 会重置 scheduler 和 executor 的 encoder cache，并从两端真实启动日志确认 POST 路由存在。只删除文件却保留内部缓存可能造成状态不一致，因此两端重置是修复的一部分。

这是启动/批间清理，**不是批内容量上限、TTL 或 LRU**。没有新增 cache 大小参数；超大单批能否超过容量没有被本次验证排除。清理也不保证释放所有其他类别的主机内存/KV cache。

### 正式 run 中实际验证

服务重建后 tmpfs 最初为 0；不能把这一步说成新清理代码释放了原来的 1TB。

随后正式训练清理日志验证：

| 完成的 epoch 0 batch | removed_entries | used_bytes_before | used_bytes_after |
| --- | ---: | ---: | ---: |
| 018 | 1389 | 20796166144 | 0 |
| 019 | 1255 | 19383316480 | 0 |
| 020 | 3670 | 53803704320 | 0 |

Proxy 对 encoder/PD reset 的真实日志均出现 status=200。下一批运行时缓存再次增长是预期现象；最后一次小时检查看到约 32G，不能把它作为停止后的当前值。

已执行一次新 subagent diff review 和一次 Claude 全仓 review。小修复后未重复 review。Claude 提到无法从仓库确认 reset 路由；已由实际安装源码和启动日志解决。没有运行语法检查、smoke test 或 preflight；验证来自正式运行。

## 7. 正式 run 产物与已知问题

W&B：<https://wandb.ai/3dsmile/video-sql-v2/runs/evnxlgp9>。Weave 项目为 `3dsmile/video-sql-v2`，client 日志有实际 call 链接；这里只确认日志存在，不据此宣称所有 trace 健康。

`config.json` 保留 run 初建的 `main_git_hash=a16fadefe0011afff9a200c66289f641778b9a06`，并非恢复时的运行代码 hash。恢复代码不重写旧 config；辨认实际代码需结合镜像和上述提交。

最后完整 batch 的真实 `batch.json`（020）：successful=30，failed=0，mean_score=2.8，fully_correct_portion=0.4，skill_commit=`3fc5d1cb3b1483ce3242144fa5b70b914e41d1c1`。successful 表示管线完成，不表示答案正确。

主机读取技能仓库 HEAD 时遇到 Git dubious ownership，未修改全局 safe.directory。因此上面的 skill_commit 是 batch 文件记录，不能假称再次读出了技能仓库当前 HEAD。

具体 trace 路径相对于 run：

- `train/epoch-000/batch-018/question-56/rollout-04/attempt-2/traces/clip-2064a35c0b8e457098b0f509e2067613-segment-00.json`：人工读取片段，见 sample_video 首次传 crop 全 0 返回明确错误，随后模型纠正为 0/0/1000/1000，返回 40 帧/4 contact sheets；再调用两个细采样窗口，均返回图像项；多轮 reasoning item 保留。此次只检查了该片段，没重新查看其全部实际图片像素，不能称已全量验证。
- `train/epoch-000/batch-020/question-133/rollout-06/attempt-1/reflection-attempt-1-error.json`：`Tool submit_answer not found in agent memory-retrieval`。
- `train/epoch-000/batch-020/question-132/rollout-01/attempt-1/error.json`：`Invalid JSON input for tool submit_answer`。
- `train/epoch-000/batch-020/question-128/rollout-09/attempt-1/error.json`：顶层 `object NoneType can't be used in 'await' expression`，异常链含 `RemoteProtocolError: peer closed connection without sending complete message body (incomplete chunked read)`；client 同期有 `Error streaming response`。未定位具体源代码根因。

这些错误所在 batch 最终写出了 30 个完成结果，不代表错误路径不存在，也不能把这些错误自动等同于用户尚未说明的新 bug。

更早的人工检查还发现某些 reflection 在未找到答案时仍很长或写出推测经验，个别 clip 输出认错游戏/角色或计数。它们说明 prompt 合规性不能只看结果分数；本轮缓存修复没有重审或修复这些模型行为。

历史 semantic_search 不可用、clip 输出 `[object Object]`、视频路径不存在、reflection clone/compaction 争议均有先前修复/事故记录。不要仅靠这份交接认定当前任意样本已无这些问题；用户指定问题后读取当前实际调用输入和返回值。

## 8. 恢复语义与原启动参数（暂不执行）

现有训练从 `config.json` 使用同一个 wandb_id，`resume="must"`；已存在 batch.json 的批次跳过。未完成 batch 重用已保存的 rollout `result.json`，只补缺失项，然后继续 summary/updater。不要删除 completed rollout 或换 run 目录来“方便恢复”。如果用户认为某类旧产物已无效，由用户决定作废范围。

原 client 命令保存在下面供核对；已有同名已停止容器，不能原样直接再 docker run。修改 src 后也不能仅 docker start 旧容器，因为旧镜像没有新代码。等用户提供 bug 并完成相应修复/审查、明确恢复后，再处理容器名与新镜像。

```bash
docker run --name video-sql-qwen38-27b-fp8-fresh-3x10-full-client \
  --network video-sql-v2 --gpus '"device=0,2,4,5"' --env-file .env \
  -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory \
  -e GIT_CONFIG_VALUE_0=/workspace/video_sql_v2 -e HF_HOME=/raid/huggingface \
  -v /home/wg25r/.netrc:/root/.netrc:ro \
  -v /raid/wg25r/video_sql_v2/runs:/workspace/video_sql_v2/runs \
  -v /raid/wg25r/video_sql_v2/data:/workspace/video_sql_v2/data:ro \
  -v /raid/wg25r/video_sql_v2/.git:/workspace/video_sql_v2/.git:ro \
  -v /raid/wg25r/video_sql_v2/AGENT_STDOUT.txt:/workspace/video_sql_v2/AGENT_STDOUT.txt \
  -v /raid/huggingface:/raid/huggingface:ro \
  video-sql-training:cache-cleanup-a0be03dc -m src.training \
  --dataset-root /raid/huggingface/hub/datasets--MM-Lifelong--MM-Lifelong/snapshots/248aa82039a574e63a2e524746a7cd8f32330443 \
  --runtime-root /workspace/video_sql_v2/data/runtime \
  --initial-skills /workspace/video_sql_v2/skills/initial \
  --run-dir /workspace/video_sql_v2/runs/qwen38_27b_fp8_epd_fresh_3x10_seed0 \
  --model-name Qwen/Qwen3.8-27B-FP8 --api-key-env LAB_OPENAI_KEY \
  --base-url http://qwen38-27b-fp8-epd-proxy:8000/v1 \
  --embedding-url http://egolife-qwen3-vl-embedding-8b:8000/v1 \
  --embedding-model qwen3-vl-embedding-8b \
  --attempts 3 --epochs 3 --batch-size 3 --rollouts 10 --seed 0 --first-epoch-skip 0
```

不要输出 `.env`/`.netrc` 内容，也不要去其他目录寻找 API key。

## 9. 操作纪律与历史交接陷阱

- 中文交流。遵循当前 AGENTS.md 与用户最新要求；只修改指定目标，不顺带清理或重构。
- 永远不要 `git status --short`。列 untracked 使用 `git ls-files --others --exclude-standard`。不要对输出数据做全仓 diff。
- 每次编辑都提交。本文写入前的无关 untracked 为 `OTHERS.md`、`notes/INCIDENT_redundant_post_patch_read.md`、`notes/index.md`，不要加入本次提交。
- 永远不要 `fork_turns="all"`，也不能省略该参数导致默认 all。独立审查用新 agent、fork none，不泄露用户已指出的结论来引导独立 trace 审查。不要复用旧 subagent。
- 主 agent 不得打开 REVIEW.md；中等修改按仓库规则派新 reviewer，prompt 使用用户要求的短格式：`Review the diff between X and Y against REVIEW.md for the purpose of A for files [files]`。正式运行前的 Claude 全仓 review 按当时有效规则处理；本轮已完成的 review 不必因为交接重复。
- 不写新 tests、不开 syntax check/smoke/preflight；从正式产物与实际 trace 验证。不要把 schema/工具错误吞掉继续生成无效结果，也不要擅改重试次数和实验参数。
- 稳定后监控间隔 **1 小时**，禁止 60 秒轮询和反复进度消息。短任务/服务启动可短等；长任务保持 foreground exec，不用 tmux/nohup/后台 watcher。
- 工具单次 poll 有上限时，在一次 functions.exec 内累计等满 1 小时，底层进程退出则立即返回；外层如返回 cell_id，继续同一 cell 的长 wait，不在模型层每 60 秒醒来。用户停止后不要自动继续此流程。
- 此环境 bwrap 存在 runtime 问题，shell 读写按用户指示在 elevated 环境执行。编辑首选 bare `apply_patch` 的持久 PTY，再用 write_stdin 发送 patch+EOF；遵守审批拒绝，不改工具绕过拒绝。
- 停容器只按精确名字；不 broad pkill，不删除数据集、技能或训练结果。
- 曾经 host `src/` 消失，按用户授权从镜像恢复了 6 个源文件：`__init__.py`、`harness.py`、`normalize.py`、`prepare_runtime.py`、`training.py`、`val.py`。内容与当时 HEAD 相同，`a465bda6` 为空提交记录恢复。不要顺手恢复其他用户删除的文件。

此前关于 Codex 强制 60 秒监控偏好的 GitHub issue 已处理：<https://github.com/openai/codex/issues/42981>。英文正文，核心是内置沟通偏好覆盖用户明确的低频监控要求，不是 Goal 功能本身；此任务目前没有待办。

`vis/README.md` 当前描述的是旧 continuation run 的静态导出器，`vis/dist/` 更旧。不要把已有网页当成本次 fresh run 的实时 trace，也不要假称本次修复已更新 Sites；本轮未重新发布，交接时未核对线上版本。

## 10. 新会话从这里开始

1. 保持训练停止，接收用户发现的具体 bug；本文没有替用户推断 bug 内容。
2. 以当前源文件和用户指定 trace 为证据定位，分别确认工具实际输入/返回、reflection/summary/updater 产物，不能只报告分数。
3. 修改严格限定在确认的问题；用户如果要求作废/重跑，再明确处理相应范围，不能自行把所有旧结果抹掉。
4. 修复、必要 review 和镜像同步后，按用户新的恢复要求继续；当前停止指令仍有效。
