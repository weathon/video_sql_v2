# 在线 teacher 指导的 skills 训练

## 训练协议

- 使用 Week 200 题，随机无放回组成 100 批，每批 2 题；保存抽样顺序以支持恢复。Month 保持为测试集。
- 从原始 6 个 skills 的副本开始。后续允许自由新增、改名、合并或删除 skills。
- 每题先运行 5 条独立完整 rollout，再注入经验运行 5 条，共 2,000 条正式轨迹，不含失败重试。
- Qwen 沿用 Week 协议：保留 thinking/history，不设 effort 或采样参数；共享 60 次调查调用，加一次强答；可重试失败最多 4 次。
- 所有 teacher 使用 Codex SDK、`gpt-5.6-sol`、high effort、normal speed。更新对象是经验和 skills，Qwen 权重保持固定。

## 每批执行流程

1. **冻结当前 skills，运行初始轨迹。** 两题各 5 条，共 10 条并发；每条使用独立历史、notes 和预算。使用 GPT-5 官方 rubric 评分，严格解析 `Final Score`。
2. **两个分析 agent 并发分析，各负责一题。** 提供该题所有原始 metadata、五条轨迹、实际工具输入输出、最终答案和评分。通过 JSON 路径清单按需读取，并允许使用 MCP 核验证据。Week clue 时间通过源视频与片内时间映射到 EgoLife 时间。
3. **归因并生成两份经验。** 每条轨迹的回报为官方映射分数：4/5→1、3→0.5、0/1/2→0。计算
   `Aᵢ = rᵢ − mean(r₁…r₅)`。
   teacher 以此为参照，为有证据支持的关键工具调用给出 `[-1, 1]` 局部贡献分、理由和实际调用引用，不要求局部分数之和等于 `Aᵢ`；没有依据的调用不强行归因。
   credit assignment 同时判断错误类别，并定位错误发生的步骤。如果最终错误来自 visual evidence 识别错物，前面正确的 retrieval / SQL 不应连带标为 negative；反过来，final answer 正确但实际是 lucky guess，也不应把整条轨迹标为 positive。区分答案是否正确和是否有实际证据支撑。
   对同一道题的多个 rollout 做 success/failure trajectory 的 contrastive comparison，比较哪些 step / strategy 反复出现在成功轨迹中、哪些只出现在失败轨迹中，并保留对应调用、出现次数和反例。
4. **回注经验并重跑，形成现有 pre/post A/B。** 各题只加入自己的经验，从头再运行 5 条。A（pre）使用当前 skills，B（post）在同一版 skills 上加入该题经验；现有 pre/post 就是这份经验整体的 A/B 验证。使用与初始轮完全相同的 skills、工具和预算。经验包含适用条件、dos、don’ts 和简短步骤；具体答案、clue 区间、证据定位信息及轨迹引用只保存在私有分析记录中。
5. **复盘并更新 skills。** 两个新的分析 agent 分别比较各题前后十条轨迹，检查答案正确性、证据链、无效探索和直接性，按需要修订经验。修订后不再 rollout。随后一个新的 Sol 编辑 agent 读取两份最终经验和当前 skills，直接修改下一版 skills；完成后才开始下一批。

每份经验输出前要求 teacher 检查是否泄漏题目答案或 clue。后续修改后的经验和 skills 不标注为已经通过本批重跑验证；本批 post 指标对应实际注入的经验版本。

## 实现与服务

- 新增 `src/train_online.py`，负责批次调度、评分、两轮分析、经验注入、skills 更新和恢复。复用现有 Week harness 的工具、轨迹记录和失败分类。
- 给 `VideoSQLHarness.answer` 增加可选经验输入，在正常 prompt 上追加通用指导，保留现有 schema、工具和 skills 路由。
- 调整 `src/mcp.py`，让分析进程加载当前批次的 skills；`src/prompt.md` 的 ROOT 与目录部分从同一版本生成。每个分析 agent 使用独立 MCP context。
- 启动一个 `Qwen/Qwen3.8-27B-FP8` vLLM 服务，TP=2。沿用 Week 的模型长度、MTP、prefix cache、4 GiB 多模态 cache 和 CPU KV offload 设置，启动前核对其实际资源需求。
- 当前候选 GPU 为 Qwen 0、1，embedding 4；执行时重新检查。Qwen 端口为 8310；8214 的 `qwen3-vl-embedding-8b` 服务使用一张 GPU，显存预算为 32 GiB。资源不足立即报告。
- 分析 agent 使用只读权限；编辑 agent 使用限定到候选 skills 工作目录的写权限，并通过自动审批执行必要的沙箱外 `apply_patch`。prompt 写明已验证的沙箱外调用方法，不启用 full access。
- Codex sessions 保存事件记录并在完成后 archive。

## 保存、恢复与指标

- 每批保存问题、skills 输入版本、两轮全部轨迹及评分、调用级归因与错误类别、success/failure 策略对照、实际注入的经验、最终经验和修改后的 skills。完成一个阶段即落盘。
- skills 编辑在独立候选副本上进行；只有正常完成的编辑结果才用于下一批。恢复时复用已完成阶段、既定问题顺序和对应 skills。
- rollout 重试耗尽按 Week 协议计 0；teacher 的可重试 API 或格式错误最多尝试 4 次，反馈具体错误，耗尽后停止当前批次。确定性错误、缺少必需服务或权限拒绝立即停止相关阶段。judge 沿用既有递增超时与严格格式重试。
- W&B 以 batch 为横轴记录：
  - pre/post accuracy：各轮 10 条轨迹中映射分数为 1 的比例。
  - pre/post 平均映射分数、原始分数、每题至少一条正确的比例。
  - 各项 `delta = post − pre`，以及累计配对结果。
  - 已完成题数、rollout 数、失败与重试数，以及现有调用量、耗时、token/cache 和吞吐统计；缺失 usage 明确标记。
- 原始记录和中间 skills 存入忽略目录。源码及文档修改及时提交；训练完成后提交最终 skills 与紧凑结果报告。

## 验证与执行

- 小范围真实调试覆盖 SQL、语义检索及过滤、视频核验、当前 skills 读取，以及完整的经验生成、回注和编辑流程。
- 检查实际请求：学生仅收到原题、当前 skills 和对应通用经验；两题状态独立；两轮使用同一 skills；teacher 读取了实际工具返回值。
- 核对同分轨迹、失败轨迹、零工具调用轨迹的归因；从保存的逐条评分重新计算 advantage 和 pre/post delta。
- 检查中断恢复不会重复抽题、重做已完成 rollout 或启用未完成的 skills 修改；检查图片配方和 trace 内容。
- 中等代码改动完成后，由新的审查子代理按 `REVIEW.md` 审查指定改动，要求正常退出并给出明确结果；本次用户明确豁免 Claude CLI 全仓库审查。
- 全程使用受控前台会话，稳定后按长间隔监控。100 批完成后核验最终产物与指标，停止并手动删除本任务 Qwen 容器，保留共享 embedding 服务。
