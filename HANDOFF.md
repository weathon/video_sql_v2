# video_sql_v2 handoff

## 用户当前目标

重新设计训练 harness。继续工作时应以本文件中的“新 harness 规格”为当前需求，不要继续启动旧 v11，也不要把旧的局部修复当成最终设计。

## 新 harness 规格

1. Agent 停止与循环检测：
   - 主 retrieval agent、`clip_qa` 和 reflection 分别增加适配自身输出的 `submit_answer` tool；模型通过调用它提交最终答案，调用后退出对应 agent loop。
   - skill updater 不使用 `submit_answer`；它完成连续文件操作后调用专用 `stop` tool 退出。
   - 所有 agent loop 的总轮数限制为 100。100 足够所有正常完成结果；触发 100 证明本次运行出问题，本次 trace 作废并按现有正式模式重试整个 rollout。
   - 如果连续 3 次 agent completion 在排除 reasoning 后的 raw characters 完全一致，本次 trace 作废，并按现有正式模式重试整个 rollout。
   - 所有 trace 都必须保存，包括因重复输出或达到 100 轮而作废的 trace，以便 debug。
   - 这条需求源于真实 `clip_qa` trace 中模型反复声称要停止、但仍调用 `add_note` 的循环。
2. 正式训练改为每 batch 5 个问题、每个问题 10 个并行 rollout。
3. 主模型改为 **Qwen 3.8 Next Flash**，不是 Qwen 3.8 27B。模型名称和启动方式必须读取仓库 `INFRA.txt` 后确定，不要凭记忆填写。
4. SKILL 不再是单个 Markdown 文件，而是一个包含多个 Markdown 文件的目录：
   - 每个文件对应一种特殊情况，例如“查找特定时间任务”。
   - 每个文件包含介绍、何时使用、主要内容等。
   - SQL 示例直接写在 Markdown 中，不单独保存脚本文件。
   - 结构参考 Anthropic 官方 skill-creator：<https://github.com/anthropics/skills/tree/main/skills/skill-creator>。
   - Anthropic 的 skill-creator 是“如何写 SKILL”；本项目的 `META-SKILL` 仍然是“如何反思”，两者不是同一概念。
5. Rollout 不把所有 SKILL 文件直接放入 prompt：
   - prompt 只提供可用 SKILL 列表及各自“何时使用”的说明。
   - 新增读取 SKILL 的工具；输入具体 SKILL 名称，返回对应文件的完整内容。
6. Reflection 完成后的 SKILL 更新改为多轮 agent loop：
   - updater 可以增加、修改和删除 SKILL Markdown 文件。
   - SKILL 目录由一个独立 Git 仓库管理。
   - 一轮完整 update 的所有文件操作完成后只 commit 一次，不是每个 patch 各 commit 一次。
   - updater 可以查看历史并参考旧版本重新写入当前文件。
   - 不允许向 updater 暴露或执行直接 Git rollback；所谓“回滚”必须是读取历史内容后重新 update，再形成新 commit。
   - updater 完成全部文件操作后调用专用 `stop` tool 退出，不使用 `submit_answer`。

## 开始实现前必须向用户确认的设计点

当前需求仍有一个会改变实验语义的未决点；受 `AGENTS.md` 的“每个有意义决定都询问用户”约束，新对话必须先确认，不能自行选择：

1. 新 SKILL 独立仓库的路径、首批文件如何从现有 `skills/initial/SKILL.md` 拆分，以及 `META-SKILL.md` 放在外层主仓库还是 SKILL 独立仓库中。

## 已核验的循环事件

- v11 路径：`runs/pilot_formal_openai_agents_tp1_seq100_v11`。
- W&B run：`07n50o4x`。
- 异常文件：
  `runs/pilot_formal_openai_agents_tp1_seq100_v11/train/epoch-000/batch-000000/questions/question-000001/rollout-00/attempt-01/traces/clip-46e519c544234f9ab66d5a764371e3c2-segment-00.json`
- 该单次 `clip_qa` 执行了 587 次模型请求，其中 585 次调用 `add_note`。
- 循环从 response index 9 左右已经明确出现；模型 message/reasoning 多次写明“不再调用工具”，但同一 response 仍附带 `add_note`。
- 末尾在以下 note 之间反复循环：
  - `Final. Conversation complete.`
  - `Final acknowledgment. Conversation complete.`
- 该 trace 累计 input tokens 34,146,937、output tokens 71,540；这些是 587 次请求的累计值，不是单个上下文长度。
- 第 588 次流式响应在 vLLM Responses 服务中触发 `ResponseFunctionToolCallItem.name=None`，客户端收到 incomplete chunked read，v11 因 `UserError` 终止。
- 这次事实已经加入 `PLANS.md`；对应提交为 `55a92e5 Record v11 clip agent loop trace`。

## 关于 MTP 的结论修正

- 模型的 `add_note` 循环不是 MTP 引起的。
- 真实顺序是模型先自行循环；高度重复、几乎确定性的输出恰好让 MTP 草稿 token 全部命中，所以接受率随后显示 100%。
- vLLM 的 XGrammar/MTP warning 和第 588 次流中断是另一个服务器问题，不能作为模型循环的原因。
- 当前容器确实缺少 vLLM PR #52805 的 XGrammar termination 修复，但除非用户另行要求，不要把升级或补丁 vLLM 当成本次 agent-loop redesign 的解决方案。

## 当前实现概况

- Harness 使用 OpenAI Agents SDK、`OpenAIResponsesModel`、Responses API 和 `Runner.run_streamed`。
- reasoning item policy 为 `preserve`；服务端 thinking/preserve_thinking 均开启。
- 当前 `clip_qa` 使用 `Agent(..., tools=[sample_video, add_note, read_notes], output_type=ClipAnswer)`。
- Agents SDK 当前默认 `tool_use_behavior="run_llm_again"`。只要 response 内仍有工具调用，即使同轮还有 message，Runner 也会执行工具并进入下一轮；这是 585 次 `add_note` 循环得以持续的直接机制。
- 当前主 agent 最大 50 turns；`clip_qa` 和 reflection 使用 `TURN_SAFETY_LIMIT=1000`。新设计要求所有 agent loop 的总轮数统一限制为 100。
- 当前代码已有接近 context limit 时关闭工具、强制结构化结果的逻辑；新设计不应误删这部分，除非用户明确改变。
- 当前正式配置仍是 2 questions/batch、5 rollouts/question；需要按新规格改为 5 和 10。
- 相关代码主要在：
  - `src/harness.py`
  - `src/training.py`
  - `src/val.py`
  - CLI 参数定义文件
  - `tests/`
  - `PLANS.md`

## 仓库状态

- 分支：`fresh`。
- 创建本 handoff 前观察到 HEAD：`e6beb1ed7fb467af3c5ec8bec1ae72cbeb805f42`，提交标题 `new plan`。
- 创建本 handoff 前已有工作区改动：
  - `D CODEX_PLAN.md`
  - `M PLANS.md`
- 上述改动不是本 handoff 创建的；必须保留，不要恢复、覆盖或顺手提交。
- 近期关键提交：
  - `55a92e5`：记录 v11 clip agent loop trace。
  - `dbdfc43`：记录 v10 clip recovery failure/fix。
  - `3342eff`：clip 结构化解析错误接近上下文时关闭工具。
  - `192c2be`：主 agent 空答案改为可重试错误。
  - `17d1716`：从真实 completed stream event 保留结构化解析失败 usage。
- 按仓库规则，每次主仓库编辑后立即提交；但新 SKILL 独立仓库的一轮 updater 操作按用户新规格合并为一次 commit。

## 服务与运行状态

- v11 正式训练已经退出，不能继续监控或恢复为当前正式方案。
- 最后核验的主 vLLM 容器名：`qwen38-nvfp4-mtp3-run`，GPU 5、TP=1、端口 8000、max model length 262144、max sequences 100、MTP 3。
- 这是旧的 Qwen 3.8 27B 服务；新流程要求改为 Qwen 3.8 Next Flash，必须按 `INFRA.txt` 重建。
- 最后核验的 embedding 服务名：`egolife-qwen3-vl-embedding-8b`，GPU 4、host 端口 8013。
- 这些是最后观测状态；新对话开始时应只读重新核验容器实际状态和参数。
- 查看旧主 vLLM stdout 的命令：
  `docker logs --follow --tail 200 qwen38-nvfp4-mtp3-run`

## 继续工作的顺序

1. 先读取当前 `AGENTS.md`、本文件、`PLANS.md` 的相关末段和 `INFRA.txt`；不要一次读取多份大型源码。此前一次批量读取三个源码文件的命令被审批器拒绝，拒绝后本对话已停止，没有绕过。
2. 阅读 Anthropic 官方 skill-creator 的实际内容，而不只看仓库目录名。
3. 向用户确认上面的一个未决设计点。
4. 把用户确认后的新 harness 设计写入 `PLANS.md`，只改相关内容并提交。
5. 为主 agent、`clip_qa` 和 reflection 实现 submit tool，为 skill updater 实现 stop tool；实现排除 reasoning 后连续三次 completion raw characters 完全一致的检测、统一 100 轮限制、失败 trace 保留、batch/rollout 配置、Flash 模型配置、SKILL 目录读取工具，以及独立 Git skill updater agent loop。
6. 不新增测试，除非用户明确要求；可以运行已有测试。严禁运行 syntax check。
7. 中等改动完成后，按 `REVIEW.md` 由一个新的 subagent 审查 diff；主 agent 不读取 `REVIEW.md`。
8. 正式 Pilot 前按规则执行一次 Claude CLI 全仓库审查；小型后续修复只需新的 subagent diff review。
9. 正式运行必须使用 Docker、W&B、前台 unified exec session，不传 OpenAI key。稳定后按 `AGENTS.md` 使用一小时监控间隔。
