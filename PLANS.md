这是一个AI Memory Agent的项目, 你的目标是构建一个agent, 用来检索视频记忆并且回答问题. 本branch是从main checkout出来的, 不要查看git历史, main是一次失败的尝试, ../video_sql也是, 除非特别指示, 不然不要去看, 以免遭受误导. 如果过程中用户更新了plan的内容, 记得要更新这个文件. 

严格按照这个plan来做, 即使你觉得不符合常理/不正确. 

## 材料:
1. 数据集: https://huggingface.co/datasets/MM-Lifelong/MM-Lifelong
2. 旧的预处理仓库: https://github.com/lijiazheng0917/MAGIC-video.git
3. 使用的Harness框架: https://github.com/openai/openai-agents-python
5. Embedding模型: Qwen/Qwen3-VL-Embedding-8B (Use VL for both visual and text-only), start it using vllm
6. 数据库引擎: https://github.com/duckdb/duckdb
7. 本地模型: Qwen/Qwen3.8-27B-FP8, 使用参考在notes/INFRA.txt第一条
8. 预构建的index(只用作试点): jiazhengli7/magic-video-artifacts
9. notes/文件夹, 上一次失败过程积累的经验和踩坑, 可以参考, 但是不一定有用甚至可能跑偏, 冲突时以当前计划为准


## 项目创新点:
1. 本项目使用SQL Dataset代替graph, 从而使得query过程可以被agent控制, 并且可以直接实现多步骤跳转而不是agent自己relay. 
2. Agent SKILL.md "训练": 训练大模型很复杂, 耗费算力, 容易过拟合, 所以我们"训练"agent的skill.md文件, 而不是模型参数. 简单来说, 每一次rollout结束之后模型复盘, 复盘经验加入一个文本文件, 这个文本被下一次模型rollout使用, 如此反复迭代实现模型进化. 
3. 以往的index构建过于繁琐复杂, 我们使用一个简单的一次过的方式快速处理. 
4. 我们让模型自己反思, 而不是把trace给另一个agent, 这样模型知道当时自己的错误原因(例如保留思维链). 
5. 我们使用多个SKILL文件, 而不是一个, 模型按需读取. 

## 试点实验
1. 使用notes/index.md的方式读取index, 用mmlifelong的数据集分割, 找出training和testing set和测试集
2. 搭建我们的harness,  见Harness架构如何搭建, 使用OpenAI Agents SDK连接本地模型, 验证他保留了过往reasoning并且支持图片或视频输入
3. 在训练集上跑我们的`"训练"过程`, 每个问题5个rollout, 一次4个问题, 一共3个epoch
4. 监控训练过程的平均打分到wandb, 结束后在测试集上测试. 
5. 使用最开始的最简单SKILL, 在测试集跑一次, 对比准确度


## 方法详情
### 介绍
本方法基于MAGIC-Video的index, 但是MAGIC-video把已经表格化的内容(triplet, caption, etc)作为graph的PPR检索, 检索过程agent不可控. 所以我们把同样的index作为sql数据库检索, agent写SQL来检索数据库. 
例子.  TIME_OF_INTEREST是占位符表示需要query的时间. 
```sql
SELECT e.time, c.text
FROM events e JOIN captions c ON e.time = c.time
WHERE e.description LIKE '%lunch%'
  AND e.time >= TIME_OF_INTEREST - INTERVAL '5 days'
  AND c.text LIKE '%beef%';
```

然后再调出对应画面确认。

其他几个例子:

统计某人上周在厨房出现的时长
```sql
SELECT SUM(e.duration)
FROM events e JOIN triplets t ON e.event_id = t.event_id
WHERE t.subject = 'Alex' AND t.object = 'kitchen'
  AND e.time >= TIME_OF_INTEREST - INTERVAL '7 days';
```

找出和 Alex 同时出现在同一场景里次数最多的人
```sql
SELECT t2.subject, COUNT(*) AS co_occur
FROM triplets t1 JOIN triplets t2 ON t1.event_id = t2.event_id
WHERE t1.subject = 'Alex' AND t2.subject <> 'Alex'
GROUP BY t2.subject
ORDER BY co_occur DESC
LIMIT 5;
```

回溯某个物体最后一次被看到的位置
```sql
SELECT e.time, t.object, c.text
FROM triplets t JOIN events e ON t.event_id = e.event_id
JOIN captions c ON c.time = e.time
WHERE t.subject = 'keys'
ORDER BY e.time DESC
LIMIT 1;
```

跨天做行为对比,比如每天做饭的时间点
```sql
SELECT DATE(e.time) AS day, MIN(e.time) AS first_cook
FROM events e
WHERE e.description LIKE '%cooking%'
GROUP BY DATE(e.time)
ORDER BY day;
```

这几个的共同点是 PPR 那种相似度扩散做不了:带时间窗口的计数、聚合、join 之后的排序,都需要显式的关系代数操作,而 agent 写 SQL 正好能表达这些。

### Harness架构
#### 工具列表: 
## Agent 接口与 compaction

### Harness 工具

- `clip_qa(video_id, start_s, end_s, question)`：建议片段不超过 10 分钟，超过 20 分钟直接拒绝。内部 agent 可调用 `sample_video`、`add_note`、`read_notes`。
- `clip_qa` 返回 `answered|unknown`、答案和至少一个证据时间点；harness 将证据时间点转换为独立截图交给主 agent。
- `sample_video`：接受时间、fps 和 0–1000 整数裁剪坐标；返回覆盖请求区间的 caption 和带时间标记的 4×3 contact sheet，每页 12 帧，所有帧完整分页。contact sheet 和证据截图只向模型暴露文件名，不暴露宿主机绝对路径。
- `run_sql(sql)`：只读执行，30 秒硬超时；最多向模型返回 100 行并说明是否存在更多结果；每个单元格最多返回前 500 字符及原长度。所有执行错误作为 tool result 返回给模型处理。
- `sql_qa(sql, question)`：和run_sql类似, 接受一个sql命令, 同时接受一个we
- `semantic_search(query, n, table)`：只允许查询已生成 embedding 的表，返回 table、row_id、video_id、start/end、规范化文本预览及 cosine score。
- 主 agent 提供 `add_note`、`read_notes` 和 `agent_raise(reason)`；普通工具执行错误返回调用它的 agent 处理，`agent_raise(reason)` 保持为 agent 主动结束当前 rollout 的信号；正式运行中同一 rollout 最多按原设置重试 3 次。
- 所有 agent loop 最大 100 turns；达到上限按整个 rollout 重试，不截断上下文或返回半成品。
- SKILL部分不再是一个文件, 而是多个, 如何构建SKILL参考antrophic官方的SKILL(https://github.com/anthropics/skills/tree/main/skills/skill-creator), 注意, 虽然这个也可以被理解为META-SKILL但是和我们的不一样, 我们的META SKILL是如何反思, 这个是如何写SKILL, 是在反思完成后更新skill文件用的. 这意味着, 我们的skill不再是一个单独的md文件, 而是一个目录, 这个目录包括多个markdown文件, 每个文件对应一个特殊情况(比如: 查找特定时间任务), 包含介绍, 何时使用, 以及主要内容等. 内容可以包含SQL脚本示例 (类似于SKILL中的脚本文件, 但是我们不单独存储脚本而是放在markdown里面).
- 为了实现上一点, 我们的harness需要
  - rollout时, 模型不读取所有skill文件, 而是一个list, 列出可用skill和何时使用, 然后多了一个工具就是读取SKILL, 是具体SKILL名字, 然后工具返回SKILL内容
  - 反思更新SKILL的时候, 模型可以增加文件, 修改文件, 和删除文件. 这个SKILL文件夹被单独的git仓库管理, 每次upadte之后(多有update, 不是单个patch), 都会commit一次, 然后反思的时候agent还可以看历史, 回滚修改 (安全问题, 不允许直接git回滚, 而是参考历史重新update). 这个是一个多轮的agent loop, 类似于直接受控操作文件系统. 注意: 这个META SKILL是注入主agent的message, 而不是单独的Agent. 


### Compaction

- 仅主检索 agent 使用 compaction；`clip_qa`、judge、reflection、summarizer 和 updater 不使用。
- 每次主 agent 响应完成后读取服务端 usage，计算 `remaining = notes/INFRA.txt.context_limit - input_tokens - output_tokens`。
- 当 `remaining <= 50,000` 时，通过流式 runner 执行 `cancel(mode="after_turn")` 并排空事件，确保当前工具轮完整结束。
- 在当前主检索 agent 的同一对话中追加固定 compaction 指令，要求它停止工具调用并总结当前对话。该 agent 的下一次输出作为新对话中的 trusted handoff，旧对话不再传入；不创建单独的 compactor agent。
- 第一次和第二次 compaction 后继续 rollout。第三次 compaction 输出生成后立即终止，状态设为 `compaction_limit`，分数设为 0，不调用 judge、不重试，直接进入重点分析绕圈原因的 reflection。
- Reflection 永不 compaction；当其 remaining 小于等于 50k 时，禁止继续调用工具并强制输出最终 reflection 和 meta-reflection。

#### "训练"过程:
1. 我们定义batch size代表n个问题和每个问题m个平行rollout. 把SKILL.md放入prompt对于一整个batch做平行rollout, 得到20个结果, 这是用用一个额外的judge call来判断模型答案正确性(打分/5), 这个分数记作s. wandb log这个s. 这个s也是Qwen/Qwen3.8-27B-FP8产生. 
2. 把正确答案和s(而不只是s)都给每一个rollout, 然后告诉他这是正确答案和你的分数(注意! 不是另一个agent, 是告诉答题agent自己), 同时插入META-SKILL.md (META-SKILL.md不是最开始放进去的), 你现在要做的就是去重新复盘这个结果, 包括重新回顾自己的trace和重新对数据库检索和探索, 找到你为什么对/为什么错的原因, 把这个原因总结出来, 记错relection, 同时这个反思过程有什么经验(帮助下一次反思), 记错meta-relection. 两个反思都要去掉具体内容变得抽象化 可以辅助任何问题而不是本次情况. 本次情况可以作为案例, 但是要去掉敏感信息防止数据泄露(比如具体某天中午午饭内容). 注意要模型不光总结新的经验, 还要让她注意什么时候要更新/替换/修改旧的经验. 不然经验会越变越大而且出现重复/冲突/相似的内容.
3. 每一个问题的m个rollout的反思被另一个call总结起来, 然后模型生成一个新的SKILL.md和META-SKILL.md, 同上, 但是注意不要一味的添加, 也要值得去除/替换. 不然会冲突和越变越大. 
4. 下一个batch带着新的SKILL.md和META-SKILL.md继续. 


#### Prompt
System Prompt包含简单的任务说明和database schema, **不需要**任何详细的任务步骤描述, 例如如何搜索如何继续. 但是, 我们需要一个起始状态的SKILL.md来告诉模型怎么做检索, 同样的, 反思阶段给的guideline也应该保持最简单, 而具体的开始反思的guideline应该放在META-SKILL.md. 这样模型不会被我们的system prompt困住或者和SKILL.md和META-SKILL.md冲突, 而是可以自由修改. 然后作为baseline的no skill baseline包含这个最开始的最初始的SKILL作为system prompt, 而不是完全没有prompt. 

## 代码结构
src/harness.py: 主harness架构, 包括tools定义
src/training.py 训练代码
src/val.py 验证集代码, 可以测试没有SKILL.md(baseline)也可以测试有
tests/ 所有单元测试
runs/ 记录训练的log和缓存的SKILL.md和META-SKILL.md, 包括超参数和git hash
notes/ 记录给你自己的note, 包括注意事项和踩过的坑, 被context compact之后可能需要参考
其他你觉得需要的文件

## 规则
1. 不允许静默失败。调试时显式失败；现在已进入正式运行，所有 FunctionTool 执行错误都作为 tool result 返回给调用它的 agent 处理，不直接使 rollout 或 TaskGroup crash。如果 agent 主动调用 `agent_raise`，则调试阶段停止、正式运行按原设置重试，3 次失败后跳过；agent loop 外的失败继续按对应的 retry 或 skip 规则处理。
2. 每次开始运行前, 使用REVIEW.md审查代码
3. 使用git做记录, 以方便回滚, 实验结果需要配对的git hash, 注意修改.gitignore防止不需要的文件进入历史. git操作时, 不要用系统默认用户名, 创建一个虚拟的name=codex email=codex@localhost来分别作者
4. 保持代码简洁干净, 不要过度设计和抽象, 这是科研代码
5. 代码必须在batch level可重新开始
6. 所有run必须log到wandb
7. 本台机器有8xB200和2TB RAM, 性能强劲, 但是不要用很多GPU, 单个GPU经常跑不满, 所以尝试用单个GPU负载. 
8. 全程前台监控, 但是不要频繁查看, 刚开始时每30s看一次, 后面逐步改成10min, 30min, 1h. 每次不要直接查看日志, 不然你会看到很多噪音. 脚本的详细log不要输出到stdout而是log文件, stdout只输出你需要监控的信息, 并且你只看这些. 
9. 用openai的agent sdk来写agent, 不要自己手搓harness, 用response api, 保留过往reasoning
10. 你的运行过程中除非停止, 不然所有消息都默认是你的自言自语用户不会读, 需要用户读取的milestone和log记录在AGENT_STDOUT.txt给用户看, 比如cost, 你自己做的决定, etc. AGENT_STDOUT是你输出给用户看的重要信息, wandb是记录run的过程. 

## 实现决定
1. 所有文本转成ASCII并转成小写.做成归一化
2. 去除文本开始和结尾的空格, 所有连续空白折叠成单个空格, 总是允许一定程度的改变和坍缩, 保持一致优先.
3. 所有依赖保持当前最新版; 如果不能工作, 回退到可以工作的版本.
4. 环境使用uv; 不需要每次都创建新的venv, 不能使用别人创建的venv, 自己创建的venv可以复用.
5. 保留所有非英文字符; 英文字符使用ASCII.
6. infra benchmark使用更长的测试文本和多轮对话.
7. `run_sql`每个单元格保留前500字符.
8. 轮数实际设置为100; 达到100说明模型自循环, 重试整个样本.
9. 允许并行工具调用.
10. uv cache放到/raid.
11. home目录满了时先清理home, 删除本次产生的无用内容.
12. 过程中严谨访问未列出的目录和当前目录. 


## Update to Plan

### Agent 停止与循环检测
1. 主检索 agent、`clip_qa` 和 reflection 各自增加一个适配自身输出结构的 `submit_answer` tool；模型通过调用它提交最终答案，调用后退出对应 agent loop。不再依赖 Agents SDK 的 `output_type` 结构化最终输出来结束 loop。
2. skill updater 不使用 `submit_answer`；它完成全部文件操作后调用专用 `stop` tool 退出。
3. 所有 agent loop 的总轮数限制统一为 100。触发 100 视为本次运行出问题：本次 trace 作废，并按现有正式模式规则重试整个 rollout。
4. 如果连续 3 次 agent completion 在排除 reasoning 后的 raw characters 完全一致，本次 trace 作废，并按现有正式模式规则重试整个 rollout。
5. 所有 trace 都必须保存，包括因重复输出或达到 100 轮而作废的 trace。
6. 现有接近 context limit 时关闭工具、强制输出的逻辑保留不变。
7. 正式运行中，所有普通 FunctionTool 执行错误以 `{"error": "..."}` tool result 返回给调用它的 agent 处理，不再让异常直接穿透并使 rollout 或 TaskGroup crash；`agent_raise` 保持为 agent 主动结束当前 rollout 的控制信号。统一使用 JSON 是因为 v18 已将 SDK 默认错误文本返回给 agent，但 reflection 的 rollout trace 会将普通工具返回解码为 JSON，默认文本导致 `JSONDecodeError` 并终止整批运行。

### 训练配置
1. 正式训练改为每 batch 4 个问题、每个问题 5 个并行 rollout，共 20 个并行 rollout；服务并发上限设为 24。
2. 主模型使用官方 `Qwen/Qwen3.8-27B-FP8`。PD consumer 运行在 GPU 0,5，TP=2，端口 8002，max-num-seqs 24，gpu-memory-utilization=0.90，不使用 `--no-async-scheduling`，MTP speculative length 为 2，default chat-template kwargs 保持 `enable_thinking=true` 和 `preserve_thinking=true`；agent reasoning effort 使用 `medium`。改回 27B 是因为 Next Flash 在 TP=2、gpu-memory-utilization=0.90 下没有可用 KV cache；使用官方 FP8 避开此前 NVFP4 重复 reasoning 问题；0.90 为运行时动态加载 encoder cache 保留额外显存余量；TP=2 是本次正式运行指定的并行配置；MTP=2 用于检查 MTP=4 是否导致正式运行吞吐过低。
3. harness 的 `context_limit` 保持 262,144，并保留接近该边界时的 compaction；vLLM 后端通过模型官方 YaRN 配置启用 314,573 token 上限、factor=1.2，作为 client compaction size 之上的 20% 缓冲。
4. 正式训练 client Docker 暴露与 vLLM 相同的 GPU 0,4,5，供 W&B 采集 GPU 指标。
5. 一个 batch 的全部 rollout 都失败时跳过该 batch 的 updater；连续两个 batch 的全部 rollout 都失败时终止训练。
6. 保留 OpenAI SDK 内部重试，并将 `max_retries` 设为 3。
7. 正式训练 client 使用 `docker/training/Dockerfile` 构建的派生镜像，并安装 `ffmpeg`；原因是 `sample_video` 会调用 `ffmpeg`。
8. 按 vLLM Encoder Disaggregation 方案，将 vision encoder producer 与 embedding 服务共同运行在 GPU 4、各使用 gpu-memory-utilization=0.45，encoder 端口为 8001、max-num-seqs 为 24，并与 PD consumer 并行运行，通过共享 EC storage 向 PD consumer 提供 encoder cache；`ECExampleConnector` consumer 使用当前 TP worker 的 CUDA device 加载 cache；`infra/responses_epd_proxy.py` 在端口 8000 保持 `/v1/responses` 请求和 SSE 输出原样，只先把其中的 `input_image` 发送到 encoder producer，encoder 失败时不把请求发送给 PD；共享 cache 放在 `/raid`，并在活跃 Responses 请求数归零后清空。原因是 v18 在 47 个纯文本 running 请求时四张 PD GPU 为 100% SM，但进入多模态阶段后仍有 21 running、0 waiting 时 SM 多数降到 14–35%，且同一采样窗口出现新的 encoder cache miss；EPD 用于消除 vision encoder 与 prefill/decode 的共置干扰，同时避免无淘汰的 `ECExampleConnector` 填满根盘；当前 device 避免 safetensors 将每个 TP worker 的 `cuda` 都解析为逻辑 `cuda:0`；GPU 4 共置避免 encoder 占用 PD GPU。

### SKILL 目录
1. SKILL 不再是单个 Markdown 文件，而是一个包含多个 Markdown 文件的目录；每个文件对应一种特殊情况，包含 YAML frontmatter（`name`、`description`，description 即“何时使用”）、Introduction、When to use、Main content；SQL 示例直接写在 Markdown 中，不单独保存脚本文件。结构参考 Anthropic 官方 skill-creator。
2. Anthropic 的 skill-creator 是“如何写 SKILL”；本项目的 `META-SKILL.md` 仍然是“如何反思”。
3. 首批文件位于 `skills/initial/`：`general-retrieval.md`、`sql-schema.md`、`semantic-search.md`、`locating-time-intervals.md`、`counting-and-ordering.md`、`visual-verification.md`，以及 `META-SKILL.md`。旧的单文件 `SKILL.md` 已删除。
4. 首批文件职责：`general-retrieval.md` 说明完整检索循环和何时读取其他 SKILL；`sql-schema.md` 说明 DuckDB 表、字段和连接关系；`semantic-search.md` 说明语义检索及从候选结果转入 SQL 验证；`locating-time-intervals.md` 说明根据时间线索定位和扩展时间窗口；`counting-and-ordering.md` 说明去重、计数、聚合和排序；`visual-verification.md` 说明何时以及如何调用 `clip_qa` 做画面核验；`META-SKILL.md` 说明答题 agent 如何复盘、抽象经验以及更新、替换或删除旧经验。
5. 每个 run 在 `<run_dir>/skills/` 内持有自己的 SKILL 独立 Git 仓库：训练启动时从 `skills/initial/` 复制并直接 `git init` 后做首次 commit；它不是主代码仓库的 worktree 或分支。`META-SKILL.md` 放在该独立仓库内。val 读取指定 run 的该目录。
6. Anthropic skill-creator 的规范提供给 skill updater，指导它创建和修改上述 SKILL 文件；它不替代本项目的 `META-SKILL.md`，也不作为答题阶段的检索 SKILL。

### Rollout 读取 SKILL
1. prompt 不再放入全部 SKILL 内容，只列出可用 SKILL 的 `name` 和 `description`。
2. 新增 `read_skill(name)` tool，返回对应文件的完整内容。
3. `read_skill` 成功返回的原始 Markdown 作为工具返回保留在答题 agent 的同一对话中；工具错误使用统一的 error JSON 返回。Reflection 不建立 trace 投影，也不对成功的 Markdown 工具返回执行 JSON 解码。

### Skill updater agent loop
1. Reflection 汇总完成后的 SKILL 更新改为多轮 agent loop；updater 可以增加、修改和删除 SKILL Markdown 文件，也可以修改 `META-SKILL.md`。
2. updater 可以查看 Git 历史并读取旧版本内容，参考后重新写入当前文件；不向 updater 暴露或执行直接 Git rollback。
3. 一轮完整 update 的所有文件操作完成后只 commit 一次。
4. updater 完成全部文件操作后调用专用 `stop` tool 退出。
5. 每个 updater Git commit 的 body 必须列出本次更新所依据的完整 rollout ID（`epoch/batch/question/rollout/attempt`）以及 updater 提交的总体修改原因。保存的 SKILL 中每个指导点后必须包含简短的 `**Reason:** ...`；如果一条原因覆盖整个 section，可以只在该 section 末尾记录一次。这样便于以后推翻或修改之前的方案以及 debug 追溯。

### 正式执行
1. 不运行 preflight 或 smoke test；直接启动正式运行，并只用正式运行的结果和完整 trace 检查正确性。
2. 之前把 trace 交给独立 reflection agent 的实现及其正式 run 结果全部作废，不恢复旧代码或旧 run；从当前已删除的 `src/`、`tests/` 和 `skills/initial/` 状态按本计划重新实现，并使用新的正式 run 目录从头执行。
