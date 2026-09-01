这是一个AI Memory Agent的项目, 你的目标是构建一个agent, 用来检索视频记忆并且回答问题. 本branch是从main checkout出来的, 不要查看git历史, main是一次失败的尝试, ../video_sql也是, 除非特别指示, 不然不要去看, 以免遭受误导. 如果过程中用户更新了plan的内容, 记得要更新这个文件. 

## 材料:
1. 数据集: https://huggingface.co/datasets/MM-Lifelong/MM-Lifelong
2. 旧的预处理仓库: https://github.com/lijiazheng0917/MAGIC-video.git
3. 使用的Harness框架: https://github.com/openai/openai-agents-python
5. Embedding模型: Qwen/Qwen3-VL-Embedding-8B (Use VL for both visual and text-only), start it using vllm
6. 数据库引擎: https://github.com/duckdb/duckdb
7. 本地模型: Qwen/Qwen3.8-27B, 使用参考在INFRA.txt
8. 预构建的index(只用作试点): jiazhengli7/magic-video-artifacts
9. notes/文件夹, 上一次失败过程积累的经验和踩坑, 可以参考, 但是不一定有用甚至可能跑偏, 冲突时以当前计划为准


## 项目创新点:
1. 本项目使用SQL Dataset代替graph, 从而使得query过程可以被agent控制, 并且可以直接实现多步骤跳转而不是agent自己relay. 
2. Agent SKILL.md "训练": 训练大模型很复杂, 耗费算力, 容易过拟合, 所以我们"训练"agent的skill.md文件, 而不是模型参数. 简单来说, 每一次rollout结束之后模型复盘, 复盘经验加入一个文本文件, 这个文本被下一次模型rollout使用, 如此反复迭代实现模型进化. 
3. 以往的index构建过于繁琐复杂, 我们使用一个简单的一次过的方式快速处理. 

## 试点实验
1. 用提前做好的index (材料 8)和原始视频(材料 1), 构建一个新的duckdb数据库, 数据库的schema完全准许材料2, 只不过我们是用SQL database而不是graph, embedding也许需要重新准备, 参考notes/VIDEO_EMBEDDING_PIPELINE.md如何构建新的embedding.
2. 搭建我们的harness,  见Harness架构如何搭建, 使用OpenAI Agents SDK连接本地模型, 验证他保留了过往reasoning并且支持图片或视频输入
3. 把材料8和材料1的数据集做graph分割, 按照问题和视频连接做成一个图, 分割出两个set, 大的作为训练集, 小的作为测试集
4. 在训练集上跑我们的`"训练"过程`, 每个问题5个rollout, 一次2个问题, 一共3个epoch
5. 监控训练过程的平均打分到wandb, 结束后在测试集上测试. 
6. 使用最开始的最简单SKILL, 在测试集跑一次, 对比准确度


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

- `run_sql(sql)`：只读执行，30 秒硬超时；最多向模型返回 100 行并说明是否存在更多结果；每个单元格最多返回前 500 字符及原长度。SQL 语法或数据错误返回给模型，数据库系统错误直接终止。
- `semantic_search(query, n, table)`：只允许查询已生成 embedding 的表，返回 table、row_id、video_id、start/end、规范化文本预览及 cosine score。
- `clip_qa(video_id, start_s, end_s, question)`：建议片段不超过 10 分钟，超过 20 分钟直接拒绝。内部 agent 可调用 `sample_video`、`add_note`、`read_notes`。
- `sample_video`：接受时间、fps 和 0–1000 整数裁剪坐标；返回覆盖请求区间的 caption 和带时间标记的 4×3 contact sheet，每页 12 帧，所有帧完整分页。contact sheet 和证据截图只向模型暴露文件名，不暴露宿主机绝对路径。
- `clip_qa` 返回 `answered|unknown`、答案和至少一个证据时间点；harness 将证据时间点转换为独立截图交给主 agent。
- 主 agent 提供 `add_note`、`read_notes` 和 `agent_raise(reason)`；正式运行中同一 rollout 最多按原设置重试 3 次，缺少必要数据或确定性代码错误直接终止。
- 主 agent 最大 50 turns；达到上限按整个 rollout 重试，不截断上下文或返回半成品。

### Compaction 状态机

- 仅主检索 agent 使用 compaction；`clip_qa`、judge、reflection、summarizer 和 updater 不使用。
- 每次主 agent 响应完成后读取服务端 usage，计算 `remaining = notes/INFRA.txt.context_limit - input_tokens - output_tokens`。
- 当 `remaining <= 50,000` 时，通过流式 runner 执行 `cancel(mode="after_turn")` 并排空事件，确保当前工具轮完整结束。
- 在当前主检索 agent 的同一对话中追加固定 compaction 指令，要求它停止工具调用并总结当前对话。该 agent 的下一次输出作为新对话中的 trusted handoff，旧对话不再传入；不创建单独的 compactor agent。
- 第一次和第二次 compaction 后继续 rollout。第三次 compaction 输出生成后立即终止，状态设为 `compaction_limit`，分数设为 0，不调用 judge、不重试，直接进入重点分析绕圈原因的 reflection。
- Reflection 永不 compaction；当其 remaining 小于等于 50k 时，禁止继续调用工具并强制输出最终 reflection 和 meta-reflection。

#### "训练"过程:
1. 我们定义batch size代表n个问题和每个问题m个平行rollout. 把SKILL.md放入prompt对于一整个batch做平行rollout, 得到25个结果, 这是用用一个额外的judge call来判断模型答案正确性(打分/5), 这个分数记作s. wandb log这个s. 这个s也是qwen 3.8 27b产生. 
2. 把正确答案和s(而不只是s)都给每一个rollout, 然后告诉他这是正确答案和你的分数, 同时插入META-SKILL.md (META-SKILL.md不是最开始放进去的), 你现在要做的就是去重新复盘这个结果, 包括重新回顾自己的trace和重新对数据库检索和探索, 找到你为什么对/为什么错的原因, 把这个原因总结出来, 记错relection, 同时这个反思过程有什么经验(帮助下一次反思), 记错meta-relection. 两个反思都要去掉具体内容变得抽象化 可以辅助任何问题而不是本次情况. 本次情况可以作为案例, 但是要去掉敏感信息防止数据泄露(比如具体某天中午午饭内容). 注意要模型不光总结新的经验, 还要让她注意什么时候要更新/替换/修改旧的经验. 不然经验会越变越大而且出现重复/冲突/相似的内容.
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
1. 不允许静默失败, 调试时显式失败, 最终运行是重试/丢弃失败样本, 工具失败可以返回给agent让agent处理, 但是如果agent call了raise, 那么就要skip/停止, 根据是调试阶段(stop)还是运行阶段(retry, if failed 3 times, skip, 对于function级别错误, 比如api错误, 重试这个function, 如果是agent造成的错误, 返回错误值让agent自己修, 如果function不能retry (比如状态已经被破坏, 一个有副作用的funtion跑了一半. 或者是rollout自己崩了), 那么retry整个rollout)
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
8. 轮数实际设置为50; 达到50说明模型自循环, 重试整个样本.
9. 允许并行工具调用.
10. uv cache放到/raid.
11. home目录满了时先清理home, 删除本次产生的无用内容.
12. 过程中严谨访问未列出的目录和当前目录. 

## 当前实现状态与已完成修改

### 框架、依赖与权限规则
1. Agent harness 已改回 `openai-agents`，使用 OpenAI Agents SDK 的 `OpenAIResponsesModel`、`Runner.run_streamed` 和 Responses API，不再使用自制 agent loop。
2. Responses API harness 使用 `reasoning_item_id_policy="preserve"`，本地模型服务同时启用 `enable_thinking=true` 和 `preserve_thinking=true`，保留多轮 reasoning item。
3. 主 agent、`clip_qa`、judge、reflection、question summarizer 和 skill updater 都使用 Agents SDK；需要结构化输出的阶段使用 Pydantic `output_type`。
4. Python 环境使用 uv，项目依赖已写入 `pyproject.toml` 和 `uv.lock`，包括 `openai-agents`、`openai`、DuckDB、Qwen VL embedding 所需依赖、W&B 和 pytest。
5. 全局 `AGENTS.md` 已加入规则：修改、停止、重启或重建容器不需要再次向用户请求允许。

### Pilot 数据、数据库与 split
1. 已实现统一文本归一化：转小写、折叠连续空白、去除首尾空白；Latin 非 ASCII 字符转成 ASCII，非 Latin 字符保留。
2. 已从预构建材料和 14 个原始视频重建 `data/pilot/pilot.duckdb`。当前行数为：videos 14、episodes 11311、visual_clips 9167、entities 23447、semantic_triples 54375、openie_triples 92296、topic_chains 1680、topic_facts 10359、event_chains 173、event_chain_entities 1130、event_steps 775、graph_nodes 98300、graph_edges 243291。
3. DuckDB 同时保存结构化 SQL 表以及用于 split 的 graph_nodes/graph_edges；检索 harness 使用 SQL 表，不使用 PPR。
4. 已按问题和 clue video 的连通分量完成 split，生成 `data/pilot/train.json` 646 个问题和 `data/pilot/test.json` 24 个问题，两个集合不共享连通的视频分量。
5. 已把原视频按实际边界缓存为 30 秒左右的 MP4 clip，并核对每个视频的 clip 数量、单段时长总和与源视频时长；`visual_clips` 的时间边界已按实际缓存片段回写数据库。
6. 数据库、clip、embedding shard、embedding index、运行 trace、media、W&B 和 runs 输出均已加入忽略规则，不进入 git 历史。

### Embedding
1. 已启动 `Qwen/Qwen3-VL-Embedding-8B` pooling 服务，served model name 为 `qwen3-vl-embedding-8b`，host 端口为 8013；文本和视频都使用同一个 VL embedding 模型。
2. 文本表按 batch 512 生成可恢复 shard；视频表按 clip 文件生成可恢复 shard；所有最终向量为 float32、4096 维并与 row_id 顺序逐项校验。
3. 已生成 episodes 11311、entities 23447、semantic_triples 54375、openie_triples 92296、topic_facts 10359、event_steps 775 的文本索引。
4. 已处理 9167 个 visual clip；3 个 clip 的解码器明确返回无下一帧错误，因此完整跳过对应 clip 并记录错误，没有替代输入；最终 visual_clips 视频索引包含 9164 行。
5. `semantic_search` 从 mmap NPY 索引执行精确 cosine similarity，并回查 DuckDB 返回对应 row_id、video_id、时间范围、文本预览和分数。

### 本地 Qwen 服务
1. 当前主模型为 `Inferact/Qwen3.8-27B-NVFP4`，运行在 GPU 5、TP=1、host 端口 8000。
2. 当前服务参数为 max model length 262144、max batched tokens 32768、max sequences 100、image limit 1024、video limit 0、KV cache FP8、GPU memory utilization 0.75、chunked prefill、async scheduling 和 MTP 3 speculative tokens。
3. 当前服务使用 qwen3 reasoning parser、qwen3_coder tool parser、auto tool choice、Responses API，并通过 chat-template kwargs 保留 thinking。
4. 主 agent 和 reflection 使用 `reasoning_effort=medium`；未设置 temperature、top_p、客户端 input token cap 或 completion token cap。

### Harness、工具与 trace
1. `run_sql` 使用只读 DuckDB，并关闭 external access；SQL 查询 30 秒超时，最多返回 100 行，每个单元格最多给模型 500 字符及原始长度。
2. `sample_video` 支持时间范围、fps、0–1000 整数裁剪坐标和 caption；输出完整分页的 4×3 contact sheet，每页最多 12 帧。
3. `clip_qa` 是嵌套 Agents SDK agent，可调用 `sample_video`、`add_note` 和 `read_notes`；返回 `answered|unknown`、答案和证据时间点，随后生成独立证据截图给主 agent。
4. 主 agent 可调用 `run_sql`、`semantic_search`、`clip_qa`、`add_note`、`read_notes` 和 `agent_raise`；问题输入不包含正确答案。
5. 主 agent 的上下文阈值、after-turn cancellation、handoff 和最多三次 compaction 状态机已实现；第三次 compaction 结束后记 `compaction_limit` 和 0 分并进入 reflection。
6. `clip_qa` 和 reflection 各只保留 `TURN_SAFETY_LIMIT=1000` 的死循环保护，不再暴露 `--clip-max-turns` 或 `--reflection-max-turns`，正常流程不以该值截断。
7. Agents SDK 结构化输出解析失败时，harness 从真实 `response.completed` 事件保留本轮失败的 reasoning 和 assistant output，把原解析错误追加到同一对话后继续生成，不改用其他 parser、模型或方法；轮数和 usage 继续累计。
8. 每段 trace 在磁盘保存清理后的 input、responses、items、raw stream events、usage 和 final output；base64 图片内容替换为占位符，工具参数、图片项和 reasoning item ID 保留。
9. Reflection 输入保留完整语义 trace，但不重复传入已由 responses/items 表达的逐 token raw stream events；磁盘原始 trace 不变。
10. 已用真实多轮模型调用验证主 agent 调用 `clip_qa`、`clip_qa` 调用 `sample_video`、图片返回、shared notes、reasoning item 保留、结构化错误纠正和最终结构化答案；生成的 contact sheet 与证据截图已实际检查。

### 训练、恢复与验证
1. 正式训练配置保持每个问题 5 个 rollout、每 batch 2 个问题、3 个 epoch、每个可重试阶段最多 3 次尝试；主 agent 仍保持 50 turns 上限。
2. Judge 只接收 `{reference_answer, candidate_answer}`，输出 0–5 整数分；主 agent 永远不接收 reference answer。
3. 每个成功执行的训练 rollout 只创建一次 reflection 任务；该任务可在一次 reflection 内进行多轮工具检索，最终输出 reflection 和 meta_reflection。
4. Reflection 接收 question、candidate answer、reference answer、score、rollout status、完整语义 trace、当前 SKILL 和 META-SKILL；它可以重新查询数据库或视频，并输出去具体内容后的可复用经验。
5. 同一问题的多个 rollout reflection 先汇总为 question summary，再由 updater 输出完整替换的 SKILL.md 和 META-SKILL.md；下一个 batch 使用更新后的文件。
6. 初始文件已创建为 `skills/initial/SKILL.md` 和 `skills/initial/META-SKILL.md`，baseline 使用同一个初始 SKILL，不使用空 prompt。
7. Retryable API、模型解析、超时、max-turn 和正式模式的 `agent_raise` 错误按原方法重试；确定性输入、权限、认证、坏请求和代码错误直接终止，不使用替代方法。
8. 进程取消会写入 `cancelled.json` 并传播，取消的 attempt 不占重试预算；恢复时创建新的物理 attempt 编号。
9. 重试耗尽的 rollout 保存为 `status=failed`、score=0；训练和验证均把所有指定 rollout 的分数纳入均值，失败项不会从分母移除。
10. 同一 batch 的并行 rollout 和 question summary 使用 `asyncio.TaskGroup`；一个任务异常时取消并等待同组任务完成取消，再关闭共享客户端。
11. 训练和验证按 batch 保存 stage input、attempt trace、usage、result、SKILL before/after、META-SKILL before/after 和 completed marker；恢复时核对已保存输入和配置，完成 batch 可直接继续。
12. W&B 初始化记录 run id、git hash、git status、路径、模型、数据和协议配置；每个 batch 先写 W&B 指标，再写 completed marker，避免恢复时跳过未记录的指标。
13. 训练记录 completed rollouts、failed rollouts 和包含 0 分失败项的 mean score；验证记录相同统计并生成最终 validation summary，空测试集直接失败。

### 当前验证结果和未执行项
1. 当前测试为 15 项，全部通过；测试覆盖归一化、schema、split、只读 SQL、工具拒绝、并发视频采样、gold answer 隔离、重试、取消恢复、失败计 0、结构化输出纠错、trace 图片清理和 reflection trace 去重。
2. 当前主服务已按 TP=1、max sequences=100 重建并完成真实 Responses API 多轮 harness 验证。
3. 正式训练、训练后测试集验证和初始 SKILL baseline 对比尚未启动。
4. 正式训练启动前仍按规则执行一次 Claude CLI 全仓库审查；若修复仅是小范围修改，再由新的 REVIEW.md 审查代理复核修复后直接开始正式运行。
