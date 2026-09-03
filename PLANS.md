这是一个AI Memory Agent的项目, 你的目标是构建一个agent, 用来检索视频记忆并且回答问题. 本branch是从main checkout出来的, 不要查看git历史, main是一次失败的尝试, ../video_sql也是, 除非特别指示, 不然不要去看, 以免遭受误导. 如果过程中用户更新了plan的内容, 记得要更新这个文件. 

## 材料:
1. 数据集: https://huggingface.co/datasets/MM-Lifelong/MM-Lifelong
2. 旧的预处理仓库: https://github.com/lijiazheng0917/MAGIC-video.git
3. 使用的Harness框架: https://github.com/openai/openai-agents-python
5. Embedding模型: Qwen/Qwen3-VL-Embedding-8B (Use VL for both visual and text-only), start it using vllm
6. 数据库引擎: https://github.com/duckdb/duckdb
7. 本地模型: Inferact/Qwen3.8-Flash-Next-NVFP4, 使用参考在notes/INFRA.txt第一条
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
4. 在训练集上跑我们的`"训练"过程`, 每个问题10个rollout, 一次5个问题, 一共3个epoch
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
- 所有 agent loop 最大 100 turns；达到上限按整个 rollout 重试，不截断上下文或返回半成品。

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
8. 轮数实际设置为100; 达到100说明模型自循环, 重试整个样本.
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
6. 本地 OpenAI SDK 客户端不要求或注入 `OPENAI_API_KEY`；不提供 `api_key`，关闭 SDK 的凭证强制检查，并使用不含密钥的空 `Authorization` 头通过客户端校验。
7. 正式训练使用 `vllm/vllm-openai:qwen38` Docker 镜像和 host network，以 root 用户运行；仓库读写挂载，host uv Python、`/usr/bin/git` 和用于 W&B 登录的 `/home/wg25r/.netrc` 只读挂载，不传入 OpenAI key。

### Pilot 数据、数据库与 split
1. 已实现统一文本归一化：转小写、折叠连续空白、去除首尾空白；Latin 非 ASCII 字符转成 ASCII，非 Latin 字符保留。
2. 已从预构建材料和 14 个原始视频重建 `data/pilot/pilot.duckdb`。当前行数为：videos 14、episodes 11311、visual_clips 9167、entities 23447、semantic_triples 54375、openie_triples 92296、topic_chains 1680、topic_facts 10359、event_chains 173、event_chain_entities 1130、event_steps 775、graph_nodes 98300、graph_edges 243291。
3. DuckDB 同时保存结构化 SQL 表以及用于 split 的 graph_nodes/graph_edges；检索 harness 使用 SQL 表，不使用 PPR。
4. 已按问题和 clue video 的连通分量完成 split，生成 `data/pilot/train.json` 646 个问题和 `data/pilot/test.json` 24 个问题，两个集合不共享连通的视频分量。
5. 已把原视频按实际边界缓存为 30 秒左右的 MP4 clip，并核对每个视频的 clip 数量、单段时长总和与源视频时长；`visual_clips` 的时间边界已按实际缓存片段回写数据库。
6. 数据库、clip、embedding shard、embedding index、运行 trace、media、W&B 和 runs 输出均已加入忽略规则，不进入 git 历史。
7. 官方 dense caption 存在但文本为空时立即抛出错误，不改用 Whisper 或 Qwen 生成内容替代。

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
6. `clip_qa` 和 reflection 各只保留 `TURN_SAFETY_LIMIT=1000` 的死循环保护，不再暴露 `--clip-max-turns` 或 `--reflection-max-turns`，正常流程不以该值截断。`clip_qa` 在 remaining 小于等于 50k 时完整结束当前工具轮，随后关闭工具并强制返回最终 `ClipAnswer`；若结构化解析错误发生在该区间，则从真实 `response.completed` usage 计算 remaining，在保留同一对话的 reasoning、assistant output 和解析错误后关闭工具继续纠正，避免保留工具 schema 的解析纠正循环先于 1000 turns 撞到服务物理上下文上限。不设置输入或输出 token cap。
7. Agents SDK 结构化输出解析失败时，harness 从真实 `response.completed` 事件保留本轮失败的 reasoning 和 assistant output，把原解析错误追加到同一对话后继续生成，不改用其他 parser、模型或方法；轮数和 usage 继续累计。
8. 每段 trace 在磁盘保存清理后的 input、responses、items、raw stream events、usage 和 final output；base64 图片内容替换为占位符，工具参数、图片项和 reasoning item ID 保留。
9. Reflection 输入只保留最新有效 `main-segment` 的 `input`、`is_complete`、`current_turn` 和包含真实工具返回的 `items`；保留该段全部 reasoning、function call 和 function output，把 JSON 字符串形式的工具参数与返回解码为等价结构化对象后传入 prompt。compaction 后该段的 developer trusted handoff 已概括旧主段，主段 `items` 已包含主 agent 实际接收的 `clip_qa` 返回，因此不再重复传入旧主段、compaction trace、嵌套 clip trace、`responses`、逐 token raw stream events、usage 或已单独提供的 final output，磁盘原始 trace 不变。Reflection 首次请求前调用本地模型 `/tokenize`；若初始输入已进入现有 50k remaining 区间，则从第一次请求起关闭工具并直接生成 reflection。结构化解析失败后从真实 `response.completed` 事件携带该失败响应的 input/output token 数并重新计算 remaining，不依赖解析失败时可能为空的 SDK `raw_responses`；若已进入 50k 区间，则在保留同一对话的 reasoning、assistant output 和解析错误后关闭工具继续纠正。不设置输入或输出 token cap。
10. 已用真实多轮模型调用验证主 agent 调用 `clip_qa`、`clip_qa` 调用 `sample_video`、图片返回、shared notes、reasoning item 保留、结构化错误纠正和最终结构化答案；生成的 contact sheet 与证据截图已实际检查。
11. 主 agent compaction 后的新上下文保留 developer trusted handoff，并重新附加原始问题作为 user 消息，确保本地 Responses 服务收到 user query。
12. 主 agent 只有返回非空最终答案才记为 `completed`；若模型耗尽物理上下文后只返回 reasoning、没有 message 或答案，则抛出可重试的 `ModelBehaviorError`，由现有正式模式规则重试整个 rollout，不把空字符串送入 judge 和 reflection。

### 训练、恢复与验证
1. 正式训练配置保持每个问题 5 个 rollout、每 batch 2 个问题、3 个 epoch、每个可重试阶段最多 3 次尝试；主 agent 仍保持 50 turns 上限。
2. Judge 只接收 `{reference_answer, candidate_answer}`，输出 0–5 整数分；主 agent 永远不接收 reference answer。
3. 每个成功执行的训练 rollout 只创建一次 reflection 任务；该任务可在一次 reflection 内进行多轮工具检索，最终输出 reflection 和 meta_reflection。
4. Reflection 接收 question、candidate answer、reference answer、score、rollout status、完整语义 trace、当前 SKILL 和 META-SKILL；它可以重新查询数据库或视频，并输出去具体内容后的可复用经验。
5. 同一问题的多个 rollout reflection 先汇总为 question summary，再由 updater 输出完整替换的 SKILL.md 和 META-SKILL.md；下一个 batch 使用更新后的文件。
6. 初始文件已创建为 `skills/initial/SKILL.md` 和 `skills/initial/META-SKILL.md`，baseline 使用同一个初始 SKILL，不使用空 prompt。
7. Retryable API、模型解析、超时、max-turn 和正式模式的 `agent_raise` 错误按原方法重试；Agents SDK `UserError` 包装的 `httpx` 或 `httpx2` request error 先取出原始原因后按同一规则分类；确定性输入、权限、认证、坏请求和代码错误直接终止，不使用替代方法。原因是正式 v16 的嵌套 `clip_qa` 收到 `httpx2.RemoteProtocolError`，该异常属于 `httpx2.RequestError`，但旧分类只包含 `httpx2.TimeoutException`。
8. 进程取消会写入 `cancelled.json` 并传播，取消的 attempt 不占重试预算；恢复时创建新的物理 attempt 编号。
9. 重试耗尽的 rollout 保存为 `status=failed`、score=0；训练和验证均把所有指定 rollout 的分数纳入均值，失败项不会从分母移除。
10. 同一 batch 的并行 rollout 和 question summary 使用 `asyncio.TaskGroup`；一个任务异常时取消并等待同组任务完成取消，再关闭共享客户端。
11. 训练和验证按 batch 保存 stage input、attempt trace、usage、result、SKILL before/after、META-SKILL before/after 和 completed marker；恢复时核对已保存输入和配置，完成 batch 可直接继续。
12. W&B 初始化通过当前 SDK 的 `wandb.sdk.lib.runid.generate_id` 生成 run id，并记录 git hash、git status、路径、模型、数据和协议配置；每个 batch 先写 W&B 指标，再写 completed marker，避免恢复时跳过未记录的指标。
13. 训练记录 completed rollouts、failed rollouts 和包含 0 分失败项的 mean score；验证记录相同统计并生成最终 validation summary，空测试集直接失败。

### 当前验证结果和未执行项
1. 当前测试为 15 项，全部通过；测试覆盖归一化、schema、split、只读 SQL、工具拒绝、并发视频采样、gold answer 隔离、重试、取消恢复、失败计 0、结构化输出纠错、trace 图片清理和 reflection trace 去重。
2. 当前主服务已按 TP=1、max sequences=100 重建并完成真实 Responses API 多轮 harness 验证。
3. Docker 正式训练已产生真实 agent、`clip_qa`、judge 和 reflection trace，并先后暴露两个已修复的确定性错误：主 agent compaction 续段缺少 user query，以及 reflection 重复传入 trace 表示导致初始上下文超过 262144。修复后的真实 reflection 调用以 216241 input tokens 进入模型，执行两次 `run_sql` 后按阈值关闭工具并返回有效结构化结果；随后 v4 运行中的 `clip_qa` 遇到一次 600 秒 `httpx2.ReadTimeout`，因 timeout 类型未被现有分类器识别而终止，该分类缺口已修复。
4. v5 正式训练产生一条已完整核验的成功 rollout，随后一个发生 compaction 的 rollout 在 reflection 首次请求前失败：旧投影仍同时传入旧主段、compaction 输入、嵌套 clip trace 和最新主段，本机 tokenizer 计为 728704 tokens，服务返回 262144 上下文错误。提交 `cbfafb3` 改为只投影最新有效主段；同一失败输入的 Docker 重放不传 OpenAI key，首轮为 203179 input tokens，实际完成 7 次 `run_sql`、1 次 `clip_qa` 及嵌套 `sample_video`，最终强制输出段为 217748 input tokens 并返回有效结构化 reflection。接触表、证据截图、全部工具输入和返回均已检查，独立 REVIEW.md 审查无阻塞项。
5. v6 第一次 Docker 启动因遗漏 Git safe-directory 环境变量而在创建 run 目录前失败；修正后的命令设置 `GIT_CONFIG_COUNT=1`、`GIT_CONFIG_KEY_0=safe.directory` 和 `GIT_CONFIG_VALUE_0=/raid/wg25r/video_sql_v2`。v6 完成两个结果：question 0 rollout 04 得分 2，主 agent 29 turns、1 次 compaction、reflection 10 次请求；question 1 rollout 03 得分 1，主 agent 15 turns、无 compaction、reflection 6 次请求。随后 question 0 rollout 03 的最新主段自身达到 928929 item chars，原始 reflection 输入为 275282 tokens，在首次模型请求前超过上下文。该 trace 包含 18 个 reasoning、49 个 function call 和 49 个 function output。一次临时修改删除 reasoning 后，独立 REVIEW.md 审查指出它丢失唯一决策内容，因此已恢复全部 reasoning。
6. 提交 `0e54628` 保留最新主段全部 reasoning、49 个 function call 和 49 个 function output，同时解码 JSON 字符串嵌套并加入首次 `/tokenize` 预检。同一 v6 失败输入的 Docker 重放不传 OpenAI key，实际仅发出一次模型请求，usage 为 input 252279、output 1021、total 253300；因初始输入已进入 50k remaining 区间而未调用工具，返回有效结构化 reflection。该重放的 trace 投影与模型返回均已检查，独立 REVIEW.md 审查无阻塞项。
7. 正式训练前的 Claude CLI 全仓库审查已完成；审查发现的空验证集防御分支已删除，并由新的 REVIEW.md 审查代理复核通过。正式训练需要使用新的 v7 run 目录重新启动，训练后测试集验证和初始 SKILL baseline 对比尚未执行。
8. v7 使用无 OpenAI key 的 Docker 客户端启动，W&B run 为 `jijpmt12`，完成 5 个 rollout。第一条完成结果的主 trace、41 次工具输入和返回、17 个 reasoning、judge、3 次 reflection 请求、3 张 contact sheet 和 6 张 evidence screenshot 已逐项检查。随后 question 0 rollout 03 的一个 `clip_qa` 在已经确认答案后进入 `add_note` 自循环：第一个 segment 为 16 requests，第二个 segment 为 519 requests 和 518 次 `add_note`；最后一次有效请求为 input 261766、output 378、total 262144，下一次请求达到 262196 并超过 262144 上限，导致 batch 终止。`TURN_SAFETY_LIMIT=1000` 未改变；修复为 remaining 小于等于 50k 时结束当前工具轮并关闭工具强制输出。同一 `clip_qa` 输入的 Docker 重放中，segment 00 到 842 turns，包含 839 次 `add_note` 和 5 次 `sample_video`；最后一次请求为 input 212050、output 170、total 212220、remaining 49924，当前工具轮结束后取消。segment 01 关闭工具，一次请求为 input 211671、output 270、total 211941，返回有效 `ClipAnswer` 并以状态 0 退出；5 张 evidence screenshot 已逐张检查。正式训练需要使用新的 v8 run 目录重新启动。
9. v8 在增量 REVIEW.md 审查和 Claude CLI 全仓库审查均无 plain-yes 项后启动，W&B run 为 `m70lgl7i`。服务器重启中断训练后，主模型按 GPU 5、TP=1、max sequences=100 重建，embedding 在 GPU 4 重建并以真实请求返回 4096 维、平方范数 0.9999999802708379 的向量；v8 从同一 run 与 W&B run 恢复。batch 000000 完成 9 个 rollout、失败 1 个，10 个分数全部写入，mean score 为 1.2。batch 000001 的 question 3 rollout 02 在 reflection 中产生 208 个 segment：第 3 至 206 段没有 function call，但请求仍包含工具 schema；每段 assistant output 都包含完整 JSON 后跟 `</parameter></function></tool_call>`，Agents SDK 按真实错误继续结构化纠正，input tokens 从 128921 增长到 261547。第 207 段请求达到至少 262145 input tokens，服务返回 400 并终止 batch。修复为结构化解析错误后依据最后一次真实 response usage 检查 remaining，进入 50k 区间后关闭工具继续同一对话的结构化纠正；`TURN_SAFETY_LIMIT=1000` 不变，不设置输入或输出 token cap。正式训练需要使用新的 run 目录重新启动。
10. 第一次用 v8 原失败 reflection 输入进行 Docker 重放时，第三个 segment 已收到 `response.completed`，usage 为 input 129574、output 792，但 Agents SDK 因结构化解析失败未把该响应加入 `raw_responses`，旧取值触发 `IndexError`。提交 `17d1716` 让 `StreamModelBehaviorError` 从真实 completed 事件携带失败响应 token 数，reflection 直接用该值计算 remaining；提交 `ae7e0f9` 同步已有测试桩。Docker 项目测试为 15 passed。第二次用相同失败输入完整重放，共 17 次模型请求、27 次工具调用和 27 个逐一配对返回，无工具错误；第一段最终结构化输出解析失败后，同一对话的第二段返回有效 `reflection` 和 `meta_reflection`。定向阈值验证记录的工具数为 `[6, 0]`，确认失败响应达到 220000 input、1000 output tokens 时，下一段移除全部工具并返回结构化结果。正式训练需要使用新的 run 目录重新启动。
11. v9 在新的增量 REVIEW.md 审查和 Claude CLI 全仓库审查均无 plain-yes 项后，以提交 `0fdc4ee` 启动，W&B run 为 `0tq672bi`；batch 000000 已有 7 个 rollout 写出结果，其中 6 个 completed、1 个 failed。question 0 rollout 00 的主 agent 最后一次响应为 input 102871、output 159273、total 262144 tokens，真实输出只有 665973 字符的 reasoning，没有 message、function call 或最终答案；旧逻辑把空字符串记为 completed，随后包含该真实 reasoning 的单段 reflection 输入超过 262144 token，首次请求返回 400 并终止 batch。提交 `192c2be` 改为主 agent 空最终答案时抛出可重试 `ModelBehaviorError`，按原方法和设置重试整个 rollout。Docker 项目测试为 15 passed；定向验证确认该错误在正式模式可重试，非空答案仍按 completed 返回。正式训练需要使用新的 run 目录重新启动。
12. v10 以提交 `c0af586` 启动，W&B run 为 `ryyg9sha`；batch 000000 写出 5 个 rollout 结果，其中 4 个 completed、1 个在 3 次 `clip_qa` `httpx2.ReadTimeout` 后 failed。question 0 rollout 03 的一个 `clip_qa` 产生 254 个 segment：模型从第 1 段起反复输出有效 `ClipAnswer` JSON 后追加 `</parameter></function></tool_call>`，结构化解析恢复分支持续保留工具 schema；第 252 段为 input 261211、output 875，第 253 段 prompt 达到 262144 后返回 400 并终止 batch。提交 `3342eff` 让 `clip_qa` 的结构化解析错误恢复也按真实失败响应 usage 检查 50k remaining，并在该区间关闭工具继续同一对话。Docker 项目测试为 15 passed。使用原视频区间、问题和调用前 7 条 shared notes 的完整 Docker 重放返回有效 `ClipAnswer`，3 次工具调用与 3 个返回逐一配对且无错误，5 页 contact sheet 和 4 张 evidence screenshot 已实际检查；该随机重放在低 token 阶段成功。定向阈值验证记录工具数为 `[3, 0]`，确认 input 220000、output 1000 的解析失败后下一段移除全部工具并返回结构化答案。正式训练需要使用新的 run 目录重新启动。
13. v11 的 question 1 rollout 00 中，`clip-46e519c544234f9ab66d5a764371e3c2-segment-00` 在一次 `clip_qa` 内执行了 587 次模型请求，其中 585 次调用 `add_note`；末尾反复交替写入 `Final. Conversation complete.` 和 `Final acknowledgment. Conversation complete.`。第 588 次流式响应触发 `ResponseFunctionToolCallItem.name=None`，客户端收到 incomplete chunked read；vLLM 引擎随后仍保留 1 个请求持续生成。MTP 接受率在此前为 88.3%–97.7%，从该残留请求开始连续显示 100.0%。



## Harness 重新设计（v12 起）

### Agent 停止与循环检测
1. 主检索 agent、`clip_qa` 和 reflection 各自增加一个适配自身输出结构的 `submit_answer` tool；模型通过调用它提交最终答案，调用后退出对应 agent loop。不再依赖 Agents SDK 的 `output_type` 结构化最终输出来结束 loop。
2. skill updater 不使用 `submit_answer`；它完成全部文件操作后调用专用 `stop` tool 退出。
3. 所有 agent loop 的总轮数限制统一为 100。触发 100 视为本次运行出问题：本次 trace 作废，并按现有正式模式规则重试整个 rollout。
4. 如果连续 3 次 agent completion 在排除 reasoning 后的 raw characters 完全一致，本次 trace 作废，并按现有正式模式规则重试整个 rollout。
5. 所有 trace 都必须保存，包括因重复输出或达到 100 轮而作废的 trace。
6. 现有接近 context limit 时关闭工具、强制输出的逻辑保留不变。

### 训练配置
1. 正式训练改为每 batch 2 个问题、每个问题 10 个并行 rollout；降低同时运行的 rollout 数量，避免 active KV cache 超出当前容量。
2. 主模型使用官方 `Qwen/Qwen3.8-27B-FP8`，启动命令原样使用 `notes/INFRA.txt` 第一条：GPU 0,2,5,7，TP=4，端口 8000，max-num-seqs 200，gpu-memory-utilization=0.95，关闭 async scheduling，MTP speculative length 为 4，default chat-template kwargs 保持 `enable_thinking=true` 和 `preserve_thinking=true`；agent reasoning effort 使用 `medium`。改用官方 FP8 是为了排除已观察到的 NVFP4 重复 reasoning 与 ModelOpt-NVFP4 路径之间的关联，同时 medium 避免长 agent 使用默认 xhigh 产生过长 reasoning。
3. harness 的 `context_limit` 保持 262,144，并保留接近该边界时的 compaction；vLLM 后端通过模型官方 YaRN 配置启用 314,573 token 上限、factor=1.2，作为 client compaction size 之上的 20% 缓冲。
4. 正式训练 client Docker 暴露与 vLLM 相同的 GPU 0,2,5,7，供 W&B 采集 GPU 指标。
5. 一个 batch 的全部 rollout 都失败时跳过该 batch 的 updater；连续两个 batch 的全部 rollout 都失败时终止训练。
6. 保留 OpenAI SDK 内部重试，并将 `max_retries` 设为 3。
7. 正式训练 client 使用 `docker/training/Dockerfile` 构建的派生镜像，并安装 `ffmpeg`；原因是 `sample_video` 会调用 `ffmpeg`。

### SKILL 目录
1. SKILL 不再是单个 Markdown 文件，而是一个包含多个 Markdown 文件的目录；每个文件对应一种特殊情况，包含 YAML frontmatter（`name`、`description`，description 即“何时使用”）、Introduction、When to use、Main content；SQL 示例直接写在 Markdown 中，不单独保存脚本文件。结构参考 Anthropic 官方 skill-creator。
2. Anthropic 的 skill-creator 是“如何写 SKILL”；本项目的 `META-SKILL.md` 仍然是“如何反思”。
3. 首批文件位于 `skills/initial/`：`general-retrieval.md`、`sql-schema.md`、`semantic-search.md`、`locating-time-intervals.md`、`counting-and-ordering.md`、`visual-verification.md`，以及 `META-SKILL.md`。旧的单文件 `SKILL.md` 已删除。
4. 每个 run 在 `<run_dir>/skills/` 内持有自己的 SKILL 独立 Git 仓库：训练启动时从 `skills/initial/` 复制并 `git init` 后做首次 commit；`META-SKILL.md` 放在该独立仓库内。val 读取指定 run 的该目录。

### Rollout 读取 SKILL
1. prompt 不再放入全部 SKILL 内容，只列出可用 SKILL 的 `name` 和 `description`。
2. 新增 `read_skill(name)` tool，返回对应文件的完整内容。
3. Reflection trace 保留 `read_skill` 成功返回的原始 Markdown，只对该工具的 error JSON 返回执行 JSON 解码；原因是正式 v15 的真实 Markdown 返回触发了 `JSONDecodeError`。

### Skill updater agent loop
1. Reflection 汇总完成后的 SKILL 更新改为多轮 agent loop；updater 可以增加、修改和删除 SKILL Markdown 文件，也可以修改 `META-SKILL.md`。
2. updater 可以查看 Git 历史并读取旧版本内容，参考后重新写入当前文件；不向 updater 暴露或执行直接 Git rollback。
3. 一轮完整 update 的所有文件操作完成后只 commit 一次。
4. updater 完成全部文件操作后调用专用 `stop` tool 退出。
5. 每个 updater Git commit 的 body 必须列出本次更新所依据的完整 rollout ID（`epoch/batch/question/rollout/attempt`）以及 updater 提交的总体修改原因。保存的 SKILL 中每个指导点后必须包含简短的 `**Reason:** ...`；如果一条原因覆盖整个 section，可以只在该 section 末尾记录一次。这样便于以后推翻或修改之前的方案以及 debug 追溯。
