这是一个AI Memory Agent的项目, 你的目标是构建一个agent, 用来检索视频记忆并且回答问题. 本branch是从main checkout出来的, 不要查看git历史, main是一次失败的尝试, ../video_sql也是, 除非特别指示, 不然不要去看, 以免遭受误导. 如果过程中用户更新了plan的内容, 记得要更新这个文件. 

严格按照这个plan来做, 即使你觉得不符合常理/不正确. 

## 材料:
1. 数据集: https://huggingface.co/datasets/MM-Lifelong/MM-Lifelong
2. 旧的预处理仓库: https://github.com/lijiazheng0917/MAGIC-video.git
3. 使用的Harness框架: https://github.com/openai/openai-agents-python
5. Embedding模型: Qwen/Qwen3-VL-Embedding-8B (Use VL for both visual and text-only), start it using vllm
6. 数据库引擎: https://github.com/duckdb/duckdb
7. 本地模型: Qwen/Qwen3.8-Flash-Next-FP8, 使用参考在根目录`INFRA.txt`第一条
8. index: 试点和正式运行全部使用现有全量index `data/pilot/stage2/index.duckdb` 和 `data/pilot/stage2/embedding_index`(读取方式见notes/index.md), 不使用jiazhengli7/magic-video-artifacts
9. notes/文件夹, 上一次失败过程积累的经验和踩坑, 可以参考, 但是不一定有用甚至可能跑偏, 冲突时以当前计划为准


## 项目创新点:
1. 本项目使用SQL Dataset代替graph, 从而使得query过程可以被agent控制, 并且可以直接实现多步骤跳转而不是agent自己relay. 
2. Agent SKILL.md "训练": 训练大模型很复杂, 耗费算力, 容易过拟合, 所以我们"训练"agent的skill.md文件, 而不是模型参数. 简单来说, 每一次rollout结束之后模型复盘, 复盘经验加入一个文本文件, 这个文本被下一次模型rollout使用, 如此反复迭代实现模型进化. 
3. 以往的index构建过于繁琐复杂, 我们使用一个简单的一次过的方式快速处理. 
4. 我们让模型自己反思, 而不是把trace给另一个agent, 这样模型知道当时自己的错误原因(例如保留思维链). 
5. 我们使用多个SKILL文件, 而不是一个, 模型按需读取. 
6. 不是一次性update(像trace2skill), 目的: 可以学习meta skill, 一次更新模型基础能力弱可能很多问题根本没法采样到正确答案并且没有回炉修改机会, 只能按照cross batch的拉扯. 我们给answer, 不是只有reward. Batch Update也需要硬件资源可以跑的起这么多Batch.

## 试点实验
1. 使用notes/index.md的方式读取index, 直接使用MM-Lifelong自带的数据集划分作为training和testing set, 不使用旧`src/splits.py`的自定义连通分量划分. 训练集=`month/train.json`(266题). 测试集从`month/val.json`、`day/test.json`、`week/test.json`各随机抽100条, 共300条, 固定seed; 抽样结果保存成文件提交到git, 初始SKILL和训练后SKILL测试使用同一份.
2. 搭建我们的harness,  见Harness架构如何搭建, 使用OpenAI Agents SDK连接本地模型, 验证他保留了过往reasoning并且支持图片或视频输入. 验证方式: 不允许preflight, 正式跑一个batch之后停下来检查, 通过后从第二个batch接着跑
3. 在训练集上跑我们的`"训练"过程`, 每个问题5个rollout, 一次4个问题, 一共3个epoch. 为了测试我们的harness, 我们用一个快速的云端模型(gpt-5.6-luna)来验证我们的想法, 用.env中的lab api key来call openai官方的api, 如果lab的不行就换personal的api key, 只跑4次update, 这个是因为云端模型更快可以快速验证我们的harness, 如果出现错误, 回去用这个云端模型反复迭代. 通过后进入下一步. 试点中所有LLM角色(主检索agent、clip_qa内部agent、sql_qa单次call、judge、summarizer、skill updater)都用gpt-5.6-luna, embedding仍用本地Qwen3-VL-Embedding-8B服务; compaction的context_limit和阈值与本地一样(262,144 / 50k); 试点直接用uv venv在宿主机跑, 不用Docker, 同样log到wandb和weave.
4. 用本地的qwen 27b跑完整的训练过程, log到wandb**和**weave. 
5. 使用`skills/initial/`作为初始SKILL版本, 通过和test相同的val接口在测试集跑一次, 与训练后的SKILL准确度对比. 测试=300题 x 每题5个rollout, 只跑答题+judge, 不跑reflection/summarizer/updater. 准确度定义为所有rollout的judge平均分 s/5. 初始和训练后各自一个wandb run.


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

#### Prompt
System Prompt包含简单的任务说明和当前全量Stage 2 database schema, **不需要**任何详细的任务步骤描述, 例如如何搜索如何继续. 答题prompt给出问题文本和该题所属的逻辑时间轴(day/week/month), 不给question_type、clue_intervals、temporal_certificate. 起始状态使用一个SKILL目录；答题prompt只列出SKILL的name和description, 由模型通过`read_skill`按需读取. 反思阶段的guideline保持最简单, 具体的反思方法放在`META-SKILL.md`, 避免system prompt与可迭代的SKILL冲突. val不实现把初始SKILL内容直接放进system prompt的独立no-skill baseline；初始版本和训练后版本使用同一接口, 区别仅是分别读取`skills/initial/`和指定run的SKILL目录.

#### Harness 工具

- `clip_qa(video_id, start_s, end_s, question)`：建议片段不超过 10 分钟，超过 20 分钟直接拒绝。内部 agent 可调用 `sample_video`、`add_note`、`read_notes`。
- `clip_qa` 返回 `answered|unknown`、答案和证据时间点；只有 `answered` 必须包含至少一个证据时间点，`unknown` 可以没有证据时间点。harness 将已有证据时间点转换为独立截图交给主 agent。
- `sample_video`：接受时间、fps、是否返回 caption 的 `include_captions`，以及 0–1000 整数裁剪坐标；由模型决定是否返回覆盖请求区间的 caption，同时返回带时间标记的 4×3 contact sheet，每页 12 帧，所有帧完整分页。contact sheet 和证据截图只向模型暴露文件名，不暴露宿主机绝对路径。
- `run_sql(sql)`：只读执行，30 秒硬超时；最多向模型返回 100 行并说明是否存在更多结果；每个单元格最多返回前 500 字符及原长度。所有执行错误作为 tool result 返回给模型处理。
- `sql_qa(sql, question)`：和run_sql类似, 接受一个sql命令, 同时接受一个问题, 这个sql命令和问题被提交执行, 但是结果不直接返回给模型, 而是由另一个single LLM call对于question进行回答和总结, 对于大面积搜索推荐agent使用这个工具, 以避免context过大. 交给这个single call的结果不限制行数, 只限制总字符数: 超过50k字符的截断, 并在末尾写明已被截断; 30秒SQL超时同样适用.
- `semantic_search(query, n, table)`：只允许查询已生成 embedding 的表，返回 table、row_id、video_id、start/end、规范化文本预览及 cosine score。允许传入 threshold min/max 或 n-th 到 m-th 的 rank range；例如已经查看 top 10 后，可以直接查询第 10–20 名，不需要重新调用 top 20。
- harness 使用当前全量 Stage 2 schema，包括 `object_states`、source mapping、evidence 表以及 `6h`/`video`/`scale` granularity；不使用旧 `src/schema.sql` 中的 `openie_triples` schema。读取画面时使用 `visual_clips.filename` 和 source offsets，不根据 row ID 拼接旧 clip 路径。
- 主 agent 还提供 `add_note`、`read_notes`、`read_skill(name)` 和 `agent_raise(reason)`。

#### Agent 停止、错误与 trace
1. 主检索 agent、`clip_qa` 和 reflection 各自使用适配自身输出结构的 `submit_answer` tool；模型调用后退出对应 agent loop，不依赖 Agents SDK 的 `output_type` 结构化最终输出来结束 loop。
2. skill updater 不使用 `submit_answer`；完成全部文件操作后调用专用 `stop` tool 退出。
3. 所有 agent loop 的总轮数限制统一为 100。触发 100 时本次 trace 作废，并按正式模式重试整个 rollout，不截断上下文或返回半成品。
4. 如果连续 3 次 agent completion 在排除 reasoning 后的 raw characters 完全一致，本次 trace 作废，并按正式模式重试整个 rollout。
5. 所有 trace 都保存，包括因重复输出或达到 100 轮而作废的 trace。JSON trace 删除 base64 图片数据，图片独立保存为 JPG，并在 trace 中保留对应文件名。
6. 普通 FunctionTool 执行错误统一以`{"error": "..."}`tool result返回调用它的agent处理，不直接使rollout或TaskGroup crash；`agent_raise(reason)`保持为agent主动结束当前rollout的信号。统一使用JSON是因为v18曾把SDK默认错误文本交给会执行JSON解码的reflection trace，导致`JSONDecodeError`并终止整批运行。
7. 接近context limit时关闭工具并强制输出；除主检索agent的正式compaction流程外，其他agent不做compaction。

#### SKILL 目录与读取
1. SKILL是包含多个Markdown文件的目录。每个普通SKILL对应一种特殊情况，包含YAML frontmatter（`name`、`description`，description即“何时使用”）、Introduction、When to use、Main content；SQL示例直接写在Markdown中，不单独保存脚本文件。结构参考Anthropic官方skill-creator。
2. Anthropic的skill-creator是“如何写SKILL”；本项目的`META-SKILL.md`是“如何反思”，二者不是同一概念。skill updater不注入skill-creator完整正文，只使用根据其格式浓缩的手写规则。
3. 首批文件位于`skills/initial/`：`general-retrieval.md`、`sql-schema.md`、`semantic-search.md`、`locating-time-intervals.md`、`counting-and-ordering.md`、`visual-verification.md`和`META-SKILL.md`；旧的单文件`SKILL.md`已删除。
4. `general-retrieval.md`说明完整检索循环和何时读取其他SKILL；`sql-schema.md`说明DuckDB表、字段和连接关系；`semantic-search.md`说明语义检索及从候选结果转入SQL验证；`locating-time-intervals.md`说明根据时间线索定位和扩展时间窗口；`counting-and-ordering.md`说明去重、计数、聚合和排序；`visual-verification.md`说明何时以及如何调用`clip_qa`做画面核验；`META-SKILL.md`说明答题agent如何复盘、抽象经验以及更新、替换或删除旧经验。
5. 每个run在`<run_dir>/skills/`持有独立Git仓库：训练启动时从`skills/initial/`复制并直接`git init`后做首次commit；它不是主代码仓库的worktree或branch。`META-SKILL.md`位于该独立仓库内，val读取指定run的该目录。
6. rollout prompt只列出可用SKILL的`name`和`description`；`read_skill(name)`返回对应文件的完整内容。成功返回的原始Markdown保留在答题agent的同一对话中；工具错误返回统一error JSON。Reflection不建立trace投影，也不对成功的Markdown工具返回执行JSON解码。

#### Compaction

- 仅主检索 agent 使用 compaction；`clip_qa`、judge、reflection、summarizer 和 updater 不使用。
- harness 的 `context_limit` 为 262,144；vLLM后端通过模型官方YaRN配置启用314,573 token上限、factor=1.2，作为client compaction size之上的20%缓冲。
- 每次主 agent 响应完成后读取服务端 usage，计算 `remaining = INFRA.txt.context_limit - input_tokens - output_tokens`。
- 当 `remaining <= 50,000` 时，通过流式 runner 执行 `cancel(mode="after_turn")` 并排空事件，确保当前工具轮完整结束。
- 在当前主检索 agent 的同一对话中追加固定 compaction 指令，要求它停止工具调用并总结当前对话。由原来的同一个 Agent instance 生成 trusted handoff，不 clone Agent，也不创建单独的 compactor agent。该输出作为新对话中的 trusted handoff，旧对话不再传入。
- 第一次和第二次 compaction 后继续 rollout。第三次 compaction 输出生成后立即终止，状态设为 `compaction_limit`，分数设为 0，不调用 judge、不重试，直接进入重点分析绕圈原因的 reflection。
- Reflection 永不 compaction；当其 remaining 小于等于 50k 时，禁止继续调用工具并强制输出最终 reflection 和 meta-reflection。

### "训练"过程
1. batch size代表n个问题和每个问题m个平行rollout。每个rollout在prompt中获得当前SKILL列表并按需读取；一个batch得到20个结果后，用额外的judge call判断答案正确性并打分（满分5），这个分数记作s并log。judge只接收reference_answer和candidate_answer, 不接收问题、trace、证据；评分标准是最终答案语义等价, 忽略附带的video id、时间戳、row id等标识符。正式训练的judge同样使用`Qwen/Qwen3.8-Flash-Next-FP8`。
2. 把正确答案和s(而不只是s)都给每一个rollout, 然后告诉他这是正确答案和你的分数(注意! 不是另一个agent, 是告诉答题agent自己), 同时插入META-SKILL.md (META-SKILL.md不是最开始放进去的). 保留答题时原来的同一个Agent instance和同一对话继续reflection, 不clone Agent, 不创建新的reflection agent, 也不把trace序列化后交给另一个agent. 你现在要做的就是去重新复盘这个结果, 包括重新回顾自己的trace和重新对数据库检索和探索, 找到你为什么对/为什么错的原因, 把这个原因总结出来, 记错relection, 同时这个反思过程有什么经验(帮助下一次反思), 记错meta-relection. 两个反思都要去掉具体内容变得抽象化 可以辅助任何问题而不是本次情况. 本次情况可以作为案例, 但是要去掉敏感信息防止数据泄露(比如具体某天中午午饭内容). 注意要模型不光总结新的经验, 还要让她注意什么时候要更新/替换/修改旧的经验. 不然经验会越变越大而且出现重复/冲突/相似的内容. reflection阶段保留答题时的全部工具(run_sql、sql_qa、semantic_search、clip_qa、add_note、read_notes、read_skill、agent_raise), 只把`submit_answer`换成`submit_reflection`.
3. 每个问题的m个rollout反思由另一个call汇总，再由skill updater更新SKILL目录和`META-SKILL.md`。更新时既可以增加，也要删除、替换重复、冲突、过时或相似的内容。updater每个batch跑一次, 一次agent loop输入该batch全部4个问题的summary, commit一次。
4. 下一个batch使用更新后的SKILL目录继续，形成跨batch的反复迭代，而不是一次性update。

#### Skill updater agent loop
1. updater是多轮agent loop，可以增加、修改和删除普通SKILL Markdown文件，也可以修改`META-SKILL.md`，但明确禁止删除`META-SKILL.md`。
2. updater可以查看独立SKILL Git仓库的历史并读取旧版本，参考后重新写入当前文件；不向updater暴露或执行直接Git rollback。
3. updater完成全部文件操作后调用专用`stop` tool退出；一轮完整update的所有文件操作只commit一次。
4. 每个updater Git commit的body列出本次更新所依据的完整rollout ID（`epoch/batch/question/rollout/attempt`）以及总体修改原因。保存的SKILL中每个指导点后包含简短的`**Reason:** ...`；一条原因覆盖整个section时，可以只在section末尾记录一次，便于以后推翻或修改旧方案以及debug追溯。
5. updater失败后在同一个SKILL worktree原地重试，不清理本次失败已经产生但尚未commit的修改；下一次attempt读取当前status和diff后继续完成同一轮update。

### 训练与运行配置
1. 正式训练每batch 4个问题、每个问题5个并行rollout，共20个并行rollout；一共3个epoch，每个epoch开始时重排训练问题顺序，seed由run参数给出，默认0，第k个epoch用seed+k；服务并发上限为24。
1a. 采样参数(temperature/top_p/top_k等)一律不设, 使用服务端/generation_config默认; 本地Qwen和云端gpt-5.6-luna都只传reasoning effort。
1b. W&B entity/project为`3dsmile/video-sql-v2`, weave同project。
2. 主模型使用官方`Qwen/Qwen3.8-Flash-Next-FP8`。负责prefill和decode的主模型服务运行在GPU 0,2,5,7，TP=4，端口8002，max-num-seqs 24，gpu-memory-utilization=0.90，不使用`--no-async-scheduling`，MTP speculative length为2，default chat-template kwargs保持`enable_thinking=true`和`preserve_thinking=true`；agent reasoning effort使用`medium`。Next Flash不使用TP=2，因为该配置此前没有可用KV cache；改用TP=4并保留官方FP8以避开NVFP4重复reasoning问题；MTP=2用于避免MTP=4造成吞吐过低。
3. 正式训练client Docker暴露GPU 0,2,4,5,7，供W&B采集三个模型服务的GPU指标；镜像使用`docker/training/Dockerfile`构建并安装`ffmpeg`，因为`sample_video`会调用`ffmpeg`。
4. 一个batch的全部rollout都失败时跳过该batch的updater；连续两个batch的全部rollout都失败时终止训练。
5. OpenAI SDK内部`max_retries`为3。`--attempts`独立应用于整个rollout、judge、reflection、summarizer和updater，这些阶段分别最多按该值重试，正式值为3。
6. 不使用Disaggregated Encoder.
7. 初始SKILL与训练后SKILL的test使用相同的batch size和每题rollout数。

## 代码结构
src/harness.py: 主harness架构, 包括tools定义
src/training.py 训练代码
src/val.py 验证集代码, 使用相同的SKILL列表和`read_skill`接口分别测试`skills/initial/`和指定run的SKILL目录
tests/ 所有单元测试
runs/ 记录训练的log和缓存的SKILL.md和META-SKILL.md, 包括超参数和git hash
notes/ 记录给你自己的note, 包括注意事项和踩过的坑, 被context compact之后可能需要参考
其他你觉得需要的文件

## 规则
1. 不允许静默失败。调试时显式失败；现在已进入正式运行，所有 FunctionTool 执行错误都作为 tool result 返回给调用它的 agent 处理，不直接使 rollout 或 TaskGroup crash。如果 agent 主动调用 `agent_raise`，则调试阶段停止、正式运行按原设置重试，3 次失败后跳过；agent loop 外的失败继续按对应的 retry 或 skip 规则处理。
2. 每次开始运行前, 使用REVIEW.md审查代码
3. 使用git做记录, 以方便回滚, 实验结果需要配对的git hash, 注意修改.gitignore防止不需要的文件进入历史. git操作时, 不要用系统默认用户名, 创建一个虚拟的name=codex email=codex@localhost来分别作者
4. 保持代码简洁干净, 不要过度设计和抽象, 这是科研代码
5. 代码必须在batch level可重新开始. 中断后重启时, 未完成的batch重用已保存的已完成rollout(含reflection), 只补跑缺失的, 然后继续该batch的summarizer和updater
6. 所有run必须log到wandb
7. 本台机器有8xB200和2TB RAM, 性能强劲, 但是不要用很多GPU, 单个GPU经常跑不满, 所以尝试用单个GPU负载. 
8. 全程前台监控, 但是不要频繁查看, 刚开始时每30s看一次, 后面逐步改成10min, 30min, 1h. 每次不要直接查看日志, 不然你会看到很多噪音. 脚本的详细log不要输出到stdout而是log文件, stdout只输出你需要监控的信息, 并且你只看这些. 
9. 用openai的agent sdk来写agent, 不要自己手搓harness, 用response api, 保留过往reasoning
10. 你的运行过程中除非停止, 不然所有消息都默认是你的自言自语用户不会读, 需要用户读取的milestone和log记录在AGENT_STDOUT.txt给用户看, 比如cost, 你自己做的决定, etc. AGENT_STDOUT是你输出给用户看的重要信息, wandb是记录run的过程. 
11. 不运行preflight或smoke test；直接启动正式运行，只用正式运行的结果和完整trace检查正确性。不允许preflight是因为不要跑没用的测试浪费时间; 只要跑的东西后面可以接着用就没有问题, 比如正式跑一个batch, 停下来检查, 然后从第二个batch开始接着跑。
12. 之前把trace交给独立reflection agent的实现及其正式run结果全部作废；不恢复旧代码或旧run，从当前已删除的`src/`、`tests/`和`skills/initial/`状态按本计划重新实现，并使用新的正式run目录从头执行。

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
13. 正式容器不使用`--privileged`、`--network host`或`--gpus all`; 使用项目专用bridge network, 并且每个容器只暴露计划分配给它的GPU.
14. 所有timeline/run都必须对模型完全隔离: agent-facing prompt、schema、tool参数、tool返回、skill内容和数据库视图只能暴露当前任务自己的世界; 不能暴露其它timeline/video/split的存在、名称、数量或可查询性.
