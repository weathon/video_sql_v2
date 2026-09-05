这是一个AI Memory Agent的项目, 你的目标是构建一个agent, 用来检索视频记忆并且回答问题. 本branch是从main checkout出来的, 不要查看git历史, main是一次失败的尝试, ../video_sql也是, 除非特别指示, 不然不要去看(包括简单的git status --short), 以免遭受误导. 如果过程中用户更新了plan的内容, 记得要更新这个文件. 

严格按照这个plan来做, 即使你觉得不符合常理/不正确. 

## 材料:
1. 数据集: https://huggingface.co/datasets/MM-Lifelong/MM-Lifelong；已有本地快照为`/raid/huggingface/hub/datasets--MM-Lifelong--MM-Lifelong/snapshots/248aa82039a574e63a2e524746a7cd8f32330443/`，直接使用，不重新下载。媒体记录中的filename相对于此目录解析，不额外假设`videos/`前缀。
2. 旧的预处理仓库: https://github.com/lijiazheng0917/MAGIC-video.git
3. 使用的Harness框架: https://github.com/openai/openai-agents-python
5. Embedding模型: Qwen/Qwen3-VL-Embedding-8B (Use VL for both visual and text-only), start it using vllm
6. 数据库引擎: https://github.com/duckdb/duckdb
7. 本地模型: Qwen/Qwen3.8-27B-FP8, 使用参考在根目录`INFRA.txt`第一条
8. index: 使用现有全量Stage 2 index `data/pilot/stage2/index.duckdb` 和 `data/pilot/stage2/embedding_index`派生的隔离运行库`data/runtime/{day,week,month}/`；各目录包含`index.duckdb`、`embedding_index`和`media-map.json`。本次直接使用已有运行库，不重建index，不使用jiazhengli7/magic-video-artifacts。
9. notes/文件夹, 上一次失败过程积累的经验和踩坑, 可以参考, 但是不一定有用甚至可能跑偏, 冲突时以当前计划为准


## 项目创新点:
1. 本项目使用SQL Dataset代替graph, 从而使得query过程可以被agent控制, 并且可以直接实现多步骤跳转而不是agent自己relay. 
2. Agent SKILL.md "训练": 训练大模型很复杂, 耗费算力, 容易过拟合, 所以我们"训练"agent的skill.md文件, 而不是模型参数. 简单来说, 每一次rollout结束之后模型复盘, 复盘经验加入一个文本文件, 这个文本被下一次模型rollout使用, 如此反复迭代实现模型进化. 
3. 以往的index构建过于繁琐复杂, 我们使用一个简单的一次过的方式快速处理. 
4. 我们让模型自己反思, 而不是把trace给另一个agent, 这样模型知道当时自己的错误原因(例如保留思维链). 
5. 我们使用多个SKILL文件, 而不是一个, 模型按需读取. 
6. 不是一次性update(像trace2skill), 目的: 可以学习meta skill, 一次更新模型基础能力弱可能很多问题根本没法采样到正确答案并且没有回炉修改机会, 只能按照cross batch的拉扯. 我们给answer, 不是只有reward. Batch Update也需要硬件资源可以跑的起这么多Batch.

## 实验流程
1. 使用notes/index.md的方式读取index, 直接使用MM-Lifelong自带的数据集划分作为training和testing set, 不使用旧`src/splits.py`的自定义连通分量划分. 训练集=`month/train.json`(266题). 测试集从`month/val.json`、`day/test.json`、`week/test.json`各随机抽100条, 共300条, 固定seed; 抽样结果保存成文件提交到git, 初始SKILL和训练后SKILL测试使用同一份.
2. 搭建我们的harness,  见Harness架构如何搭建, 使用OpenAI Agents SDK连接本地模型, 验证他保留了过往reasoning并且支持图片或视频输入. 验证方式: 不允许preflight, 正式跑一个batch之后停下来检查, 通过后从第二个batch接着跑
3. 之前的gpt-5.6-luna云端试点属于历史实验，不是本次重启步骤。本次直接用本地`Qwen/Qwen3.8-27B-FP8`从`skills/initial/`开始正式训练，每batch 3个问题、每题10个rollout、3个epoch；不切换云端模型或API key作为失败后的替代路径。
4. 正式训练所有LLM角色（主检索agent、clip_qa内部agent、sql_qa单次call、judge、summarizer、skill updater）使用同一本地模型，embedding使用本地Qwen3-VL-Embedding-8B服务；log到wandb**和**weave。
5. 使用`skills/initial/`作为初始SKILL版本, 通过和test相同的val接口在测试集跑一次, 与训练后的SKILL准确度对比. 测试=300题 x 每题5个rollout, 只跑答题+judge, 不跑reflection/summarizer/updater. 准确度定义为所有rollout的judge平均分 s/5. 初始和训练后各自一个wandb run.


## 方法详情
### 介绍
本方法基于MAGIC-Video的index, 但是MAGIC-video把已经表格化的内容(triplet, caption, etc)作为graph的PPR检索, 检索过程agent不可控. 所以我们把同样的index作为sql数据库检索, agent写SQL来检索数据库. 
下面是说明关系查询能力的概念性SQL例子，不是当前运行库可直接执行的schema。实际工具SQL以`src/harness.py`的SCHEMA为准；TIME_OF_INTEREST是占位符表示需要query的时间。
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
System Prompt包含简单的任务说明、当前隔离运行库的完整SCHEMA，以及自动注入的`ROOT-SKILL.md`全文；具体检索方法保留在可更新的SKILL中。答题prompt给出问题文本，不给question_type、clue_intervals、temporal_certificate，也不暴露其它timeline/video/split；逻辑时间轴由调用方选择对应运行库。普通SKILL只列name和description，由模型通过`read_skill`按需读取；ROOT-SKILL告诉模型什么时候读取哪些SKILL。反思时插入`META-SKILL.md`，并保留最低工具调用次数和证据要求。val不实现独立no-skill baseline；初始版本和训练后版本使用同一接口，包括自动注入各自的ROOT-SKILL，区别仅是读取`skills/initial/`或指定run的SKILL目录。

#### Harness 工具

- `clip_qa(start_s, end_s, question)`：片段超过20分钟直接拒绝，起止时间必须非负且递增。内部agent可调用`sample_video`、`add_clip_note`、`read_clip_notes`和`submit_clip_answer`；不向模型暴露video_id。
- `clip_qa` 返回 `answered|unknown`、答案和证据时间点；只有 `answered` 必须包含至少一个证据时间点，`unknown` 可以没有证据时间点。harness将answered结果的证据时间点转换为独立截图交给主agent。
- `sample_video`：接受时间、fps、是否返回 caption 的 `include_captions`，以及 0–1000 整数裁剪坐标；由模型决定是否返回覆盖请求区间的 caption，同时返回带时间标记的 4×3 contact sheet，每页 12 帧，所有帧完整分页。contact sheet 和证据截图只向模型暴露文件名，不暴露宿主机绝对路径。
- `run_sql(sql)`：只读执行，30 秒硬超时；最多向模型返回 100 行并说明是否存在更多结果；每个单元格最多返回前 500 字符及原长度。所有执行错误作为 tool result 返回给模型处理。
- `sql_qa(sql, question)`：和run_sql类似, 接受一个sql命令, 同时接受一个问题, 这个sql命令和问题被提交执行, 但是结果不直接返回给模型, 而是由另一个single LLM call对于question进行回答和总结, 对于大面积搜索推荐agent使用这个工具, 以避免context过大. 交给这个single call的结果不限制行数, 只限制总字符数: 超过50k字符的截断, 并在末尾写明已被截断; 30秒SQL超时同样适用.
- `semantic_search(query, table, n, rank_start, min_score=null, max_score=null)`：schema明确列出已生成embedding的7张表：`episodes`、`entities`、`semantic_triples`、`topic_facts`、`event_steps`、`object_states`、`visual_clips`。返回table、row_id、start_s/end_s、preview和cosine score，不返回video_id。rank_start从1开始，n为返回排名数；分数边界使用JSON null或省略，不使用字符串`"None"`。校验失败返回实际参数、schema和纠正提示。
- harness使用隔离后的Stage 2 schema，包括`object_states`和evidence表；不使用旧的`openie_triples` schema。读取画面时通过内部`media-map.json`映射时间、原视频路径和source offset，不使用不存在于运行库的`visual_clips.filename`，也不根据row ID拼接旧clip路径。
- 主 agent 还提供 `add_note`、`read_notes`、`read_skill(name)` 和 `agent_raise(reason)`。
- 本地Qwen通过Responses EPD proxy访问；proxy先把每个`input_image`提交给独立encoder实例，再把原始Responses请求提交给prefill/decode实例。

#### Agent 停止、错误与 trace
1. 主检索agent、clip_qa和reflection分别使用`submit_answer`、`submit_clip_answer`和`submit_reflection`退出对应agent loop；不依赖Agents SDK的`output_type`结构化最终输出来结束loop。
2. skill updater 不使用 `submit_answer`；完成全部文件操作后调用专用 `stop` tool 退出。
3. 所有 agent loop 的总轮数限制统一为 100。触发 100 时本次 trace 作废，并按正式模式重试整个 rollout，不截断上下文或返回半成品。
4. 如果连续 3 次 agent completion 在排除 reasoning 后的 raw characters 完全一致，本次 trace 作废，并按正式模式重试整个 rollout。
5. 所有 trace 都保存，包括因重复输出或达到 100 轮而作废的 trace。JSON trace 删除 base64 图片数据，图片独立保存为 JPG，并在 trace 中保留对应文件名。
6. 普通 FunctionTool 执行错误统一以`{"error": "..."}`tool result返回调用它的agent处理，不直接使rollout或TaskGroup crash；`agent_raise(reason)`保持为agent主动结束当前rollout的信号。统一使用JSON是因为v18曾把SDK默认错误文本交给会执行JSON解码的reflection trace，导致`JSONDecodeError`并终止整批运行。
7. 主检索agent和reflection接近context limit时使用下面规定的compaction流程；clip_qa、judge、summarizer和updater不做compaction，也没有单独的接近阈值强制输出分支。

#### SKILL 目录与读取
1. SKILL是包含多个Markdown文件的目录。每个普通SKILL对应一种特殊情况，包含YAML frontmatter（`name`、`description`，description即“何时使用”）、Introduction、When to use、Main content；SQL示例直接写在Markdown中，不单独保存脚本文件。结构参考Anthropic官方skill-creator。
2. Anthropic的skill-creator是“如何写SKILL”；本项目的`META-SKILL.md`是“如何反思”，二者不是同一概念。skill updater不注入skill-creator完整正文，只使用根据其格式浓缩的手写规则。
3. 首批文件位于`skills/initial/`：`ROOT-SKILL.md`、`general-retrieval.md`、`sql-schema.md`、`semantic-search.md`、`locating-time-intervals.md`、`counting-and-ordering.md`、`visual-verification.md`和`META-SKILL.md`；旧的单文件`SKILL.md`已删除。
4. `general-retrieval.md`说明完整检索循环和何时读取其他SKILL；`sql-schema.md`说明DuckDB表、字段和连接关系；`semantic-search.md`说明语义检索及从候选结果转入SQL验证；`locating-time-intervals.md`说明根据时间线索定位和扩展时间窗口；`counting-and-ordering.md`说明去重、计数、聚合和排序；`visual-verification.md`说明何时以及如何调用`clip_qa`做画面核验；`META-SKILL.md`说明答题agent如何复盘、抽象经验以及更新、替换或删除旧经验。
5. 每个run在`<run_dir>/skills/`持有独立Git仓库：训练启动时从`skills/initial/`复制并直接`git init`后做首次commit；它不是主代码仓库的worktree或branch。`META-SKILL.md`位于该独立仓库内，val读取指定run的该目录。
6. ROOT-SKILL全文自动注入；普通SKILL只列`name`和`description`，`read_skill(name)`返回对应文件全文。ROOT-SKILL和META-SKILL不作为普通SKILL列入catalog，由updater直接更新维护。成功返回的原始Markdown保留在答题agent的同一对话中；工具错误返回error JSON。Reflection不建立trace投影，也不对成功的Markdown工具返回执行JSON解码。

#### Compaction

- 主检索agent和reflection各自允许3次compaction；clip_qa、judge、summarizer和updater不使用。
- harness 的 `context_limit` 为 262,144；vLLM后端通过模型官方YaRN配置启用314,573 token上限、factor=1.2，作为client compaction size之上的20%缓冲。
- 每次主检索agent或reflection响应完成后读取服务端usage，计算`remaining = CONTEXT_LIMIT - input_tokens - output_tokens`，常量位于`src/harness.py`。
- 当 `remaining <= 50,000` 时，通过流式 runner 执行 `cancel(mode="after_turn")` 并排空事件，确保当前工具轮完整结束。
- 在当前主检索 agent 的同一对话中追加固定 compaction 指令，要求它停止工具调用并总结当前对话。由原来的同一个 Agent instance 生成 trusted handoff，不 clone Agent，也不创建单独的 compactor agent。该输出作为新对话中的 trusted handoff，旧对话不再传入。
- 第一次和第二次 compaction 后继续 rollout。第三次 compaction 输出生成后立即终止，状态设为 `compaction_limit`，分数设为 0，不调用 judge、不重试，直接进入重点分析绕圈原因的 reflection。
- Reflection使用原session的同一个Agent instance，不clone；当remaining小于等于50k时生成reflection handoff，保留问题、答案、分数、证据、已做检查及未解决问题。前两次compact后继续检索，第三次compact后仅允许提交reflection和meta-reflection。Handoff和instructions保留在原session的agent中。
- Reflection阶段至少完成2次工具调用才允许提交；答题阶段调用和submit_reflection本身不计入。未满足最低调用次数时，不得强制产出有效reflection。

### "训练"过程
1. batch size代表n个问题和每个问题m个平行rollout。每个rollout在prompt中获得当前SKILL列表并按需读取；每个rollout答题后立即用额外的judge call判断答案正确性并打分（满分5），再进入reflection；这个分数记作s并log。完整batch汇集30个rollout结果，epoch末不足3题的batch按实际题数运行。judge只接收reference_answer和candidate_answer, 不接收问题、trace、证据；评分标准是最终答案语义等价, 忽略附带的video id、时间戳、row id等标识符。正式训练的judge同样使用`Qwen/Qwen3.8-27B-FP8`。
2. 把正确答案和s(而不只是s)都给每一个rollout, 然后告诉他这是正确答案和你的分数(注意! 不是另一个agent, 是告诉答题agent自己), 同时插入META-SKILL.md (META-SKILL.md不是最开始放进去的). 保留答题时原来的同一个Agent instance和同一对话继续reflection, 不clone Agent, 不创建新的reflection agent, 也不把trace序列化后交给另一个agent. 你现在要做的就是去重新复盘这个结果, 包括重新回顾自己的trace和重新对数据库检索和探索, 找到你为什么对/为什么错的原因, 把这个原因总结出来, 记错relection, 同时这个反思过程有什么经验(帮助下一次反思), 记错meta-relection. 两个反思都要去掉具体内容变得抽象化 可以辅助任何问题而不是本次情况. 本次情况可以作为案例, 但是要去掉敏感信息防止数据泄露(比如具体某天中午午饭内容). 注意要模型不光总结新的经验, 还要让她注意什么时候要更新/替换/修改旧的经验. 不然经验会越变越大而且出现重复/冲突/相似的内容. reflection阶段保留答题时的全部工具(run_sql、sql_qa、semantic_search、clip_qa、add_note、read_notes、read_skill、agent_raise), 只把`submit_answer`换成`submit_reflection`.
2a. Reflection prompt和`META-SKILL.md`要求：必须在reflection过程中通过实际检索证据成功找到正确答案，才可以写入经验；获得reference answer本身不算找到答案，不能把随意推测的错误原因写成经验。如果最终没能找到正确答案，只产生一个极小的patch，记录可能的错误原因并明确包含不确定性，由merger决定怎么处理。汇总和updater必须保留这种不确定性，不得将其改写成已证实经验。
3. 每个问题的m个rollout反思由另一个call汇总，再由skill updater更新SKILL目录和`META-SKILL.md`。更新时既可以增加，也要删除、替换重复、冲突、过时或相似的内容。updater每个batch跑一次, 一次agent loop输入该batch全部3个问题的summary, commit一次。
4. 下一个batch使用更新后的SKILL目录继续，形成跨batch的反复迭代，而不是一次性update。

#### Skill updater agent loop
1. updater是多轮agent loop，可以增加、修改和删除普通SKILL Markdown文件，也可以修改`META-SKILL.md`和`ROOT-SKILL.md`，但明确禁止删除这两个文件。
2. updater可以查看独立SKILL Git仓库的历史并读取旧版本，参考后重新写入当前文件；不向updater暴露或执行直接Git rollback。
3. updater完成全部文件操作后调用专用`stop` tool退出；一轮完整update的所有文件操作只commit一次。
4. 每个updater Git commit的body列出本次更新所依据的完整rollout ID（`epoch/batch/question/rollout/attempt`）以及总体修改原因。保存的SKILL中每个指导点后包含简短的`**Reason:** ...`；一条原因覆盖整个section时，可以只在section末尾记录一次，便于以后推翻或修改旧方案以及debug追溯。
5. updater失败后在同一个SKILL worktree原地重试，不清理本次失败已经产生但尚未commit的修改；下一次attempt读取当前status和diff后继续完成同一轮update。
6. 每次update建议像小学习率/gradient clipping一样，只做少量、局部、证据支持的修改，尽量保留已有有效经验；这是建议性语言，不设置硬性修改额度。

### 训练与运行配置
1. 正式训练每batch 3个问题、每个问题10个并行rollout，共30个rollout；一共3个epoch，每个epoch开始时重排训练问题顺序，seed由run参数给出，默认0，第k个epoch用seed+k；服务并发上限为24。
1c. 保留平均分，并记录完全正确的rollout占比：得分严格大于4分算完全正确，分母为该batch全部rollout，未完成的不计为完全正确。指标名为`train/fully_correct_portion`，同时保存到batch结果。
1a. 采样参数(temperature/top_p/top_k等)一律不设, 使用服务端/generation_config默认; 本地Qwen和云端gpt-5.6-luna都只传reasoning effort。
1b. W&B entity/project为`3dsmile/video-sql-v2`, weave同project。
2. 主模型使用`Qwen/Qwen3.8-27B-FP8`。独立encoder实例运行在GPU 5；负责prefill和decode的实例运行在GPU 0,2，TP=2；Responses EPD proxy使用主机端口8002。prefill/decode实例的max-num-seqs为24、gpu-memory-utilization为0.90，不使用`--no-async-scheduling`，MTP speculative length为3，default chat-template kwargs保持`enable_thinking=true`和`preserve_thinking=true`；agent reasoning effort使用`medium`。两个模型实例使用`ECExampleConnector`，共享的encoder cache位于Docker tmpfs volume中。
3. 正式训练client Docker暴露GPU 0,2,4,5，供W&B采集三个模型服务的GPU指标；镜像使用`docker/training/Dockerfile`构建并安装`ffmpeg`，因为`sample_video`会调用`ffmpeg`。
4. 一个batch的全部rollout都失败时跳过该batch的updater；连续两个batch的全部rollout都失败时终止训练。
5. OpenAI SDK内部`max_retries`为3。`--attempts`独立应用于整个rollout、judge、reflection、summarizer和updater，这些阶段分别最多按该值重试，正式值为3。
6. 使用vLLM Disaggregated Encoder，部署为一个encoder实例和一个prefill/decode实例；encoder cache位于内存。
7. 初始SKILL与训练后SKILL的test使用相同的batch size和每题rollout数。
8. 本次从头训练：从`skills/initial/`初始化新的run，不继承旧run的训练后SKILL，epoch 0不跳过问题；保留旧run和已有数据，不重新下载数据集或重建index。

## 代码结构
src/harness.py: 主harness架构, 包括tools定义
src/training.py 训练代码
src/val.py 验证集代码, 使用相同的SKILL列表和`read_skill`接口分别测试`skills/initial/`和指定run的SKILL目录
src/prepare_runtime.py: 从已有Stage 2 index派生各timeline运行库和媒体映射；本次训练不重跑
runs/ 记录训练的log和缓存的SKILL.md和META-SKILL.md, 包括超参数和git hash
notes/ 记录给你自己的note, 包括注意事项和踩过的坑, 被context compact之后可能需要参考
其他你觉得需要的文件

## 规则
1. 不允许静默失败。调试时显式失败；现在已进入正式运行，所有 FunctionTool 执行错误都作为 tool result 返回给调用它的 agent 处理，不直接使 rollout 或 TaskGroup crash。如果 agent 主动调用 `agent_raise`，则调试阶段停止、正式运行按原设置重试，3 次失败后跳过；agent loop 外的失败继续按对应的 retry 或 skip 规则处理。
2. 代码审查按AGENTS.md执行：REVIEW.md只由新的review subagent自行读取，主agent不读。正式运行前使用Claude CLI做全仓库审查。委派禁止`fork_turns="all"`或省略该参数。
3. 使用git做记录, 以方便回滚, 实验结果需要配对的git hash, 注意修改.gitignore防止不需要的文件进入历史. git操作时, 不要用系统默认用户名, 创建一个虚拟的name=codex email=codex@localhost来分别作者
4. 保持代码简洁干净, 不要过度设计和抽象, 这是科研代码
5. 代码必须在batch level可重新开始. 中断后重启时, 未完成的batch重用已保存的已完成rollout(含reflection), 只补跑缺失的, 然后继续该batch的summarizer和updater
6. 所有run必须log到wandb和weave。
7. GPU按INFRA.txt和本计划的训练配置分配，不再尝试将当前TP=2及独立encoder/embedding部署改为单GPU。
8. 全程使用harness前台session监控；稳定后按AGENTS.md降低轮询频率，不每30秒或60秒轮询。详细日志落盘，终端只读取有界里程碑和错误；正式结果必须核对实际trace，不能只看输出数量或进程存活。
9. 用openai的agent sdk来写agent, 不要自己手搓harness, 用response api, 保留过往reasoning
10. 你的运行过程中除非停止, 不然所有消息都默认是你的自言自语用户不会读, 需要用户读取的milestone和log记录在AGENT_STDOUT.txt给用户看, 比如cost, 你自己做的决定, etc. AGENT_STDOUT是你输出给用户看的重要信息, wandb是记录run的过程. 
11. 不运行preflight或smoke test；直接启动正式运行，只用正式运行的结果和完整trace检查正确性。不允许preflight是因为不要跑没用的测试浪费时间; 只要跑的东西后面可以接着用就没有问题, 比如正式跑一个batch, 停下来检查, 然后从第二个batch开始接着跑。
12. 之前把trace交给独立reflection agent的实现及其正式run结果全部作废；使用当前已实现的同一session reflection代码和新的正式run目录从头训练，不删除或重新实现现有src和skills/initial，也不覆盖旧run。

## 实现决定
1. 文本归一化采用NFKD、移除组合标记、将映射的标点转成ASCII并转小写；保留非英文字符，不将全部文本强制丢弃为ASCII。
2. 去除文本开始和结尾的空格, 所有连续空白折叠成单个空格, 总是允许一定程度的改变和坍缩, 保持一致优先.
3. 使用当前uv.lock锁定的依赖和现有服务镜像；不因本次训练重启自动升级、降级或切换依赖。
4. 环境使用uv; 不需要每次都创建新的venv, 不能使用别人创建的venv, 自己创建的venv可以复用.
5. 保留所有非英文字符; 英文字符使用ASCII.
6. infra benchmark使用更长的测试文本和多轮对话.
7. `run_sql`每个单元格保留前500字符.
8. 轮数实际设置为100; 达到100说明模型自循环, 重试整个样本.
9. 允许并行工具调用.
10. uv cache放到/raid.
11. 不自动清理home；HF cache使用`/raid/huggingface`。删除操作必须明确目标、符合用户授权，不删除现有数据集或训练产物。
12. 过程中严谨访问未列出的目录和当前目录. 
13. 正式容器不使用`--privileged`、`--network host`或`--gpus all`; 使用项目专用bridge network, 并且每个容器只暴露计划分配给它的GPU.
14. 所有timeline/run都必须对模型完全隔离：使用已有day/week/month三个独立运行库，每次只连接当前任务自己的数据库；agent-facing prompt、schema、tool参数、tool返回和skill内容不能暴露其它timeline/video/split的存在、名称、数量或可查询性。
