这是一个AI Memory Agent的项目, 你的目标是构建一个agent, 用来检索视频记忆并且回答问题. 本branch是从main checkout出来的, main是一次失败的尝试, ../video_sql也是, 除非特别指示, 不然不要去看, 以免遭受误导. 如果过程中用户更新了plan的内容, 记得要更新这个文件. 

## 材料:
1. 数据集: https://huggingface.co/datasets/MM-Lifelong/MM-Lifelong
2. 旧的预处理仓库: https://github.com/lijiazheng0917/MAGIC-video.git
3. 使用的Harness框架: https://github.com/HKUDS/OpenHarness.git
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
2. 搭建我们的harness,  见Harness架构如何搭建, 使用open harness连接本地模型, 验证他保留了过往reasoning并且支持图片或视频输入
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
