**This plan is a draft and has been refined into CODEX_PLAN.md**

这是一个AI Memory Agent的项目, 你的目标是构建一个agent, 用来检索视频记忆并且回答问题. 

## 材料:
1. 数据集: https://huggingface.co/datasets/MM-Lifelong/MM-Lifelong
2. 预处理仓库: https://github.com/lijiazheng0917/MAGIC-video.git
3. pilot使用模型: OpenAI的gpt-5.6-luna(支持高并发(100左右)和图片输入)
4. 使用SDK: https://github.com/openai/openai-agents-python (文档在docs/)
5. Embedding模型: Qwen/Qwen3-VL-Embedding-8B (Use VL for both visual and text-only), start it using vllm
6. 缓存的试点数据集和预构建的index: /raid/wg25r/mmlifelong_pilot
7. 数据库引擎: https://github.com/duckdb/duckdb
8. OpenAI Doc Skill: 本次运行明确允许且需要使用这个doc来完成构建, 上面的agent辅助doc也需要使用
9. 本地模型: Qwen/Qwen3.8-27B

## 项目创新点:
1. 本项目使用SQL Dataset代替graph, 从而使得query过程可以被agent控制, 并且可以直接实现多步骤跳转而不是agent自己relay. 
2. Agent SKILL.md "训练": 训练大模型很复杂, 耗费算力, 容易过拟合, 所以我们"训练"agent的skill.md文件, 而不是模型参数. 简单来说, 每一次rollout结束之后模型复盘, 复盘经验加入一个文本文件, 这个文本被下一次模型rollout使用, 如此反复迭代实现模型进化. 

## 成本控制和监控
所有的API都应该统计token的消耗, luna的pricing如下. 建议监控长context和短context. 始终需要开启cache write. 如果总消耗达到$100, 保留可以resume的状态并且暂停, 告知用户, 用户决定继续. pilot使用luna, 不要fast模式; pilot之后的所有实验全部使用本地模型, 不再使用luna. 
| Model | Short context input | Short context cached input | Short context cache writes | Short context output | Long context input | Long context cached input | Long context cache writes | Long context output |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-5.6-sol | $4.00 | $0.40 | $5.00 | $20.00 | $8.00 | $0.80 | $10.00 | $30.00 |
| gpt-5.6-terra | $2.00 | $0.20 | $2.50 | $12.00 | $4.00 | $0.40 | $5.00 | $18.00 |
| gpt-5.6-luna | $0.20 | $0.02 | $0.25 | $1.20 | $0.40 | $0.04 | $0.50 | $1.80 |

## 实验
### 阶段一: 试点
0. (跳过这个, 这个已经完成)在试点期间, 同步进行这个操作: 启动一个subagent, 让他从网上(英文社区和中文社区, 还有官方文档)找到最好的vllm/sglang配置跑Qwen/Qwen3.8-27B用的, research哪一个bug更少. 同时这个subagent在一个空的GPU自主测试效果, 用合成长图像文本混合数据集. 每个setting不需要stress test很多prompt, 但是每个prompt需要是合成数据多伦对话长context, 用一个脚本自动的添加用户回复或者让两个对话相互几乎, 包含图片, 简单看看效果就好. 把所有setting和throughput放到INFRA.txt作为best_setting, 子agent不打断你做rollout. 做test的时候要测试高并发(25-50)长context(100k token)测试.
1. 使用已经下载的mm lifelong数据集和预处理的MAGIC-video index (/raid/wg25r/mmlifelong_pilot), 构建我们自己的index: 所有表格全部转化成一个duckdb数据库, 保留原本表格结构, normalize所有的文本和时间格式(绝对时间还是相对时间, 人类可读时间还是秒数, 等, 建议用SQL数据库支持的相对时间格式(有些视频是不知道绝对时间的, 尤其是游戏))使得JOIN更好兼容. duckdb需要开启只读模式. artifacts里面没有embedding, 需要重新抽取embedding, 抽取使用极高并发(around 8192 for text and 512 for video).
2. 构建我们的harness (参考: 方法详情和代码结构)和简单单元测试, 完成这些测试. 完成harness之后, 停下来不要继续, 用户审核代码后再继续. 
3. 对于pilot dataset: 根据视频与问题的连接关系构建图，并进行图划分（graph partitioning），保证训练集和测试集之间不存在重叠的问题连接。较大的分区作为训练集，较小的分区作为验证集.
4. 在训练集上"训练"我们的SKILL.md和META-SKILL.md (Batch Size=5 questions x 5 rollouts, epochs=1), 然后在验证集上**不迭代更新**验证我们的方法. 注意: 测试集不需要META-SKILL.md. 打分方法也用openai api. 训练过程中检查INFRA.txt是否完整.
5. 重跑一次验证集, 但是不包含SKILL.md作为baseline
6. 记录结果, 通过记录的花费推算 cost per question, 报告出来, 推算总花费

### 阶段二: 预处理
1. 下载MM-Lifelong数据集, 有可能已经存在cache
2. 找一个空的GPU, 用docker开启Qwen/Qwen3-VL-Embedding-8B作为embedding server. 找另一个空的GPU, 用best_setting启动Qwen/Qwen3.8-27B本地模型. 如果没有两张空的卡, 挤在一张卡, embedding不需要很多空间, 或者找另一个显存还有剩余但是compute是空的的卡启动embedding. Embedding高并发同上. 
3. 修改MAGIC-video中的预处理脚本, 用本地模型Qwen/Qwen3.8-27B, 生成index, 所以用它作为所有模型, reasoning设置成medium. 使用embedding server作为embedding. 需要全部重新生成不能使用pilot的, 因为模型变了. 产生artifacts非常耗时和资源, 所以必须用本地模型而不是云端. 
4. 储存所有index并且转化成SQL格式保存在一个duckdb database

### 阶段三: "训练"和测试
1. 和试点一样(使用luna), 只不过这次在完整的训练集(mmlifelong官方的split)上面训练, 一样的batch size, 只训练一个epoch, 但是SKILL.md和META-SKILL.md要重新训练, 初始状态和阶段一的**初始**状态一样. 
2. 在mmlifelong的官方测试集采样测试, 每个长度(week, day, month)采样20个问题做测试. 同时测试有SKILL和没有SKILL
3. 记录比较结果. 

### 阶段四: 跨模型测试
1. 启动一个vllm/sglang, 根据best_setting在一个GPU跑Qwen/Qwen3.8-27B [如果所有GPU不可用(VRAM<80% free OR always busy, it is fine if it has a small amount of VRAM being used), 暂停], medium, 测试mmlifelong测试集.由于是本地跑没有成本考量, 所以跑所有的mmlifelong测试集. 同样测试用SKILL.md和没有SKILL.md. 
2. 记录和比较结果, 同样用deepseek/deepseek-v4-flash-0731的judge. 主要内部比较(with/without skill), 但是同时和阶段三比较, 和阶段三比较时比较同一子集. 


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
- run_sql: 简单的运行sql, 需要强行timeout 30s, 不然有可能会写死. 后端连接duckdb来支持高并发. 长caption做截断. 返回数量也做阶段到100行并且提示后面的被隐藏. 防止context爆炸. 
- clip_qa: 由于video信息密度很低而且极度占用token, 我们把video_qa单独拿出来, 这个是一个subagent, 主agent给他一个视频片段(tool介绍里面建议小于10min但不强制, 强制小于20min, 如果输入超过20min返回消息让agent重来选择)和一个问题, 他可以通过类似VideoARM的方式在这个片段进一步检索和回答这个问题, 视频不进入主agent的架构. 这个字agent只有一个简单的sample_video(start_time, end_time, fps, x1, x2, y1, y2, return_caption=True), add_note(text), and read_notes(), 可以在这个10min长度的视频中先低fps采样, 检索然后截取一小段高fps来仔细查看, 还可以使用坐标进行裁剪, 仔细看视频的一个部分. 注意统一一下坐标格式, 用x/1000相对坐标. 这个agent在sample_video时一样可以看见caption用来辅助, 但是返回caption是可选的, 但是一旦选择返回, 返回这个区间内所有caption, 不做裁剪. caption的intervel和agent选择的time frame不一定一样, 优先返回更多内容覆盖而不是减少. 两个note是给agent做笔记用的, 防止agent自己的reasoning记忆在长上下文中丢失. clip_qa要被允许返回无法得知并且告诉他不要瞎猜. 需要返回得到证据的一个或者多个时间戳. harness会把这些时间戳转化为静态截图一同返回给主agent. 
- add_note(text), and read_notes(): 主agent一样有这两个工具用于笔记
- semantic_search(query, n, table): 用一个文本, 搜寻图片embedding或者文字embedding(具体看agent写了哪个table), 返回表名和行id以及时间, agent可以继续做检索. 返回top_n, 长caption做截断.table给一个限定(tools介绍和返回消息)只允许有embedding的table, MAGIC Video的index只有少量table有embedding. 
- agent_raise(): 特殊工具, agent发现当前状态明显不正常, 继续rollout没有意义, 例如数据库永远失败, 这时候agent raise一个error, 停止本次rollout, 上游决定是skip还是stop (参考规则).

#### "训练"过程:
1. 我们定义batch size代表n个问题和每个问题m个平行rollout. 把SKILL.md放入prompt对于一整个batch做平行rollout, 得到25个结果, 这是用用一个额外的judge call来判断模型答案正确性(打分/5), 这个分数记作s. wandb log这个s. 这个s也是qwen 3.8 27b产生. 
2. 把正确答案和s(而不只是s)都给每一个rollout, 然后告诉他这是正确答案和你的分数, 同时插入META-SKILL.md (META-SKILL.md不是最开始放进去的), 你现在要做的就是去重新复盘这个结果, 包括重新回顾自己的trace和重新对数据库检索和探索, 找到你为什么对/为什么错的原因, 把这个原因总结出来, 记错relection, 同时这个反思过程有什么经验(帮助下一次反思), 记错meta-relection. 两个反思都要去掉具体内容变得抽象化 可以辅助任何问题而不是本次情况. 本次情况可以作为案例, 但是要去掉敏感信息防止数据泄露(比如具体某天中午午饭内容)
3. 每一个问题的m个rollout的反思被另一个call总结起来, 然后对于n个总结和当前SKILL.md状态, 另一个agent(不是single call)生成一个patch来修改它产生新的skill. 这个修改必须使用patch而不是直接输出新的是为了防止大范围改动. 使用linux标准的patch格式, 如果patch失败就让这个这个agent得到错误(需要agent的原因)然后重写一个patch. reflection进入SKILL.md meta-reflection进入META-SKILL.md. 这个patch需要包含修改和删除, 而不只是无限量添加, 不然会爆炸. 删掉旧的错误的, 或者merge相似的, 或者补充现有的, 而不是永远加下去. 
4. 下一个batch带着新的SKILL.md和META-SKILL.md继续. 
5. 只训练一个epoch, 不然很可能会过拟合

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
1. 所有文本转成ASCII并转成小写.
2. 去除文本开始和结尾的空格, 所有连续空白折叠成单个空格, 总是允许一定程度的改变和坍缩, 保持一致优先.
3. 所有依赖保持当前最新版; 如果不能工作, 回退到可以工作的版本.
4. 环境使用uv; 不需要每次都创建新的venv, 不能使用别人创建的venv, 自己创建的venv可以复用.
5. 保留所有非英文字符; 英文字符使用ASCII.
6. infra benchmark使用更长的测试文本和多轮对话.
7. `run_sql`每个单元格保留前500字符.
8. 轮数实际设置为50; 达到50说明模型自循环, 重试整个样本.
9. 允许并行工具调用.
10. `$100`限制只在整个rollout结束后检查, 允许少量超过.
11. uv cache放到/raid.
12. home目录满了时先清理home, 删除本次产生的无用内容.
13. 过程中严谨访问未列出的目录和当前目录. 
