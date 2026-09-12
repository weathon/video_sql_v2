# 对比在没有训练的情况下, 我们的方法和MAGIC-Video方法的性能

## 目标
对我们的方法和MAGIC-Video在EgoLifeQA上面的表现

## 材料:
MAGIC-Video的github仓库: https://github.com/lijiazheng0917/MAGIC-video.git
Magic-video的artifacts: /raid/huggingface/hub/datasets--jiazhengli7--magic-video-artifacts/snapshots/8573f317684ce55c0999f86386666d3d3fe76ae3/egolife/A1_JAKE/
Embedding模型: https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B
QA问题集: /raid/wg25r/egolife/EgoLifeQA/EgoLifeQA_A1_JAKE.json
EgoLifeQA视频: /raid/wg25r/egolife/A1_JAKE

## 步骤:
### 准备工作
0. 在/raid/wg25r checkout一个新的worktree来完成这项任务从最新的commit, 避免干扰. 复制或者link必要的ignore文件(例如.env和data)到新的目录
1. 转化Magic-video的nodes变成一个数据库. 自然语言文本（caption、实体名、三元组文本、chain描述和检索查询）统一先做Unicode NFKC规范化，再用casefold做大小写折叠，将连续空白合并为一个空格并去掉首尾空白；保留标点和重音字符. 建图、入库、文本embedding和查询使用同一规则；节点ID、外键、文件路径、URL、日期时间和类型标识保留原值. 一个node type一个表格, edge用外键表示. topic/event chain用单独的表格表示. chain仅对已有明确关联建立外键，没有关联则没有外键，不从描述中推断新关联. 用duckdb做数据库. 保留所有schema. 我们的方法需要对每一个问题限制sql query的时间, 提前用sql过滤把所有在query time之后的行去掉, 对于每一个问题, 提前生成一个过滤后的数据库(反正很小)存储起来, 后面用这个过滤后的. 按照完全排除过滤, 即如果这个index跨越了query time, 则被去掉. Various、GENERAL、Overall habit等没有明确时间的记录从每题过滤数据库中排除. MAGIC的工具沿用自带的截止逻辑，不修改.
2. 按照MAGIC-Video的仓库的embedding方法, 用我们的新的Qwen3-VL-Embedding-8B模型对于所有需要embedding的物品进行embedding. 用vllm启动embedding模型, 视频embedding参考 notes/VIDEO_EMBEDDING_PIPELINE.md 的请求和缓存流程，按caption的明确时间区间裁剪embedding缓存片段，再均匀采样，按原始时间顺序作为视频embedding输入；两边共享相同向量. 文本embedding方法直接用batch query. 空caption保留，对应的caption文本embedding设为全0. 所有的artifatcs (magic video format, embedding, database放到/raid/videosql-video-artifacts)然后用HF_TOKEN push到一个private的hf repo(名字你来定), 记录下来这个位置. artifacts包括: 视频切片, 所有embedding, 所有数据库, magic video的index, 以及一个如何使用的文档(不是handoff文档, handoff是如何改进/继续开发, 这里是如何直接使用的文档) 
3. 清理所有文档关于artifacts和index, 用一个新的简洁的文档描述我们的index的结构和方式.
4. 校验embedding, 通过text-to-video和text-to-text验证recall是否稳定. 没有具体阈值, 这里要看的不是recall有多好, 而是recall是否合理, 即embedding是否正确. 
5. 对于EgoLife的QA问题, 用规则修改他, 使其变成open question而不是选择题, 增加难度, 抽样验证规则处理去掉了几乎所有(不需要保证完美, 95%以上就好)的问题选项. 但是要注意排除的是MCQ的答案选项, 不是题干中的信息. 比如: 以下谁先吃蛋糕: lucy, jack, marshall, alex. 这里名字不属于选项. 
6. 复制一份我们的src/变成training_free/修改我们的harness, 使其不做训练只做简单的一次性rollout. 修改我们的harness限制最多turn(模型调用次数)变成10, 到达10后强行输出答案. clip_qa的模型调用也计入；replace_tool自身不计入；被替换掉的历史模型调用仍计入，回退不重置累计次数. 推荐模型进行并行tool call. 提问时加入query time. 

### 运行
1. 参考INFRA.txt的方式, 启动一个新的vllm server，使用新的Docker容器名，不复用现有容器名, TP=2, encoder分离, encoder和embedding放在一个vram空闲计算量偏少的GPU. 服务器max num seq变成50写一个监控脚本, 监控总in/cache/out token消耗. 等效节省指这一批query如果使用API会产生的费用，包含该批query实际所有LLM请求的三种token，价格均为USD per 1M tokens：in $0.15、out $0.47、cache $0.016. in指未命中缓存的输入token，cache指缓存命中的输入token，out指输出token；若API返回的input总数包含cache，先扣除cache得到in. 费用（USD）= (in × 0.15 + out × 0.47 + cache × 0.016) / 1,000,000，记录在一个log文件里面.
2. 修改MAGIC-Video的harness代码, 使其使用http请求embedding而不是线程内, 这样可以避免死锁. 同时修改llm端点使用本地qwen
3. 同时修改MAGIC-VIDEO的harness. 因为其并发时 egolife 会开很多线程，每个并发里 OMP/MKL/BLAS/NumExpr 又各自用满所有核心，线程数暴涨会把机器搞崩（有时还会系统 OOM，连带 NVIDIA 驱动崩掉导致 NCCL 不可用）。解决：用线程环境变量把 OMP/MKL/OPENBLAS/NUMEXPR/VECLIB 各限制到 10 线程，并记住以后使用时用 systemd 限制最大内存（本次 剩余内存的20%）。
4. 启动MAGIC-Video的harness, 并发20, 回答每个问题, 每个问题回答1次. 用llm judge(judge获得问题, 正确答案, 和模型答案)对每个回答打分, 0-5分, 看有多对. 给出rubic每个分数代表什么确定稳定性
5. 记录下平均分数和准确度(完全对5分的比例)
6. 启动我们的方法, 用从magic video的index构建的数据库做我们的方法的运行和judge. 最后记下比分. 并发20线程运行. 
7. 在没有并发, vllm完全空载的情况下, 对magic和我们的方法分别按顺序做一次性能测试. 同样的随机采样20个问题, 并发为1, 顺序跑完magic video跑我们的. 比较两边的时间

## 其他信息
1. 所有方法在客户端不设置采样参数, 采样参数在服务端使用默认, 所有reasoning level用xhigh
2. 确保两个harness都保留past reasoning回传
3. 两种方法均在本批问题完成后，将失败的问题集中再次重试一次，保持同一方法和设置. 最终按实际执行trace中的失败原因区分算法层级和维护层级，不仅凭异常名称分类.
   - 算法层级：超出context window；用尽调用预算仍无有效答案；模型生成无效SQL、工具参数或答案格式且最终未能纠正；拒答或未生成有效答案. 最终重试后仍属于算法失败的题计入分母；没有有效答案的题计0分，有有效答案的题按judge评分.
   - 维护层级：vLLM超时、连接或服务中断、服务端崩溃、硬件故障、必要数据或文件缺失损坏、实现错误导致工具不可用. 最终重试后仍属于维护失败的题不计入平均分和准确度的分母，记录具体失败原因.
