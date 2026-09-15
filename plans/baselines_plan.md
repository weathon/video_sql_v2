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
2. 按照MAGIC-Video的仓库的embedding方法, 用我们的新的Qwen3-VL-Embedding-8B模型对于所有需要embedding的物品进行embedding. 用vllm启动embedding模型, 视频embedding参考 notes/VIDEO_EMBEDDING_PIPELINE.md 的请求和缓存流程，抽帧明确采用均匀采样，按原始时间顺序作为视频embedding输入. 文本embedding方法直接用batch query. 空caption保留，对应的caption文本embedding设为全0. 所有的artifatcs (magic video format, embedding, database放到/raid/videosql-video-artifacts)然后用HF_TOKEN push到一个private的hf repo(名字你来定), 记录下来这个位置. artifacts包括: 视频切片, 所有embedding, 所有数据库, magic video的index, 以及一个如何使用的文档(不是handoff文档, handoff是如何改进/继续开发, 这里是如何直接使用的文档) 
3. 清理所有文档关于artifacts和index, 用一个新的简洁的文档描述我们的index的结构和方式.
4. 校验embedding, 通过text-to-video和text-to-text验证recall是否稳定. 没有具体阈值, 这里要看的不是recall有多好, 而是recall是否合理, 即embedding是否正确. 
5. 对于EgoLife的QA问题, 用规则修改他, 使其变成open question而不是选择题, 增加难度, 抽样验证规则处理去掉了几乎所有(不需要保证完美, 95%以上就好)的问题选项. 但是要注意排除的是MCQ的答案选项, 不是题干中的信息. 比如: 以下谁先吃蛋糕: lucy, jack, marshall, alex. 这里名字不属于选项. 
6. 复制一份我们的src/变成training_free/修改我们的harness, 使其不做训练只做简单的一次性rollout. 修改我们的harness限制最多turn(模型调用次数)变成30, 到达30后强行输出答案. clip_qa的模型调用也计入；replace_tool自身不计入；被替换掉的历史模型调用仍计入，回退不重置累计次数. 推荐模型进行并行tool call. 提问时加入query time. 

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



# Codex Refined Plan (这个优先级高于上面)

## 1. 目标与已确定设置

基于 [baselines_plan.md](/raid/wg25r/video_sql_trace2skill/baselines_plan.md)，比较 MAGIC-Video 与我们的 SQL 方法在 A1_JAKE 全部 500 道开放式问题上的质量和耗时。

- 在主目录 `/raid/wg25r/video_sql_trace2skill` checkout 已有分支 `codex/egolife-baselines`，继续使用已有实现和离线 artifacts。
- 答题、MAGIC retriever/responder、子调用和 judge 均使用 `Qwen/Qwen3.8-27B-FP8`，reasoning 为 `xhigh`，保留历史 reasoning。客户端不设置采样参数或输入、输出 token 上限。
- 两种方法分别以并发 20、每题一次运行；每批结束后，失败题集中重试一次。
- 我们的方法每次尝试允许 **60 次调查模型调用，再额外一次强制回答**。MAGIC 保持上游检索轮数和截止逻辑。
- 抽样检查与最终 20 题计时使用 seed 0，并保存抽中题目 ID。

## 2. 数据、索引与 artifacts

- 从 [MAGIC 官方仓库](https://github.com/lijiazheng0917/MAGIC-video) 获取源码并记录版本，使用指定的现成 artifacts 建图，不重新生成 captions、三元组或 chains。
- 自然语言统一执行 `NFKC → casefold → 合并空白 → strip`，保留标点和重音；ID、外键、路径、URL、日期时间和类型标识不变。先按原始键完成 caption/OpenIE 关联，再规范化内容，避免破坏已有 `chunk-md5` 关联。
- DuckDB 为 entity、episode、semantic、visual_clip 分别建表，保留全部原始非向量字段和 metadata。边按类型及端点表建立外键关联，保留方向、类型和权重；topic/event chains 及其 facts/steps 单独存储。只采用显式关联，不从描述或时间重叠补造关系。
- 为每题提前生成独立过滤数据库：删除未来、跨越 query time 或无明确时间的记录；边仅保留仍存在的端点。**Chain 按 fact/step 过滤**，保留剩余记录及所属 chain。Semantic 保留截止前的快照，不擅自改成 MAGIC 的“仅最新快照”规则。
- 文本和视频 embedding 均使用 `Qwen3-VL-Embedding-8B`。文本按既有流程每批 512 条；空 caption 保留，对应文本向量为全零。视频每请求一个缓存片段、最多 64 个请求在途，完成即保存。
- 按 caption 的明确时间区间裁剪 embedding 缓存片段，再均匀采样；两边共享相同向量。保留原始顺序并禁止再次采样。
- 向量按节点 ID 独立保存并共享读取；每题检索先限定当前数据库允许的 ID，再计算排名，避免未来节点进入候选集。
- QA 将答案字母映射成正确选项文本，仅向答题模型提供开放题和 query time。删除独立选项及明确的题干选项表达，保留题干必要的人名、对象和条件；随机检查 100 题，要求至少 95 题处理正确。

全部 artifacts 存入 `/raid/videosql-video-artifacts`，包括视频切片、向量、完整及逐题数据库、MAGIC 格式索引和开放题数据。通过项目 `HF_TOKEN` 上传至其所属账号下的 private dataset repo `egolife-a1-jake-magic-videosql-qwen3vl8b`，记录实际地址。提供可直接使用的 README 和简洁索引说明，清理现有文档中重复、过时的 artifact/index 描述。

## 3. Harness 与本地服务

- 复制当前 `src/` 为独立 training-free 实现，只运行静态 skills 和一次答题流程。适配真实 MAGIC schema、节点 ID 和时间字段，同步修改工具说明与静态 SQL skills，不补造旧 schema 的字段。
- 保留 `run_sql`、`semantic_search`、`clip_qa`、`replace_tool`、证据帧 recipe 和完整 trace。SQL 与检索使用逐题数据库；视频工具拒绝读取 query time 之后的证据。
- 已停止的 SQL+PPR 方案（保留 idea 记录，当前运行不启用）：组合工具 `sql_ppr_search`：自然语言 query 通过 MAGIC 的文本向量、BM25、视频向量种子及 PPR 检索。可用前置 SQL 在逐题数据库的 `memory_nodes` 视图中选择候选 ID，再在这些节点及其显式边构成的子图上检索；也可先检索，再用后置 SQL 在 `ppr_hits` 表上筛选、排序或提取所需信息。`memory_nodes` 是 entity、episode、semantic、visual_clip 的联合视图；`ppr_hits` 保留节点原字段及 rank、score。前置 SQL 用于硬性限定证据范围（如昨天），后置 SQL 用于整理语义命中结果，不能节省前面的检索工作。告诉模型通常只做一次 SQL，按目的选择 pre 或 post；需要时仍可使用两者。原 SQL 和向量检索工具保留。
- 上述已停止方案的组合检索只使用当前逐题数据库允许的节点和显式边，前置范围外的节点不参与种子选择或图传播。保留 MAGIC 上游 PPR、BM25 权重及多尺度筛选；SQL 方法继续保留截止前的 semantic 快照。组合工具不增加隐式 LLM 子调用。
- 原本 SQL60 运行已完成并记录结果。下一阶段保留 `run_sql`、`clip_qa`、`replace_tool`，以支持 pre/post SQL 的 PPR 替换 `semantic_search`；仅在需要语义匹配时使用，不默认或优先推荐，没有单独的 `sql_ppr_search` 工具。
- 在实际模型请求发出前统一计数：主 agent、`clip_qa`、SQL/检索结果 QA 等调查调用共享预算；并行子调用先占用剩余额度。调用 `replace_tool` 的模型轮次计数，工具执行不另计，历史回退不减计数。
- 用完 60 次后关闭调查工具，额外调用一次生成最终答案；judge 不占调查预算。移除旧的客户端 token 阈值提前强答规则，真实 context 溢出按失败协议处理。
- MAGIC 的 embedding 接口改为 HTTP，所有查询和建索引路径使用同一规范化及向量模型。保留其 PPR、BM25、chain 检索和时间截止语义；开放题不再传入选择项。模型接口保留每题真实消息及 reasoning 历史，结构化输出使用 SDK parse。
- 时间检索修正日期与秒数单位，并应用 query time 截止；保留时间检索分支。Chain 相似度检索保留上游前 2000 字符规则。
- 每题隔离可变检索状态，避免串行抽样或批后重试继承其他问题的时间范围。缺失必要数据或 embedding 时明确失败，不继续运行缺失的检索分支。

新服务使用独立容器名前缀、网络和 encoder cache volume：

- 主服务 TP=2、`max-num-seqs=50`；初始选择 GPU 4、5，encoder 和 embedding 共用 GPU 6。
- CPU KV cache 合计 **256 GiB，每个 TP rank 128 GiB**；其余以当前核对过的 INFRA 配置为基础。
- 使用新端口：主服务 8124、encoder 8125、embedding 8114、proxy 8103；挂载本任务实际媒体目录。
- 启动时重新核对资源；资源不足立即报告，不等待或调整实验规格。
- Harness 启动前将 OMP、MKL、OPENBLAS、NUMEXPR、VECLIB 各限制为 10 线程，并通过 systemd 将内存上限设为服务启动后可用内存的 20%。
- 服务和任务均以前台会话运行，日志落盘，稳定后采用长间隔等待；结束后停止并手动删除本任务容器。

## 4. 评分、重试、token 与计时

两边使用同一个 judge 实现，输入开放题、正确答案文本、模型答案，以及提问者 A1_JAKE/Jake、第一人称身份、query time 原值、DAY 编号、时分秒和帧号，固定采用以下 rubric：

| 分数 | 标准 |
|---|---|
| 0 | 无有效答案，或核心答案错误 |
| 1 | 仅少量相关正确信息，核心问题未答对 |
| 2 | 部分正确，但存在主要遗漏或错误 |
| 3 | 核心内容大体正确，仍有实质遗漏或错误 |
| 4 | 基本完整正确，仅有可指出的、与问题相关的次要事实错误或遗漏 |
| 5 | 问题所需信息完整正确，在问题要求的精度上与参考答案语义等价，无实质矛盾 |

评分以问题要求的信息为准。接受同义表达、可识别的同一人名拼写变体、列表顺序变化以及与 query time 对应的绝对/相对时间；“我”和 Jake/A1_JAKE 指同一提问者。正确的补充细节、引用、行 ID、时间戳、格式、篇幅和谨慎措辞本身不扣分，不因简短参考答案未列出某个细节就认定该细节错误。低于 5 分须在结构化 justification 中指出实际事实错误或所需信息的遗漏；score 与 justification 应一致。

谁的问题要求答对人物，不要求复述参考答案中的附带事件。DAY3 是日期编号，不自动等于三天前；时间按参考答案精度比较，约几小时前或中午等表达不要求精确到同一分钟。确实错误的补充断言或改变答案的未消解备选项可扣分。

已统计旧 SQL 10 次预算的实际强制回答比例，并用上述 judge 重新评分两边已有答案，保留原答案、失败分类和旧评分。用户停止了 SQL30 运行，保留其部分 artifacts 并标记为未完成、不用于正式对比。SQL+PPR 15 次运行也已由用户停止，保留其部分 artifacts 并标记作废。恢复原本 SQL 方法，以 60 次调查调用加一次强制回答重新运行全部 500 题，使用同一新版 judge；新结果写入独立目录，不复用停止运行的答案。

每道成功答题后立即 judge，保持并发 20 和批后一次集中失败重试。W&B 的 1 Hz 日志增加已完成且可评分题目与 MAGIC 新评分结果的共同题目数、两边同题均分、满分准确率（5 分比例）和差值。未完成或尚未分类的失败不当作 0 分；最终算法失败分类后计 0 分，维护失败排除。答题与 judge 交错执行时，以同一阶段的服务端计数记录总 token，保存每个请求的原始 usage，不能把混合阶段的 token 都标成答题 token。

- 保存每题答案、judge、全部尝试和完整请求/工具 trace，完成即落盘，恢复时复用已完成结果。
- 不沿用旧评估器的三次整题重试；只执行本批结束后的那一次集中重试。模型可纠正的工具错误回传原错误，后续模型调用仍消耗预算。
- 根据实际 trace 区分算法失败与维护失败。最终算法失败计入分母，无答案记 0 分，有有效答案则 judge；最终维护失败排除并记录原因。
- 分别报告平均分、`5 分题数 / 有效分母`、有效分母和两类失败数量。
- Token 统计覆盖实际全部 LLM 请求，包括子调用、judge、被替换历史及失败重试；保存原始 usage，并与独立服务端计数核对。未返回 usage 的请求不能按零处理。
- 日志分别累计未缓存输入 `in`、缓存输入 `cache` 和输出 `out`；输入总数包含缓存时先扣除。等效费用为 `(in×0.15 + out×0.47 + cache×0.016) / 1,000,000 USD`。
- 持续拉取服务端 `/metrics`，以 1 Hz 实时记录 token throughput、累计 token usage、运行进度和 GPU 使用情况并上传 W&B；不得等待请求 usage 返回或批次结束才记录。
- 上一次质量和计时结果作废，修复及审查完成后重新运行；旧答案不得作为新运行的完成结果复用。
- 最后用同一随机 20 题、相同顺序、并发 1，先 MAGIC 后我们的方法。每种方法开始前清理 prefix/KV、encoder 和运行时查询缓存，保留模型与离线索引。计时从每题处理开始到最终答案，包含在线检索和模型调用，不包含 judge；报告逐题时间、20 题总时间和平均时间。

## 5. 验证与交付

- 核对原始记录到图、DuckDB 和向量 ID 的对应关系；检查截止前、截止点、跨截止及无明确时间记录，确认 SQL、向量检索和视频证据均符合各自规则。
- 用小范围真实调用检查原 SQL/检索工具及时间范围、并行预算、60+1 强答、replacement 回退、reasoning 回传、逐题 judge、实时同题对比、失败重试与 token 账目。逐项查看工具实际输入和返回值，不仅检查完成状态。
- 不新增测试套件，不运行语法检查。本次跳过 Claude 检查，启动新的独立 subagent 对照本计划审查整个目录，不只是 diff；处理审查问题后再重新运行。
- 每次源码或文档编辑后提交。Git 只保留源码、使用说明和紧凑最终结果；原始 traces、逐题结果、大型数据库和向量保持忽略。
- 最终交付可运行的两种方法、完整私有 HF artifacts、直接使用文档，以及质量、失败分类、token/等效费用和串行耗时对比。

## 6. 下一阶段

- 原 SQL 60 次调查预算的运行已完成，结果已先记录于 `notes/egolife-baselines-results.md` 和同名 JSON，提交为 `acd5cf0`。
- 然后将当前 `semantic_search` 替换为支持 pre/post SQL 的 PPR 检索。
- PPR 仅作为语义搜索工具使用，不设为默认工具，不提示模型优先使用它；保留原本的 SQL 工具。
