# 视觉 Agent Memory 上下文效率诊断

审计日期：2026-09-10 UTC。检查对象为当前磁盘源码（审计开始时 HEAD 为 `8253bab`）和最新实际运行 `runs/luna-day-week-tree-month-val-seed0`。两名新子代理分别检查 harness/SDK/技能加载和原始 trace；主代理检查分析阶段、快照存储，并核对三个案例的实际请求与工具返回。本次只交付诊断，没有修改实验代码、调用模型 API 或重新运行实验。开始时已有的 `src/harness.py` 和 `REVIEW.md` 修改未纳入本次提交；主代理未读取 `REVIEW.md`。

**最主要的问题是大量工具证据进入上下文后持续重放，而现有替换能力几乎没有使用。** 视觉子代理占累计输入 token 的 52.23%，主检索代理占 47.73%；两侧都需要处理。优先候选是工具返回的无损去重与紧凑编码，以及针对已读大结果的证据保留策略。技能文本有重复，但样本不支持把它当作主要原因。已有缓存确实生效，不能将累计输入全部按未缓存价格解释。

## 1. 范围与计量口径

最新运行由当前 [baseline 的 RUN 配置](/raid/wg25r/video_sql_trace2skill/src/trace2skill_baseline.py:32)、实际 API 时间戳及 [停止记录](/raid/wg25r/video_sql_trace2skill/runs/luna-day-week-tree-month-val-seed0/stopped-summary.json)共同确认，不依据文件名或 mtime。该运行已完成响应的 `created_at` 范围是 `2026-09-09T05:25:56Z` 至 `2026-09-10T00:00:10Z`；最后的 `completed_at` 为 `2026-09-10T00:00:13Z`。停止记录为 `2026-09-10T00:02:23.394400+00:00`，原因是用户因费用停止运行。实际存在 1,422 份 `answer.json` 和 1,419 份 `result.json`。当前运行没有 `analysis`、`evolution`、`month-initial` 或 `month-evolved` 目录，后文关于这些阶段的发现仅来自源码。

全量逐行扫描 collection 下 11,987 个事件文件，共 12,772,127,637 字节，提取 58,724 条 `model_input` 和 58,082 条带 usage 的 `response.completed`；完成响应的 ID 没有重复。统计直接来自原始事件，不累加 segment 中重复保存的 `api_usage`、`path_responses` 或父分支费用。两种事件计数相差 642，不能将其直接称作 642 个 API 失败，也不把没有完成 usage 的请求消耗推定为零。

三种量必须区分：

| 对象 | 本次使用的证据 | 解释 |
|---|---|---|
| 实际请求上下文 | 过滤后的 `model_input`、完成响应的 API usage | 判断每轮保留了什么、输入如何增长 |
| 物理请求累计用量 | 原始事件中唯一的完成 response ID | 包含中断尝试中已经完成的请求；不是一条最终答案的逻辑路径 |
| 磁盘 trace 体积 | `history`、`raw_responses`、`prefix_state`、事件文件 | 包含多份审计与恢复副本，不能直接换算为模型 token |

`interrupted-*` 路径中有 28,960 个已完成请求、1,353,454,007 输入 token；其余 attempt 路径有 29,122 个、1,314,817,853 输入 token。这是物理尝试的分类，不能把前者全部称为无用计算，也不能把整个运行用量除以答案数就称作单条最终路径的上下文。树分支实际有 2,249 条 `prefix_verified` 事件；[分支恢复和路径指标](/raid/wg25r/video_sql_trace2skill/src/trace2skill_baseline.py:177)会携带父前缀。

人工检查覆盖下述三个案例的工具参数、返回边界、具体文本片段与逐轮输入；长 SQL/caption 只读取有界真实内容，没有逐字通读全库。落盘图像 URL 已被替换为占位文本，图像统计因此是 `input_image` 项出现次数，不是独立帧数或图像 token 数。本次没有验证图像像素和答案正确性。

## 2. 实际开销分布

| Agent | 已完成请求 | 累计 input tokens | cached tokens | cache_write_tokens | output tokens | 单请求 input 峰值 |
|---|---:|---:|---:|---:|---:|---:|
| clip-qa | 29,477 | 1,393,555,922 | 988,659,277 | 398,916,296 | 13,854,967 | 383,036 |
| memory-retrieval，含 compaction | 27,125 | 1,273,634,548 | 935,442,955 | 338,110,218 | 8,686,470 | 234,812 |
| judge | 1,424 | 703,332 | 0 | 0 | 117,953 | 936 |
| sql-qa | 56 | 378,058 | 0 | 373,320 | 47,094 | 46,708 |
| 合计 | 58,082 | 2,668,271,860 | 1,924,102,232 | 737,399,834 | 22,706,484 | — |

所有已完成响应的实际 reasoning effort 都是 high。累计 output 中，API 报告的 reasoning tokens 为 13,912,647；这个量不能从 `encrypted_content` 的字符数推算。clip-qa 输入的中位数/P95/P99 是 34,901/158,389/269,330；主代理为 37,825/128,181/158,162。分位数取排序后索引 `floor((N−1)×p)`。

| Agent | 实际 model_input 数 | 含图请求数 | 图像项累计出现次数 | 单请求最多图像项 | 工具返回项累计出现次数 |
|---|---:|---:|---:|---:|---:|
| clip-qa | 29,976 | 21,232 | 2,711,358 | 650 | 170,517 |
| memory-retrieval | 27,262 | 3,542 | 33,560 | 57 | 445,516 |

这张表包含没有完成 usage 的输入快照，因此请求数与上一张表不同。重复发送同一帧也会再次计数。主代理收到的是 clip 最终引用的证据图，没有收到全部内部采样图。

全库实际只有 **1 次 `replace_tool` 调用，clip-qa 中为 0 次**。`compaction-events-*` 有 411 个文件，其中完成调用 402 次，消耗 65,213,604 输入和 1,012,160 输出 token，已经计入主代理一行。不能把 411 个文件当成 411 次已完成压缩。

72.11% 的累计 input 被 API 记为 cached，737,399,834 token 为 cache write，普通未缓存且未写入部分为 6,769,794。仓库 [费用函数](/raid/wg25r/video_sql_trace2skill/third_party/trace2skill/tracing_bridge.py:20)使用 `ordinary = input − cached − written`，再按 `ordinary + 0.1×cached + 1.25×written` 加权；这一缓存口径与当前 [OpenAI 缓存文档](https://developers.openai.com/api/docs/guides/prompt-caching)一致。这里不据此重算账单，也不把原始输入 token 的下降直接当成同比例费用下降。

## 3. 已验证的增长机制与改进方向

### 3.1 视觉子代理的大批采样和后续重放是首要目标

[clip-qa 的创建与调用](/raid/wg25r/video_sql_trace2skill/src/harness.py:1077)使用独立上下文，初始输入只有区间和视觉问题；[返回主代理的部分](/raid/wg25r/video_sql_trace2skill/src/harness.py:1103)仅包括答案、时间及引用证据帧。隔离已经存在，不能将问题诊断成“子代理完整 trace 被塞回主代理”。

真正的增长发生在子代理内部：每次 `sample_video`、`crop_frame` 的图像与文字一直留在其后续输入中。采样的 650 图额度限制累计采样/裁切数量，[调用处的 `watch_context=False`](/raid/wg25r/video_sql_trace2skill/src/harness.py:1092)使主代理的压缩检查不保护这个循环。这里的 314,573 是 harness 配置，并不是本次核实过的提供商模型窗口。

**案例 A：day question-42 / rollout-02 / attempt-1。** [原始事件](/raid/wg25r/video_sql_trace2skill/runs/luna-day-week-tree-month-val-seed0/collection/day/question-42/rollout-02/attempt-1/traces/clip-a0230ae29c5d4209bbba18aefb32795a-events-bff95581817346a3914cd511a83fc15b.jsonl)与 [segment](/raid/wg25r/video_sql_trace2skill/runs/luna-day-week-tree-month-val-seed0/collection/day/question-42/rollout-02/attempt-1/traces/clip-a0230ae29c5d4209bbba18aefb32795a-segment-02.json)中的视觉任务是检查 23,400–24,600s 的敌人击杀。

| 本地请求轮次 | 输入图像项 | input tokens | cached tokens | cache_write_tokens |
|---|---:|---:|---:|---:|
| 1 | 0 | 832 | 0 | 0 |
| 2 | 600 | 347,820 | 0 | 347,817 |
| 3 | 606 | 352,222 | 347,817 | 4,402 |
| 4 | 649 | 377,411 | 352,219 | 25,189 |

首轮并行调用四次 `sample_video`，覆盖连续四个 300s 区间，均为 0.5fps、完整画面、附 captions。tool_id 23–26 每次实际返回 150 个图像项和 10 条 captions，首批共 600 图。第二轮七次 crop 中六次返回图像，一次把时间字符串 `24330.012` 当成 frame_id，实际返回 `frame_id must refer to an image returned in this clip_qa`。第三轮再采样 `[24100,24115]@1fps` 和 `[24320,24348]@1fps`，分别返回 15/28 图。第四轮提交答案和四个证据 ID。

第二至第四轮仍完整保留首批 600 图、40 条 captions，连失败 crop 的错误也持续存在。四次请求累计输入 **1,078,285 token**，输出 4,734；649 个采样/裁切图像项在输入中出现 1,855 次。此例不仅有重放成本，第二轮还一次引入了约 348k 的新输入。因此仅优化旧历史，不控制每批进入上下文的证据量，仍会留下很大的首批开销。

建议分两类评估：

- **信息无损候选：** 对完全相同来源时间、裁切和图像内容的重复证据，以及重复 caption/元数据，保留一份内容和完整引用映射；不要按“看起来相似”合并不同时间的动作帧。原始输出和全部证据 ID 仍须可追溯。
- **改变上下文策略的候选：** 在已有 `replace_tool` 中保留已检查区间、证据 ID、发现及未解决事项，移出已确认不再需要的批次；或把长区间调查划分为明确的阶段，并保留可恢复的证据。需要先验证问题是否依赖整段连续动作或稀有事件。这些策略会改变模型观察历史；不能直接声称与原实验等价。

不建议用降低模型、reasoning effort、图像分辨率/detail 或采样总预算来代替上下文优化。只添加 notes 也不会移除图像；源码中笔记与历史目前是并存的。

### 3.2 主代理的大段检索返回同样需要处理

[输入过滤器](/raid/wg25r/video_sql_trace2skill/src/harness.py:470)只对明确被替换的 `function_call_output` 改写输出，其余历史继续发送。`run_sql` 已限制 100 行、每字符串 500 字符，但这仍允许数万字符的单次回包；[semantic_search](/raid/wg25r/video_sql_trace2skill/src/harness.py:804)每条 preview 最多 500 字符，`n` 只要求为正数。限制单元格长度并不等于控制整个结果大小。

**案例 B：day question-52 / rollout-00 / attempt-1。** 主代理 [answer segment](/raid/wg25r/video_sql_trace2skill/runs/luna-day-week-tree-month-val-seed0/collection/day/question-52/rollout-00/attempt-1/traces/answer-segment-01.json)和 [实际输入事件](/raid/wg25r/video_sql_trace2skill/runs/luna-day-week-tree-month-val-seed0/collection/day/question-52/rollout-00/attempt-1/traces/answer-events-5063b953b18a4d48948cc1f0c32cb13f.jsonl)有七次请求：

`1,371 → 2,807 → 8,307 → 21,009 → 23,229 → 37,263 → 41,278 input tokens`，累计 135,264。

首轮并行加载五个技能后仅增至 2,807。随后八次语义检索返回 120 个 hits，其中按 `(table,row_id)` 计算，23 个是同一调查内重复出现的行；这包括并行请求之间的重叠，不能都称作模型重复读取旧轮次。tool_id15 的 SQL 是 `SELECT row_id,start_s,end_s,granularity,text FROM episodes WHERE start_s BETWEEN 45000 AND 47000 ORDER BY start_s`，实际返回 80 行、52,037 字符，混合 67 条 30sec、12 条 3min、1 条 1h 记录。相邻下一轮输入从 23,229 增至 37,263；这个增量同时包含上一轮模型输出，不是独立测得的 SQL token 数。

最后一个 clip 返回六张证据图，主代理输入为 41,278；它没有接收该 clip 内部全部 120 图。[对应 clip segment](/raid/wg25r/video_sql_trace2skill/runs/luna-day-week-tree-month-val-seed0/collection/day/question-52/rollout-00/attempt-1/traces/clip-d8234f56cb3c41f2aa26f0a08e363a61-segment-00.json)四轮输入为 `846 → 28,653 → 55,382 → 74,360`，输入图像项为 `0 → 46 → 92 → 120`。实际采样逐步缩小区间、提高 fps，说明不同样本的增长来源不能只用一种解释概括。

建议先处理表示上的重复，再评估检索行为：

- SQL 已有 `columns`，行对象仍逐行重复字段名。按现有列顺序改为行数组，保留所有值、顺序、截断标记和 `has_more`，上述同一回包的序列化字符数从 52,037 变为 47,797，少 4,240 字符（8.15%）。这是离线表示比较，不是 API token 或答案质量收益。
- semantic hits 的外层和每个 hit 重复 `table`；相同行内容可通过引用复用，但保留各次查询的 rank、score、时间、row_id 及来源。不能删除不同语义粒度的记录，或擅自改变检索排序。
- 查询已经定位到候选区间时，优先提出能直接区分候选的列、时间范围和粒度，避免继续搬入大量背景；大列表在确认哪些行有用之后，才用保留来源的摘要替换。此类提示/检索策略会改变实验行为，应单独比较，不能作为格式清理混入。

### 3.3 `replace_tool` 已实现且确实生效，问题是使用极少

[工具定义](/raid/wg25r/video_sql_trace2skill/src/harness.py:375)允许替换包括图像在内的已完成工具结果，并提示优先替换最新结果以保留前缀缓存。长 clip 的工具清单也包含它，因此零次使用不是因为没有暴露工具。当前机制完全依赖模型主动决策，没有持续管理大回包的行为保障。

**案例 C：day question-57 / rollout-00 的一个中断尝试。** [事件文件](/raid/wg25r/video_sql_trace2skill/runs/luna-day-week-tree-month-val-seed0/collection/day/question-57/rollout-00/interrupted-1-9aaceb5e843e4fb8a996388623e56463/traces/answer-events-884ea9c36c434f7a9a0ecefe878d4cd8.jsonl)中，tool_id14 使用包含 `%cap%`、`%martial%`、`%melon%` 等宽泛词的 SQL 查 `visual_clips`，实际返回 100 行、62,278 字符，且 `has_more=true`。首尾文本是开场猴面战士和雪地人物/狐狸，确有大量背景内容。

第四轮 `call_yTRwrS1IkHp7zuHa4V8YSbi0` 调用替换，工具返回 `{"replaced_tool_id":14}`；第五轮真实 `model_input` 中原 call_id 的输出只有 183 字符，input 从 **28,386 降至 12,568**。缓存前缀仍为 7,025，写入量从 21,358 降至 5,540。对应 segment 的 `history` 中仍保存 62,278 字符的原始输出，因而只看 `history` 会误判替换没有生效。

这一次下降不能证明整体节省或无损：替换文本中甚至包含不完整时间 `37,770–38,? s`；本段随后又读取多批新结果，input 增至 31,271、106,595、140,386。它位于中断尝试，不能作为完成答案的质量验证。

建议把“大结果读完之后保留何种证据”写成明确的可评估行为：保存查询条件、关键行/时间/证据 ID、结论、反例和未解决项，再针对冗余部分调用已有替换工具。不要默认每轮加一次摘要调用。替换文本既出现在目标输出，又保留在 `replace_tool` 自身 arguments；原始 SQL arguments 和 reasoning 也仍存在。[replacement 计量](/raid/wg25r/video_sql_trace2skill/src/trace2skill_baseline.py:256)只数局部 before/after，不包含额外决策/调用、后续重查以及缓存改变，不能把该数直接称为整条轨迹的净节省。

### 3.4 compaction 是较晚的整体交接，并未覆盖视觉循环

[阈值计算](/raid/wg25r/video_sql_trace2skill/src/harness.py:554)是 `314573 − input − output <= 178000`，即已有 input+output 达到 **136,573** 后触发；178,000 是剩余空间阈值。判断发生在响应完成时，当前工具返回造成的下一轮增长不能被预先限制，因此它不是严格的请求上限。

[交接代码](/raid/wg25r/video_sql_trace2skill/src/harness.py:1286)再次携带完整历史生成 handoff，然后从 base instructions + handoff 继续；第三次交接返回 `compaction_limit`。主代理初始用户消息中的原问题和技能 catalog 不会作为原始消息自动带到新段，保留情况取决于交接文本。更早压缩会增加交接调用，并可能损失证据或引发重新检索，不能只把阈值降低就宣布更省。

建议优先验证局部替换是否足以处理大结果；若评估更早的整体交接或 clip 内部交接，必须保留原问题、已查区间、证据定位、反例和待解项，检查真实后续 trace 是否重复查回被丢弃的内容。阈值、交接频率和视觉上下文策略都属于实验方法变化，本次没有修改这些设置。

### 3.5 技能/schema 与笔记的重复可清理，但优先级较低

[主 instructions](/raid/wg25r/video_sql_trace2skill/src/harness.py:1246)已经包含 12 表 schema；[sql-schema skill](/raid/wg25r/video_sql_trace2skill/skills/initial/sql-schema.md:18)再次列出同一 schema。技能 catalog 和按需 `read_skill` 已存在，不是所有技能都被预先注入。案例 B 的五技能加载只对应 1,371→2,807 的输入变化，远小于之后的大结果增长。

可评估将完整 schema 保留一处，让 sql-schema 只保留额外的 granularity、join 和时间语义；保留按需加载机制。`read_skill` 没有阻止重复加载，但案例 B 中没有同技能重复读取，不能把代码允许当成已观察到的行为。修改初始技能会改变基线，需要作为独立改动记录。

[add_note](/raid/wg25r/video_sql_trace2skill/src/harness.py:882)把笔记同时放在调用参数和返回 `saved.note` 中；`read_notes` 每次返回所有笔记。可以把写入确认缩为简短 ack，笔记内容仍保留在原始 trace 和持久化文件。全库只有 21 次 add_note、58 次 read_notes，因此此项不是当前主开销。已有 `sql_qa` 会把完整、无行数截断的查询结果交给另一模型调用，不能无条件将所有 SQL 改走它：这会转移输入开销并引入信息压缩，目前只有 56 个完成调用。

## 4. 已有缓存、reasoning replay 和本地快照的边界

当前 [rollout](/raid/wg25r/video_sql_trace2skill/src/trace2skill_baseline.py:168)已设置按问题分组的 `prompt_cache_key`；实际缓存数也证实重用。改变历史中间的内容会改变其后可重用的前缀，因此要比较真实 cached/write usage，不能把删除一个旧结果后的净收益只按删除 token 数计算。可保留稳定的 instructions、工具定义和一致的前缀顺序；不应无证据地改缓存配置或扩充 prompt 来追求命中率。

[harness 的模型设置](/raid/wg25r/video_sql_trace2skill/src/harness.py:1177)保留 stateless reasoning replay。案例 B 末轮有六个带 `encrypted_content` 的 reasoning 项，API 实际 `reasoning.context=all_turns`。当前 [OpenAI reasoning 文档](https://developers.openai.com/api/docs/guides/reasoning)要求工具链保留相应 reasoning 和调用/返回项；不能随意删密文、把摘要当作等价 reasoning，或改成较弱推理设置来声称压缩成功。改用 `previous_response_id` 也不会自动免除历史输入计费，见 [会话状态文档](https://developers.openai.com/api/docs/guides/conversation-state)。

另一个已证实的问题是存档重复，但它需要单独归因。[每轮快照](/raid/wg25r/video_sql_trace2skill/src/harness.py:481)会遍历已有证据图写 JPG，并复制 input、tool_calls、tool_outputs、path_responses、prior_histories；[save_trace](/raid/wg25r/video_sql_trace2skill/src/harness.py:298)再保存 history/raw_responses/tool_calls。案例 B 的主 segment 为 2,194,539 字节；按紧凑 JSON 序列化测量，history 为 161,518 字符，raw_responses 为 1,712,159 字符。末轮快照中的 input/tool_calls/tool_outputs 又分别占 157,835/182,577/191,799 字符。

这些副本没有全部发给模型，最终返回给 SDK 的是 `ModelInputData(input=items, instructions=...)`。可以评估让不变的工具结果与 JPG 只落盘一次，快照引用内容或记录可精确恢复的增量，完整保留分支 prefix、reasoning、图像字节和原始轨迹。此项可能降低本地序列化、内存、磁盘和上传负担；本次没有测量这些步骤的耗时，不能认定它是运行慢的主因，也不能将其体积减少计入 API token 节省。

## 5. 尚未运行的分析阶段：存在另一处输入重复

[analyze](/raid/wg25r/video_sql_trace2skill/src/trace2skill_baseline.py:384)把每个 artifact 的 agent、history、tool_calls 写入 bundle。history 已包含调用参数与返回；tool_calls 又保留调用记录，尽管 original_output 被排除，调用参数仍重复。多个 segment 的 agent 配置和前缀历史也可能重复。成功分析一次发送整个 bundle 和全部 caller-visible evidence 图；失败分析虽然支持按行 read_trace，但返回后也按一般工具历史累计，并且 `watch_context=False`。

建议为分析提供一份按调用 ID 对齐、每个调用和结果只出现一次的有序视图，保留成功/失败尝试及因果关联，并能定位完整原始 trace；不要先删除失败探索只留下正确路径。重复的 agent 配置可引用一次，证据图按实际引用去重，模型需要时再读取详细片段。此处必须继续遵守已有 `clip_qa` 不透明边界：成功/失败分析只能看到主调用参数、返回答案和返回证据，不能引入视觉子代理内部 reasoning、采样轨迹或未返回帧。

该运行未进入分析和演化阶段，所以这些是源码确认的后续风险，不是本次 2.668B 输入的已测来源。

## 6. 建议评估顺序

| 优先级 | 候选改动 | 已有证据 | 需要保留或核对的边界 |
|---|---|---|---|
| 1 | 大工具返回的无损紧凑编码、完全相同内容的引用复用 | 案例 B 的 80 行 SQL 和重复 hits | 保留全部值、顺序、来源与可读取内容；格式变化仍需比较模型表现 |
| 1 | 明确使用已有 replace_tool 管理已读大结果，包括视觉批次 | 全库只调用一次；案例 C 真实输入下降；案例 A 持续保留 600 图 | 不能丢失动作连续性、反例或证据；计入替换调用、缓存变化和重查 |
| 2 | 限定检索问题与候选区间，评估分阶段视觉工作集 | 案例 A 首批即 600 图；案例 B 查回大量不同粒度背景 | 属于检索/上下文策略变化，不默认减少采样预算、降低细节或跳过样本 |
| 2 | 主代理与 clip 分别评估 compaction | 主代理 402 次完成交接，clip 不监控上下文 | 检查交接后的问题、证据和技能信息是否保留，是否增加重查 |
| 3 | schema 单处保存、简短笔记 ack | 源码重复明确，样本静态输入较小 | 不将小项收益外推为主要瓶颈改善 |
| 3 | 快照与图像落盘去重；分析输入改为单份有序视图 | 本地重复已测；分析仅为代码风险 | 精确分支恢复与原始审计轨迹必须保留；本地 I/O 与 API 用量分开计量 |

后续若比较这些候选，应保持相同题集、分支方法、模型、high reasoning、采样总预算和现有 0–5 judge，分别核对实际请求输入、缓存读写、输出、现有耗时和答案结果；同时人工检查工具参数与返回、交接/替换前后的证据。不能以文件变小、替换计数变多、平均 token 下降或输出文件存在，代替方法正确性与净收益验证。本次没有得出任何改动已节省费用或保持准确率的结论。

检查过程说明：代码子代理曾尝试用 `ast.parse` 提取字符串长度，被自动审批判为禁止的语法检查并拒绝。该操作未执行，已停止且未绕过；本报告依赖正常源码读取与真实 trace，不依赖那项统计。

## 7. 单次检索 QA、摘要替换与 normal 重跑

按用户后续决定实现：`run_sql` 和 `semantic_search` 接受 `question`，以独立单次 API 调用回答，主 prompt 和相关初始技能优先推荐 QA；`replace_tool` 接受摘要请求 `prompt`，独立 API 仅接收目标工具的原始结果及其参数，生成替换内容和理由；工具返回文本超过 10,000 字符时添加 reminder。成功替换后，后续输入删除 replace 调用及 acknowledgement，原工具结果显示 `Replaced with:`。实现提交为 `991c979`；独立评审发现并修复了续跑时原始工具图像未恢复的问题（`0df5ee6`）。用已有真实 trace 检查了 50 张上下文图像和 64 张原始工具结果图像，恢复后的字节与保存的 JPG 一致。正式运行前 Claude 全仓评审没有阻塞项。

先运行了同一条 day/196 的 Flex rollout：第一次尝试耗时 2,112.726 秒后遇到 API 过载；第二次耗时 1,089.547 秒，模型直接输出自然语言而未调用 `submit_answer`，被现有 JSON 校验拒绝。用户要求换回 normal 后，停止正在运行的第三次尝试，保留原始 trace。随后以 `service_tier="default"` 重新运行同一条 day/196，只有 `rollout-00`，没有运行完整 set；保留模型、high reasoning、采样和重试设置。该配置提交为 `a47cac8`。Normal 的全部 87 个已记录 API 响应均返回 `default`。

### Normal 单条结果

- 在第一次 rollout 尝试完成，耗时 **2,891.502 秒（48.19 分钟）**，主模型 24 次调用、主工具 70 次调用。最终回答把目标 boss 判为黑熊精，给出辟火罩 → 定风珠 → 绣花针 → 芭蕉扇；参考答案是凤翅将军，之后均为芭蕉扇，judge 为 **0/5**。
- 主模型单次输入峰值 **40,076 tokens**；输入加输出峰值 **40,693 tokens**。此前只有主 prompt replacement 指引的单条测试为 134,388，峰值下降 **69.72%**；主模型累计输入由 1,106,150 降到 **580,467**，下降 **47.52%**。两条轨迹的调查路径不同，且此次使用 normal，不能把费用差额解释为单一改动的效果。
- `run_sql` 的 QA 使用率 **16/16（100%）**；`semantic_search` 为 **32/32（100%）**。提前摘要共 **48 次**，显式 `replace_tool` **5 次**，合计 **53 次**。
- 主模型接收的 48 份 QA 回答共 **68,954 字符**；这些 QA API 接收的原始 SQL 结果/检索 hits 共 **5,049,929 字符**。这是工具返回文本量的比较，不是累计计费 token 节省量。
- 5 次显式替换针对 4 个视频结果和 1 个 SQL QA 结果；逐个 before/after 的 API token count 差额合计 **13,345 tokens**。核对了 **55 个**后续请求中的替换状态，目标结果已改写，replace 调用和 acknowledgement 均已移除。
- 本条没有任何工具返回文本超过 10,000 字符，因此 **reminder 没有触发**；这次自然 rollout 未覆盖阈值触发路径。

下表包含这条 rollout 内所有已记录响应，包括未完成的视频响应；不包含 judge。Reasoning 取自 API usage，已包含在 output 中。

| API 类型 | 调用数 | Input tokens | Output tokens | 其中 reasoning | 按现有计价代码计算的费用 USD |
|---|---:|---:|---:|---:|---:|
| 主模型 | 24 | 580,467 | 11,705 | 5,285 | 0.03937653 |
| Semantic QA | 32 | 875,126 | 41,808 | 31,606 | 0.26894630 |
| SQL QA | 16 | 375,947 | 40,824 | 27,588 | 0.14287500 |
| Clip QA | 9 | 1,678,155 | 395,663 | 394,367 | 0.89433300 |
| Replacement QA | 5 | 15,912 | 4,027 | 3,027 | 0.00880965 |
| 合计 | 86 | 3,525,607 | 494,027 | 461,873 | 1.35434048 |

加上 1 次 judge（727 input、72 output，其中 50 reasoning，$0.00023180），总费用为 **$1.35457228**。仅统计 completed QA 响应会得到 $0.96815678，漏掉同一次 rollout 内未完成响应的 **$0.38618370**；这部分不是另一次 rollout 或另一轮整条重试。

### 实际 trace 中的问题和边界

- 10 次 `clip_qa` 工具调用产生 9 次 API 请求：1 次请求超过既有 650 图限制，在调用 API 前返回错误。9 次 API 中 7 次 completed，2 次 incomplete；两次 incomplete 均为 `max_output_tokens`，各产生 **128,000 reasoning tokens**。还有一个 completed 视频响应产生 **87,017 reasoning tokens**，API 的 created/completed 时间差为 **446 秒**。这些工具错误返回给主模型，没有触发整条重试。请求中没有设置客户端输出 token 上限。
- 16 个 SQL QA 的原始结果与重新执行同一 SQL 得到的真实数据库结果一致；32 个 Semantic QA 的原始 preview 与其 row ID 对应的真实数据库文本一致。48 份 API 回答均实际作为对应工具结果返回；每份请求只有该次问题和证据，没有工具循环。
- 5 个 replacement 请求中的原始内容与目标工具原始结果逐项一致，没有携带其他工具结果。主模型仍在部分 `prompt` 中写入断言，但本次摘要 API 没有直接照抄：例如 SQL 工具 32 的替换拒绝了其他视频结果中的 76751、78021、80289 等精确装备结论，只保留目标原文支持的时间范围及不确定性。
- 视频 QA 的文字与所返回帧存在冲突。实际打开了 3 张返回图像：76751.508 的帧是叙事画卷，没有装备界面；80289.525 的帧是绣花针菜单，按钮为“装备”，没有文字回答所称的芭蕉扇“已装备”。对应 replacement 摘要指出了这些冲突。主模型最后仍回答了错误的 boss 与更换顺序。

原始数据与日志保留在 ignored 目录；本报告仅保存紧凑统计。运行结束后已停止本次启动的 embedding 容器。

- [Normal 完整统计与边界核对](../runs/day196-tool-qa-normal/audit.json)
- [与此前主 prompt 测试的对照](../runs/day196-tool-qa-normal/comparison.json)
- [Normal 最终答案和评分](../runs/day196-tool-qa-normal/collection/day/question-196/rollout-00/result.json)
- [Normal W&B run](https://wandb.ai/3dsmile/video-sql-v2/runs/day196-tool-qa-normal)

## 8. QA 改回可选，主模型直接替换最新结果

按用户新要求，撤掉主 prompt、工具描述和三个初始技能中优先使用 SQL/semantic QA 的推荐，保留可选 `question`。`replace_tool(tool_id, replacement, reason)` 改由主模型直接输出摘要，不再调用摘要 API；prompt 要求只总结目标结果，引用其他结果时明确标注 tool_id。运行时只接受当前输入中最新的工具结果。成功替换后，将该结果之后至本次替换轮结束的全部历史移除，包括 reasoning、消息、replace 调用及确认、同轮其他工具记录；之后新生成的 reasoning 正常保留。实现提交为 `1728e20`、`812fcd9`。

独立 diff 评审及 Claude 全仓代码评审均无阻塞项。原 GPU 4 embedding 服务启动时空闲 79.48 GiB，低于原配置所需 80.25 GiB，因此在空闲 GPU 0 上以相同镜像、有效启动命令、环境和挂载运行 embedding；两卡均为 B200，真实查询返回有效 4096 维向量。测试结束后已停止并删除本次新建容器，原 GPU 4 容器保留为停止状态。

只运行了一条 day/196、rollout-00，normal/default、同一模型与 high reasoning；第一次尝试完成，无工具错误、无整条重试。13 个已记录 API 响应全部为 default：12 次主模型和 1 次 judge。

| 指标 | 上一条：推荐 QA＋独立摘要 API | 本条：可选 QA＋直接 replace |
|---|---:|---:|
| 耗时（秒） | 2,891.502 | 76.705 |
| 总费用 USD，含 judge 和未完成响应 | 1.35457228 | 0.05726132 |
| 主模型调用数 | 24 | 12 |
| SQL / Semantic 检索调用数 | 16 / 32 | 17 / 8 |
| SQL / Semantic QA 使用数 | 16 / 32 | 0 / 0 |
| 显式 replace 调用数 | 5 | 2 |
| 独立 replacement API 调用数 | 5 | 0 |
| 视频 QA 工具调用数 | 10 | 0 |
| 主工具调用总数 | 70 | 33 |
| 主模型输入加输出峰值 tokens | 40,693 | 117,988 |
| 全部问答 API 累计 input tokens，不含 judge | 3,525,607 | 582,855 |
| 全部问答 API 累计 output tokens，不含 judge | 494,027 | 6,259 |
| 其中 API reasoning tokens | 461,873 | 3,206 |
| Judge 分数 | 0 | 0 |

本条没有调用视频 QA，调查路径也改变了，耗时和费用差额不能仅归因于 QA 推荐或 replace 实现。最终答案仍把 boss 判为黑熊精，给出辟火罩 → 定风珠 → 绣花针 → 芭蕉扇，与参考答案不符。

实际裁剪核对：两次 replace 分别针对工具 15 和 28，均为当时最新结果，并且独占该轮工具调用。下一次 API 输入逐项等于“原输入保留至目标结果，并改写该结果”，没有留下后缀。两轮共 3 个旧 reasoning item 被移除，在之后合计 8 个请求快照中没有再次出现；接下来生成的 3 个新 reasoning item 均正常保留。全部 12 个主请求的 function_call 与 function_call_output 成对匹配。before/after API token count 分别为 4,920 → 464、9,202 → 333，局部减少合计 13,325 tokens。

13 个工具结果超过 10,000 字符，全部实际带有 reminder；其中工具 15、28 被替换。其余大结果未被替换，主模型峰值 context 因而仍达 117,988 tokens。

**仍有摘要内容问题：** 第二次 replace 将 37500、61740、68820、76650、78000、78540、80280 秒的后期装备信息写进工具 28 的摘要，但目标的 75 行原始返回中没有定风珠、绣花针、芭蕉扇及对应英文名称，也没有这些精确时间。模型未标注外部 tool_id 引用。因此，本次历史/reasoning 裁剪检查通过，但仅靠当前 prompt 尚未消除跨结果混写。

- [本条完整指标](../runs/day196-latest-replace/audit-summary.json)
- [替换、reasoning 和 reminder 核对](../runs/day196-latest-replace/replacement-audit.json)
- [前后对照](../runs/day196-latest-replace/comparison.json)
- [W&B run](https://wandb.ai/3dsmile/video-sql-v2/runs/day196-latest-replace)
