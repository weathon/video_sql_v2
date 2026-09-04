已按 REVIEW.md 全文审查了当前工作区代码(src/harness.py、src/training.py、src/val.py、skills/initial/、data/test-seed-0.json),并用一个 verifier 子代理独立复核。下面只列会影响正式运行的 plain-yes 项;测试集抽样(300 题,day/week/month 各 100,timeline 与 split 一致)、index 与 embedding 产物(index.duckdb 存在,7 表 x 2 共 14 个 npy,clips 有 day/week/month)、.gitignore(runs/、data/pilot/ 已忽略)、compaction 三次逻辑(第三次先产出 handoff 再抛 CompactionLimit,score=0,不判分,进入 reflection)均核对无误。

发现:

1. src/training.py:26 `timeline = item.get("timeline") or item.get("video_id") or "month"`。val 走 data/test-seed-0.json,每题都带显式 "timeline",正确;但训练走原始 month/train.json,而 "timeline" 和 "source_split" 是被删掉的抽样脚本后加的字段(source_split 的值是源文件名,原始文件不可能有),所以训练时会落到 item.get("video_id")。按 notes/index.md,month 的 video_id 是具体视频标识,不是字符串 "month",而 PLANS.md:90 要求答题 prompt 携带的是逻辑时间轴 day/week/month。若原始 month/train.json 含 video_id 键,则 266 道训练题的 prompt 会全部被标成 `timeline: <video_id>` 而非 `timeline: month`,与协议不符。该数据集路径被沙箱屏蔽,我无法读取确认它是否含 video_id 键,需你在开跑前核对原始文件字段。此外这一行本身就是 .get 加多级 or 兜底(REVIEW.md 第 6/7 项),用替身填补缺失字段。标签 TRUST-FATAL(取决于原始文件是否有 video_id)。

2. src/harness.py:701-714 judge 用 `client.responses.parse(text_format=JudgeScore)`,而 JudgeScore(harness.py:54-55)只有一个 `score: int`,前面没有 reasoning 字段,同时又传了 `reasoning={"effort": ...}`。按你自己全局规则,结构化/受限解码会从第一个 token 起强制符合 schema,从而关闭思考。正式运行 judge 是本地 Qwen3.8-Flash-Next-FP8 走 vLLM,若服务端未在 `</think>` 之后才施加语法约束,judge 会直接吐裸整数、reasoning effort 失效;而准确度 = judge 平均分 / 5,会直接受影响。试点用 gpt-5.6-luna(Responses API,reasoning 是独立 item)不受此影响。需按正式服务实际的 reasoning-parser/结构化输出配置核对。标签 TRUST-FATAL(取决于服务端是否在结构化输出下保留思考)。

其余为非阻塞的次要项,仅列出供你判断,不影响能否开跑:

3. src/harness.py:149-181 has_three_repeated_messages 只提取 type=="message" 的文本,忽略只含 tool-call、无文本的 completion(response_message_text 返回 "" 会被当成重置)。最常见的自循环——反复发同一个 tool call 且无正文——不会被这个检测捕获,只能靠 100 轮上限兜底。PLANS.md:107 的"排除 reasoning 后 raw characters 完全一致"可理解为也包含 tool-call 参数,故实现比规格窄。标签 ANNOYING。

4. src/harness.py:233-235 与 269-271,连接已是 read_only=True,却又按词边界拦截含 update/insert/copy/load 等词的查询,会误拒像 `WHERE text LIKE '%update%'` 这类合法只读检索,可能轻微影响个别题的检索。标签 ANNOYING。

5. src/harness.py:467 sample_video 取帧时 `next(row ... or row is rows[-1])`,当采样时间点不落在任何 clip 覆盖区间时,静默退回最后一个 clip 并按 `t - start_s` seek,属于用替身而非 raise/skip 的边缘兜底。标签 ANNOYING。

我尝试建自己的 venv 去查 visual_clips.filename 的存储格式(harness 用 repo_root / filename 解析,格式不对会静默让所有画面读取失败),`uv venv` 被沙箱拦截,按规则我停在这里,未创建任何 venv,这一项未验证。
