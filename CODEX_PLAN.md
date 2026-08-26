# 基于 PLANS.md 的完整实施计划

## 总结

- 保留基础设施研究、索引与 harness、pilot、正式训练与评测四个阶段；原 Phase 4 删除，因为 Phase 3 已进行完整评测。
- Pilot 使用 Luna；Pilot 之后的主检索、judge、reflection、skill updater、compactor、字幕与结构化处理全部使用本地 Qwen。
- Pilot 训练每题 5 个 rollout；正式训练每题 10 个并发 rollout；所有验证与测试每题 1 个 rollout。
- 仅主检索 agent 启用 context compaction。完成第三次 compaction 后立即终止该 rollout，记 0 分并进入复盘。
- 固定 compaction prompt，不允许训练过程修改。
- 每條 skill 規則帶穩定 ID；Experience DB 只做規則與歷史證據的索引，updater 修改某條規則前先查詢該規則的歷史證據。

## 阶段与实现

### 0. 完成本地 Qwen 基础设施研究

- 继续完成 `INFRA.txt`，不能接受当前失败的 vLLM 日志作为结论。
- 使用真实约 100k-token 多模态对话测试 25–50 个并发请求，检查实际回答和完整调用轨迹。
- 记录最终服务命令、服务框架和版本、GPU 数量、context limit、并发能力、官方 generation config、图像限制、吞吐量、首 token 延迟及整体延迟。
- 移除现有 `temperature=0`；客户端不覆盖 temperature、top-p、top-k，也不设置输入或输出 token 上限。
- 后续所有本地 Qwen 运行只使用最终 `INFRA.txt` 中确认的配置。

### 1. 仓库、索引和 harness

- 初始化 Git，身份设为 `codex <codex@localhost>`；使用 `uv` 和仓库内 `/raid` 缓存，提交锁定后的 `uv.lock`。
- 从 [MAGIC-video](https://github.com/lijiazheng0917/MAGIC-video) 只移植当前项目需要的预处理逻辑，并记录上游 commit；不引入其截断、替代数据或静默恢复路径。
- 使用 [MM-Lifelong](https://huggingface.co/datasets/MM-Lifelong/MM-Lifelong) 的官方顺序、时长和区间信息。所有时间统一保存为视频内秒数和全局相对 `INTERVAL`；只有来源提供绝对时间时才保存 `TIMESTAMP`。
- DuckDB 保存 videos、questions、clue intervals、ASR segments、各粒度 captions、entities、episodic/semantic triples、activities、event chains、topic chains 和 embeddings。
- 文本入库前统一执行：拉丁字符转 ASCII 并小写、保留非拉丁字符、去除首尾空白、折叠连续空白。数据库只保留规范化文本。
- 每条可检索文本记录和每个 30 秒视频片段均生成稳定 row ID 和向量。文本及视频使用 [Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B)，采用 DuckDB 精确 cosine scan。
- 构建期间使用写连接和可恢复的磁盘分片；正式 harness 使用 `read_only`，每个并发调用创建独立连接，符合 [DuckDB 并发约束](https://duckdb.org/docs/current/connect/concurrency)。
- Experience DB 與上述檢索庫是兩個獨立檔案，寫入 `runs/<run_id>/experience.duckdb`，不註冊到 harness 的任何工具，agent 無法查詢自己的訓練歷史、judge 分數或反思。
- 建议目录为 `src/preprocess/`、`src/indexing.py`、`src/harness.py`、`src/training.py`、`src/val.py`、`src/experience.py`、`skills/initial/`、`prompts/COMPACTION.md`、`tests/`、`notes/`；运行产物写入忽略的 `runs/`。

### 2. Pilot

- 将当前 47 个 train 和 623 个 val 问题按共享 clue video 建图，使用已确认的两个连通分量作为固定划分：646 train、24 val，禁止跨集合共享视频。
- Pilot 主 agent、judge、reflection、summarizer 和 updater 使用 `gpt-5.6-luna`，reasoning effort 为 high；使用 Responses API 与 Agents SDK 的流式运行接口。[Agents SDK 运行方式](https://developers.openai.com/api/docs/guides/agents/running-agents)和 [Luna 规格与定价](https://developers.openai.com/api/docs/models/gpt-5.6-luna)作为实现依据。
- Pilot 训练 1 个 epoch，每批 5 个问题，每题 5 个 rollout；顺序以 seed 2026 固定打乱，保存完整 question ID 顺序，最后不足 5 题的批次正常执行。
- 每个 rollout 独立由 judge 评为 0–5 分，然后生成 reflection；每题生成一次汇总，每批由 updater 对运行副本中的 `SKILL.md` 和 `META-SKILL.md` 应用一个统一 Linux patch。
- updater 產生 patch 前先按 `skill_id` 取出該規則的歷史證據，連同本批彙總一起輸入；patch 成功套用後寫回對應的 `skill_evidence`。
- 初始 skill 提供简短检索指导；初始 meta-skill 提供复盘指导。评测基线使用原始 `SKILL.md`，训练组使用更新后的 skill；测试 prompt 不包含 meta-skill。
- 初始 `SKILL.md` 與 `META-SKILL.md` 的每條規則都帶穩定 ID；baseline 使用的原始 skill 也是同一份帶 ID 的版本。
- 24 个 val 问题分别执行 trained-skill 和 initial-skill 两组，每题各 1 个 rollout。
- 启用 Luna prompt cache，记录 cache write、cached input、普通 input 和 output token。Pilot API 累计费用超过 100 美元时，仅在当前顶层 rollout 完成后保存恢复状态并暂停。
- 完成 Pilot 后报告每题费用及正式规模外推。

### 3. 正式 Qwen 训练与完整评测

- 使用 datasets/Hugging Face 下载机制获取正式数据；重新生成全部正式数据，不复用 Pilot 产物。
- ASR 使用 [Qwen3-ASR-1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf)；caption merge、OpenIE、semantic、topic、event 等步骤均使用 `INFRA.txt` 指定的本地 Qwen。
- 从与 Pilot 相同的初始 `SKILL.md` 和 `META-SKILL.md` 重新开始，不继承 Pilot 训练结果。
- Experience DB 同樣重新建立，不複用 Pilot 的 provenance。
- 使用 month train 的 266 个问题训练 1 个 epoch，每批 5 题，每题 10 个并发 rollout；seed 2026 固定顺序，最后一批正常处理。
- 每个 rollout 由本地 Qwen judge 独立评分并复盘；summarizer、skill updater 和 compactor 也使用本地 Qwen。
- 对 month val 623、day test 200、week test 200 共 1023 题进行完整评测；trained-skill 和 initial-skill 各跑一次，每题每组 1 个 rollout。
- 报告 overall、day/week/month、question_type 以及每题 trained-vs-baseline 配对差值，并保留全部逐题原始结果。不存在额外 Phase 4。

## Agent 接口与 compaction

### Harness 工具

- `run_sql(sql)`：只读执行，30 秒硬超时；最多向模型返回 100 行并说明是否存在更多结果；每个单元格最多返回前 500 字符及原长度。SQL 语法或数据错误返回给模型，数据库系统错误直接终止。
- `semantic_search(query, n, table)`：只允许查询已生成 embedding 的表，返回 table、row_id、video_id、start/end、规范化文本预览及 cosine score。
- `clip_qa(video_id, start_s, end_s, question)`：建议片段不超过 10 分钟，超过 20 分钟直接拒绝。内部 agent 可调用 `sample_video`、`add_note`、`read_notes`。
- `sample_video`：接受时间、fps 和 0–1000 整数裁剪坐标；返回覆盖请求区间的 caption 和带时间标记的 4×3 contact sheet，每页 12 帧，所有帧完整分页。
- `clip_qa` 返回 `answered|unknown`、答案和至少一个证据时间点；harness 将证据时间点转换为独立截图交给主 agent。
- 主 agent 提供 `add_note`、`read_notes` 和 `agent_raise(reason)`；正式运行中同一 rollout 最多按原设置重试 3 次，缺少必要数据或确定性代码错误直接终止。
- 主 agent 最大 50 turns；达到上限按整个 rollout 重试，不截断上下文或返回半成品。

### Compaction 状态机

- 仅主检索 agent 使用 compaction；`clip_qa`、judge、reflection、summarizer、updater 和 compactor 不使用。
- 每次主 agent 响应完成后读取服务端 usage，计算 `remaining = INFRA.txt.context_limit - input_tokens - output_tokens`。
- 当 `remaining <= 50,000` 时，通过流式 runner 执行 `cancel(mode="after_turn")` 并排空事件，确保当前工具轮完整结束。
- 将规范化后的完整 transcript 交给同一 Qwen 的无工具 compactor。compactor 的唯一输出作为新对话中的 trusted handoff，旧对话不再传入。
- 第一次和第二次 compaction 后继续 rollout。第三次 compaction 输出生成后立即终止，状态设为 `compaction_limit`，分数设为 0，不调用 judge、不重试，直接进入重点分析绕圈原因的 reflection。
- Reflection 永不 compaction；当其 remaining 小于等于 50k 时，禁止继续调用工具并强制输出最终 reflection 和 meta-reflection。

### 固定项目专用 compaction prompt

```text
You are compacting an in-progress video-memory SQL agent rollout. Produce only a self-contained handoff for the same agent in a fresh conversation. Do not answer the original question, call tools, propose a new retrieval strategy, or add conclusions that are not already supported by the transcript.

Preserve all information needed to continue the exact rollout:
1. The original question, required answer format, and success condition.
2. The current stage of the investigation and the next concrete action.
3. Every verified fact, including exact table names, row IDs, video IDs, timestamps, entities, quantities, SQL predicates, joins, ordering clauses, and aggregation logic needed to reproduce it.
4. Decisive SQL and semantic-search results, compressed into the smallest set of rows or aggregates that preserves their meaning.
5. Clip observations, screenshot evidence, contact-sheet findings, caption evidence, and exact evidence timestamps.
6. Agent notes and any constraints imposed by the read-only database, tool interfaces, evidence requirements, or the user.
7. Every attempted path, its tool inputs at a reproducible level, its result, and whether it failed, was rejected, or remains unresolved.
8. Errors, contradictory evidence, hypotheses already ruled out, and the evidence that ruled them out.
9. Open questions, unresolved conflicts, and missing evidence.
10. The number of compactions already completed for this rollout.

Use exactly these sections:
Original task
Verified evidence
Working interpretations
Failed or rejected attempts
Unresolved items
Current state
Next action

Compress repetition and long tool output, but do not drop identifiers, timestamps, SQL details, contradictory evidence, failed paths, or current progress. Distinguish verified evidence from interpretation. Never invent missing information or convert an interpretation into a fact. The output will replace the entire prior conversation, so it must be sufficient to continue without access to that conversation.
```

- `prompts/COMPACTION.md` 为只读训练资产。updater 的 patch allowlist 只包含该次运行的 `SKILL.md` 和 `META-SKILL.md`，因此训练不能改动 compaction prompt。
- 第三次 compaction 后的固定复盘指令为：`Stop tool use now. Produce the final reflection and meta-reflection from the existing trace. Focus on repeated paths, why the rollout failed to make progress, which evidence was repeatedly revisited, and what skill change would prevent the loop. Do not continue solving the question.`

## Skill provenance 與 Experience DB

### 動機

`SKILL.md` 是 distill 出來的檢索 policy，`META-SKILL.md` 是複盤 policy，但兩者都只記錄「目前的規則是什麼」，不記錄「為什麼相信這條規則」。訓練若干個 batch 之後，無法回答某條規則是被哪些 rollout 建立的、被哪些 rollout 支持、被哪些 failure 推翻。缺的不是資料 —— 每個 rollout 的完整 JSON、reflection、question summary、patch、batch skill snapshot 都已經落盤 —— 缺的是 skill 與這些產物之間的 link。

### 非目標

- 不改動回答路徑。推理仍然是 question → `SKILL.md` → SQL / semantic / clip 檢索 → answer，harness 工具集不變。
- 不把 raw trace 複製進 Experience DB。DB 只存指標和 `artifact_path`，原始證據留在 `runs/`。
- 不建立「用 search 去 guide search」的語意記憶層。預設路徑是按 `skill_id` 的確定性查詢，語意檢索只在下面寫明的一種情況下使用。

### Schema

兩張表，寫入 `runs/<run_id>/experience.duckdb`：

```sql
CREATE TABLE experience (
  experience_id      TEXT PRIMARY KEY,
  question_id        TEXT,
  rollout_id         TEXT,
  batch_id           INTEGER,
  score              INTEGER,
  status             TEXT,
  reflection_summary TEXT,
  artifact_path      TEXT
);

CREATE TABLE skill_evidence (
  skill_id      TEXT,
  skill_version INTEGER,
  experience_id TEXT,
  relation      TEXT,
  patch_id      TEXT
);
```

- `status` 沿用 harness 既有的終止狀態，取值固定為 `completed`、`compaction_limit`、`agent_raise`、`turn_limit`，不允許模型自由產生新值，否則無法聚合。`agent_raise` 與 `turn_limit` 在重試耗盡、該樣本被 skip 之後才落一筆，讓訓練過程的失敗分佈保持完整。
- `score` 為 judge 的 0–5 分，`compaction_limit` 記 0 分。
- `relation` 取值固定為 `created`、`supports`、`contradicts`、`refined`、`deprecated`。
- `artifact_path` 指向該 rollout 已有的 JSON，不複製其內容。
- 規模：Pilot 646 題 × 5 rollout 約 3.2k 筆，正式訓練 266 題 × 10 rollout 約 2.7k 筆，兩張表都是千筆量級。

### 穩定 skill ID

`SKILL.md` 與 `META-SKILL.md` 的每條規則以穩定 ID 作為標題前綴，skill 用 `S`、meta-skill 用 `M`：

```text
### [S017] Temporal Aggregation

Prefer SQL aggregation over per-row clip inspection when the question asks for counts or durations.
```

- ID 一經分配不再改變，也不因規則改寫、合併或重新排序而變化。
- updater 新增規則時分配下一個未使用的編號；細化規則時保留原 ID 並遞增 `skill_version`；淘汰規則時寫入 `deprecated` 關係，而不是讓該 ID 從歷史中消失。
- 沒有穩定 ID 就無法判斷 v1 的 `prefer sql for counting` 與 v7 的 `for structured counting, aggregate in sql` 是同一條規則的演化，還是兩條獨立規則，provenance 也就無從建立。

### Updater 流程

每批結束、產生 patch 之前，updater 增加一步確定性查詢：

1. 從目前的 `SKILL.md` 解析出本批 reflection 涉及的 `skill_id`。
2. 對每個 `skill_id` 執行 `SELECT ... FROM skill_evidence JOIN experience USING (experience_id) WHERE skill_id = ?`，取出該規則歷史上的 supports 與 contradicts。
3. 把這些歷史證據連同本批彙總一起交給 updater，再產生 patch。
4. patch 成功套用後，為本批每條 reflection 寫入對應的 `skill_evidence`，`patch_id` 指向該次 patch 的落盤檔案。

只有當某條 reflection 描述的是一個對不上任何現有 `skill_id` 的新 failure 時，才對歷史 `reflection_summary` 做一次語意檢索，找出可能相關的規則。這是 fallback，不是預設路徑。

這一步只改變 updater 的輸入和 prompt，不改變 patch allowlist：允許寫入的檔案仍然只有該次執行副本中的 `SKILL.md` 和 `META-SKILL.md`，`prompts/COMPACTION.md` 依然不可訓練。

### 隔離與資訊洩漏

- Experience DB 不出現在 `run_sql` 或 `semantic_search` 看得到的表裡。否則 agent 能查到自己過往的 judge 分數、正確答案線索和 reflection，評測結果不再可信。
- reflection 進入 `SKILL.md` 前仍按原規則抽象化並去除敏感內容，具體案例只留在 `runs/` 的原始 JSON 中，由 `artifact_path` 索引。provenance 不放寬這條規則，反而是抽象化之後的規則仍然能回溯到具體案例的唯一手段。

## 持久化、测试与阶段门

- 每个 rollout 完成后立即保存独立 JSON，包括原始 responses、reasoning items、tool inputs/outputs、usage、截图路径、评分和状态；reflection、question summary、patch、batch skill snapshot 也分别落盘。
- `experience` 在該 rollout 的 reflection 落盤後立即寫入一筆，`skill_evidence` 在該批 patch 成功套用後立即寫入；兩者都只記錄指標和指向 `runs/` 的路徑。
- 恢复运行时依据实际已有产物继续，不增加完成标记、哈希清单或替代结果。
- W&B project 固定为 `video-sql-v2`，记录每个 rollout 分数、每题均值、batch 均值、模型 usage、compaction 次数及最终状态。
- 测试覆盖规范化、时间映射、表连接、只读并发、SQL 超时和返回限制、semantic ranking、notes、`agent_raise`、clip 时长与裁剪、contact-sheet 分页、caption 覆盖、证据截图、646/24 划分、第三次 compaction 终止、reflection 强制结束、prompt 不可修改、固定顺序、尾批、恢复和 patch 路径限制。
- 額外測試涵蓋：skill ID 在規則改寫與合併後保持穩定、provenance 寫入完整、Experience DB 不被 harness 工具存取。
- 集成测试必须读取真实 Pilot 数据库行、查看实际 contact sheet 和截图，并人工检查完整 agent trace、每次工具的真实输入输出和 usage。
- 阶段 1 完成并提交后暂停，交由用户审查代码；获得确认后才进入 Pilot。
- 每次大型运行前先提交改动，再按 `REVIEW.md` 让一个 reviewer 检查相对上一提交的变更行，修复所有明确问题并复查。
- 所有长期任务在命名 tmux session 中运行，并持续前台观察真实 pane 或日志直至结束；完整日志写文件，stdout 和 `AGENT_STDOUT.txt` 只记录里程碑。
