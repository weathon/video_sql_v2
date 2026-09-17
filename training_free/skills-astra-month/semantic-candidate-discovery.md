---
name: semantic-candidate-discovery
description: 用于关键词未知、存在改写或视觉描述未被文字索引时发现候选事件。
---

---
name: semantic-candidate-discovery
description: 用于关键词未知、存在改写或视觉描述未被文字索引时发现候选事件。
---

# Introduction

`semantic_search` 对选定嵌入表中的行计算相似度，再通过 SQL 筛选和排序。相似度提供候选优先级，不证明事件或属性成立。

# When to use

需要从含义定位对话、行为、对象、属性或视频片段时使用。

# Main content

1. 调用 `semantic_search(query, table, sql, question)`。按内容选择 `episode.text`、`entity.text`、`semantic.text`、`topic_facts.fact`、`event_steps.description`，或使用视频嵌入的 `visual_clip`。
2. 每次 query 聚焦一个独特线索。复合问题可分别搜索事件、人物行为或结果，再用时间和关系连接。字面原话已知时，可直接结合 SQL 精确检索。
3. SQL 查询临时表 `semantic_hits`。它包含所选表原列，以及 `cosine_sim`、`start_ts`、`end_ts`、`start_s`、`end_s`。`video_id` 标识源视频，所有数值时间均为该视频内的秒数。全部 14 个视频中的行都参与检索，没有 query-time 截止。
4. 发现候选时通常按 `cosine_sim DESC` 排序；已获得相关集合后再按时间排序。已知日期或窗口必须写入 SQL 条件，QA 问句中的日期不会约束返回行。
5. `question=null` 返回最多 100 行原始结果，长文本会缩短。提供 question 会调用独立 QA，读取完整结果并返回支持 ID 和时间，同时消耗共享调查模型调用。
6. 检查候选的实际内容和时间，再用原始记录确认。空字面搜索不证明事件不存在；可改用动作描述或邻近锚点。不要不加理由地重复完全相同的 query 和 SQL。
7. 对相似度阈值做 COUNT 只得到匹配行数，真实事件数还需要去重和事件资格核验。
