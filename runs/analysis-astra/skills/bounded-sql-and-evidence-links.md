---
name: bounded-sql-and-evidence-links
description: 用于编写受限 SQL、确认实际表字段，以及通过明确关系追踪原始证据。
---

---
name: bounded-sql-and-evidence-links
description: 用于编写受限 SQL、确认实际表字段，以及通过明确关系追踪原始证据。
---

# Introduction

数据库包含节点、主题链、事件链及明确的边关系。查询必须使用实际字段，并限制结果范围。

# When to use

调用 `run_sql`、连接表、读取元数据、聚合或查找证据来源前使用。

# Main content

每条查询，包括带 QA 的查询，都必须具有相关的外层 WHERE、有限 LIMIT、HAVING/QUALIFY、采样，或紧凑的全局 COUNT/MIN/MAX/SUM/AVG 聚合。ORDER BY 和 QA 问句不构成范围限制。先按日期、时间、实体或已知 ID 缩小范围，再读取详细文本。

表结构：

- `entity`、`episode`、`semantic`、`visual_clip`：`id, node_type, text, start_ts, end_ts, metadata`。
- `metadata` 是原生 JSON。episode 包含 `granularity, video_path, date, start_time, end_time`，粒度有 `30sec, 3min, 10min, 1h`；entity 包含 `normalized_name, frequency`；semantic 包含 `subject, predicate, object`；visual_clip 包含 `video_path, date, start_time, end_time`。用 `metadata->>'granularity'` 取文本值。
- `topic_chains(id, entity, has_lifecycle)`；`topic_facts(id, chain_id, time, fact, location)`。
- `event_chains(id, name, key_entities)`；`event_steps(id, chain_id, step_index, day, start_time, end_time, description)`。
- `topic_entities(chain_id, entity_id)`；`event_entities(chain_id, entity_id)`。
- 边表：`mentioned_in` 连接 entity→episode，`appeared_in` 连接 entity→visual_clip，`has_property` 连接 entity→semantic，`temporal_next` 和 `contains` 连接 episode→episode；`co_clip_visual_episode`、`co_clip_episode_visual` 保留两种原始方向。边表列为 `src_id, dst_id, edge_type, weight`。

事实和步骤通过 `chain_id` 连接。只使用明确存在的边，不虚构字幕关联或物体状态表。原始 topic_facts/event_steps 没有语义搜索临时表补充的时间字段。

`run_sql(question=null)` 最多返回 100 行并缩短长文本；QA 可读取完整查询结果，但其总结仍需对应来源。文本列已 casefold，LIKE 模式使用小写；结构化 day 字段按实际值匹配，例如 `DAY3`。

被范围校验拒绝时重写为更小的合法查询。跨日 UNION 若被拒绝，可分别查询各日；不要将这种限制扩展为所有 UNION 都不可用。
