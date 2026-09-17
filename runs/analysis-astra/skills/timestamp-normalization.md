---
name: timestamp-normalization
description: 用于解释 DHHMMSSFF 时间戳、构造日期范围，以及将候选窗口转换为视频工具的全局秒数。
---

---
name: timestamp-normalization
description: 用于解释 DHHMMSSFF 时间戳、构造日期范围，以及将候选窗口转换为视频工具的全局秒数。
---

# Introduction

原生节点时间、文字形式的链时间和视频工具秒数使用不同表示。时间基准错误会检索错误日期，或产生无效片段。

# When to use

按日查询、计算时间差、调用 `clip_qa`，或时间过滤意外返回空结果时使用。

# Main content

1. 节点时间编码为 `DHHMMSSFF`：D 为日序号，FF 为该秒内的源帧号。DAY5 的原生日期范围是 `500000000 <= ts < 600000000`，不是少一位的范围。
2. 视频工具使用从 DAY1 00:00:00 起的连续秒数。优先采用语义结果的 `start_s/end_s`；手动转换整秒部分时使用 `(D-1)*86400 + HH*3600 + MM*60 + SS`。
3. FF 不是默认的百分之一秒。只有已知源帧率时才据其计算分数秒；不要将原生整数或当天钟表秒数直接传给视频工具。
4. 空范围查询后，先检查已知行和受限 MIN/MAX，核实编码量级、日序号及结构化 day 的大小写，再解释为空缺事件。
5. `topic_facts.time` 和 `event_steps` 的 day/start_time/end_time 按实际字符串格式解释；语义结果补充的字段不自动存在于原表。
6. 视频返回后核对实际来源时间与场景。工具接受区间仍可能采到错误日期，不能只检查调用是否成功。
7. 时间差在统一时间基准下计算，不直接相减 DHHMMSSFF 整数。
