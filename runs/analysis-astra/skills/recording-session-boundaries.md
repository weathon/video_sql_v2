---
name: recording-session-boundaries
description: 用于录像或某日记录的起止时间及跨日起止差，避免混入场景中的拍摄动作。
---

---
name: recording-session-boundaries
description: 用于录像或某日记录的起止时间及跨日起止差，避免混入场景中的拍摄动作。
---

# Introduction

记录边界应从源媒体对应的时间记录查询。场景里说“开始拍摄”可能发生在录像开始很久之后。

# When to use

问题询问某天录像何时开始、结束，或两天结束时间相差多久时使用。

# Main content

1. 先区分记录会话边界与会话中某次拍摄行为。
2. 在目标日期范围内，对 `visual_clip` 等源媒体记录查询 `MIN(start_ts)` 或 `MAX(end_ts)`，再读取首尾几行及元数据核对。
3. 可用 episode 边界交叉检查，但不要用 semantic 等派生表更晚的最小时间替代媒体起点。
4. 比较多天时分别取得每一天的受限聚合结果，保持日期标签；混合聚合会隐藏各日端点。
5. 如需读取画面时钟，核验开头或结尾的窄片段。计算差值前统一时间表示，并按问题要求报告钟表时间或时长。
