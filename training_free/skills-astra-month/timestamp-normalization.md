---
name: timestamp-normalization
description: 用于保留 video_id、解释片内秒数，以及将候选窗口对应到源视频。
---

---
name: timestamp-normalization
description: 用于保留 video_id、解释片内秒数，以及将候选窗口对应到源视频。
---

# Introduction

Month 的数值时间是各源视频内的秒数，包含分数秒。缺失视频身份或混淆时间基准会检索错误场次，或产生无效片段。

# When to use

按日查询、计算时间差、调用 `clip_qa`，或时间过滤意外返回空结果时使用。

# Main content

1. 保留每个候选的 `video_id`；`start_ts/end_ts` 和语义结果的 `start_s/end_s` 都是该视频内的秒数，不是跨日连续时间。
2. 调用 `clip_qa` 时传入同一 `video_id` 及片内 `start_s/end_s`，明确指定 fps。无需应用 EgoLife 的日期编码转换。
3. 数值视频 ID 给出直播顺序；跨视频比较时同时保留视频身份和局部时间，不按局部秒数直接排序全部视频。
4. 空范围查询后，先检查已知行和受限 MIN/MAX，核实 `video_id` 及片内区间，再解释为空缺事件。
5. `topic_facts` 和 `event_steps` 有 `video_id, start_ts, end_ts`；源时间字符串使用 HH:MM:SS，`event_steps.day` 为 NULL。
6. 视频返回后核对实际源视频、时间与场景。工具接受区间仍可能采到错误场次，不能只检查调用是否成功。
7. 同一视频内可直接计算秒数差；跨视频的钟表时间或经过时长需要明确的源时间对应关系，不直接相减不同视频的局部秒数。
