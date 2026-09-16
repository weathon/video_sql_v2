# MM-Lifelong Week rollout 结果

Week test 全部 200 题已完成，共 200 条主线和 996 条分支，1196 条最终轨迹均有答案和 GPT-5 评分。

| 指标 | 最终结果 |
| --- | ---: |
| 每题内平均后再跨题平均的映射分数 | 0.5045833333333334 |
| At least one correct rate | 0.695（139/200） |

题目 10、20、27、72 各有 4 个候选点并生成 4 条分支，其余题目各生成 5 条分支。分数映射为 4/5→1、3→0.5、0/1/2→0。

- 紧凑逐题结果：[mmlifelong_week_results.json](mmlifelong_week_results.json)。
- W&B：[sn5c3p7j](https://wandb.ai/3dsmile/video-sql-v2/runs/sn5c3p7j)，最终状态为 finished，云端指标与磁盘重算一致。
- 完整 batch、原始请求响应、工具轨迹与图片配方：`runs/mmlifelong-week/collection/`，保持 Git 忽略。
- 最终核验记录：`final-artifact-audit.json`、`wire-media-audit.json`、`persisted-media-scan.json`、`wandb-final-verification.json`，均位于上述收集目录。

## 方法与核验

使用 Qwen/Qwen3.8-27B-FP8，前后各 100 题分别绑定两个 TP=2 服务，每端并发 10，保留 thinking/history。检索使用完整七天 EgoLife 数据和原生时间格式，无 query-time 截止；原题、原答案与题目顺序保持不变。SQL/semantic QA 沿用获准的 16384-token 上限及截断重试；没有设置采样参数或 reasoning effort。GPT-5 使用官方 rubric 和明确的 Final Score。

已核对全部 996 条分支的首轮输入、工具定义、notes、替换状态和预算与选定快照一致；全部 1196 份评分均与实际 judge 响应相符。实际工具输入输出覆盖 SQL、向量检索、视频问答及工具结果替换。持久化请求及轨迹未发现图片或视频 data URL、JPEG/PNG base64，也没有保存图片或视频文件；保存的是可重建帧的配方。小范围真实调试另行覆盖零候选、少于五个候选、失败主线分支、四次重试、断点恢复和图片重建字节与画面。

正式运行前完成独立改动审查、Claude 全仓库审查及遗留验证闭环，记录位于 `runs/mmlifelong-week/full-claude-review.json` 和 `claude-review-closure.json`。运行期间的局部修正处理了录像空档、视频单帧输入、vLLM 数字错误码对应的上下文超长、SQL TokenError 分类和共享 embedding 缺失检查；最终源码提交为 `081ce4e`。

## 恢复与收尾

共享 embedding 容器在运行期间被删除，收集停止后按保存的原始启动命令恢复；同一真实请求的 4096 维向量与原输出逐项一致。第 97、192 题受影响的 `branch-002` 及旧评分已归档至 `runs/mmlifelong-week/maintenance/embedding-service-removed/`，随后从原选定前缀完成，最终评分分别为 5、4。已完成答案与随机选点均复用。

两个 Week Qwen 容器已正常退出并手动删除。共享 `mmlifelong-month-embedding` 服务保留在 GPU 2、端口 8214。收尾记录位于 `runs/mmlifelong-week/cleanup.json`。
