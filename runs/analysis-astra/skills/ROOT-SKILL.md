# Introduction

本指南将视频记忆检索拆分为可组合的小技能。先确定问题要求的事件、人物、对象、时间范围和答案单位，再选择相应流程。技能名称和描述仅用于路由；应用前调用 `read_skill(name)` 阅读完整内容。

# When to use

开始检索、出现新的证据分支，或发现当前方法混淆了时间、身份、对象或统计口径时使用。

# Main content

- 开始调查或整理多条线索：读取 `retrieval-plan-and-evidence-ledger`。
- 用含义发现候选片段：读取 `semantic-candidate-discovery`。
- 编写 SQL、读取元数据或追踪表间关系：读取 `bounded-sql-and-evidence-links`。
- 限定日期、解释原生时间戳或调用视频工具：读取 `timestamp-normalization`。
- 结果被截断、QA 未提供依据或需要精确措辞：读取 `complete-source-text-retrieval`。
- 同一活动重复出现，需确定题目所指场次：读取 `recurring-event-disambiguation`。
- 判断开始、结束、经过时间或持续时间：读取 `temporal-boundaries-and-durations`。
- 查询录像或某日记录的起止时间：读取 `recording-session-boundaries`。
- 排序、寻找首次、末次或第 n 次：读取 `event-order-and-ordinal-selection`。
- 已定位片段，需要安排视觉核验：读取 `visual-check-scoping`。
- 视觉回答相互矛盾、返回 unknown，或截图与答案不符：读取 `visual-evidence-reconciliation`。
- 关联人物外观、姓名、说话者和行为角色：读取 `person-role-and-appearance-grounding`。
- 判断物品归属、交接方向或是否为同一实例：读取 `object-instance-and-handoff-tracing`。
- 识别屏幕、品牌、文字、数字或标签：读取 `exact-screen-and-label-reading`。
- 判断容器关系、物体类型或颜色部位：读取 `spatial-relations-and-object-classification`。
- 回答说了什么、谁提出建议或某句话的含义：读取 `dialogue-facts-and-attribution`。
- 回答原因、动机、用途或计划改变：读取 `causal-and-purpose-retrieval`。
- 追踪选择、购买、使用、归还及最终状态：读取 `object-state-and-completion-tracing`。
- 统计跨片段、跨阶段的重复事件：读取 `event-coverage-and-deduplication`。
- 统计短暂可见动作及其完成次数：读取 `atomic-visual-action-counting`。
- 统计不同物品、类别、地点或来源：读取 `distinct-item-and-source-counting`。
- 统计某一时刻或最终摆放状态的可见物品：读取 `instantaneous-visual-inventory`。
- 统计参与者、初始成员、全过程成员或坚持至完成的人：读取 `participant-set-and-role-counting`。
- 使用现场报数、明确总数或对话中的数量：读取 `session-tallies-and-numeric-claims`。
- 计算付款、成交、修正后金额或多笔总额：读取 `final-values-and-transaction-totals`。
- 求最大值、最小值、排名或并列：读取 `scoped-extrema-and-rankings`。
- 回答多选、例外、缺失标签或完整候选集合：读取 `option-wise-and-exception-retrieval`。
- 跨日比较活动或追溯物品来源：读取 `cross-day-provenance-and-comparison`。
- 提交答案前：读取 `evidence-grounded-answer-handoff`，核对回答是否完整、精确且有对应依据。

按实际需要组合技能。例如，“第二次交接中拿了多少件物品”需要场次消歧、序数定位、交接追踪和相应计数技能；只有仍存在可见细节缺口时才增加视觉核验。
