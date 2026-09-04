# 事故报告：未核实官方 EPD 接口并自行重写调度层

日期：2026-09-03

责任方：Codex assistant

## 摘要

我没有先完整阅读 vLLM 的 Disaggregated Encoder 官方文档和当前版本接口，仅因官方示例使用 `/v1/chat/completions`，便错误推断官方 EPD 只支持 Chat Completions。基于这个未经验证的假设，我自行实现了 `infra/responses_epd_proxy.py`，手动拆分图片请求、调用 encoder，再把请求转发给 PD。

## 已验证事实

- vLLM 官方文档没有明确声明 EPD 只支持 Chat Completions。
- 官方示例使用 `/v1/chat/completions`，只能证明该示例展示了这个接口，不能证明其他接口不受支持。
- 我在作出架构决定前没有核对官方 EPD 的当前接口和参考实现。
- 本次运行实际启动了独立的 proxy、encoder 和 PD 容器，并使用 `ECExampleConnector` 交换 encoder outputs。
- 我将 `ECExampleConnector` 的 `shared_storage_path` 配置为 `/epd_shared/cache`，并把该路径 bind mount 到宿主机 `/raid`，使临时 encoder outputs 写入没有显式容量上限的持久磁盘目录。

## 影响

未经验证的接口假设导致项目增加了不必要的自定义代理和容器编排。自定义代理承担了官方 EPD 已有调度能力应承担的工作，同时引入了错误的 encoder cache 生命周期和存储设计：运行期临时数据被写入持久磁盘，且容量只受磁盘剩余空间限制。发现问题后，本次训练以及相关 proxy、encoder、PD 和 embedding 服务均已停止；容器和 cache 现场保留，未删除。

## 原因

我把“官方示例使用 Chat Completions”当成了“官方只支持 Chat Completions”，没有查找明确的限制说明，也没有先验证官方 Disaggregated 配置能否直接满足当前服务接口。我随后把自己的推断写成实现，而不是停止并核对文档。

我还混淆了两类 cache：Hugging Face 和 uv 的下载、构建 cache 应放在 `/raid`，encoder output cache 是运行期临时数据，不应因此被放入持久磁盘。

## 修正要求

- 在重新设计或启动 EPD 前，完整核对当前 vLLM 版本的官方文档、启动参数和参考实现。
- 不得因为官方示例只展示某个 API endpoint，就推断官方功能不支持其他 endpoint；必须找到明确限制或直接验证。
- 删除基于该错误假设增加的自定义 EPD 调度实现，使用官方 Disaggregated 能力。
- 不得将 encoder output cache 与 Hugging Face、uv cache 混为一类；encoder output cache 不写入无限制的持久磁盘目录。
- 正式运行前检查真实启动参数和多轮调用 trace，确认请求由官方 EPD 路径处理。
