# 事故报告：将单请求测试参数 `--max-num-seqs 1` 写入最终基础设施计划

日期：2026-08-31

责任方：Codex assistant

## 摘要

在 Qwen3.8-27B 的 1024 张图片、图片加文本共 230k tokens、单 request 超长上下文测试中，我将 `--max-num-seqs` 设为 1，用于限制该次测试只运行一个序列。随后我把这条特定测试命令写入 `INFRA.txt`，标为 fastest verified vLLM Docker launch，并让 `CODEX_PLAN.md` 将该文件作为后续本地 Qwen 运行的最终配置来源。这个参数只能描述单请求测试条件，不应被写成通用最终服务配置。

## 证据

- `notes/INFRA.txt` 的旧 Qwen3.8-27B 命令仍包含 `--max-num-seqs 1`。
- `CODEX_PLAN.md` 要求后续本地 Qwen 使用 `notes/INFRA.txt` 确认的配置，并要求测试 25–50 个并发请求。
- `--max-num-seqs 1` 最多允许一个序列同时运行，与 25–50 个并发请求的计划不一致。

## 影响

如果按旧 Qwen3.8-27B 命令启动正式服务，服务端并发序列数会被限制为 1，无法执行计划中的 25–50 路并发测试，也不能作为通用吞吐配置。

## 原因

我把单请求超长上下文实验的隔离参数当成了模型的最终服务参数，没有区分实验专用命令与通用服务命令；写入最终计划前也没有将 `--max-num-seqs 1` 与计划中的并发要求逐项核对。

## 当前状态

`notes/INFRA.txt` 最上方的 Flash Next 当前最佳配置使用 `--max-num-seqs 200`，`CODEX_PLAN.md` 目前明确引用最上方 Current best choice，因此当前正式 Flash Next 配置不是 1。旧 Qwen3.8-27B 条目仍保留 `--max-num-seqs 1`，本报告明确将其认定为单请求测试配置错误进入最终基础设施文件的历史记录。

## 防止再次发生

- 单请求、压力测试和正式服务命令必须分别标记，实验隔离参数不得标为通用最佳配置。
- 写入最终计划前，逐项核对并发限制参数与计划要求，至少包括 `--max-num-seqs` 和 `--max-num-batched-tokens`。
- 最终计划引用配置时必须同时指定模型和配置段，不能只引用文件名。
