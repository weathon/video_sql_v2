# 事故报告：将不必要的 `--privileged` 和 `--gpus all` 写入最终 vLLM Docker 配置

日期：2026-09-03

责任方：Codex assistant

## 摘要

我在记录 `Inferact/Qwen3.8-Flash-Next-NVFP4` 的最终 Docker 启动命令时，直接保留了官方示例中的 `--privileged` 和 `--gpus all`，没有先验证 vLLM、GPU 映射、CPU KV offload 或当前容器配置是否需要这些参数。该服务只应访问指定 GPU，现有测试和设备要求均未证明必须使用特权容器，最终配置不应包含这两个参数。

## 已验证事实

- 提交 `a73942b` 中的 `notes/INFRA.txt` 将 Flash Next 命令标为 Current best choice，并包含 `--gpus all --privileged` 与 `CUDA_VISIBLE_DEVICES=5,6`。
- `--gpus all` 将全部 GPU 授予容器，`CUDA_VISIBLE_DEVICES=5,6` 只在容器内部限制 CUDA 进程可见设备，没有在 Docker 设备授权层将容器限制为 GPU 5 和 6。
- 使用 `--privileged --gpus device=5,6` 启动的受控检查中，容器内 PyTorch 实际看到 8 张 GPU，而不是指定的两张。
- 在同类容器中加入 `CUDA_VISIBLE_DEVICES=5,6` 后，PyTorch 只看到两个逻辑设备，其 UUID 对应物理 GPU 5 和 6。
- 早先的 Qwen3.8-27B 测试因此实际使用了物理 GPU 0 和 1，而不是指定的 GPU 5 和 6；该次结果按用户决定保留且不重跑。
- Flash Next 测试使用 `CUDA_VISIBLE_DEVICES=5,6` 后实际运行在物理 GPU 5 和 6，但该测试没有证明 `--privileged` 是必要参数。

## 影响

`--privileged` 扩大了容器对宿主机设备和内核能力的访问范围；`--gpus all` 又明确将全部 GPU 授予容器。两者使最终命令没有在 Docker 设备授权层隔离 GPU 5 和 6，只依赖额外的 `CUDA_VISIBLE_DEVICES` 才让 CUDA 进程选择目标 GPU。早先一次测试也因设备隔离失效而使用了错误的物理 GPU。

## 原因

我把官方示例中的继承参数当作可直接保留的配置，没有逐项确认 `--privileged` 和 `--gpus all` 对当前服务的必要性，也没有要求 Docker 在设备授权层只暴露目标 GPU。

## 修正要求

- 从最终 Flash Next Docker 命令中删除 `--privileged`。
- 将 `--gpus all` 改为 `--gpus '"device=5,6"'`，在 Docker 设备授权层限制容器只看到目标 GPU。
- 在模型加载前检查容器内 PyTorch 的设备数量和每张卡的 UUID，确认其对应指定的物理 GPU。
- 官方示例中的权限、挂载、环境变量和设备参数必须逐项验证后才能写入最终配置。
