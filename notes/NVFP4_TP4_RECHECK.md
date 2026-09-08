# NVFP4 四卡复测（2026-09-08）

循环或重复 token 不能直接证明 NVFP4 权重量化导致模型退化。[NVIDIA 的讨论](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4/discussions/4)包含切换 vLLM 版本后恢复正常的报告，也有未恢复的反馈，并建议多卡尝试 `--disable-custom-all-reduce`。该讨论针对 Qwen3.6-27B；不能据此断言本机 Flash Next 异常具有相同根因。

本次实际启动 `Inferact/Qwen3.8-Flash-Next-NVFP4`，权重 revision 为 `103a7608316173ca6edd49929544244de7ffda70`，使用 GPU 0、1、2、3（B200），TP=4。镜像 `vllm/vllm-openai:qwen38-flash-next` 中实际版本为 vLLM `0.1.dev20073+g8e685d198`、Torch `2.13.0+cu130`、FlashInfer `0.6.17`。

设置保留 `max-model-len=314573`、YaRN factor 1.2、`max-num-seqs=50`、MTP 3、prefix cache、CPU KV offload，以及 medium reasoning、thinking/preserve_thinking。启用了 `--disable-custom-all-reduce`；未设置请求生成参数或输入/输出 token 上限。

人工检查的实际输出：

| 请求 | 结果 |
| --- | --- |
| Responses：西班牙语要求只回答 OK | reasoning 为 `O`，没有最终答案 |
| Responses：7 × 24 − 39 的中文计算题 | 开头已经语义异常，随后持续重复“带”；人工中止时已记录 30,176 字符，其中 30,015 个“带” |
| Chat Completions：同一 OK 指令 | reasoning 为 `O4, the short "OKV" - by OK, OK`，没有最终答案 |
| Chat Completions：2 + 2 | reasoning 为 `2068`，没有最终答案 |

本次四卡配置未通过提示词检查；异常并非只出现在 Responses 路径。关闭 custom all-reduce 未消除这些异常，但本次测试没有确定底层根因，也没有证明权重退化。后续 SQL、结构化输出、工具往返和图像测试未执行。

完整启动命令、服务日志、请求及流式原始输出保存在 [runs/nvfp4-tp4-recheck-20260908](../runs/nvfp4-tp4-recheck-20260908/)。上述测试实际使用的命令保留在 `server-command-before-removing-disable-custom-all-reduce.json`；`server-command.json` 已按用户要求移除 `--disable-custom-all-reduce`。

## 移除参数后的复测

按修改后的同一四卡命令启动成功。实际启动日志记录 `disable_custom_all_reduce=False`，TP 通信后端为 `CUSTOM`、`SYMM_MEM`、`PYNCCL`。

再次人工检查四条相同提示词的完整输出：Responses 的 OK 指令只产生 reasoning `T`，中文计算题只产生 `1 1 算式`；Chat Completions 的 OK 指令产生 `user: OK、APT：`，2 + 2 问题只复述 `What is 2 + 2?Answer in one sentence.`。四条均结束但没有最终答案，本次未观察到长循环，仍未通过回答检查。

本轮命令、请求和原始输出保存在 [custom-all-reduce](../runs/nvfp4-tp4-recheck-20260908/custom-all-reduce/)。
