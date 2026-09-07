# Qwen3.8-27B-FP8：B200 / vLLM 最大 batch 吞吐配置研究

日期：2026-09-07。对象：video_sql 项目的部署决策者。研究范围：现有 FP8 模型服务配置，不包括 Trace2Skill 接口、模型更换、实际启动、profiling 或 benchmark。

## 直接建议

建议采用 **官方 FP8 权重 + TP2 + BF16 KV + 32768 batched tokens + MTP1 + FlashInfer autotune** 作为下一次正式运行的具体候选；保留 encoder 分离、异步调度、prefix caching、SimpleCPUOffloadConnector eager 模式及 thinking/preserve_thinking。客户端总 API 并发仍为 128；服务器 max_num_seqs 由用户确定的原生容量流程计算，不能直接写成 128 或沿用 24。

这是一组有证据支持的候选，**还没有本项目实测证明它比原配置快，更没有证明它是全局最优**。检索未找到同时满足 Qwen3.8-27B 官方 FP8、B200 TP2、314573 上下文、128 API 并发、视频 encoder 分离和 CPU KV offload 的公开对照实验。32768、MTP1、autotune 三项的证据强度不同，见下表。

| 参数 | 原配置或当前实际默认 | 建议 | 依据与适用限制 |
|---|---|---|---|
| 模型、并行 | Qwen/Qwen3.8-27B-FP8；GPU 0,2，TP2 | 保留 | 不套用短上下文 DP8 的结果；GPU 3 留给 encoder，4 留给 embedding |
| max_model_len / YaRN | 314573；factor=1.2，original=262144 | 原样保留 | 长度不能为了获得漂亮吞吐数字而缩到 2.5K/4K |
| max_num_batched_tokens | 未显式传；镜像对 ≥160GiB GPU 默认为 16384 | **32768** | 提高单次迭代可处理的 prefill token 数；当前实现的 throughput 模式也会把未指定的 16384 翻倍。属于合理候选，不是已测得提升；activation 峰值必须在 N=128 profile 中重算 |
| speculative-config | MTP，num_speculative_tokens=3 | **MTP1** | 最接近硬件的一手 B200/Qwen3.5-27B FP8 报告中 MTP1 胜过 MTP2；精确 Qwen3.8 模型在 RTX PRO 6000 上又有 MTP2/MTP3 胜出的相反报告。长多轮 prefix reuse 还会承担 spec cache-tail 重算，不能只看 acceptance |
| FlashInfer autotune | 显式关闭 | **开启** | 当前 O2/O3 配置均默认启用；找到的 B200 crash 是旧 BF16 MoE+EP 路径且已关闭修复，不是本项目 dense FP8。未找到本模型开启/关闭的独立吞吐增益量化 |
| attention backend | B200 causal 路径默认优先 FLASHINFER | 显式 FLASHINFER | 是固定已存在的选择，不能计为新增收益；保持原生 GDN 路径，不把所有注意力都强行切成 FA4 |
| async scheduling | 当前镜像对兼容 MTP 默认开启 | 显式开启 | 是固定已有默认；与异步 spec 同用时镜像会关闭 cascade attention，不能把二者收益叠加 |
| 权重 GEMM | qwen3_5_text 在 Blackwell 自动排除 DeepGEMM E8M0 | 保持实现的选择 | 源码注明该格式对这类架构有精度问题；不能从“Blackwell 上 DeepGEMM 更快”推导出应强开 |
| KV dtype | auto → BF16 | **保留 auto/BF16** | 官方 FP8 权重并不包含 KV calibration scales；本镜像 parser 没有 calculate-kv-scales 参数。FP8 KV 是额外量化，不能当作无精度代价的普通配置开关 |
| Mamba/GDN cache | prefix caching 对应 align，dtype auto | 保留 align、原生 dtype | Qwen 实现直接拒绝 all 模式；模型声明 mamba_ssm_dtype=float32，不改成 FP16/BF16 来扩大容量 |
| MM processor cache | Qwen 服务默认 4GiB；encoder 共享缓存另有 tmpfs | 显式 4GiB，保留共享 encoder cache | processor cache、GPU encoder cache、EC 共享缓存、KV cache 是不同层。没有工作集数据支持随意扩到 16/64GiB；禁止把 0 当作无害默认 |
| CPU KV offload | SimpleCPUOffloadConnector，220GiB/rank，lazy_offload=false | **保留** | 当前实现支持 hybrid cache 和异步传输；没有匹配负载的证据证明换 LMCache/另一 connector 或 lazy=true 更快。CPU 容量不计入 resident C |
| GPU memory utilization | PD 0.90；encoder 0.45 | 保留 | 预算从正式原生 profile 得出；不能把别人的 0.92/0.95 直接视为本项目最优 |
| 编译与 CUDA graph | PD 默认 O2；encoder eager | 保留 | 当前 O2 已配置 FULL_AND_PIECEWISE；不盲目添加大 capture-size 或 O3 并宣称提速。encoder 的 eager 保留原架构选择 |

实现依据为从现有 stopped 容器只读导出的实际源码，尤其 [B200 默认和 throughput 模式](sources/engine__arg_utils.py)、[调度、编译与 GEMM 策略](sources/config__vllm.py)、[FlashInfer 能力](sources/v1__attention__backends__flashinfer.py)、[MM cache](sources/config__multimodal.py)、[CPU connector](sources/distributed__kv_transfer__kv_connector__v1__simple_cpu_offload_connector.py)。方向性公开依据包括 [vLLM chunked prefill 调优](https://docs.vllm.ai/en/latest/configuration/optimization/)、[B200 原始部署文件](https://github.com/m4r1k/vllm-1mtps/blob/main/k8s/single-replica-qwen35-27b.yaml)和下列原始研究。

## 可直接用于原生 N=128 启动 profile 的候选命令

这是部署建议文本，本研究没有执行以下命令。网络与 tmpfs 卷使用 INFRA 中现有资源。encoder GPU 按本次用户指定改为 3；embedding 继续用 GPU 4。不改变 embedding 的实验参数。

```bash
docker run --name qwen38-27b-fp8-epd-encoder \
    --network video-sql-v2 \
    --gpus '"device=3"' \
    --shm-size=32g \
    -e HF_HOME=/hf_cache \
    -e VLLM_SERVER_DEV_MODE=1 \
    -v /raid/huggingface:/hf_cache \
    -v video-sql-ec-cache:/epd_shared/cache \
    video-sql-vllm-epd:ec-tp-device \
    Qwen/Qwen3.8-27B-FP8 \
    --host 0.0.0.0 --port 8000 \
    --mm-encoder-only \
    --enforce-eager \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.45 \
    --max-num-batched-tokens 114688 \
    --max-num-seqs 24 \
    --mm-processor-cache-gb 4 \
    --enable-request-id-headers \
    --ec-transfer-config '{"ec_connector":"ECExampleConnector","ec_role":"ec_producer","ec_connector_extra_config":{"shared_storage_path":"/epd_shared/cache"}}'

docker run --name qwen38-27b-fp8-epd-pd \
    --network video-sql-v2 \
    --gpus '"device=0,2"' \
    --shm-size=32g \
    -e HF_HOME=/hf_cache \
    -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    -e VLLM_SERVER_DEV_MODE=1 \
    -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
    -v /raid/huggingface:/hf_cache \
    -v video-sql-ec-cache:/epd_shared/cache \
    video-sql-vllm-epd:ec-tp-device \
    Qwen/Qwen3.8-27B-FP8 \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 314573 \
    --hf-overrides '{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":1.2,"original_max_position_embeddings":262144}}}' \
    --max-num-seqs 128 \
    --max-num-batched-tokens 32768 \
    --gpu-memory-utilization 0.90 \
    --kv-cache-dtype auto \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --mamba-cache-mode align \
    --async-scheduling \
    --attention-backend FLASHINFER \
    --enable-flashinfer-autotune \
    --mm-processor-cache-gb 4 \
    --tensor-parallel-size 2 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
    --kv-transfer-config '{"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use_per_rank":236223201280,"lazy_offload":false}}' \
    --ec-transfer-config '{"ec_connector":"ECExampleConnector","ec_role":"ec_consumer","ec_connector_extra_config":{"shared_storage_path":"/epd_shared/cache"}}' \
    --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true}'
```

不另加 performance-mode=throughput：预算已显式给出，当前实现对显式数值不会再翻倍。服务器 N=128 是用户指定的容量 profile 输入，不是最终并发上限。

### 最终 B、C、N 的填入规则

1. 在上述最终其余参数下，用 N=128 完成原生服务启动，包括模型、activation、非 Torch 和 CUDA graph 的内存 accounting。首次不传 kv-cache-memory-bytes，以免跳过 native memory profiling。
2. 取 TP 各 rank 真实可用 resident KV 预算的最小值 B。使用用户确定的扣除项和该次真实 profile 数据，不把 CPU 440GiB 或 encoder GPU 空闲空间加进 B。
3. 固定 B 后，由本镜像实际 hybrid KV groups/layout 和 L=314573 取得精确 `cache_config.kv_cache_max_concurrency=C`，再令 `N=floor(1.5*C)`。不能拿日志四舍五入的 `%.2fx` 反算 N，也不能用 `num_blocks*block_size/L` 替代 group-aware C。
4. 最终命令只把 `--max-num-seqs 128` 换成该 N，并加 `--kv-cache-memory-bytes B`。该参数在当前 parser 中存在；日志建议文本出现的旧拼写 `--kv-cache-memory` 不应照抄。B、C、N 在本次研究中都没有实测值，不能预先捏造。

当前 [gpu_worker.py](sources/v1__worker__gpu_worker.py) 的手动 KV bytes 分支仍调用 profile_run 以完成编译，但明确跳过 memory profiling；这与“首次先 N=128 原生启动，再固定 B”的顺序有关。CUDA graph 估计是否扣入预算由 `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` 控制，所以命令显式设为 1。启动后须结合实际 graph 捕获内存完成用户所要求的扣除，不能只摘第一条 Available KV cache memory 日志。[容量实现](sources/v1__core__kv_cache_utils.py)把精确 C 写入 cache_config，打印时才保留两位小数。

## 一手 benchmark 对照：哪些数字能说明什么

| 原始来源 | 模型、精度、硬件、软件 | 输入/输出/并发与统计口径 | 原文结果 | 本项目可用信息与不能外推处 |
|---|---|---|---|---|
| Federico Iezzi / m4r1k，2026-03 | Qwen3.5-27B FP8；8×B200，DP8；vLLM 0.18.0；FP8 KV | 随机 ISL=1024、OSL=512，range ratio=.5；最终 manifest L=2560；多客户端，各1000并发。不是本项目128并发 | 单节点95317是 **input+output total tok/s**；文章给 MTP1/util.95=95391，MTP2/util.95=90567，无MTP/util.95=66021 | 支持在 B200 dense FP8 上优先考虑浅 MTP；不支持用95K预测本项目 output tok/s。实际命令是 mtp，不是转载帖里的 ngram。文章叙述有4K/2560不一致及权重大小纠正，原始客户端JSON未公开 |
| cmarsili，2026-08，HF discussion #9 | 精确 Qwen3.8-27B-FP8；RTX PRO 6000 Max-Q 300W，TP1；vLLM .27.1，BF16 KV | 单请求，400生成token；max L262144，实际输入长度未给 | 无MTP46.8、MTP1 56.3、MTP2 62.2、MTP3 59.6 tok/s | 是精确模型、相同 KV dtype 的相反证据：MTP1未必赢。不能从单请求 RTX 结果决定 B200/N128 |
| 同 discussion 的 iphands | Qwen3.8-27B FP8，自定义栈；RTX PRO 6000 600W | 生产/代理图，输入长度与可复现实验未完整给出 | 作者表示MTP3胜1/2，并贴80–120 tok/s片段 | 保留为低强度相反证据；其100% acceptance 被另一参与者质疑与重复/热prompt有关，不能当一般负载 |
| jcartu，2026-05-05，原始FP8研究 | Qwen3.6-27B FP8+MTP3；双RTX PRO 6000 TP2；Repne vs upstream .20.1 | c=1/2/4；ctx目标0/131072；每cell30秒sustained decode，10秒warmup，skip-prefill，N=1 | c4长ctx upstream344.97 tok/s；原始JSON为10461 output tokens/30.324074s，真实ISL128678、每请求OSL2048 | 与 FP8/TP2 有关，但只测试构建差异，不是 batched-token 对照；不能把32758配置当“最优32K”的证据 |
| jcartu，2026-05-11，buffer/graph sweep | Qwen3.6-27B **BF16+DFlash**；双RTX PRO 6000 TP2；Repne dev16530 | batched8192/16384/32768 × graph64/128/256；c1/2/4、目标ctx0/16K/32K/64K/128K；每cell60秒decode；OSL cap2048 | 9配置均值183.48–190.10 tok/s；16384/c256=186.38，32768/c256=190.10，N=1 | 直接反驳“32K必然大幅提速”；只有约2%差且无重复方差。本项目FP8、MTP、128API并发、视频prefill均不同 |

来源：[B200 原始文章](https://medium.com/google-cloud/1-million-tokens-per-second-qwen-3-5-27b-on-gke-with-b200-gpus-161da5c1b592)、[部署 manifest](https://github.com/m4r1k/vllm-1mtps/blob/main/k8s/single-replica-qwen35-27b.yaml)、[真实 benchmark 脚本](https://github.com/m4r1k/vllm-1mtps/blob/main/scripts/parallel-bench.sh)、[精确模型 MTP 讨论](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/discussions/9)、[FP8 研究](https://github.com/jcartu/qwen36-27b-fp8-repne-vs-upstream)、[FP8 原始 JSON](https://github.com/jcartu/qwen36-27b-fp8-repne-vs-upstream/blob/main/upstream-v0.20.1/c4_ctx131072.json)、[buffer/graph 研究](https://github.com/jcartu/qwen-bench-2026-05-dflash-v2-sweep)、[原始 ranking CSV](https://github.com/jcartu/qwen-bench-2026-05-dflash-v2-sweep/blob/main/configs/stage-a/_ranking.csv)。

B200 benchmark 脚本逐客户端求 `Total token throughput` 的和，不是统一起止时间下的全局 token 总数除 wall time；不同客户端持续时间造成的影响未量化。原始 manifest/脚本可以核对设置，无法独立重算95K headline。jcartu 的两个研究实际读了 raw JSON 的输入长度、usage、measurement_seconds 和 request_samples；FP8 文件还读取了 event_log，确认 headline 属 sustained decode，并非端到端完整 batch wall throughput。

## Issues 的相反证据与当前版本核对

**MTP 并非没有成本。** [#53670](https://github.com/vllm-project/vllm/issues/53670)报告双RTX5090、Qwen3.8-27B FP8+FP8 KV、dev926、本地DFlash补丁环境中，32K重复prompt、4轮×8并发、256输出的 spec 路径有大粒度 cache-tail 重算；表中的327→206 tok/s来自 DFlash/tiered-K0 比较，不是 MTP3→MTP1 测量。正文称 MTP 也复现，但不能将37%数字直接移植给本项目。评论还有GB10/NVFP4和Gemma/H100样本，仅用于限定机制，均不是本项目的精度/硬件证据。当前镜像没有 `disable_eagle_block_drop` 配置字段；不推荐应用帖子中关闭保护的私有补丁，也不加动态K0 tier。

**不要从旧标题误判官方FP8。** [#52873](https://github.com/vllm-project/vllm/issues/52873)最初声称过32768后MTP永久零接受，作者后续先撤回固定位置解释，最终明确归因特定 Avuja AutoRound INT4 checkpoint，并表示官方FP8没有collapse，已关闭。它不能成为关闭本项目MTP或换框架的依据。

**不要加 MTP 的 max_model_len 门限。** [#52756](https://github.com/vllm-project/vllm/issues/52756)在精确Qwen3.8-27B-FP8、RTX PRO 6000、vLLM.27.1、FP8 KV、MTP2下，将drafter limit设80000，约115K输入后出现仍draft但0接受的trace；与本项目没有设置drafter cutoff不同。本建议保持没有cutoff，不把整批受限或忽略门限当优化策略。

**autotune 旧 crash 不对应本项目。** [#46083](https://github.com/vllm-project/vllm/issues/46083)是B200、Qwen3.5-35B-A3B **BF16 MoE、DP2+EP** 的FlashInfer MoE autotune故障，末尾维护者说明FI已修复、重测不复现，且移除了CI关闭autotune的参数。dense27B FP8没有该专家MoE路径。这个issue足以反对沿用旧的“B200必须关闭autotune”规则，但不足以证明开启后有特定百分比收益。

**CPU offload 的成功必须看到 load。** [#53498](https://github.com/vllm-project/vllm/issues/53498)包含“store增加而外部hits永久0”的真实trace。当前镜像在 `_process_store_completion` 已把CPU块插入hash map：本研究同时检查了store赋值和completion注册，未把前半段赋值误判为旧bug仍在。eager 路径依然有 finish时最后一整块未flush的TODO；当前也已按group累积和扫描块，不能据旧issue声称仍会漏44%或48%。实际丢失比例只能看本次正式trace。

**换 connector / lazy 不是有证据的免费优化。** 当前 lazy=true 是沿GPU free queue提前复制接近淘汰的cached blocks；切换模式没有“完全相同只是更快”的保证。lazy减少无用D2H的可能收益，要与高负载突发淘汰前来不及复制的代价一起测。现有220GiB/rank足够与否取决于工作集，不能只看主机剩余1.9TiB。[#53868](https://github.com/vllm-project/vllm/issues/53868)给出hybrid TP2、CPU eager offload、重prefix reuse下Running0/Waiting3的活跃但停滞trace；使用的是RTX3090、dev1214和本地补丁，不证明当前镜像有同bug。另一个[#42085](https://github.com/vllm-project/vllm/issues/42085)是DeepSeek-V4-Flash/旧fork，不能直接类比Qwen。当前Simple和OffloadingConnector都声明SupportsHMA；仅凭旧兼容性表更换LMCache没有根据。

**容量不能用attention-only算式。** [#55580](https://github.com/vllm-project/vllm/issues/55580)在Qwen3.8 INT4权重、FP8 KV、2×5060Ti、vLLM.28.0下给出512→256、c32吞吐随profile KV池变化的现象，c8不受影响。它不是本模型FP8/B200性能预测；其价值是说明hybrid状态池和调度约束会改变有效并发，支持使用本镜像精确group-aware C，而非机械计算KV tokens。

## FP8 KV、attention 与缓存的边界

本地官方 checkpoint snapshot `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` 的 config 是 `qwen3_5_text`、64层、4 KV heads、head_dim256、BF16计算类型、SSM float32；权重量化是dynamic E4M3、128×128 blocks。完整weight_map中没有q_scale/k_scale/v_scale或对应Q/K/V output_scale tensor key。FP8权重不能据此推出KV已校准。

当前FlashInfer支持fp8_e4m3 KV，但B200的FP8 KV路径还会把Q转为FP8；这是更多数值变化。[官方KV说明](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/)明确未校准默认scale为1.0，推荐校准数据集。结合本镜像缺少calculate-kv-scales CLI，本报告不提供未经支持的“自动校准flag”，不为吞吐临时量化checkpoint，最终候选保持KV auto/BF16。也不强改GDN状态dtype。

FLASHINFER只是full-attention backend。GDN/linear attention有单独实现；consumer encoder仍使用自己的非因果ViT路径。不能把FlashAttention4的纯attention microbenchmark乘到整个27B吞吐上。也没有证据支持将sm120的`VLLM_USE_DEEP_GEMM=0`、FA2限制、PCIe TP技巧原封不动移植到B200 SM100；此处DeepGEMM保留自动排除是实际镜像针对qwen3_5_text的明确策略。

MM processor cache的4GiB是每API/engine进程复制的host cache预算，并非4GiB GPU encoder容量。[源码](sources/config__multimodal.py)明确0会完全禁用。shared EC tmpfs则保存已算好的encoder tensor，TP当前设备加载修复来自本项目Dockerfile。首个正式多轮视频trace应真实经过encoder、EC write/read和PD consumption，不能只见API200或文件数量就认定复用生效。

## 正式运行中需要回答的有限问题

不另开smoke/preflight或短benchmark；只从用户授权的正式运行和原生启动profile核对：

1. 记录真实startup args及resolved config：模型官方FP8、L/YaRN、MTP1、token budget32768、TP2、BF16 KV、async、FlashInfer、autotune、thinking/preserve_thinking。观察真实GEMM/attention/GDN kernel选择及CUDA graph捕获，不以参数字符串代替实际路径。
2. 保存每rank原生内存扣除项和最终B、精确C、floor(1.5C)。客户端128与running/resident上限分别记录；不把排队请求算成同时resident。
3. 阅读真实多轮视频输入、encoder返回、EC缓存读写及PD消费；检查新一轮reasoning与保留的prior reasoning。由主agent核查应用层trace，本报告不改接口。
4. MTP看每位置draft/accepted counters及真实输出；prefix看cached token数与重算tail；CPU offload同时看store和**实际load/hit**，并核对恢复请求输出，不能只看复制字节增加。
5. 性能结论用正式batch wall time及实际完成输出token；另列prefill量、输出长度、成功样本和真实API并发。32768若使activation上涨、C下降或ITL变差，不能只挑更高prefill tok/s；MTP1若acceptance较低或整批时间变长，也不能用B200外部帖子覆盖本地结果。

## 范围、证据缺口和交付核验

本研究读取了项目AGENTS.md、INFRA.txt、Dockerfile.vllm-epd；没有读取REVIEW.md，没有更改仓库、服务或容器状态，没有启动模型或运行测试。镜像版本字符串 `0.1.dev20073+g8e685d198` 来自主agent前序实查；本研究另从原stopped容器导出相关实际源码核验。公开upstream查该短SHA返回422，不能靠版本号声称它对应某个上游release或包含某PR；兼容性结论以上述实际代码为准。

检索包含精确模型+B200、Qwen3.5/3.6 dense27B FP8近邻、一手MTP梯度、batch-token sweep、attention、async、KV offload及相关GitHub issues。GitHub issues均通过gh CLI读取正文和评论；最初gh旧版GraphQL默认字段失败后改用明确JSON字段，未绕过权限拒绝。发现阶段后用gap matrix追相反证据；没有匹配的性能数据时保留“候选/未知”，未补造提升百分比。没有再继续宽泛搜集相同或更弱证据，因为主要配置已有实现证据，剩余不确定性需要本机正式运行才能解决。

研究产物是Markdown。交付前执行完整文本回读及链接/命令/表格结构检查；未使用视觉渲染，不宣称进行过PDF或页面视觉校验。详细来源记录另存claim-source-ledger.md，供主agent复核。
