# Claim-to-source ledger（内部复核记录）

访问日期均为2026-09-07。研究主题：官方Qwen3.8-27B-FP8、B200 TP2、N128 API、L314573、encoder分离、CPU KV offload。没有执行本机测试；所有性能数字是来源作者报告，不是本项目实测。以下GitHub issue正文和全部可见评论均由gh CLI取得；其他仓库以gh API读取原文或网页发现后访问，未使用GitHub connector读取issue。

| ID | 支持的claim | 原始作者/发布者、日期、链接 | 读取与限定 | 信心 |
|---|---|---|---|---|
| S01 | B200 Qwen3.5 dense FP8的浅MTP候选；total与output TPS区别 | Federico Iezzi / m4r1k，2026-03，[文章](https://medium.com/google-cloud/1-million-tokens-per-second-qwen-3-5-27b-on-gke-with-b200-gpus-161da5c1b592) | 已读Phase1–4、方法、结果。MTP1/MTP2/noMTP不是本模型或本并发；内文有不严谨解释；原始输出JSON未公开 | 中，不能迁移绝对TPS |
| S02 | 正确方法=mtp、spec1、L2560、budget16384、DP8、FI、.18.0 | m4r1k，[manifest](https://github.com/m4r1k/vllm-1mtps/blob/main/k8s/single-replica-qwen35-27b.yaml) | gh API完整读取；不是转载帖的ngram/4K参数 | 高，限配置事实 |
| S03 | random1024→512，ratio.5，每client1000，sum(client totalTPS) | m4r1k，[脚本](https://github.com/m4r1k/vllm-1mtps/blob/main/scripts/parallel-bench.sh) | 全文读取、实际聚合循环核对；存在缺值取0的作者脚本，不能凭headline独立验证所有client | 高，限统计口径 |
| S04 | 精确Qwen3.8 FP8+BF16 KV在sm120上MTP2胜1/3；另一作者说MTP3胜 | cmarsili / iphands / evilperson068，2026-08，[HF #9](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/discussions/9) | 阅读各参与者正文、表格和日志，Max-Q300W与600W区分；输入长度缺失；单请求不等于总吞吐 | 中/低 |
| S05 | FP8 TP2研究中32758不是被对照出的最优值 | Josh Cartu，2026-05-05，[研究](https://github.com/jcartu/qwen36-27b-fp8-repne-vs-upstream) | gh API完整README；构建对照，N1，c≤4，skip-prefill；两个fork-only flags也不同 | 中 |
| S06 | 344.973TPS的真实分子/分母及ISL128678、OSL2048 | 同作者，[原始JSON](https://github.com/jcartu/qwen36-27b-fp8-repne-vs-upstream/blob/main/upstream-v0.20.1/c4_ctx131072.json) | 阅读metadata、全部event_log和全部results/request_samples；10461/30.324074；存在未完成request，sustained窗口允许；不是完整batch wall指标。未取得生成文本，不据此宣称质量 | 高，限该artifact记录 |
| S07 | 8192/16384/32768 sweep差仅3.6%，不是32K大幅提速证据 | Josh Cartu / Repne，2026-05-11，[研究](https://github.com/jcartu/qwen-bench-2026-05-dflash-v2-sweep) | 全文readme；BF16+DFlash、2×sm120、c1/2/4、目标ctx0–128K、N1，不能外推FP8+MTP+B200+128 | 中，限制充分 |
| S08 | 原始ranking与一cell usage一致 | 同作者，[ranking](https://github.com/jcartu/qwen-bench-2026-05-dflash-v2-sweep/blob/main/configs/stage-a/_ranking.csv)，[JSON](https://github.com/jcartu/qwen-bench-2026-05-dflash-v2-sweep/blob/main/configs/stage-a/b32768_c256_n8/throughput.json) | ranking全文；JSON metadata+首cell完整；ctx0实际ISL77，OSL2048；不声称检查195cell全部generation | 高，限所读内容 |
| S09 | prefix reuse会出现spec-tail重算；不能用DFlash损失当MTP1收益 | Suppressor72等，2026-08-25起，[#53670](https://github.com/vllm-project/vllm/issues/53670) | 正文及评论完整；FP8双5090正文数字为DFlash，后续GB10/NVFP4、Gemma/H100不匹配；评论中的最新开关本镜像不存在 | 中，版本约束 |
| S10 | “过32768 MTP永久零接受”不适用于官方FP8结论 | noonghunna，2026-08，[#52873](https://github.com/vllm-project/vllm/issues/52873) | 含作者两次更正和最后关闭原因；特定INT4 checkpoint问题；官方FP8未collapse | 高，限作者更正 |
| S11 | 不添加drafter max_model_len cutoff | hr168，2026-08-18，[#52756](https://github.com/vllm-project/vllm/issues/52756) | 正文完整；.27.1、sm120、精确FP8、FP8KV，limit80K、输入115K、drafted非0/accepted0；无评论 | 中，当前镜像未实测 |
| S12 | autotune旧B200 crash是BF16 MoE+EP且已关闭 | vLLM contributors，2026-06到08，[#46083](https://github.com/vllm-project/vllm/issues/46083) | 正文相关范围和完整末尾维护者关闭评论；最后指向[#51293](https://github.com/vllm-project/vllm/pull/51293)；未将评论推测当根因实证 | 高，限问题范围/关闭状态 |
| S13 | CPU store需同时核验lookup/load，不能只看写入计数 | wickist等，2026-08-24，[#53498](https://github.com/vllm-project/vllm/issues/53498) | 正文+维护者评论；旧自定义.26.1，本文当前源码已有completion注册；finish TODO仍在但漏44%不能外推 | 高，限当前源码存在注册 |
| S14 | CPU offload可能使健康接口仍200但无实际进展 | wickist，2026-08-26，[#53868](https://github.com/vllm-project/vllm/issues/53868) | 正文完整，.26.1dev1214、3090、DFlash7、本地补丁；只能列正式trace待验证，不声称本镜像复现 | 中 |
| S15 | 旧hybrid offload allocator崩溃不足以证明Qwen要换connector | jasl fork使用者，2026-05-08，[#42085](https://github.com/vllm-project/vllm/issues/42085) | 正文+stale评论；模型DeepSeekV4-Flash、sm120、旧fork | 高，限不可迁移性 |
| S16 | hybrid容量不能用单attention group的token数推并发 | 报告者及wickist，2026-09-06，[#55580](https://github.com/vllm-project/vllm/issues/55580) | 全文及评论；INT4而非FP8权重，5060Ti、.28.0；comment的机制仍是作者解释 | 中，非本机容量数字 |
| S17 | 更大prefill预算可能利TTFT/吞吐、增加decode干扰 | vLLM，[optimization](https://docs.vllm.ai/en/latest/configuration/optimization/) | 完整相关chunked-prefill章节；>8192是泛化推荐，不证明32768优于16384 | 高，限方向性 |
| S18 | FP8 KV默认未校准scale1；不是FP8权重自动带来的属性 | vLLM，[quantized KV cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/) | 完整FP8 overview、scale calibration段；当前latest没有运行时自动calibration建议；本镜像parser亦无旧flag | 高 |
| S19 | 精确模型B200运行记录存在，但不能作为吞吐tuning比较 | numinous-technology，2026-09，[trimtab](https://github.com/numinous-technology/trimtab) | 已读Measured段；是hot knob生效时延/请求丢失控制研究，Qwen3.8FP8、B200、c32；没有可用MTP/budget吞吐对照，不列为推荐证据 | 限定排除 |

## 实际镜像源码来源

来源容器：`qwen38-27b-fp8-epd-pd`，已停止；只读`docker cp … - | tar`同类导出。没有启动、改镜像或安装依赖。以下文件保存在本报告的sources/。版本字符串由主agent前序实查；GitHub upstream SHA 8e685d198查询返回422，未以它虚构公开永久链接。

| 本地文件 | 核验内容 |
|---|---|
| engine__arg_utils.py | B200默认budget16384；performance_mode=throughput仅翻倍未显式指定值；CLI含--kv-cache-memory-bytes；没有calculate-kv-scales |
| config__vllm.py | async允许EAGLE/MTP，默认自动开启；async+spec关闭cascade；O2已有FULL_AND_PIECEWISE/autotune=true；当前DeepGEMM排除逻辑 |
| platforms__cuda.py | SM100因果full-attention优先FLASHINFER；非因果不按同一优先顺序 |
| utils__deep_gemm.py | qwen3_5_text/qwen3_5_moe_text列于Blackwell E8M0排除集合 |
| config__cache.py | KV auto=model dtype；Mamba align规则；kv_cache_max_concurrency存储字段；CPU容量分离 |
| config__multimodal.py | processor cache默认4GiB，每API/engine复制；0完全禁用；不同于encoder缓存 |
| v1__attention__backends__flashinfer.py | 支持BF16/FP8 KV/head_dim256；SM100 FP8KV对应FP8 Q路径；backend名称不能代表全部GDN/ViT路径 |
| model_executor__models__qwen3_5.py | mamba all模式明确NotImplementedError；使用原生state dtype |
| distributed__kv_transfer__kv_connector__v1__simple_cpu_offload_connector.py | SupportsHMA；220GiB/rank参数覆盖；requires prefix caching；异步传输在get_finished驱动 |
| v1__simple_kv_offload__manager.py | eager按group累积已确认块；completion处hash map insert；lazy free queue扫描；finish TODO尚在 |
| distributed__kv_transfer__kv_connector__v1__offloading_connector.py | 也SupportsHMA；不因此推导比Simple更快 |
| v1__worker__gpu_worker.py | 首次原生profile，graph估计扣除开关；显式bytes跳过memory profiling；实际capture结果和后续预算输出 |
| v1__core__kv_cache_utils.py | group-aware capacity及exact C写入；日志仅%.2fx |
| config__speculative.py | MTP配置存在；本镜像没有disable_eagle_block_drop |

本地模型来源：`/raid/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a/{config.json,model.safetensors.index.json}`。解析完整JSON，只输出相关字段和全weight_map中KV scales的匹配列表（空）。未打开模型tensor作精度试验，因此不宣称量化准确率通过。

## Gap matrix与停止理由

| 关键问题 | 当前证据 | 剩余缺口 | 处理 |
|---|---|---|---|
| 32768是否快于16384 | 当前throughput默认翻倍；sm120 BF16 sweep近乎持平 | 无B200 FP8+N128视频对照 | 候选，正式batch结果判定 |
| MTP1是否快于3 | B200近邻正例；精确模型sm120相反例 | 输入/输出长度、acceptance、并发和prefix reuse不匹配 | 候选，不承诺收益 |
| autotune收益 | 默认支持；旧MoE crash已关闭 | 无本模型on/off量化对照 | 开启候选，检查真实kernel和正式产物 |
| FP8 KV准确率 | 无calibration tensor，默认scale1，额外Q量化 | 无当前模型/YaRN/视频准确率证据 | 不采用，保留BF16 KV |
| CPU lazy/LMCache吞吐 | native源码明确模式差异，相关失效trace | 无匹配负载的对照 | 保留当前Simple/eager |
| resident B、C、N | 原生实现可取精确值 | 尚未启动profile | 由主agent按用户已确定流程填入，不估算 |

已完成一次宽发现、定向补缺、相反证据追踪和最高影响源码复核。后续更多泛搜不可能补齐同硬件/模型/负载的缺失实验，停止检索。没有创建目标、自动化、长进程、子agent或新实验协议。最终artifact检查仅为Markdown文本结构/命令内容回读，不运行syntax/smoke/preflight。

原生网页引用记录：B200文章turn15view0；原manifest仓库turn18view0；HF梯度turn15view2；FP8研究turn18view2；buffer sweep turn21view2；KV文档turn22view0；optimization turn22view1。此行仅为内部工具provenance，不在用户报告中展示。
