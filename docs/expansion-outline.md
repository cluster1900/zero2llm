# 从基础 Transformer 到现代大模型：扩展大纲（待审核）

状态：**仅大纲，未实现新增章节或实验**。2026-10-18，基于主分支 `72c39df`。
任务：[bot-hub #14](https://github.com/cluster1900/bot-hub/issues/14)；对照[原目录快照](https://github.com/cluster1900/bot-hub/blob/main/materials/14/current-toc.md)。
由 Normal Work 审核本大纲后，才按每批 2–3 章撰写。本文不是已经出版的新版目录。

## 设计原则与阅读路线

- 保留生活类比 → 小图解 → 张量形状 → 逐行代码 → 自测题的节奏；每章给出类比失效边界。
- 先跑通小模型，再每次只换一个部件；不把框架安装命令当成算法解释，也不把三首诗的记忆当成通用语言能力。
- CPU 必修主线止于教学级实现；GPU 性能、多卡训练、大模型权重为选修，必须写明显存、版本、耗时和跳过条件。不宣称普通电脑可完整复现 DeepSeek。
- 基础篇 P00–P10；架构与推理篇 P11–P18；训练与应用篇 P19–P24；案例与更新篇 P25–P28。
- P 编号是**规划 ID**，不重命名现有章节，也不生成不存在的链接。当前可读正文仍见 [README](../README.md)。
- 每章以下“实验”均为审核后的交付计划，不是本 PR 的运行结果。新增代码路径、文件名等到对应批次落地再登记。

## 逐章大纲

### P00 阅读指南与实验地图
- 目标：理解本书的基础、实现、系统、前沿四层边界，选择 CPU 或 GPU 路线。
- 小节：环境与张量入门；如何读图与代码；实验记录模板；复现和性能数字的区别。
- 代码 / 实验：现有纯 Python 示例启动；记录设备、版本、种子和输出。
- 前置：Python 变量、循环、函数。
- 旧章关系：改写 `00_preface.md`；保留初学者导读，替换路线图。
- 来源：S01、S14。

### P01 词嵌入与空间向量
- 目标：区分 token ID、one-hot 和可训练 embedding，理解点积与相似度。
- 小节：水果坐标系；点积和余弦；查表与梯度；词向量类比的局限。
- 代码 / 实验：复用 `code/01_vector_similarity.py`，增加零向量与尺度改变的自测。
- 前置：P00。
- 旧章关系：保留 `01_word_embedding.md` 主体，校正“手工坐标就是学到的语义”等过度类比。
- 来源：S01、S14。

### P02 Tokenizer：从字符到 BPE
- 目标：在进入模型前看清文本如何成为序列，不再到 GPT 源码时才突然遇到分词器。
- 小节：字符、字节与子词；BPE 合并；训练与编码；特殊 token、Unicode、chat template；词表与长度成本。
- 代码 / 实验：纯 Python 微型 BPE；中文、英文、emoji 的 encode/decode round-trip；对照锁定版本 HF Tokenizers。
- 前置：P01；字典与字符串。
- 旧章关系：新增；提取旧 `08_baby_transformer.md` 分词器段落作对比，保留 Baby-GPT 字符基线。
- 来源：S02、S14。

### P03 为什么需要注意力
- 目标：理解上下文依赖和并行处理，不把注意力权重直接解释成因果理由。
- 小节：一词多义；RNN 与信息路径；查询相关信息；注意力的适用边界。
- 代码 / 实验：手工画代词的候选依赖图，比较不同上下文的玩具权重。
- 前置：P01–P02。
- 旧章关系：保留 `02_why_attention.md`，补充权重解释的边界。
- 来源：S01。

### P04 QKV、Softmax 与掩码
- 目标：逐步算出注意力，掌握因果约束与数值稳定性。
- 小节：投影；缩放点积；稳定 Softmax；padding / causal mask；训练并行与生成逐步计算。
- 代码 / 实验：复用 `code/02_manual_attention.py`、`code/03_scaled_dot_product.py`；验证改动未来 token 不影响过去输出。
- 前置：P03；矩阵乘法和概率归一化。
- 旧章关系：改写 `03_qkv_self_attention.md`，引入旧第 7 章的掩码细节但不提前堆叠模型。
- 来源：S01。

### P05 多头注意力与张量账本
- 目标：读懂 B、T、H、D 维度，形成后续 GQA / KV 内存分析基础。
- 小节：独立投影与分头；reshape / transpose；拼接与输出投影；参数量和计算量条件。
- 代码 / 实验：复用 `code/04_multi_head_attention.py`；每一步断言形状，单头退化与参考实现对齐。
- 前置：P04。
- 旧章关系：保留 `04_multi_head_attention.md`，补充维度与性能说法的适用条件。
- 来源：S01、S05。

### P06 位置编码：先理解座位号
- 目标：区分可学习绝对位置、正弦位置和相对位置信息，为 RoPE 铺路。
- 小节：置换与顺序；正弦频率；相加；距离性质的误区；训练长度与测试长度。
- 代码 / 实验：复用 `code/05_positional_encoding.py`；画不同频率，检验余弦相似度不是随距离单调下降。
- 前置：P05；三角函数的图形直觉。
- 旧章关系：保留 `05_positional_encoding.md`，RoPE 详细推导后移 P12。
- 来源：S01、S03。

### P07 残差、LayerNorm 与 FFN
- 目标：能独立组装并解释一个基础 Transformer Block。
- 小节：残差的信息路径；均值方差与 epsilon；Pre-LN / Post-LN；FFN 非线性与参数预算。
- 代码 / 实验：复用 `code/06_transformer_block.py`；恒等残差、归一化统计、前后向有限性检查。
- 前置：P05–P06；梯度的直觉。
- 旧章关系：保留 `06_residual_layernorm_ffn.md` 基础主体，RMSNorm / SwiGLU 系统讲解移 P13。
- 来源：S01、S06、S07。

### P08 编码器、解码器与生成循环
- 目标：明确三类 Transformer 架构和 token 生成过程，避免把 CED 与传统翻译模型混为一谈。
- 小节：encoder-only / decoder-only / encoder-decoder；cross-attention；teacher forcing；prefill 和 decode；采样与停止。
- 代码 / 实验：手工追踪输入、标签右移和下一个 token；比较 greedy 与 temperature，验证 EOS / 最大长度停止。
- 前置：P04–P07。
- 旧章关系：改写 `07_encoder_decoder.md`；加 P17 的概念路标但不提前声称实现 CED。
- 来源：S01、S14。

### P09 训练最小闭环：数据、损失与评估
- 目标：先讲清训练循环的每个环节，再呈现完整 Baby-GPT，补上“突然出现源码”的断层。
- 小节：样本切分；输入标签错位；交叉熵；autograd / optimizer；训练验证分离；过拟合和生成检查。
- 代码 / 实验：拆解现有 Baby-GPT 一次 batch 的前向、反向与更新；人为标签泄漏反例；固定种子与重复实验。
- 前置：P02、P07–P08；导数只需直觉。
- 旧章关系：新增，从 `08_baby_transformer.md` 拆出训练基础；不将三首诗随机窗口当成独立泛化集。
- 来源：S14、S18。

### P10 实战：Baby-GPT 基线
- 目标：把前面已讲过的组件拼成可运行模型，保留一个后续可比较的基线。
- 小节：配置；完整数据流；模块逐行对照；训练日志；背诵与泛化边界；保存加载。
- 代码 / 实验：复用 `code/07_baby_gpt.py`；基础输出、loss、因果性、checkpoint round-trip；建立 S0 基准。
- 前置：P01–P09。
- 旧章关系：改写 `08_baby_transformer.md`；保留三首诗例子，去掉此前尚未解释的概念跳跃。
- 来源：S14；仓库现有代码。

### P11 KV Cache：生成时不再重复抄作业
- 目标：掌握缓存的正确性，再讨论速度和显存。
- 小节：prefill / decode 数据流；逐层 K/V 形状；位置偏移；mask；缓存容量、重置和批处理；内存公式。
- 代码 / 实验：S1 给 S0 加 KV cache；eval 模式下完整前向与增量 logits 对齐；T=1、chunked prefill、连续两次请求、长度上限测试；分别记录首 token 与后续 token 时延。
- 前置：P10。
- 旧章关系：新增，扩展旧 `09_modern_llms.md` 的 KV Cache 简介。
- 来源：S08、S14、S15。

### P12 RoPE 与 YaRN：旋转位置和长上下文
- 目标：把位置编码落实到 Q/K 与缓存偏移，理解扩展上下文并非修改一个长度常数。
- 小节：二维旋转；相对相位；旋转维度；缓存 position offset；插值 / 外推；YaRN 与长上下文评估。
- 代码 / 实验：S2 在 S1 中用 RoPE 替换绝对位置；旋转范数、相对位置与缓存等价测试；短序列训练后做长度外推，展示失败而非承诺能力。
- 前置：P06、P11。
- 旧章关系：新增，扩写旧第 9 章 RoPE，衔接旧第 5 章。
- 来源：S03、S04。

### P13 RMSNorm 与 SwiGLU：现代 Block
- 目标：理解现代归一化和门控 FFN 的变化及公平对比方法。
- 小节：RMS 与 LN；epsilon / dtype；门控与激活；中间层宽度；参数量相近的消融。
- 代码 / 实验：S3 在 S2 中逐项替换；参考算式、反向有限性、参数统计和单变量 ablation。
- 前置：P07、P10–P12。
- 旧章关系：新增，扩写旧第 6 / 9 章相关摘要。
- 来源：S06、S07。

### P14 MHA、MQA、GQA 与 MLA
- 目标：区分共享 KV 头与潜变量压缩，不把所有压缩都叫 GQA。
- 小节：KV 头数；分组映射；缓存容量；训练与 checkpoint 转换区别；MLA 的低秩压缩和解耦位置项。
- 代码 / 实验：S4 在 S3 加 GQA；Hq 能被 Hkv 整除、MHA / MQA 边界、缓存对齐测试；独立小矩阵演示 MLA，不声称复现生产 kernel。
- 前置：P05、P11–P13。
- 旧章关系：新增；旧第 4 章只保留 MHA 入门。
- 来源：S05、S09。

### P15 FlashAttention、PagedAttention 与服务调度
- 目标：分清计算 kernel、KV 内存管理和请求调度三个层次。
- 小节：显存带宽与 IO；分块在线 Softmax；精确注意力的浮点误差；分页块表；continuous batching / prefix caching；TTFT、TPOT、吞吐。
- 代码 / 实验：CPU 分块 attention 数值对照与块表模拟；GPU 选修使用受支持 SDPA 后端，检查实际 kernel；固定 prompt/输出长度和并发做显存及延迟比较。
- 前置：P11、P14；内存层级直觉。
- 旧章关系：新增；不把分页或 FlashAttention 当成改变模型语义的新架构。
- 来源：S08、S10、S15。

### P16 滑动窗口、稀疏、混合与线性注意力
- 目标：知道不同机制改变哪些信息路径及复杂度，不只记模型名。
- 小节：局部窗口与感受野；全局加局部；线性递推状态；混合层；质量 / 内存 / 速度权衡；MLA 与稀疏选择的不同维度。
- 代码 / 实验：构建 SWA mask；比较窗口边界；独立线性注意力教学递推与并行表达对齐；固定状态大小下的检索失败示例。
- 前置：P11–P15。
- 旧章关系：新增；不在主线默认替换 attention，以免把多个改动混在一起。
- 来源：S11、S12。

### P17 CED：DeepSeek-V4.1-Flash 的因果编码器—解码器
- 目标：回答 CED 在架构哪里、为什么用于输入密集长上下文，而不是只说“encode/decode 分离”。
- 小节：普通 decoder-only 对照；因果 encoder 到 decoder 的 global KV 路径；局部 SWA 与 replay；与 CSA2 的分工；与经典 seq2seq、服务端 P/D disaggregation 的区别。
- 代码 / 实验：独立教学数据流 / 形状追踪，画 encoder 输出到 decoder global KV 的边；计数全量与简化 prefill 的计算路径；只验证教学模型的因果性，不冒充真实 CED 权重复现或保证官方加速比。
- 前置：P08、P11、P14–P16。
- 旧章关系：新增；旧第 7 章引路，P26 DeepSeek 案例回扣。
- 来源：S12、S13。核验范围和重要限制见下方“CED 证据闸门”。

### P18 MoE：专家、路由与负载均衡
- 目标：解释总参数、激活参数与实际推理成本为何不同。
- 小节：top-k 路由；共享 / 细粒度专家；辅助损失和偏置式均衡；token dispatch；容量与丢弃；通信成本。
- 代码 / 实验：S5 从 S4 的 FFN 升级教学 MoE；路由权重、token 计数、梯度、空专家和全部 token 集中测试；打印负载直方图，不宣称单机 MoE 必然更快。
- 前置：P13–P14；P15 性能指标。
- 旧章关系：新增，扩写旧第 9 章 MoE。
- 来源：S09、S19。

### P19 预训练：从三首诗到数据流水线
- 目标：理解数据质量、计算预算和评估决定什么。
- 小节：许可与隐私；清洗去重；文档边界与 packing；训练 / 验证 / 测试；混合比例；学习率与混合精度；scaling laws 的条件；污染检测。
- 代码 / 实验：用小型公开可用或合成数据做去重、边界 mask、恢复训练；记录 tokens、loss 与数据版本；不外传私人数据。
- 前置：P09–P10、P13。
- 旧章关系：新增，扩展旧第 9 章尺度定律；不把“涌现”写成统一物理定律。
- 来源：S18、S09、S14。

### P20 训练框架：Transformers、DeepSpeed 与 Megatron
- 目标：理解从手写循环到框架、多卡的对应关系，而非复制大型训练配置。
- 小节：config / model / tokenizer / Trainer；DDP；ZeRO 状态切分；tensor / pipeline / expert parallel；checkpoint 与版本；框架选择边界。
- 代码 / 实验：小模型 Transformers 前向和训练 smoke；CPU 模拟分片内存账本；DeepSpeed / Megatron 的多 GPU 实验为选修，保存命令、配置和硬件条件。
- 前置：P19；P18 对 expert parallel 可选。
- 旧章关系：新增。
- 来源：S14、S20、S21。

### P21 SFT、LoRA 与对话数据
- 目标：区分预训练与指令微调，准确计算 assistant-only loss。
- 小节：对话模板；角色和特殊 token；mask；全量 / LoRA / QLoRA；数据质量与评估；训练推理模板一致性。
- 代码 / 实验：小型合成指令集 SFT；标签 mask 断言；LoRA 可训练参数统计；量化微调细节与 P24 交叉引用。
- 前置：P02、P19–P20。
- 旧章关系：新增。
- 来源：S14、S17、S22。

### P22 RLHF、DPO 与 GRPO
- 目标：区分偏好优化、奖励模型、策略学习和可验证奖励。
- 小节：SFT 起点；reward model / PPO 的 RLHF 流程；DPO 偏好对；GRPO 组内相对优势；KL 与采样；奖励投机和评估。
- 代码 / 实验：手算一个 DPO batch、一个 GRPO group（含奖励全相同的稳定性）；TRL 小模型配置；验证 reward / prompt 对齐与更新前后 KL，训练性能结论需实测。
- 前置：P21；对数概率和期望的补充小节。
- 旧章关系：新增。
- 来源：S17、S23、S24。

### P23 推理模型、MTP 与投机解码
- 目标：区分训练目标、多步预测、草稿验证及测试时计算，不把“多说话”当成更会推理。
- 小节：R1 式推理训练路线；预算与正确率；MTP 辅助目标；draft / verify；接受率；确定性与随机采样验证；蒸馏。
- 代码 / 实验：可验证算术题预算曲线；S5 旁路增加教学 MTP head（S6，可关）；演示贪心 draft/verify 与基线一致；随机采样必须实现正确接受校正才声称分布保持。
- 前置：P18、P22、P11。
- 旧章关系：新增。
- 来源：S09、S25、S12（仅用于比较 V4.1 的 DSpark 与 V3 的 MTP，不混写训练方案）。

### P24 量化与推理框架实战
- 目标：能依据硬件、模型格式和延迟需求选择工具，并理解量化精度损失。
- 小节：权重 / 激活 / KV 量化；scale、zero-point、group；校准；GGUF；vLLM、SGLang、llama.cpp、TensorRT-LLM 的定位与兼容性；API 服务、流式输出和边界处理。
- 代码 / 实验：CPU int8 量化 / 反量化误差实验；小模型 llama.cpp 本地路线；GPU 选修比较其余三种框架；同权重与 tokenizer、相同量化和采样条件测质量 / TTFT / TPOT / 显存，缺少硬件则明确未测。
- 前置：P15、P20–P23。
- 旧章关系：新增；不保证任一框架支持任一最新模型。
- 来源：S15、S16、S26、S27、S22。

### P25 Qwen3 案例：读配置而不是背参数表
- 目标：把一个明确版本的开源模型映射到全书机制。
- 小节：指定 dense / MoE checkpoint；tokenizer 与模板；RoPE / GQA / RMSNorm / SwiGLU；专家路由；thinking / non-thinking 与预算；参数和版本口径。
- 代码 / 实验：固定模型 revision，读 config 并生成组件对照表；可选小型号推理；用已有章节测试检查形状与缓存，不下载大型权重作为必修。
- 前置：P02、P12–P14、P18、P21–P24。
- 旧章关系：新增，替换旧第 9 章笼统架构对照表的一部分。
- 来源：S28；具体 checkpoint 在撰写时锁定，不能将 Qwen3 全系列当成同一架构。

### P26 DeepSeek 案例：V3、R1 与 V4.1-Flash
- 目标：把架构、训练方法与服务优化分开，建立可追溯的跨版本地图。
- 小节：V3 → MLA / DeepSeekMoE / MTP；R1 → 后训练 / 推理与蒸馏；V4.1-Flash → CED / CSA2 / SWA / KV 压缩；多模态入口；mHC / Engram / DSpark 阅读路标；版本限定。
- 代码 / 实验：对官方配置和论文生成“组件—章节—证据”矩阵；小尺寸形状实验；只报告已运行的测试，不把官方 benchmark 当成本书复现结果。
- 前置：P14–P18、P19–P24。
- 旧章关系：新增，改写旧 `09_modern_llms.md` 案例材料；该旧章的概览职责转到 P00 / 案例路线图。
- 来源：S09、S12、S13、S25；明确 V4.1-Flash backbone 预训练省略 MTP、另训 DSpark，不能沿用 V3 的标签。

### P27 综合实验：给自己的 miniGPT 做体检
- 目标：把逐步升级串起来，能判断改动正确性与性能影响。
- 小节：S0–S6 对照；配置与 checkpoint 兼容；消融；错误定位；模型卡；哪些结论不能外推。
- 代码 / 实验：一条 CPU smoke 命令覆盖基线、KV cache、RoPE、GQA、MoE；单变量报告；GPU 基准单独记录；坏 mask、错位置、缓存污染的负例测试。
- 前置：P10–P18、P23；训练扩展选读 P19–P22。
- 旧章关系：新增；保留 `code/07_baby_gpt.py` 作为历史基准，不把所有开关塞进初学者第一份源码。
- 来源：上述组件原始论文与官方实现；实测报告由后续 PR 提供。

### P28 附录：术语、数学与前沿跟踪
- 目标：提供可维护、可纠错的学习参考，而非追逐未核实消息。
- 小节：术语中英对照；张量 / 概率 / 梯度速查；硬件指南；论文阅读；事实状态（已核实 / 作者报告 / 本书复现 / 待核实）；更新日志与失效链接处理。
- 代码 / 实验：执行章节链接、公式、图表、示例测试；拿一条新模型新闻走完“官方来源 → 固定版本 → 差异审查 → 实验 → PR”流程。
- 前置：P00；按需查询，其余章节非强制前置。
- 旧章关系：改写 `10_appendix.md`，保留词典、公式卡与书单，新增持续更新流程。
- 来源：本文件来源索引与仓库构建工具。

## miniGPT 代码主线与验收边界

| 阶段 | 对应章 | 只引入的主要变化 | 必须验证 |
| --- | --- | --- | --- |
| S0 | P09–P10 | 当前 Baby-GPT 基线与可测试训练接口 | 因果性、loss 有限、保存加载、数据边界 |
| S1 | P11 | KV cache | 同权重完整 / 增量 logits；缓存隔离；边界长度 |
| S2 | P12 | RoPE | 位置偏移；chunked prefill；旋转数学性质 |
| S3 | P13 | RMSNorm / SwiGLU，逐项消融 | 参考算式、梯度、参数预算 |
| S4 | P14 | GQA，MHA/MQA 是边界配置 | 头数约束；KV 内存；缓存数值对齐 |
| S5 | P18 | 教学 MoE | 路由 / 计数 / 梯度；空专家；负载记录 |
| S6 | P23 / P27 | 可选 MTP / draft-verify 教学支线 | loss 对齐；贪心验证；不假称生产 DSpark |

后续拟以独立阶段脚本或清晰模块组织；共享数据、配置和测试，避免每章复制一份无法同步的大脚本。所有阶段先在 CPU、小张量和固定种子下验证；容差须随 dtype 记录。不同阶段改变模型结构时须重新训练或明确权重转换，不能要求新旧模型语义等价。只有同配置、同权重的缓存 / 非缓存和数学等价实现做数值对齐。

MLA、分页、线性注意力和 CED 先做独立可读实验，不强行加入每个主线阶段。性能结果记录设备、软件版本、序列长度、batch、warm-up、同步方法、重复次数和误差范围；CPU 模拟只证明机制，不证明 GPU 加速。

## CED 证据闸门（已核实核心，非凭描述推断）

2026-10-18 已直接读取 DeepSeek 官方模型卡、官方模型仓库 inference README 和 DeepSeek-AI 的版本化技术报告。**CED 确有官方来源**，具体对象为 **DeepSeek-V4.1-Flash**，不泛化为全部“V4.1”。

- 架构位置：语言 backbone 的因果 encoder / decoder 分段，以及 decoder global KV 的来源。不是 tokenizer 的 encode/decode，也不是把两个推理服务放在不同机器上。[S12 §2.2；S13]
- 官方模型卡说明 40 层分成 20+20 层；decoder 的 global KV 从 encoder 最终 hidden states 投影。局部 SWA 另有状态，不能写成 decoder 不需要处理任何 prompt。[S13]
- 报告的 SWA Bounded Replay 是近似重建而非完整等价重算。CED、CSA2 跨层缓存共享和量化各自解决不同问题，不把所有显存收益归给 CED。[S12 §2.2、§2.3、§3.2.2]
- 官方 inference README 说明它是可读参考实现，generation 为普通自回归采样；其小模型 self-test 使用未初始化权重，仅检查形状 / kernel plumbing，不能当成数值正确性或模型质量验证。[S13]
- 本次未运行官方大模型、未复现速度 / 质量指标。正文撰写时固定模型 revision 与配置；缺乏证据的 checkpoint 支持、端到端收益或兼容性一律标“待核实”，不能由博客或模型名推断。

## 旧章节处置与发布一致性

| 现有文件 | 规划落点 | 处理方式 |
| --- | --- | --- |
| `00_preface.md` | P00 | 改写路线图，保留导读 |
| `01_word_embedding.md` | P01 | 保留并审校 |
| `02_why_attention.md` | P03 | 保留并审校 |
| `03_qkv_self_attention.md` | P04 | 改写，补掩码与数值边界 |
| `04_multi_head_attention.md` | P05 | 保留并补张量账本 |
| `05_positional_encoding.md` | P06 / P12 | 保留基础，现代扩展新增 |
| `06_residual_layernorm_ffn.md` | P07 / P13 | 保留基础，现代扩展新增 |
| `07_encoder_decoder.md` | P08 / P17 | 改写基础，CED 新增 |
| `08_baby_transformer.md` | P09 / P10 | 拆解训练基础后改写实战 |
| `09_modern_llms.md` | P11–P26 | 分主题扩写；不保留“摘要替代整章”的结构 |
| `10_appendix.md` | P28 | 保留词典公式，新增更新流程 |

原 #11 审校结果保留，后续对相关章节逐项复用与重测；本 PR 不混入审校补丁，不改变历史正文、代码或 `dist/`。
大纲放在 `docs/` 而非 `chapters/`，不会被当前 build.py 误编译成正式章节。README 保留可读的旧目录，新增独立的“规划中”入口；正文、源码、README、目录、HTML/EPUB 和实验记录在每批实现 PR 中一起更新。未存在的文件只写计划名称，不创建虚假链接。

## 分批实施与审核停点

1. 本 PR：审核主题覆盖、难度、CED 事实和章节顺序；**不写正文，等待 Normal Work 审核**。
2. 审核后每批 2–3 章，建议依序：P00–P02、P03–P05、P06–P08、P09–P10、P11–P13、P14–P16、P17–P18、P19–P20、P21–P23、P24–P26、P27–P28。
3. 每批单独分支和 PR；交付正文、图解、源码、测试和 README 进度更新。现有章号在迁移方案审核前不变；真正重编号时修复全部内部链接并保留旧入口或明确迁移表。
4. CPU 核心实验须可运行；GPU / 多卡实验要么实际验证，要么标明未测及硬件条件。跑书籍构建、资源 / 链接校验；图解改变时补布局检查；失败不得写成“通过”。
5. 对尚无实测支持的性能宣传、模型私有架构、未经核验的新版本不作事实陈述。不直接 push main、不合并、不触发发布替换。

### 请 Normal Work 审核

- 29 个规划单元（P00–P28）是否适合本书体量；基础 11 单元和进阶选修的边界是否清楚？
- CED 放在 KV、变体、SWA 之后，并在 DeepSeek 案例回扣，是否认可？
- miniGPT 单一主线加独立机制实验是否认可；是否需要调整框架实战的选修比例？
- 认可后按上述 2–3 章批次推进；若调整，先更新本大纲与 README，再开始正文。

## 官方 / 原始资料索引

核验日 2026-10-18。下列为章节写作的依据和入口，不是已跑通的软件版本清单；动态文档在实现时锁定版本。核心新事实优先版本化论文与官方模型仓库，不采用第三方 issue、未合并 PR 或营销摘要作为架构证据。

- S01：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)，基础 attention、位置编码、encoder-decoder。
- S02：[HF Tokenizers](https://huggingface.co/docs/tokenizers/index)，分词流水线与官方实现入口。
- S03：[RoFormer / RoPE](https://arxiv.org/abs/2104.09864)，旋转位置编码。
- S04：[YaRN](https://arxiv.org/abs/2309.00071)，上下文扩展。
- S05：[GQA](https://arxiv.org/abs/2305.13245)，MHA/MQA/GQA 对照。
- S06：[RMSNorm](https://arxiv.org/abs/1910.07467)。
- S07：[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)，SwiGLU。
- S08：[PagedAttention](https://arxiv.org/abs/2309.06180)，缓存分页与服务内存管理。
- S09：[DeepSeek-V3 官方仓库](https://github.com/deepseek-ai/DeepSeek-V3)，MLA、MoE、MTP 及其技术报告入口。
- S10：[FlashAttention 官方仓库](https://github.com/Dao-AILab/flash-attention)，精确 attention 的 IO 优化及原始论文入口。
- S11：[Flash Linear Attention 官方实现](https://github.com/fla-org/flash-linear-attention)，线性 / 混合注意力实现与各算法原始论文入口。
- S12：[DeepSeek-V4.1-Flash 技术报告 v1](https://arxiv.org/html/2609.19969v1)，重点 §2.2、§2.3、§3.2.2。
- S13：[DeepSeek-V4.1-Flash 官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)；[官方 inference README](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/README.md)。
- S14：[Transformers 官方文档](https://huggingface.co/docs/transformers/index)。
- S15：[vLLM 官方文档](https://docs.vllm.ai/en/latest/)。
- S16：[SGLang 官方文档](https://docs.sglang.io/)。
- S17：[TRL 官方文档](https://huggingface.co/docs/trl/index)，SFT / DPO / GRPO 等训练入口。
- S18：[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)，计算预算与数据规模。
- S19：[DeepSeekMoE](https://arxiv.org/abs/2401.06066)，专家划分和共享专家。
- S20：[DeepSpeed 官方站](https://www.deepspeed.ai/)，ZeRO 与训练实现入口。
- S21：[Megatron-LM 官方仓库](https://github.com/NVIDIA/Megatron-LM)，并行训练实现。
- S22：[QLoRA](https://arxiv.org/abs/2305.14314)，量化微调。
- S23：[DPO](https://arxiv.org/abs/2305.18290)。
- S24：[DeepSeekMath](https://arxiv.org/abs/2402.03300)，GRPO 原始方案。
- S25：[DeepSeek-R1 官方仓库](https://github.com/deepseek-ai/DeepSeek-R1)，推理模型、训练路线和蒸馏。
- S26：[llama.cpp 官方仓库](https://github.com/ggml-org/llama.cpp)。
- S27：[TensorRT-LLM 官方文档](https://nvidia.github.io/TensorRT-LLM/)。
- S28：[Qwen3 官方介绍](https://qwenlm.github.io/blog/qwen3/)，模型族、思考模式和官方模型入口。
