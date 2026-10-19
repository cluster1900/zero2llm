# 模型案例（P26）：DeepSeek 的跨版本证据地图 {#p26-deepseek}

[]{#第9章走向未来从经典-transformer-到-gpt-4deepseek}

[]{#第9章-走向未来-从经典-Transformer-到-GPT-4-DeepSeek}

> 本章不是“在普通电脑上复现 DeepSeek”，而是练习更重要的能力：面对一个真实模型，分清**主干架构、训练方法、推理系统和证据来源**。先读 [P14 的 MLA](#p14-grouped-kv)、[P17 的 CED](#p17-ced)、[P18 的 MoE](#p18-moe) 与 [P23 的 MTP](#p23-mtp)。本章 CPU 实验只用 Python 标准库，不下载权重、不运行远程代码、不需要 GPU。

## P26.1 先锁定对象，再讨论“DeepSeek 用了什么”

“DeepSeek 使用 MoE”“R1 使用强化学习”“V4.1 使用 MTP”看起来像同一种句子，实际在说不同层面的事情。前一句可能描述计算图，中间一句描述权重怎样训练，后一句甚至可能把配置字段误读成训练事实。

我们把一次可靠阅读分成四问：

1. **哪个对象？** 精确到仓库、模型名和不可变 revision，而不只是品牌或系列。
2. **哪类证据？** 配置给出尺寸；源码给出数据流；论文给出训练流程与实验；模型卡概括发布范围。它们不能相互代替。
3. **本书做了什么？** 提取字段、运行小矩阵、验证坐标，还是跑了真实权重？只有实际运行的部分才能叫本书实验结果。
4. **哪里仍未知？** 缺字段不等于功能不存在，参考实现也不等于线上部署实现。

本章核查日期为 **2026-10-19**。以下是官方模型 API 当日返回、随后实际读取配置的固定提交，不是 `main` 别名；后来发布的新模型不自动继承本章结论。

| 本章代号 | 官方仓库 | 固定 revision |
| --- | --- | --- |
| V3 | deepseek-ai/DeepSeek-V3 | e815299b0bcbac849fa540c768ef21845365c9eb |
| R1 | deepseek-ai/DeepSeek-R1 | 56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad |
| 学生 | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562 |
| V4.1-Flash | deepseek-ai/DeepSeek-V4.1-Flash | 314113f8a7dadb388215396ba5fbee2c610a04d6 |

对应报告分别固定为 [V3：2412.19437v2][V3-paper]、[原始 R1：2501.12948v1][R1-paper]、[V4.1-Flash：2609.19969v1][V41-paper]。这里没有把 R1 后续变体、其他 V4 型号或 API 产品自动算作同一个模型。

[本章脚本](../code/27_deepseek_case.py) 配套五份原始 JSON 配置快照和一份[来源清单](../code/data/deepseek/manifest.json)。清单记录完整 URL、revision、日期和 SHA-256。运行时先检查配置字节，再生成矩阵，全程离线。**哈希只证明当前文件与记录的快照一致，不证明论文正确，更不证明程序刚刚联网复核过。** 论文行是有出处的人工核对记录，程序不会假装自己从 JSON 推导出了训练历史。

![DeepSeek 三条证据路线：V3 把 MLA、MoE 与顺序 MTP 分开；R1 区分后训练教师与异构蒸馏学生；V4.1-Flash 将 CED、CSA2、SWA 和 DSpark 分别对应架构、缓存、局部状态与草稿。底部说明配置、论文和本书实验不能互相冒充。](assets/diagrams/p26-deepseek.svg)

## P26.2 V3：一份参数表里有三种不同问题

### MLA 关心“历史状态存什么”

[V3 报告 §2.1.1][V3-paper] 的 MLA 将内容 K、V 共同关联到低维 latent，另保留解耦的位置键。[固定配置][V3-config] 对应 `kv_lora_rank=512`、`qk_rope_head_dim=64`、`num_attention_heads=128`。这三个量分别是潜变量宽度、位置分支宽度和查询头数，不能把 512 当成头数，也不能因为 `num_key_value_heads=128` 就断定实现缓存了 128 份完整 K、V。

对于这个配置，按“每 token、每层、缓存内容 latent 加共享位置键”的逻辑口径，元素数为：

$$
512+64=576.
$$

若假设都用两字节存储，payload 是 1,152 字节。但这是**特定缓存表示的元素账**，不是整个请求的显存：层数、batch、token 数、分配器、量化 scale、工作区与权重尚未计入；具体后端也可能采用不同布局。先说清“谁的哪一块缓存”，再谈节省多少。[P14](#p14-grouped-kv) 解释为什么线性上投影可以吸收到查询或输出侧，本章稍后用三行数据复核它。

### DeepSeekMoE 关心“这次执行哪些 FFN”

[]{#杀手锏-4混合专家架构moedeepseek-的低成本核武器}

[V3 配置][V3-config] 明确：主干 61 层，前 3 层 FFN 为 dense；其后 MoE 每层 256 个路由专家，选择 8 个，并有 1 个共享专家。**不是每一层都有 256 个专家。** 报告中的 671B 总参数与每 token 37B 激活参数是官方模型口径，不是把 `8/256` 直接乘总参数就能得到的数字，也不是本书测得的运行成本。[V3 报告 §4.2][V3-paper]

共享专家像每次都要经过的公共工序，路由专家是当前 token 选择的几条额外支路；它们不是预先贴好“数学专家”“历史专家”的知识库。路由分工由训练形成。

V3 的路由还有一个容易混淆的区别：**纠偏 bias 参与选择专家，选中后的混合权重仍来自原始 affinity。** 配置的 `scoring_func=sigmoid`、`norm_topk_prob=true`、`routed_scaling_factor=2.5` 要分开解释；“auxiliary-loss-free”也不能扩写成“整个训练完全没有任何辅助损失”，报告还保留 sequence-wise 项。[V3 报告 §2.1.2][V3-paper]

本章只做选择与权重的手算，不实现实际的分组/节点受限路由、负载更新、通信、专家网络或辅助目标。完整教学路由和负载实验见 [P18](#p18-moe)。尤其不要将 V3 的 sigmoid 公式照搬到 V4.1-Flash：后者固定配置写的是 `sqrtsoftplus`、384 个路由专家、选择 6 个。[V4.1 固定配置][V41-config]

### MTP 关心“训练时多监督哪个未来位置”

V3 的 MTP 是**顺序模块**，不是在同一 hidden state 上并排挂几个互不相干的线性分类头。后一个预测深度接收前一个深度表示和相应 token 的 embedding；embedding 与输出头共享。报告把额外预测深度设为 1，与固定配置 `num_nextn_predict_layers=1` 对应。[V3 报告 §2.2、§4.2][V3-paper]

因此，主任务预测下一词，额外深度再监督更远一个词。训练时可以提供移位后的真实 token，部署时却不能偷看尚未生成的答案。若用作草稿，仍要由目标模型按投机解码规则验证。**MTP 训练、多 token 草稿、最终接受几个 token，是三个不同问题。** [P23](#p23-mtp) 的独立线性未来头用于学习标签对齐，不冒充 V3 的顺序 MTP。

## P26.3 R1：改变训练路线，不等于发明另一种 Attention

[R1 固定配置][R1-config] 的 `model_type` 仍是 `deepseek_v3`，61 层与 `kv_lora_rank=512` 等字段也能直接读取。但仅靠这些字段，既不能证明模型经过 RL，也不能证明它擅长推理。要回答训练问题，必须读 [R1 原始报告 §2.2–2.4][R1-paper]。

| 对象 | 这份报告描述的训练路线 | 不能偷换成什么 |
| --- | --- | --- |
| R1-Zero | 从基础模型开始 RL，不先做冷启动 SFT | 不能据此说所有 R1 都不需要 SFT |
| R1 | 冷启动 SFT → 推理导向 RL → 拒绝采样与新数据 SFT → 综合任务 RL | 不是简单“在 prompt 里加一步步思考” |
| R1 蒸馏学生 | 用 R1 整理出的示范数据微调 Qwen/Llama 学生；该报告学生阶段只做 SFT | 不是把教师 MoE 权重逐层缩小，也不是报告证明学生做了同样 RL |

这里的蒸馏首先指**通过教师产出的训练样本传递行为**，不必等同于逐 token 匹配教师所有 logits 的 KL 蒸馏。是否使用 logits、怎样过滤样本、是否再做 RL，都要逐条找证据，不能由“distill”这个词补全。

最小反例就是本章锁定的 [R1-Distill-Qwen-1.5B 配置][Student-config]：`model_type=qwen2`，12 个查询头、2 个 KV 头，28 层。它沿用学生的架构形状，**不能因为名字里有 DeepSeek 就贴上 MLA、256 专家或 CED 的标签**。报告说明该学生基座为 Qwen2.5-Math-1.5B；这和教师属于哪一族是两条不同的谱系。[R1 报告 §2.4][R1-paper]

还有一个适合自己核对的陷阱：学生配置同时包含 `sliding_window=4096` 和 `use_sliding_window=false`。只抽取前一个数字就宣传“用了 4K SWA”是错误读法。配置中的数值、开关、模型代码路径需要一起看。

本章没有微调学生，也没有比较教师与学生分数。想实际理解训练信号，接着看 [P21 的 SFT](#p21-sft)、[P22 的偏好优化](#p22-preference) 和 [P23 的推理/投机解码边界](#p23-mtp)。

## P26.4 V4.1-Flash：必须把 CED、CSA2 和 SWA 分开

### CED：global KV 从哪里来

[V4.1-Flash 报告 §2.1–2.2][V41-paper] 与[固定模型卡][V41-card]描述 40 层语言主干：20 层 causal encoder，再接 20 层 decoder。这里 encoder 仍受因果约束，不是把未来文本全部读完的 BERT 式双向编码器。

decoder 的 **global KV 从最终 encoder 表示投影**，不必等待每个 decoder 层自己的历史表示；但各层 **SWA KV 仍来自本层输入**。一个是全局历史的来源，一个是局部路径的深度，不能把“共享 global KV”误说成“所有层只剩同一份状态”。[固定参考源码 `Attention`][V41-model]也保留这两个入口。

这解释了 CED 为什么能为某些 prefill 路径省去大量 decoder prompt 计算，却没有证明普通逐层参考代码已经实现生产调度。官方参考 `Transformer.forward` 仍遍历主干；报告中的 bounded replay 与部署组织是另一个层面的设计。[P17](#p17-ced) 已用独立数值 oracle 演示：有界 replay 是近似路径，多层叠加后不保证与完整历史严格相等。

### CSA2：global KV 和索引分别由谁生产

根据[报告 §2.3][V41-paper]与[固定 inference 配置][V41-inference-config]，三种模式要画两条来源线，而不是画一条“共享注意力”箭头：

| 模式 | main KV / indexer K | 当前 query 的 Top-K 索引 | 本层 main Q 与 SWA KV |
| --- | --- | --- | --- |
| Full | 本层生产 | 本层计算 | 本层计算 |
| Reindex | 复用前面 Full 层的 bank | 用本层 indexer Q 重新选择 | 本层计算 |
| Reuse | 复用同一 bank | 复用对此 bank 最近产生的选择 | 本层计算 |

Full 是 **CSA2 路径名**，不是“不加因果 mask 的 dense attention”。Reindex 是重新选择条目，不是重新算整份 KV。Reuse 也不意味着输出相同，因为本层 query 与局部路径仍会改变。[报告 §2.3.1][V41-paper]

固定配置使用从 0 开始的层号：

- `n_layers=40`，`window_size=128`。
- global KV source 为 `[2,8,14,20]`。
- index source 为 `[2,8,14,20,24,28,32,36]`。
- 前 2 层只有 SWA；第 2–19 层压缩比为 2；第 20–39 层为 1。
- 压缩比数组总长 **43**，尾部 3 项对应额外预测模块，不能因此宣布“主干有 43 层”。

这些值在[外层 `text_config`][V41-config]与[参考 inference 配置][V41-inference-config]中以不同字段名出现；测试逐项对齐它们。本章解析器只遍历前 `n_layers` 项，得到 **2 SWA、4 Full、4 Reindex、30 Reuse**。这是从配置解析的层来源账本，不是执行了 CSA2 attention。

### KV 压缩至少有四本不同的账

请分别记录：每条缓存有多宽、多少 token 合成一条、多少层复用一份、每个数占多少 bit。SWA 的短期状态、indexer K、量化 scale、压缩中间状态和分配器开销还要另算。

[V4.1 报告 §2.4.4][V41-paper]与[固定参考 `_compress_kv`][V41-model]描述 main KV 的 FP4 路径，源码按 16 通道一组使用 E4M3 scale；不能把根配置的权重量化信息直接当作 KV 格式。**权重是 FP8/FP4、KV 是 FP4、文件体积减半、HBM 实占减半，是四种不同主张。** 本章不实现 FP4 编解码，也不从只算数据位宽的账本推断实际服务内存。

## P26.5 同名预测字段，为何不是同一种训练

这是整章最值得保留的一张冲突消解表：

| 证据 | 能直接说什么 | 不能直接说什么 |
| --- | --- | --- |
| V3 配置 `num_nextn_predict_layers=1` | 配置记录一个额外预测层 | 单靠此值还不能证明训练流程 |
| V3 报告 §2.2、§4.2 | 顺序 MTP 与 backbone 联合预训练 | 不能说所有未来版本都照搬 |
| V4.1 配置 `num_nextn_predict_layers=3`；参考字段 `n_mtp_layers=3` | 发布实现存在三个额外预测模块的配置 | 不能推出 backbone 预训练使用 V3 MTP |
| V4.1 报告 §2.1、§2.4.3 | backbone 预训练省略 MTP；其后专门训练 DSpark | 不能把字段名当作反驳训练报告的证据 |

这里没有真正的矛盾：**字段名是软件接口，论文陈述的是训练时序。** DSpark 专门训练阶段冻结 backbone；后训练阶段随 backbone 继续训练，但 DSpark 目标的梯度不反传进 backbone。该报告还区分了半自回归草稿与按置信度安排验证的策略。[V4.1 报告 §2.4.3][V41-paper]

固定配置给出 `dspark_block_size=5`、目标层 `[37,38,39]`；这是模块接口，不是“每次一定接受五个词”的保证。[固定 inference 配置][V41-inference-config] 以及[官方 inference README][V41-inference-readme]还提醒我们：参考实现包含 DSpark forward 路径，但 generation 本身仍是普通自回归采样。**存在一个 forward 函数，不等于生成脚本启用了完整投机调度器。**

## P26.6 多模态入口，以及三张进一步阅读的路标

### 视觉 token 怎样接进文本主干

这次讨论的 V4.1-Flash 有图像入口，但不要把这个能力倒灌给本章早先锁定的 V3 或 R1。官方描述的路线是：图像 → DeepSeek-ViT → 局部特征重排 → MLP projector → 与文本 embedding 一起进入语言主干。[固定模型卡][V41-card]

[固定配置][V41-config]给出视觉宽度 1,024、downsample ratio 3、语言宽度 5,120。于是一个 3×3 邻域原来有九个 1,024 维位置，重排后成为一个 **9,216 维向量**，再由 projector 对齐到语言宽度。重排减少位置数，但在投影前并没有丢掉原有九块的通道数。

尤其要看[参考 `Aligner.forward`][V41-vision]：先把 H×W×C 改成通道在前，向**右侧和底部补零**使边长能被 3 整除，再执行不重叠的 `unfold`。一个新 token 内部的顺序是“通道 → 邻域行 → 邻域列”，不是“邻域位置 → 通道”。最后才经过两层 MLP（中间 GELU）。本章只复现前面的坐标重排，不运行 ViT、projector、视觉位置编码或图像理解。

### mHC、Engram、DSpark 各解决什么

| 阅读路标 | 先问的问题 | 本章核对范围 |
| --- | --- | --- |
| Single-Pass mHC | 多条 residual stream 怎样混合？哪些依赖妨碍融合？ | 报告 §2.4.1；输入混合系数错开一 block 使用，配置 `hc_mult=4` |
| Engram | 哪些信息通过条件查表记忆，而不只靠每次 dense 计算？ | 报告 §2.4.2；tokenizer 压缩、哈希和门控；参考配置层号 `[1,14]` |
| DSpark | 怎样产生草稿，决定验证长度，又保持目标模型负责最终输出？ | 报告 §2.4.3；读训练时序，再区分 forward 与部署调度 |

阅读入口为同一份[版本化报告的 §2.4.1–2.4.3][V41-paper]与[固定源码][V41-model]。不要把三者当作 MLA 的三个别名：一个改 residual stream，一个提供参数化的条件记忆，一个服务于草稿生成。Engram 的内部查表也不能直接称为“联网搜索”或“用户文档 RAG”。

本章只检查 `hc_mult`、`engram_layer_ids`、DSpark 接口字段并建立阅读路线，**未实现这些完整模块，未训练它们，也未验证各自官方收益**。遇到报告之外的新说法，应在矩阵中先记为 unknown，而不是靠熟悉的缩写把空白补满。

## P26.7 自动生成“组件—章节—证据”矩阵

运行 `--matrix-only` 会得到 19 行带出处的完整矩阵，其中 11 行从配置提取，8 行是有版本/位置的人工文献或源码核对。正文压缩展示其中几条，完整字段和来源以脚本输出为准：

| 组件 | 回到本书哪里 | 本章读取的证据 | 输出边界 |
| --- | --- | --- | --- |
| V3 MLA | P11 / P14 | [配置][V3-config]：512、64、128 | 尺寸不是语言质量 |
| V3 MoE | P18 | [配置][V3-config]：dense 前缀 3、256 选 8、共享 1 | 不含真实路由通信 |
| V3 MTP | P23 | [配置][V3-config] + [报告 §2.2][V3-paper] | 层数与训练流程分开记录 |
| R1 后训练 | P21 / P22 / P23 | [报告 §2.3–2.4][R1-paper] | 不由配置字段推测 RL |
| 蒸馏学生 | P14 / P25 | [配置][Student-config]：qwen2、12/2 头 | 不继承教师 MLA 标签 |
| CED / CSA2 / SWA | P16 / P17 | [配置][V41-inference-config] + [报告 §2.2–2.3][V41-paper] | 解析来源，不执行生产 attention |
| DSpark | P23 | [配置][V41-inference-config] + [报告 §2.4.3][V41-paper] | 不推出 backbone MTP=true |
| 视觉重排 | P26 | [Aligner 源码][V41-vision] | 坐标与补零，不是视觉能力测试 |
| FP4 KV | P24 / P26 | [报告 §2.4.4][V41-paper] | payload 账本，不是总 HBM |

`extract_fields` 遇到缺失字段输出 `null`、记录 missing 路径并标记 `unknown`；它不会用零或 false 顶替。真正存在的 `false` 则保留为 false。程序也不会因为一个配置缺少 `n_routed_experts` 就自动断言“模型完全没有条件计算”。

来源清单与程序是一起维护的教学数据，不是防篡改审计系统：同时修改清单和文件可以产生新的自洽哈希。所以测试还锁住本章声明的 revision 和关键配置值；更新版本时应重新查看原始证据，而不是只改 expected 让测试变绿。

## P26.8 四个可以手算的小实验

### 实验 A：MLA 的两条计算路径

将内容 latent 写成行向量，设三行数据、一个教学头：

$$
C=\begin{bmatrix}1&2\\-1&1\\2&0\end{bmatrix},\quad
W_K=\begin{bmatrix}1&2\\-1&1\end{bmatrix},\quad
W_V=\begin{bmatrix}2&0\\1&3\end{bmatrix}.
$$

内容 query 为 `[2,-1]`，位置 query 为 `[1,2]`，三行已经旋转过的位置 key 为 `[[0,1],[1,0],[2,-1]]`。内容与位置宽度各为 2，所以此例分母为 √4=2。

路径一展开 `K=CW_K`、`V=CW_V` 再注意力；路径二把 query 改为 `qW_Kᵀ`，先对 C 做加权平均，最后乘 `W_V`。两个路径的 logits 都应为 `[-2,-1,0]`，输出约 `[2.776358,1.274369]`。

这里检查的是线性代数恒等式与位置分支没有丢失；它省略 V3 的多头、query 低秩投影、归一化、真实 RoPE 生成和训练权重。**小实验正确不等于完整 MLA 内核正确。** 测试除比较两路，还直接用 `exp(-2)、exp(-1)、1` 对手工展开后的值 `[4,6]、[-1,3]、[4,0]` 求加权平均，避免两条路径共享同一个错位后一起通过。

### 实验 B：选择 bias 不该污染混合权重

假设三个专家的 sigmoid 后分数为 `[3/4,1/2,1/4]`，选择 bias 为 `[-1,0,1]`，选两个专家。选择用的分数是 `[-1/4,1/2,5/4]`，所以选中专家 **2、1**。

但混合权重来自原始分数，因此是 `(1/4)/(3/4)=1/3` 与 `(1/2)/(3/4)=2/3`。让专家输出分别为 `[2,0]、[0,4]、[6,2]`，共享输出为 `[1,-1]`，教学 scale 暂取 1：

$$
y=[1,-1]+\tfrac13[6,2]+\tfrac23[0,4]=[3,7/3].
$$

只要选择顺序不变，改变 bias 的大小不应改变这组权重。若把 bias 加进归一化，数值就变了。固定 V3 配置的 routing scale 是 2.5，脚本参数也可取 2.5；上述输出随之变成 `[6,22/3]`，共享分支不跟着缩放。本实验不把这三位专家宣传成真实训练后的专家。

### 实验 C：CSA2 来源账本，不是压缩算法

用官方层调度，但把序列改成 10 tokens、每条 main latent 仅 4 个数、每数假想 4 bits。为便于手算，暂时不计 scale、indexer、SWA、未满压缩组的状态和字节对齐。

- 18 个 encoder global 层的压缩比为 2，每层对应 5 条已完成的 main 记录。
- 20 个 decoder global 层的压缩比为 1，每层对应 10 条。
- 假想“不共享”时共 `18×5+20×10=290` 条。
- 按固定 source 共享时只有三份 ratio=2 bank 和一份 ratio=1 bank，共 `3×5+10=25` 条。

| 教学口径 | bank 数 | 完整记录数 | 仅 main 数据 payload |
| --- | --- | --- | --- |
| 假想逐层独立 | 38 | 290 | 580 bytes |
| 按配置 source 共享 | 4 | 25 | 50 bytes |

这不是官方“890 bytes/token”等指标的复现，也不是同模型两个真实实现的显存比较。脚本没有生成任何压缩向量、学习 compressor 权重、计算 sparse indexer 或调用 GPU kernel。50 与 580 只回答这里明示的四维小账本问题。

因果性还有一个边界：ratio=2 时，原始位置 0 到来后还没有完整组；位置 1 到来后才有第一组。`completed_entries(q,r)=(q+1)//r`，不能向上取整让第一组在位置 0 就可见。组的代表位置或 RoPE 位置，与组内所有 token 是否已经到齐，是两个不同问题。测试枚举各组末位置作为独立 oracle。

### 实验 D：九个视觉位置如何变一个

用 `grid[y][x][c]=100y+10x+c` 构造 6×6×2 特征。按 3×3 重排后是 4 个 token，每个 18 维，72 个数仍然是 72 个数。第一行应为：

```text
[0,10,20,100,110,120,200,210,220,
 1,11,21,101,111,121,201,211,221]
```

如果把通道放到最内层，会得到另一个看起来同样是 `[4,18]` 的张量，却无法与 projector 的输入通道对齐。**形状对，不等于坐标对。** 测试用“由输出 feature 编号反推输入坐标”的独立公式检查每个元素，而不是只检查 shape。

4×5×2 的输入会补成 6×6×2，再输出 4×18；原来 40 个数都应保留，另有 32 个填充零。它不是把 3×3 邻域求平均，不能用平均池化来替换这一步。

## P26.9 运行、可验证结果和失败边界

在仓库根目录运行，不需要安装 PyTorch：

```bash
python code/27_deepseek_case.py
python code/27_deepseek_case.py --matrix-only
python code/27_deepseek_case.py --json
python -m unittest discover -s tests -p 'test_deepseek_case.py' -v
```

默认输出完整证据矩阵和四个实验；`--json` 输出可被程序读取的同一份报告，`--matrix-only` 只打印矩阵。三个模式互斥；从其他工作目录运行也会按脚本位置找到快照。程序不联网、不下载权重，也不执行配置中的 `auto_map`。

本章实际运行的标准库 CPU 结果：

| 检查 | 结果 | 结论上限 |
| --- | --- | --- |
| MLA 展开/吸收输出最大差 | 约 2.22×10⁻¹⁶ | 双精度小矩阵等价；末位允许平台差异 |
| 路由选中与输出 | `[2,1]`；`[3,2.333333333333333]` | 选择偏置与混合权重分离 |
| source 共享 main 账本 | 25 条、50 bytes | 明示假设下的整数账 |
| 视觉重排 | `[6,6,2] → [4,18]` | 元素与坐标检查 |
| 专项测试 | 17 项通过，无依赖跳过 | 只覆盖本章代码与快照契约 |

专项测试包含六个**实际执行的数值错误变体**：漏转置、漏位置分支、bias 污染权重、按每层重复存 bank、提前暴露未完成组、调换视觉通道顺序。它们都能返回数值，由数值/坐标断言抓住；没有把语法错误或程序崩溃伪装成“成功检测”。另执行 SWA 来源与 Markdown 竖线转义两项契约变异；它们分别由精确拒绝原因和输出断言检出。此外还测试坏哈希、浮动 revision、路径越界、缺字段、非法 source 调度、补零、空序列账本和配置别名一致性。

我们没有运行官方权重、训练、GPU 内核、质量 benchmark 或服务吞吐。本书输出不会将官方分数拷贝成“本书实测”。即使官方参考自测能跑通，也只说明它承诺的内容：其 README 明说自测使用未初始化权重，检查形状与 kernel 连接，**不验证真实数值质量**。[官方 inference README][V41-inference-readme]

## P26.10 旧版第 9 章的直觉桥梁：保留有用内容，修正大而化之的结论

[]{#架构选择为什么开放式生成常用-decoder-only}
[]{#三种架构各自擅长什么}
[]{#表-9-1三大-transformer-派系历史战役复盘表}

### 三种架构不是“淘汰赛”

| 结构 | 主要信息流 | 适合用来理解的任务 |
| --- | --- | --- |
| Encoder-only | 允许输入位置双向交互，再读出表示 | 分类、检索、抽取；BERT 是典型阅读入口 |
| Encoder-decoder | 编码输入，decoder 读取编码结果并因果生成 | 翻译、摘要；T5 提供 text-to-text 例子 |
| Decoder-only | 每步只看前文，继续预测下一 token | 开放式续写；本书 Baby-GPT 使用此路线 |

[BERT 原论文 v2](https://arxiv.org/abs/1810.04805v2)与 [T5 原论文 v4](https://arxiv.org/abs/1910.10683v4)给出前两种代表。选择某种结构并不证明另外两种被淘汰。尤其 V4.1-Flash 的 causal encoder，与表中双向 encoder 不是同一个因果设定；不能看到 encoder-decoder 两个单词，就把它与原版翻译 Transformer 的信息流完全等同。

[]{#现代大模型的五种关键改进原理与直觉}

### 五个基础构件仍然有用，但不是每个模型的固定套餐

[]{#杀手锏-1旋转位置编码rope复数平面上的几何旋转}

**RoPE 的手算直觉。** 对二维向量旋转：

$$
R(\theta)\begin{bmatrix}x\\y\end{bmatrix}
=\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}
\begin{bmatrix}x\\y\end{bmatrix}.
$$

座位 m 的 query 旋转 mθ，座位 n 的 key 旋转 nθ；点积可整理成净旋转 `(n−m)θ`，所以相对位置进入分数。例如 `[1,0]` 旋转 π/2 得到 `[0,1]`，但其长度仍为 1。这个几何关系不自动保证任意长上下文都可靠，完整数值与 YaRN 边界见 [P12](#p12-rope)。

[]{#杀手锏-2rmsnorm先不管平均数只控制振幅}

**RMSNorm 先控制尺度，不减均值。** 实际公式要保留 epsilon：

$$
\operatorname{RMSNorm}(x)=\frac{x}{\sqrt{\operatorname{mean}(x^2)+\epsilon}}\odot\gamma.
$$

例如 `[3,4]` 的均方根为 √12.5；令 γ 为 1、忽略小 epsilon，归一化后约 `[0.849,1.131]`，均值不为零。它与 LayerNorm 不是同一个操作，也不能保证固定“快 10%–50%”；速度依赖实现和硬件。[P13](#p13-modern-block) 有带 epsilon 的实现与反例。

[]{#杀手锏-3swiglu-激活函数带有开关的神经元}

**SwiGLU 的两条支路。** 一路准备内容，另一路用 SiLU 调节它：

$$
\operatorname{SwiGLU}(x)=
\bigl(\operatorname{SiLU}(xW_g)\odot xW_u\bigr)W_d.
$$

中间的 ⊙ 是逐元素乘法，不是点积。若某个 gate 输入恰为 0，SiLU(0)=0，该通道的内容支路就被关掉。门控不等于 MoE 路由：前者通常调节 dense 中间通道，后者决定哪些专家支路参与。具体参数量与隐藏宽度要一起算，见 [P13](#p13-modern-block)。

**MoE 的公共工序与选修工序。** 本章 P26.2 和实验 B 保留了“共享专家每次参加、路由专家只选一部分”的直觉，并补上权重归一化手算。不要把总参数、激活参数、显存驻留参数当作同一个数，也不要给专家预先贴知识领域标签。

[]{#杀手锏-5kv-cache-机制消除重复运算的打字机}

**KV Cache 消除重复计算，不消除历史。** 对固定因果模型和未修改的前缀，旧 K、V 不必每个生成步骤重新计算。朴素逐次重算长度 t 的 dense attention，单步约 t²，累积到 N 约 N³；缓存后每次新 query 对历史约 t，累积约 N²。这里仅计算 attention 位置对，不包含投影、FFN、prefill、batch 或内存带宽。缓存既不把全局注意力变成常数，也不能跨权重修改后继续当成有效状态；具体契约见 P11 与 [P15](#p15-attention-systems)。

[]{#顶级大模型架构进化对照表}
[]{#表-9-2从经典-transformer-到-deepseek-的参数与架构进化演变}

### 用可追溯对照代替“从旧到新全都更强”的大表

| 精确对象 | 本章有证据的重点 | 不应外推 |
| --- | --- | --- |
| 固定 V3 | MLA、DeepSeekMoE、顺序 MTP | 不是所有 DeepSeek 的统一模板 |
| 原始 R1 与固定教师配置 | V3 形状主干与特定后训练流程 | 训练路线不能只由 config 推断 |
| 固定 R1-Distill-Qwen-1.5B | Qwen 学生架构、12/2 查询/KV 头 | 学生不继承教师所有构件 |
| 固定 V4.1-Flash | CED、CSA2、局部 SWA、独立训练 DSpark、图像入口 | 不能只按编号认为是“V3 加几层” |

来源分别在本章版本表、组件矩阵和原始配置链接中。旧式“最大上下文”一列也需要拆开：V3 本章固定配置 `max_position_embeddings=163840`，模型卡/报告描述的上下文能力另有口径，**配置允许的索引边界不是已验证质量范围，更不等于某个 API 当前限制**。[V3 配置][V3-config] [V3 模型卡][V3-card]

[]{#尺度定律scaling-law与神奇的涌现现象}

### 尺度与“涌现”：先问测量方法

扩大参数、数据与计算预算，经常是研究训练损失如何变化的一条路线，但性能也取决于数据质量、优化、架构与评测。某项离散任务从“全错”到“答对”，可能显得突然；不能据此断言内部能力也在同一刻跳变，更不能由一张平滑下降曲线推导人类式心智。

对本书读者更实用的问题是：改变了哪个变量？有没有固定训练数据与评测？损失、准确率、延迟、内存是否各自测量？[P19](#p19-pretraining) 和本章的证据矩阵，就是避免把“规模更大”当作万能解释的起点。

## P26.11 小练习与参考答案

1. 删掉 V3 快照中的 `kv_lora_rank` 后直接运行，会先遇到 unknown，还是校验错误？为什么这两种机制都需要？
2. 为什么 V4.1 的 43 项压缩数组只得到 40 行主干调度？第 25 层的 KV 与索引分别来自哪里？
3. ratio=2 时，query 位置 0、1、2 各能看见几个完整压缩组？把向下取整改成向上取整错在哪里？
4. 保持实验 B 的专家排序不变，只把 bias 改成 `[-4,0,2]`。输出应不应该变？若把 routing scale 改成 2.5 呢？
5. 4×5×2 视觉特征以 r=3 重排后形状是多少？增加的零有几个？只检查 shape 为什么不够？
6. 看到 `n_mtp_layers=3`，能否在矩阵里添加 `backbone_mtp=true`？看到学生 `sliding_window=4096`，能否宣布它启用了 SWA？

**参考答案（前五项的数值/契约和第六项的证据边界均有专项测试）：**

1. 先触发 SHA-256 不匹配；它保护离线快照。测试直接把缺字段字典送给 extractor 时才得到 unknown，保护解析语义。两者分别回答“文件是否是那份”和“字段是否已知”。
2. `n_layers=40` 是主干边界，尾部三项另属额外模块。第 25 层是 Reuse：KV 来自 20，最近索引来自 24，本层 SWA 仍是 25。
3. 分别为 0、1、1；向上取整让尚未收齐的组提前进入历史，违反因果可见性。
4. 不变，仍为 `[3,7/3]`；scale=2.5 时为 `[6,22/3]`。纠偏只改选择，scale 只乘路由混合项。
5. 输出 4×18，补 32 个零；交换通道与局部坐标的顺序不会改变 shape，却会改变每个维度的意义。
6. 都不能。训练时序要读报告，而窗口开关在该学生配置中为 false。既不应把缺证据写成 false，也不应把出现一个数字写成 true。

[]{#本章小结-1}

本章完成的不是“复现一个品牌”，而是搭起一条可检查的链：**固定对象 → 找到证据 → 说明边界 → 做小实验 → 用独立 oracle 与错误变体检查**。V3、R1、蒸馏学生和 V4.1-Flash 可以放在同一张地图上，但不能被拼接成一个现实中不存在的超级架构。

## P26.12 原始资料与版本

- [V3 技术报告 2412.19437v2][V3-paper]：§2.1 MLA/MoE，§2.2 顺序 MTP，§4.2 参数设置；[固定配置][V3-config]与[固定模型卡][V3-card]。
- [R1 原始报告 2501.12948v1][R1-paper]：§2.2 R1-Zero，§2.3 四阶段后训练，§2.4 蒸馏；[固定教师配置][R1-config]、[固定学生配置][Student-config]。
- [V4.1-Flash 报告 2609.19969v1][V41-paper]：§2.1–2.4 架构与训练时序、§3.2.2 SWA bounded replay；[固定模型卡][V41-card]。
- V4.1-Flash 同一 revision 下的 [config.json][V41-config]、[inference/config.json][V41-inference-config]、[model.py][V41-model]、[vision.py][V41-vision]、[inference README][V41-inference-readme]。阅读源码，不导入官方模型、不运行未知权重路径。

[V3-paper]: https://arxiv.org/html/2412.19437v2
[R1-paper]: https://arxiv.org/html/2501.12948v1
[V41-paper]: https://arxiv.org/html/2609.19969v1
[V3-config]: https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/e815299b0bcbac849fa540c768ef21845365c9eb/config.json
[V3-card]: https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/README.md
[R1-config]: https://huggingface.co/deepseek-ai/DeepSeek-R1/resolve/56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad/config.json
[Student-config]: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/resolve/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562/config.json
[V41-config]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/314113f8a7dadb388215396ba5fbee2c610a04d6/config.json
[V41-card]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/README.md
[V41-inference-config]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/config.json
[V41-model]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/model.py
[V41-vision]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/vision.py
[V41-inference-readme]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/README.md
