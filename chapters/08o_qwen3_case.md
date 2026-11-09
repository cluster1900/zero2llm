# 模型案例（P25）：Qwen3，读配置而不是背参数表 {#p25-qwen3}

拿到一张汽车宣传单，你能知道它叫“小型车”，却不能因此确定轮胎尺寸、座位数和油箱容量。模型名里的 `0.6B`、`30B`、`A3B` 也像宣传单上的分类：有用，但不能代替配置、实现与版本。

本章做一次完整的“拆说明书”练习：**选定两个真实 checkpoint，读出它们的组件，算清参数和缓存，再用很小的随机模型验证机制。** 目标不是背诵“Qwen3 都用了什么”，而是能够指出：我说的是哪个仓库、哪次提交、哪个字段，哪些结论还需要看实现。

配套 [code/26_qwen3_case.py](../code/26_qwen3_case.py)，测试为 `tests/test_qwen3_case.py`。默认命令只用 Python 标准库、读取本地小文件；`--torch` 才运行 PyTorch/Jinja2 的额外探针。**本章没有下载或运行任何 Qwen 预训练权重，也没有实现完整 Qwen tokenizer、checkpoint 加载器或生产推理服务。**

先修：[Tokenizer](#a.1-三种切法整词字符与字节)、[RoPE](#p12-rope)、[RMSNorm / SwiGLU](#p13-modern-block)、[GQA](#p14-grouped-kv)、[MoE](#p18-moe)，以及前面关于后训练、推理预算和部署的区分。

## P25.1 先把研究对象钉住

核对日期见附录 H。本章研究的是 2025 年首发路线的两个混合 thinking / non-thinking 文本 checkpoint，**不是**名字中带 `Instruct-2507`、`Thinking-2507` 的后续版本，也不是 Qwen3-Next、视觉模型或整个 Qwen3 家族。

| 对象 | 官方仓库 | 本章锁定的完整 revision |
| --- | --- | --- |
| Dense | `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` |
| MoE | `Qwen/Qwen3-30B-A3B` | `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` |

官方不可变入口：[0.6B 配置](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/config.json)、[30B-A3B 配置](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/config.json)。`main` 是会移动的分支名；完整 revision 才能让几年后的读者看到同一组字节。

仅有配置还不够。例如，`rms_norm_eps` 说明数值稳定项，却不直接告诉你 Q、K 是否另做归一化。我们还核对 Hugging Face Transformers **v4.51.3** 的实现，锁到 commit `5f4ecf2d9f867a1255131d2461d75793c0cf1db2`：[dense 实现](https://github.com/huggingface/transformers/blob/5f4ecf2d9f867a1255131d2461d75793c0cf1db2/src/transformers/models/qwen3/modeling_qwen3.py)、[MoE 实现](https://github.com/huggingface/transformers/blob/5f4ecf2d9f867a1255131d2461d75793c0cf1db2/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py)。这里选择的是可追溯的实现快照，不宣称它是当前最新版，也不要求安装它。

本地 `code/data/qwen3/` 只保存各 checkpoint 的 `config.json`、`tokenizer_config.json`、`generation_config.json` 和原始 Apache-2.0 `LICENSE`，合计 44,317 字节，另有来源清单。每份文件的完整 URL、字节数、SHA-256 都在 `manifest.json` 中。`load_checkpoint` 先核对 revision 和哈希再读取；损坏时直接报错，**不会偷偷改取最新配置**。

这形成三个证据层次：

1. 官方配置回答“这个 checkpoint 声明了什么”。
2. 固定实现回答“这些字段怎样变成运算”。
3. 本章测试回答“我们的小型教学探针实际验证了什么”。

第三层通过，不能替代官方模型质量、训练过程或部署性能的复现。

## P25.2 两列配置，先找不同，再找共同点

下表直接对应上述两个不可变配置，不从模型名猜测数值。

| 字段／含义 | 0.6B dense | 30B-A3B MoE |
| --- | ---: | ---: |
| `model_type` | `qwen3` | `qwen3_moe` |
| 层数 $L$ | 28 | 48 |
| 隐藏宽度 $D$ | 1024 | 2048 |
| Q 头数 $H_q$ | 16 | 32 |
| KV 头数 $H_{kv}$ | 8 | 4 |
| `head_dim`，每头宽度 $d_h$ | 128 | 128 |
| `vocab_size`，embedding 行数 $\lvert\mathcal V\rvert$ | 151936 | 151936 |
| dense FFN `intermediate_size` | 3072 | 6144，当前全 MoE 层不使用它 |
| 单专家 `moe_intermediate_size` | 不适用 | 768 |
| 专家总数／每 token 选中数 | 不适用 | 128 / 8 |
| `tie_word_embeddings` | true | false |
| `rope_theta` | 1000000 | 1000000 |
| `rope_scaling` | null | null |
| `rms_norm_eps` | 0.000001 | 0.000001 |
| `max_position_embeddings` | 40960 | 40960 |
| `use_sliding_window` | false | false |

读表时先做三个检查：

- **不是每个字段都代表当前启用的模块。** MoE 的 `intermediate_size=6144` 留给 dense MLP 路径；本快照所有层都选择专家路径。把 6144 的 dense FFN 再加一份，会把不存在的参数算进去。
- `torch_dtype=bfloat16` 是配置声明，不意味着所有运行时张量都只能是 BF16。我们的数值探针主动使用 CPU float64。
- `max_window_layers` 的存在不等于滑动窗口已经开启。两个对象均关闭 `use_sliding_window`，不能给它们贴上“局部注意力模型”的标签。

![两份固定配置如何映射到注意力、FFN 和输出预算](assets/diagrams/p25-qwen3.svg)

## P25.3 一个最容易算错的维度：D 不一定等于 Hq × d

以前的小型教学模型常把 `head_dim` 写成 `D // Hq`。本章两个 checkpoint **显式指定 $d_h=128$**，所以：

其中：$H_q$ 是 Q 头数，$d_h$ 是每头宽度，$D$ 是隐藏宽度；本章的 $H_qd_h$ 不等于 $D$，这正是要强调的地方。


- dense：D=1024，但 Q 投影宽度为 16×128=2048；
- MoE：D=2048，但 Q 投影宽度为 32×128=4096。

这不是表格矛盾，而是 Q 投影可以改变宽度。输入一个 `[B,T,D]` 张量，线性层先投影，再分头；注意力完成后，O 投影把拼接的头映回 D，才能与 residual 相加。

以 dense 为例，按 PyTorch 的 `[输出维度, 输入维度]` 写权重形状：

```text
输入 hidden                  [B,T,1024]
q_proj.weight               [2048,1024]  → Q [B,16,T,128]
k_proj.weight / v_proj      [1024,1024]  → K/V [B,8,T,128]
注意力各头输出               [B,16,T,128]
拼接                         [B,T,2048]
o_proj.weight               [1024,2048]  → 输出 [B,T,1024]
```

因此不能只把 [S4](#p14-grouped-kv) 的模型尺寸改成 1024，就称它为“Qwen3-0.6B 实现”。S4 的旧假设是 $D=H_qd_h$；本章的微型探针另写显式 head dimension，并且仍不提供官方 checkpoint 兼容性承诺。

Q/K 的头数关系遵循 GQA：dense 每组 2 个 Q 头共用 1 个 KV 头，MoE 每组 8 个 Q 头共用 1 个 KV 头。分组按连续顺序，例子 $H_q=4$、$H_{kv}=2$ 的对应关系是 `[0,0,1,1]`，不是轮流的 `[0,1,0,1]`。

## P25.4 走一遍 block：QK-norm 不是又一个 LayerNorm

在本章选定的实现里，一个 block 的主顺序为：

```text
x → RMSNorm → Attention → 加回 x
  → RMSNorm → dense SwiGLU 或稀疏 MoE → 加回上一步结果
```

注意力内部还有一层细节：Q、K 在投影并按头切开后，分别做 **QK-norm**，随后才应用 RoPE；V 不做这两项变换。

RMSNorm 对最后一维计算：

$$
\operatorname{RMSNorm}(z)_k
= \gamma_k\frac{z_k}{\sqrt{\frac{1}{n}\sum_{k'=1}^{n}z_{k'}^2+\epsilon}}.
$$

**大白话：**先算这一行数字的“均方根”当作音量，把整行除以它统一音量，再让每一维乘上自己可学的增益。不减均值，所以比 LayerNorm 少一步。

其中：$z$ 是被归一化的那一行向量，$z_k$ 是它的第 $k$ 维，$k'$ 是求和用的维度下标；$n$ 是最后一维的长度；$\gamma_k$ 是第 $k$ 维的可学习 gain（代码里的 `weight`）；$\epsilon$ 是防止除以零的小常数（配置里的 `rms_norm_eps`）。

对 block 输入，$n=D$；对单个 Q/K 头，$n=d_h$。这里不减均值，也没有 LayerNorm 的偏置向量。Q 和 K 各有一个长度 $d_h$ 的可学习 gain，它们在同类头之间共享，**不是每个头各拥有一份 $H_q\times d_h$ 或 $H_{kv}\times d_h$ 的 gain**。这会直接影响参数统计。

本章探针保留 float64 精度来做标量对照；固定官方实现的 RMSNorm 有转到 float32 求统计量再转回的逻辑。即使数学结构一样，两种数值路径也不能宣称逐位等价，更不代表 BF16 GPU kernel 的误差行为已经复现。

RoPE 也有布局约定。本快照使用 **split-half** 配对：$d_h=4$ 时，维度 0 与 2 配一对，1 与 3 配一对。若记角度为 θ，第一对是：

$$
(z_0,z_2) \mapsto
(z_0\cos\theta-z_2\sin\theta,\ z_2\cos\theta+z_0\sin\theta).
$$

**大白话：**把第 0 维和第 2 维看成平面上一个点的两个坐标，绕原点转 $\theta$ 角；长度不变，只改方向。

其中：$z_0,z_2$ 是一个 Q/K 头向量的第 0、2 维；$\theta$ 是这一对维度在当前位置的旋转角（局部符号），等于位置乘以这一对维度的角频率 $\omega_i$（附录 A.5；$i$ 是频率对的编号）。

另一种实现可能把相邻维度 0/1、2/3 配对。它们都能表达旋转位置机制，但权重布局、旋转方式和 checkpoint 转换必须一致；不能把一种公式塞进另一种布局，再仅凭“都叫 RoPE”认定兼容。

缓存时保存已经做过位置旋转的 K 和原始投影 V。追加 chunk 的位置从旧缓存长度开始，旧 K 不能再旋转一次。

## P25.5 SwiGLU 与专家：激活少，不代表存储少

Dense FFN 与每个 MoE 专家都使用三矩阵的 SwiGLU 结构：

$$
\operatorname{FFN}(x)=
\big[\operatorname{SiLU}(xW_{\mathrm{gate}})\odot(xW_{\mathrm{up}})\big]W_{\mathrm{down}}.
$$

**大白话：**一路（up）准备内容，另一路（gate）过 SiLU 后当“音量旋钮”逐元素调节它，最后用 down 投影压回原宽度。

其中：$x$ 是一个位置的输入行向量，长 $D$；$W_{\mathrm{gate}},W_{\mathrm{up}}\in\mathbb R^{D\times d_{\mathrm{ff}}}$，$W_{\mathrm{down}}\in\mathbb R^{d_{\mathrm{ff}}\times D}$（按全书行向量写法；checkpoint 里按 `[输出,输入]` 存的是它们的转置）；$\odot$ 是逐元素乘法；$d_{\mathrm{ff}}$ 是中间宽度。

宽度为 $d_{\mathrm{ff}}$ 时，三个无偏置矩阵一共 $3Dd_{\mathrm{ff}}$ 个参数。**gate_proj 是 SwiGLU 的门，不是 MoE 的路由器。** MoE 另有一个从 $D$ 到专家数 $N_E$ 的 `mlp.gate`，它决定调用哪些完整专家。

对本章的 30B-A3B，固定配置是：$N_E=128$、$k=8$、每个专家 $d_{\mathrm{ff}}=768$，`decoder_sparse_step=1`、`mlp_only_layers=[]`。因此 48 层均使用 MoE；固定实现中没有额外的 shared expert 分支。不要把 [S5 教学 MoE](#p18-moe) 默认的共享专家设置照搬进来。

单个 token 的路由流程是：

1. 计算 128 个 router logits。
2. 对全部 logits 做 softmax，选取最大的 8 项。
3. 因为 `norm_topk_prob=true`，把这 8 项重新归一化到和为 1。
4. 只计算选中专家，把结果按重新归一化的权重相加。

手算一个更小的例子：4 个专家的 logits 为 `[2,1,-1,0]`，取前 2 个，则选择下标 0、1 的专家；混合权重等于 `[exp(2),exp(1)]` 再归一化，约为 `[0.731059,0.268941]`。不能直接拿全体 softmax 的前两项相加，因为它们的和小于 1。

我们的实现采用“选中 logits 再 softmax”，在正常有限数值下与上述归一化等价；相等 logits 用较小专家 ID 优先，只是为了教学可复现，**不是官方 `topk` 对 ties 的契约**。没有实现路由辅助训练损失、专家容量丢弃或跨 GPU 分发，也不把这种实现称为官方性能复刻。

## P25.6 参数到底有多少：先声明口径，再按矩阵加总

`parameter_inventory` 只生成名字与形状，不分配真正的大矩阵。参数数目由各维乘积得到，再按组件求和。

对这两个对象，一层的注意力投影参数为：

$$
N_{\mathrm{attn}}=2D(H_qd_h)+2D(H_{kv}d_h).
$$

**大白话：**Q 投影把 $D$ 维映到 $H_q$ 个头、每头 $d_h$ 维，O 投影再映回来，所以是两份 $D\times H_qd_h$；K、V 只有 $H_{kv}$ 个头，是两份 $D\times H_{kv}d_h$。

其中：$N_{\mathrm{attn}}$ 是一层注意力投影的参数数；$D$ 是隐藏宽度，$H_q$、$H_{kv}$ 是 Q 头数与 KV 头数，$d_h$ 是每头宽度。本章 $H_qd_h\ne D$，所以不能化简成 $2D^2$ 之类。

前一项对应 Q 与 O，后一项对应 K 与 V。再加 Q/K norm 的 $2d_h$、两个 block RMSNorm 的 $2D$。最终输出前的 RMSNorm 另加 $D$。

Dense 每层再加 $3Dd_{\mathrm{ff}}$。MoE 每层再加 router 的 $DN_E$ 与全部专家的 $N_E\cdot3Dd_{\mathrm{ff}}$。embedding 为 $\lvert\mathcal V\rvert D$；输出 head 也是 $\lvert\mathcal V\rvert D$，但只有未绑定时才新增一份独立参数。

| 组件 | 0.6B dense | 30B-A3B MoE |
| --- | ---: | ---: |
| 输入 embedding | 155,582,464 | 311,164,928 |
| 独立输出 head | 0，和输入绑定 | 311,164,928 |
| 全层注意力投影 | 176,160,768 | 905,969,664 |
| 全层 Q/K norm | 7,168 | 12,288 |
| 全层 block RMSNorm | 57,344 | 196,608 |
| dense FFN | 264,241,152 | 0 |
| 全层 router | 0 | 12,582,912 |
| 全部专家 FFN | 0 | 28,991,029,248 |
| 最终 RMSNorm | 1,024 | 2,048 |
| **唯一架构参数总数** | **596,049,920** | **30,532,122,624** |
| 排除输入 embedding／独立输出 head | 440,467,456 | 29,909,792,768 |

“唯一架构参数”指按上述实现结构、绑定关系去重后的参数元素数，**不是实际权重文件的字节数、序列化键的机械求和、显存实测或优化器状态大小**。本章不读取权重文件，因此也没有核验某个存储格式是否重复保存了绑定矩阵。

还需要单独定义“激活参数”。本章采用可复算的 **active-matrix convention**：完整计入 embedding／输出矩阵、注意力、norm、router；每个 MoE 层只计 k 个完整专家矩阵。按此定义，30B-A3B 为：

$$
30,532,122,624 - 28,991,029,248
+ \frac{8}{128}\times28,991,029,248
= 3,353,032,704.
$$

**大白话：**从总数里先拿掉全部专家，再按“每个 token 只用 128 个专家中的 8 个”的比例把专家加回来，就是这一口径下的激活参数。

其中：30,532,122,624 是唯一架构参数总数，28,991,029,248 是全部专家 FFN 的参数，$8/128=k/N_E$。

这与[该版本模型卡](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/README.md)描述的约 30.5B 总参数、3.3B 激活参数处在相应量级，但我们明确保留自己的完整计算口径，不把近似标签当作严格恒等式。

为何强调“完整矩阵”？输入 embedding 实际查表只访问选中 token 的行；计算次数、内存读取、算术 FLOPs 不等于这一参数计数。一个 batch 中不同 token 还可能走不同专家，选中专家的并集超过 k。**A3B 不等于只需装下 3B 权重，更不意味着它一定比某个 dense 3B 更快。**

## P25.7 KV cache：这次真的按 Hkv 计算

不量化、各层都保存 K/V 时，逻辑 payload 为：

$$
\text{bytes}=2\times L\times B\times H_{kv}\times T\times d_h\times b_{\mathrm{elem}},
$$

**大白话：**和 P24 的 KV 账本是同一个乘法：K、V 两份，每层、每条序列、每个 KV 头、每个 token 存一条长 $d_h$ 的向量。

其中：2 对应 K 与 V；$L$ 是层数，$B$ 是批大小，$H_{kv}$ 是 KV 头数，$T$ 是缓存的 token 数，$d_h$ 是每头宽度，$b_{\mathrm{elem}}$ 是 K/V 张量每个元素的字节数。假设 $B=1$、$T=4096$、BF16 的 $b_{\mathrm{elem}}=2$：

- 0.6B：469,762,048 字节，即 **448 MiB**；
- 30B-A3B：402,653,184 字节，即 **384 MiB**。

这是一个很好的反直觉检查：**总参数更多，不代表每条序列的 KV cache 一定更大。** MoE 对象有更多层，但 KV 头数更少。缓存量主要取决于上述维度，不直接由专家总参数推出来。

如果误把 $H_q$ 代替 $H_{kv}$，dense 会多算 2 倍，MoE 会多算 8 倍。本章注意力在运算时临时展开 KV 头，返回的缓存仍是 $[B,H_{kv},T,d_h]$；这不是 FlashAttention，不声称避免了注意力矩阵或临时展开的内存。

数字只覆盖张量逻辑内容，不包括权重、临时工作区、allocator、页面碎片、通信缓冲、padding、量化 scale 或多请求调度。不要据此直接宣布“某张显卡一定装得下”。

## P25.8 Tokenizer 与模板：模型看到的不是裸问题

本地保存的是 `tokenizer_config.json`，其中含有正式的 Jinja chat template 与 added-token 元数据。**它不是完整 BPE tokenizer**：没有 `vocab.json`、`merges.txt` 或 `tokenizer.json`，所以本章不把中文字符串真正编码成官方 token 序列，也不做 Qwen encode/decode 的 round-trip 宣称。

同样不要把 `vocab_size=151936` 当作“中文字符有 151936 个”，或从几个 added token 推断整套 BPE 词表规模。

`render_prompt` 真正执行保存的官方模板，但把练习范围限制为一个 user 文本消息。对问题 `1+1?`、`add_generation_prompt=True`：

```text
enable_thinking=True 的结尾：
<|im_start|>user
1+1?<|im_end|>
<|im_start|>assistant

False 比 True 多加的后缀（空行有意义）：
<think>

</think>

```

注意 True 的模板**没有预先写入 `<think>`**。False 则把一个空 thinking 块作为 assistant 的前缀放进 prompt；这部分已属于输入，不是后来生成的输出 token。为了显示清楚，上面排版有说明行。

两份快照的 added-token 元数据都给出 `<think>` 为 ID 151667、`</think>` 为 151668；它们的 `special` 标记是 false。带尖括号的保留文本，不一定就被 tokenizer 标成“special”；`skip_special_tokens` 不能代替明确的 reasoning 解析。

另一个容易混淆的字段是 BOS：模型配置含 `bos_token_id=151643`（固定 tokenizer 配置中这个 ID 就是 `<|endoftext|>`），但 tokenizer 配置的 `bos_token` 为 null、`add_bos_token=false`。不能看见前者就手工往每段对话前重复加 BOS。具体输入应遵循所选 tokenizer 与模板的处理规则。[固定 tokenizer 配置](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer_config.json)

完整模板还包含历史 assistant、工具调用等分支；本章只测试单轮文本路径，不假装单轮测试已经覆盖所有工具／多轮语义。

## P25.9 Thinking、non-thinking、预算是三件事 {#p25.9-thinking-budget}

对这两个 checkpoint，模型卡描述了硬开关 `enable_thinking`，以及在启用 thinking 时使用 `/think`、`/no_think` 的软提示机制。**软提示是给模型的文本，不是模板解析器里的另一个布尔开关。**[0.6B 固定模型卡](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/README.md)

至于“想多久”，又是预算问题。`max_new_tokens` 限制整个新生成序列，包括思考、闭合标记、最终答案等；它不是“答案最多多少 token”，也不会自动给答案另留一份额度。`completion_budget` 只是对已经得到的 ID 记账，不会迫使模型结束思考，更不是推理预算控制器。

手算一个假想输出，不把它当作真实模型生成：

```text
生成 ID：[151667, 7, 8, 151668, 9]
总额度：6
思考段：3 个 ID（本口径包括开头 <think>）
闭合：  1 个 ID（</think>）
答案段：1 个 ID
合计：  5 个 ID，剩余 1
```

若额度只有 3，已拿到 `[151667,7,8]`，还没有 `</think>`：应报告“思考未闭合、答案段长度 0”，不能把末尾 8 擅自当作最终答案。此练习把第一个闭合标记作为分界，假设输入是单个正常完成序列；不是任意嵌套标签、工具输出或流式协议的通用解析器。non-thinking 模式则把后续生成序列全记为答案段，包括末尾 EOS 等控制 ID，并不声称所有 ID 都是可显示文本。

这里必须区分教学约定与官方解析：上述固定 revision 的官方模型卡采用最后一个闭合位置（反向查找 151668）；同 revision 的 chat template 对 reasoning 取第一个 `</think>` 之前，对 content 取最后一个之后。教学练习把这个第一个闭合称为首闭合，方便在一条序列里记账。多个闭合标记时这些口径会不同，本练习仅为记账选择第一个，不宣称它是官方统一口径，并保留最后闭合的官方示例口径。[固定模型卡](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/README.md)；[固定模板](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer_config.json)。

此外还有上下文预算：`prompt token 数 + max_new_tokens` 不能超过选用的运行限额。模板中的角色边界、空 thinking 块、历史消息也属于 prompt；字符数不能代替 token 数。`context_budget(12,8,20)` 正好用完 20，`context_budget(12,9,20)` 必须拒绝。

最后，**配置长度不等于能力保证**。两个 config 的 `max_position_embeddings` 都是 40960；对应模型卡描述的原生上下文是 32768，两者相差 40960 − 32768 = 8192。30B-A3B 的卡另外说明借助 YaRN 达到 131072，并解释 40960 默认值的预算安排。两个快照的 tokenizer_config.json 里 `model_max_length` 是 131072，这是 tokenizer 截断和长度警告用的上限，不是模型配置，也不证明模型能处理这么长。本地快照的 `rope_scaling=null`，本章没有启用或验证 YaRN，更不能从 40960 的字段宣布 131072 已经可用。[30B-A3B 长文本说明](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/README.md#processing-long-texts)

本章保存的 generation config 声明采样、temperature=0.6、top-p=0.95、top-k=20。官方卡对 non-thinking 推荐的是另一组采样设置；模板开关本身不会自动重写所有生成参数。不要把上一章用于严格对照的教学贪心解码直接包装成这里的推荐生产配置。

## P25.10 运行：不下载权重，也能得到有意义的证据

在仓库根目录运行：

```bash
# 标准库：校验本地快照、输出每个组件的来源与预算。
python code/26_qwen3_case.py

# 仓库基线 PyTorch/Jinja2：执行随机小模型和模板路径。
OMP_NUM_THREADS=1 python code/26_qwen3_case.py --torch

OMP_NUM_THREADS=1 python -m unittest discover -s tests -p test_qwen3_case.py -v
```

默认输出为每项组件附上带完整 revision 的 source URL，并打印本章精确参数和缓存数字。程序无网络请求。

`--torch` 使用 CPU、附录 H 所列 PyTorch、Jinja2 `3.1.6`、种子 250、单线程、float64。缺少 PyTorch 或 Jinja2 时给出安装 `requirements.txt` 的提示并以状态 2 退出；不是“只打印配置就算可选实验通过”。本章没有额外安装 Transformers，也不修改仓库的依赖锁定。

微型配置为 $D=8$、$H_q=4$、$H_{kv}=2$、$d_h=4$、2 层、词表 17，仍然保留 $H_qd_h\ne D$ 的关键结构。dense 的 FFN 宽度 12；MoE 是 4 个宽度 6 的专家、每 token 选 2 个，没有共享专家。它们只随机初始化，没有训练。

本次实际输出的稳定字段为：

```text
tiny dense: params=1536; shape=(1, 5, 17); cache_bytes=1280
tiny moe:   params=2312; shape=(1, 5, 17); cache_bytes=1280
pretrained inference: NOT RUN; no weights or BPE vocabulary loaded
```

脚本还报告完整输入与 `[2,3]` 分块输入之间的最大 logits 误差；`max_cache_error` 的具体值以本次运行日志为准，检查阈值为 `1e-10`。CPU 架构、BLAS kernel 与矩阵分块可能改变浮点末位，因此不把某台机器上的误差数值或恰好为零当作跨平台保证。1280 字节按 float64 的每元素 8 字节计算，不要拿它与上一节 BF16 的结果混比。

这个 probe 支持等长、无 padding 的 eval 批次，并保护缓存 owner、参数版本、形状、dtype 和上下文边界。缓存应由调用者视为只读；手工修改 tensor 内容或用 `.data` 绕过 PyTorch 版本跟踪不属于支持的流程。它没有训练目标、量化、fused attention、分布式路由或预训练 state 加载，不能作为真实服务的替代品。

## P25.11 为什么这次不是让两个错误实现互相作证

本章的数值结论不靠两份实现互相比对：结构与数值以独立的标量参考为准，另可与官方实现对照，避免两份实现共享同一个错误而一起通过；分层做法与变体清单见 `docs/testing.md`。

运行本章测试文件，末尾应显示 `OK`；测试分类、计数与错误版本清单见 `docs/testing.md`。

本章的数值对照只验证小型结构和数值路径，不表示已经加载或运行上述真实 checkpoint。

这些检查能证明这份教学代码在已覆盖条件下符合我们明确写出的数学规则，**不能证明真实 checkpoint 的输出质量、GPU 性能、长上下文可靠性或 reasoning 能力**。

## P25.12 可选真实推理：明确标记为未运行

如果读者有足够内存、磁盘和下载条件，可以另外使用 0.6B 做真实推理，但这不是本章必修，也不是 `--torch` 的含义。本书本次没有运行以下路线：

1. 建立独立环境，核对所选 Transformers 版本及依赖；这里的解释快照是 v4.51.3，不据此要求替换仓库基线环境。
2. tokenizer 与模型都传入同一个 `Qwen/Qwen3-0.6B` 和完整 revision `c1899de289a04d12100db370d81485cdf75e47ca`，不能只锁模型、不锁 tokenizer。
3. 按该版本的正式 `apply_chat_template` 生成输入，并用真实 tokenizer 测量 prompt 长度。选择 thinking 开关后，还要单独核对采样设置与总输出预算。
4. 记录设备、软件版本、精度、实际生成长度、是否截断、原始输出与解析结果；不要把“能加载”写成质量复现。

不要为了跟着本章先下载 30B-A3B 权重。读取几 KB 的配置与运行几十 GB 的权重，是完全不同的资源承诺。

## P25.13 自测：每道题都有可运行的答案

先手算，再在 `tests/test_qwen3_case.py` 中找到同名测试。

1. **0.6B 的 Q 投影输出为什么不是 1024？** 因为显式 $d_h=128$、$H_q=16$，输出为 2048；O 再映回 1024。见 `test_exercise_explicit_head_width_and_qk_norm_weights`。
2. **Q norm 是否有 16×128 个 gain？** 没有，本快照只有 128 个，按最后一维归一化、跨头共享。对应同一形状测试及 `test_exercise_qk_norm_and_split_half_rotation_hand_reference`。
3. **0.6B 的输入／输出 embedding 是否应该计算两遍？** 不应，绑定后唯一架构参数为 596,049,920。见 `test_exact_real_parameter_totals_independent_hand_arithmetic`。
4. **30B-A3B 应该再加宽度 6144 的 dense FFN 吗？** 不应，这份配置所有层都选择专家路径。把稀疏步长改成 2、层数改成 4，并排除下标 1 的层后，只有下标 3 的层是 MoE。见 `test_mixed_sparse_layer_selection_no_unused_dense_mlp_count`。
5. **四个专家 logits 为 `[2,1,-1,0]`、top-k=2，实际调用谁？** 仅下标 0、1 的专家，权重重新归一化约为 0.731059、0.268941。见 `test_sparse_moe_dispatch_and_selected_probability_hand_oracle`。
6. **BF16、单条 4096-token 序列，两个真实配置的缓存谁大？** dense 为 448 MiB，MoE 为 384 MiB；都不含权重和临时内存。见 `test_exercise_cache_bytes_and_head_ratio`。
7. **`enable_thinking=False` 和在文本里写 `/no_think` 相同吗？** 不同。前者改变模板后缀；后者是保留给模型解释的软提示。见 `test_exercise_official_template_not_naive_concatenation`。
8. **额度为 3、没有闭合思考标签时，是否已经有答案？** 不能这样认定；本练习记答案长度 0。闭合标记也占总输出额度。见 `test_exercise_budget_counts_closure_and_missing_answer`。
9. **把五个 token 分成 1、2、2 三块，缓存长度该是什么？** 依次为 1、3、5；最终 logits 要与独立完整前缀一致，微型 float64 K/V 共 1280 字节。见 `test_arbitrary_chunks_match_independent_full_prefix_and_cache_bytes`。
10. **微型参数表只是公式吗？** 不是，还核对了实际模块的每个参数名字、形状和唯一总数：dense 1536，MoE 2312。见 `test_exercise_tiny_actual_named_shapes_and_unique_counts`。

## P25.14 延伸阅读：读其他模型的配置 {#p25.14-config-compare}

读其他模型时，可以照本章的方法逐字段对照官方配置或报告，例如[Kimi K3 技术报告，arXiv:2607.24653](https://arxiv.org/abs/2607.24653)的 Table 1。

## 本章小结

读完本章，应能把“Qwen3 用了 GQA 和 MoE”改写为更有信息量的句子：**这份完整 revision 的 dense／MoE checkpoint，按这些配置和实现证据使用了这些组件；我们只在这个小型、可复算的范围内验证过它们。** 这也是阅读下一个模型案例时应保留的习惯。
