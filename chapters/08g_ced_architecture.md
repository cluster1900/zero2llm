# 架构进阶（P17）：CED 的两条 KV 路径 {#p17-ced}

前两章分别问：怎样安排计算、内存和请求，怎样改变 token 之间的信息路径？本章把镜头移到另一条边：**上半部分网络所读取的历史 K/V，一定要由上半部分自己逐层生成吗？**

这是 DeepSeek-V4.1-Flash 的 Causal Encoder-Decoder（CED，因果编码器—解码器）要解决的问题之一。长任务中，模型常常读入很长的工具结果，只生成较短的回复。如果每次读新输入都要完成全部层的逐 token 计算，prefill 就很昂贵。CED 改变 global KV 的来源，让多数输入 token 可以不经过完整 decoder 计算，同时保留 decoder 的局部处理能力。

本章配套 [code/18_ced_architecture.py](../code/18_ced_architecture.py) 和独立测试 `tests/test_ced_architecture.py`。它们是 **CPU、float64、固定小矩阵的数据流实验**，不是官方模型的缩小权重版，也不是训练或性能 benchmark。完成后，你应当能指出 global / local KV 各从哪里来，解释 replay 为什么有误差，并检查缓存位置有没有接错。

## P17.1 先把三个“编码—解码”分开

第 7 章中的经典 seq2seq 有两条序列：例如先编码中文句子，再自回归生成英文句子。经典 Transformer 的源句 encoder 可以双向看源句，目标句 decoder 则只能看已经生成的目标词。

本章的 CED 不是把同一张图换个名字：

| 容易混淆的说法 | 实际在划分什么 | 本章是否讨论 |
| --- | --- | --- |
| 经典 seq2seq encoder / decoder | 源序列和目标序列的建模角色 | 只作对照 |
| CED causal encoder / decoder | 同一自回归序列中，下半层与上半层，以及 global KV 的来源 | 本章重点 |
| 服务端 prefill / decode 分离 | 请求的两个执行阶段放在哪些设备、怎样调度和传输缓存 | 属于 P15 的服务层 |

tokenizer 的 encode / decode 又是第四件事：文本与 token ID 的转换，既不是这里的网络层，也不是生成阶段。

**网络的 decoder 不等于推理的 decode 阶段。** 在 CED 中，生成一个新 token 的表示，仍然要经过 causal encoder 和 decoder。反过来，prefill 也不是绝对不运行 decoder，稍后会看到需要处理一段后缀。

## P17.2 哪些是已经核验的官方事实

本节只谈 **DeepSeek-V4.1-Flash**，不把结论泛化成所有叫“V4.1”的模型。2026-10-19 核验了技术报告 v1，以及官方 Hugging Face 仓库的固定 revision：

```text
314113f8a7dadb388215396ba5fbee2c610a04d6
```

[技术报告 §2.1–2.2](https://arxiv.org/html/2609.19969v1#S2.SS1) 与 [固定版官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/314113f8a7dadb388215396ba5fbee2c610a04d6/README.md) 相互印证：语言主干为 40 层，分成 20 层 causal encoder 和 20 层 decoder；decoder global KV 来自 encoder 最终表示，而非各 decoder 层自己的表示。官方报告的激活参数口径是 **prefill 每 token 8B、decode 每 token 16B**。这是官方架构数据，**不是本书测得的速度、内存或质量结果，也不是模型总参数量**。

实际配置也必须看，不能从模型名反推：

- [固定版 config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/314113f8a7dadb388215396ba5fbee2c610a04d6/config.json) 的 text_config 中，num_hidden_layers=40，sliding_window=128。
- [固定版 inference/config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/config.json) 使用 n_layers=40、window_size=128；kv_source_layers 为 [2,8,14,20]，索引从 0 开始。
- 主干前两层只有 SWA；随后 18 个 encoder 层的压缩比是 2，20 个 decoder 层的压缩比是 1。配置数组末尾还包含额外预测层，不能把数组长度直接当成主干层数。[报告 §4.2.1](https://arxiv.org/html/2609.19969v1#S4.SS2.SSS1)

再向源码走一步：[固定版 inference/model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/model.py) 中，Attention._compress_kv 由指定 source 层产生 global latent，其他层读共享缓存；decoder 边界的第 20 号层是 source，之后的 decoder 层复用它。每层的 _window_kv 仍从本层 attention 输入生成局部状态。真实路径还包括归一化、位置旋转、量化和 residual-stream mixing，不能将它们抹去后声称与生产张量逐项相同。

这里也发现一个重要边界：参考代码的 Transformer.forward 遍历全部主干层，并不是报告部署系统的“跳过多数 decoder prompt 计算”调度器。[inference README](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/README.md) 明确定位为可读参考实现；其 generation 是普通自回归采样，内置小模型 self-test 的权重未初始化，只查形状与 kernel 连接。本书没有运行官方权重或用该 self-test 证明数值正确性。

## P17.3 普通 decoder-only 的依赖链

设位置 i 的输入向量是 x_i，第 l 层的输入表示是 h_i^(l)。先忽略归一化与多头，普通 self-attention 的 K/V 通常来自本层输入：

$$
K^{(l)}=H^{(l)}W_K^{(l)},\qquad V^{(l)}=H^{(l)}W_V^{(l)}.
$$

因此，如果想得到很高一层的历史 K/V，通常要先算完历史 token 在下面各层的表示。KV cache 可以避免“已经算过又重算”，但没有缓存的新输入仍要经过这条层间链。

这不是说全量 prefill 必须按 token 一个一个串行执行。因果 mask 允许同一层的多个位置并行计算；**沿层的依赖**和**沿时间的可见性**是两个不同问题。CED 改变的是前者中的 KV 来源，而不是放弃后者的因果约束。

## P17.4 CED 改了哪条边

把下半部分的最终输出记为 E，形状 [B,N,D]。B 是 batch，N 是输入长度，D 是每个 token 的表示宽度。上半部分 decoder 的某层输入仍记为 H^(l)。

为了先看清路径，本章教学模型使用分开的 K、V：

$$
K_g^{(l)}=E W_{Kg}^{(l)},\qquad V_g^{(l)}=E W_{Vg}^{(l)}.
$$

下标 g 是 global。**右边是 E，不是 H^(l)。** 查询 Q 则仍由当前 decoder 层提出。可以把它理解成：多个深层处理步骤各自提问，但查阅的历史资料可以先由下面的网络准备好。

生产模型的报告用 KV latent C 和压缩权重 Z 表达投影，不是本章这两块独立矩阵；CSA2 还会跨层复用，因此并非每个 decoder 层都独占一份 global cache。先分开写 K/V 是为了看清来源，不是对官方 checkpoint 的逐张量复刻。

与此同时，局部路径仍保留原来的依赖：

$$
K_{local}^{(l)}=H^{(l)}W_{Kl}^{(l)},\qquad
V_{local}^{(l)}=H^{(l)}W_{Vl}^{(l)}.
$$

local 的意思是滑动窗口，不是“来自另一台本地机器”。这两套 KV 不能互换：global 可以承接较远的历史，local 保留本层对最近上下文的加工。

![教学 CED 数据流：encoder 最终输出投影为 decoder global KV；decoder 本层输入产生 local KV；有界 replay 只处理后缀且为近似](assets/diagrams/p17-ced.svg)

### 因果 encoder 为什么不会偷看未来

“encoder”这个词本身不保证双向或单向，mask 才决定可见性。这里的 encoder 是因果的：E_j 只依赖 x_0 到 x_j。

decoder 在位置 i 又只读取位置 j≤i 的 global 条目。两条规则串起来，任何可读条目 E_j 都不可能包含 x_(i+1) 之后的信息。局部路径同样必须满足 j≤i。因此允许提前算出整段 E 和 global bank，**不代表每个 query 都可以读 bank 的所有行**。

这里的 global 指覆盖历史的范围，不保证生产系统每次读取所有历史条目。真实模型还会做稀疏选择；本章为了隔离数据来源，用稠密的因果 global attention，省去索引器。

## P17.5 为何还要 replay

如果 global KV 已经准备好了，为什么不把整个 decoder 都跳过？因为生成开始时还需要各 decoder 层的 **local KV**。高层 local KV 依赖低层处理后的表示，不能直接拿 encoder 的 global KV 充当替身。

replay（重放）就是重新运行一段历史的计算，用它恢复当前需要的状态。它不是把历史文字再生成一遍，也不是修改用户输入。

设局部窗口 W 包含当前 token，每层最多再读取 W−1 个旧位置。单层看起来只需要一小段，但多层会把影响向前传播：第二层的某个局部 value 已经混合了第一层更早的信息。因此“最后 W 行还在”不等于“能从这 W 个原始输入精确重建所有深层状态”。

[报告 §3.2.2](https://arxiv.org/html/2609.19969v1#S3.SS2.SSS2) 的 Bounded Replay 只重放一段固定长度后缀，并把局部注意力截断在段内。若段起点为 s、当前 query 为 i，局部可见范围变成：

$$
\max(s,i-W+1)\le j\le i.
$$

这会丢掉 replay 边界之前的局部依赖。**报告明确说这不是全量计算的数学等价重建。** 官方对响应质量影响的判断来自其模型与评估，本书的小矩阵误差不能验证或否定那项质量结论。

还要分清两个用途：

- Encoder replay：持久缓存命中 global、却缺少 encoder SWA 状态时，重放缓存前缀的末尾，再处理新后缀；不覆盖已命中的 global cache。近似状态可能使新后缀的结果依赖缓存命中边界。
- Decoder replay：global KV 已由 encoder 准备好，再把 prompt 末尾的一段 encoder 输出送进 decoder，准备之后生成所需的 local 状态。

**本实验只模拟第二种。** 我们完整计算 encoder，因而 full 与 bounded 两条路径中的 encoder 和 global KV 完全相同；生产 encoder replay 不享有这个简化保证。代码不模拟 SSD、主机内存、驱逐、缓存命中策略或分布式服务。

## P17.6 CED、CSA2 与量化各管什么

不要把这些名字合成一个“省显存技巧”：

| 机制 | 主要改变的问题 | 不能据此推出 |
| --- | --- | --- |
| CED | decoder global KV 从哪里生成，哪些 prompt 位置要经过完整 decoder | 所有 decoder prompt 工作都消失 |
| SWA Bounded Replay | 缺失局部状态时重算多长的后缀，接受什么近似 | 与全量前向完全等价 |
| CSA2 | 哪些层共享 global KV、哪些层重做或复用稀疏索引 | 所有层的 Q 和 local KV 都相同 |
| KV 量化 | 每个缓存数值怎样表示、占多少字节 | 数据流完全不变就一定无误差 |
| P/D 服务分离 | 计算阶段怎样分配到设备与队列 | 模型已经采用 CED |

[报告 §2.3](https://arxiv.org/html/2609.19969v1#S2.SS3) 与固定版模型卡把 CSA2 分成 Full、Reindex、Reuse 三种模式：Full 产生 global KV 并选择索引；Reindex 复用 global KV、重新选择索引；Reuse 连选择结果一起复用。它们仍保留本层 query 与 SWA 状态。这里的 Full 是 **CSA2 模式名**，不能理解成“取消因果约束的全注意力”。

所以，CED 解释“资料从下半网络准备”，CSA2 进一步解释“哪些层共用资料和选取结果”。两者可以协同，但不应把所有缓存节省都归功于 CED。相关头数 / latent 概念见 [P14](#p14-grouped-kv)，局部与稀疏路径见 [P16](#p16-sparse-linear)。

## P17.7 先把教学模型的边界写在前面

ToyCED 输入不是 token ID，而是已经准备好的二维向量。默认两层 encoder、两层 decoder，窗口 W=3；所有权重都显式初始化并固定，全部在 CPU 上用 float64 运算。

本模型故意只保留：

1. 因果的局部 encoder。
2. 每个 decoder 层从同一个最终 E 投影 global K/V。
3. decoder 本层输入产生 local K/V。
4. 完整前向、增量缓存和近似后缀 replay 三条路径。

它省略 tokenizer、词嵌入训练、输出词表和 loss，也省略生产的 MoE、压缩 latent / Z、CSA2 共享与 Top-K、层级索引器、RoPE / YaRN、RMSNorm、FFN、mHC、Engram、量化、视觉输入、DSpark 与 GPU kernel。因此输出是二维 hidden states，**不是语言模型 logits，不能用来评估语言质量**。

还有两项特意不同的数学约定：

- 为了让新手逐项手算，本模型分别计算 global 和 local 的 Softmax，然后把两路结果加到残差上。官方参考 Attention.forward 则合并局部与 global KV，送入同一次 sparse_attn，不能拿我们的“两个平均值相加”解释生产 attention 的归一化。
- 本模型给初始向量加上 `[0.01×position, −0.01×position]`，只为让位置错误可见。它不是 RoPE，也不提供长上下文外推能力。

因此这是“保留关键依赖边”的实验，不是通过换小维度就自动忠实复现完整模型。

## P17.8 手算：三行输入如何走完两条路径

先使用更简单的配置：encoder 一层、decoder 一层、W=2，位置项关闭；所有 Q、K 投影为零，所有 V 投影为单位矩阵。分数都是 0，Softmax 就是可见位置的平均权重。第二个坐标始终为 0，只写第一个坐标：

```text
输入 x = [1, 2, 3]
```

encoder 只看自身及前一个位置，再加回残差：

| 位置 | encoder 可见输入 | 均值 | 加上本位置输入后的 E |
| --- | --- | --- | --- |
| 0 | [1] | 1 | 2 |
| 1 | [1,2] | 3/2 | 7/2 |
| 2 | [2,3] | 5/2 | 11/2 |

于是 E=[2,7/2,11/2]。decoder 的 global V 直接来自这三行 E，local V 来自本层输入；第一层的输入恰好也是 E，但多层时不能继续混为一谈。

位置 2 的完整 decoder 输出为：

$$
\underbrace{\frac{11}{2}}_{\text{残差}}
+\underbrace{\frac{2+7/2+11/2}{3}}_{\text{global 平均值 }11/3}
+\underbrace{\frac{7/2+11/2}{2}}_{\text{local 平均值 }9/2}
=\frac{41}{3}.
$$

完整的输出是 `[6,9,41/3]`。

现在只 replay 末尾 W=2 个位置，即位置 1、2。**global bank 仍有全部三行 E**，只是 decoder 的局部计算从位置 1 重新开始。

- replay 位置 1 的 global 均值仍为 (2+7/2)/2=11/4；local 却只能看到自己，为 7/2。输出是 7/2+11/4+7/2=39/4=9.75，而完整结果是 9。
- replay 位置 2 恰好已经拥有完整的单层局部窗口，所以这一行仍是 41/3。

这说明两件事：有界 replay 已经在第一行产生差异；**最后一行偶然相同，也不能证明多层情况下等价**。下一层可以读取已经改变的前一行，差异便继续传播。默认两层实验会实际展示最后一个位置也不同。

## P17.9 对照源码：不要只看图上写了 encoder

最重要的实现不长：

```python
# global_from_encoder()：在运行 decoder 前准备好各层 global bank
kv = join_kv(encoder @ layer.global_k, encoder @ layer.global_v,
             offset, None if old is None else old[i])
```

这里的 encoder 就是两层 causal encoder 的最终输出，不是把 decoder 的输出换个变量名。每层 global bank 都在 decoder 开始前生成，不依赖 decoder 本身。

相反，decoder 的局部路径是：

```python
kv = join_kv(h @ layer.k, h @ layer.v, offset,
             None if old is None else old[i])
q = h @ layer.q
h = h + attention(q, global_kv[i], offset) + attention(q, kv, offset, window)
```

h 每经过一层就改变。于是第二层 local V 会跟着第一层的处理改变，而提前准备的 global V 不会。

测试对此做了“干预”，不只检查变量名：把第一层 decoder 的 local V 投影放大三倍，重新计算。结果应当是：encoder-derived global banks 保持不变，第二层 decoder 的 local cache 和最终输出改变。如果 global bank 也被改变了，说明路径很可能接回了 decoder 自己。

## P17.10 形状和绝对位置必须一起记

默认例子 B=1、N=6、D=2、W=3。这里 D 始终是向量宽度；后文用 L_e、L_d 表示 encoder、decoder 的层数。

| 对象 | 完整 prefill 后的形状 / 位置 | 来源 |
| --- | --- | --- |
| 输入、encoder 输出 | [1,6,2] | 输入二维向量与 encoder 计算 |
| 每个 decoder 层的 global K、V | 各 [1,6,2]，绝对位置 0…5 | encoder 最终输出 |
| 每个 encoder / decoder 层的 local K、V | 各 [1,2,2]，绝对位置 4、5 | 各自层的输入 |
| full 输出 | [1,6,2]，位置 0…5 | 完整 decoder |
| bounded 输出 | [1,3,2]，位置 3…5 | 只运行最后 3 行 decoder |
| 下一次位置 | next_position=6 | 已处理 6 个位置 |

为什么 local cache 留 W−1=2 行而不是 W=3 行？因为下一次会先加入当前新行。处理位置 6 时，局部可见位置正好是 `[4,5,6]`。这只是本实验的存储约定；生产实现可以用大小为 W 的环形缓冲区。

代码用 KV.start 记录第一行的绝对位置，不能把缓存的行号 0 当成原序列的位置 0。增量输入 C 行时，位置项要从 offset 开始；注意力的比较也是 `key_position <= query_position`，而不是按本次小块的行号比较。

当 W=1 时，不需要保留旧 local 行。切片要专门处理长度 0：Python 中 `-0` 等于 `0`，`tensor[:, -0:]` 会意外保留全部历史。tail_kv 使用 `size-n` 作为起点，测试覆盖了这个边界。

缓存是调用者持有的快照，新请求传 None。增量调用不原地覆盖之前的快照，可以从同一个前缀分叉实验。它还记录模型身份和配置 / buffer 版本；常规原地改权重、换配置或换模型后会拒绝旧缓存。不要编辑缓存张量，也不要用 `.data` 或底层共享存储写入绕过版本跟踪；这里不承诺检测这些破坏。

bounded 路径会把 approximate 标记保留到后续 decode。它记录“来自截断 replay 的状态”，不是错误大小；某些窗口或输入上误差恰好为零，也不能把来源标记擅自改成精确。

## P17.11 算清少走的路径，别把路径数叫速度

定义：B 为 batch，N 为 prompt 长度，L_e / L_d 分别为 encoder / decoder 层数，R=min(N,W) 为 replay 长度。

本实验把“一个 token 的表示经过一个层”记作一次 **token-layer visit**。完整和有界路径分别是：

$$
C_{full}=BN(L_e+L_d),\qquad
C_{bounded}=B(NL_e+RL_d).
$$

令 B=1、L_e=L_d=L/2 且 N≥W，则 C_bounded=NL/2+WL/2，对应报告 §2.2 的 O(NL/2+n_win·L/2) 路径计数写法；短输入仍应使用 R=min(N,W)。

这里的 full 基线是同一个 CED 教学模型的完整执行，不是另一个普通 decoder-only 模型，更不是两个训练模型的质量对比。这是本模型实际循环累计的层处理次数，不是仅打印一条假想公式。被省掉的是 `B(N−R)L_d` 次 decoder token-layer visit。

但不能把所有工作都塞进这个计数：两条路径仍为全部 N 个 encoder 输出做 global K/V 投影。本教学模型每个 decoder 层各有一对投影，故两边都要 `BNL_d` 对投影。**官方 CSA2 有跨层复用，不能把本实验这个投影数量直接套到真实模型。**

默认 B=1、N=6、L_e=L_d=2、W=3：

| 计数项目 | 完整路径 | 有界路径 |
| --- | --- | --- |
| encoder token-layer visits | 12 | 12 |
| decoder token-layer visits | 12 | 6 |
| 合计 block visits | 24 | 18 |
| global K+V 投影对 | 12 | 12 |

24 变成 18，**不等于运行时间变成 3/4**。不同路径的每次 attention 涉及的 key 数不同，投影和内存操作也有成本；本脚本甚至显式构造稠密分数矩阵，再用 mask 遮住局部不可见位置，根本不是稀疏 GPU kernel。

当 N≤W 时，R=N，有界路径不会省掉 decoder 的任何位置。只有输入显著长于窗口时，计数差异才突出。即使如此，也不能由这个固定小矩阵实验推出官方吞吐、TTFT、TPOT、显存节省或语言质量。

## P17.12 运行与验证

在仓库根目录安装依赖后运行：

```bash
python -m pip install -r requirements.txt
OMP_NUM_THREADS=1 python code/18_ced_architecture.py
OMP_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_ced_architecture.py' -v
```

若使用仓库的虚拟环境，把 python 换成 `.venv/bin/python`。没有 PyTorch 时，CLI 给出安装提示并以非零状态退出；测试按仓库稳定 torch 版本门槛跳过数值部分，**跳过不是通过**。

本章在 CPU / torch 2.14.1 / float64 下验证，核心输出为：

```text
手算 encoder: [2.0, 3.5, 5.5]
手算 full decoder: [6.0, 9.0, 13.666666666666666]
手算 bounded 后缀: [9.75, 13.666666666666666]
完整 / 有界 replay 的 block visits: 24 18
两条路径各自 global K+V 投影对计数: 12
有界 replay 的输出范围: 3 到 5
最后位置 full / replay 最大差: 0.014462788
```

末行是两组教学 hidden states 的最大绝对差，不是困惑度、回答错误率或可接受质量阈值；不同平台的末位浮点数可略有变化。

### 为什么 full/cache 对齐还不够

如果完整路径和缓存路径共用一个错误 mask，两条路径可能同时看未来，对齐测试仍然会通过。因此测试同时覆盖：

1. **独立标量 oracle**：用 Python 循环做二维乘法、逐位置 Softmax 和残差，不调用脚本的 attention、projection 或 cache helper；检查 full 和 bounded 两条路径。
2. **手算结果**：直接断言 `[2,7/2,11/2]`、`[6,9,41/3]` 与 `[39/4,41/3]`。
3. **未来隔离**：对每一个 prefix cut 改写之后的输入，前缀 encoder / global / 输出都不应变化；replay 也不能提前读 global bank 的未来行。
4. **缓存等价**：每一种两块切分和逐 token 增量都与独立 full oracle 对齐，包含 offset、W=1、短 prompt、快照分叉和新请求。
5. **近似差异**：多层 replay 的末位置以及后续 decode 确实可与完整路径不同；不把“不等价”误写成 bug。
6. **错误变体**：故意把 global 来源接成当前 decoder h、取消局部窗口、放开未来、重置增量位置、把 replay 全局 query 位置重置为 0。必须被实际数值断言抓住，不能把变体运行崩溃当成“成功检测”。

完整数值比较使用 rtol=atol=1e−12。这只适用于本章小规模 float64 实验；生产低精度算子不应照搬这个阈值。

## P17.13 小练习与本章结论

1. 将 W 从 3 改成 8，输入仍为 6 行。为什么 bounded 输出长度变回 6、block visits 也不再减少？
2. 不看图，只检查 global_from_encoder 与 decode 的参数。哪一条调用证明第二层 decoder 的 global KV 不依赖第一层 decoder？
3. 手算例子里最后一行相同，为什么不能删除“近似”警告？试着把 decoder_layers 改成 2，追踪第一层位置 1 的误差会流向哪里。
4. 如果把本实验换成真正的 CSA2，你需要新增哪些缓存、source 映射和索引有效性测试？为什么不能只把类名改成 CSA2？

### 参考答案

1. R=min(6,8)=6，整个输入都在 replay 范围内；两层 encoder 与两层 decoder 共计 6×4=24 次 block visits，和完整路径相同。
2. forward 先调用 global_from_encoder(encoder, ...)，再调用 decode(encoder, ..., global_kv, ...)；所有 decoder 层的 global bank 在第一层 decoder 运行之前就已经生成。测试放大第一层 decoder 的 local V 后，global bank 也确实不变。
3. 延续手算配置，只把 decoder 改为两层。第二层位置 2 的 global 平均仍为 11/3，残差仍为 41/3。完整 local 均值为 (9+41/3)/2=34/3，结果是 **86/3**。replay local 均值为 (39/4+41/3)/2=281/24，结果是 **697/24**，比完整结果大 **3/8**。第二层读到了第一层已改变的位置 1，单层末位的巧合就消失了。self-test 和独立测试都断言这组数。
4. 至少要分别跟踪 main KV、indexer K、Top-K 选择及各自 source，处理共享层的写入与读取时机，并重新验证索引与压缩条目的因果可见性。压缩块没有收齐时不能提前公开含有未来输入的摘要；修改名字不提供这些保证。

记住这条主线即可：**CED 改变 decoder global KV 的来源；局部 SWA 仍有自己的逐层状态，所以需要 replay；有界 replay 是明确接受误差的取舍，不是免费的等价省算。** 本章已经用实际张量和反例验证这三件事，但没有复现官方训练、权重、服务系统或性能。
