# 架构进阶（P14）：MHA、MQA、GQA 与 MLA {#p14-grouped-kv}

上一章把归一化和 FFN 换成现代 Block，但每个 query 头仍有自己的一份 K/V。现在进入 S4：**保留 S3 的 RoPE、RMSNorm、SwiGLU，只把注意力改成可以共享 K/V 的 GQA**。另做一个独立的小矩阵实验，解释 MLA 为什么不是另一种“少几个头”。

配套 [code/15_grouped_kv.py](../code/15_grouped_kv.py)。不下载模型，不跑 GPU，不实现 DeepSeek 生产 kernel；普通 CPU 就能检查形状、缓存字节数和数值关系。

## P14.1 从四个读者、四套索引说起

把 query 头想成四个读者：每个人带着自己的问题，去查历史 token 的索引 K，再按照匹配程度读取内容 V。

- **MHA**：四个读者各有一套 K/V。
- **MQA**：四个读者共用一套 K/V，但仍然各提各的问题。
- **GQA**：分成两组，每组两个读者，共用这一组的 K/V。
- **MLA**：不是简单合并索引套数，而是保存一段较短的内容潜变量，再通过各头不同的映射参与运算；还要保存单独的位置键。

类比到此为止：K/V 和潜变量不是人能直接阅读的摘要，也不是可无损压缩任何资料的 ZIP 文件。它们是模型训练得到的向量。减少表示自由度可能影响质量，必须靠训练和评估判断。

![GQA 按连续组共享 KV；MLA 缓存低维内容和解耦位置键，两者不是同一个机制](assets/diagrams/grouped_kv_mla.svg)

## P14.2 先区分两个头数

继续用 B 表示 batch，T 表示序列长度，D 表示主干宽度。本章新增两个量：

- Hq：query 头数。
- Hkv：key/value 头数。本实现 K 和 V 的头数相同。
- d：每头宽度，本章令 d=D/Hq，Q、K、V 每头宽度相同。

这三个名字的边界可以放进同一张表：

| 名称 | KV 头数 | Hq=4 时的例子 | 每个 KV 头服务多少个 query 头 |
| --- | --- | --- | --- |
| MHA | Hkv=Hq | Hkv=4 | 1 |
| GQA 的中间情况 | 1<Hkv<Hq | Hkv=2 | 2 |
| MQA | Hkv=1 | Hkv=1 | 4 |

从广义参数化看，MHA 和 MQA 是 GQA 的两个边界。实际说“采用 GQA”时，人们常指表格的中间情况。MQA 中的“单”指只有一个 KV 头，**不是整个模型只剩一个注意力头**。

本实现要求 `Hq % Hkv == 0`，使每组大小相同。例如 4 个 query 头分成 3 个等大的组就不合法。这是本教学实现的分组约定，不是说所有想象得到的不等分共享都数学上不可能。

## P14.3 分组映射不能凭感觉写

Hq=4、Hkv=2 时，每组有 g=Hq/Hkv=2 个 query 头。编号从 0 开始：

| query 头 h | 使用的 KV 头 `h // g` |
| --- | --- |
| 0 | 0 |
| 1 | 0 |
| 2 | 1 |
| 3 | 1 |

因此映射是 `[0,0,1,1]`，不是 `[0,1,0,1]`。两种排列都能得到合法张量形状，但对应不同的权重组织方式，不能混用。

不用随机数也能抓住这个错误。设只有一个历史位置，KV 头 0 的 V 是 `[10,11]`，KV 头 1 的 V 是 `[20,21]`。只有一个可见 key 时，Softmax 权重必为 1。四个头的输出就必须依次为：

```text
头 0：[10,11]
头 1：[10,11]
头 2：[20,21]
头 3：[20,21]
```

脚本的 `mapping_walkthrough()` 正是这个算例。测试还写了“每个 batch、每个 query 头、每个位置分别算”的慢参考，不与快速路径共享映射函数。否则两个实现一起写错，full/cache 对齐也可能照样通过。

## P14.4 真正要变小的是投影和缓存

PyTorch 的 `Linear(in_features, out_features)` 把输入最后一维映射到输出最后一维；它保存的 weight 形状是 `[out_features,in_features]`。这一点不要与纸面上右乘矩阵的写法弄反。

```python
D = config.d_model
kv_width = config.num_kv_heads * self.head_dim
self.W_q = nn.Linear(D, D, bias=False)
self.W_k = nn.Linear(D, kv_width, bias=False)
self.W_v = nn.Linear(D, kv_width, bias=False)
self.W_o = nn.Linear(D, D, bias=False)
```

逐行看：

1. Q 仍然输出 Hq×d=D 个数，query 头没有减少。
2. K 和 V 各只输出 Hkv×d 个数，不是先算出全部 MHA 头再丢弃。
3. 每个 query 头依然产生一个 d 维输出，所以拼接后仍是 D 维，W_o 不变。

当已有 P 个缓存位置、本次输入 C 个 token 时：

| 张量 | 形状 | Hq=4、Hkv=2 时的头数 |
| --- | --- | --- |
| 当前 Q | [B,Hq,C,d] | 4 |
| 当前 K、V | 各 [B,Hkv,C,d] | 2 |
| 拼接后的缓存 K、V | 各 [B,Hkv,P+C,d] | 2 |
| 注意力计算临时使用的 K、V | 各 [B,Hq,P+C,d] | 4 |
| scores、weights | 各 [B,Hq,C,P+C] | 4 |
| 拼接后的本块输出 | [B,C,D] | 头轴已拼回 D |

本章为了易读，注意力乘法前临时扩展 K/V：

```python
repeats = Hq // Hkv
keys = k.repeat_interleave(repeats, dim=1)
values = v.repeat_interleave(repeats, dim=1)
scores = q @ keys.transpose(-2, -1) / math.sqrt(d)
```

`repeat_interleave(2, dim=1)` 沿头轴连续重复每个 KV 头，得到刚才的 `[0,0,1,1]`。`repeat(1,2,1,1)` 则会把整组头再排一遍，得到 `[0,1,0,1]`，不是同一映射。

**缓存保存的必须还是原始 k、v，而不是临时的 keys、values。** 本实现确实会为临时重复张量和完整 scores 分配内存，因此“缓存较小”不等于“本代码峰值内存按相同比例缩小”，更不等于“运行一定更快”。优化 kernel 可以避免某些显式复制，但本章没有实现或测速这种 kernel。

## P14.5 S4 怎样接在 S3 后面

`GroupedConfig` 继承上一章的 `ModernConfig`，只新增 `num_kv_heads`。`GroupedGPT` 保留词嵌入、Block 的归一化/FFN、最终归一化和输出层，替换各层 `.attn`。新模型仍返回 `(logits, loss)`，训练代码不需要另造一种损失。

在 Hkv=Hq 的边界，投影形状和算法都退回 MHA。测试把 S3 的同一份 state_dict 装入这个边界模型，逐项检查参数、logits、K/V 完全一致。**不同 Hkv 的随机模型不能用来宣称质量相同；这里证明的是 MHA 边界，以及各模型自身 full/cache 的一致性。**

GQA 的缓存不能直接交给 P12 的 MHA 缓存函数。P12 校验的是 Hq 个 KV 头，S4 必须校验 Hkv。源码专门定义 `GroupedCache` 和自己的 `cached_forward`，保留三个规则：

1. 新 Q/K 的 RoPE 位置是 `P,...,P+C-1`。只旋转新 K；旧缓存里的 K 已经转过，V 不转。
2. 矩形 mask 仍是 `key_position <= query_position`。例如 P=3、C=2 时，两行可见性分别是 `[1,1,1,1,0]`、`[1,1,1,1,1]`。
3. 各层缓存保存 `[B,Hkv,P+C,d]`；前一份缓存不被原地改写，新请求用 `None` 开始。

下面是缓存路径的核心，不是另一份模型：

```python
q, k, v = block.attn.project(block.ln1(x), offset=offset)
if cache is not None:
    k = torch.cat((cache.layers[i][0], k), dim=2)
    v = torch.cat((cache.layers[i][1], v), dim=2)
x = x + block.attn.attend(q, k, v, offset=offset)
x = x + block.mlp(block.ln2(x))
layers.append((k, v))
```

`project` 用两种头数做 reshape；`attend` 才临时扩展 KV；最后一行不保存扩展后的副本。缓存同时记录模型身份和参数/位置设置签名。修改权重、RoPE 设置、归一化 epsilon 或头数后，必须重建缓存。不要通过 `.data` 偷改参数，也不要编辑缓存张量；这样的绕过不在检测保证内。缓存推理要求 eval，不支持 padding、变长 batch 或 autocast；训练走正常 full forward。

## P14.6 先算内存账，再测实际张量

本章 K、V 每头同宽，且不量化。L 层、batch B、长度 T、每个数 s 字节时：

$$
\text{KV 字节数}=2\times L\times B\times T\times H_{kv}\times d\times s
$$

开头的 2 表示 K、V 两份。固定 Hq 和 d，缓存相对 MHA 的比例是 Hkv/Hq，不是其平方。query 不在这个缓存里。

脚本取 D=48、Hq=4、d=12、L=2、B=2、T=12，float32 的 s=4：

| 结构 | Hkv | 两层 W_k/W_v 参数总数 | 实际 KV 缓存字节 | 相对 MHA |
| --- | --- | --- | --- | --- |
| MHA | 4 | 9216 | 18432 | 1 |
| GQA | 2 | 4608 | 9216 | 1/2 |
| MQA | 1 | 2304 | 4608 | 1/4 |

例如中间一行的参数数目是 `2×L×D×Hkv×d=2×2×48×2×12=4608`。它只数 W_k/W_v，不是整模型参数。缓存则是 `2×2×2×12×2×12×4=9216` 字节。两者恰好在这个例子相差 2 倍，没有普遍意义。

源码用 `numel() * element_size()` 实测每层 K/V；测试还检查底层存储大小，避免只在 shape 上做一个漂亮的切片，底下仍占着大张量。统计不包含模型权重、临时注意力矩阵、分配器和 Python 对象开销。float64 改为每数 8 字节，缓存字节数翻倍；这也是测试的一部分。

## P14.7 从头训练，不等于转换旧 checkpoint

有两条不同路线：

- **从头训练 GQA**：直接创建 Hkv 较小的 W_k/W_v，所有参数按新结构学习。本章模型支持正常反向传播；脚本不训练、不报告语言质量。
- **把已训练的 MHA 改成 GQA**：旧 W_k/W_v 的形状不同，需要明确的转换策略，再评估是否保留能力。改一个配置值或用不严格加载跳过权重，不是完成转换。

GQA 原论文的做法是把同组 KV 头的投影取平均作为初始化，再继续预训练（uptraining）；论文中的训练预算和质量结果是其特定实验，不是对任意 checkpoint 的保证。[原论文 v3，§2](https://arxiv.org/html/2305.13245v3#S2)

举一个足够小的权重例子：四个头，每头输出宽度 d=1，输入宽度为 2：

```text
旧 weight 的四行：[0,1]、[2,3]、[4,5]、[6,7]
组 0 平均：([0,1]+[2,3])/2 = [1,2]
组 1 平均：([4,5]+[6,7])/2 = [5,6]
```

`mean_pool_kv_weight()` 实现的就是这一层初始化算术，没有执行继续训练。平均后不能声称输出相同：只有一个可见 token，原来四个头的 V 为 `[1,3,5,7]`，两两平均并共享后输出变成 `[2,2,6,6]`。Softmax 此时全为 1，差异仍然存在。

只有组内原本就满足特定相同权重等条件，或者在 Hkv=Hq 不压缩的边界，才可能保持对应函数。**普通 MHA → GQA 的均值转换不是无损等价变换。**

## P14.8 MLA：共享低维内容，不是共享完整 KV 头

MLA 是 Multi-head Latent Attention，首先由 DeepSeek-V2 提出。其关键设计是：用同一个较短的潜变量共同表示内容 K 和 V，并把 RoPE 位置键单独保存；各头的内容上投影可以不同。本章的权重吸收与解耦位置解释以原始 [DeepSeek-V2 报告 v5，§2.1.2–2.1.3](https://arxiv.org/html/2405.04434v5#S2.SS1.SSS2) 为依据。DeepSeek-V3 延续这一结构，可对照 [V3 报告 v2，§2.1.1](https://arxiv.org/html/2412.19437v2#S2.SS1.SSS1)。

为了与 PyTorch 批量张量的方向一致，下面都用**行向量右乘矩阵**。这与论文常用的列向量写法互为转置，不是换了算法。设单个 token 输入为 x_t，潜变量宽度 r，内容 key 宽度 d_c，value 宽度 d_v：

$$
c_t=x_t W_{down},\qquad k^C_{t,h}=c_t U^K_h,\qquad v_{t,h}=c_t U^V_h
$$

| 对象 | 形状 | 作用 |
| --- | --- | --- |
| x_t | [D] | 本层输入 |
| W_down | [D,r] | 降到潜变量 |
| c_t | [r] | 内容 K/V 共同使用 |
| U^K_h | [r,d_c] | 第 h 头的内容 key 映射 |
| U^V_h | [r,d_v] | 第 h 头的 value 映射 |

GQA 中，同组 query 使用**相同完整 K/V**。这里不同头的 U 可以不同，所以同一个 c_t 可以产生不同头的 K/V；共享的是潜在表示，不是说这些头的 K/V 必须相等。

**教学简化要单独记住**：完整 MLA 还包含 query 的低秩降维/升维分支，用来减少训练激活内存，但不进一步缩小历史 KV 缓存。本实验省略该分支，直接给定两头的内容 query 数值，把其一半作为旋转前的位置 query；也不包含完整输出投影 W_O、量化或并行实现。它验证的是 KV 侧吸收与解耦位置的矩阵关系，不是完整 MLA 模型。[V2 v5，§2.1.2，式 12–13](https://arxiv.org/html/2405.04434v5#S2.SS1.SSS2)

“先降维、再升维”也不是任意矩阵的免费无损压缩。`W_down @ U` 的秩最多是 r。比如目标矩阵 `diag(3,2,1)` 的秩为 3，用秩 2 的 `diag(3,2,0)` 代替，最后一个方向被丢掉，Frobenius 误差为 1。源码会打印这个反例。模型必须在这种表示约束下学习；本章不把已有 MHA 权重直接变成等价 MLA。

## P14.9 为什么能不存展开的内容 K/V

先暂时拿掉位置项。当前 query 的内容部分是 q^C_h，某个历史位置的 key 是 `c_t @ U^K_h`。利用矩阵乘法的结合律：

$$
q^C_h(k^C_{t,h})^\top
= q^C_h(U^K_h)^\top c_t^\top
$$

因此可以先算一个 r 维的新 query：`q_absorbed = q_content @ U_k.T`，再与缓存的 c_t 打分。**这一步只移动括号，没有宣称任意低秩近似不损失信息。**

value 也有类似关系。若注意力权重为 a_t：

$$
\sum_t a_t(c_t U^V_h)=\left(\sum_t a_t c_t\right)U^V_h
$$

先对潜变量加权求和，再做一次上投影即可。每个头的 a_t 仍可不同，因此不会强迫所有头输出相同。

原始论文把这种推理重排描述为：内容 key 的上投影可以吸收到 query 投影，value 的上投影可以吸收到输出投影。这里为了让每一步可见，只演示把 value 上投影移到加权求和之后，不把它继续融合进 W_O。[V2 v5，§2.1.2](https://arxiv.org/html/2405.04434v5#S2.SS1.SSS2)

小矩阵实验分别计算：

1. 先显式展开各头的内容 K/V，再做注意力。
2. 不把展开的内容 K/V 当作缓存，只使用潜变量计算打分和加权结果。

对同一组低秩参数，这两种括号位置的结果应当只差浮点舍入，但两条路径相等仍不足以证明算对：它们可能一起忘记位置项，或一起用了错误的缩放分母。测试因此另写一个不调用演示函数内部步骤的标量循环参考：独立列出输入和权重，用 sin/cos 手算位置 0、1、2 的旋转，拼成 4 维 Q/K，直接点积除以 2，再对可见前缀做 Softmax 和 value 加权。两条实现各自对齐这个参考，并逐项核对旋转后位置键。

本函数为核对而同时保留两条路径的张量，**它本身并不是一段峰值内存优化实现**。

## P14.10 RoPE 位置项为什么单独算

不能把位置从内存账本里删掉。若把每个历史位置的内容 K 全部旋转，不同位置带来不同旋转矩阵；一般不能再把它们吸收成一个与历史位置无关的固定 query 投影。原始论文 §2.1.3 明确指出位置旋转夹在相应投影之间，矩阵通常不可交换，因此破坏这种固定权重吸收；随后引入独立位置 query 和共享位置 key。[V2 v5，§2.1.3](https://arxiv.org/html/2405.04434v5#S2.SS1.SSS3)

本章演示的解耦方案把打分拆成两项：

$$
\text{score}_{h,i,t}
=\frac{q^C_{i,h}(k^C_{t,h})^\top
+q^R_{i,h}(k^R_t)^\top}{\sqrt{d_c+d_r}}
$$

- 内容项：K 从 c_t 得到，不对这一部分施加 RoPE，才能使用上面的矩阵结合律。
- 位置项：单独投影并旋转得到 q^R 和 k^R。这里 k^R_t 跨头共享，但 q^R_{i,h} 可以随头不同。
- 因果 mask：依然只能看 t≤i，压缩不取消因果约束。
- 缩放：本例拼接后的 Q/K 总宽度是 d_c+d_r，所以除以它的平方根，不是潜变量宽度 r 的平方根。

推理要缓存的是 **c_t 加 k^R_t**，不是只缓存 c_t。理想紧凑 payload 为 `L×B×T×(r+d_r)×s`，这里不再另外乘 2，因为一个 c_t 同时服务内容 K 和 V。若采用别的表示、复制方式或量化，还要重算账本。

独立实验固定 H=2、T=3、r=2、d_c=d_v=d_r=2，float64，每数 8 字节：

| 表示 | 每 token 保存的数 | 三个 token 的字节数 |
| --- | --- | --- |
| 展开内容 K/V，仍共享位置键 | H×(d_c+d_v)+d_r=10 | 240 |
| 潜变量 + 位置键 | r+d_r=4 | 96 |
| 只算潜变量，漏掉位置键 | r=2 | 48，但这个账本不完整 |

展开对照已经把位置键跨头共享，没有故意为每个头多复制一份来夸大收益。省下的比例由具体维度决定；**MLA 不保证在所有配置下都比 MQA 更省，也不等于“GQA 再少一个头”**。

官方推理源码的吸收路径同样把内容缓存与位置缓存分开；本章只借此核验设计含义，不复刻其并行、量化或 kernel。[DeepSeek-V3 官方 inference/model.py](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py)

## P14.11 运行、读输出、检查失败

在仓库根目录：

```bash
python -m pip install -r requirements.txt
OMP_NUM_THREADS=1 python code/15_grouped_kv.py
python -m unittest discover -s tests -p test_grouped_kv.py -v
```

本批在 Python 3.12.14、PyTorch 2.14.1+cpu、单 CPU 线程、seed=42 下运行。模型表格使用 float32，独立 MLA 矩阵使用 float64；不训练、不下载权重。关键输出：

```text
Hq=4 Hkv=2 mapping: (0, 0, 1, 1)
one-key head outputs: [[10.0, 11.0], [10.0, 11.0], [20.0, 21.0], [20.0, 21.0]]
Hkv=4: KV parameters=9216, cache=18432 bytes, formula=18432, shape=(2, 4, 12, 12), max_error=3.58e-07
Hkv=2: KV parameters=4608, cache=9216 bytes, formula=9216, shape=(2, 2, 12, 12), max_error=3.58e-07
Hkv=1: KV parameters=2304, cache=4608 bytes, formula=4608, shape=(2, 1, 12, 12), max_error=4.77e-07
mean-pooled example: [[1.0, 2.0], [5.0, 6.0]] (initialization, not equivalence)
MLA matrix scores error=1.11e-16, output error=4.44e-16
MLA toy payload compact=96 bytes vs expanded=240 bytes; both include RoPE keys
Rank example: 3 -> 2, Frobenius error=1.0
```

末位浮点误差可随平台变化。模型按 `[3,4,1,4]` 分块，把每块 logits 拼回完整序列，与同一模型的 full forward 比较；误差阈值是 `atol=3e-6, rtol=1e-5`。不同 Hkv 的行不是相互比 logits，也不是质量或吞吐排行榜。

测试覆盖四类问题：

- **算对了什么**：独立逐头参考、手算映射、MHA/MQA 边界、部分 RoPE、同权重全量/单 token/多种分块对齐、有限梯度。
- **真的存少了吗**：K/V 投影宽度、缓存头数、tensor 字节与底层存储字节，而不是只检查打印的公式。
- **哪些输入会被拒绝**：非法头数、空序列、错误 dtype、越界 token、跨模型缓存、长度溢出、坏缓存形状、权重/位置设置改变。
- **错误实现能被抓住吗**：GQA 检查错误的重复映射、RoPE offset、矩形 mask 和扩展后缓存。MLA 再故意把两条路径的缩放一起改成 √r、一起删除位置打分项，或让位置 key 不旋转；每个变体虽然仍能通过内部两路径对齐，却必须被独立参考判为 assertion failure，且运行错误数为 0。这样不会把坏导入或异常崩溃误报为“抓住数值错误”。源码文件不会被这些 mutation 测试改写。

本章共 23 项测试。没有安装兼容的 torch 2.14.x 时，数值类按共享测试门控明确跳过；缺依赖的 CLI 给出安装提示。**skip 不是通过，不能靠只有 CLI 检查成功来宣称 GQA 数值实现已验证。**

## P14.12 自测题与参考答案

1. Hq=8、Hkv=2 时，query 头 5 使用哪一个 KV 头？还能输出 8 个头吗？
2. Hq=4、Hkv=3 为什么被本实现拒绝？MQA 的 query 头数是不是 1？
3. P=3、C=2、Hq=4、Hkv=2、B=1、d=12 时，Q、紧凑 K 缓存和 scores 分别是什么形状？
4. 表格的 GQA 缓存如果从 float32 换成 float64，是多少字节？为什么不是 4 倍？
5. MHA 四个头的单位置 V 为 `[1,3,5,7]`，两两平均后还能声称输出不变吗？
6. MLA 的代数验证误差接近零，是否证明把任意 MHA checkpoint 压成秩 2 都无损？
7. MLA 小实验为什么是 96 字节，而不是 48？打分除以 √2 还是 √4？
8. full/cache 完全相等，能否单凭这个结果证明 `[0,0,1,1]` 的映射写对了？

参考答案：

1. 每组 4 个 query 头，`5//4=1`。仍有 8 个 query 及对应输出头。
2. 4 不能被 3 整除，无法按当前约定等分。MQA 只有 1 个 KV 头，query 头数仍可为 4。
3. Q `[1,4,2,12]`；K 缓存 `[1,2,5,12]`；scores `[1,4,2,5]`。第二行新 query 能看五个 key，第一行只能看前四个。
4. 18432 字节。单个数从 4 字节到 8 字节，只有 2 倍；张量元素数未变。
5. 不能，变成 `[2,2,6,6]`。平均是初始化策略，不是一般的函数等价证明。
6. 否。它只证明同一组已因式分解参数的矩阵结合律；`diag(3,2,1)` 的秩 2 反例仍有误差 1。
7. 每 token 除了 2 个潜变量数，还存 2 个位置键数，共 `3×4×8=96` 字节；内容 key 2 维加位置 2 维，缩放分母是 √4=2。
8. 不能，两个路径可能共享同一个映射错误。还要用手算或独立逐头参考验证。

到这里，S4 仍是一条普通 decoder-only 主线，只更换了 KV 共享方式。下一单元 [P15](#p15-attention-systems) 研究 kernel、缓存管理和服务调度时，要把“模型表示减少了什么”与“系统怎样高效执行”分开记账。

## P14.13 延伸：VHA，用“虚拟头”让 GQA 缓存接近 MHA 质量

GQA 省缓存的代价，是多个 query 头只能读同一份 K/V。2026-10-13 提交的 [VHA 论文（arXiv 2610.17959）](https://arxiv.org/abs/2610.17959) 换了个思路：**缓存仍按 GQA 只存少数 KV 头，但训练时让每个真实 query 头对每个 KV 组各派生一个“虚拟头”，再在头之间做一次低秩混合。**推理前，这两步混合都能吸收到普通权重里，部署形态还是一个 GQA。

两个可训练的小矩阵（下式按论文记号）：

- **Premix**：真实 query 头 i 面对 KV 组 g 时，多乘一个 `W_g^premix`，初始化为 `I + αε`（论文取 α=0.1，ε 为标准正态）。吸收后 `W̃_Q^(i,g) = W_Q^(i) W_g^premix`，只是 query 投影变多，缓存不变。
- **Postmix**：把所有虚拟头输出 Y 按头混合，`Y' = (I + B Aᵀ) Y`，秩 r 取**物理 query 头数**的一半（论文 r = H_q′/2；1.7B 模型里虚拟头有 16 个，r=4），不是虚拟头数的一半，B 初始化为 0，所以训练开始时 Postmix 就是恒等映射。吸收到输出投影：`W_O' = ((I + A Bᵀ) ⊗ I_d) W_O`。注意吸收后是 `A Bᵀ`，转置来自“先混头再乘 W_O”等于“对 W_O 的行块做转置混合”。

[`../code/15b_vha.py`](../code/15b_vha.py) 用 float64 在 CPU 上做三件事，均为断言而非打印：训练形式（显式 Premix/Postmix）与吸收后的 GQA-2 形式最大误差小于 1e-5；B=0 时 Postmix 精确等于恒等；并打印 MHA 与 GQA-2 每层每 token（BF16）的 KV 缓存字节数（d=4 时分别为 128 与 32 字节）。`tests/test_vha.py` 另外检查两个故意写错的吸收：把 `A Bᵀ` 写成 `B Aᵀ`、或忘记把 Premix 并进 W_Q，都必须让输出明显偏离。

论文报告的结果（均为原文表格，本书未复现）：1.7B 规模下，相对基线的平均分差 GQA 16-2-2 为 −1.00、8-2-2 为 −1.71，VHA 为 +0.41，每层每 token（BF16）KV 缓存由 4 KB 降到 1 KB（Table 2）；1.7B、序列长 4096 时训练/prefill FLOPs 降低 7.9%（Table 5）；30B-A3B MoE 上 VHA 50.36、MLA 49.66、GQA 47.76（Table 3）。消融中给混合加 SiLU 反而变差，Postmix 秩 r=4/32/128 的分数 54.50/54.24/54.15 相差很小。

边界：本节代码只验证吸收恒等式和缓存账，不训练模型、不复现论文分数，虚拟头与 KV 组的排列方式是本书按论文公式做的教学重建。

## 资料与实现边界

- [GQA 原论文 v3](https://arxiv.org/html/2305.13245v3)：核验 MHA/MQA/GQA 边界及分组均值初始化、继续预训练的区别。
- [DeepSeek-V2 原始报告 v5，§2.1.2](https://arxiv.org/html/2405.04434v5#S2.SS1.SSS2) 与 [§2.1.3](https://arxiv.org/html/2405.04434v5#S2.SS1.SSS3)：权重吸收、query 低秩分支、RoPE 不可交换性及解耦位置缓存的原始依据；固定版本核验日 2026-10-19。
- [DeepSeek-V3 技术报告 v2，§2.1.1](https://arxiv.org/html/2412.19437v2#S2.SS1.SSS1)：核验 MLA 的联合内容压缩、解耦位置项与缩放维度。
- [DeepSeek-V3 官方推理源码](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py)：核验内容/位置缓存分离的实现入口；动态链接核验日 2026-10-19。

- [VHA 论文 arXiv 2610.17959](https://arxiv.org/abs/2610.17959)（提交日 2026-10-13；核验日 2026-10-25）：Premix/Postmix 初始化、吸收公式与 Table 2/3/5 数字。

本章实现 GQA 的完整 tiny-model 前向和紧凑缓存；MLA 仅实现独立矩阵教学实验，未实现完整训练模型、checkpoint 转换流程、分布式部署或生产性能复现。
