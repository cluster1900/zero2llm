# 第4章：多头注意力（Multi-Head Attention）——给每个维度记一本账

> 多头不是把同一张注意力表复印几份，而是学习多套投影，分别计算权重，再把结果合起来。

上一章我们用 $Q$ 和 $K$ 打分，做 Softmax，再用权重汇总 $V$。本章把这套计算分成几个并行的“头”，重点追踪形状怎样变化、参数和计算成本怎样计数。

配套代码是 [code/04_multi_head_attention.py](../code/04_multi_head_attention.py)。本章只讨论**标准、等宽的多头自注意力**：Q、K、V 来自同一段输入，三者头数相同，每个头的 Q、K、V 宽度也相同。共享 KV 头等变体留到后续章节（大纲 S05：[GQA 原始论文](https://arxiv.org/abs/2305.13245)）。

---

## 4.1 为什么要“多头”：多张权重表，而不是预先分配的专家

假设我们在读一句话：“小林把书递给小周，因为她要准备考试。”理解它可能涉及动作的参与者、物品、指代和上下文等关系。

单头注意力并不是“只能理解语法，不能理解指代”。它也可以让一个位置同时关注多个位置，向量中的不同特征还可以承载不同信息。它的具体结构限制是：**对一个查询位置，这个头只有一组沿 key 位置归一化的权重，同一组权重用于汇总该头的所有 value 特征。**

例如，一个头在三个可见位置上的权重是 `[0.7, 0.2, 0.1]`，那么这个头的每一个输出特征，都按这三个比例对相应的 value 特征加权。多头则允许另一套投影得到另一张权重表，例如 `[0.1, 0.3, 0.6]`，再把两种汇总结果一起保留下来。

```text
同一段输入 X
   ├── 学习得到的投影 1 ── 注意力表 A1 ── 头输出 Z1
   ├── 学习得到的投影 2 ── 注意力表 A2 ── 头输出 Z2
   └── 学习得到的投影 H ── 注意力表 AH ── 头输出 ZH
                                            │
                            按特征拼接，再经过输出投影
```

“分别计算”指每个头有自己的投影和权重；它们仍共享输入、一起训练，不意味着统计上互不相关。可以把多头想成几位读者各记笔记，但程序没有安排“你只看语法，你只看情感”的岗位。训练后各头可能形成不同模式，也可能彼此相似。**类比的边界：允许分工，不保证语义专家，更不保证头越多越好。**

原始 Transformer 论文把多头描述为在不同表示子空间中分别做注意力、再拼接与投影。我们接下来把这件事落实到具体的维度和代码上。[来源：Attention Is All You Need，§3.2.2](https://arxiv.org/html/1706.03762v7#S3.SS2.SSS2)

---

## 4.2 先投影，再分头：不是把原始输入硬切成几块

先统一本章的符号。遇到形状时，先读每根轴的名字，再看轴有多长。

| 符号 | 含义 | 本章小例子 |
| :--- | :--- | :--- |
| $B$ | batch size，一批有多少条序列 | 2 |
| $T$ | sequence length，每条序列有多少个 token 位置 | 3 |
| $D$ | model dimension，每个位置的总特征宽度 | 8 |
| $H$ | head count，注意力头数 | 2 |
| $d_h$ | head dimension，每个头的特征宽度 | 4 |

本章约定：

$$D = H d_h, \qquad d_h = D / H$$

因此 $D$ 必须能被 $H$ 整除，而且二者都应是正整数。`D=8, H=2` 合法，`D=8, H=3` 就不能按本章方式均分。这里 $T$ 数的是 **token 位置**，不等于汉字数或自然语言的词数。

### 两种等价的写法

在数学上，可以为第 $h$ 个头分别写出：

$$Q_h = XW_h^Q, \qquad K_h = XW_h^K, \qquad V_h = XW_h^V$$

其中每个 $W_h^Q, W_h^K, W_h^V$ 的形状都是 $[D, d_h]$。注意左边乘的是**完整的输入 $X$**，所以每个头的投影都能利用输入的全部 $D$ 个特征，而不是只看原始输入中的某一小段。

实现时，把各头投影沿输出特征方向并排拼成一个 $[D,D]$ 矩阵，就不必写 $H$ 个 Python 循环：

$$W_Q = [W_1^Q\; W_2^Q\; \cdots\; W_H^Q]$$

一次 `W_q(x)` 得到 `[B,T,D]` 的全部查询，再把最后的 $D$ 个特征整理为 `[H,d_h]`。K/V 同理。**不同头的可学习差异来自投影参数，reshape 本身不学习任何东西。**

数学里采用行向量右乘矩阵的记法；PyTorch `nn.Linear` 则把权重存为 `[out_features,in_features]`，计算相当于 `x @ weight.T`。不要混淆这两种记法。

![图 4-1：多头注意力空间切分与拼接流程](assets/diagrams/multi_head_attention.svg)

**读图提醒：**图中 $D=512$、4 个头，每头宽 128；8 个头则每头宽 64。图省略了投影细节，实际顺序是“完整输入 → 线性投影 → 分头”。图里的“专家”是类比，“可能关注语法”不是保证或预设标签。

---

## 4.3 张量账本：每一步到底换了什么

我们故意选择 $T=3$、$d_h=4$，让“序列长度”和“单头宽度”不相等。这样转错轴时，更容易从形状发现问题。

### 表 4-1：从输入到输出的完整形状账本

| 阶段 | 一般形状 | $B=2,T=3,D=8,H=2$ 时 | 这一步在做什么 |
| :--- | :--- | :--- | :--- |
| 输入 `x` | `[B, T, D]` | `[2, 3, 8]` | 两条序列，每个位置 8 维 |
| 投影后 Q/K/V，各一份 | `[B, T, D]` | `[2, 3, 8]` | 改变特征，不改变位置数 |
| 拆开特征轴 | `[B, T, H, d_h]` | `[2, 3, 2, 4]` | 把每个位置的 8 维分为 2 组 |
| 换轴后 Q/K/V，各一份 | `[B, H, T, d_h]` | `[2, 2, 3, 4]` | 让 batch 和 head 成为前导轴 |
| 仅转置 K 的最后两轴 | `[B, H, d_h, T]` | `[2, 2, 4, 3]` | 为点积对齐特征轴 |
| 分数 `scores` | `[B, H, T, T]` | `[2, 2, 3, 3]` | 行是 query 位置，列是 key 位置 |
| 掩码后的分数 | `[B, H, T, T]` | `[2, 2, 3, 3]` | 禁止某些连接，形状不变 |
| 权重 `attn_weights` | `[B, H, T, T]` | `[2, 2, 3, 3]` | 沿最后的 key 轴归一化 |
| 各头输出 `head_outputs` | `[B, H, T, d_h]` | `[2, 2, 3, 4]` | 每个头对 V 加权求和 |
| 换回位置优先 | `[B, T, H, d_h]` | `[2, 3, 2, 4]` | 把同一位置的各头放在一起 |
| 拼接 `out_concat` | `[B, T, D]` | `[2, 3, 8]` | 合并 H 与单头宽度两根轴 |
| 输出投影 `W_o` | `[B, T, D]` | `[2, 3, 8]` | 混合各头特征，得到最终结果 |

`[B,H,T,T]` 存放**位置之间的权重**，`[B,H,T,d_h]` 存放**各位置的特征**。即使 $T=d_h$，含义也不同。

### 第一步：reshape 是拆轴，transpose 是换轴

假设 `Q_full` 是投影后的 `[B, T, D]` 张量：

```python
Q_split = Q_full.reshape(B, T, H, d_h)
assert Q_split.shape == (B, T, H, d_h)

Q = Q_split.transpose(1, 2)
assert Q.shape == (B, H, T, d_h)
```

`reshape` 把一个长度为 $D$ 的特征轴拆成两个轴，元素总数必须相同；`transpose(1, 2)` 则交换第 1 和第 2 根轴，轴从 0 开始编号。这两种操作**不能互相替代**。

用整数给元素编号最容易看清。以下数字只用于追踪位置，不代表真实词向量：

```python
import torch

z = torch.arange(24).reshape(1, 3, 8)
split = z.reshape(1, 3, 2, 4)
heads = split.transpose(1, 2)

print(heads[0, 0])  # 第 0 条序列，第 0 个头
# tensor([[ 0,  1,  2,  3],
#         [ 8,  9, 10, 11],
#         [16, 17, 18, 19]])

print(heads[0, 1])  # 同一条序列，第 1 个头
# tensor([[ 4,  5,  6,  7],
#         [12, 13, 14, 15],
#         [20, 21, 22, 23]])
```

每个头都保留了三个 token 位置。如果直接做 `z.reshape(1,2,3,4)`，最终形状虽相同，第一个头拿到的却是连续的 0 到 11，混淆了位置与头的归属。**形状正确，不代表元素排布正确。**

### 第二步：为什么把 H 放到前面

对这里使用的高维张量，`torch.matmul` 把最后两根轴当作矩阵轴，其余轴用作批量维度。于是：

```text
Q:                         [B, H, T, d_h]
K.transpose(-2, -1):       [B, H, d_h, T]
                                    ↓ 消去 d_h
scores:                    [B, H, T, T]
```

我们得到每条序列、每个头各自的 $T\times T$ 打分表。不是在不同头之间做点积，也不是在两条序列之间交换信息。

```python
scores = Q @ K.transpose(-2, -1) / math.sqrt(d_h)
assert scores.shape == (B, H, T, T)
```

缩放分母是 $\sqrt{d_h}$，不是 $\sqrt{D}$，因为点积只对 $d_h$ 个特征求和。归一化沿最后的 key 轴做；有掩码时先处理掩码，规则见下一节。

加权汇总又是一次矩阵乘法：

```text
attn_weights:              [B, H, T, T]
V:                         [B, H, T, d_h]
                                    ↓ 对 key 位置加权求和
head_outputs:              [B, H, T, d_h]
```

### 第三步：拼接不是求和，也不是再做一次注意力

要把同一个位置的各头输出放在一起，先换回 `[B,T,H,d_h]`，再把后两根轴合并：

```python
head_outputs = attn_weights @ V
assert head_outputs.shape == (B, H, T, d_h)

by_token = head_outputs.transpose(1, 2)
assert by_token.shape == (B, T, H, d_h)

out_concat = by_token.contiguous().view(B, T, D)
assert out_concat.shape == (B, T, D)

output = W_o(out_concat)
assert output.shape == (B, T, D)
```

如果某个 token 的两个头分别输出 `[a,b,c,d]` 和 `[e,f,g,h]`，拼接就是 `[a,b,c,d,e,f,g,h]`，不是把两份向量相加成 4 维。之后的 $W_O$ 才是可学习的特征混合层，它仍对每个位置单独作用，不再次混合不同 token。

为什么有 `contiguous()`？换轴后的普通张量通常仍共享原存储，但布局可能不再满足 `view` 的要求，需要先整理成连续布局。也可以写 `by_token.reshape(B,T,D)`，必要时由它复制数据。形状变换不一定零开销；`contiguous()` 也不能修好转错轴的逻辑。[来源：PyTorch reshape](https://docs.pytorch.org/docs/2.14/generated/torch.reshape.html)、[transpose](https://docs.pytorch.org/docs/2.14/generated/torch.transpose.html)

---

## 4.4 多了头，掩码规则没有变

多头把分数扩展成 `[B,H,T,T]`，不会自动阻止模型看未来。

**本章配套代码的布尔 mask 统一使用 `True = 允许访问`，`False = 禁止访问`。**例如三个 token 的因果掩码是：

```text
              key 0   key 1   key 2
query 0       True    False   False
query 1       True    True    False
query 2       True    True    True
```

可以这样建立一个供所有 batch、所有头共用的 `[T,T]` 掩码：

```python
allowed = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
output, weights = mha(x, mask=allowed)
```

广播是在缺省轴上复用规则。不同序列可用 `[B,1,T,T]` 表达各自的规则，再供各头共用。不要把 `[B,T]` 的有效 token 标记直接当成注意力 mask：先明确它控制 key 列、query 行，还是二者。屏蔽 padding key 不自动保证 padding query 输出为零。

### 全被挡住的一行怎么办

对于至少有一个允许位置的行，禁止位置的分数设为负无穷，Softmax 后权重为零，允许位置的权重之和为 1。

但如果某个 query 在某个头中**一个 key 都不能看**，就没有合法的概率分布。把这一整行都设为负无穷后直接 Softmax，会遇到未定义的归一化；把它们都改成 `-1e9` 也不是解决办法，因为相同分数会变成均匀权重，反而读到了被禁止的值。

本章实现采用明确的工程约定：

1. 先找出全被屏蔽的行，让它们以安全的临时分数经过 Softmax，避免出现 `NaN`。
2. 再把这些行的**所有权重置零**，该头的加权输出因此为零。
3. 其余非空行正常归一化。

全屏蔽行的权重和是 0，不是 1。若一个 query 在**所有头**中都被全屏蔽，拼接向量为零；本例 `W_o` 无 bias，最终输出也为零。只屏蔽一个头则未必如此。若输出层有 bias，结果可能等于 bias；更不能据此推断后续带残差的整个模块为零。

最后注意一个容易踩到的 API 陷阱：本书的 allowed mask 与 PyTorch `scaled_dot_product_attention` 的布尔 mask 一样，`True` 表示参与；但 `nn.MultiheadAttention` 的布尔 `attn_mask` 和 `key_padding_mask` 使用 `True` 表示屏蔽。切换接口时要检查语义和形状，不能原样照搬。[来源：PyTorch SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)、[MultiheadAttention](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MultiheadAttention.html)

---

## 4.5 算清三本账：参数、运算和中间张量内存

“头数变成 8，开销是否变成 8 倍？”不能只回答“是”或“不是”。先固定比较条件：**总宽度 $D$ 不变，$H$ 个头等宽，$d_h=D/H$，Q/K/V 与输出都投影到 $D$，四个线性层都不带 bias。**单头对照也保留同样的四个投影。

### 参数账：标准无偏置 MHA 是 $4D^2$

Q、K、V 各有一个 $D\times D$ 投影，最后再加一个 $D\times D$ 的输出投影：

$$N_{\mathrm{params}} = D^2+D^2+D^2+D^2 = 4D^2$$

按头拆开数也一样：Q 投影共有 $H\cdot Dd_h=D^2$ 个参数。$D=8$ 时，四个矩阵共 256 个参数；改成 1、2、4 或 8 个头，都不改变这一数量。

这只是**注意力模块**的参数，不含 embedding、FFN、归一化等。若四个投影都加上长度为 $D$ 的 bias，还需加 $4D$。若固定的是每头宽度 $d_h$，增加头数会让 $D=Hd_h$ 一起增大，当然就不能再套用“参数不变”的结论。

### 运算账：投影成本不能漏掉

一次完整自注意力前向的主要矩阵乘法可以分成两类：

- **投影：**共有 $BT$ 个位置，每个位置做 $D$ 到 $D$ 的变换。Q/K/V 和输出投影合计是 $O(BTD^2)$。
- **头内注意力：**每个头的 $QK^\top$ 与权重乘 V，各需 $O(BT^2d_h)$；所有头加起来是 $O(BHT^2d_h)=O(BT^2D)$。

所以在上述条件下，主要计算量为：

$$O(BTD^2 + BT^2D)$$

掩码和 Softmax 还要处理 $BHT^2$ 个元素。在本章 $1\leq H\leq D$ 的等宽设置下，这不改变上述总体大 O 阶数，但会影响实际成本。大 O 隐去了常数、内存访问和执行方式，并不承诺两个实现耗时相同。

原论文在每头宽度随头数缩小的条件下，比较的是与全宽单头相近的计算成本；它不是“加头必然提速”的定理。实际耗时还依赖硬件、数据类型、矩阵尺寸、内存布局和具体计算内核。CPU 教学实现可用于验证数值，不足以证明 GPU 性能。[来源：Attention Is All You Need，§3.2.2](https://arxiv.org/html/1706.03762v7#S3.SS2.SSS2)

### 内存账：分数表仍有 $H$ 份

这里最容易漏算。Q/K/V 每份共有 $BHTd_h=BTD$ 个元素，但**分数表没有 $d_h$ 这一根轴**：它的形状是 `[B,H,T,T]`，所以需要 $BHT^2$ 个元素。

例如 $B=1,H=8,T=1024$，一份 float32 分数张量就占：

$$1\times8\times1024^2\times4\ \text{bytes}=32\ \text{MiB}$$

这里 $1\ \text{MiB}=2^{20}$ 字节。这还不含权重张量、Q/K/V、输出以及训练时保存的中间量。固定 $B,T,D$ 增加头数，显式保存分数或权重表的开销会随 $H$ 增加；把序列长度翻倍，则这部分元素数变成四倍。

这个内存账本针对**本章显式构造分数和权重的实现**。优化内核可能不把完整分数表留在显存中，所以它不是所有实现的峰值内存下界。FlashAttention 一章会展开这个区别。

还有一个以后会用到的区别：标准 MHA 的 K 和 V 总元素数是 $2BHTd_h=2BTD$，并不是 $2BHT^2$。后续 KV Cache 存的是 K/V 特征，不是所有历史注意力分数；现在先把这两类张量分清。

---

## 4.6 动手实验：先验形状，再验数值

在仓库根目录运行：

```bash
python3 code/04_multi_head_attention.py
python3 -m unittest discover -s tests -v
```

示例使用小尺寸输入打印形状和各头的权重。阅读时先把日志对照表 4-1，再看权重：同一个 query 的哪一行、哪一个头、对哪些 key 求和？不要只因为数字看起来合理，就认定实现正确。

演示里的“机器”“学习”“真”“酷”只是位置标签，输入和投影都是随机的。不同百分比不证明语义分工；固定种子便于复查，也不保证跨设备、版本逐位相同。

### 实验一：退化成一个头，必须等于直接的参考公式

将 $H$ 设成 1 后，$d_h=D$。此时分头不应改变计算含义。我们不用重新初始化另一套投影，而是读取**同一个模型的参数**，手动计算单头公式，再和模块输出比较。

下面的代码可以在仓库根目录的 Python 会话中执行：

```python
import math
import runpy
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path("code").resolve()))
module = runpy.run_path("code/04_multi_head_attention.py")
SimpleMultiHeadAttention = module["SimpleMultiHeadAttention"]

torch.manual_seed(100)
B, T, D = 2, 3, 8
x = torch.randn(B, T, D)
mha = SimpleMultiHeadAttention(d_model=D, num_heads=1)

with torch.no_grad():
    actual, actual_weights = mha(x)
    q, k, v = mha.W_q(x), mha.W_k(x), mha.W_v(x)
    reference_weights = torch.softmax(
        q @ k.transpose(-2, -1) / math.sqrt(D), dim=-1
    )
    reference = mha.W_o(reference_weights @ v)

assert actual.shape == (B, T, D)
assert actual_weights.shape == (B, 1, T, T)
torch.testing.assert_close(actual, reference)
torch.testing.assert_close(actual_weights[:, 0], reference_weights)
print("单头退化与直接公式一致")
```

为什么还要保留 `W_o`？因为我们比较的是**完整模块**，不是只比较注意力核心。一个有输出投影、另一个没有，即使头数都是 1，也不是同一个计算。

这能发现分头、缩放、拼接等错误，但还需检查多个头、不同 $B,T,D$ 与掩码边界。

### 实验二：全屏蔽行不能偷偷变成均匀分布

继续沿用前面的导入，再创建两个头。把最后一个 query 的所有 key 都挡住：

```python
mha = SimpleMultiHeadAttention(d_model=D, num_heads=2)
allowed = torch.ones(T, T, dtype=torch.bool).tril()
allowed[-1, :] = False

with torch.no_grad():
    output, weights = mha(x, mask=allowed)

assert weights.shape == (B, 2, T, T)
assert torch.isfinite(weights).all()
assert torch.isfinite(output).all()
assert torch.count_nonzero(weights[:, :, -1, :]).item() == 0
assert torch.count_nonzero(output[:, -1, :]).item() == 0

torch.testing.assert_close(
    weights[:, :, :-1, :].sum(dim=-1),
    torch.ones(B, 2, T - 1),
)
print("非空行归一化，全屏蔽行权重与输出为零")
```

这检查的是本章的空行约定，不是说零权重是一种概率分布。真实任务仍要追查空行来源，以及 padding 位置是否应排除在损失之外。

还可以把 `H` 改成 4，检查输出形状仍为 `[B,T,D]`，权重形状则变为 `[B,4,T,T]`；再试 `H=3`，确认不合法的配置会被拒绝。固定 $D=8$ 时，`sum(p.numel() for p in mha.parameters())` 应为 256。这些检查分别针对形状、输入约束和参数账，不要混成一个“能运行就通过”的标准。

---

## 4.7 自测：遮住答案再读一遍账本

1. $B=2,T=5,D=12,H=3$ 时，分头后的 Q、分数表和拼接结果分别是什么形状？
2. 为什么 `reshape(B,H,T,d_h)` 不能直接替代“先拆成 `[B,T,H,d_h]`，再换轴”？
3. 标准 MHA 从 2 个头改成 4 个头，固定 $D$，参数量是否翻倍？显式权重表是否变大？
4. 两个随机初始化的头得到不同的权重表，能否据此宣布一个懂语法、另一个懂情感？
5. 一个 query 只在其中一个头中被全屏蔽，最终输出一定是零吗？

**参考答案：**

1. $d_h=4$；Q 是 `[2,3,5,4]`，分数是 `[2,3,5,5]`，拼接结果是 `[2,5,12]`。
2. reshape 只按元素顺序改变形状；换轴改变的是元素所属的轴位置。最终尺寸相同，元素归属仍可能不同。
3. 四个无偏置投影仍为 $4D^2$；固定 $B,T$ 时，权重表从 $2BT^2$ 增至 $4BT^2$ 个元素。
4. 不能。不同数字没有证明语义分工，更没有证明对模型决策的因果解释。
5. 不一定。其他头仍可能产生非零特征，随后被 $W_o$ 混合。

---

## 4.8 本章小结与下一步

- 多头的核心是**不同的可学习投影、分别归一化的权重、拼接后的输出投影**，不是把一张表复制多遍，也不是预先指定语义专家。
- 记住主路径：`[B,T,D] → [B,T,H,d_h] → [B,H,T,d_h] → [B,H,T,T] → [B,H,T,d_h] → [B,T,D]`。
- reshape 拆轴，transpose 换轴，concat 合并各头特征。写对形状之后，还要检查元素是否去了正确的位置。
- 本章 mask 使用 `True=允许`；非空行权重和为 1，全屏蔽行按明确约定置零。
- 固定 $D$、标准等宽、无偏置设置下，参数是 $4D^2$，主要计算是 $O(BTD^2+BT^2D)$；显式分数存储仍有 $BHT^2$ 个元素。相同计算阶数不意味着相同速度或内存。

多头本身不添加“我在第几个位置”的信息。没有位置编码、也没有依赖位置的掩码时，重排输入 token，输出只会跟着重排。因果掩码限制可见方向，但与为表示加入位置信息不同。下一章继续讨论**位置编码（Positional Encoding）**。

文档核验日：2026-10-19，来源见相关段落。文档版本不等于本书的完整兼容性测试范围；本章实验验证教学计算，不提供未测的速度或显存提升比例。
