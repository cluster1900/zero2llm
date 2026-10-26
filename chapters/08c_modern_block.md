# 现代 Block（P13）：RMSNorm 与 SwiGLU {#p13-modern-block}

[P12 的 S2](#p12-rope) 已经把位置处理换成 RoPE，但 Block 内部仍然是 LayerNorm 和 GELU FFN。本章得到 S3：**保留残差路线、因果注意力和 RoPE，只研究归一化与 FFN**。不同时加入 GQA，不借“现代”二字暗示每种设置都更好。

配套 [code/14_modern_block.py](../code/14_modern_block.py) 在 S2 类上做替换，不复制另一套注意力代码。你会先算两行小向量，再检查真实参数量，最后做两个分开的消融实验。初次接触这些名词时，请先回顾 P07 的 Pre-LN 和逐位置 FFN，以及 P09 的前向、反向和 optimizer。

## P13.1 先认出没有变的骨架

一个 Block 仍然做两次“先归一化、再计算、加回主干”：

$$u=x+\operatorname{Attention}(\operatorname{Norm}_1(x))$$
$$y=u+\operatorname{FFN}(\operatorname{Norm}_2(u))$$

Norm 和 FFN 都分别作用于每个 token 的最后一维。它们本身不把不同位置混起来；跨位置读信息的是 Attention。输入和输出都是 `[B,T,D]`，所以残差相加仍然合法。这里 B 是 batch，T 是序列长度，D 是主干宽度。

![S3 主干与 SwiGLU 两路门控：注意力不改，FFN 内部宽度可以改](assets/diagrams/p13-modern-block.svg)

图中“×”是逐坐标相乘，不是矩阵乘法。FFN 内部临时变宽，输出投影再回到 D；它没有偷偷改变下一个 Block 的输入宽度。模型最后的 `ln_f` 也要跟随归一化选择，不能只改每个 Block 的两处 Norm，漏掉输出头之前的一处。

## P13.2 LayerNorm 与 RMSNorm 究竟除以什么

先取某个 token 的向量 $x=[x_1,\ldots,x_D]$。为看清区别，暂时令可学习缩放 $\gamma$ 全为 1，偏移 $\beta$ 全为 0。

LayerNorm 先减均值，再除以标准差：

$$\mu=\frac{1}{D}\sum_i x_i,\qquad v=\frac{1}{D}\sum_i(x_i-\mu)^2$$
$$\operatorname{LN}(x)_i=\frac{x_i-\mu}{\sqrt{v+\epsilon}}\gamma_i+\beta_i$$

这里方差除以 D，不是统计学无偏估计的 D−1。PyTorch LayerNorm 对应 `correction=0`。它在训练和推理时都使用当前输入的统计量，没有 BatchNorm 那种运行均值表。[来源 3]

RMS 是 root mean square，中文是“均方根”：先平方、求平均，再开平方。RMSNorm **不减均值**：

$$r^2=\frac{1}{D}\sum_i x_i^2,\qquad
\operatorname{RMSNorm}(x)_i=\frac{x_i}{\sqrt{r^2+\epsilon}}\gamma_i$$

本章实现只有 D 个可学习的 $\gamma$，没有 $\beta$。它把向量的整体尺度调到较稳定的范围，却不强制把中心搬到 0。RMSNorm 原论文研究的正是省去重新居中后的归一化方式；论文中的实验结果不能直接成为我们这份 CPU 实现的速度保证。[来源 1]

一个很有用的恒等式是 $r^2=v+\mu^2$。例如 `[1,3]` 的均值是 2，方差是 1，均方是 5，因此 $5=1+2^2$。RMSNorm 不是“LN 换个名字”：只要均值不是 0，两个分母一般不同，LN 的分子也还减了均值。

### 表 P13-1：两行向量，逐项手算

为了使分母容易核对，这张教学表故意用 **ε=0.5**，不是模型训练默认的 `1e-5`。每一行独立归一化，γ=1，β=0。

| 输入 x | 均值 μ | 方差 v | 均方 r² | LN 分母 | RMS 分母 | LN 输出 | RMSNorm 输出 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [1,3] | 2 | 1 | 5 | √1.5 | √5.5 | [-0.816497,0.816497] | [0.426401,1.279204] |
| [3,5] | 4 | 1 | 17 | √1.5 | √17.5 | [-0.816497,0.816497] | [0.717137,1.195229] |

第二行等于第一行的每个坐标都加 2。LN 把这次整体平移消掉，RMSNorm 则保留差异。以第一行 RMS 输出为例，`1/√5.5≈0.426401`，`3/√5.5≈1.279204`，输出的均值明显不是 0。

如果学到 γ=[2,0.5]，还要把两个输出坐标分别乘 2 和 0.5；γ 不是整条向量共用一个数。LN 在 γ 不同或 β 非零时，其最终输出也不必保持均值 0，不能把“标准化后的中间量”与“可学习仿射变换后的输出”混为一谈。

`norm_walkthrough()` 产生表中的原始张量；测试独立使用两种平方根公式核对，不是拿同一个实现与自己比较。

## P13.3 ε 是数值约定，不是可省略的装饰

若 x 全为 0，直接除以 RMS 会出现 0/0。加上正数 ε 后，分母至少是 √ε，输出仍为 0，反向也可保持有限。但 ε 很小时，零点附近导数可能很大；“有限”不等于“绝不会梯度爆炸”。

位置也很重要：

$$\sqrt{\operatorname{mean}(x^2)+\epsilon}\ \ne\ \sqrt{\operatorname{mean}(x^2)}+\epsilon$$

右边不是本章实现。表 P13-1 的较大 ε 能让这类错误很容易暴露，生产配置的小 ε 反而可能让粗糙测试漏掉它。`norm_eps` 必须是有限正数；0、负数、NaN、无穷和布尔值都被拒绝。

再看“缩放不变”这句话：当忽略 ε 且乘数 a>0 时，RMS(ax)=a·RMS(x)，因此归一化可抵消这种正的整体缩放。保留 ε 后只能在 ε 相对均方很小时近似成立；a<0 还会反转符号。不要把论文里的理想化推导写成任意输入、任意精度下的逐位相等保证。

### 低精度为什么先升到 float32

float16 能表示 40000 左右的数，但它的平方已经远超 float16 范围。错误地先在 float16 中平方，之后再转换 float32，已经救不回无穷了。源码先转换，再做平方、均值和倒平方根：

```python
work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
inv_rms = torch.rsqrt(work.square().mean(dim=-1, keepdim=True) + self.eps)
return (work * inv_rms * self.weight.to(work.dtype)).to(x.dtype)
```

这是源码原句。float32 输入保持 float32；float64 输入保留 float64，而不是为了“统一精度”反而降成 float32。最后返回输入 dtype，方便后续残差计算。γ 的转换仍在自动求导图里，并没有切断梯度。

我们的 `eps=1e-5` 是明确写入实验的选择，不依赖库的默认值。PyTorch 2.14 的 RMSNorm 在 `eps=None` 时按计算类型选择机器 epsilon；比较第三方实现时，应把 ε、统计计算 dtype 和最终输出 dtype 一并核对。[来源 4] 此实现没有宣称能处理任意大的 float32 数，也没有替你实现 AMP 或 fused kernel。半精度测试只验证这个局部数值策略，整模型消融仍在 CPU float32 下运行。

## P13.4 从 GELU FFN 到“内容乘以门”

旧 FFN 是两次线性投影，中间加 GELU：先 D→H，再 H→D。默认 H=4D。它在同一个 token 的特征间重组信息，不是沿时间轴做卷积。

SwiGLU 在同一个输入上走两条 D→M 支路。一条产生内容 `up(x)`，另一条经过 SiLU 后调节内容 `gate(x)`，最后通过 `down` 回到 D：

$$g=xW_g,\quad u=xW_u,\quad z=\operatorname{SiLU}(g)\odot u,\quad y=zW_d$$
$$\operatorname{SiLU}(t)=t\sigma(t)=\frac{t}{1+e^{-t}}$$

这里公式采用行向量右乘矩阵的记法，PyTorch `nn.Linear` 则储存转置方向的权重。`gate.weight` 与 `up.weight` 的形状是 `[M,D]`，`down.weight` 是 `[D,M]`。源码只有一行核心计算：

```python
return self.down(F.silu(self.gate(x)) * self.up(x))
```

SiLU 是 Swish 在参数 β=1 时的形式。GLU 变体论文将这类两路相乘的结构用于 Transformer FFN；SwiGLU 对门支路使用 Swish，而不是把 FFN 中的 GELU 单独换成 SiLU 就结束。[来源 2、5]

把 gate 类比成音量旋钮有助于记忆，但类比不能过头：**SiLU 的输出不是概率，不限于 [0,1]，可以为负，也可以大于 1**。SwiGLU 没有在若干专家中做 top-k 选择，也不是后续 MoE 的路由器。两条支路都计算，每个中间坐标都相乘。

### 表 P13-2：真正走完两路投影

令输入 x=[1,2]，选以下 PyTorch 权重，没有 bias，down 是单位矩阵：

```text
gate.weight = [[1,-1], [0,1]]
up.weight   = [[1, 1], [2,1]]
down.weight = [[1, 0], [0,1]]
```

于是 gate 的第一坐标是 1×1+2×(−1)=−1，第二坐标是 1×0+2×1=2；up 的两坐标分别是 3 和 4。

| 中间坐标 | gate 投影 g | sigmoid(g) | SiLU(g)=g×sigmoid(g) | up 内容 u | 相乘结果 z |
| --- | --- | --- | --- | --- | --- |
| 0 | -1 | 0.268941 | -0.268941 | 3 | -0.806824 |
| 1 | 2 | 0.880797 | 1.761594 | 4 | 7.046377 |

down 是单位矩阵，所以输出恰好是 `[-0.806824,7.046377]`。如果误用 sigmoid 而不是 SiLU，第一项会变成正数；如果误把两路相加，也不会得到表中的结果。`gate_walkthrough()` 打印每一层，测试把最终结果独立写成 `-3/(1+e)` 和 `8/(1+exp(-2))` 核对。

## P13.5 为什么不能把两种 FFN 都设置为 4D 宽

先全部关闭 bias，GELU FFN 有两个矩阵，参数数是：

$$N_{\text{GELU}}=D H+H D=2DH$$

SwiGLU 有三个矩阵，参数数是：

$$N_{\text{SwiGLU}}=D M+D M+M D=3DM$$

如果 H=M=4D，后者的矩阵参数多 50%。然后拿训练 loss 比较并宣布“门控优于 GELU”，就把增加参数的影响混进来了。令两者相等，可得 M=2H/3；当 H=4D 时，M≈8D/3。这种收窄门控 FFN 的比较思路也出现在 GLU 变体论文中。[来源 2]

### 表 P13-3：D=48 的每层 FFN 账本

| FFN | 中间宽度 | 矩阵参数计算 | 无 bias 参数 |
| --- | --- | --- | --- |
| GELU 基线 | H=192 | 2×48×192 | 18432 |
| SwiGLU，错误地保持同宽 | M=192 | 3×48×192 | 27648 |
| SwiGLU，匹配预算 | M=128 | 3×48×128 | 18432 |

这里第二行不是“非法模型”，而是不适合用来隔离门控结构效果的同预算对照。M 不一定能整除，硬件对齐也可能要求它是某个数的倍数。`matched_swiglu_width` 选最接近参数预算的合法正整数宽度，平局选较小值；如设置 `multiple_of`，要重新报告真实误差。

有 bias 时也别忘了数：GELU 多 H+D，SwiGLU 多 2M+D。上述宽度对应 18672 与 18736，不再严格相等，相差 64。源码的计数函数包含这两种情况，测试再用真实 `numel()` 交叉检查。参数数相同也不等于峰值激活内存、执行时间或训练质量相同：门控要保留额外支路中间量，kernel 和硬件的实现也不同。

## P13.6 S2 不被悄悄改写，S3 有明确配置

`ModernConfig` 扩展 S2 的 `RopeConfig`；词表、主干宽度、头数、层数、容量、RoPE base 和 rotary_dim 都继承。新增的开关是归一化类型、FFN 类型、ε、内部宽度与 bias 策略。

| 项目 | 原始 S2 | S3 默认 |
| --- | --- | --- |
| 注意力与位置 | MHA + RoPE | 完全保留 |
| Norm | LayerNorm，γ 与 β | RMSNorm，只有 γ |
| FFN | GELU，H=4D，有 bias | SwiGLU，预算匹配 M，无 bias |
| 残差顺序 | Pre-Norm | 完全保留 |
| cache 路径 | S2 的 cached_forward | 调用同一函数 |

同时改变 Norm、FFN 和 bias 才得到这份默认 S3，所以 **“默认 S2 对默认 S3”不能称为单变量消融**。默认 S3 也不是把一个已训练 S2 checkpoint 无损升级后的模型；结构不同的参数需要合适训练。

为守住旧行为，测试还构建一个兼容设置：`norm_kind='layernorm'`、`ffn_kind='gelu'`、`norm_bias=True`、`ffn_bias=True`、ε=1e-5、H=4D。它在相同 seed 下与 S2 的每个 state tensor、每个输出 logit 精确相同。S2 自己的默认配置和代码没有因本章而变成 RMSNorm。

## P13.7 单变量消融，先控制什么再比较什么

先建立一个**公开说明的无 bias 对照 A**：LayerNorm 保留 γ，去掉 β；FFN 的两个线性层也去掉 bias。这是实验准备，不是归一化或门控胜出的证据。随后只比较相邻两行：

| 实验 | Norm | FFN | FFN 宽度 | 相对上一行的研究变量 |
| --- | --- | --- | --- | --- |
| A | LayerNorm，无 β | GELU，无 bias | 192 | 无 bias 控制组 |
| B | RMSNorm | GELU，无 bias | 192 | 归一化方式 |
| C | RMSNorm | SwiGLU，无 bias | 128 | 同预算 FFN 结构 |

A→B 的 ε、γ、FFN、注意力等都固定，比较的是居中与均方归一化机制。B→C 固定所有非 FFN 部件；FFN 结构与它所需的预算匹配宽度一起作为处理项。**这不是“只改一个激活函数”**，不能把结论归因于 SiLU 单独的效果。

`make_ablation_models()` 除了设置同一 seed，还真的调用 `copy_unchanged`：A→B 复制所有非 Norm state tensor，并复制对应 γ；B→C 复制全部非 FFN state tensor。它检查 key、shape 和数值一致，不通过参数共享来冒充复制。因此训练 A 不会顺便更新 B。新 FFN 形状不同，不能要求它与旧 FFN 逐项同值；用相同初始化规则与 seed 生成，并明确它是被研究的组件。

仅写两次 `manual_seed(42)` 不够：如果第一个模型多初始化一层，第二个模型后续参数的随机数消耗顺序就可能不同。**复制并检查不变部分**才是这里可执行的控制条件。

### 整模型账本：为什么三行都是 61008

设置 V=57、D=48、L=2、H=4，embedding 与输出 head 不共享权重。每个 Block 两个 Norm，加最后一个 Norm，共 5 个。

| 部分 | 参数计算 | 参数数 |
| --- | --- | --- |
| token embedding + 输出 head | 2×57×48 | 5472 |
| 两层 MHA 投影 | 2×4×48² | 18432 |
| 两层 FFN，按表匹配 | 2×18432 | 36864 |
| 5 个仅有 γ 的 Norm | 5×48 | 240 |
| 合计 | 5472+18432+36864+240 | 61008 |

原 S2 默认是 61728 个参数。对照 A 去掉两层 FFN 的 2×(192+48)=480 个 bias，再去掉 Norm 的 5×48=240 个 β，合计少 720，得到 61008。RoPE 没有可学习位置表，不能把 S0 的那张位置表又算进来。

## P13.8 实际跑一次，但不要把 smoke 当榜单

在仓库根目录安装固定依赖后执行：

```bash
python -m pip install -r requirements.txt
OMP_NUM_THREADS=1 python code/14_modern_block.py
OMP_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_modern_block.py' -v
```

没有 PyTorch 时，CLI 会简短提示安装方式并以退出码 2 结束。张量测试使用仓库共享的稳定版 torch 2.14.x 检查；未安装或版本不匹配会明确 skip，**skip 不是通过**。不需要下载模型或数据。

固定实验条件：CPU、PyTorch 2.14.1+cpu、float32、单线程、seed=42，B=2、T=12、8 次更新。数据用单独的 generator，seed=43，先生成长度 T+1 的 ID，再取相邻输入/目标。三行模型使用同一份 batch、同一输入顺序和步数，每次都创建全新的 AdamW，lr=0.001、betas=(0.9,0.999)、eps=1e-8、weight_decay=0，无 dropout。

### 表 P13-4：这一次实跑的训练闭环结果

| 配置 | 每层 FFN 参数 | 整模型参数 | 更新前 loss | 8 次更新后 loss | 所有参数梯度有限 |
| --- | --- | --- | --- | --- | --- |
| A：LN + GELU | 18432 | 61008 | 4.429393 | 2.889954 | True |
| B：RMSNorm + GELU | 18432 | 61008 | 4.441514 | 2.896931 | True |
| C：RMSNorm + SwiGLU | 18432 | 61008 | 4.403015 | 3.212187 | True |

这几行表明：配置有实数结果、梯度能传到全部参数、优化器确实更新、参数预算确实匹配。它们**不能证明任何一种结构在语言任务上更优**。这里只有一个 seed、一份反复训练的合成 batch，没有验证集、自然语言语料或质量指标；C 的训练 loss 更高也不证明 SwiGLU 一般更差。本章不计时，因此也不从这些结果推断速度。

若真要研究质量，应沿用 P09 的训练/验证分离，固定数据与 token 预算，在多个 seed 下记录验证 loss 的分布，并把超参数搜索预算也控制住。不要把这里的八步示例包装成复现原论文实验。

## P13.9 换了组件，旧的 cache 正确性仍要重测

归一化和 FFN 只处理当前 token，不改变 P11 的因果缓存理由。但每层缓存的 K/V 已经受到之前层的 Norm 与 FFN 影响。若完整前向使用 RMSNorm，而缓存路径偷偷硬编码了 LayerNorm，两条路径会计算不同的模型。

S3 直接复用 S2 的 `cached_forward`。这个函数访问当前 Block 的 `ln1`、`ln2` 和 `mlp`，因此自然调用被替换后的模块；没有另写一份旧 FFN 藏在 decode 分支里。测试把同一串 token 分成 `[12]`、12 次单 token、`[3,5,4]` 和 `[11,1]`，拼接每块 logits，与同权重的完整前向逐项比较。B>1、float32 和 float64 都覆盖。

在表 P13-4 更新后的模型上，用长度 5 的块重放同一输入，本机实测三行 float32 最大 logit 误差均小于 1e-6；末位可随硬件与算子实现变化，不作逐位快照。测试容差为 atol=3e-6、rtol=1e-5。缓存等价比较的是**同一个模型的两种计算路线**，不是要求 A、B、C 三个不同模型的输出彼此相等。

缓存仍然是每请求私有、eval-only，不接受权重更新后的旧 cache。配置、Norm 的 ε 或 GELU 近似模式变化也需要重置。手工改 Tensor 的 `.data` 会绕过版本追踪，仍不在支持范围；不要把这个教学缓存当成可以任意热替换参数的服务系统。

## P13.10 让测试真的能抓错

正常通过只能说明当前样例没发现问题。这里还在内存里编译几个错误版本，确认独立参考测试会失败；生产源码不会被改动。

| 故意制造的错误 | 会被什么抓住 |
| --- | --- |
| RMS 分母误用居中后的方差 | 非零均值输入和独立均方参考 |
| ε 加在平方根外 | ε=0.5 的精确手算表 |
| SwiGLU 使用 sigmoid 而不是 SiLU | gate=-1、2 的符号与幅值 |
| 门与内容相加而非相乘 | 两路投影的最终输出参考 |
| 漏换最后的 ln_f，或忽略配置中的 ε | 三种配置下逐个检查 ln1/ln2/ln_f 的类型与 ε=1e-6 |
| 消融学习率由 0.001 改成 0.002 | 表 P13-4 的八步 loss 快照，绝对容差 2e-5 |
| 预算宽度平局时误选较大值 | D=2、H=4、有 bias 的等距例子及独立穷举参考 |

FFN 中间宽度，以及表 P13-4 的每层 FFN 参数和整模型参数，也由实际运行结果核对。γ 初值全为 1 时，省略对应复制可能不改变结果，不能把这种等价变异声称为已发现错误；另用非全 1 的 γ 验证复制助手的值与非共享存储约定。

还有 double 精度自动微分数值检查、全零与半精度大输入的有限反向、真实参数计数、显式复制后的数值一致性、S2 兼容模式及缓存对齐。测试一旦失败，应先定位公式、dtype、配置或共享状态，而不是扩大容差直到它变绿。

## P13.11 自测与参考答案

1. `[1,3]` 整体加 2 后，为什么 LN 表中不变，而 RMSNorm 变了？
2. 为什么“先 float16 平方，再转 float32”还会出错？
3. SwiGLU 的门输出为 1.76 是否违反概率范围？
4. D=48，GELU 的 H=192。无 bias 的 SwiGLU 应取多宽才能匹配矩阵参数？
5. 同一个 seed 是否足以证明消融只有一个变量变化？
6. 为什么表 P13-4 不能用来排“哪种 Block 最好”？
7. A→B 为什么把 LayerNorm 的 β 也关闭？
8. 改完 Norm 后只看生成出的一个 token 一样，能证明缓存等价吗？

参考答案：

1. LN 在两个输入上都减去各自均值，得到同样的 `[-1,1]`；RMSNorm 不居中，均方从 5 变 17，分子也改变。
2. 溢出在平方那一步已经发生，随后升精度只能保留无穷；必须先转换再平方。
3. 不违反。SiLU 门不是概率，可以大于 1、为负；sigmoid 本身才在 0 与 1 之间。
4. M=128，因为 2×48×192=3×48×128=18432。开启 bias 后要把 bias 额外计算，不能直接声称总参数严格相等。
5. 不够。初始化时随机数消耗顺序可能变化，必须复制并验证不变部分，同时控制数据、优化器、步数和其他设置。
6. 单 seed、重复 batch、无验证集，只能演示训练与数值闭环；没有足够证据做泛化质量或速度结论。
7. 避免把“多一组可训练偏移参数”也夹在归一化机制对比里。关闭 bias 是对照准备，不宣称它单独带来收益。
8. 不能。argmax 相同可掩盖很大的 logit 差异，应比较每个位置、每个词表项，并覆盖分块与参数变更边界。

## P13.12 本章带走什么

RMSNorm 的关键是**均方缩放，不居中**；SwiGLU 的关键是**两路投影，其中一路过 SiLU，再逐元素相乘**。数值稳定需要明确 ε 和计算精度；公平比较需要明确参数、bias、数据、初始化和优化器，不是换掉两个类名就算完成。

下一单元 [P14](#p14-grouped-kv) 保持 S3 的 Norm 与 FFN，再改变 KV 头的共享方式。这样每次升级都有明确的变化范围，不让“现代大模型积木”变成无法追踪的名词堆叠。

## P13.13 延伸：SiTU-GLU，给两个因子都装上软上限 {#p13.13-situ-glu}

先看动机。P13.4 的 SwiGLU 是 `Swish(W_g x) ⊙ W_u x`。两个因子都没有上界。两个大坐标碰在一起，乘积就是一个离群值，低精度下可能溢出。原始 GLU 的 sigmoid 门有界，但丢掉了 Swish 在正半轴近似线性的形状。

Kimi K3 报告（S29 §2.3.2，Eq.12）用软上限 `softcap(x, β) = β tanh(x/β)` 同时限制两个因子：

`SiTU-GLU(x) = [β₁ tanh(W_g x / β₁) ⊙ Sigmoid(W_g x)] ⊙ [β₂ tanh(W_u x / β₂)]`。

K3 取 `β₁ = 4`、`β₂ = 25`。所以 `|f| ≤ β₁β₂ = 100`。tanh 在原点附近近似线性，所以小输入时它和 SwiGLU 几乎一样。S29 的 Table 1 显示，K2 用 SwiGLU，K3 换成 SiTU-GLU。

[`../code/14b_attnres_situ.py`](../code/14b_attnres_situ.py) 的标量对照（g = u = x）：

| x | SwiGLU | SiTU-GLU |
|---|---|---|
| 0.1 | 0.0052 | 0.0052 |
| 1 | 0.7311 | 0.7158 |
| 10 | 99.9955 | 37.4846 |
| 100 | 10000.0000 | 99.9329 |

测试在 ±10⁴ 的输入上检查上界 100。它还检查原点附近两者相差小于 10⁻⁴。一个变异去掉 up 分支的上限，输出立刻超过 100。本书没有训练对比，不声称 SiTU-GLU 提高了质量。

## 来源与实现核对

1. [Root Mean Square Layer Normalization，Zhang 与 Sennrich，2019](https://arxiv.org/abs/1910.07467)：RMSNorm 的原始定义与研究；本章手算、dtype 约定和 CPU 测试为独立教学例子。
2. [GLU Variants Improve Transformer，Shazeer，2020](https://arxiv.org/abs/2002.05202)：GLU 变体及匹配预算时缩小中间宽度；本章不复现其 T5 训练结论。
3. [PyTorch 2.14 LayerNorm](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html)：方差口径、γ/β、ε 和 bias 参数。
4. [PyTorch 2.14 RMSNorm](https://docs.pytorch.org/docs/2.14/generated/torch.nn.RMSNorm.html)：公式与 epsilon 计算类型；本代码始终显式指定 ε。
5. [PyTorch 2.14 SiLU](https://docs.pytorch.org/docs/2.14/generated/torch.nn.SiLU.html)：SiLU 的逐元素定义。

6. [Kimi K3 技术报告，arXiv:2607.24653](https://arxiv.org/abs/2607.24653)（S29）：§2.3.2 Eq.12、β₁=4、β₂=25 与 Table 1 的激活函数一行；核验 2026-10-26。

来源核对日期：2026-10-19。实际实验由固定依赖环境运行；其它设备或库版本的最后几位可能不同，应先核对配置与测试，而不是把表中的舍入值当作所有环境的逐位承诺。
