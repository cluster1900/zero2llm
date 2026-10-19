# 架构进阶（P18）：MoE 的专家、路由与负载 {#p18-moe}

想象一间维修店：前台看过问题，把工单交给几位师傅；有些检查则每张工单都要做。店里可以有很多师傅，但一张工单未必请所有人动手。这是理解 **Mixture of Experts（MoE，混合专家）** 的起点。

不过，“专家”不是预先命名的数学老师、程序员或翻译官。它通常只是一组可训练的 FFN 权重；路由器根据当前 token 的隐藏向量选它。某个专家是否出现了可解释的分工，需要实验，不能由名字保证。多个 token 也可以同时选择同一专家，所以“前台把一个问题交给一个人”的类比很快就不够用了。

在 [P14 的 S4](#p14-grouped-kv) 中，我们已经有 RoPE、RMSNorm、SwiGLU、GQA 和 compact KV cache。本章回到主线，构造 **S5：只把 S4 的 FFN 换成教学 MoE**。P15–P17 的系统、稀疏注意力和 CED 独立实验没有偷偷混进来。

配套 [code/19_moe.py](../code/19_moe.py)，测试为 `tests/test_moe.py`。CPU 即可运行，不下载模型，不训练出专业分工，也不复现完整 DeepSeek。

## P18.1 先找准：换的是哪一块

回想 Pre-Norm Block 的两条残差支路：

$$
u=x+\operatorname{Attention}(\operatorname{Norm}_1(x)),\qquad
h=u+\operatorname{FFN}(\operatorname{Norm}_2(u)).
$$

S5 不改变第一条支路，只把第二式的 FFN 换成 MoE：

$$
h=u+\operatorname{MoE}(\operatorname{Norm}_2(u)).
$$

因此，MoE 本身不再加一次输入残差。注意力负责读取其他位置，MoE 在本层逐 token 加工已经得到的表示。换 FFN 不意味着换掉因果 mask，也不意味着取消 GQA 的 K/V 缓存。

![S5 教学 MoE：输入分为路由专家与共享专家两条路径，top-k 权重只组合被选专家，容量丢弃仅为独立实验，语言模型保持不丢弃](assets/diagrams/p18-moe.svg)

图中的 E 是可选的路由专家数量，k 是每 token 选择的数量，S 是始终参与的共享专家数量。**E 和 k 不能混写。** k=2、E=4 时，模型保存四个路由专家，单个 token 只调用其中两个。

## P18.2 路由器先打分，再选择，再加权

设一个 token 的输入是行向量 x，宽度为 D。教学路由器是一层不带 bias 的 Linear：

$$
z=xW_r^T,\qquad W_r\in\mathbb{R}^{E\times D},\qquad
p_e=\frac{\exp(z_e)}{\sum_j\exp(z_j)}.
$$

z 是分数，p 是本章选用的 Softmax affinity，意思是 token 对专家的相对偏好。它不是词表上的下一个 token 概率，维度 E 也不是词表大小 V。

接着取 p 最大的 k 个专家，记为集合 K。本章再把**所选专家**的权重归一化：

$$
w_e=\frac{p_e}{\sum_{j\in K}p_j}\quad(e\in K),\qquad
\operatorname{MoE}(x)=\sum_{e\in K}w_e F_e(x)+\sum_{s=1}^{S}F_s^{shared}(x).
$$

没有被选中的路由专家不计算 F_e(x)，也不参与这次加权和。共享专家全部执行，输出直接相加，不与路由专家一起归一化，也不除以 S。

这是本书明确选择的一套教学约定，不是“所有 MoE 的唯一公式”。尤其要区分原始 Softmax、所选权重归一化、Sigmoid affinity、输出缩放等细节。即便都叫 top-2，公式不同也会产生不同输出。

### 三张工单，手算一次

不必先学习路由器的训练。假设它已经给出下面的 p，专家编号从 0 开始，k=2。相同分数时，本章固定选编号较小者在前，以便复现。

| token | p0 | p1 | p2 | p3 | 选择顺序 | 组合权重 |
| --- | --- | --- | --- | --- | --- | --- |
| A | 0.4 | 0.3 | 0.2 | 0.1 | 0、1 | 4/7、3/7 |
| B | 0.1 | 0.2 | 0.3 | 0.4 | 3、2 | 4/7、3/7 |
| C | 0.4 | 0.4 | 0.1 | 0.1 | 0、1 | 1/2、1/2 |

A 选择的原始概率和为 0.7，所以不是“0.4 倍专家 0 加 0.3 倍专家 1”，而是 4/7 与 3/7。C 的平局不随机打破。

请求负载是 `[2,2,1,1]`，总和 6=N×k。它数的是 **token—专家分配次数**，不是不同 token 的数量，也不包括共享专家。一个 token 选择两个专家，会计两次；每个共享专家另外处理全部 N=3 个 token。

### 再把输出算成一个具体数字

为了单独检查组合规则，暂时不用 SwiGLU，把三个专家换成简单倍率：F0(x)=x，F1(x)=10x，F2(x)=100x。令 affinity 比例为 1:2:4，top-2 选专家 2 和 1，归一化权重为 2/3 和 1/3：

$$
y_{routed}=\tfrac23\times100x+\tfrac13\times10x=70x.
$$

再加两个共享分支 3x 和 5x，得到 78x，而不是 74x。这个倍率实验只用来验算组合，不把线性倍率当成真正的专家能力；正式模块和另一组独立梯度测试都使用 SwiGLU。

## P18.3 张量账本：先摊平，最后放回

设输入 X 的形状为 [B,T,D]，N=B×T。MoE 的路由和专家计算不需要先混合不同 token，因此把前两维暂时摊平即可。

| 张量 | 形状 | 含义 |
| --- | --- | --- |
| flat | [N,D] | 当前调用的全部 token 向量 |
| router.weight | [E,D] | 为所有路由专家打分 |
| logits、probs | [N,E] | 每 token 对每个路由专家的分数、affinity |
| indices | [N,k] | 被选中的专家编号 |
| weights | [N,k] | 丢弃前的组合权重 |
| kept | [N,k] | 容量实验中是否接受该分配 |
| requested、accepted | [E] | 容量处理前后各专家的分配次数 |
| output | [N,D] → [B,T,D] | 专家结果按原 token 位置合回 |

**dispatch** 是按专家收集 token；**combine** 是把结果送回 token 原位置并加权。它们不能只做 reshape：同一个 token 可能出现在多个专家的小批次中，结果要相加，而不是覆盖。

例如上面的 A、B、C：专家 0 收 A、C；专家 1 收 A、C；专家 2 收 B；专家 3 收 B。专家 0 的两个输出再分别回到 A、C 的位置。并没有把 A 的结果交给 C。

## P18.4 专家多了，参数和计算怎样算

沿用 P13 的无 bias SwiGLU，中间宽度为 d_f 的一个专家有三张矩阵：gate、up、down。参数量为 3D d_f。路由器有 DE 个参数。

设路由专家宽度为 d_r，共享专家宽度为 d_s，则**仅 MoE 模块**：

$$
P_{total}=E(3Dd_r)+S(3Dd_s)+DE,
$$

$$
P_{active/token}=k(3Dd_r)+S(3Dd_s)+DE.
$$

这里的 active 是不丢弃时，每 token 参与运算的权重预算。路由器需要给全部 E 个专家打分，所以不能只算 k 个 router 行；共享专家也不能漏算。两式都不包含 attention、norm、embedding 和输出 head，更不是整模型内存公式。

手算 D=4、E=4、k=2、d_r=2、S=1、d_s=2：

- 一个路由专家：3×4×2=24。
- 共享专家：24；路由器：4×4=16。
- 总参数：4×24+24+16=136。
- 每 token 激活参数预算：2×24+24+16=88。

**136/88 不是速度比。** 权重仍需保存或调度；路由、收集、合回、访问内存都需要工作。训练时还有梯度、优化器状态和中间激活。对一批 token，不同 token 的选择取并集后，可能所有专家都被用到。

### 细粒度专家：把宽专家换成更多窄专家

考虑 D=4、没有共享专家的两个方案：

| 方案 | E | k | 专家宽度 | 全部专家参数，不含 router | 每 token 专家参数，不含 router | 含 router 总参数 |
| --- | --- | --- | --- | --- | --- | --- |
| 较粗 | 2 | 1 | 8 | 192 | 96 | 200 |
| 较细 | 4 | 2 | 4 | 192 | 96 | 208 |

专家总预算与每 token 专家预算都相同，但细粒度方案能选择更多组合，同时 router 更大，dispatch 次数也增加。本章通过 expert_width、num_experts 和 top_k 表达这组设计，不提供把一个已训练宽专家无损切成窄专家的转换器。**预算相同不等于函数相同，也不保证质量更好。**

共享专家则是另一维选择：每个 token 都会经过它们，可以学习跨 token 的共同处理，但“学到共同知识”仍是训练假设。若添加共享专家却不相应调整 k 或宽度，激活预算会增加，不能再声称公平对比。

## P18.5 负载失衡：总共六张分配，不等于每人一样忙

如果全部 token 只选择少数专家，会有两个问题：一些专家很少接到任务、很少得到直接的任务梯度；部署到多设备时，忙设备可能拖住其他设备。

负载直方图可以帮助发现问题，却不证明模型质量。自然语言不要求每个小 batch 都完美均匀；过强地追求均匀，也可能干扰有用的选择。本章区分两类调节办法。

### 办法一：辅助均衡损失，通过梯度影响路由器

本章将整个 [B,T] 摊平，定义：

$$
f_e=\frac{n_e}{Nk},\qquad
\bar p_e=\frac1N\sum_{t=1}^{N}p_{t,e},\qquad
L_{aux}=E\sum_e f_e\bar p_e.
$$

n_e 使用丢弃前的 requested 计数；f 的总和为 1。计数来自离散选择，本实现不对它求梯度，梯度通过平均 affinity 流向 router。训练时再乘一个小系数 alpha：

$$
L=L_{CE}+\alpha\cdot\operatorname{mean}_{layers}(L_{aux}).
$$

S5 默认 alpha=0.01，且只有提供 targets 才返回训练损失。均衡辅助项不是标签预测的交叉熵；对验证损失或困惑度做比较时，应单独计算 CE，不把 CE+辅助项直接拿去算 perplexity。

在 A、B、C 的例子里，f=[1/3,1/3,1/6,1/6]，平均 p=[0.3,0.3,0.2,0.2]，所以未乘系数的 L_aux=16/15≈1.066667。

不要把“辅助损失等于 1”读成“硬路由已均衡”。例如四个专家的 p 都等于 1/4，在我们的固定平局规则下，所有 token 都选 0、1，计数仍失衡，但上述辅助项恰好也是 1。应同时查看计数、概率和训练效果。

### 办法二：偏置式均衡，改变选择而不篡改组合权重

为每个专家保存一个不通过梯度学习的 balancing bias b_e。**只在选择时**使用 p_e+b_e；组合时仍按未加 bias 的原始 p 归一化。

以 1:2:4 的 affinity 为例，若给专家 1 加 0.5，它可以排到专家 2 前面，但同样选中 1、2 时，权重仍为 1/3、2/3。不是把加过 0.5 的分数再当成输出权重。

一步训练结束后，本章演示以下显式更新：超出平均请求负载的专家减小 bias，低于平均的增加 bias，相等则不变。例子 `[4,0,2,2]` 的平均为 2，更新步长 0.1 时，bias 增量为 `[-0.1,+0.1,0,0]`。

代码将这件事放在 `update_balance_bias`，不放进 forward。否则同样的输入多调用一次，下一次结果就可能改变，缓存也可能悄悄过期。多 batch 或多设备时需要先按既定统计窗口聚合计数，本章没有实现分布式聚合或生产训练调度。

本实现用 balance_mode 选择 `none`、`aux` 或 `bias`，使两种教学机制可以单独观察。偏置模式不自动叠加辅助项；这项简化不代表真实模型不能组合两者。

## P18.6 容量与丢弃：丢的是专家支路，不是删除句子

某些 MoE 实现为每个专家一次接收的 token 数量设上限，便于控制缓冲区大小。超过容量时，系统可能丢弃、改派或采用不丢弃的可变长度实现，必须看具体规则。

本章的**独立容量实验**用固定整数 capacity，表示每个专家在本次调用最多接受多少次分配。顺序按 flat 的 token 索引从小到大；容量用完后，该专家分支贡献变成 0：

- 不删除原 token，不改序列长度。
- 不改派到其他专家。
- 不把存活分支重新归一化。
- 共享专家不受此容量限制；Block 的残差也仍在。

A、B、C 例子设置 capacity=1 后，requested 仍是 `[2,2,1,1]`，accepted 变成 `[1,1,1,1]`。C 的两个路由分支都被拒绝，共享分支仍能输出。

再看部分丢弃：第一行 p=[0.5,0.1,0.4]，第二行 p=[0.4,0.3,0.3]，k=2、capacity=1。第一行占用了专家 0 和 2；第二行本来选 0、1，但只剩专家 1 被接受，所以其路由权重总和是 **3/7，而不是 1**。这是本实验的语义，不代表它是最佳训练策略。

### 为什么 S5 语言模型明确禁止这个选项

“每次调用重新计算容量”会让同一串 token 的结果依赖分块方式：整段 prefill 可能让后面的 token 丢弃，而逐 token decode 每次都拿到新容量。batch 中另一条序列还可能抢占同一专家的配额。若容量根据 batch token 总数推导，改变未来长度甚至可能改变已有位置的结果。

因此，`TeachingMoE` 可独立演示容量，`MoEGPT` 的完整前向与缓存前向都要求 **dropless，即 capacity=None**。运行中偷偷改成整数也会报错，而不是给出两种语义不一致的 logits。这不是“缓存功能不够强就不测”，而是先把正确性契约说清楚。

## P18.7 梯度去哪儿：选择离散，权重仍可微

在一次前向里，top-k 的专家编号是离散的。本实现不会对“编号从 1 跳到 2”求导。但在选中的分支内部，SwiGLU、归一化权重和输入仍参与 autograd。

需要分开看：

1. 被接受的专家收到来自相应 token 的任务梯度；本次完全没执行的专家，其参数 `.grad` 可以是 None，不代表程序故障。
2. top-k 组合权重通常让任务损失影响 router，但本章的 **top-1 加所选权重归一化**有特殊性：唯一权重恒为 1，所以该任务路径的 router 梯度为 0。
3. 辅助损失通过全部 affinity 影响 router，不能把“空专家没有直接任务梯度”误说成“对应 router 行永远没有梯度”。

在 78x 的手算测试里，x=[1,0]、router 第一列为 [0,log2,log4]、第二列为 0。令损失为输出各维之和，router 第一列梯度为 `[0,-20,+20]`：权重变化会在 10x 与 100x 间重新分配。输入第一维梯度为 78+20log2，第二维为 78。测试把这些字面算式与 autograd 对齐，没有拿生产 dispatch 函数自己验证自己。

“今天没收到梯度”不等于“永远不会变化”：其他 batch 可能路由到它，优化器的历史动量、权重衰减和梯度清理方式也会影响实际更新。我们的空专家测试只断言本次反向的直接梯度。

## P18.8 顺着代码走一次

关键实现没有写成一个与主线无关的玩具。`MoEGPT` 继承 `GroupedGPT`，保留 attention、norm、残差、embedding 和 head，只替换每个 Block 的 mlp。

```python
for block in self.blocks:
    block.mlp = TeachingMoE(
        config.d_model, config.num_experts, config.top_k,
        config.expert_width, config.shared_experts,
        config.shared_width, config.balance_mode)
```

专家调用处可以按六步读：

```python
flat = x.reshape(-1, self.d_model)
info = route(self.router(flat), self.top_k, bias, self.capacity)
output = flat * 0
for expert_id, expert in enumerate(self.experts):
    token_ids, slots = ((info.indices == expert_id) & info.kept).nonzero(as_tuple=True)
    if token_ids.numel() == 0:
        continue
    values = expert(flat.index_select(0, token_ids))
    values = values * info.weights[token_ids, slots, None].to(values.dtype)
    output = output.index_add(0, token_ids, values)
for expert in self.shared:
    output = output + expert(flat)
return output.reshape_as(x)
```

1. 摊平 token，但保留 D。
2. 计算 router，再得到选择、权重和容量 mask。这里 bias 只在偏置模式提供。
3. 按 expert_id 找出属于它的 token 和在 top-k 列表中的位置 slots。
4. 只收集并计算这些 token，空专家直接跳过。
5. 查出对应权重乘到专家输出上，`index_add` 按 token 索引相加，避免覆盖其他专家的贡献。
6. 运行所有共享专家，恢复原形状。残差在外层 Block 中添加。

route 里不用默认 topk 的偶然平局顺序，而使用稳定降序排序，保证同分时专家编号较小的在前。没有非零均衡偏置时，直接按 logits 排序：Softmax 单调，而直接排序可避免极小概率下溢成相同的 0，制造假的平局。组合权重则直接对**所选原始 logits**做 Softmax：代数上等于先取原始 p 再归一化，但若 bias 选中极小概率专家，可避免原始 p 下溢后出现 0/0。

这仍然是强调可读性的 CPU 实现：Python 循环、有限性检查、逐专家收集都不是生产 fused kernel。不要拿它的运行时间推断大型 MoE 的性能。

### 缓存也必须认识路由配置

S5 继续调用 S4 的 compact GQA cache，形状仍为 [B,Hkv,T,d]。但高层的历史 K/V 依赖低层 MoE 的输出，所以改变低层路由也会让缓存失效。

已有签名追踪权重、位置参数、norm epsilon 等；S5 另外追踪实际模块上的 top_k、capacity、balance_mode，以及 balancing bias 等 buffer 的身份、原地修改版本、dtype 和 device。bias 虽然不是 Parameter，也必须纳入。

以下情况都需要清空 cache：更新权重、更改 router 设置、执行 bias 更新、替换 bias buffer、切换 dtype/device。不同模型不能共用缓存。和 S1–S4 一样，故意通过 `.data` 绕过版本跟踪或手动改 cache 张量都不受支持；签名不是防篡改哈希。

`last_info` 只是最近一次调用的诊断，不是累计请求状态。逐 token decode 后看到的计数，只描述最后一次输入的那几个 token。不要把它当成整段请求直方图。

## P18.9 在 CPU 上跑通，并知道测试证明了什么

使用仓库固定依赖，在仓库根目录运行：

```bash
python -m pip install -r requirements.txt
OMP_NUM_THREADS=1 python code/19_moe.py
OMP_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_moe.py' -v
```

2026-10-19，CPU、PyTorch 2.14.1+cpu、seed=180 的实际输出包括：

```text
capacity=None; selected=[[0, 1], [3, 2], [0, 1]]
requested=[2, 2, 1, 1]; accepted=[2, 2, 1, 1]; aux=1.066667
capacity=1; selected=[[0, 1], [3, 2], [0, 1]]
requested=[2, 2, 1, 1]; accepted=[1, 1, 1, 1]; aux=1.066667
module budget: {'expert': 24, 'router': 16, 'shared': 24, 'total': 136, 'active': 88}
finite training loss/gradients: True
layer 0 requested load: [4, 6, 11, 3]
layer 1 requested load: [8, 6, 5, 5]
full/cache maximum absolute error: 1.79e-07
compact KV payload bytes: 1536; no GPU timing measured
```

后一组直方图来自未训练的两层 S5，B=2、T=6、k=2，所以每层计数和为 24。不能把随机初始化下的分工当成学到的专家能力。脚本用合成 token 做一次梯度 smoke，循环左移标签也只是用于验证反向，并非本书推荐的语料构造或验证集评估方法。

KV 的 1536 字节可以独立复算：2（K/V）×2 层×2 batch×6 token×2 KV 头×4 头宽×4 字节。MoE 没有把 Hkv 改成 E；专家参数内存与 KV payload 是两本不同的账。

测试覆盖的不只是 full/cache 相同：

- 固定路由权重、平局规则、负载、辅助损失和参数预算手算。
- 独立逐标量 SwiGLU 与路由参考，对比输出、输入梯度和全部参数梯度。
- 空 token、空专家、所有 token 集中到一个专家、top-1 和 top-E 边界。
- 容量为零、完全丢弃、部分丢弃，以及共享分支仍保留。
- 稀疏 dispatch 确实不调用空专家；梯度 None 与零的区别。
- 多种 chunk 划分、单 token decode、未来 token 改动不影响前缀、batch 独立和请求重置。
- bias、top-k、权重、dtype、位置参数改变后，旧 cache 被拒绝。
- 故意改成均匀权重、全专家执行、把 bias 混进组合、共享输出取平均、丢弃后重归一化、后到先收等错误时，独立判据确实失败。

CPU 数值对齐不是 GPU 速度测试，也不是质量评测。浮点末位可能随环境变化，判定依据是测试容差，而非逐字复制日志。没有 PyTorch 时，CLI 给出安装提示并退出码 2，数值测试明确跳过；**跳过不是通过**。

## P18.10 多设备时，还要付“把工单送过去”的成本

当专家分布在不同设备上，dispatch 不再只是本机索引：需要把 token 向量送到相应专家所在的设备，算完后把输出送回。常见 expert parallel 会用到 all-to-all 一类通信，但具体分组、复制和合并方案各不相同。

一个粗略账本是：N 个 token，每个发往 k 个远端专家，每份 D 维、每元素 q 字节，那么一次向外发送的向量 payload 约为 NkDq；返回还要再算一趟。它不是网络实测：同节点合并、专家共置、本地命中、元数据、padding、反向通信和通信计算重叠都会改变真实开销。

因此，决定时延的至少包括：

- 每个专家的实际 token 数，最忙的设备是否成为瓶颈。
- 矩阵乘法能否形成足够大的批次。
- 网络带宽、延迟、路由限制与通信重叠。
- 模型权重能否驻留，加载或搬运要花多久。
- 服务目标是 prefill 吞吐、单请求 decode 延迟，还是并发服务。

“激活参数较少，所以普通电脑一定更快”是不成立的。小 batch 下，大量小矩阵乘法和调度可能更慢。性能要回到 P15 的相同请求长度、并发、硬件与 TTFT/TPOT 等指标上测；本章没有 GPU 或多卡实验。

## P18.11 和官方资料对照：学机制，不冒充复现

本章于 2026-10-19 核验以下固定论文版本，并沿用大纲来源编号：

- **S19：** [DeepSeekMoE，arXiv 2401.06066v1，§3.1–3.3](https://arxiv.org/html/2401.06066v1#S3)。它讨论细粒度专家、共享专家隔离与负载均衡。共享专家始终执行，细分专家时要同时调整激活数量，不能只比较专家数。该版本公式（9）–（11）的 Softmax gating 不应被自动改写为本章的 selected renormalization。
- **S09：** [DeepSeek-V3 官方仓库](https://github.com/deepseek-ai/DeepSeek-V3) 所链接的 [技术报告 arXiv 2412.19437v2，§2.1.2](https://arxiv.org/html/2412.19437v2#S2.SS1.SSS2)。V3 使用 Sigmoid affinity、所选 affinity 归一化、只影响选择的均衡偏置，并保留很小的序列级辅助项；报告明确说明训练和推理不丢弃 token。因此 “auxiliary-loss-free” 不能解释成全模型绝对没有任何辅助损失，也不能把本章容量丢弃示例说成 V3 的做法。

本书 S5 使用 Softmax affinity、单进程逐专家 dispatch、可分开观察的 batch 辅助项或偏置模式。没有复现 V3 的 Sigmoid 路由、节点限制、分布式均衡、FP8、通信 kernel、训练数据或官方权重。引用生产系统的设计动机，不意味着继承其性能和质量结论。

## P18.12 自测：先写答案，再运行断言

1. A 的 p=[0.4,0.3,0.2,0.1]、k=2，输出权重是 0.4、0.3 吗？三行 A、B、C 一共发生多少次路由分配？
2. D=4、E=4、k=2、路由和共享宽度都为 2、一个共享专家时，MoE 总参数与每 token 激活预算各多少？是整模型参数吗？
3. 把 E=2、k=1、宽度 8 换成 E=4、k=2、宽度 4，专家参数预算保持什么？包含 router 后还严格相等吗？
4. A、B、C 的请求计数为何是 [2,2,1,1]？capacity=1 后计数和 C 的路由贡献怎样变？
5. 部分丢弃例子第二行只保留专家 1，权重应为 1 还是 3/7？共享分支受不受影响？
6. 所有 token 的四个 affinity 都是 1/4，top-2 的硬负载会均匀吗？L_aux=1 能证明均衡吗？
7. top-1 且所选权重归一化时，单靠任务损失能在本实现中更新 router 吗？辅助项呢？
8. bias 更新采用负载 [4,0,2,2]、步长 0.1 时，增量是多少？已建的 KV cache 能否继续用？
9. 缓存和完整前向都用同一个错误的“所有专家平均”函数，二者一致能证明 MoE 正确吗？
10. 本次 CPU 日志中的 KV payload 1536 字节怎样算？为什么它不等于专家权重占用？

### 参考答案与对应测试

1. 权重为 4/7、3/7；三行共 3×2=6 次。见 `test_hand_route_weights_counts_aux_and_ties`。
2. 总参数 136，激活预算 88，只覆盖 MoE 模块。见 `test_parameter_active_budget_and_fine_grained_example`。
3. 专家总参数都是 192，每 token 专家参数都是 96；含 router 总参数分别为 200、208。见同一预算测试。
4. A、C 都选 0、1，B 选 3、2；accepted=[1,1,1,1]，C 的路由贡献为零，共享和残差不因此删除。见 `test_capacity_fixed_order_no_renormalization_shared_survives`。
5. 3/7，不重新归一化；共享分支不受容量限制。该测试同时检查部分丢弃与 capacity=0。
6. 不会。固定平局规则使每个 token 都选 0、1；损失仍可为 1。见手算路由测试中五 token 的 [5,5,0,0] 断言。
7. 不能通过这条任务路径，唯一组合权重恒为 1，router 梯度为零；辅助项在测试的集中路由案例中提供非零梯度。见 `test_sparse_execution_empty_experts_and_top1_router_gradient`。
8. [-0.1,+0.1,0,0]；更新 buffer 后必须重置缓存。见偏置方向测试与 `test_runtime_and_buffer_mutations_invalidate_cache`。
9. 不能。测试还用独立标量输出/梯度参考、78x 手算和故意注入的错误验证判据。见 `test_mutations_fail_independent_oracle_not_only_cache_agreement`。
10. 2×2×2×6×2×4×4=1536；这是当前请求的 K/V 张量，形状依赖层数、KV 头与长度，而不是专家矩阵。见 `test_full_cache_alignment_each_mode_and_extreme_k`。

到这里，S0→S5 已经把“多保存一些参数”和“每 token 真正调用哪些参数”分开。接下来进入训练流水线：即使路由与缓存都正确，数据边界、去重、切分和预算没有处理好，也训练不出可靠的模型。
