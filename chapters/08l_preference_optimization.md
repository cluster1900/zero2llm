# 训练进阶（P22）：RLHF、DPO 与 GRPO {#p22-preference}

[P21：SFT、LoRA 与对话数据](#p21-sft) 教模型“照着好答案写”。但有时我们只有两份答案的比较，或者只能在答案写完后判断对错。怎样把这种反馈变成参数更新？

想象一位学徒练习做菜。SFT 是跟着师傅的菜谱做；偏好数据是品尝者说“这盘比那盘好”；奖励是为一盘菜打分。**比较、打分、学习如何做菜，是三件不同的事。** 品尝者也可能偏爱摆盘，评分器也可能漏查食材。分数上涨不等于真的做得更好。这是本章最重要的边界。

本章不从几十个训练参数开始。先补齐对数概率和期望，再手算一批 DPO、一个 GRPO 组，最后让 CPU 真正改变六个参数。配套 [code/23_preference_optimization.py](../code/23_preference_optimization.py) 有两层：

- 标准库路径：DPO 数值、组内优势、精确离散 KL，不需要安装 PyTorch。
- PyTorch 路径：两道合成题、每题三个固定答案的策略，实际运行 SFT，以及从同一 SFT 起点分出的 DPO / GRPO 两条更新路线；另有语言模型式变长 token 概率与 mask 测试。

**六参数策略不是 Transformer，不生成任意文本，不学习算术规则；TRL 配置只讲解，未运行。** 本章既不复现 InstructGPT、DeepSeekMath 或 R1，也不提供大模型训练速度、显存或质量结论。

## P22.1 先把几个名字放回各自的位置

| 名称 | 它主要回答的问题 | 本章的边界 |
| --- | --- | --- |
| SFT | 给定示范，应该提高哪些 token 的概率？ | 常见后训练起点，不是所有 RL 路线的必需前置 |
| 人类偏好数据 | 同一问题的两个答案，哪个更好？ | 可能有分歧、噪声与评价偏差 |
| 奖励模型 | 用模型预测一个答案的评分 | 拟合人的判断，不是真理检测器 |
| PPO 式 RLHF | 用人类反馈训练的奖励信号，怎样更新生成策略？ | 一条常见 RLHF 流程，不代表所有 RLHF |
| DPO | 怎样直接用偏好对优化策略？ | 标准离线形式不需要单独训练奖励模型，也不需要在线 rollout |
| GRPO | 同题采样多份答案，怎样用组内比较更新策略？ | 奖励可来自模型或可验证规则；并不天然等于“人类反馈” |

“策略”在这里就是生成答案的概率分布，记为 π。θ 是可训练参数，x 是问题，y 是答案。对语言模型，选择下一个 token 是一次动作；本章 CPU 玩具则把选择一整份预置答案当作一次动作。两者都能解释概率更新，但不能混淆动作空间。

![从 SFT 起点分出 PPO 式 RLHF、DPO 和 GRPO；奖励全相同的组优势为零；固定参考与采样旧策略职责不同](assets/diagrams/p22-preference.svg)

图中的三条路线是对照，不是要求读者先做 PPO，再做 DPO，最后做 GRPO。代码中 DPO 与 GRPO 都从同一份 SFT 参数复制，便于看清各自的作用。

## P22.2 补课：log probability、期望与 KL

### 概率相乘，对数相加

假设在已有问题和前文条件下，两个答案 token 的概率依次是 0.5、0.25。那么这条答案的条件概率是：

$$
\pi(y\mid x)=0.5\times0.25=0.125.
$$

使用自然对数 ln，有：

$$
\log\pi(y\mid x)=\log0.5+\log0.25\approx-2.07944.
$$

这里的第二个概率已经以第一个 token 为条件，**不是假设 token 彼此独立**。对数把很长的概率连乘变成相加，数值上更方便。概率在 0 到 1 之间，对数概率通常为负；从 −3 提高到 −2，是概率增加，不是下降。

在 P09 的交叉熵里，我们最小化负对数概率；在 DPO 中，我们比较两个答案的对数概率。注意区分：

- 完整答案的 log probability：对有效答案 token 求和，包括真正生成的 EOS。
- 每 token 平均 log probability：再除以答案长度，是另一种量。
- 原始 logits：尚未归一化的分数，不能直接当 log probability。

例如两个 token 都为 −1，序列和是 −2，均值是 −1。偷偷把和换成均值，不是“只优化实现”，而是改变目标和长度效应。

### 期望是按概率加权，不是总取最好的一项

三个答案的概率为 [0.5, 0.3, 0.2]，奖励为 [1, 0, 0]。期望奖励为：

$$
\mathbb E_{y\sim\pi}[r(y)]
=\sum_y\pi(y)r(y)=0.5.
$$

符号 y∼π 表示按策略抽样。多抽几次再取奖励均值，可以估计这个期望，但有限样本有波动。贪心只选最大概率答案，不是在进行同一种抽样。本章 `sample_rollouts` 用固定种子的有放回抽样，temperature=1，不做 top-k / top-p 截断。

### KL 比较两份完整分布

参考策略 π_ref 像一张固定的起点照片。我们希望提高奖励，又不无节制地偏离它。一种约束量是：

$$
D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})
=\sum_y\pi_\theta(y)
\left[\log\pi_\theta(y)-\log\pi_{\mathrm{ref}}(y)\right].
$$

两份分布相同时 KL=0；一般不对称，不能调换左右。代码在有限 logits 上用 log-softmax，枚举所有动作计算。手算 p=[0.6,0.3,0.1]、q=[0.2,0.5,0.3]，得到约 0.396058 nats。它不是“只看 chosen 的概率变化”，也不是两个 logits 的欧氏距离。

对真正的自回归模型，在一个固定前缀上枚举词表，得到的是**该前缀下的条件下一 token KL**。对旧策略采到的若干前缀取平均，并不自动成为完整答案分布的精确 KL。本章后面打印的“精确答案 KL”之所以成立，是因为玩具策略只有三个完整答案，能够全部枚举。

## P22.3 经典流程：SFT → 奖励模型 → PPO 式 RLHF

[InstructGPT 论文，2022-03-04](https://arxiv.org/abs/2203.02155v1) 展示了示范微调、答案比较、强化学习三个阶段。这里解释数据怎样流动，不运行这套完整系统。

1. 收集示范答案，做 SFT。这样策略已经具备基本的回答方式。
2. 对同一个问题生成多个答案，请标注者比较，得到 chosen / rejected。
3. 训练奖励模型 r_φ(x,y)，让 chosen 的评分倾向于更高。
4. 用当前策略回答新的一批问题，奖励模型为这些 rollout 打分。
5. 使用 PPO 等策略优化方法更新生成模型，同时控制它与参考策略的偏离。

奖励模型可以用成对损失：

$$
\mathcal L_{\mathrm{RM}}
=-\log\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big),
\qquad \sigma(z)=\frac{1}{1+e^{-z}}.
$$

w 表示较受偏好的答案，l 表示较不受偏好的答案。只比较差值，所以对同一问题把两个评分都加 100，并不改变偏好概率。评分没有天然的“满分 100”含义。

常见 PPO 式语言模型训练还会用价值函数 V 来估计未来回报，并计算“比预期好多少”的优势 A。**奖励模型负责评价输出；价值函数负责估计回报基线。** 它们可能都是神经网络，却不是同一个职责。

PPO 的 clipped surrogate 抑制一批旧样本上的过大更新。它比较当前策略与生成这批样本的旧策略，不是把所有参数硬裁剪到一个范围。[PPO 原论文 v2，2017-08-28](https://arxiv.org/abs/1707.06347v2) 给出了这一目标。我们在 GRPO 小节继续拆开其中的概率比与 clip。

特别记住两份快照：

- π_ref：本章固定为 SFT 起点，用于衡量偏离，整个偏好优化期间不更新。
- π_old：生成当前这批 rollout 时的策略快照，同一批的多次更新期间不变；换批采样时可以刷新。

把 old 和 ref 互换，会改变概率比或惩罚项；每一步都把 old 改成正在训练的策略，也会让概率比失去“相对于采样时”的含义。成熟训练系统的生命周期更复杂，本章先把这两种职责分开。

## P22.4 DPO：直接学习“相对起点，更偏向哪份答案”

一条偏好样本必须是同一问题的三元组 (x,y_w,y_l)。不能把“1+1=?” 的好答案和“2+1=?” 的坏答案随意凑成一对，也不能让两侧使用不同的 chat template。

标准 DPO 的每对损失为：

$$
\begin{aligned}
\Delta_\theta &= \log\pi_\theta(y_w\mid x)-\log\pi_\theta(y_l\mid x),\\
\Delta_{\mathrm{ref}} &= \log\pi_{\mathrm{ref}}(y_w\mid x)-\log\pi_{\mathrm{ref}}(y_l\mid x),\\
z &= \beta(\Delta_\theta-\Delta_{\mathrm{ref}}),\\
\mathcal L_{\mathrm{DPO}} &= -\log\sigma(z)=\operatorname{softplus}(-z).
\end{aligned}
$$

来源是 [DPO 论文 v3，式 (7)，2024-07-29](https://arxiv.org/html/2305.18290v3#S4)。直觉是：不只问策略是否喜欢 chosen，而是问它**相对于原来，更偏向 chosen 了吗？** 参考策略可能本来就偏爱某个答案，减掉参考差值可以把这一点纳入目标。

DPO 的推导把 KL 正则化奖励最大化与特定偏好模型联系起来，再将奖励重新参数化。标准离线 DPO 不单独拟合 reward model，也不必在每个优化步在线生成答案。它仍然依赖偏好数据、参考策略和假设，**不能因此说“DPO 就是全部强化学习”，更不能说任何 DPO 数据都等价于任意 RL 问题**。

### 手算一个两对答案的 batch

取 β=0.5。表中只列 chosen / rejected 的概率，其余概率质量属于其他答案。

| 样本 | 当前 chosen / rejected | 参考 chosen / rejected | 相对 margin | z | loss |
| --- | --- | --- | --- | --- | --- |
| A | 0.6 / 0.2 | 0.4 / 0.4 | ln 3 ≈ 1.098612 | 0.549306 | 0.455746 |
| B | 0.2 / 0.5 | 0.1 / 0.5 | ln 2 ≈ 0.693147 | 0.346574 | 0.534800 |

batch 对两对样本取平均，结果约 0.495273。B 的 chosen 概率仍小于 rejected，但相对参考已经改善，所以 z 仍为正。如果当前策略和参考完全相同，每对 loss 都是 ln 2≈0.693147，而不是零。

对应的核心代码只有几行：

```python
margin = (chosen - rejected) - (
    ref_chosen.detach() - ref_rejected.detach()
)
loss = torch.nn.functional.softplus(-beta * margin).mean()
```

四个输入都是 [B] 的**答案 log probability**，不是 token ID，也不是 reward。detach 切断参考分支的梯度；参考参数本身也不应交给 optimizer。softplus 的稳定实现避免先算巨大的 exp 再溢出。

单对 loss 对 chosen log probability 的导数是 −βσ(−z)，对 rejected 则为 +βσ(−z)。梯度下降倾向于增加相对 margin。共享神经网络参数会同时影响其他答案，不保证每条 chosen 的绝对概率在每一步都上涨。

### β 不是“越大越听话”的旋钮

在 DPO 的推导中，β 与对参考策略的约束强度有关。落实到这条损失，β 同时影响 sigmoid 的输入与梯度尺度；例如起点 z=0 时，梯度大小与 β 成正比。因此不能只凭“β 大”就断言实际训练每步更小，或者最终一定更好。应固定数据、学习率等条件，比较偏好评估、KL 和其他能力变化。

原始 DPO 用答案 token log probability 的和。改成长度平均、删掉 reference、把乘 β 写成除 β，都是不同目标或错误实现，不能仍声称与这条公式相同。

## P22.5 从变长 token 到一对概率：形状不能代替对齐

语言模型 logits 为 [B,T,V]，input_ids 和 attention_mask 为 [B,T]。本章的 `completion_log_probs` 做三步：

```python
next_logps = logits[:, :-1].log_softmax(dim=-1)
targets = input_ids[:, 1:]
selected = next_logps.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
```

这就是一次 causal shift：输入位置 t 预测 token t+1。然后根据每行实际的 prompt 长度和 padding mask，只保留答案目标：

```python
mask = attention_mask[:, 1:].bool() & (positions >= starts)
sequence_logps = (selected * mask).sum(dim=-1)
```

positions 是原序列中目标的位置 1…T−1，starts 是该行答案开始的位置。**最后一个 prompt token 的 logits 要参与预测第一个答案 token**，不能把“输入来自 prompt”误解成“这行 logits 一定不算 loss”。

例如 T=5，第一行 prompt 长 2、真实总长 4，第二行 prompt 长 1、真实总长 5。shift 后的答案 mask 为：

```text
第一行：[False, True, True, False]   两个答案目标，末尾 PAD 不算
第二行：[True,  True, True, True ]   四个答案目标
```

EOS 如果确实属于答案，应当计入；PAD 不计入。这个 helper 只支持右 padding、有至少一个 prompt token 和一个答案 token 的输入，遇到左 padding、中间断开的 mask、空答案或非法 ID 就报错。它不负责 tokenizer 拼接边界，调用方还要保证 chosen / rejected 共用同一 prompt 和一致的结束规则。

测试不仅比较 loss，还逐元素检查梯度：无效位置应为零，有效位置应等于“目标指示值减去该 token 的概率”。修改被排除位置的 logits，不应改变结果。这样才能发现“维度正确但监督错位”的错误。

## P22.6 GRPO：同一道题，先比较一组答案

[DeepSeekMath v3，第 4.1 节，2024-04-27](https://arxiv.org/html/2402.03300v3#S4.SS1) 引入 GRPO：对同一个问题采样一组答案，用组内奖励建立相对基线，不必另外训练 PPO 式价值网络。本章讲 **outcome reward**：每份完整答案得到一个分数，同一答案的有效 token 共用一个优势值；不是逐步过程奖励。

设一个问题采样 G 份答案，奖励为 r₁…r_G。本章明确选择总体标准差，即分母为 G：

$$
\bar r=\frac1G\sum_i r_i,\qquad
s=\sqrt{\frac1G\sum_i(r_i-\bar r)^2},\qquad
A_i=\frac{r_i-\bar r}{s+\delta},\quad\delta=10^{-8}.
$$

δ 放在平方根**外面**。这里的选择用于手算和可重复的教学实现，不意味着所有 GRPO 实现都采用同一个标准差、epsilon 或归一化方式。

### 手算一组 [0,1,1,2]

平均奖励为 1，方差为 (1+0+0+1)/4=0.5，标准差约 0.707107。忽略很小的 δ 后，优势约为：

```text
奖励：  0          1       1       2
优势： -1.414214   0       0      +1.414214
```

相对本组平均更好的是最后一份答案；第一份较差。这里没有给另一道更难的题“借用平均值”。代码按 prompt_id 分组，即使样本交错排列也不会混组。每组要求至少两份答案。

如果一组奖励为 [2,2,2,2]，减去平均后全为 0，分母为 δ，优势正好全为 0，不会产生 NaN。它可能表示全部答对、全部答错，或者评分器太粗糙；**仅凭零优势无法知道是哪一种**。这组没有奖励比较带来的策略梯度，但若策略已经偏离 reference，非零 KL 惩罚仍可能产生梯度。

### 概率比、clip 和最小化符号

对第 i 份答案的第 t 个有效 token，记当前与采样旧策略的概率比为：

$$
\rho_{i,t}
=\exp\left(\log\pi_\theta(y_{i,t}\mid x,y_{i,<t})
-\log\pi_{\mathrm{old}}(y_{i,t}\mid x,y_{i,<t})\right).
$$

本章使用 clip ε=0.2，并把“最大化收益”写成优化器常用的“最小化负收益”：

$$
\mathcal L=
\frac1N\sum_{i=1}^{N}\frac1{L_i}
\sum_{t=1}^{L_i}
\left[-\min\left(\rho_{i,t}A_i,
\operatorname{clip}(\rho_{i,t},0.8,1.2)A_i\right)
+\beta K_{i,t}\right].
$$

N 是这批答案总数，L_i 是第 i 份答案的有效长度，K 是该前缀的 KL 项。本章各题采样数相同，因此对答案均权也使题目均权。不同组大小时还需明确是否要对题目再次加权。

两个检查符号的例子：

- A=+1、ρ=1.4：收益取 min(1.4,1.2)=1.2，不再鼓励这项收益无限上涨。
- A=−1、ρ=0.6：收益取 min(−0.6,−0.8)=−0.8，不能错误地取 max。

clip 的作用依赖优势符号和更新方向，并不是所有范围外的概率比都“没有梯度”；它也不保证实际 KL 永远小于一个固定值。更新后的概率仍需检查。

### 本章的 KL 和长度归约，刻意写清楚

`grpo_loss` 输入 logits/reference=[N,L,V]，targets/old_logps/mask=[N,L]，advantages=[N]。old、reference、advantages 都停止梯度；当前 logits 保留梯度。mask 中每行至少一个 True。有限的 log probability 不保证 exp 后仍然有限；如果概率比、逐 token loss 或归约超出 dtype 的有限范围，代码明确报错，不用 clip 掩盖溢出，也不让 0×∞ 静默变成 NaN。

本章词表很小，K 使用枚举整个词表的精确条件 KL，而不是采样估计。原论文使用的一个逐 token KL 估计写成：

$$
k=\frac{\pi_{\mathrm{ref}}(a\mid s)}{\pi_\theta(a\mid s)}
-\log\frac{\pi_{\mathrm{ref}}(a\mid s)}{\pi_\theta(a\mid s)}-1.
$$

若动作 a 真按当前 π_θ 抽样，且支持集条件满足，它的期望等于相应 KL；单个样本不是精确分布 KL。若样本来自固定 old，已经更新后的 current 与 old 不同，也不能不加说明地把旧样本均值称为精确 KL。本章不拿这个估计量冒充精确监测值，亦不把替换成精确 K 的教学目标说成原论文代码逐行复现。

长度归约先做“每个答案内 token 平均”，再对答案平均。若答案长度分别为 2 和 4，每份答案总权重都是 1/2；若改成六个 token 统一平均，两份答案的总权重则为 1/3 和 2/3。两者都能写成 loss，但不是同一个目标。测试刻意使用不同长度，以免相同长度掩盖错误。

## P22.7 真正跑一次：参数、概率和 KL 都由更新得到

从仓库根目录运行：

```bash
# 标准库手算，不导入 torch
python code/23_preference_optimization.py

# 安装仓库基线依赖后，实际 CPU 更新
python code/23_preference_optimization.py --torch

# 独立数值与梯度测试
python -m unittest discover -s tests -p test_preference_optimization.py -v
```

缺少 PyTorch 时默认手算正常退出；显式请求 `--torch` 会打印 `SKIP` 并以状态码 2 退出，不能把“没有运行”当成训练成功。可用 `python -S` 模拟不加载第三方包；测试中 PyTorch 项会明确跳过，标准库项仍执行。

### 一个非常小，但真实更新的策略

问题只有 “1+1=?” 和 “2+1=?”；答案动作都为 [“2”, “我猜是 0”, “3”]。每题各有三个可训练 logits，共六个 float64 参数。从全零开始，先用正确答案做 4 次 SFT SGD 更新，学习率 0.4。

之后复制两份策略：

- DPO：chosen 为各题正确答案，rejected 都是“我猜是 0”，β=0.5，学习率 0.4，更新 6 次。
- GRPO：每题从 SFT 策略有放回采样 8 个动作，精确匹配正确答案得 1，否则得 0；保存 old 概率，同一批更新 3 次，学习率 0.2，β=0.04，clip ε=0.2。

GRPO 的一次动作代表一整份答案，所以优化张量的 L=1；“我猜是 0”虽然字符更多，也不是多个策略动作。语言模型变长 token 的实现契约在前面和测试中单独验证。不要把这两层混写成“已经训练了一个生成式小 LLM”。

2026-10-19，在 CPU、PyTorch 2.14.1+cpu、seed=22、单线程下实测：

| 指标 | SFT 起点 | DPO 后 | GRPO 后 |
| --- | --- | --- | --- |
| 两题平均的精确期望奖励 | 0.506496 | 0.608010 | 0.560679 |
| 对 SFT 的精确答案 KL，题 0 | 0 | 0.024651 | 0.006216 |
| 对 SFT 的精确答案 KL，题 1 | 0 | 0.024651 | 0.006595 |
| DPO 固定偏好对 loss | 0.693147 | 0.562092 | 不作跨目标对照 |

例如题 0 的三个 logits 从 [0.479421,−0.239710,−0.239710] 变为 DPO 的 [0.761335,−0.521624,−0.239710]；这些数字来自 backward 和 SGD，而不是为了展示方向手写的“更新后概率”。reference 保持原值，并与偏好更新前独立 clone 的快照逐值比较；不能仅与可能同步改变的另一份对象比较。示例用 `fork_rng` 与 `finally` 在正常返回和异常退出时恢复调用者的 CPU 随机数状态、线程数，不触碰 CUDA 随机数发生器。

这不是 DPO 优于 GRPO 的实验：优化步数、目标、样本和学习率都不同，数据只有两道训练题。精确期望奖励也不是独立测试集准确率。两题各用一行独立参数，没有任何跨题泛化能力。我们只证明目标、梯度、数据对齐、采样和参数更新构成了可检验闭环。

### reward 对齐不能只靠 zip

一次采样保存 sample_id、prompt_id、action_id 和 completion；奖励保存对应 sample_id、prompt_id 和 value。`aligned_rewards` 在转 tensor 前检查条数、ID、顺序、题号、动作与文本一致性、有限奖励。

“2” 对题 0 正确，对题 1 错误。把奖励列表交换顺序，虽然形状完全相同，仍会把错误行为加强。代码遇到这种错配会报错，而不是悄悄 zip 截断或继续训练。实际数据流水线也应保留能追溯到样本的标识，不要在排序、截断、异步打分后丢失对应关系。

## P22.8 接到 TRL：固定版本配置阅读，明确未运行

本节核对的是 **TRL 0.16.1** 的 [DPOConfig 源码](https://github.com/huggingface/trl/blob/v0.16.1/trl/trainer/dpo_config.py) 与 [GRPOConfig 源码](https://github.com/huggingface/trl/blob/v0.16.1/trl/trainer/grpo_config.py)，核验日期为 2026-10-19。这是固定版本接口导读，不是对当前最新版的描述，也不是与本仓库依赖组合的兼容性证明。

**为什么相邻章节引用的 TRL 版本不一样？** [P20](#p20-frameworks) 用 [0.17.0 官方总览](https://huggingface.co/docs/trl/v0.17.0/en/index) 建立框架与训练入口的地图；[P21](#p21-sft) 用 [0.18.2 SFTTrainer 源码](https://github.com/huggingface/trl/blob/v0.18.2/trl/trainer/sft_trainer.py) 说明 SFT 训练器的职责。本章则选择 **0.16.1 作为固定的历史 GRPO 实现切片**，逐项核对奖励标准化、KL 估计和 loss 归约，再与手写教学变体对照。选择这个较早标签是为了固定具体的比较对象，不是推荐退回旧环境。

这三个版本号属于三处资料引用，**不是同一套训练环境，不是一条逐章升级或降级的安装路线，也没有经过跨版本兼容性验证**。下面对 0.16.1 行为的结论只适用于所链接的源码；本章没有比较另外两版的相同函数，因此不推断它们的 GRPO 行为一定相同或一定不同。三个章节均未运行 TRL，P20 的真实框架实验是另行锁定的 Transformers / Accelerate 实验。

下面只是一份**未执行的配置示意**，不包含模型下载、Trainer 构造或 train 调用：

```python
# NOT RUN：仅供阅读，需在单独环境核对完整依赖。
from transformers import GPT2Config
from trl import DPOConfig, GRPOConfig

# 若开展框架实验，可从这个结构创建随机本地模型；不是预训练权重。
tiny_architecture = GPT2Config(
    vocab_size=12, n_positions=32, n_ctx=32,
    n_embd=16, n_layer=1, n_head=2, n_inner=32,
    resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
    bos_token_id=2, eos_token_id=3, pad_token_id=0,
)

dpo_config = DPOConfig(
    output_dir="p22-dpo-local", use_cpu=True,
    per_device_train_batch_size=2, max_steps=2,
    max_prompt_length=8, max_completion_length=8, max_length=16,
    beta=0.1, loss_type="sigmoid",
    fp16=False, bf16=False, gradient_checkpointing=False,
    report_to=[], save_strategy="no",
)
grpo_config = GRPOConfig(
    output_dir="p22-grpo-local", use_cpu=True,
    per_device_train_batch_size=4, num_generations=4,
    max_steps=2, max_prompt_length=8, max_completion_length=8,
    temperature=1.0, top_p=1.0, top_k=None,
    beta=0.04, epsilon=0.2, num_iterations=2,
    use_vllm=False, fp16=False, bf16=False,
    gradient_checkpointing=False, report_to=[], save_strategy="no",
)
```

先读懂这些设置，再谈运行：

- 12 项词表、16 维、1 层、2 个头把模型结构限制在小规模；tokenizer 的 ID 必须一致。输入最多 8+8 个 token，不超出 32 个位置。
- DPO 数据需要同题 prompt / chosen / rejected；`loss_type="sigmoid"` 指向本章讨论的标准形式，不能随意换成另一个 loss 名称再声称目标未变。
- GRPO 的每题生成数是 4。该固定版本要求总 batch size 可被它整除；这里单进程 batch=4，符合算术约束，不意味着已经验证训练能启动。
- GRPO 需要与样本对应的奖励函数或奖励模型，还需要模板、padding、EOS、生成截断和 reward 解析测试。一个随机模型可能所有输出都得零分，配置合法不等于学习有效。
- `use_cpu=True`、关闭半精度和 vLLM、`report_to=[]` 只表达运行意图。没有安装并运行完整框架，就没有 CPU 兼容性、性能或训练结果。

尤其不能预期 TRL 与教学脚本逐数相同。[该版本 GRPOTrainer 源码](https://github.com/huggingface/trl/blob/v0.16.1/trl/trainer/grpo_trainer.py) 的具体实现使用组内 `std` 默认修正，即样本标准差，分母为 G−1；奖励标准化稳定项为 10⁻⁴；loss 按全 batch 的有效 token 数归约，并使用采样 KL 估计。本章则采用总体标准差、10⁻⁸、每答案均权、精确条件 KL。**名字相同不保证变体相同，源代码与版本才是复现账本的一部分。**

未来要运行 TRL，应新建独立环境锁定整套依赖，不要为这段未运行配置改变 P20 已验证的环境。至少保存：完整版本、随机种子、数据和模板、采样参数、每组奖励、mask、旧策略保存时点、有效长度、KL 的方向与估计方式、参数更新证据。只有真正跑完，才能把 NOT RUN 改成 EXECUTED。

## P22.9 奖励投机：学会拿分，不等于学会任务

把奖励设计为“输出中含数字 2 就得分”，那么“答案不是 2，其实是 999”也能拿分；如果奖励只检查格式，模型可能输出漂亮的格式和错误的内容。把答案写长当作奖励，则可能得到冗长重复。这里的问题不一定出在优化器，往往出在目标与真正需求之间的缝隙。

本章精确匹配答案规避了上述“包含即可”的漏洞，却也有局限：“二”与“2”可能表达同一个数，严格字符串匹配却不同。更复杂的数学、代码、事实问题还涉及等价表达、执行隔离、超时、测试覆盖和标注争议，不能用两题匹配器替代。

做评估时至少分开记录：

1. 训练奖励及其各组成部分，而不只看总分。
2. 未用于训练与挑参数的题目上的正确率或盲评偏好率。
3. 不同任务、长度、语言、难度上的退化情况。
4. 答案长度、重复、格式通过率、无效输出率及采样设置。
5. KL、奖励方差、零方差组比例、clip 比例，并说明各自统计口径。
6. 多个种子的变化和评价者不确定性；对奖励模型，还需关注它是否在训练分布之外被利用。

比较方法时固定题集、token 预算与采样预算。某方法尝试 64 次后选出最好答案，不能直接与另一个方法只答一次的正确率并列作公平结论。小样本奖励变好值得继续检查，不是宣告“已对齐”或“会推理”。

## P22.10 自测与可执行验证

配套 `tests/test_preference_optimization.py` 将题目中的关键契约变成测试。不是让实现调用自己再比较，而是使用独立的标量公式、闭式导数和有限差分：

| 自测题 | 应当得到的结论 | 自动检查 |
| --- | --- | --- |
| 当前策略等于参考，DPO loss 是多少？ | 每对 ln 2，不是 0 | 标准库手算与 PyTorch 梯度 |
| B 样本 chosen 仍低于 rejected，为什么 z>0？ | 与参考相比的相对概率已经改善 | 两对独立概率比公式 |
| 对数概率求和能换成平均吗？ | 不能冒称仍是原始序列 DPO | 变长 mask、shift、求和突变 |
| [0,1,1,2] 用总体标准差怎么算？ | 约 [−√2,0,0,√2]；epsilon 略缩放 | 交错 prompt 分组与标准差突变 |
| 所有奖励一样时一定没有任何更新吗？ | 奖励优势为零；非零 KL 仍可更新 | β=0 与 β>0 的梯度对照 |
| 为何负优势 clip 要取 min 而非 max？ | 最大化目标必须保留较保守收益 | 正负优势、多 ratio 分段标量 oracle |
| 长 2 与长 4 的答案权重是多少？ | 本章各 1/2，不是按 token 数加权 | 不同长度 loss 与有限差分 |
| 同一文本在不同题目的奖励能否交换？ | 不能；sample/prompt/reward 一起对齐 | 错序、漏项、错题号与重复 ID 拒绝 |
| KL 为何不只看采到的动作？ | 精确分布 KL 需要所有动作及概率权重 | 独立 KL 值、方向与闭式梯度 |
| 参数真的按这些目标更新了吗？ | SFT、DPO、GRPO 的最终 logits 可独立复算 | 标量 SGD 与完整 GRPO 有限差分复算 |

测试另外检查示例调用前后的 CPU 随机数状态与线程数，并注入一次中途异常验证恢复路径；将 reference 和原 SFT 对象同时改动的反例，必须被独立快照检查识别。

测试还将错误真正注入可执行源码：翻转 DPO 符号、反转 β、删参考、混组、错标准差、漏对齐、错误 shift / mask / 长度平均、反转或丢掉 KL 权重、让 old 每步跟随 current、错误 clip、全 token 归约等。测试必须抓住这些行为改变，不能只检查源码里有没有某个词。

在上面的基线环境实测，21 个测试通过；不加载第三方包时，7 个标准库测试通过、14 个 PyTorch 测试明确跳过。测试通过说明这份微型实现满足所声明的契约，不说明 RLHF 安全性、真实人类偏好或大模型能力已被验证。

本章可以带走的一句话是：**偏好、奖励、策略、参考和评估是不同角色，先把每个数字属于谁说清楚，再让优化器动手。** 下一章会把这一框架接到推理预算、MTP 和投机解码，并继续区分训练目标与推理时计算。

## P22.11 延伸：on-policy 蒸馏与多教师合并（MOPD） {#p22.11-mopd}

**先看动机。** 普通蒸馏是 off-policy 的：老师先写好答案，学生照着做 SFT。学生自己生成时，一旦走到老师没写过的前缀，就没人教了。On-policy 蒸馏反过来：让学生自己采样，再请老师给学生写下的**每一个 token** 打分。这个思路的公开讲解见 [Thinking Machines Lab，On-Policy Distillation（2025）](https://thinkingmachines.ai/blog/on-policy-distillation/)，K3 报告的 §4.1.3 引用了它。

**Kimi K3（S29 §4.1）。** 先做 SFT。再按三个领域（通用任务、通用 agent、编程 agent）各训一个 RL 专家。每个领域又分 low、high、max 三档推理强度，共 9 个专家。最后用 MOPD 把它们合进一个模型。领域 d、强度 e 的样本，由对应的那位老师打分。逐 token 奖励是 Eq.15：

`r(y_t) = clip( sg( log π_teacher^{(d,e)}(y_t | x, y_<t) / π_θ(y_t | e, x, y_<t) ), −R_max, R_max )`。

`sg` 是 stop-gradient：这个比值只当作奖励数字，不让梯度穿过它。clip 限制极端的优势信号，让 RL 更稳。报告还说，他们试过更细的 top-k 蒸馏目标，在他们的设置下没有看到明显优势。

**Nemotron 3.5 Super（S30）。** 官方文档列出 6 类专科老师：agent、推理、视觉工具使用、视频推理与跟踪、视觉推理、对齐（RLHF）。MOPD 分两个阶段，每个 prompt 交给拥有该领域的老师。其余 prompt 交给一个冻结的 self-anchor。文档说明，RLVR 第 120 步是 MOPD 第一阶段的学生。self-anchor 是这个学生起始 checkpoint 的冻结副本，用作没有专科老师认领的 prompt 的老师。

[`../code/23b_mopd_reward.py`](../code/23b_mopd_reward.py) 在一个 4 词表上手算：

| token | 老师 | 学生 | 原始 log 比 | clip(R=1) |
|---|---|---|---|---|
| 0 | 0.70 | 0.25 | +1.0296 | +1.0000 |
| 1 | 0.20 | 0.25 | −0.2231 | −0.2231 |
| 2 | 0.05 | 0.25 | −1.6094 | −1.0000 |
| 3 | 0.05 | 0.25 | −1.6094 | −1.0000 |

不 clip 时，按学生分布求期望，正好等于 `−KL(学生 ‖ 老师)` = −0.603100。所以这是一个逐 token 的反向 KL 信号。测试还检查：学生与老师一致时奖励为 0，所以起点处的 self-anchor 给出零奖励；没认领的领域会路由到 self-anchor；把比值写反，符号就翻转。本书没有跑任何蒸馏训练。

P22.11 资料（核验 2026-10-26）：S29 [Kimi K3 技术报告，arXiv:2607.24653](https://arxiv.org/abs/2607.24653) §4.1.1–4.1.3；S30 [Nemotron 3.5 Super 官方训练文档](https://docs.nvidia.com/nemotron/latest/nemotron/super35/README.html)（厂商文档）。K3 §4.1.3 同时引用 [MiMo-V2-Flash，arXiv:2601.02780](https://arxiv.org/abs/2601.02780) 与 [DeepSeek-V4，arXiv:2606.19348](https://arxiv.org/abs/2606.19348) 的多教师蒸馏做法，本书未逐条核对这两篇的细节。

## P22.12 前沿补充：学生等于老师，梯度却不为 0 {#p22.12-softmax-cancellation}

> [!NOTE]
> 本节依据 [A Fix for Softmax and Cross-Entropy](https://arxiv.org/abs/2610.31642) 的 §3 与 §7，以及[项目页](https://jonbarron.info/softmaxfix/)的代码。本书只做单行梯度演示，没有复现蒸馏训练。

**问题从哪来。** 蒸馏常用软目标交叉熵：老师分布 $p=\operatorname{softmax}(z_T)$，学生分布 $q=\operatorname{softmax}(z_S)$。它对学生 logits 的梯度是 $q-p$。它与 KL$(p\Vert q)$ 只差一个与学生无关的常数，所以梯度相同。

学生接近老师时，$q$ 和 $p$ 是两个几乎相等的数。浮点里相减，有效位会大量抵消。论文把这叫 Softmax Cancellation（§3）。学生完全等于老师时，真梯度是 0，算出来却可能不是。

**修法。** 论文不先算两个概率再相减，而是从 logit 差 $\delta=z_S-z_T$ 出发（Eq. 8–9）：

1. 在 $p$ 下中心化：$d=\delta-\sum_j p_j\delta_j$。
2. 求 $s=d-\operatorname{log1p}\left(\sum_j p_j\operatorname{expm1}(d_j)\right)$，它等于 $\log(q/p)$。
3. 求 $u=p\cdot\operatorname{expm1}(s)$，它等于 $q-p$。
4. 令 $g=u-q\sum_j u_j$。精确算术下梯度和为 0；浮点下这一步把残差压到舍入量级。

`log1p` 和 `expm1` 专为接近 0 的输入设计，不会先加 1 再减 1。$\delta=0$ 时每一步都是精确 0。项目页代码多了一个分支：$\lvert s\rvert\ge1$ 时直接用 $q-p$。arXiv 正文的式子没有写这个分支。

**跑一下。** `code/23c_softmax_cancellation.py` 按论文 §3 的算法（Eq. 8–9）用 PyTorch 重新实现，并注明出处。前向 loss 不变，只替换梯度。项目页没有标注代码许可证，所以本书代码与注释均为自写，不复制原代码。

论文例子是 $z_T=[2,1,0]$，$z_S=z_T+[-10^{-7},10^{-7},0]$，fp32：

| 实现 | 学生梯度 |
| --- | --- |
| 论文 §3 末尾 JAX 代码示例：标准版 | [-5.960e-8, 1.452e-8, -1.123e-9] |
| 本书：PyTorch 标准版 | [-5.960e-8, 4.470e-8, 7.451e-9] |
| 修正版（两者相同） | [-4.596e-8, 4.144e-8, 4.513e-9] |
| fp64 精确值 | [-4.596e-8, 4.144e-8, 4.513e-9] |

两个标准版的误差不同，因为舍入顺序依赖框架实现。修正版与 fp64 精确值一致，也与论文 §3 末尾 JAX 代码示例注释中的数字一致（arXiv v1 HTML 版，核对日期 2026-10-26；表中为其 4 位有效数字舍入）。

再取固定种子 0，生成 4 行 8 类的 bf16 logits，令学生等于老师。标准版最大梯度约 3.5e-3，修正版恰好为 0。论文报告的残差量级是：fp64 约 1e-17，fp32 约 1e-8，bf16 约 1e-3（§7）。本书的 bf16 实测与之同一量级。

**边界。** 论文在 bf16 下观察到学生持续漂离老师（§7），本书没有复现。on-policy 蒸馏与多教师合并另见 [P22.11 的 MOPD 小节](#p22.11-mopd)，本书不在这里重复。`tests/test_softmax_fix.py` 固定输入断言上表的修正版数字、fp64 下梯度和在 1e-14 以内、去掉第 4 步后残差变大，以及学生等于老师时的精确 0。
