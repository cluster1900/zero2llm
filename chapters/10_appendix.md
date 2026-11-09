# 附录：术语、数学与证据更新（P28） {#p28-reference}

[]{#附录-数学公式速查卡与深度学习黑话翻译词典}
[]{#附录数学公式速查卡与深度学习黑话翻译词典}

> 这不是要背完的最后一章，而是读书时随手翻的工具箱。先修只有前言：遇到公式查附录 A，遇到术语查附录 B，不知道下一步读什么查附录 C，准备运行实验查附录 D，看到新模型新闻查附录 E，自测查附录 F，来源查附录 G，版本与日期查附录 H。公式手算、自测与固定配置证据由仓库测试核对；本页不宣布任何未运行硬件的性能。

## 附录 A：大模型核心数学公式速查卡 {#附录-a大模型核心数学公式速查卡}

### 表 A-1：数学知识点与大模型机制对应速查表 {#表-a-1数学知识点与大模型机制对应速查表}

| 概念 | 公式 | 用在哪里与必要条件 |
|---|---|---|
| 向量 | $x=(x_1,\ldots,x_d)$ | embedding 的一行；坐标不是天然有名字的语义属性 |
| 长度 | $\|x\|_2=\sqrt{\sum_i x_i^2}$ | 衡量数值大小，不能单凭长度判断模型理解程度 |
| 点积 | $x\cdot y=\sum_i x_i y_i$ | 注意力打分；两向量最后一维相同 |
| 余弦 | $\cos\theta=(x\cdot y)/(\|x\|_2\|y\|_2)$ | 比较方向；零向量时分母为零，需要约定或拒绝 |
| 稳定 softmax | $p_i=e^{z_i-z_{\max}}/\sum_j e^{z_j-z_{\max}},\ z_{\max}=\max_j z_j$ | 减共同最大值不改变概率，避免大正数指数溢出；全 mask 行另行处理 |
| 交叉熵 | $\mathcal L=-\log p_y$ | 对正确目标的自然对数损失，单位 nat；有效目标数决定平均分母 |
| 均值/方差 | $\mu=d^{-1}\sum_i x_i,\quad v=d^{-1}\sum_i(x_i-\mu)^2$ | LayerNorm 的总体方差，不用 $d-1$ |
| LayerNorm | $y_i=\gamma_i(x_i-\mu)/\sqrt{v+\epsilon}+\beta_i$ | 归一化后还有可学习仿射；有 epsilon 时方差不严格等于 1 |
| RMSNorm | $y_i=\gamma_i x_i/\sqrt{d^{-1}\sum_j x_j^2+\epsilon}$ | 不减均值；不能用 LayerNorm 的输出当参考 |
| 方差相加 | $\mathrm{Var}(X+Y)=\mathrm{Var}(X)+\mathrm{Var}(Y)$ | 独立或协方差为零时成立；一般还要加 $2\mathrm{Cov}(X,Y)$ |
| 缩放注意力 | $\mathrm{softmax}(QK^\top/\sqrt{d_k}+M)V$ | 行是 query，列是 key；禁看位置的加性 mask 为负无穷 |
| 周期 | $\sin(\omega t),\quad \lambda=2\pi/\lvert\omega\rvert$ | 正弦位置编码的频率与波长；频率非零才有这个有限周期（这里 $\omega$ 是局部角频率，$\lambda$ 是对应波长，与 A.5 的 $\omega_i,\lambda_i$ 同族） |
| 和角 | $\cos(a+b)=\cos a\cos b-\sin a\sin b$ | 说明相对位移的三角关系，不自动保证长度外推质量 |
| 平面旋转 | $R(\theta)=\begin{pmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{pmatrix}$ | RoPE 在成对维度旋转；相邻配对与 split-half 约定不能混用 |
| 链式法则 | $\partial\mathcal L/\partial w=(\partial\mathcal L/\partial z)(\partial z/\partial w)$ | 反向传播沿计算图传递导数 |
| SGD | $w_{t+1}=w_t-\eta\nabla\mathcal L(w_t)$ | 参数更新，和梯度清零是不同动作 |

这些公式在正文中都有推导，旋转约定详见 [P12 RoPE](#p12-rope)，归一化和门控详见 [P13](#p13-modern-block)。注意力原式见 [Vaswani 等 1706.03762v7 §3.2][attention]；本表的简单算术是教学推导，不是某个真实模型的测评。

**其中（表 A-1 记号）：** $\mathcal L$ 是损失；$z_i$ 是第 $i$ 个 logit，$z_{\max}$ 是同一行里最大的 logit（减它只为数值稳定，不改概率）；$\omega$ 是局部角频率，$\lambda=2\pi/\lvert\omega\rvert$ 是对应波长；$Q$、$K$ 是查询、键矩阵，$^\top$ 表示转置；$M$ 是加性 mask；$y$ 是正确目标的编号，$p_y$ 是模型给它的概率；$d_k$ 是单头 Q/K 宽度；$d$ 是向量长度，对 LayerNorm/RMSNorm 即模型宽度 $D$。**白话：** 这张卡是翻书时的速查，字母已与 A.5 对齐；细节推导仍回各章。

### A.2 一张形状卡：先写轴，再乘矩阵

令 batch $B=2$、长度 $T=3$、query 头数 $H_q=4$、KV 头数 $H_{kv}=2$、每头维度 $d_h=8$。本例另设模型宽度 $D=32=H_q d_h$；这是这个小 GQA 玩具的选择，不是所有真实配置的恒等式：

| 对象 | 形状 | 元素数 |
|---|---|---:|
| Q | `[2,4,3,8]` | 192 |
| 未展开的 K 或 V | `[2,2,3,8]` | 96 |
| 逻辑注意力分数 | `[2,4,3,3]` | 72 |
| 拼回模型宽度的输出 | `[2,3,32]` | 192 |

Q 的下标 0、1 的头共享 KV 下标 0 的头；Q 下标 2、3 的头共享 KV 下标 1 的头。对单层未展开 K+V，float32 payload 是 $2\times96\times4=768$ 字节，不是分数矩阵的大小。这里不包含 allocator、索引、量化 scale 或工作区，详见 [P14](#p14-grouped-kv)。换轴后元素数可能不变，所以“shape 对了”不是全部正确性证明。

其中：$B$、$T$、$H_q$、$H_{kv}$、$d_h$、$D$ 见 A.5；$96=B\cdot H_{kv}\cdot T\cdot d_h$ 是未展开的单份 K（或 V）元素数；乘 2 是 K+V，再乘 4 是 float32 的 $b_{\mathrm{elem}}$。**白话：** 先数清轴上有几个数，再乘每元素字节，得到的是 cache 载荷，不是整网显存。

### A.3 三个可以手算的校验点

**向量。** $a=(3,4)$、$b=(4,0)$：长度分别 5 和 4，点积 12，余弦 $12/20=0.6$。

**概率。** logits 为 $[\log1,\log2,\log3]$，softmax 是 $[1/6,1/3,1/2]$。目标是第三类时，交叉熵 $-\log(1/2)=\log2\approx0.693147$。三个 logits 都加 1000 后结果不变，直接算 $e^{1000}$ 却会溢出。这正是“数学等价”和“浮点算法可靠”要分开检查的地方。

**梯度。** 只有一个权重，$z=wx$、$\mathcal L=(z-y)^2/2$。取 $w=1,x=2,y=0$，则 $z=2,\mathcal L=2$，$\partial\mathcal L/\partial w=(z-y)x=4$。学习率 0.1 的一步 SGD 得到 $w'=0.6$，新损失 $\mathcal L'=0.72$。这个例子只说明这一小步下降，不保证任意学习率或非凸网络每步都下降。

### A.4 别把三种平均混在一起

两个文档分别有 1 和 3 个有效目标，文档平均 loss 为 2 和 4：

- 文档等权平均是 $(2+4)/2=3$；
- token 等权平均是 $(1\times2+3\times4)/4=3.5$；
- 训练中每步各自求平均再更新，并不等于把四个 token 合成一大步。

报告必须写明分母。padding、prompt-only SFT 位置以及文档跨界 label 都可能被忽略，但 attention mask 和 loss mask 仍是两件事，见 [P19](#p19-pretraining)。

### A.5 全书符号约定：一个字母，一种意思 {#p28.a5-symbols}

公式里的字母就像地图上的图例：图例统一了，换一章也不会迷路。下表是全书的**全局符号**，每个符号只表示一种东西；每条公式后面的「其中：」会再提醒一次它在这里指什么。本表是各章公式**要改成的写法**；某章还没改写完成时，以该式的「其中：」为准。附录自己的 A.1、A.2、D.1、F.0 已按本表改写（F9）：交叉熵写 $\mathcal L$，稳定 softmax 的最大值写 $z_{\max}$，波长写 $\lambda=2\pi/\lvert\omega\rvert$，注意力写 $QK^\top$，A.2 每头宽度写 $d_h$，D.1 写 $\mathrm{Mem}_{KV}$ 与 $d_h$、$b_{\mathrm{elem}}$，F.0 平均损失写 $\bar{\mathcal L}$；E.0 的 $\Delta$ 本就表示差值，保持不变。

**尺寸与计数**

| 符号 | 唯一含义 | 常见形状或取值 | 不要和谁混 |
|---|---|---|---|
| $B$ | 批大小：一次并排处理几条序列 | 例：$B=2$ | 不作缩放定律里的常数，也不作 LoRA 矩阵名 |
| $T$ | 序列长度：一条序列有几个 token；做 KV cache 时是已缓存的 token 数 | 例：$T=3$ | 转置写 $^\top$；温度写 $\tau$；周期写 $\lambda$ |
| $T_q,\ T_{kv}$ | 同一个式子里同时出现两种长度时：本次 query 的个数、可读 key/value 的个数 | 增量解码时 $T_q=1$ | 只有一种长度时直接写 $T$ |
| $T_{\mathrm{train}}$ | 训练时见过的最长上下文长度（位置外推时用） | 例：$T_{\mathrm{train}}=8$ | 不再写成 $L$ |
| $T_{\mathrm{win}}$ | 滑动窗口长度（含当前位置自己） | 例：$T_{\mathrm{win}}=3$ | 不写 $W$（$W$ 只指权重） |
| $D$ | 模型宽度：每个位置的总特征数，代码里叫 `d_model` | 例：$D=8$ | 正文公式统一写 $D$；KL 散度写 $\operatorname{KL}$，数据并行份数写 $N_{\mathrm{dp}}$ |
| $\mathcal V$ / $\lvert\mathcal V\rvert$ | 词表（全部 token 的集合）/ 词表大小 | embedding 表 $E\in\mathbb R^{\lvert\mathcal V\rvert\times D}$ | 大写 $V$ 只指 Value |
| $H$ | 注意力头数（GQA 中指 query 头数，也可写 $H_q$） | 例：$H=2$ | FFN 隐藏宽度不用 $H$；P17 的层表示 $H^{(l)}$ 改为 $X^{(l)}$ |
| $H_{kv}$ | KV 头数：MHA 时 $H_{kv}=H$，MQA 时为 1 | $1\le H_{kv}\le H$ | 正文写数学符号 $H_{kv}$，不写代码标识符 `H_kv` |
| $d_h$ | 单个头的宽度（Q/K/V 各头等宽时） | 例：$d_h=4$ | 不再写 $d$、$D_{head}$、$d_{\mathrm{head}}$ |
| $d_k,\ d_v$ | 单头注意力里 Q/K 与 V 的宽度（第 3 章）；多头且等宽时 $d_k=d_v=d_h$ | $W_Q\in\mathbb R^{D\times d_k}$ | |
| $d_{\mathrm{ff}}$ | FFN 中间层宽度；要比较两种 FFN 时写 $d_{\mathrm{ff}}^{\mathrm{GELU}}$、$d_{\mathrm{ff}}^{\mathrm{SwiGLU}}$ | 常见 $d_{\mathrm{ff}}=4D$ | 不用 $H$、$M$ |
| $L$ | 层数；编码器、解码器分别为 $L_e$、$L_d$ | 例：$L=2$ | 损失写 $\mathcal L$，长度写 $T$ |
| $N_E$ | MoE 的路由专家个数；第 $e$ 个专家记 $F_e$ | 路由矩阵按 $[N_E,D]$ 存，打分 $z=xW_r^\top$ | $E$ 只指 embedding 表 |
| $N_{\ast}$ | 带下标的 $N$ 一律表示“个数”：$N_S$ 共享专家数、$N_{\mathrm{valid}}$ 有效目标数、$N_{\mathrm{dp}}$ 数据并行份数、$N_{\mathrm{GELU}}$ 等参数个数 | 例：$N_{\mathrm{valid}}=2$ | 正文不再用不带下标的 $N$ |
| $N_{\mathrm{param}},\ N_{\mathrm{tok}}$ | 整个模型的参数元素个数（缩放定律与 ZeRO 等处共用；embedding 计入，见 P19/08i 玩具账本）、训练 token 数 | $\mathcal L(N_{\mathrm{param}},N_{\mathrm{tok}})$ | 不借用 $N$、$D$；MoE 模块内的 $N_{\mathrm{total}}$/$N_{\mathrm{active}}$ 另有下标 |
| $b_{\mathrm{elem}}$ | 每个张量元素占几个字节；带文字下标的 $b$ 都是字节数（如 ZeRO 的 $b_p,b_g,b_o$） | float32 为 4，float16/bfloat16 为 2 | 不用 $s$；FFN 偏置写 $b_1,b_2$ |

**头宽与模型宽度的关系要带条件。** 只有在「各头等宽，且所有头拼起来的投影总宽等于 $D$」时，才有 $D=Hd_h$、$d_h=D/H$，输出投影 $W_O$ 才是 $[D,D]$ 的方阵。一般情况下 $W_O$ 的形状是 $[Hd_v,D]$（原论文写作 $hd_v\times d_{\mathrm{model}}$）。反例就在本书里：附录 E.2 的 Qwen3 配置 $D=1024$、16 个头、每头 128，Q 投影宽 $16\times128=2048\neq D$；P25 的微型配置也刻意保留 $H_qd_h\neq D$。

**张量与运算**

| 符号 | 唯一含义 | 形状 |
|---|---|---|
| $E$ | embedding 表，第 $i$ 行是 token $i$ 的向量 | $[\lvert\mathcal V\rvert, D]$ |
| $X$、$X^{(l)}$ | 一层的输入表示（隐藏状态），一行一个位置；$X^{(l)}$ 是第 $l$ 层的输入，其中第 $i$ 行记 $x_i^{(l)}$ | $[T,D]$，带 batch 时 $[B,T,D]$ |
| $Q,\ K,\ V,\ O$ | 查询、键、值矩阵与注意力输出；小写 $q_i,k_j,v_j,o_i$ 是它们的第 $i$、$j$ 行 | 单头 $[T,d_k]$、$[T,d_k]$、$[T,d_v]$、$[T,d_v]$ |
| $W_Q,\ W_K,\ W_V,\ W_O$ | Q/K/V 投影与输出投影；第 $h$ 个头的写 $W_h^Q$ 等 | $W_Q:[D,d_k]$；$W_O:[Hd_v,D]$，等宽且总宽为 $D$ 时为 $[D,D]$ |
| $W_{\mathrm{gate}},\ W_{\mathrm{up}},\ W_{\mathrm{down}}$ | 门控 FFN 的三块权重 | $[D,d_{\mathrm{ff}}]$、$[D,d_{\mathrm{ff}}]$、$[d_{\mathrm{ff}},D]$ |
| $s_{ij}$ | 第 $i$ 个 query 对第 $j$ 个 key 的缩放分数；一行里的最大值记 $s_{\max}$ | 标量，整张表 $[T,T]$ |
| $M$ | 加性 mask：允许看的位置为 0，禁看的位置为 $-\infty$ | $[T,T]$ |
| $p_i$ | 概率：softmax 输出的第 $i$ 项 | 标量，各项和为 1 |
| $u$ | 注意力子层加完残差后的结果，$u=x+\operatorname{Attention}(\cdot)$ | 与 $x$ 同形 |
| $\mathcal L$ | 损失（默认对有效目标取平均的交叉熵） | 标量 |
| $\Omega$ | 参与损失的有效位置集合，$N_{\mathrm{valid}}=\lvert\Omega\rvert$ | 由 $(b,t)$ 组成 |
| $\tau$ | 温度：采样温度或注意力 logits 的温度 | 标量，$\tau>0$ |
| $\omega_i$、$\lambda_i$ | 第 $i$ 对维度的角频率（每前进一个位置转多少弧度）、对应波长 $\lambda_i=2\pi/\omega_i$ | 位置 $t$ 的转角是 $t\omega_i$ |
| $\kappa$ | 位置插值、YaRN 的上下文扩展倍数 | 例：$\kappa=4$ |
| $\epsilon$ | 防止除以零的小常数 | 例：$10^{-5}$ |
| $\eta$ | 学习率 | 例：$0.1$ |

**改名对照：各章旧写法换成什么**

| 出现在哪里 | 旧写法 | 新写法 |
|---|---|---|
| 第 3 章、P15 稳定 softmax 的最大值 | $m$ | $s_{\max}$（对 logits $z$ 做时写 $z_{\max}$） |
| 第 1 章、P20、P23 的词表大小（含形状表） | $V$、`[B,T,V]`、`[V,D_model]` | $\lvert\mathcal V\rvert$；形状写 $[B,T,\lvert\mathcal V\rvert]$、$[\lvert\mathcal V\rvert,D]$ |
| 第 5 章的位置、最大位置数 | $pos$、$L_{max}$ | $t$、$T_{\max}$ |
| 第 5 章相对位置里的两个位置 | $p$、$q$ | $m$、$n$ |
| 第 6 章残差要拟合的目标映射 | $H(x)$ | $\mathcal H(x)$ |
| 第 7 章采样温度 | $T$ | $\tau$ |
| P11、P12、P14、P24、P25、P27 的 KV 字节式 | $M_{KV}$、$s$、$b_{KV}$、$d$、$D_{head}$ | $\mathrm{Mem}_{KV}=2\cdot L\cdot B\cdot H_{kv}\cdot T\cdot d_h\cdot b_{\mathrm{elem}}$，$b_{\mathrm{elem}}$ 按每个张量元素计 |
| 各章每头宽度 | $d$ | $d_h$ |
| P12 RoPE 的位置 | $p$ | $t$；讨论两个位置的相对距离时用 $m$、$n$ |
| P12 RoPE 每对维度的频率、YaRN 混合后的频率 | $\theta_i$、$\theta'$；P12.3 不带下标的 $\theta$（$R(m\theta)$、例子里的 $\theta=1$） | $\omega_i$、$\omega'$（与第 5 章一致；原论文写作 $\theta$），转角 $t\omega_i$；只看一对维度时写 $R(m\omega_i)$、$\omega_i=1$；第 5 章正文写 $\omega_k$，因为该章 §5.2 已用 $i$ 作 query 位置，维度对编号改用 $k$，与表中 $\omega_i$ 是同一类量 |
| P12 YaRN 的扩展倍数 | $s$ | $\kappa$（原论文写作 $s$） |
| P12 YaRN 的原训练长度 | $L$ | $T_{\mathrm{train}}$ |
| P12 YaRN 的注意力温度 | $t$ | $\tau$ |
| P13 SiLU 的自变量 | $\operatorname{SiLU}(t)=t\sigma(t)$ | $\operatorname{SiLU}(g)=g\sigma(g)$（$g=xW_{\mathrm{gate}}$ 正是 SiLU 的输入；$z$ 仍表示乘积 $\operatorname{SiLU}(g)\odot a_{\mathrm{up}}$；$t$ 只作位置编号） |
| P13 FFN 中间宽度 | $H$、$M$ | $d_{\mathrm{ff}}^{\mathrm{GELU}}$、$d_{\mathrm{ff}}^{\mathrm{SwiGLU}}$ |
| P13、P26 门控 FFN 的权重与 up 分支 | $W_g,W_u,W_d$；$u=xW_u$ | $W_{\mathrm{gate}},W_{\mathrm{up}},W_{\mathrm{down}}$；$a_{\mathrm{up}}=xW_{\mathrm{up}}$（$u$ 只指残差结果） |
| P14 MLA 的 KV 压缩权重 | $W_{down}$ | $W_{\mathrm{DKV}}$（$W_{\mathrm{down}}$ 只指 FFN） |
| P15、P17 的序列长度 | $N$ | $T$（两种长度同时出现时用 $T_q$、$T_{kv}$） |
| P15、P24 TPOT 式里的输出 token 数（同时是最后一个时刻的下标） | P15：$t_1\ldots t_n$、$(t_n-t_1)/(n-1)$；P24：$t_1\ldots t_N$、$(t_N-t_1)/(N-1)$ | $t_1\ldots t_{T_{\mathrm{out}}}$、$(t_{T_{\mathrm{out}}}-t_1)/(T_{\mathrm{out}}-1)$，$T_{\mathrm{out}}>1$ |
| P16、P17 的窗口长度 | $W$、$n_{win}$ | $T_{\mathrm{win}}$ |
| P17 有界重放的段起点 | $s$ | $i_0$ |
| P17 的层表示 | $H^{(l)}$、$h_i^{(l)}$ | $X^{(l)}$、$x_i^{(l)}$ |
| P17 的编码器输出 | $E$ | $X_{\mathrm{enc}}$ |
| P18 的专家个数 | $E$ | $N_E$ |
| P18 的共享专家个数与编号 | $S$、$s$ | $N_S$、$e'$（共享专家记 $F^{\mathrm{sh}}_{e'}$） |
| P18 的 top-k 选中集合 | $K$ | $\mathcal I_k$ |
| P18 摊平后的 token 数 | $N$ | $BT$ |
| P18、P21、P25 的参数个数 | $P_{\ast}$ | $N_{\ast}$（如 $N_{\mathrm{total}}$、$N_{\mathrm{LoRA}}$） |
| P19 Chinchilla 式 | $L(N,D)\approx E+A/N^\alpha+B/D^\beta$ | $\mathcal L\approx c_{\mathrm{irred}}+c_{\mathrm{param}}/N_{\mathrm{param}}^{\alpha}+c_{\mathrm{tok}}/N_{\mathrm{tok}}^{\beta}$ |
| P20 的有效位置集合、有效目标数 | $\mathcal V$、$N$ | $\Omega$、$N_{\mathrm{valid}}$ |
| P20 ZeRO 显存式 | $M_k$、$P$、$D$ | $\mathrm{Mem}_k$、$N_{\mathrm{param}}$、$N_{\mathrm{dp}}$ |
| P21 SFT 损失的 mask 与分母 | $m_{b,t+1}$、$N$ | 对 $\Omega$ 求和，$N_{\mathrm{valid}}$ |
| P21 LoRA 的两块矩阵与乘法方向 | 列向量写法：$W_0\in\mathbb R^{d_{\mathrm{out}}\times d_{\mathrm{in}}}$，$A\in\mathbb R^{r\times d_{\mathrm{in}}}$，$B\in\mathbb R^{d_{\mathrm{out}}\times r}$，$\Delta W=\frac{\alpha}{r}BA$，$y=W_0x+\frac{\alpha}{r}B(Ax)$ | $W_A:[r,d_{\mathrm{in}}]$，$W_B:[d_{\mathrm{out}},r]$，$\Delta W=\frac{\alpha}{r}W_BW_A$，与 $W_0:[d_{\mathrm{out}},d_{\mathrm{in}}]$ 同形；行向量前向写 $y=x(W_0+\Delta W)^\top=xW_0^\top+\frac{\alpha}{r}(xW_A^\top)W_B^\top$（两项都要转置，不能把 $x\Delta W$ 加到 $xW_0^\top$ 上） |
| P22 的 KL 散度 | $D_{\mathrm{KL}}$ | $\operatorname{KL}$ |
| P22 GRPO 的奖励标准差、回复长度、回复个数 | $s$、$L_i$、$N$ | $\sigma_r$、$T_i$、$N_{\mathrm{resp}}$ |
| P22 GRPO 损失里的逐 token KL 项（本章枚举整个词表算出的**精确**条件 KL） | $K_{i,t}$ | $\operatorname{KL}_{i,t}$（不加帽子） |
| P22 原论文的逐 token KL **估计量**（单个样本，不是精确 KL） | $k$ | $\widehat{\operatorname{KL}}_{i,t}$（只有估计量加帽子） |
| P22 `grpo_loss` 的张量形状 | `[N,L,V]`、`[N,L]`、`[N]` | $[N_{\mathrm{resp}},T,\lvert\mathcal V\rvert]$、$[N_{\mathrm{resp}},T]$、$[N_{\mathrm{resp}}]$（这里的 $T$ 是填充后的长度，每份回复的有效长度 $T_i\le T$） |
| P22 KL 估计里的“动作 / 状态” | $a$、$s$ | $y_t$、$(x,y_{<t})$ |
| P23 MTP 的额外 head 个数（深度） | $D$ | $n_{\mathrm{mtp}}$ |
| P23 第 $j$ 个 horizon 的有效位置集合 | $M_j$ | $\Omega_j$ |
| P23 的辅助损失权重 | $\lambda$ | $w_{\mathrm{mtp}}$（$\lambda$ 只指波长） |
| P23 的 head 乘法 | $W_jh_t$ | $h_tW_j^\top$（$W_j$ 按 $[\lvert\mathcal V\rvert,D]$ 存） |
| P24 量化步长（格点间距） | $s$、$s_x$、$s_w$ | 量化 $x$ 时写 $\Delta_x$，量化权重时写 $\Delta_w$；步长一律带下标，不写裸 $\Delta$（裸 $\Delta$ 留给差值） |
| 各章损失 | $L$、$L_{CE}$、$L_{aux}$、$L_j$ | $\mathcal L$、$\mathcal L_{\mathrm{CE}}$、$\mathcal L_{\mathrm{aux}}$、$\mathcal L_j$ |

**下标与写法**

- 编号字母（主要当编号用）：$i$、$j$ 是 query、key 的位置，$t$ 是时间步或当前位置，$m$、$n$ 是一对位置（第 5 章、P12 和第 9 章 RoPE 例子里的“座位 $m$、$n$”都是这个意思），$h$ 是第几个头，$e$ 是第几个专家，P23 的预测距离（horizon）也用 $j$，$l$ 是第几层，$b$ 是 batch 里第几条。指数里的 $e$ 仍是自然常数。有两处例外要认得：$i$ 在 RoPE、YaRN 里也是维度对的编号（第 $i$ 个频率对，$\omega_i$、$\lambda_i$ 的 $i$）；$h$ 作下标（如 $W_h^Q$）是第几个头，单独出现的 $h$、$h_t$ 则是章内局部量（见下面的局部字母）。带文字下标的小写 $n$ 表示个数（如 $n_{\mathrm{mtp}}$、P18 的 $n_e$）；带下标的 $d$ 都是某种宽度（如 MLA 的 $d_c$、$d_r$，LoRA 的 $d_{\mathrm{in}}$、$d_{\mathrm{out}}$）。
- 向量按**行向量**写，线性层写 $xW$；形状用方括号 `[B,T,D]`，从左到右是轴的顺序。权重若按 $[\text{输出},\text{输入}]$ 存（如 MoE 路由 $W_r$、MTP head $W_j$、LoRA 的 $W_0$），乘的时候写转置 $xW^\top$。
- 转置统一写 $A^\top$；Softmax 统一写 $\operatorname{softmax}$，沿最后一维（key 所在的维）归一化；$\log$ 默认是自然对数；期望写 $\mathbb E$；单位矩阵写 $I$。

**局部字母：每条公式自己说清楚**

有些字母在不同章节各管一摊，硬统一反而难读。它们是**局部符号**：只在所在公式和紧跟的「其中：」里有效，翻到下一条公式要重新看「其中：」，不要把上一章的意思带过来。

- 希腊字母：$\sigma$（标准差，或 sigmoid 函数）、$\alpha$（LoRA 缩放、辅助损失系数、接受概率、缩放定律指数等）、$\beta$（LayerNorm 偏移、DPO/KL 系数、缩放定律指数）、$\gamma$（LayerNorm 缩放，或 YaRN 的过渡系数）、$\theta$（夹角；A.1 与第 9 章通用二维旋转矩阵 $R(\theta)$ 的转角，RoPE 里代入 $\theta=m\omega_i$，P12 直接写 $R(m\omega_i)$；或模型参数 $\pi_\theta$、$p_\theta$）、$\pi$（圆周率，或策略 $\pi_\theta$）、$\phi$（旋转角、特征映射，或奖励模型参数）、$\rho$（概率比）、$\delta$、$\Delta$（不带下标的 $\Delta$ 只表示差值，如 P12 的 $\Delta=n-m$、E.0 的 $\Delta=x_{\mathrm{new}}-x_{\mathrm{old}}$；$\Delta W$ 是更新量；P22 DPO 的 $\Delta_\theta$、$\Delta_{\mathrm{ref}}$ 是两个对数比的差；量化步长只写带下标的 $\Delta_x$、$\Delta_w$）、$\mu$（均值）。
- 拉丁字母：$x$（某个式子的输入：一个位置的向量、它的第 $i$ 个分量，或 P21–P23 里的 token 序列）、$y$（输出向量、目标 token，或 P22 里的回复；$y_l$ 是 DPO 的 rejected 答案，下标 $l$ 不是层号）、$h$（第 2 章 RNN 的记忆状态 $h_t$、第 6 章 FFN 激活前的中间结果、P23 主干最后一层的隐藏向量 $h_t$）、$a$（第 6 章 ReLU 的输出、P12 YaRN 的长度缩放系数、第 1 章的示例向量、第 3 章的注意力权重 $a_{ij}$、P15 的请求到达时刻）、$b$（第 1 章、P12 的示例向量）、$f$（第 2 章 RNN 的更新函数、P18 的分配比例 $f_e$）、$g$（SwiGLU 的门控分支、P20 的梯度）、$r$（LoRA 秩、奖励 $r_\phi$、频段比值、特征宽度、第 3 章的求和下标）、$k$（第 5 章的维度对编号（$\omega_k$，即表中的 $\omega_i$；该章位移改记 $\Delta$）、top-k 的个数、P20 的训练步编号 $\theta_k$）、$q$（P23 的草稿分布 $q(x)$、P24 的量化整数）、$z$（多数时候是 softmax 之前的 logits；也作 SwiGLU 的门控乘积、线性注意力的归一化量、DPO 的间隔、P24 的零点）、$w$（单个权重、路由权重 $w_e$、注意力权重 $w_{tj}$）、$c$（第 2 章的上下文向量、P14 MLA 的压缩向量 $c_t$）、$t_1,t_{\mathrm{start}}$ 等 P15、P24 的计时时刻、$A$（注意力子层函数、GRPO 的优势 $A_i$）、$F$（第 6 章的残差分支 $F(x)$）、$P$（第 5 章的换序操作、P15 的概率矩阵）、$C$、$G$、$R$、$S$、$U$ 等只在一章出现的大写字母。
- 第 2 章 $h_t$、第 6 章 $h$、P23 $h_t$ 都不是全局的层表示；逐层的隐藏状态统一写 $X^{(l)}$。

## 附录 B：深度学习常见“黑话”大白话翻译词典 {#附录-b深度学习常见黑话大白话翻译词典}

### 表 B-1：AI 行业黑话通俗翻译词典 {#表-b-1ai-行业黑话通俗翻译词典}

| 中文 | 英文 | 简明含义与容易混淆的地方 |
|---|---|---|
| 张量 | Tensor | 带 shape/dtype/device 的多维数组；轴的含义由程序规定 |
| 词元 | Token | tokenizer 产生的单位，可是一段字节、字符或片段，不保证一个汉字一个 token |
| 嵌入 | Embedding | ID 查表后的向量；不是把 ID 大小当作语义距离 |
| 损失 | Loss | 给定目标函数下的数值；更小不自动代表更有帮助或更安全 |
| 梯度 | Gradient | 每个参数的局部变化方向；不是参数本身 |
| 反向传播 | Backpropagation | 按链式法则计算导数；不等于优化器已更新参数 |
| 超参数 | Hyperparameter | 学习率、层数等外部设定；也可通过搜索选择 |
| 过拟合 | Overfitting | 对训练集适应过度而泛化不佳；必须先检查划分、标签分布与评估协议 |
| 欠拟合 | Underfitting | 模型/训练过程还未充分拟合目标规律；并非只看一个 loss 数字判断 |
| 自回归 | Autoregressive | 用已知前缀预测后继；训练并行计算不代表可以偷看未来 |
| 参数量 | Parameter count | 按声明的去重/绑定规则数权重元素；不是智商、文件大小或激活 FLOPs |
| 激活参数 | Activated parameters | 一次计算实际使用的参数口径，MoE 尤其要说明是否包含 embedding |
| 对齐 | Alignment | 使行为更符合目标与偏好的训练/评估过程；不保证自动解决事实性与安全问题 |
| 预训练 | Pretraining | 在大量数据上学习基础预测能力；后训练通常从已有权重出发 |
| 监督微调 | SFT | 在示例输入与目标上继续优化；哪些 token 计 loss 必须明确 |
| 低秩适配 | LoRA | 用低秩增量训练部分权重变化；不是自动量化，也不是免除前向计算 |
| 偏好优化 | Preference optimization | 用相对偏好信息优化；DPO、奖励模型、GRPO 不是同一算法 |
| 预填充/解码 | Prefill / Decode | 处理已有 prompt / 逐步产生新 token；不要与 [P17](#p17-ced) 的 CED（Causal Encoder-Decoder，因果编码器—解码器）架构中的两段网络混同 |
| 缓存 | KV cache | 保存历史 K/V；所有者、权重、位置、dtype 必须匹配，不能混入另一请求 |
| 贪心/采样 | Greedy / Sampling | 选最大概率 / 按分布抽样；greedy 等价不证明随机采样分布等价 |
| 精度/量化 | Precision / Quantization | 数值表示与映射规则；权重、激活、KV 可分别使用不同格式 |
| checkpoint | Checkpoint | 保存的模型/训练状态；精确恢复还可能需要优化器、随机数与数据位置 |
| 消融 | Ablation | 尽量只改一个变量的对照；换模型宽度又换数据不能归因给某个模块 |
| 基准测试 | Benchmark | 固定任务、输入、计时与评分协议的比较；一次 smoke 不是 benchmark |
| 首 token 延迟 | TTFT | 从请求到达至首 token 的时间，单位必须给出 |
| 后续 token 间隔 | TPOT | 首 token 后的平均间隔；一个 token 时没有此平均，不能硬除以零 |

## 附录 C：后记与初学者进阶书单推荐 {#附录-c后记与初学者进阶书单推荐}

你已经从向量和概率走到了可测试的小模型。下一步不必追求一次读完最多论文：选一个现象，写下预测，做一个能推翻自己预测的实验，再解释差异。

### 进阶推荐阅读清单 {#进阶推荐阅读清单}

1. [Attention Is All You Need，1706.03762v7][attention]：先读 §3 的形状与公式，再看 §5 的训练设定。论文里的质量数字属于作者报告，不是本书重新测得。
2. [3Blue1Brown 的 attention 原作者讲解][visual]：用于建立直觉，再回到正文手算和源码检验。视频讲解不是模型配置的权威来源。
3. [Andrej Karpathy 的 nanoGPT 仓库][nanogpt]：读模型与训练循环的边界。该项目会更新；本书没有执行它的全套训练，也不把其 README 中的硬件用时作为本书测量。查看时记录 commit，勿把浮动链接当精确复现实验版本。
4. 需要更具体的问题时，回到 [P25](#p25-qwen3) 与 [P26](#p26-deepseek) 的固定配置与论文节号，不从一个模型名推断它的所有版本。

读论文时可以画四栏：**问题、假设、方法、证据**。例如“减少 KV 内存”是问题，“共享哪些层/哪些头”是方法，“同长度同 dtype 的字节数”是一个证据；“因此所有机器生成更快”仍是未经测试的外推。

## 附录 D：运行与硬件：先做小实验，再决定是否扩展 {#p28.d-运行与硬件先做小实验再决定是否扩展}

本书的必需路线是小规模 CPU 教学，不要求购买 GPU。安装命令以仓库 README 与固定 requirements 为准；按平台选安装包时查 [PyTorch 官方安装页][install]。这个动态页面的核对日期见附录 H，不是永远不变的安装承诺。

### D.1 KV cache 载荷速查

对未量化的 K/V 张量，按实际保存的头数计算：

$$\mathrm{Mem}_{KV}=2\cdot L\cdot B\cdot H_{kv}\cdot T\cdot d_h\cdot b_{\mathrm{elem}}.$$

其中：$L$ 是层数，$B$ 是批大小，$H_{kv}$ 是 KV 头数，$T$ 是已缓存长度，$d_h$ 是每头宽度，$b_{\mathrm{elem}}$ 是每元素字节数；乘 2 是因为 K 与 V 各存一份。**白话：** 这是 KV cache 张量自己的字节数，不是整进程显存峰值。

用 P27 教学配置 $L=2,B=2,H_{kv}=2,T=8,d_h=4,b_{\mathrm{elem}}=4$（float32）代入：$2\cdot2\cdot2\cdot2\cdot8\cdot4\cdot4=2048$ bytes。把 GQA 的 $H_{kv}=2$ 改成 MHA 的 $H_{kv}=4$，同一账本得到 4096 bytes。这只核算缓存张量 payload，可与 `numel × element_size` 核对，不代替进程内存峰值。

| 你要做的事 | 先运行什么 | 不能据此声称什么 |
|---|---|---|
| 学数学与证据提取 | 标准库手算、P24/P26 默认 CLI | 不能说 Torch 模型前向已运行 |
| 检验模型模块 | CPU Torch、小 batch/短序列、固定 dtype | 不能推断 GPU 吞吐或 BF16 质量 |
| 对照第三方框架 | 独立依赖环境与明确选修入口 | 安装成功不等于训练/恢复/多卡通过 |
| 测 GPU 性能 | 单独设计记录设备/版本/预热/同步/重复次数的实验 | 本书未运行的 GPU 格子必须写“未运行” |

最粗的权重 payload 是“参数数 × 每参数字节”。100 万个 float32 参数约 4,000,000 字节，即 3.814697 MiB；不是 4 MiB，更不是训练峰值内存。梯度、优化器状态、激活、缓存、临时工作区与框架分配器还要另算。P20 的 16 字节 Adam 账本是一种假设，不适用于所有优化器或分片方式。

出现 OOM 时，先减 batch/序列/层宽，记下变化，不要一开始就把精度、注意力实现和数据一起改掉。若结果不同，先固定 dtype、train/eval、随机数与输入，再检查 mask/位置。PyTorch 2.14 的[复现说明][repro]明确区分随机源控制与跨版本/平台一致性；设 seed 不保证所有硬件逐位相同。本书给 tiny float64 MoE 1e-12 的特定因果容差，不意味着任何模型都能直接复用这个阈值。

**不要为了跑通教程关闭浏览器 sandbox 或忽略证书警告。** 正常构建受限时，说明实际错误、使用有源码匹配检查的备用制品路线，并请有条件的环境补验；“未验证”比伪造 PASS 更有用。

## 附录 E：看到一条新模型新闻，怎样不被标题带着走 {#p28.e-看到一条新模型新闻怎样不被标题带着走}

### E.0 证据变化率速查

比较两个固定版本的同一字段时，先把绝对差和相对差分开：

$$\Delta=x_{\mathrm{new}}-x_{\mathrm{old}},\qquad r=\Delta/x_{\mathrm{old}}.$$

其中：$\Delta$ 是新旧字段的差（裸 $\Delta$ 只表示差值，见 A.5）；$x_{\mathrm{new}}$、$x_{\mathrm{old}}$ 是同一字段的两个固定取值；$r$ 是相对变化。**白话：** 先问“变了多少”，再问“相对原来占几成”，两者不要混成一句广告语。

例如旧配置最大长度为 40960，新版本为 32768，则 $\Delta=-8192$，$r=-8192/40960=-0.20$，即减少 20%。这个结果只描述已锁定字段；硬件速度或模型质量需要另外的测量记录。

### E.1 四种状态，四种不同承诺

| 状态 | 允许写什么 | 不允许偷换成什么 |
|---|---|---|
| 已核实 | 固定官方配置/源码确实包含某字段或机制 | 该机制在所有版本都有、或必然带来某种质量 |
| 作者报告 | 论文/模型卡在指定设定下报告某结果 | 本书已重复得到该结果 |
| 本书复现 | 本书实际运行并记录输入、环境、误差/指标 | 更大模型、更长上下文或另一硬件也相同 |
| 待核实 | 目前证据不足、来源不可用或版本不明 | “大概如此”写成确定事实 |

一个组件可以同时有多项不同状态。例如配置字段已核实，而“该模型在我的 GPU 上快一倍”仍待核实。状态附在**具体主张**上，不是给整个品牌盖一个“可信”章。

### E.2 完整演示：审查“Qwen3 的每头维度就是 hidden_size / heads”

这是一个待检验的说法，不把标题当作事实。我们采用 [P25](#p25-qwen3) 已固定的 Qwen3-0.6B 对象，不趁机引入另一个未批准的新模型。

**第一步，找官方对象。** 官方 repo 是 `Qwen/Qwen3-0.6B`，固定 revision `c1899de289a04d12100db370d81485cdf75e47ca`。使用[此 revision 的 config.json][qwen-config]，不只给 `main` 链接。

**第二步，锁定原始字节。** 仓库 `code/data/qwen3/manifest.json` 保存获取日期、URL、长度和 SHA-256。本配置为 **726 字节**，SHA-256 为 `660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd`。哈希证明当前文件与已记录字节相同，不证明来源自己没有错误。

**第三步，做差异审查。** 文件写的是 hidden_size=1024、num_attention_heads=16、head_dim=128。1024/16=64，与显式 128 不同；Q 投影宽度是 $16\times128=2048$（K/V 各为 $8\times128=1024$），不能擅自把 Q 投影改成 1024。新 revision 即使文件长度不变也可能改字段，必须同时检查 hash 和语义 diff，不能只看文件大小。

**第四步，小实验验证。** 在仓库根目录运行：

```bash
python code/26_qwen3_case.py
python -m unittest discover -s tests -p 'test_reference_appendix.py' -v
```

第一条仅读取固定小文件并打印账本；第二条用标准库重新核对原始字节、上述三个字段、形状/概率/梯度卡和本节自测。若另运行 P25 `--torch` 或官方参考测试，记录它们的独立依赖和实际结果。没有下载真实权重，所以不能把此处称作“复现 Qwen3 的问答能力”。

**第五步，提交可复审的 PR。** 把具体主张改成：“这个固定配置使用显式 head_dim=128；tiny 数值对照范围见 P25。”附来源版本、原始 hash、字段 diff、运行命令、失败负例与未运行范围。新结论先在独立分支审核，不直接改 main。若发现旧结论错了，应在变更记录写明为什么错，而不是悄悄替换文件后留住旧测试数字。

这个示例已完成“已核实配置＋本书标准库算术”的闭环；它没有生成一个新模型的性能结论。

### E.3 失效链接不是删除证据的理由

链接打不开时，先记录日期与错误，判断是临时网络问题、页面迁移还是 revision 删除。优先找同一官方仓库的固定文件或同一论文版本；保留原URL、替代URL和变更原因。只能找到二手转载时把状态降为待核实，不用转载给原主张重新背书。外链 HTTP 200 也只说明能打开，不说明内容支持当前句子。

维护记录至少包含：日期、对象/revision、主张、证据路径、验证命令、环境、通过/失败/跳过、限制、对应 PR。既往正常浏览器检查属于既往 head，不能复制到新 head。源代码、README、章节状态、HTML/EPUB 与验证报告应同批更新；旧的来源与版本信息集中保留在[附录 H](#p28.h-版本与来源)。

## 附录 F：自测、校验与最后一步 {#p28.f-自测校验与最后一步}

1. A.2 的两头 KV 若换为四头，单层 float32 K+V payload 变为多少？它证明生成速度减半吗？
2. A.3 的三个 logits 都加 1000，第三类交叉熵是否仍为约 0.693147？为什么不能直接指数？
3. 两文档有效目标数为 1、3，平均 loss 为 2、4，token 平均和文档平均分别是多少？
4. A.3 单权重若学习率改为 1，更新后权重与 loss 是多少？
5. 一个配置写 head_dim=128，而 hidden_size/heads=64，哪个值应进入这份实现的投影形状？文件等长是否能证明未篡改？
6. 新闻说“模型快两倍”，你只跑了标准库参数账本。用哪种状态描述这条速度主张？最少还缺什么？

**答案。** 1）1536 字节，只是 payload 翻倍，不能推出速度。2）相同；减最大值后指数安全，直接指数可能溢出。3）3.5 与 3。4）$w'=-3$、$\mathcal L'=18$，大步长使本例更差。5）该固定实现使用显式 128，需核对代码；等长不等于等字节，仍需 hash。6）对本书而言待核实；若官方论文确有该结果，可另标作者报告，必须给硬件、软件、输入长度/batch、比较基线与计时方法，不能写本书复现。

### F.0 平均值与一步更新速查

$\bar{\mathcal L}_{\mathrm{doc}}=(2+4)/2=3$，而 $\bar{\mathcal L}_{\mathrm{token}}=(1\times2+3\times4)/4=3.5$。学习率 $0.1$ 后 $w'=1-0.1\times4=0.6$，新损失 $\mathcal L'=(0.6\times2)^2/2=0.72$。这两个分母和更新顺序是核对训练日志的最小账本。

仓库根目录的 `python -m unittest discover -s tests -v` 检查示例与契约，`python scripts/check_book.py` 检查已生成制品的资源/链接，`npm run check:layout` 检查正常浏览器视口布局。公式正确、链接存在、图片可读与代码通过互不替代。具体通过与跳过数以当前 CI 运行结果为准，运行环境版本见[附录 H](#p28.h-版本与来源)，不在这里保留会迅速过时的全书总数。

学习的终点不是背熟每个缩写，而是能说清：**我改了什么，为什么这样改，哪些证据支持它，还有什么没有验证。**

## 附录 G：本页原始来源与版本 {#p28.g-本页原始来源与版本}

来源校验把长度和摘要同时锁定。对固定配置字节串计算 SHA-256，并记录 `(bytes, sha256, revision)`。本地 `code/data/qwen3/dense/config.json` 是 726 bytes，摘要为 `660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd`。把最后一个字节从 125 改为 124（异或 1），长度仍为 726 bytes，但摘要变为 `e338758cad0bc78e2ac6b2e298c0e621b78a6d2e80bac7df11df3a828bf8b73e`；同长度、不同内容必须同时查 hash。

- [Attention Is All You Need，arXiv:1706.03762v7][attention]，§3.2 attention 定义；历史书单保留，非新性能复现。
- [PyTorch 2.14 reproducibility][repro]，固定文档版本；[官方安装选择器][install]为动态页，核对日期见附录 H。
- [Qwen3-0.6B 固定官方 config][qwen-config]，revision/hash 如 E.2；相关官方实现版本见 P25。
- [3Blue1Brown 原作者 attention 页面][visual]与 [nanoGPT 原作者仓库][nanogpt]：阅读入口，核对日期见附录 H，不当作本书实验版本。

[attention]: https://arxiv.org/abs/1706.03762v7
[repro]: https://docs.pytorch.org/docs/2.14/notes/randomness.html
[install]: https://pytorch.org/get-started/locally/
[qwen-config]: https://huggingface.co/Qwen/Qwen3-0.6B/resolve/c1899de289a04d12100db370d81485cdf75e47ca/config.json
[visual]: https://www.3blue1brown.com/lessons/attention/
[nanogpt]: https://github.com/karpathy/nanoGPT

## 附录 H：版本与来源 {#p28.h-版本与来源}

下表保留从正文移出的日期与版本。种子、线程数、dtype 和数值结果仍在原章。正在讲授的库版本锁定留在原处，不计入本表。

| 章 | 内容 | 版本 / 日期 |
| --- | --- | --- |
| 00 | 来源核验日期 | 2026-10-18 |
| 00 | CI 环境 | Python 3.12、PyTorch 2.14.1（CPU 版） |
| 01 | 来源核验日期 | 2026-10-18 |
| 01a | 核验和运行日期 | 2026-10-18 |
| 03 | 动态文档核验日期 | 2026-10-19 |
| 04 | 文档核验日 | 2026-10-19 |
| 05 | 来源核验日期 | 2026-10-19 |
| 06 | 核验环境 PyTorch | 2.14.1+cpu |
| 06 | 访问核验日期 | 2026-10-19 |
| 07 | 文档核验日期 | 2026-10-19 |
| 07a | 复核记录日期 | 2026-10-19 |
| 07a | 复核记录 PyTorch | 2.14.1+cpu |
| 07a | 来源核验日期 | 2026-10-19 |
| 07a | 运行数值 PyTorch | 2.14.1+cpu |
| 08 | 运行环境 PyTorch | 2.14.1+cpu |
| 08 | 来源核验日 | 2026-10-19 |
| 08a | 来源核验日期 | 2026-10-19 |
| 08a | 访问核验日 | 2026-10-19 |
| 08b | 验证环境 PyTorch | 2.14.1+cpu |
| 08b | 来源核验日期 | 2026-10-19 |
| 08c | 实验条件 PyTorch | 2.14.1+cpu |
| 08c | 来源核对日期 | 2026-10-19 |
| 08d | Python | 3.12.14 |
| 08d | PyTorch | 2.14.1+cpu |
| 08d | 动态链接核验日 | 2026-10-19 |
| 08d | 固定版本核验日 | 2026-10-19 |
| 08e | Python | 3.12.14 |
| 08e | PyTorch | 2.14.1+cpu |
| 08e | 导入检查 torch | 2.14.1 |
| 08e | CPU 测试环境 torch | 2.14.1+cpu |
| 08e | 来源核验日 | 2026-10-19 |
| 08f | Python | 3.12.14 |
| 08f | PyTorch | 2.14.1+cpu |
| 08f | 来源与官方入口核验日期 | 2026-10-19 |
| 08g | 验证 torch | 2.14.1 |
| 08g | 技术报告与固定 revision 核验日期 | 2026-10-19 |
| 08h | 实测日期 | 2026-10-19 |
| 08h | 实测 PyTorch | 2.14.1+cpu |
| 08h | 固定论文版本核验日期 | 2026-10-19 |
| 08i | 实测日期 | 2026-10-19 |
| 08i | Python | 3.12.14 |
| 08i | torch | 2.14.1+cpu |
| 08i | 来源核验日 | 2026-10-19 |
| 08i | 数据版本指纹（SHA-256） | `1b1a8d64d76a38761d8166a145550efdab89a852874994c19be8333947c75705` |
| 08j | 实测日期 | 2026-10-19 |
| 08j | 来源核验日期 | 2026-10-19 |
| 08k | 实测日期 | 2026-10-19 |
| 08k | 实测 PyTorch | 2.14.1+cpu |
| 08k | 来源核验日期 | 2026-10-19 |
| 08k | TRL 源码快照（只读，未执行） | 0.18.2 SFTTrainer；P20 引 0.17.0 官方总览，P22 引 0.16.1 GRPO 源码，分别固定资料快照，不是同一套环境，未作跨版本兼容验证 |
| 08l | 实测日期 | 2026-10-19 |
| 08l | 实测 PyTorch | 2.14.1+cpu |
| 08l | TRL 源码核验日期 | 2026-10-19 |
| 08l | TRL 源码快照（只读，未执行） | 0.16.1 DPOConfig / GRPOConfig / GRPOTrainer，作为固定的历史 GRPO 实现切片，不是推荐退回旧环境 |
| 08m | 已验证环境 PyTorch | 2.14.1+cpu |
| 08m | 跨版本核对截止 | 2026-10-19 |
| 08m | 来源核验日期 | 2026-10-19 |
| 08n | 固定历史版本核验日 | 2026-10-19 |
| 08o | --torch 环境 PyTorch | 2.14.1+cpu |
| 08o | 来源核对日期 | 2026-10-19 |
| 09 | 核查日期 | 2026-10-19 |
| 09a | 重新运行日期 | 2026-10-30 |
| 09a | 重新运行 Python | 3.13.5 |
| 09a | 重新运行 PyTorch | 2.14.1+cpu |
| 09a | 来源核验日 | 2026-10-20 |
| 10 | 安装页核对日期 | 2026-10-20 |
| 10 | 安装选择器核对日期 | 2026-10-20 |
| 10 | 阅读入口核对日期 | 2026-10-20 |
