# 训练进阶（P20）：从手写循环到训练框架 {#p20-frameworks}

在 [P19：预训练数据与训练流程](#p19-pretraining) 中，我们已经能说明一批文本怎样变成输入、标签和一次参数更新。现在换一个问题：**当这些工作交给框架，哪些东西只是换了名字，哪些东西真的改变了？**

初学者常有两种相反的困惑。一种是“我会写 backward，为什么还需要 Trainer？”另一种是“示例能启动，就证明训练正确了吧？”本章既不要求你重写工业训练系统，也不把配置文件当咒语。我们先把一个 2,704 参数的小模型接上真正的 Transformers，再用手写计算检查它，最后才谈多卡。

配套 [code/21_training_frameworks.py](../code/21_training_frameworks.py) 有两条互不冒充的路线：

- **必修机制实验：**只用 Python 标准库，逐 rank 计算 DDP 与 ZeRO 的持久模型状态字节数。
- **可安装的 CPU 框架实验：**用固定版本 Transformers + Accelerate，运行随机初始化小 GPT-2 的前向、3 次真实 Trainer 更新、独立 loss / 更新对照，以及本地保存和重新加载。

第二条不会下载任何预训练权重或 tokenizer；不是 GPT-2 语言能力复现。DeepSpeed、Megatron 和多 GPU 训练**没有在本章运行**，不能把内存算术结果当成它们的显存或速度实测。完成后，你应当能读懂一个训练入口，并指出它的标签、优化器、并行方式和检查点到底由谁负责。

## P20.1 框架没有取消训练循环

先回想最朴素的一步：取 batch，前向得到 logits，算 loss，反向得到梯度，再让 optimizer 更新参数。训练框架主要把这个流程及周边工作组织起来：搬运设备、分发数据、累积梯度、记录日志、保存状态、恢复进度。

**训练目标仍由标签和 loss 决定，不由框架名字决定。** 如果标签错移两次，换成 Trainer 不会让目标自动正确。如果验证集泄漏，换成多卡只会更快地完成一次有问题的实验。

本章用下表做翻译，而不是背 API 清单：

| 手写代码中的职责 | Transformers 中的主要对象 | 不应误以为它会代办 |
| --- | --- | --- |
| 模型的结构参数 | config，例如 GPT2Config | 自动提供训练好的权重 |
| 神经网络和前向运算 | model，例如 GPT2LMHeadModel | 自动理解任何 tokenizer 的词表 |
| 文本转 ID、特殊 token、补齐 | tokenizer | 自动决定任务该监督哪些位置 |
| 一条条样本 | dataset | 自动保证无泄漏、无重复、来源合法 |
| 合并样本为 batch | data collator | 永远使用正确的 padding 和 label mask |
| 学习率、步数、保存策略等 | TrainingArguments | 替你选择合理的实验预算 |
| 训练、评估、保存的控制流程 | Trainer | 证明模型学到了泛化能力 |

[Transformers 4.51.3 的 Trainer 文档](https://huggingface.co/docs/transformers/v4.51.3/en/trainer) 把它定位为训练与评估循环，并说明它通过 Accelerate 处理设备和分布式能力。可以把 Trainer 看成“把你已经理解的步骤可靠串起来的调度者”，不要把它当成替代理解的黑盒。

![训练框架地图与 ZeRO 状态切分：数据和 tokenizer 进入模型，Trainer 控制更新；分片只改变持久状态的分布](assets/diagrams/p20-frameworks.svg)

这里还有一条重要边界：本书之前的 miniGPT 是用来逐层解释机制的代码主线；本章换用 GPT2LMHeadModel，是学习成熟 API 的独立实验。两者没有自动共享权重。即使都叫 Transformer，也不能把旧 checkpoint 直接塞进新模型，然后宣称转换成功。

## P20.2 config、model、tokenizer：三样东西不能混成一样

### config 是结构说明，不是模型的记忆

本章的关键配置很小：

```python
config = GPT2Config(
    vocab_size=12, n_positions=16, n_ctx=16,
    n_embd=16, n_layer=1, n_head=2, n_inner=32,
    resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
    bos_token_id=2, eos_token_id=3, pad_token_id=0,
    use_cache=False, attn_implementation="eager",
)
model = GPT2LMHeadModel(config)
```

vocab_size 是输出分类的类别数；n_embd 是每个 token 的表示宽度；n_layer 是 Transformer block 数；n_head 是注意力头数。16 维分成 2 个头，每头 8 维。n_inner=32 是 FFN 的中间宽度。位置表容纳 16 个位置，本实验输入最长只有 6。

**从 config 构造 model 得到随机权重，不是下载一个缩小的预训练 GPT-2。** config 说明“神经网络长什么样”，训练后的参数才承载学到的信息。这一区别在 [GPT2Config 的固定版说明](https://huggingface.co/docs/transformers/v4.51.3/en/model_doc/gpt2#transformers.GPT2Config) 中也很明确。

本模型仍是 GPT-2 风格的绝对位置 embedding 和相应 block，不是把 P12 的 RoPE、P13 的 RMSNorm / SwiGLU 自动装进去。我们关掉 dropout，是为了让本章的数值对照尽量只考察训练接口，不是在建议真实训练永远不用 dropout。

### tokenizer 是 ID 契约，不只是切字工具

为了完全离线，脚本用 Tokenizers 的 WordLevel 构造一个 12 项词表。它只按空白分隔，输入必须像“我 爱 学习 。”这样显式隔开。这是本地测试夹具，**不是 GPT-2 的 byte-level BPE，也不是一个合格的通用中文 tokenizer**。

```text
ID:       0       1       2       3     4  5   6    7    8     9    10  11
token:  [PAD]   [UNK]   [BOS]   [EOS]   我 爱 学习  模型 预测 下一个 词  。
```

模板在句首加 BOS，在句尾加 EOS。两个样本补齐后为：

```text
input_ids[0] = [2, 4, 5, 6, 11, 3]
input_ids[1] = [2, 7, 8, 10, 3, 0]
attention_mask[0] = [1, 1, 1, 1, 1, 1]
attention_mask[1] = [1, 1, 1, 1, 1, 0]
```

因此 input_ids 的形状为 [B,T]=[2,6]。模型输出 logits 的形状是 [B,T,V]=[2,6,12]：每个输入位置各给出一个“下一个 token”的 12 类预测。tokenizer 的词表大小、特殊 token ID、模型 embedding 行号必须一致；把两个相同大小但 ID 排列不同的词表互换，形状检查仍可能通过，语义却已经错了。

unknown 词会变成 ID 1。测试要求“陌生词”得到 [2,1,3]，也要求保存后加载的词表和固定句子 ID 不变。这比只检查“tokenizer.json 文件存在”更有意义。

## P20.3 框架已经 shift 了，就不要再 shift 一遍

GPT2LMHeadModel 接受 labels 后会在内部完成 causal language modeling 的错位：位置 t 的 logits 对应位置 t+1 的标签。我们的 labels 和 input_ids 保持相同形状，只把不该监督的位置设为 -100：

```text
labels[0] = [-100, 4, 5, 6, 11, 3]
labels[1] = [-100, 7, 8, 10, 3, -100]
```

- PAD 不是真实文本，所以第二行末尾不进入 loss。
- EOS 是真实的结束目标，仍然进入 loss；不能因为它“特殊”就一并忽略。
- 第一列没有前一个输入位置来预测它。本例显式将它设为 -100，下面会解释一个与版本有关的原因。
- attention_mask 控制注意力可见性；labels=-100 控制监督位置。二者职责不同，一个不能代替另一个。

独立参考计算不调用模型返回的 loss，也不调用框架的交叉熵函数，而是自己写：

$$
\mathcal L=\frac{1}{N}\sum_{(b,t)\in\mathcal V}
\left[\log\sum_v\exp(z_{b,t,v})-z_{b,t,y_{b,t+1}}\right],
$$

其中有效集合 $\mathcal V$ 只包含下一位置标签不是 -100 的预测位置，N 是这些位置的数量。本例第一行 5 个目标、第二行 4 个目标，所以 N=9，不是输入张量的 12 个格子。

代码先取 logits[:, :-1] 与 labels[:, 1:]，筛掉忽略项，再用 logsumexp 减去目标 logit。测试还用 Python 标量循环、稳定的指数求和独立算一次，避免两个实现都复制同一个索引错误。

### 一个真实发现的版本细节

固定版本 4.51.3 的 Trainer 在某条 loss 路径中会统计**未 shift 的 labels 中非 -100 的数量**，作为 num_items_in_batch 传给模型。若保留两行首位 BOS 标签，这个数是 11，而真正参与 next-token loss 的目标只有 9，梯度会被错误缩小为本例正确值的 9/11。

这不是说所有版本、所有模型都有同一个问题，而是本章用的组合需要明确约定。这里把首位本就不会被预测的标签也设为 -100，让框架计数和实际目标对齐。对应实现可核对 [4.51.3 Trainer 的 get_batch_samples 与 compute_loss 源码](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/trainer.py)。测试故意恢复首位 BOS，要求独立更新对照报错，而不是只接受一个仍然下降的 loss。

**因此，loss 在下降不够，程序退出码是 0 也不够。先对齐目标、分母和梯度，再谈训练结果。** 改到其他版本、加入 packing、梯度累积或多卡时，要重新检查有效 token 计数，不能把本例补丁无条件推广。

## P20.4 真正运行一次 Trainer，并保留可解释的对照

本章把 TrainingArguments 的变量尽量减少：CPU、float32、固定种子 20、每批 2 条样本、3 次更新、学习率 0.02、SGD、常数学习率、无 weight decay、无梯度裁剪、不向外部日志服务上报、不向 Hub 推送。

为什么不直接用 AdamW？不是说 SGD 更适合大模型，而是本实验要先回答“Trainer 是否执行了同一个更新”。SGD 的式子很简单：

$$
\theta_{k+1}=\theta_k-0.02\nabla_\theta\mathcal L_k.
$$

我们从相同的初始权重复制两份模型：

1. 第一份进入真正的 Trainer，执行 train()。
2. 第二份进入手写循环，用上一节的独立 loss、backward 和 torch.optim.SGD 更新。
3. 每份都执行 3 步；两个样本每步都在同一个完整 batch 中，所以打乱行顺序不改变目标，只可能有很小的浮点舍入差。
4. 逐参数比较结果，同时检查 global_step、参数是否真的变化、loss 是否有限并下降。

独立对照参数使用 atol=2e-6、rtol=2e-5；不是把所有训练轨迹都要求逐位相同。本地保存后以**同一个 eager attention 后端**恢复的 logits 则要求零误差。在开发时，恢复入口曾自动选择 SDPA，造成约 6e-8 的微小差异；解决方法是固定同一个后端，而不是不问原因就放宽容差。

2026-10-19，在本章下述固定组合、Linux CPU 环境得到的一次结果为：

| 检查项 | 实际结果 |
| --- | --- |
| 模型可训练参数 | 2,704 |
| logits 形状 / 有效目标数 | [2,6,12] / 9 |
| Trainer 更新步数 | 3 |
| 更新前 / 后 loss | 2.51291084 / 2.34189987 |
| 模型 loss 与独立参考差 | 0 |
| Trainer 与手写 SGD 参数最大差 | 约 3.73e-9 |
| 参数最大变化量 | 约 0.0195597 |
| 本地恢复后 logits 最大差 | 0 |

这些数字是测试夹具的数值证据，不是语言质量、吞吐量或大模型收敛结论。Trainer 会额外打印 train_runtime 和 samples_per_second，它们受进程启动、机器负载等影响，本章不拿它们作 benchmark。两句话被重复训练后 loss 下降，也不证明模型具备中文能力。

### 如何运行

只运行标准库账本，不需要安装 PyTorch：

```bash
python code/21_training_frameworks.py
python -m unittest discover -s tests -p 'test_training_frameworks.py' -v
```

第二条会执行标准库测试；若可选包缺失或版本不在本章验证组合中，HF 测试会明确显示 skipped。**跳过不等于 HF 通过。** 脚本默认也明确打印 HF smoke: NOT RUN，必须加 --hf 才会真正训练。

需要框架实验时，另建环境，避免为了本章升级全书原有依赖。例如在支持该 CPU wheel 的 Linux / Python 3.12 环境中：

```bash
python3.12 -m venv /tmp/zero2llm-frameworks
/tmp/zero2llm-frameworks/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
/tmp/zero2llm-frameworks/bin/python -m pip install -r requirements-frameworks.txt
/tmp/zero2llm-frameworks/bin/python code/21_training_frameworks.py --hf
/tmp/zero2llm-frameworks/bin/python -m unittest discover \
  -s tests -p 'test_training_frameworks.py' -v
```

官方 CPU wheel 的平台列表见 [PyTorch 官方索引](https://download.pytorch.org/whl/cpu/torch/)。若平台没有匹配 wheel，不要从不明镜像拼装包；可以先完成标准库部分，再选择官方支持的环境。

本章锁定 torch 2.14.1（本次 wheel 带 +cpu）、transformers 4.51.3、accelerate 1.6.0、tokenizers 0.21.4、huggingface-hub 0.30.2、safetensors 0.5.3。它是**已验证的教学组合，不是最新版推荐，也不代表任意跨版本组合都兼容**。requirements-frameworks.txt 锁定附加框架包，PyTorch 单独从官方 CPU 源安装，避免意外引入 CUDA 大包。正式归档环境还应保存完整 pip freeze、Python 版本和平台信息，因为只锁直接依赖不等于锁了全部传递依赖。

本次验证为节省重复安装空间，使用独立 /tmp/zero2llm-p20-venv，并只读复用原环境的 Torch / Tokenizers 路径；新增包写入独立环境，核对原环境版本未变。上面的干净环境命令是读者的安装路线，不要求复刻这个路径优化。

缺少可选包或版本不符时，--hf 输出 SKIP 并返回退出码 2；它不会临时联网补下载。满足版本但出现真正的运行错误时，程序正常报错，而不是吞掉异常并说“兼容性跳过”。

## P20.5 DDP：先让每张卡各做一份完整工作

rank 是分布式进程的编号；world size 是进程总数。常见 GPU 训练把一个 rank 对应一张 GPU，但这不是这些词的定义。

最容易理解的数据并行是 DDP：各 rank 持有同一模型副本，读取不同数据，反向传播时同步梯度，再按一致规则更新。它提高了可以共同处理的数据量，**没有自动把模型参数、梯度和普通优化器状态都分成几份**。DDP 也不会替你切分输入数据，数据采样器必须配合。相关职责见 [PyTorch 2.14 固定版 DDP API](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html)；这里引用其机制说明，不声称本次运行了 PyTorch 2.14 的多卡测试。

假设每个 rank 一次处理 b 条，累积 A 次后更新，数据并行组有 D 个 rank，理想情况下每次更新包含 b×A×D 条样本。若长度不一，有效监督 token 数不一定按样本数平均；还要检查各 rank 的 loss 归一化与梯度汇总口径。P19 中关于有效 token 和梯度累积的约定，并不会因为加了 DDP 就消失。

一个反直觉的结论是：单卡放不下的模型，简单复制到 4 张卡上仍可能每张都放不下。要减少每卡状态冗余，需要看下一节的分片。

## P20.6 ZeRO：分三步去掉重复的状态

训练时，模型参数不是唯一的大对象。以一种常见的低精度权重 + Adam 混合精度组织为例，每个参数元素可能对应：

- 低精度参数：2 bytes。
- 梯度：本例假设 2 bytes。
- Adam 的一阶矩 m：4 bytes。
- Adam 的二阶矩 v：4 bytes。
- FP32 master weight：4 bytes，归入优化器侧状态。

本例共 16 bytes/参数元素。**这不是所有 Adam、BF16 或 PyTorch 训练的固定规律。** 有的路径保留 FP32 梯度，有的没有单独 master copy；本章真实 CPU smoke 用的是 float32 + SGD，也不能套用这份 Adam 账本。

设参数元素数为 P，参数 / 梯度 / 优化器侧每元素字节分别为 b_p、b_g、b_o。这里 b_o=2×moment_bytes+master_bytes，默认 12。经典 ZeRO 的分级思路是：Stage 1 切优化器状态，Stage 2 再切梯度，Stage 3 再切参数。出处是 [ZeRO 原始论文 v3](https://arxiv.org/abs/1910.02054v3) 以及 [DeepSpeed v0.16.5 的 ZeRO 官方说明源码](https://github.com/deepspeedai/DeepSpeed/blob/v0.16.5/docs/code-docs/source/zero3.rst)。

| 方式 | 参数 | 梯度 | 优化器侧状态 |
| --- | --- | --- | --- |
| DDP / 本账本 Stage 0 | 每个 rank 完整保留 | 完整保留 | 完整保留 |
| ZeRO-1 | 完整保留 | 完整保留 | 分片 |
| ZeRO-2 | 完整保留 | 分片 | 分片 |
| ZeRO-3 | 分片 | 分片 | 分片 |

对于恰好整除的理想情况，每 rank 的持久状态近似为：

$$
\begin{aligned}
M_0 &= P(b_p+b_g+b_o),\\
M_1 &= P(b_p+b_g)+\frac{P}{D}b_o,\\
M_2 &= Pb_p+\frac{P}{D}(b_g+b_o),\\
M_3 &= \frac{P}{D}(b_p+b_g+b_o).
\end{aligned}
$$

Stage 3 并不是每个 rank 永远只看自己的一小片参数。执行某层计算时仍需按实现聚合所需权重，并在合适时机释放或重新切分。节省常驻冗余往往要付出通信、调度和临时空间成本。CPU/NVMe offload 又是把状态放到别的位置，不是与 Stage 1/2/3 同义的第四个数字。

### 十个参数，四个 rank，亲手算一遍

P=10、D=4 不能整除。我们的教学分配不加 padding，前两个 rank 各负责 3 个元素，后两个各负责 2 个，记作 [3,3,2,2]。这只是平衡所有权模型，不模拟 DeepSpeed 内部的 bucket / 对齐布局。

以 rank 0 为例：

- Stage 0：10×2 + 10×2 + 10×12 = 160 bytes。
- Stage 1：10×2 + 10×2 + 3×12 = 76 bytes。
- Stage 2：10×2 + 3×2 + 3×12 = 62 bytes。
- Stage 3：3×2 + 3×2 + 3×12 = 48 bytes。

全部 rank 的结果是：

| 方式 | rank 0 | rank 1 | rank 2 | rank 3 | 所有 rank 合计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage 0 | 160 | 160 | 160 | 160 | 640 |
| Stage 1 | 76 | 76 | 64 | 64 | 280 |
| Stage 2 | 62 | 62 | 48 | 48 | 220 |
| Stage 3 | 48 | 48 | 32 | 32 | 160 |

单位全部是 bytes。最大的 rank 分片应使用 ceil(P/D)=3，而不是 floor(P/D)=2。若用浮点 2.5 个参数来估算最满的 rank，会低估；若每个 rank 都实际分配 3 个槽位，则又是含 padding 的另一种账本，集群合计不能仍写 10 个槽位。

脚本用 divmod 得到商和余数，逐 rank 返回真实元素数，保证总和严格等于 P；D>P 时允许部分 rank 拥有 0 个分片元素，不能凭空多出参数。测试覆盖这些非整除和空分片边界，也检查 D=1 时四种策略完全相同。

### 账本不是显存峰值

这里明确排除：激活、模型 buffer、attention 临时张量、通信 bucket、all-gather 临时完整权重、梯度累积实现的额外副本、内存分配器保留空间、分片 padding、CPU/NVMe offload。真实峰值可能远高于持久状态的总和。

bytes、GB 和 GiB 也要区分：1 GB=10^9 bytes，1 GiB=2^30 bytes。10^9 bytes 大约是 0.9313 GiB，而不是 1 GiB。本章 byte_units 同时返回两种单位；不要把 GPU 标称容量与程序输出混用后得出“刚好能放下”的结论。

现在做一个修改练习：把 gradient_bytes 从 2 改成 4，同时把 master_bytes 改成 0。P=10、D=4 时，Stage 2 的每 rank 结果应为 [56,56,44,44] bytes。这个练习强迫我们把“梯度什么 dtype”和“优化器状态什么 dtype”分开，而不是机械背诵“每参数 16 字节”。

## P20.7 TP、PP、EP：切的不是同一条轴

ZeRO 主要回答“数据并行组中谁保存哪些训练状态”。其他并行术语回答的问题不一样：

**Tensor parallel，张量并行（TP）：**把一层里的张量运算分给多个设备。例如线性层 Y=XW，可以按 W 的输出列拆成 W_1、W_2，两个设备算出部分输出；后续运算按布局拼接或归约。拆行的方式则需要合并部分和。它不是让两个 rank 各读不同样本，而是合作完成同一层的计算。通信经常落在层内，互联条件很重要。[Megatron-LM 原始论文](https://arxiv.org/abs/1909.08053)

**Pipeline parallel，流水线并行（PP）：**把不同层放到不同设备。例如 4 层网络，设备 A 负责第 1–2 层，设备 B 负责第 3–4 层。样本的激活先从 A 传给 B，反向梯度再反方向流回。把大 batch 切为 microbatch 可以让不同阶段交错工作，但启动、收尾或阶段不均衡会产生空闲时间，称为 pipeline bubble。不是把层拆开就能永远满载。[Megatron 的大规模训练论文](https://arxiv.org/abs/2104.04473)

**Expert parallel，专家并行（EP）：**把 MoE 专家分散到不同设备，路由后把 token 发给相应专家，再把输出送回。它和 P18 的“一个 token 只激活少数专家”有关，但不是同一概念：稀疏激活描述模型计算，EP 描述专家放在哪。路由不均衡会让某些设备忙、某些设备闲；跨设备 token 交换也有成本。[Megatron-Core core_v0.12.0 的 MoE 与并行说明](https://github.com/NVIDIA/Megatron-LM/tree/core_v0.12.0)

这些策略可以组合，但**不能把所有并行度数字无脑相乘**。简单的纯 dense DP×TP×PP 布局常可按独立进程组规划；加入 EP、context parallel 或不同专家组后，要按框架实际 rank 分组规则检查哪些维度独立、哪些共享。不清楚自己的 data-parallel group 大小时，前一节的 D 也就可能填错。

## P20.8 Transformers、TRL、DeepSpeed、Megatron 应该怎么选

它们不是四个功能完全相同、只需挑“最强”的品牌。

- **Transformers：**常用模型结构、配置、tokenizer 和 Trainer 接口。本章首先用它把模型训练跑通，并建立正确性对照。
- **TRL：**在 Hugging Face 生态上提供后训练相关的训练器和数据处理入口。它与 Transformers 集成，不是另一个 tokenizer，也不是取代 PyTorch 的张量计算库。
- **DeepSpeed：**面向训练执行和内存等系统问题；ZeRO 是其重要机制，但“用了 DeepSpeed”和“用了某个固定 ZeRO stage”不是一句话。
- **Megatron-LM / Megatron-Core：**面向大模型训练组织与并行模块，特别要读懂 TP、PP、EP 和相应 checkpoint 约定。大规模环境的优势不能从本章 CPU 小模型直接推出。

[TRL 0.17.0 官方总览](https://huggingface.co/docs/trl/v0.17.0/en/index) 列出 SFT、DPO、GRPO、奖励模型等入口；[同版 SFTTrainer 说明](https://huggingface.co/docs/trl/v0.17.0/en/sft_trainer) 展示它与模型、数据和训练配置的连接。本章只建立定位，不安装 TRL，也不把 SFTTrainer 的名字等同于“自动正确的对话微调”。assistant-only 标签、LoRA、DPO 的偏好对和 GRPO 的奖励分组分别留给 P21、P22，那里会单独锁版本和检查目标。

一个实用顺序是：先用单进程证明数据、loss 和保存加载正确；确实需要更多吞吐再试数据并行；确认状态占用是瓶颈再考虑分片；当单层或网络规模、通信条件要求进一步切分时，再研究 TP / PP / EP。这个顺序不是性能定律，而是降低排错变量的工程方法。

## P20.9 保存模型，不一定等于能继续训练

本章产生两类本地输出，概念上不能混用。

**推理导出 export：**model.save_pretrained 保存 config 与 model.safetensors；tokenizer.save_pretrained 保存词表、特殊 token 和处理配置；experiment.json 保存版本、种子、设备、精度和 attention 后端。再次从本地目录加载，检查固定输入的 ID、logits 和 loss 一致。这个检查已经运行。

**训练检查点 trainer/checkpoint-3：**本例显式设置 save_only_model=False，另检查 optimizer.pt、scheduler.pt、trainer_state.json、rng_state.pth 和 training_args.bin 等文件存在，且 global_step=3。实际保存逻辑可查 [4.51.3 Trainer 源码](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/trainer.py)。不要根据概览页的一句默认行为推测所有版本和选项下都保存了什么。

虽然这些训练状态文件已经产生，本章**没有运行断点续训等价实验**，所以结果中明确记录 resume_tested=false。真正验证恢复，应该比较“连续训练 K+M 步”和“训练 K 步、保存、重启、再训练 M 步”的最终状态，并检查数据采样位置、随机数、优化器、scheduler、梯度累积边界等。只有模型权重相同还不够。

若想保留本章输出，给一个新的或空目录：

```bash
python code/21_training_frameworks.py --hf --output-dir /tmp/p20-checkpoint-demo
```

目录非空时脚本拒绝覆盖；默认不传目录则使用临时目录并清理。只加载本次自己产生的训练状态文件，不要随意反序列化来源不明的 optimizer 或 Python 对象。safetensors 也不能替代对模型来源、配置和代码的审核。

分布式训练还要问：“这是一整份模型，还是某个 rank 的分片？”并行拓扑改变后是否需要转换？词表是否一并保存？模型实现的版本是否相同？例如 Megatron 的固定版说明区分与 TP/PP 布局绑定的旧格式及其他分布式格式，不能默认不同框架的 checkpoint 可以直接交换。

## P20.10 多 GPU 选修：先写实验单，不在 CPU 假演示

本章没有安装 DeepSpeed / Megatron，也没有可报告的多 GPU 运行、通信量实测、显存峰值或加速比。下面只是实验设计与配置阅读题，不是已经通过的运行命令。

一个最小 ZeRO-2 配置片段可以先读成：

```json
{
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 1,
  "zero_optimization": {"stage": 2},
  "bf16": {"enabled": false},
  "fp16": {"enabled": false}
}
```

它只说明每 rank 的 microbatch、累积次数、状态分片级别与未启用混合精度，**不是完整训练程序**。启动器模板形如：

```text
DeepSpeed：deepspeed --num_gpus=2 YOUR_GPU_TRAIN.py --deepspeed YOUR_DS_CONFIG.json
Megatron：torchrun --nproc_per_node=2 YOUR_MEGATRON_ENTRY.py [与该入口一致的模型和数据参数]
```

大写项是待填写的占位符，不是本仓库文件。YOUR_GPU_TRAIN.py 必须自己解析配置、正确接入 DeepSpeed 或 HF Trainer；不能把本章强制 use_cpu=True 的 --hf 脚本塞进去冒充多卡训练。Megatron 入口、数据预处理和 flags 应来自选定 revision 的官方示例；仅把 --tensor-model-parallel-size=2 写在任意脚本后面没有意义。可从 [固定版官方仓库](https://github.com/NVIDIA/Megatron-LM/tree/core_v0.12.0) 的 examples 与训练参数说明开始。

Megatron 还有一个可供选修的具体入口：[core_v0.12.0 的 run_simple_mcore_train_loop.py](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.12.0/examples/run_simple_mcore_train_loop.py)。本章已读取该版本源码，确认它使用 MockGPTDataset、本地空 tokenizer、2 层模型、TP=2、PP=1，并运行 5 次更新。选好官方兼容依赖、进入该版本仓库后，待验证命令为：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  examples/run_simple_mcore_train_loop.py
```

这条命令**未在本书环境执行**。需恰好暴露两张可用 CUDA GPU，因为该入口用可见 GPU 数初始化 world size，源码又把 TP 固定为 2；还会编译数据 helper，并在当前目录写 ckpt。应使用新的实验目录并先检查源码，不能直接覆盖已有检查点。它是理解 Megatron-Core 接线的入口，不是通用生产训练配方，也不自动证明保存了可恢复的优化器状态。

动手前把下列内容保存成自己的实验单：

1. 代码 tag / commit、PyTorch / CUDA / 驱动 / NCCL / DeepSpeed 或 Megatron 版本；与本章 CPU 依赖分开管理。
2. 至少两张兼容 GPU 的型号、每卡显存、互联、CPU / RAM，跨机器时记录网络；是否真的支持计划使用的 BF16 / FP16 路径。
3. 完整展开后的启动命令和配置，不只保存一张截断的终端截图；写明 DP / TP / PP / EP 的 rank 分组。
4. 固定模型、tokenizer、数据、序列长度、有效 token 数、batch 与优化器；先用最小模型检查目标与梯度，再逐步扩张。
5. 对比 DDP 与 ZeRO-2 时尽量只改变状态分片，分别记录持久状态估计、实测峰值分配 / 保留显存、通信和速度。
6. warm-up、设备同步、重复次数、均值和波动；保存 checkpoint 后真正重启恢复一次。

这份实验单的价值不在于把配置写得很长，而在于让别人知道结果能比较什么、不能比较什么。在条件未具备时停在“未测”，比把理论账本包装成实测更专业。

## P20.11 练习、排错与验收

### 练习

1. “我把 config.json 改成两倍 hidden size，原权重就变强了。”这句话有什么问题？
2. 两条长度分别为 6、5 的含 BOS/EOS 序列，右侧补齐到 6。首列与 PAD 忽略时，监督位置数是多少？若恢复首列 BOS，本例版本的 Trainer 计数和实际 next-token 目标数有什么不同？
3. P=7、D=3，默认每元素 2+2+12 bytes。Stage 3 每个 rank 的所有权、状态字节和合计是多少？
4. 为本章代码故意制造一个错误，说明哪条测试应该发现它。只让 loss 下降，能否作为验收？

### 参考答案

1. config 改的是张量结构，不会替你训练或生成匹配权重，甚至会直接导致 shape 不兼容。结构说明与已学习参数是两个不同对象。
2. 监督目标为 5+4=9。删去首列忽略后，模型内部 shift 的目标数仍是 9，但本例 Trainer 的外部计数变成 11，因为它统计未 shift 的非忽略标签。此时会把本例梯度缩小为正确值的 9/11。
3. 所有权为 [3,2,2]，持久状态为 [48,32,32] bytes，合计 112 bytes。不能用 112/3 作为每个 rank 都安全的容量上界。
4. tests/test_training_frameworks.py 实际执行八种语义变异：丢掉分片余数、提前切分梯度、提前切分参数、漏算 master weight、错移 CE 标签、恢复首位 BOS 计数、取消 optimizer.step、损坏恢复后的权重。前四种被独立手算账本发现；错移标签被独立 CE 参考识别；后三种被真实 Trainer 更新、非零变化和恢复 logits 断言识别。程序能运行或 loss 下降，都不能代替这些检查。

常见排错顺序：

- 出现 index out of range：先查 token ID、词表大小和位置上限，不要先加 GPU。
- loss 下降但更新对照不符：查 shift、-100、有效 token 分母、梯度累积、裁剪、dropout、优化器和 scheduler。
- 保存加载只差几个末位：先查 dtype、attention 后端和评估模式，再判断合理容差；不把所有小误差都叫“随机性”。
- 只装基础依赖时显示 skipped：这是清楚报告未运行，不是脚本损坏；需要框架覆盖时在隔离环境装固定组合。
- ZeRO 账本很小但真实训练仍 OOM：检查激活、临时聚合、通信 buffer 和实际 dtype；账本从未承诺峰值。

本章验收是三句话：**我能把框架对象翻译回手写循环；我能独立核对一次真实参数更新；我能说明内存账本哪些状态算了、哪些没有。** 能做到这些，再读后训练配置时，你就不必靠“这份 YAML 看起来很专业”来判断正确性。

## 资料与版本边界

核验日期：2026-10-19。以下优先固定官方版本或原始论文；资料中的大规模结果不算作本书复现。

- [Transformers 4.51.3 Trainer](https://huggingface.co/docs/transformers/v4.51.3/en/trainer)、[GPT-2 API](https://huggingface.co/docs/transformers/v4.51.3/en/model_doc/gpt2)、[对应 Trainer 源码](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/trainer.py)：对象职责、标签计数与保存逻辑。本章实际运行此版本。
- [Transformers 4.51.3 PyPI 元数据](https://pypi.org/pypi/transformers/4.51.3/json)、[Accelerate 1.6.0 PyPI 元数据](https://pypi.org/pypi/accelerate/1.6.0/json)：附加依赖选择；实际锁定项见 requirements-frameworks.txt。
- [TRL 0.17.0 总览](https://huggingface.co/docs/trl/v0.17.0/en/index) 与 [SFTTrainer](https://huggingface.co/docs/trl/v0.17.0/en/sft_trainer)：仅核验框架关系和后训练入口，未安装或运行 TRL。
- [PyTorch 2.14 DDP 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html)：固定机制参考；不代表本书跑过该版本多卡。
- [ZeRO 论文 v3](https://arxiv.org/abs/1910.02054v3)；[DeepSpeed v0.16.5 ZeRO 文档源码](https://github.com/deepspeedai/DeepSpeed/blob/v0.16.5/docs/code-docs/source/zero3.rst)：已读取该 tag 的官方原始文件，作为分片机制参考；未安装 DeepSpeed。
- [Megatron-LM 原始论文](https://arxiv.org/abs/1909.08053)、[流水线与大规模训练论文](https://arxiv.org/abs/2104.04473)、[Megatron-Core core_v0.12.0 仓库](https://github.com/NVIDIA/Megatron-LM/tree/core_v0.12.0)：并行与 checkpoint 参考，未运行其 GPU 示例。
