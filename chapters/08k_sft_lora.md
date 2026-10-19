# 指令微调（P21）：对话数据、SFT 与 LoRA {#p21-sft}

在 [P19：预训练数据流水线](#p19-pretraining) 中，我们问“下一段文字是什么”；在 [P20：训练框架](#p20-frameworks) 中，我们把同一个目标交给框架。现在的问题变成：**给定一段对话，模型应该学习续写哪一部分？又允许修改哪些参数？**

这其实是两道题。SFT（Supervised Fine-Tuning，有监督微调）规定从示范答案中学习什么；全量更新或 LoRA 决定怎样调整参数。SFT 不等于 LoRA，LoRA 也不只用于 SFT。换了训练方法，错误的标签依旧是错误的标签。

读完本章，你应能给一段多轮对话逐 token 标出 loss，手算一个 LoRA 层，并用测试证明基础权重没有被偷偷更新。前置是 P02 的 tokenizer、P09 的交叉熵与梯度，以及 P19–P20 的数据和框架契约。不需要 GPU，也不下载模型。

配套 [code/22_sft_lora.py](../code/22_sft_lora.py) 分为标准库手算和真实 PyTorch CPU 更新两部分。**实验用随机初始化的微型 Transformer 学习 SFT 形式的目标，再冻结它继续训练 LoRA；不是对一个已有通用预训练模型做能力适配。** 这里的四条合成对话足以验证机械过程，不足以评价指令遵循、泛化或安全性。没有运行 HF SFTTrainer、PEFT、QLoRA 或任何大模型训练。

![SFT 只评分 assistant 内容和结束标记；LoRA 冻结主路，仅训练低秩支路，再合并验证](assets/diagrams/p21-sft.svg)

## P21.1 同一本练习册，两种学习目标

预训练和指令微调都可以使用 next-token 交叉熵。它们并不是“一个预测词、另一个不用预测词”，而是数据分布、上下文形式和被评分的位置不同。

想象一张带标准答案的练习卡：上方是规则，中间是问题，下方是示范回答。我们希望学生读规则、看问题，再学习如何回答，而不是主要练习复述题目。这个类比解释了 assistant-only loss，但不能推出模型像人一样理解规则，更不能把“正确格式”当成“可靠知识”。

本章选择的目标是：

- system 和 user 内容仍然作为模型的输入。
- 每轮 assistant 的回答正文，以及其真实结束标记，作为预测目标。
- role header、system/user 的结束标记和 padding 不评分。
- 同一条多轮对话中的**所有** assistant 轮都评分，不只最后一轮。

这是一个明确的教学策略，不是 SFT 唯一合法的定义。有的任务训练整段文本，有的只训练最后一个回答，有的分别加权不同片段。关键是把选择写进数据与测试，不能仅凭类名里出现“SFT”就断言得到了 assistant-only loss。

[TRL 0.18.2 的 SFTTrainer 源码](https://github.com/huggingface/trl/blob/v0.18.2/trl/trainer/sft_trainer.py) 展示了成熟训练入口如何处理数据和训练配置。这里引用它作为框架定位，不调用其中 API；更不把其他版本的参数名当成这个版本已经验证过的接口。

## P21.2 对话模板：角色不是另一个输入通道

输入常以消息列表保存：

```python
messages = [
    {"role": "user", "content": "问红"},
    {"role": "assistant", "content": "红"},
]
```

但本章的 decoder-only 模型最终收到的仍是一串 token ID，而不是一个会自动理解 role 字段的聊天对象。模板决定怎样把消息列表变成序列。

本章写死一个透明的教学模板，版本名为 `p21-char-chat-v1`：

```text
<bos>
<system>系统正文<end>        # 可选，只能在最前面
<user>问题正文<end>
<assistant>答案正文<eos>
<user>下一轮问题<end>
<assistant>下一轮答案<eos>
```

上面的换行和注释只是便于阅读，**不会成为实验输入**。在这份模板中，`<end>` 结束 system/user 消息，`<eos>` 结束 assistant 回答；多轮训练时 EOS 后面可以继续出现下一轮 user。这个命名是本章约定，不是所有模型的共同协议。实际模型可能区分“消息结束”“当前轮结束”“整个序列结束”，必须查看它自己的 tokenizer 和模板。

词表恰好有 17 项：

| ID | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| token | PAD | BOS | SYSTEM | USER | ASSISTANT | END | EOS | UNK |

剩余九个字依次是：`只=8，答=9，颜=10，色=11，问=12，红=13，蓝=14，绿=15，黄=16`。特殊标记由模板直接插入 ID；正文逐字编码，未知字变为 UNK。

因此，用户在正文输入字符串 `<assistant>`，不会凭这个拼写变成 role ID 4；每个未知字符只是 UNK。测试会确认内容中不能凭特殊标记的外观插入新角色。**这只是一条序列化性质，不是提示注入防护或模型安全保证。** 模型仍可能受普通语言内容影响。

真实 BPE/tokenizer 的词表远大于本例，不能把这个 17 项字符表套到已有 checkpoint 上。新加 special token 时，也要处理模型 embedding / 输出层的大小及其训练状态，而不是只修改字符串。

### 训练与推理，必须用同一套标点

训练输入包含示范回答；推理时没有答案，模板停在 assistant header 后：

```text
训练：<bos><user>问红<end><assistant>红<eos>
推理：<bos><user>问红<end><assistant>
```

`generation_prompt` 与 `encode_chat` 共用消息编码逻辑。测试不只看第一轮，还逐轮确认：推理 prompt 必须与训练序列到该轮回答起点为止的前缀完全相同。不能训练时用一种 role 标记、推理时随手换另一种，也不能多加一遍 BOS/EOS。

[Transformers 4.51.3 的模板文档](https://huggingface.co/docs/transformers/v4.51.3/en/chat_templating) 说明了模板与生成提示的关系，并提醒避免在模板已经加入特殊 token 后重复加入。本章只借助这些接口概念，不把手写模板称为 `apply_chat_template` 的实际运行结果。P20 已经验证过的 HF tokenizer 也没有被本章暗中改写。

## P21.3 八个位置，只有两个评分点

先完整写出“问红 → 红”的数据。这里的位置从 0 开始，`-100` 是本章 loss 的忽略值，不是词表中的一个 token。

| 位置 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| input token | BOS | USER | 问 | 红 | END | ASSISTANT | 红 | EOS |
| input ID | 1 | 3 | 12 | 13 | 5 | 4 | 13 | 6 |
| labels | -100 | -100 | -100 | -100 | -100 | -100 | 13 | 6 |
| 是否为目标 | 否 | 否 | 否 | 否 | 否 | 否 | 是 | 是 |

**labels 与 input_ids 保持同样的位置。不要在构造 labels 时提前移动答案。** 移动发生在计算 loss 的函数里，并且只发生一次：

```python
targets = labels[:, 1:]
predictions = logits[:, :-1]
```

所以位置 5 的 logits 预测位置 6 的“红”，位置 6 的 logits 预测位置 7 的 EOS。我们**不是**用“红”位置的 logits 预测同一个“红”；那样会让模型看到答案再复述答案。最后一个输入位置的 logits 没有下一个真实标签，本例不使用它。

写成数学式，令 m 表示一个目标位置是否应评分：

$$
\mathcal L=-\frac{1}{N}\sum_{b=1}^{B}\sum_{t=0}^{T-2}
 m_{b,t+1}\log p_\theta(x_{b,t+1}\mid x_{b,\le t}),
\qquad N=\sum_{b,t}m_{b,t+1}.
$$

这个 N 是有效目标数，不是 batch 大小，也不是 `B×T`。本例 N=2。假设模型给正确“红”的概率是 0.5，给正确 EOS 的概率是 0.25：

$$
\mathcal L=-\frac{\ln 0.5+\ln 0.25}{2}
=\frac{0.69314718+1.38629436}{2}
\approx1.03972077.
$$

这是自然对数下的平均负对数似然。除以八个输入格子，会错误得到四分之一大小的 loss 与梯度；只要分母错误固定，loss 依然可能下降，因此“曲线看起来正常”不能代替索引检查。

### “不评分 user”不等于“看不见 user”

这里有三个职责不同的 mask：

1. **causal mask**：位置 t 不能看到未来，防止预测时偷看答案。
2. **attention padding mask**：有效 query 不读取补齐用的 PAD key。
3. **label mask**：指定哪些 next-token 目标进入 loss。

system/user 的 attention mask 仍为 1，它们是有效上下文；labels 设为 -100 只意味着“不要求模型预测这些 token”。代码测试会发现：这些忽略目标对应的 logits 没有直接 loss 梯度，但 user 字符的 embedding 仍可能通过注意力影响 assistant 预测，因此收到梯度。

所以“忽略用户 loss”与“冻结用户表示”不是一回事。也不要把系统提示全部藏起来，却以为自己实现了 assistant-only loss。

## P21.4 多轮、结束、补齐与截断：四个常见陷阱

### 每一轮的答案都要能追溯

增加 system 和第二轮后，完整序列长 21：

```text
<bos><system>只答颜色<end>
<user>问红<end><assistant>红<eos>
<user>问蓝<end><assistant>蓝<eos>
```

有效标签位于 12、13、19、20，分别是“红、EOS、蓝、EOS”。对应评分 logits 位于 11、12、18、19。第一轮 assistant 正文的表示也会成为下一轮的上下文；这是 teacher forcing，不是模型先生成了第一轮答案再继续训练。

`ChatRow.assistant_spans` 用左闭右开区间保存 `[12,14)` 与 `[19,21)`，包含答案和 EOS，但不包含 assistant role header。它便于人工检查；测试还使用手写的完整标签序列对照，避免实现与测试都照抄同一个错误 span。

### EOS 是一个需要学习的目标

如果把所有 special token 都排除，EOS 也会消失，模型就缺少这份数据里的结束监督。这里即使 assistant 正文为空，也保留一个 EOS 目标，明确表示“立即结束”。是否允许空回答是数据质量决策，不是 mask 自动决定的事情。

PAD 则没有真实语义，补齐时 input_ids 加 PAD，attention_mask 加 0，labels 加 -100。**本例 PAD=0、EOS=6，二者不同。** 对某些把 pad_token_id 设为 eos_token_id 的真实配置，更不能简单按“ID 等于 EOS”把所有标签删掉；应按实际 padding 位置屏蔽，保留真实回答里的结束目标。

### 本章宁可丢完整轮，也不造半个答案

盲目 `ids[:max_length]` 可能在 assistant header、答案中间或 EOS 前截断。有时还会把只含问题、不含答案的样本交给 loss。另一种坏补丁是截断后强加 EOS：原本应继续的回答被错误教成“到这里结束”。

本章采用保守、可测试的策略：保留 BOS、可选 system，以及从开头开始能装下的**完整 user/assistant 对**，只在轮与轮之间截断。第一轮都放不下就明确报错，不返回一个零目标样本，不跳过长轮后拼上后面的轮，也不创造 EOS。

去掉 system 的两轮例子长度为 15；第一轮长度为 8：

- max_length=15：两轮都保留，4 个目标。
- max_length=8 到 14：只保留第一轮，2 个目标，`dropped_turns=1`。
- max_length=1 到 7：没有完整轮，拒绝这条样本。

这是本实验的契约，不是生产最佳策略。它会偏向较早的轮次，也可能丢掉有价值的长回答。实际项目可设计保留最近上下文、切分会话或拒绝过长样本等规则，但必须重新检查角色、目标、上下文完整性和分布变化，记录丢弃比例。改变规则就是改变数据版本。

### 零个目标不能“平均一下”

如果 shift 之后所有标签都是 -100，N=0。不能产生 NaN 再让 optimizer 继续，也不能悄悄说“loss=0，训练很好”。`masked_causal_loss` 在这种情况下抛出清晰异常；测试把一整个 batch 的标签设为忽略值，验证参数未变、没有 backward，也没有 optimizer 更新。

我们的 encoder 通常会更早拒绝没有完整监督轮的样本；loss 层仍保留这道检查，防止别的数据入口绕过前置约束。

## P21.5 全量更新与 LoRA：改整张表，还是加一条窄支路

对一个线性层，采用列向量记法：

$$
y=W_0x,\qquad W_0\in\mathbb R^{d_{out}\times d_{in}}.
$$

全量更新允许直接调整 W0 中的全部元素。普通 LoRA 冻结它，增加两个小矩阵：

$$
A\in\mathbb R^{r\times d_{in}},\quad
B\in\mathbb R^{d_{out}\times r},\quad
\Delta W=\frac{\alpha}{r}BA,\quad
y=W_0x+\frac{\alpha}{r}B(Ax).
$$

可以把它想成给固定机器加一条可调节的小支路，而不是重新铸造所有零件。但“支路很窄”不保证任何任务都适合，也不表示原模型本身是低秩的。受约束的是**这次更新**：`rank(BA)≤r`，还可能低于 r。

这里是 [LoRA 原论文 v2](https://arxiv.org/abs/2106.09685v2) 的普通低秩更新及 alpha/r 缩放形式，不包含 DoRA、rsLoRA 或其他变体。PyTorch 的 Linear 接受行式 batch，所以实现写成先 `F.linear(x,A)` 再 `F.linear(...,B)`；不要因为记法切换就把 BA 的乘法顺序颠倒。

### 先数参数，再谈节省

不计 bias 等额外可训练项时：

$$
P_{full}=d_{out}d_{in},\qquad P_{LoRA}=r(d_{in}+d_{out}).
$$

例如 `d_in=d_out=4096，r=8`：原矩阵有 16,777,216 个参数，A/B 合计 65,536 个，是原矩阵的 1/256，约 0.390625%。这只是这一层的**新增可训练参数比例**；冻结的 16,777,216 个原参数仍需保存和参与前向。

当矩阵很小或 r 很大时，LoRA 甚至可能比原矩阵参数更多。例如 2×3 的层，r=2 时有 2×(3+2)=10 个适配参数，超过原来的 6 个。低秩不是免费的效果保证。

本章小模型只给 q_proj 和 v_proj 接上 LoRA。每个原矩阵为 16×16，r=2，A/B 合计 64 个；两个投影合计 **128 个可训练参数**。其余 3,760 个基础参数全部冻结，包括 embedding、位置表、norm、FFN 和输出层。加入 adapter 后模型实际总参数为 3,888，不是变成只有 128 个参数。

少算基础权重的梯度和优化器状态，通常能降低相应训练状态开销；但基础权重、激活、中间结果及部分反向计算仍存在。可训练参数减少 99%，不能据此声称峰值显存或训练时间也减少 99%。本章不做 GPU 显存和速度测试。

## P21.6 一个可以手算到底的 LoRA 层

取输入 x、基础矩阵 W0 和适配矩阵如下，r=1、alpha=2，因此缩放系数为 2：

$$
x=\begin{bmatrix}2\\-1\\0.5\end{bmatrix},\quad
W_0=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix},\quad
A=\begin{bmatrix}1&0&2\end{bmatrix},\quad
B=\begin{bmatrix}3\\-1\end{bmatrix}.
$$

逐步相乘：

$$
W_0x=\begin{bmatrix}1.5\\6\end{bmatrix},\quad
Ax=3,\quad
2B(Ax)=\begin{bmatrix}18\\-6\end{bmatrix},\quad
y=\begin{bmatrix}19.5\\0\end{bmatrix}.
$$

只写一句“输出等于基础输出加 adapter 输出”很容易让两个实现一起错。测试直接以这组手算数值为参考，另用 r=2、alpha=6 的独立矩阵检查“漏除 rank”错误，同时检查 A/B 一共 5 个可训练参数、W0 的 6 个参数被冻结。

### 冻结参数不等于关闭整条计算图

`requires_grad_(False)` 让基础参数不接收梯度更新；它不等于把经过基础层的全部运算包进 `no_grad()`。有些早先层的 adapter 仍需要通过后面的冻结层回传梯度。把整条主路断开，可能切断正确的学习路径。

令这个手算例子的标量目标为 `L=y₁+y₂`，链式法则给出：

$$
\frac{\partial L}{\partial B}=\begin{bmatrix}6\\6\end{bmatrix},\quad
\frac{\partial L}{\partial A}=\begin{bmatrix}8&-4&2\end{bmatrix},\quad
\frac{\partial L}{\partial x}=\begin{bmatrix}9\\7\\17\end{bmatrix}.
$$

这些数值由测试独立写死，用来检查 autograd，而不是仅比较两个使用同一段代码的模型。W0 的梯度为 None，因为它被显式冻结。输入 x 仍有梯度，说明“权重固定”和“信息不能穿过”是两件事。

### 为什么初始 B 为零，不代表学习卡死

代码使用随机初始化 A、零初始化 B，保证刚插入 adapter 时 `BA=0`，不改变原模型预测。具体采用 Kaiming-uniform 初始化 A；它是本实现的选择，不声称逐项复现原论文的高斯初始化。固定版 [PEFT 0.15.0 LoRA 配置说明](https://huggingface.co/docs/peft/v0.15.0/package_reference/lora) 也区分初始化、rank、alpha 与 target modules；本章没有调用 PEFT。

在第一步，如果 B=0，则对 A 的梯度为 0，但对 B 的梯度通常不为 0。B 更新之后，下一步 A 也能获得梯度。测试确实执行这两次 backward，验证这一变化。若 A、B 都初始化为零，两个方向都可能没有初始学习信号；不要因为“初始更新应为零”就把两个矩阵都清零。

## P21.7 真正训练一次，明确哪些结果不能比较

CPU 实验生成四条记录，问题分别问“红、蓝、绿、黄”，答案是相应颜色字；红色记录再附加“问蓝 → 蓝”的第二轮。每条都有 system“只答颜色”。因此共四条对话、五个 assistant 回答，每个回答含一个颜色和一个 EOS，一共 **10 个有效目标**。

它们没有来自网络、用户文件或私人聊天。也没有设置真正的验证/测试集：同样十个目标反复参与更新，报告的是训练 loss，不是 held-out 性能。

### 模型与步骤

`TinyChatLM` 是独立的 1 层、1 头、宽度 16 的 causal Transformer：字符 embedding + 可学习绝对位置、LayerNorm、q/k/v/o 投影、两层 GELU FFN、输出层。最大位置长度 64，无 dropout，不共享输入/输出权重，一共 3,760 参数。它不是在 S5 主线上偷偷替换组件，也不读取之前章节的 checkpoint。

训练过程分两阶段：

```python
# 第一阶段：整个随机小模型学习这四条合成示范。
_updates(model, batch, steps=24, lr=0.15)

# 第二阶段：冻结刚得到的基础模型，给 q/v 加普通 LoRA。
model.zero_grad(set_to_none=True)
model.requires_grad_(False)
model.q_proj = LoRALinear(model.q_proj, rank=2, alpha=4)
model.v_proj = LoRALinear(model.v_proj, rank=2, alpha=4)
_updates(model, batch, steps=24, lr=0.5)
```

`_updates` 只把 `requires_grad=True` 的参数交给 SGD，并按“清梯度 → 前向 → assistant-only loss → backward → step”的顺序更新。固定 seed=21，CPU float32，单线程；无 momentum、weight decay、梯度累积或外部日志服务。SGD 是为了让过程容易检查，不是推荐真实 SFT 一律用这些超参数。

**这不是全量微调与 LoRA 的公平效果对比。** LoRA 从第一阶段的终点继续训练同一任务，学习率也不同。我们只检验：完整训练循环能更新参数、低秩支路能单独更新、基础权重不变，以及合并前后的计算一致。想比较质量，需要相同起点、独立评估数据、明确预算和调参规则。

### 本次实测

2026-10-19，Python 3.12 / PyTorch 2.14.1+cpu 的一次结果如下。数值为教学夹具记录，不是吞吐 benchmark；其他平台可能有末位浮点差异。

| 检查项 | 结果 |
| --- | --- |
| input_ids / logits 形状 | [4,21] / [4,21,17] |
| 有效监督目标数 | 10 |
| 全量参数 / 新增 LoRA 可训练参数 | 3,760 / 128 |
| 第一阶段 loss，24 次更新 | 2.83717895 → 0.61146003 |
| 第一阶段参数最大变化 | 约 0.76645482 |
| Torch loss 与独立标准库参考差 | 约 5.62e-8 |
| 第二阶段 loss，24 次更新 | 0.61146003 → 0.61064279 |
| LoRA 参数最大变化 | 约 0.03599648 |
| 冻结基础参数 / 梯度 | 逐元素未变 / 全部为 None |
| 合并前后 logits 最大差 | 约 7.15e-7 |

训练 loss 的下降证明这个目标在这批数据上被优化，不证明模型会回答新问题。脚本显式输出 `heldout_quality_evaluated=false` 与 `hf_peft_qlora_run=false`，避免把未测项目混进“通过”的结果。

### 运行方式

只查看模板、标签和手算，不需要 PyTorch：

```bash
python code/22_sft_lora.py
python -m unittest discover -s tests -p 'test_sft_lora.py' -v
```

测试在缺少 PyTorch 时会明确跳过真实训练相关部分。**skipped 不算训练验证成功。** 默认脚本打印 `training: NOT RUN`；它不会因为输出了 loss 手算数字，就假装跑过训练。

已经安装全书基础依赖后，执行真实 CPU 更新：

```bash
.venv/bin/python code/22_sft_lora.py --train
.venv/bin/python -m unittest discover -s tests -p 'test_sft_lora.py' -v
```

本章不新增 requirements，也不需要安装 transformers、trl、peft 或 bitsandbytes。若没有 PyTorch 却指定 `--train`，标准库演示仍可输出，但训练明确提示 SKIP 并以退出码 2 结束。已经满足依赖后的实际训练错误会正常报错，不被吞成“可选功能跳过”。

## P21.8 合并：把两条线变回一条线

普通、无量化、无 LoRA dropout 的线性层，可以构造：

$$
W_{merged}=W_0+\frac{\alpha}{r}BA.
$$

于是 `W_merged x` 与原先两条支路之和在实数算术中相等。上一节手算例子的合并权重为：

$$
W_{merged}=\begin{bmatrix}7&2&15\\2&5&2\end{bmatrix}.
$$

代码的 `merged()` 返回一个**新的普通 Linear**，保留原来的 bias，不改原 adapter 或基础层。调用两次也各自从同一个原始 W0 出发，不会把 BA 重复加两遍。合并后的模型不再额外执行 adapter 支路；若既改了原权重又保留支路，更新就会被算两次。

测试分别检查手算矩阵和整个模型的 logits。float64 小整数例子得到精确对齐；float32 整模因乘法顺序不同，按 `atol=2e-6、rtol=2e-5` 检查。这里验证的是数值等价，不是实测推理加速。

实际保存 adapter 时，还需要记录基础 checkpoint 的精确版本、目标层名、rank/alpha、tokenizer/template、额外训练的 embedding 或其他模块。只有 A/B 文件并不等于一个独立完整模型。**本章没有实现 adapter 保存/重载或跨框架转换**，不声称这条部署链已经通过测试。

## P21.9 QLoRA：冻结的底座换成低比特存储

QLoRA 的 Q 指 quantization。它不是“把 A/B 变成 4 bit 再训练”，也不是“训练中直接对 4 bit 整数权重做普通梯度下降”。[QLoRA 原论文 v1](https://arxiv.org/abs/2305.14314v1) 的关键思路是冻结量化后的基础模型，通过相应计算路径，把梯度传到可训练的低秩 adapter；论文还讨论 NF4、量化常数的再次量化和 paged optimizers。

需要区分三件事：基础权重的存储格式、运算时使用的计算精度、adapter / 梯度 / 优化器状态的精度。它们不必相同。基础参数“4 bit”也不表示整个训练进程只占 `参数数×0.5` 字节；量化元数据、激活、临时反量化结果及优化器状态仍要计算。

本章只运行普通 float32 LoRA，没有 bitsandbytes、NF4 kernel、量化训练或 GPU 内存测量。普通 LoRA 的合并等式，也不能直接推出“量化、反量化、合并、再量化”全过程零误差。

P24 将进一步讲 scale、zero-point、group、量化误差，以及推理格式和框架选择。**这是后续阅读边界，不是本章已经完成的功能。** 不要为了这个章节临时安装 GPU 依赖，更不要把本文几十毫秒级的小矩阵算术换算成真实大模型所需显存。

## P21.10 数据质量与评估：训练正确还只是第一关

一份有用的 SFT 数据，应能说明问题来自哪里、回答为什么可信、允许怎样使用，以及它与验证集有没有重叠。沿用 P19 的来源记录、去重、固定划分和隐私边界；不要把私人聊天直接当成现成的训练集。

对话任务还有一些额外检查：

- 同一问题是否出现互相矛盾的示范？模型可能学到混合分布，而不是自动挑对的那条。
- 系统规则与答案是否一致？“只答颜色”后面接长篇解释，会改变学习目标。
- 真实部署会遇到的多轮结构、长度和角色，训练里是否覆盖？
- 长样本被过滤了多少？丢掉长回答是否恰好丢掉某一类任务？
- 训练推理的模板、EOS、tokenizer 和截断策略是否完全对齐？
- 评估是否按独立问题或来源划分，而不是把同一对话的改写随机分到两边？

验证时除了 held-out assistant loss，还要根据任务检查答案正确性、格式、是否遵守约束、停止行为，以及不应回答时的适当处理。teacher forcing loss 使用真实历史答案，而真实生成会受自己之前错误的影响，因此二者不能互相代替。需要明确失败案例，不只挑最漂亮的一条演示。

LoRA 不是数据审校器，也不能保证不遗忘既有能力。SFT 通过模仿示范调整分布，不会自动把“看起来专业”的错误答案变真。下一章 P22 会把“示范该怎么答”进一步区分为“比较哪个回答更好”和“按奖励优化”，不能把这几种目标混叫成同一件事。

## P21.11 自测与已验证参考答案

下面所有数值和边界都在 `tests/test_sft_lora.py` 中有对应检查。先独立回答，再看参考答案。

### 题 1：标签与错位

“问红 → 红”的八个输入位置，哪些 labels 保留？哪些 logits 被评分？若两个正确目标概率分别为 0.5、0.25，平均 loss 是多少？

**参考答案：** labels 仅位置 6=13、7=6，其余为 -100；logits 位置 5、6 被评分；N=2，loss≈1.03972077。不能把 labels 提前左移，也不能除以八。对应 `test_hand_example_literal_ids`、`test_scalar_loss_hand_answer_and_large_offsets` 与独立 log-probability 测试。

### 题 2：多轮与 EOS

加入 system“只答颜色”和第二轮“问蓝 → 蓝”，目标位置与 span 是什么？空 assistant 正文是否必然有零个目标？

**参考答案：** 目标位置为 12、13、19、20；span 为 `[12,14)`、`[19,21)`，总计四个。空正文仍有一个 EOS 目标。对应 `test_multi_turn_system_and_both_assistant_eos` 和 `test_empty_answer_still_supervises_stop`。

### 题 3：截断到 14

没有 system 的两轮对话长 15；max_length=14、15、7 各怎样处理？为什么不在长度 14 的位置强加 EOS？

**参考答案：** 14 只保留完整第一轮，得到 8 个输入、2 个目标；15 保留两轮、4 个目标；7 报错。强加 EOS 会把被截掉的未完答案改成结束示范。对应 `test_truncation_exact_fit_and_all_boundaries`，测试逐一遍历边界，不只抽查 14。

### 题 4：LoRA 的参数账

4096×4096 的层用 r=8，A/B 有多少参数？本章两个 16×16 投影用 r=2 又有多少？是否可以说本章模型只剩 128 个参数？

**参考答案：** 前者 65,536，相对原矩阵 16,777,216 是 1/256；后者每层 64，共 128。不能说模型只有 128 个参数，它还保存 3,760 个被冻结的基础参数，总计 3,888。对应参数算术测试和真实训练报告测试。

### 题 5：矩阵与合并

使用 P21.6 的 W0、A、B、x，算出输出与合并权重；若忘记 alpha/r，结果还能一样吗？

**参考答案：** 输出 `[19.5,0]`，合并权重 `[[7,2,15],[2,5,2]]`。忘记系数 2 会得到错误输出 `[10.5,3]`。矩阵 oracle 直接检查手算答案；实际删掉缩放的变异代码会被拒绝，不只是口头说“应该测”。

### 题 6：梯度到哪里

仍用该例子，令 L=y₁+y₂，求 A、B、x 的梯度。为什么冻结 W0 后 x 仍可能有梯度？

**参考答案：** A 梯度 `[8,-4,2]`，B 梯度 `[6,6]`，x 梯度 `[9,7,17]`；固定 W0 不等于切断对输入的导数。`test_lora_gradients_match_manual_chain_rule` 逐项核对。另一个测试验证 user 内容虽然不被当成标签，仍可通过上下文影响 loss。

### 题 7：两个“零”是否一样

B 初始为零时，A 第一轮梯度为零，是 bug 吗？所有 labels 都忽略时，loss 返回零再更新是否合适？

**参考答案：** 前者不一定是 bug：B 先获得梯度，更新后 A 也能获得梯度。后者不合适：有效目标数为零，平均 loss 未定义；本代码在更新前报错，并验证参数完全不变。对应两步初始化梯度测试和 all-ignored 测试。

### 题 8：什么叫“验证过”

默认 CLI 打印了手算 loss，能说 SFT 训练通过了吗？LoRA 的训练 loss 更低，能说 LoRA 比全量微调更好吗？

**参考答案：** 都不能。默认命令明确输出 NOT RUN，缺 Torch 的 `--train` 以退出码 2 结束；测试用无 site-packages 的独立进程实际核对。LoRA 是从全量阶段终点继续训练同一批数据，不是同起点对比；报告明确没有 held-out 质量评估。

### 本章的反例不是注释

测试在内存中真正修改并运行 12 种语义错误：监督 user、漏掉 assistant EOS、误监督 role header、截断等号边界错误、错移 logits、错误 loss 分母、丢掉 LoRA 缩放、缩放漏除 rank、合并漏缩放、基础层未冻结、B 初始非零、移除 causal mask。每种都要求独立参考或明确的边界契约拒绝它，且不修改仓库文件。测试因此不是只证明“两个一起写错的实现彼此相同”。

本次有 PyTorch 环境中 28 个测试通过；无 PyTorch 路线执行 13 个标准库测试，另外 15 个明确跳过。未来修改后应重新运行，不应把这些数字当成永远有效的标签。

## 本章小结与来源边界

SFT 的第一步不是选显卡，而是确定“谁的哪几个 token 在被学习”。模板决定输入，causal shift 决定预测对齐，label mask 决定评分位置；三者必须一起检查。LoRA 另管一件事：冻结基础权重，用低秩 A/B 表示允许学习的更新。参数更少不代表模型更小到只剩 adapter，也不代表数据更可靠。

来源按 **2026-10-19** 核验，固定版本而非追随 latest：

- [Transformers 4.51.3 chat templates](https://huggingface.co/docs/transformers/v4.51.3/en/chat_templating)：模板、生成前缀、特殊 token 重复的注意事项。
- [TRL 0.18.2 SFTTrainer 源码](https://github.com/huggingface/trl/blob/v0.18.2/trl/trainer/sft_trainer.py)：框架职责参照，未在本章运行。
- [LoRA，arXiv:2106.09685v2，2021-10-16](https://arxiv.org/abs/2106.09685v2)：冻结基础层、低秩更新与合并思想。手算矩阵、数据、代码和测试是本章独立教学实例。
- [PEFT 0.15.0 LoRA 配置参考](https://huggingface.co/docs/peft/v0.15.0/package_reference/lora)：参数与初始化概念参照，未运行 PEFT。
- [QLoRA，arXiv:2305.14314v1，2023-05-23](https://arxiv.org/abs/2305.14314v1)：量化基础模型与低秩适配，本文仅作机制说明，不把论文硬件或质量数字当成本实验结果。

下一步进入 P22：从示范答案转向偏好对与奖励，先区分目标，再看更新方法。
