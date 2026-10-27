# 生成优化（P11）：KV Cache，不再重复算历史

第 8 章得到一份明确的 S0：相同字符词表、可学习绝对位置、两个 Pre-LN Block。现在实现 S1，**权重一项都不改，只缓存生成中可以复用的中间结果**。先证明 logits 对齐，再问时间和内存，不能把“生成的字看起来一样”当作正确性证明。

配套 [code/12_kv_cache.py](../code/12_kv_cache.py) 直接使用 S0 的模型模块，不另训一份近似模型。实验是 CPU、float32、无 dropout、eval 模式；不下载权重，不用 GPU。

## P11.1 重算发生在哪里

假设提示有“床前”两个字符。S0 先计算这两个位置，再选出“明”；下一轮把“床前明”三个位置全部送进模型，先前“床”“前”的计算又做一次。

在因果模型中，较早位置不能看后来追加的 token。如果权重、位置、历史 token 和确定性的前向计算都不变，较早位置在每层的表示也不会因追加后缀而变化。因此它们的 K 和 V 可以保存，下一步直接读。Q 是本次新位置要发出的查询，不必保存整个历史 Q。

```text
S0：床 前          → 全部位置前向 → 选 明
    床 前 明       → 全部位置前向 → 选 月

S1：床 前          → prefill，存各层 K/V → 选 明
          明       → 新 Q/K/V + 历史 K/V → 选 月
             月    → 新 Q/K/V + 历史 K/V → ...
```

可以把它比作保存已经整理好的借阅索引，下一次只补新条目。类比的边界是：真实缓存不是保存自然语言摘要，也不是模型长期记忆；它保存的是当前请求、当前参数与位置下的浮点张量。缓存本身不会让模型学会新的事实。

## P11.2 Prefill 与 decode，首字到底从哪来

**Prefill** 一次处理提示，建立每一层的 K/V。提示最后一个位置的 logits 已经能选出第一个新 token；不用先把这个尚未选出的 token 输入模型。

**Decode** 通常每次送入刚选出的一个 token。这个新位置产生 Q、K、V；把新 K/V 接在对应层的历史后面，Q 对全部可见 K 打分，然后更新本层缓存。下一层使用的是本层更新后的新 token 表示，而不是直接复用第一层的 K/V。

模型有两层就有两组缓存。每组分别保存 K 和 V，共四个张量。我们的接口：

```python
logits, cache = cached_forward(model, prompt_ids, cache=None)
# logits: [B, prompt_T, vocab_size]
# ... 选出 next_ids: [B,1]
new_logits, cache = cached_forward(model, next_ids, cache)
# new_logits: [B,1,vocab_size]
```

这里的片段是调用顺序，完整可运行示例在源码。`cached_forward` 不选 token，只返回当前新块的 logits；`generate_cached` 才负责选择与循环。它每次调用从空缓存开始，防止第二个请求误接在第一个请求后面。

## P11.3 逐层形状与位置偏移

设已有 P 个位置，本次新输入 C 个位置；头数 H，每头宽度 d，主干宽度 D=H×d。

| 张量 | 形状 | 本次是否新算 |
| --- | --- | --- |
| 输入 ID | [B,C] | 本次输入 |
| 新 Q/K/V | [B,H,C,d] | 是 |
| 历史 K/V | [B,H,P,d] | 否 |
| 拼接后的 K/V | [B,H,P+C,d] | 保存历史和新块 |
| 注意力 scores / weights | [B,H,C,P+C] | 是，通常不是方阵 |
| 当前块输出 | [B,C,D] | 是 |
| 当前块 logits | [B,C,57] | 是 |

默认模型 H=4、d=12。当 B=1、P=3、C=2 时，新 Q 是 `[1,4,2,12]`，K/V 总长为 5，scores 是 `[1,4,2,5]`。

新块位置必须是 **P,P+1,…,P+C−1**。不是每次 decode 都从 0 开始，否则每一个新字都会拿到错误的可学习位置向量。源码的关键语句：

```python
pos = torch.arange(offset, offset + T, device=idx.device)
x = model.tok_emb(idx) + model.pos_emb(pos)
```

这里 T 是本次新块长度，`offset` 是缓存长度。旧位置向量已经进入旧 K/V 的计算，不应给旧缓存再加一次位置向量。参数表最大 128 行，索引只能是 0–127。

## P11.4 矩形 mask：最容易写错的两行

把提示分块计算，或者一次追加两个 token，同样必须阻止本块早位置看到本块未来。P=3、C=2 时，新 query 的绝对位置是 3、4，key 的位置是 0–4。可见规则仍是 **key_position ≤ query_position**。

### 表 P11-1：三格历史加两格新输入

| 新 query 位置 | key 0 | key 1 | key 2 | key 3 | key 4 |
| --- | --- | --- | --- | --- | --- |
| 3 | True | True | True | True | False |
| 4 | True | True | True | True | True |

直接 `tril(ones(2,5))` 会得到第一行只看 key0、第二行只看 key0/1，错误地屏蔽了大部分历史。反过来完全不加 mask，又会让位置 3 偷看位置 4。C=1 时本行可以看全部历史加自己，但这个特殊情况不能代替通用实现。

```python
query_positions = torch.arange(offset, offset + T, device=x.device)[:, None]
key_positions = torch.arange(offset + T, device=x.device)[None, :]
allowed = key_positions <= query_positions
scores = q @ k.transpose(-2, -1) / math.sqrt(attn.head_dim)
weights = scores.masked_fill(~allowed, float("-inf")).softmax(dim=-1)
```

这是源码片段，`True` 表示允许；先填负无穷，再 Softmax。本实现不接受空块，也不提供 padding，因此每行至少有自己可见，不会产生全 mask 行。带 padding/变长批次是另一层接口契约，不能把这个限制忘掉后套到任意批次。

### 表 P11-2：把可见性变成可手算的输出

暂时只看一个头，让两行 scores 都是 `[ln1,ln2,ln3,ln4,ln5]`，五个标量 V 是 `[10,20,30,40,50]`。这些是独立手算输入，不是随机模型碰巧得到的分数。

| query | 允许项 exp(score) | 分母 | weights | 加权 V |
| --- | --- | --- | --- | --- |
| 位置 3 | [1,2,3,4,0] | 10 | [0.1,0.2,0.3,0.4,0] | 30.000000 |
| 位置 4 | [1,2,3,4,5] | 15 | [1/15,2/15,3/15,4/15,5/15] | 36.666667 |

第一行分子是 10+40+90+160=300，除以 10 得 30；第二行再加 250，550/15≈36.666667。错误的小矩形下三角会分别得到 10 和 50/3≈16.666667。这张表由脚本手算函数与测试核对，不用生成句子的表面效果来掩盖错误。

## P11.5 证明 S0 与 S1 等价，要比较什么

对同一个 eval 模型与同一串 ID，先完整前向取得 `[B,T,57]` 的参考 logits，再分块调用缓存路径，将新块 logits 沿时间拼起来。比较的是每个位置的每个词表项，而不是仅比较选出的 token。

检查覆盖：

- 一次完整 prefill、逐 token 输入、7-token chunked prefill，最后一块不足 7 也要正确。
- T=1 和 B>1；每层 K/V 长度应随新块增加。
- 同一请求分支：保留旧 cache，从它分别追加不同后缀，不应偷偷原地修改旧 cache。
- 两次独立请求：重新传 `None`，第二次结果不能包含第一次历史。
- 错误边界：train 模式、空输入、超过位置上限、错误 batch/层数/头维、不同模型或更新权重后的旧缓存。

float32 的运算顺序可能因为矩阵大小不同而有微小舍入差；测试使用明确的 `atol/rtol`，不是要求逐位相同，也不是放宽到“都能生成一句诗”。脚本打印最大绝对误差，超出容差时必须排查位置、mask、拼接维度或参数是否相同。

`KVCache` 的容器不可重赋字段，但其中的张量仍可被调用者修改。因此约定缓存张量只读，不承诺 Python 的 frozen dataclass 能深度冻结 Tensor。它不是持久化 checkpoint，也不要跨模型、跨参数更新复用。

## P11.6 用空间换计算：先算真实 K/V 负担

MHA 有 L 层，batch 为 B，缓存 T 个 token，H 个头，每头 d 个数，每个元素 s 字节，则：

$$M_{KV}=2LBHTds=2LBTDs$$

前面的 2 是 K 和 V 两份，不是两个 Block。默认 L=2、B=1、H=4、d=12、float32 的 s=4。

| 缓存长度 T | 计算 | K/V 逻辑字节数 |
| --- | --- | --- |
| 5 | 2×2×1×4×5×12×4 | 3840 |
| 48 | 2×2×1×4×48×12×4 | 36864 |
| 128 | 2×2×1×4×128×12×4 | 98304 |

脚本用 `numel() * element_size()` 把每层 K/V 实际相加，再与公式比较；float64 会将这些数字翻倍。此处的“内存”仅指逻辑张量载荷，不包括 Python 对象、模型权重、临时 scores、分配器保留空间和拼接复制的峰值。

S1 用 `torch.cat` 动态增长，旧新张量可能在某一瞬间同时存在。因此公式不是进程 RSS 或显存峰值。每步注意力仍需读全部历史 K/V；缓存不把任意长注意力变成与历史长度无关的常数成本。

GQA/MQA 会改变 KV 头数，量化会改变元素存储，分页会改变分配策略；都不在 S1 实现。这里用 MHA 的 H 直接计算，只把后续主题留作路线，不能用本章数字声称完成了 PagedAttention。[来源 2]

## P11.7 长度、重置与批处理的诚实边界

S0 的旧 `generate` 超长后保留最后 128 字、重置位置编号。S1 **拒绝超过容量**：只删掉最老 K/V 并不能重现这种重编号，因为保留下来的 K/V 已含旧位置与旧上下文影响。需要新的窗口语义和重算策略才能比较，不能静默丢缓存就声称等价。

`generate_cached` 采用保守规则：提示长度+请求新增预算 ≤128。最后一个选出的 token 不一定还需送回模型，但预算规则仍把它计算在内，方便调用者预先知道边界。零新增返回原输入，不做模型前向。它和 S0 一样没有 EOS，按预算停止。

批量实验只支持同长、无 padding 的 B 个请求，位置从 0 开始对齐。若要把长短不一的用户请求动态加入/退出，需要分别记录有效长度、注意力可见性与缓存所有权，本教学实现没有做到。对不同 batch 直接复用旧 cache 会报错，不会替你猜如何重排请求。

## P11.8 计时：首 token 与后续 token 分开记

```bash
OMP_NUM_THREADS=1 python3 code/12_kv_cache.py
python3 -m unittest discover -s tests -p 'test_kv_cache.py' -v
```

基准使用同一个默认 S0 的随机权重，固定 48 个输入 ID，前 16 个作提示，后 32 个逐个回放。随机权重足以检查数值等价和计算路径，不用于评价文本质量。

- 首 token 指标：提示前向产生可选择首字的 logits 的耗时；不含分词、采样、网络或排队，不是端到端服务 TTFT。
- 后续 token 指标：同一次运行中 32 次后续前向的平均耗时，再跨重复运行取统计量；不是逐 token 尾延迟分位数。
- 先预热 2 轮，再计 7 轮；交替 S0/S1 测量顺序，输出各指标 min/median/max。
- CPU 单线程用 `perf_counter`；没有异步 GPU 工作，因此本实验不涉及 CUDA 同步。将方法搬到 GPU 时需要另外处理同步。

本机测量值和版本见本批验证记录；每次运行都会重新打印，不将抖动的毫秒数写成单元测试常数。小矩阵下 Python 验证、`cat` 和分配可能抵消省下的计算，缓存不保证在这个微型 CPU 例子里更快。正确的优化报告应同时列出两条路径和计时范围，而不是只挑最漂亮的一次。

## P11.9 先预测，再看答案

1. P=3、C=2 时 scores 的最后两维是什么？第二个新 query 能看几个 key？
2. 已缓存 5 个 token，下一块 2 个 token 的位置编号是什么？可以重新从 0 开始吗？
3. 默认模型 float32、B=1、T=5 时，K/V 逻辑载荷多少字节？等于进程总内存吗？
4. 提示 prefill 结束后，第一个新 token 是否还要先输入模型才能被选出？
5. 两次独立请求使用同一个模型，第二次如何开始？能在更新权重后继续旧 cache 吗？
6. 缓存与完整前向生成相同贪心 token 就算验证通过吗？删最老缓存能重现 S0 的超长滑窗吗？

### 参考答案

1. `[2,5]`；第二行能看 5 个。矩形可见性由绝对位置比较得到。
2. 5、6。重新从 0 开始会查错位置嵌入。
3. 3840 字节。不是总内存，也不是 `cat` 时的峰值。
4. 不需要；提示最后位置的 logits 已可选首字。选出后才将它作为后续 decode 输入。
5. 传 `cache=None`；`generate_cached` 自动为每次调用创建空缓存。参数更新后旧缓存无效。实现核对参数对象身份、版本、dtype 与 device，包含 `assign=True` 替换参数的情况；不支持通过 `.data` 等方式故意绕过 PyTorch 版本跟踪。
6. 不算，需比较每个位置完整 logits；也不能，因为旧 K/V 的位置与上下文没有重新计算。

`tests/test_kv_cache.py` 对应形状/偏移、矩形 mask、全 logits 对齐、内存公式、独立请求和上限契约；还实际构造错误 offset 和错误 mask 的负面对照。后续 RoPE、GQA 或缓存布局的改动，都要继续通过这种同参数对齐，而不是跳过 S0 直接看速度。

## P11.10 前沿补充：不改文本，直接改 KV（cache engineering） {#p11.10-cache-engineering}

> [!NOTE]
> 这一节讲一个新做法：不重算，直接从缓存里“删掉”旧内容。依据是清华 NICS 的 [Cache Engineering for Agents](https://arxiv.org/abs/2610.28577)（arXiv:2610.28577v1，2026-10-20）§3.3–§3.4，代码见 [thu-nics/AKV](https://github.com/thu-nics/AKV)。本书只演示单条序列上的两个操作，不实现 AKV 的服务系统。

**问题。** Agent 一轮轮调用工具，历史越来越长。常见做法是改文本：删掉或摘要旧的工具结果。可本章一直在说，后面 token 的 K/V 是在前面内容的基础上算出来的。前缀一改，后面的缓存全部作废，保留下来的内容要重新 prefill。论文 §1 把这叫作重 prefill 成本与准确率之间的两难：删得越早，要重算的后缀越长。

**两个新操作。** 论文 §3.3 在普通的 `append`（本章的增量写入，Eq. 3）之外定义了两个操作，文本历史本身不改：

- `drop(P)`（Eq. 4）：把集合 P 里的 KV 条目排除在以后的注意力之外。保留条目的位置编码不变，下一个 `append` 的位置编号也不变。实现上可以只在 mask 里把它们设成“不看”，就像 P11.4 的矩形 mask 多了几列“不看”。
- `repos`（Eq. 5）：把保留下来的 r 个 key 按逻辑顺序改到位置 0…r−1，下一个 token 用位置 r。对 RoPE，把一个 key 从位置 p 挪到 q 只需 $K\leftarrow R(q)R(p)^{-1}K$；**V 原样不动**，因为 V 本来就没有旋转。为什么旋转一下就够，见 [P12.3](#p12-rope) 的相对位置性质。

AKV 的接口在原来的 OpenAI 格式请求上加两个字段（[README](https://github.com/thu-nics/AKV/blob/c567a429a058f1abce3b2879e69978b09e0fda4c/README.md) 的 Quick Start 示例和字段表）：`"drop_message": {"4": [2]}` 表示处理完第 4 条消息后丢掉第 2 条消息的 KV，`"reposition": [4]` 表示处理完第 4 条后压紧位置。论文 Fig. 6 画的是另一种写法（`"cache": {"drop": …}`），本书以仓库 README 为准。

**关键：drop + repos 不等于重新 prefill。** 保留下来的 token 当初算 K/V 时，是**看过**被删内容的。第 1 层的 K/V 只由 token 本身和位置决定，所以 repos 之后能和重算完全一样；但从第 2 层起，输入已经是混入了上下文的隐藏状态，被删内容的信息就留在保留 token 的 K/V 里。论文 §3.4 和 Fig. 5 说的正是这个“历史依赖”：同样的 token、同样的位置，操作顺序不同，缓存状态就不同；Fig. 1 也注明保留的 KV 仍可能带着被删历史的信息。

**跑一下。** `code/12b_cache_engineering.py` 用 P12 的 RoPE 小模型（随机初始化、float64、种子 2026）。10 个 token，删掉中间第 2–5 个（像一段旧工具结果），再追加 2 个 query token，比较两条路：A 是完整 prefill → `drop` → `repos` → 追加；B 是只把保留的 6 个 token 重新 prefill → 追加。

| 模型层数 | 第 1 层（layer 0）K/V 最大差 | 更高层 K/V 最大差 | query logits 最大差 |
| --- | --- | --- | --- |
| 1 | 2.2e-16 / 0 | 无 | 4.4e-16 |
| 2 | 2.2e-16 / 0 | layer 1：0.128 / 0.126 | 1.53e-2 |
| 4 | 2.2e-16 / 0 | 最大到 layer 3：0.384 / 0.499 | 7.34e-2 |

三组对照把原因钉住：

- 2 层模型的 layer 1 里，排在被删段**前面**的 token 0、1 差值为 0（它们本来就看不到后面的被删段），排在**后面**的 token 6–9 差 0.03–0.13。差异只出现在“看过被删内容”的 token 上。
- 只删最后两个 token（没人看过它们）再 repos，4 层模型每层 K/V 都与重算完全一致，logits 差 3.6e-16。
- 1 层模型若只 drop 不 repos，layer 0 的 K 与重算差 2.23，logits 差 1.64e-2：位置没压紧，key 与 query 的相对距离就和重算不同。

`tests/test_cache_engineering.py` 断言以上结论，并检查：不做任何操作时 `append` 与完整前向一致；mask 式 drop 与物理删除结果相同；先转到旧位置再 `rerotate` 等于直接转到新位置；把 `repos` 换成空操作后，1 层的相等就被破坏。

**边界。** 差异不是 bug，也不说明哪条路更好：论文 §3.2 在讲直接改缓存的动机时写道，保留的缓存可能仍带着计算时用过的被删历史信息，并把直接 drop 同时提高准确率、降低成本列为实验观察；这些属于作者报告。本书只验证了“二者在数值上不同、以及为什么不同”，没有验证对准确率的影响。论文摘要报告的效果（数据见 §5.2 Table 1、§5.3 Table 2；相比阈值触发摘要平均 +1.7 分、成本 −27.7%；AKV 输出吞吐最高 1.2×、P95 延迟 1.5× 加速）均为作者报告，本书未复现。本书的 `rerotate` 在 float64 下直接转角度差；AKV 的内核 [`context_reposition.py`](https://github.com/thu-nics/AKV/blob/c567a429a058f1abce3b2879e69978b09e0fda4c/akv/system/sglang/kernels/ops/attention/context_reposition.py) 第 93–95 行注释说明，它刻意用 FP32 先逆转再正转，因为角度差展开会改变舍入，在深层模型里被放大；第 114–115 行注释说明只改旋转子空间，K 的其余部分和全部 V 原样复制。radix tree、淘汰与恢复（论文 §4）属于服务系统，本书不实现，P15 讲分页缓存时可以对照。

## 原始来源与版本

核验日期：2026-10-19。本章 S1 是教学实现，既不是 HF Cache API 的完整实现，也不是 vLLM 服务实现。

1. [Transformers 4.57.1 Caching](https://huggingface.co/docs/transformers/v4.57.1/en/cache_explanation)：逐层 K/V、增量计算、cache position 与组合注意力 mask；只在推理使用缓存的边界。
2. Kwon 等，[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)，§2–3：KV cache 与服务内存管理动机。分页和连续批处理留给后续，不将本地 `cat` 称为 PagedAttention。
3. [code/07_baby_gpt.py](../code/07_baby_gpt.py) 与 [code/12_kv_cache.py](../code/12_kv_cache.py)：本章形状、容量策略、计时范围及可运行数值的直接依据。
4. 大纲 S15 的 [vLLM Paged Attention 设计文档](https://docs.vllm.ai/en/latest/design/paged_attention/)（动态文档，访问核验日 2026-10-19）仅作后续服务实现导航。本章不依赖 vLLM、没有运行其内核，也未实现分页缓存；本地 S1 的行为以配套脚本为准。
5. Fu 等，[Cache Engineering for Agents](https://arxiv.org/abs/2610.28577)（arXiv:2610.28577v1），§1、§3.2–§3.4（Eq. 3–5，Fig. 1、Fig. 5）、§4，核对日期 2026-10-27；代码 [thu-nics/AKV](https://github.com/thu-nics/AKV)（Apache-2.0，核对提交 `c567a42`）的 README 与 `akv/system/sglang/kernels/ops/attention/context_reposition.py`。P11.10 的依据；[code/12b_cache_engineering.py](../code/12b_cache_engineering.py) 为本书自写实现。

下一单元 [P12](#p12-rope) 将绝对位置表换成 Q/K 旋转，并继续用同参数全量/增量 logits 对照检查缓存。
