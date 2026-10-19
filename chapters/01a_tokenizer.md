# 第1A章：Tokenizer——从字符、字节到 BPE

上一章把“苹果”当成一个编号，再从 embedding 表中取出一行。现在我们打开这个暂时封住的盒子：谁决定“苹果”是一个 token，还是“苹”“果”两个 token？一串 emoji 又该怎样处理？

**Tokenizer（分词器）的工作是把文本变成 token ID 序列，并提供对应的解码方式。** Token 不一定是词，也不一定是字。它可能是子词、一个标点、带空格的片段，甚至一个字符的部分字节。训练语言模型之前，我们需要先知道预测的单位是什么。

本章属于扩展计划 P02，暂以“1A”插在词嵌入与旧第 2 章之间，保留旧章节链接。本章的两个实验都使用 CPU 和小型合成语料，不下载任何语言模型。

## 1A.1 三种切法：整词、字符与字节

想象要用积木拼一句话。整词积木大，拼起来步数少，但新词可能不在盒子里；字符积木小，组合灵活，但词表仍要考虑哪些字符；字节积木更基础，UTF-8 文本可以拆成 0～255 的字节值，不过同一句话通常需要更多块。

| 单位 | 直观好处 | 需要留意 |
| --- | --- | --- |
| 整词 | 常见词可用一个 ID 表示 | 新词、拼写变化和中文边界；词表可能很大 |
| Unicode 字符 | 实现简单，便于教学 | “字符”不等于视觉上的一个字形；未收录字符仍要处理 |
| UTF-8 字节 | 256 个基础字节覆盖合法 UTF-8 文本 | 一个汉字、emoji 可能跨多个字节，序列更长 |
| 子词 / 字节 BPE | 把频繁小片段合成较大单位 | 结果依赖训练语料、预处理、词表与合并规则 |

先用标准 Python 观察：

```python
text = "你🙂"
print(len(text))                # 2 个 Unicode 码点
print(list(text.encode("utf-8")))
# [228, 189, 160, 240, 159, 153, 130]
```

这段文本有两个码点，却有七个 UTF-8 字节。`len(text)` 也不总等于你肉眼数出的字形数：`e` 加组合重音可以显示成一个字形，却是两个码点。后面测试会特意保留这种情况。

> [!NOTE]
> **“覆盖文本”不等于“理解文本”。** 字节词表能表示没见过的文字，并不意味着模型学会了这门语言。Tokenizer 解决的是编码问题，语言能力仍取决于数据、模型和训练。

### 对照旧第 8 章：保留 Baby-GPT 的字符基线

旧第 8 章和配套的 `code/07_baby_gpt.py` 刻意采用更简单的字符级分词器：

```python
unique_chars = sorted(set(text_corpus))
vocab_size = len(unique_chars)
char2idx = {ch: i for i, ch in enumerate(unique_chars)}
idx2char = {i: ch for i, ch in enumerate(unique_chars)}
encode = lambda text: [char2idx[ch] for ch in text]
decode = lambda ids: "".join(idx2char[i] for i in ids)
```

这里的 `text_corpus` 是源码中的三首训练诗句拼接，词表恰好有 **57 个不同字符**，包括汉字和标点；57 不是汉语字符总数，也不是模型的固定常数。换训练语料，词表和编号就可能改变。每个已收录字符对应一个 token，例如“床前”编码为两个 ID，解码还原为“床前”。

它没有未知字符 token，也没有字节回退：训练诗句没有“你”，所以 `encode("你")` 会在字典查找时抛出 `KeyError`，不能编码任意中文或 emoji。相比之下，本章字节 BPE 从全部 256 个字节开始，能编码任意合法 UTF-8 文本；之后才按训练语料中学到的规则把常见字节片段合并成较长 token。能编码未见字符，不等于语言模型理解它。

**本书保留 Baby-GPT 的字符分词器作为基线，不在本章偷偷替换其训练输入。** 字符基线让初学者直接观察“一字接一字”；字节 BPE 展示开放字符覆盖与序列长度的取舍。若未来替换 tokenizer，就要同时重建词表、embedding 和输出层，并重新训练，不能让新 ID 直接查旧模型的表。

## 1A.2 BPE：把常一起出现的两块积木粘起来

BPE 是 Byte Pair Encoding 的缩写。用于文本时，其核心思想是反复找出常见的相邻符号对，为合并结果分配一个新符号。最初的符号可以是字符，也可以是字节；本章选择字节，便于让未见过的中文和 emoji 仍能往返。

先用 ASCII 字符 `banana` 看直觉，ASCII 字母的字节值正好容易读：

1. 从 `b a n a n a` 开始，统计所有相邻对。
2. `a n` 出现两次，`n a` 也出现两次。出现并列时必须规定规则，本书按 ID 对排序，所以先选 `a n`。
3. 把不重叠的 `a n` 合并成新 token，得到 `b an an a`。
4. 再统计当前序列，继续选择并合并，直到达到指定次数或没有相邻对。

单个字符串上不同并列规则可能产生不同路线，算法不是在寻找唯一正确的“英语词根”。我们的完整实验同时训练三篇短文，前四条规则实际为：

| 顺序 | 相邻 ID 对 | 新 ID | 对应字节片段 |
| --- | --- | ---: | --- |
| 1 | `(97, 110)` | 256 | `an` |
| 2 | `(98, 256)` | 257 | `ban` |
| 3 | `(256, 97)` | 258 | `ana` |
| 4 | `(257, 258)` | 259 | `banana` |

![BPE 训练与使用](assets/diagrams/bpe_pipeline.svg)

语料中的空格也有字节值。本章没有按空格预切分，所以某个 token 可以包含空格，甚至覆盖一整句训练文本。这是为了看清最小机制而做的简化，不代表所有生产分词器都这样处理边界。每篇训练文档则独立统计，绝不把前一篇的结尾和后一篇的开头粘起来。

## 1A.3 先分清 train、encode、decode

- `train`：从语料得到词表与合并规则。此后不要为了每个新问题都重新训练。
- `encode`：把新文本转成初始字节 ID，再按已学到的优先顺序合并。
- `decode`：查出各个 ID 对应的 bytes，拼接完整字节串，然后进行 UTF-8 解码。

为什么不能逐 token 解码成字符再拼接？因为一个 token 可能只有汉字的第一个字节，单独看不是合法 UTF-8。先拼 bytes，才能恢复完整字符。

同一个模型必须配套使用它训练时的 tokenizer。换了词表顺序却不调整 embedding 和输出层，就像把图书馆的索引卡全部换位：程序仍会取到一行数字，但取的不是原来的那行。增加特殊 token、改模板或改归一化规则也可能影响模型输入，不能把所有 tokenizer 当成可随意互换的插件。

## 1A.4 逐行代码：最小字节 BPE

完整实现是 [code/08_byte_bpe.py](../code/08_byte_bpe.py)，只有标准库依赖。先看基础字典：

```python
self.vocab = {i: bytes([i]) for i in range(256)}
self.merges = {}
```

`vocab[97]` 是字节 `b'a'`，不是某个语言模型里通用的“a 的 ID”。编号规则只属于这个教学实现。新合并 token 从 ID 256 开始追加；`merges` 按插入顺序记录优先级。

训练的关键几行是：

```python
counts = Counter()
for ids in sequences:
    counts.update(zip(ids, ids[1:]))
pair = min(counts, key=lambda item: (-counts[item], item))
new_id = len(self.vocab)
self.merges[pair] = new_id
self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
sequences = [merge_pair(ids, pair, new_id) for ids in sequences]
```

`zip(ids, ids[1:])` 把每个位置和紧跟它的位置配对；`Counter` 累加出现次数；`min` 的排序键先比较负频数，所以频数最高的优先，再用 ID 对解决并列。随后给合并结果分配 ID、拼接其 bytes，并更新每篇文档。完整代码在 `counts` 为空时停止，避免对空集合求最小值。

`merge_pair` 必须处理重叠。例如把 `aaa` 里的 `aa` 合并，结果应是 `[aa, a]`，不能同时把中间的 `a` 用两次：

```python
while i < len(ids):
    if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
        result.append(new_id)
        i += 2
    else:
        result.append(ids[i])
        i += 1
```

匹配成功时向前走两格，否则只走一格。频数统计可以数到重叠候选，真正替换时只合并不重叠对。

编码阶段不重新找“新文本里最常见的一对”，而是按训练好的规则顺序应用合并。解码的核心则是：

```python
raw = b''.join(self.vocab[token_id] for token_id in ids)
text = raw.decode('utf-8')
```

完整脚本会先检查 ID 是否存在。对 `decode(encode(text))`，合法 UTF-8 文本可以原样恢复；任意乱写的 ID 序列则不一定构成合法 UTF-8，例如单个字节 255。代码保留解码异常，不用替换字符掩盖错误。

## 1A.5 跑一次：短序列不等于好语言模型

在仓库根目录执行：

```bash
python3 code/08_byte_bpe.py
python3 -m unittest discover -s tests -v
```

训练语料是 `banana banana`、`banana bandana` 和 `你好，你好！`。请求最多 20 次合并，但语料提前没有相邻对可合并，因此实际词表大小为 275，即 256 个基础字节加 19 个新 token。实测输出中的计数如下：

| 文本 | Python 字符数 | UTF-8 字节数 | 教学 BPE token 数 | 往返 |
| --- | ---: | ---: | ---: | --- |
| `banana banana` | 13 | 13 | 1 | 原样恢复 |
| `你好，你好！` | 6 | 18 | 1 | 原样恢复 |
| `新词🙂 café` | 8 | 16 | 16 | 原样恢复 |
| 空字符串 | 0 | 0 | 0 | 原样恢复 |
| `e` 加组合重音 | 2 | 3 | 3 | 原样恢复 |

把整句训练文本合成一个 token，可以让训练集看起来非常短，却没有证明它对新文本也有效。第三行从未出现在训练语料中，仍然可编码和恢复，但几乎没有合并收益。这正好提醒我们：**能编码、token 少、模型理解得好，是三个不同的问题。**

测试不只检查几个好看的输出，还检查空字符串、换行空格、带修饰符的 emoji、组合字符、非法 ID、确定性并列规则、重叠合并，以及文档边界。实现优先让你能读懂，反复扫描语料会很慢，不能拿它直接训练大规模词表。

## 1A.6 对照真实库：Hugging Face Tokenizers

可选实验 [code/09_hf_tokenizer.py](../code/09_hf_tokenizer.py) 固定使用 `tokenizers==0.21.4`。核验和运行日期为 2026-10-18，官方接口来源锁定到 [v0.21.4 代码与文档目录](https://github.com/huggingface/tokenizers/tree/v0.21.4)。这不是要求你使用最新版本，而是把本章实验条件固定下来。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-tokenizer.txt
.venv/bin/python code/09_hf_tokenizer.py
```

关键配置如下：

```python
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
    add_prefix_space=False, use_regex=False
)
tokenizer.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=276,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    show_progress=False,
)
```

`ByteLevel` 提供字节层面的表示；完整的初始字节字母表让未见过的 UTF-8 文本仍有表示；不自动补前缀空格，是为了保留原输入；禁用正则预切分，是为了更接近上面的教学设置。我们没有设置会改变大小写或 Unicode 的 normalizer，也没有添加特殊 token。

随后用同样三篇短文训练，用相同样例检验 round-trip。比较的是“是否原样恢复”和“词表与序列长度”，**不要求 token ID 相同**：真实库对字节的内部表示、初始 ID 和并列合并规则可能不同。这个实验也不是某个现成聊天模型的 tokenizer。

## 1A.7 生产环境还多了哪些步骤？

真实流水线常包含归一化、预切分、子词编码、特殊 token 插入、padding / truncation 和解码。逐项检查，比只问“用了 BPE 吗”更有用。

- **归一化**可能把大写变小写或统一 Unicode 表示。这有时有帮助，却可能让解码无法恢复原文；本章刻意不做。
- **特殊 token**可表示序列结束、角色边界或填充。它们是明确约定，不是看到字符串 `<eos>` 就一定自动成为控制符。本章教学实现把这样的字符串当普通文字。
- **Chat template**把角色与消息内容组织成模型期望的输入。通常应使用模型配套模板，不要自行把“用户：”等字样当成所有模型通用的格式。
- **截断**会丢掉内容；**填充**需要配合 mask；这两件事不等于改变原文本的 token 含义。

Token 数会影响上下文容量、部分 API 的计费和计算开销；但不能用“一个中文字就是一个 token”来估算所有模型。序列长度变化会影响后面的 attention 和 KV cache 成本，词表变大又会增加 embedding / 输出层的规模，选择词表是折中而不是越大越好。

## 1A.8 自测与下一站

1. 一个 emoji 是一个 token 吗？
2. `train` 和 `encode` 为什么不能混成一步？
3. 把一段训练文本合成一个 token，是否证明 tokenizer 在任何新文本上都更好？
4. 为什么解码时先拼 bytes，再转成字符串？
5. 只用语料里出现过的字符建立词表，与从 256 个字节起步有什么区别？

参考答案：取决于具体 tokenizer；训练生成规则，编码使用已有规则；不能，需要在代表性的未见数据上评估；单个 token 可能不是完整 UTF-8 字符；字符词表可能遇到未收录字符，而完整字节表可表示合法 UTF-8，但并不赋予模型理解能力。

现在可以把前两章连起来：**文本 → tokenizer → token ID → embedding → 后续网络**。查表得到的初始向量还不知道当前“苹果”指水果还是电脑，接下来回到注意力章节，看模型如何结合周围内容。

原始资料：[Neural Machine Translation of Rare Words with Subword Units（v5）](https://arxiv.org/abs/1508.07909v5) 解释子词 BPE 的语言建模背景；本章字节实现是教学变体，不是该论文实验的复现。库接口依据上述 Hugging Face 固定标签；本文没有以可变的 `main` 文档作为未锁定的版本承诺。
