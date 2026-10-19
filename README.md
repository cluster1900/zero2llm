# 写给初学者的 Transformer：从零开始的大模型极简数学与代码之旅

[![Deploy to GitHub Pages](https://github.com/cluster1900/zero2llm/actions/workflows/pages.yml/badge.svg)](https://github.com/cluster1900/zero2llm/actions/workflows/pages.yml) [在线阅读](https://cluster1900.github.io/zero2llm/) · [下载最新 EPUB](https://github.com/cluster1900/zero2llm/releases/latest/download/transformer_from_scratch.epub)

这是一本循序渐进的 Transformer 入门教程：用生活类比、图解、逐行 Python/PyTorch 代码和 Baby-GPT 实验，把词向量、QKV 注意力、位置编码、Transformer Block 与现代大模型串成一条从 0 到 1 的学习路线。

> **阅读准备**：能够阅读 Python 变量、循环和列表即可开始；用到的代数、向量与三角函数知识会结合例子说明。
>
> **核心宗旨**：用生活实例、几何直觉、逐行代码讲解与全彩图解，带你看懂当今大模型（ChatGPT、Claude、DeepSeek）共同使用的核心架构——**Transformer**。这里的“小模型”是教学实验，不等于真实聊天模型。

![图书封面](assets/cover.jpg)

---

## 🚧 前沿大模型扩展进度

[扩展大纲](docs/expansion-outline.md) 已在 [PR #1](https://github.com/cluster1900/zero2llm/pull/1) 审核通过，按 [bot-hub #14](https://github.com/cluster1900/bot-hub/issues/14) 每批 2–3 个单元推进。

- **P00–P02 正文与示例已实现（[PR #2](https://github.com/cluster1900/zero2llm/pull/2)）**：P00 阅读指南、P01 词嵌入、P02 Tokenizer/BPE；CPU 字节 BPE、HF Tokenizers 0.21.4 对照与回归测试。
- **当前可读**：23 个章节文件（含前言、插入的第 1A 章与附录），21 个编号示例脚本及共享 attention_core.py 辅助模块；本分支 HTML / EPUB 同步更新。公开站点与 Release 在合并/发布前仍可能是旧版。
- **P03–P05 正文与 CPU 示例已实现**：注意力动机、QKV/稳定 Softmax/掩码、多头张量账本；因果性、padding/全 mask 行、形状与参考实现回归。验收记录见[第二批验证](docs/batch2-validation.md)。
- **P06–P08 正文与 CPU 示例已实现**：位置编码、残差/LayerNorm/FFN、三类架构与生成循环；保留手算表，增加数值、自测和变异回归。见[第三批验证](docs/batch3-validation.md)。
- **P09–P11 正文与 CPU 示例已实现**：训练最小闭环、Baby-GPT S0 基线和 S1 KV cache；训练/留出边界、checkpoint、完整/增量 logits 对齐与内存账。见[第四批验证](docs/batch4-validation.md)。
- **P12–P14 正文与 CPU 示例已实现**：S2 RoPE 与长上下文边界、S3 RMSNorm/SwiGLU、S4 GQA 与独立 MLA 小矩阵；各阶段全量/增量对齐、数值和参数/缓存账。YaRN 是论文机制导读，不宣称完整复现其训练。见[第五批验证](docs/batch5-validation.md)。
- **P15–P17 正文与独立 CPU 实验已实现**：分块在线 Softmax、分页缓存与调度账本，SWA / 线性递推，以及官方 CED 数据流的教学模型。三个实验不改变 S4 主线，不代表 CUDA kernel、生产服务或真实 DeepSeek 权重复现。见[第六批验证](docs/batch6-validation.md)。
- **P18–P20 正文与 CPU 实验已实现**：S5 教学 MoE、预训练数据与恢复训练、Transformers 小模型 CPU 对照与分片内存账本；TRL/DeepSpeed/Megatron 的角色及未运行多卡边界另行说明。见[第七批验证](docs/batch7-validation.md)。
- **尚未按扩展大纲实现**：P21–P28；旧第 9–10 章仍保留可读，但不代表新专题已完成。
- **后续路线**：Baby-GPT → KV cache → RoPE → RMSNorm / SwiGLU → GQA → MoE；每批审核后才继续，不把路线图当成已有功能。

为保持旧链接，Tokenizer 插为第 1A 章；P09/P11 分别以 07a/08a 文件插入，P12–P14 以 08b/08c/08d 插入，P15–P17 以 08e/08f/08g 插入，P18–P20 以 08h/08i/08j 插入，不重编号旧第 8–10 章。验证条件和结果见[首批记录](docs/batch1-validation.md)、[第二批记录](docs/batch2-validation.md)、[第三批记录](docs/batch3-validation.md)、[第四批记录](docs/batch4-validation.md)、[第五批记录](docs/batch5-validation.md)、[第六批记录](docs/batch6-validation.md)与[第七批记录](docs/batch7-validation.md)。

## 📖 当前版本目录导航

当前版本包含 23 个章节文件（含前言、Tokenizer 插入单元与附录）。每一章都配有生活类比、数学原理拆解、表格或流程图；能运行的部分同时提供纯 Python 手算版或 PyTorch 实验：

| 章节与标题 | 核心生活类比 / 几何直觉 | 用到的数学知识 | 配套可运行代码 |
| :--- | :--- | :--- | :---: |
| [**00. 前言与阅读指南**](chapters/00_preface.md) | 大模型的黑盒魔法与探索路线图 | 初等代数与逻辑 | - |
| [**01. 从词语到空间向量**](chapters/01_word_embedding.md) | 水果特征打分坐标系、“国王 - 男人 + 女人 = 女王” | 平面向量、模长、点积、夹角余弦 | [`code/01_vector_similarity.py`](code/01_vector_similarity.py) |
| [**01A. Tokenizer：字符、字节与 BPE**](chapters/01a_tokenizer.md) | 由小积木合成频繁片段；编码与理解的区别 | 频数统计、字节、确定性合并 | [`code/08_byte_bpe.py`](code/08_byte_bpe.py)<br>[`code/09_hf_tokenizer.py`](code/09_hf_tokenizer.py)（选修） |
| [**02. 注意力机制的直觉**](chapters/02_why_attention.md) | 传话游戏、鸡尾酒会聚光灯、一词多义危机 | 矩阵并行 vs 串行路径 | - |
| [**03. 拆解注意力三剑客 Q、K、V**](chapters/03_qkv_self_attention.md) | 图书馆借书标签检索、为什么除以 $\sqrt{d_k}$ | 向量数量积、方差爆炸、Softmax 归一化 | [`code/02_manual_attention.py`](code/02_manual_attention.py)<br>[`code/03_scaled_dot_product.py`](code/03_scaled_dot_product.py) |
| [**04. 多头注意力机制**](chapters/04_multi_head_attention.md) | 三个臭皮匠诸葛亮、低维空间切片魔术 | 空间投影、张量维度切分与拼接 | [`code/04_multi_head_attention.py`](code/04_multi_head_attention.py) |
| [**05. 位置编码与座位号**](chapters/05_positional_encoding.md) | 小猫吃小鱼顺序危机、石英手表三针定乾坤 | 正弦波 $\sin$、余弦波 $\cos$、和角公式 | [`code/05_positional_encoding.py`](code/05_positional_encoding.py) |
| [**06. 积木组件：残差、LN 与 FFN**](chapters/06_residual_layernorm_ffn.md) | 考试保底底分、全科标准化、每个词闭门深思 | 导数恒等 $+1$ 路径、Z-Score 均值与方差 | [`code/06_transformer_block.py`](code/06_transformer_block.py) |
| [**07. 编码器与解码器**](chapters/07_encoder_decoder.md) | 资料整理与写作、因果掩码闭卷考试、逐步生成 | 标签错位、温度分布、EOS/长度停止 | [`code/10_generation_loop.py`](code/10_generation_loop.py) |
| [**P09. 训练最小闭环**](chapters/07a_training_loop.md) | 练习与留出考试、错误标签作弊反例 | 交叉熵、梯度、loss reduction | [`code/11_training_loop.py`](code/11_training_loop.py) |
| [**08. 实战：手搭 Baby-GPT**](chapters/08_baby_transformer.md) | 从第一行代码训练一个能背三首短诗的微型模型 | 交叉熵损失、温度采样、梯度下降 | [`code/07_baby_gpt.py`](code/07_baby_gpt.py) |
| [**P11. KV Cache**](chapters/08a_kv_cache.md) | 保存历史 K/V，只计算新位置 | 矩形因果 mask、偏移、张量字节数 | [`code/12_kv_cache.py`](code/12_kv_cache.py) |
| [**P12. RoPE 与 YaRN**](chapters/08b_rope_yarn.md) | 旋转时钟与历史位置 | 二维旋转、相对相位、缓存偏移与外推边界 | [`code/13_rope.py`](code/13_rope.py) |
| [**P13. RMSNorm 与 SwiGLU**](chapters/08c_modern_block.md) | 幅度刻度与可学习门控 | 均方根、SiLU、参数预算与单变量比较 | [`code/14_modern_block.py`](code/14_modern_block.py) |
| [**P14. MHA / MQA / GQA 与 MLA**](chapters/08d_grouped_kv_mla.md) | 查询分组共享资料与低秩压缩 | KV 头映射、缓存字节、低秩矩阵与位置项 | [`code/15_grouped_kv.py`](code/15_grouped_kv.py) |
| [**P15. FlashAttention、分页与调度**](chapters/08e_attention_systems.md) | 分块算账、分页账本与排队服务 | 在线 Softmax、块表、TTFT / TPOT | [`code/16_attention_systems.py`](code/16_attention_systems.py) |
| [**P16. 稀疏、混合与线性注意力**](chapters/08f_sparse_linear_attention.md) | 有限窗口与压缩的累积笔记 | SWA mask、感受野、递推与核函数 | [`code/17_sparse_linear_attention.py`](code/17_sparse_linear_attention.py) |
| [**P17. CED 因果编码器—解码器**](chapters/08g_ced_architecture.md) | 整理全局资料与局部续写 | 全局 KV 来源、因果性、重放近似与路径计数 | [`code/18_ced_architecture.py`](code/18_ced_architecture.py) |
| [**P18. MoE 专家与路由**](chapters/08h_moe.md) | 分流工作与共享知识 | top-k 权重、负载、容量与参数预算 | [`code/19_moe.py`](code/19_moe.py) |
| [**P19. 预训练数据与恢复训练**](chapters/08i_pretraining_pipeline.md) | 整理资料与保存练习进度 | 去重切分、文档边界、损失归约与状态恢复 | [`code/20_pretraining_pipeline.py`](code/20_pretraining_pipeline.py) |
| [**P20. 训练框架与分片账本**](chapters/08j_training_frameworks.md) | 手写步骤与框架分工 | 标签错位、模型状态、数据并行与分片字节数 | [`code/21_training_frameworks.py`](code/21_training_frameworks.py) |
| [**09. 走向现代大模型 (LLMs)**](chapters/09_modern_llms.md) | RoPE 旋转位置编码、RMSNorm、SwiGLU、DeepSeek MoE | 复数平面旋转、均方根、尺度定律与涌现 | - |
| [**10. 附录与黑话词典**](chapters/10_appendix.md) | 数学公式速查卡、深度学习黑话翻译大白话词典 | 全书数理汇总 | - |

---

## 💻 21 个配套代码实验（含可选框架对照）

历史第二批验证了当时的 9 个编号示例；当前共有 21 个，最新执行环境与结果见[第七批记录](docs/batch7-validation.md)。

原有第 1～7 个实验的验证环境为 Python 3.12 与 PyTorch 2.14.1。首批新增第 8、9 个实验在 Python 3.12.14 上运行验证。第 1、2、8、10、16 个，以及第 21 个的默认账本模式只依赖标准库；第 3～7、12～15、17～19 个需要 PyTorch；第 11 个的 --scalar-only、第 20 个的 --data-only 为标准库，其完整训练实验需要 PyTorch；第 9 个选修需要单独安装 tokenizers==0.21.4；第 21 个 --hf 对照另需 requirements-frameworks.txt 中固定的依赖。

1. **[`code/01_vector_similarity.py`](code/01_vector_similarity.py)**：纯 Python 手写向量内积与余弦相似度，验证水果空间距离与国王-女王向量平移。
2. **[`code/02_manual_attention.py`](code/02_manual_attention.py)**：纯 Python 零依赖手算 3 个字（“我”、“爱”、“学”）在注意力机制内部经历的每一步数字变化。
3. **[`code/03_scaled_dot_product.py`](code/03_scaled_dot_product.py)**：用数值实验揭开“为什么必须除以 $\sqrt{d_k}$”的方差危机，以及因果掩码如何用 $-\infty$ 阻隔未来。
4. **[`code/04_multi_head_attention.py`](code/04_multi_head_attention.py)**：多头注意力的 PyTorch 完整实现与张量维度变换追踪。
5. **[`code/05_positional_encoding.py`](code/05_positional_encoding.py)**：正弦余弦位置编码矩阵计算与控制台 ASCII 字符波浪热力图打印。
6. **[`code/06_transformer_block.py`](code/06_transformer_block.py)**：Pre-LN Transformer 积木块拼装（LayerNorm + MHA + 残差 + FFN）。
7. **[`code/07_baby_gpt.py`](code/07_baby_gpt.py)**：三首诗分开训练的微型自回归模型。种子 42 时，损失从约 4.13 降到 0.0009，能从真实开头背出三首诗；提示“明月”则不会接回《静夜思》。

8. **[`code/08_byte_bpe.py`](code/08_byte_bpe.py)**：确定性字节 BPE，中文、emoji 与未见文本的往返验证；无需模型下载。
9. **[`code/09_hf_tokenizer.py`](code/09_hf_tokenizer.py)**（选修）：固定 HF Tokenizers 0.21.4，在相同合成语料上对照；不要求两种实现的 ID 相同。

10. **[`code/10_generation_loop.py`](code/10_generation_loop.py)**：标准库手算 mask、标签错位、温度概率与可复现生成追踪，验证 EOS 和新增长度停止；固定 logits，不是训练模型。

11. **[`code/11_training_loop.py`](code/11_training_loop.py)**：标准库交叉熵/梯度手算，CPU 数据切分、一次更新、整首留出与重复种子实验；不将共享词表设置称为严格独立评估。
12. **[`code/12_kv_cache.py`](code/12_kv_cache.py)**：同一 S0 参数的增量 S1；完整 logits 对齐、矩形 mask、缓存字节账、首 token/后续 token CPU 计时。

13. **[`code/13_rope.py`](code/13_rope.py)**：S2 旋转 Q/K、缓存位置偏移、短序列训练后的长度外推边界。
14. **[`code/14_modern_block.py`](code/14_modern_block.py)**：S3 逐项替换 RMSNorm / SwiGLU，参考公式、梯度与参数预算比较。
15. **[`code/15_grouped_kv.py`](code/15_grouped_kv.py)**：S4 紧凑 GQA 缓存、MHA/MQA 边界和独立教学 MLA 小矩阵；不是生产 MLA kernel。

16. **[`code/16_attention_systems.py`](code/16_attention_systems.py)**：CPU 分块注意力、分页所有权与服务调度模拟，不是 GPU 性能基准。
17. **[`code/17_sparse_linear_attention.py`](code/17_sparse_linear_attention.py)**：滑动窗口、独立线性注意力与检索边界实验。
18. **[`code/18_ced_architecture.py`](code/18_ced_architecture.py)**：独立因果 encoder/global-KV/decoder 教学数据流；区分精确前向与近似 replay。

19. **[`code/19_moe.py`](code/19_moe.py)**：S5 教学 MoE 路由、共享专家、负载与缓存对照。
20. **[`code/20_pretraining_pipeline.py`](code/20_pretraining_pipeline.py)**：合成数据清理、packing 边界与可恢复 CPU 训练。
21. **[`code/21_training_frameworks.py`](code/21_training_frameworks.py)**：标准库分片账本与可选固定版本 Transformers CPU smoke；不下载大模型权重。

### CPU 快速验证

```bash
python3 code/01_vector_similarity.py
python3 code/08_byte_bpe.py
python3 code/10_generation_loop.py
python3 code/11_training_loop.py --scalar-only
python3 -m unittest discover -s tests -v
# 可选 HF 对照：在虚拟环境内安装，不需要 GPU 或模型权重
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-tokenizer.txt
.venv/bin/python code/09_hf_tokenizer.py
```

### P20 可选框架 CPU 对照

使用独立环境，不要原地升级基础 .venv。固定组合为 Transformers 4.51.3、Accelerate 1.6.0、Tokenizers 0.21.4 与书中 PyTorch 基线；这是验证快照，不是最新版本推荐。安装之后，实验使用本地随机 tiny 模型与合成 token，不下载预训练权重：

```bash
python3 -m venv .venv-frameworks
.venv-frameworks/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
.venv-frameworks/bin/python -m pip install -r requirements.txt -r requirements-frameworks.txt -r requirements-build-static.txt
.venv-frameworks/bin/python code/21_training_frameworks.py --hf
.venv-frameworks/bin/python -m unittest discover -s tests -v
```

默认运行 code21 只检验标准库内存账本；只有 --hf 真正执行框架对照。依赖缺少或版本不符会明确跳过，不能把跳过当作通过。DeepSpeed/Megatron 多卡、CUDA 和 TRL 后训练没有在本批运行。

### 快速运行代码：

```bash
# 第一次使用时（已有 .venv 可跳过）
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# 激活预置虚拟环境
source .venv/bin/activate

# 运行任一脚本，例如微型大模型训练
python code/07_baby_gpt.py
```

如果想先从手算和纯 Python 实验入手，可以直接运行 `python3 code/01_vector_similarity.py` 和
`python3 code/02_manual_attention.py`，它们不需要安装 PyTorch。重新编译 EPUB/HTML 还需要系统中的
`pandoc`（本书验证版本为 **3.9**）和 Node.js 22 或更高版本（macOS 可运行 `brew install pandoc node`），并先运行 `npm ci` 安装锁定版本的图表构建工具。阅读成品不需要这些依赖。

---

## 📱 交付成果物与随时阅读方式

项目已全自动编译并生成了三大格式的成品交付物，位于 `dist/` 目录下：

```
dist/
├── transformer_from_scratch.html  # 🌟 现代化响应式 Web 阅读器（图表可离线阅读，公式排版需网络）
├── index.html                       # 🌟 对应 Web 站点入口文件
├── transformer_from_scratch.epub  # 📚 标准 EPUB 电子书 (支持 iPad/Apple Books/Kindle)
├── transformer_from_scratch.md    # 📝 完整合并版 Master Markdown 文档
├── code/                            # 🧪 网页中“打开配套源码”所需的 Python 示例
└── assets/                          # 🎨 高清封面图与矢量架构图 (SVG)
```

### 1. 网页端观看（最佳排版体验）：
- **直接双击打开** [`dist/index.html`](dist/index.html) 或 [`dist/transformer_from_scratch.html`](dist/transformer_from_scratch.html) 即可在任何浏览器（Safari, Chrome, Edge）中畅读！
- 特性：
  - 支持 **深色 (Dark) / 浅色 (Light)** 主题一键切换；
  - 集成 **MathJax 3** 高清数学公式渲染；
  - **预渲染流程图**，断网仍显示，点击可查看原尺寸；
  - 左侧常驻章节目录跳转，顶部实时阅读进度条；
  - 代码块高亮显示与一键下载功能。

  公式排版和代码高亮通过 CDN 加载，需要网络；正文、全部图表和代码可离线阅读。

### 2. 电子书阅读器观看（EPUB）：
- 文件路径：[`dist/transformer_from_scratch.epub`](dist/transformer_from_scratch.epub)
- 将文件导入 Apple Books 或其他支持 EPUB 的阅读器；Kindle 可通过 Send to Kindle 转换导入，具体支持以阅读器为准。
- 图表在构建时转为 2 倍分辨率 PNG 并打包，离线无需脚本；长流程拆成小图，保持比例并限制宽高。
- 如果阅读器缓存了旧版，移除旧书后重新导入。配套完整代码链接指向 GitHub，打开链接需要网络。

---

## 🛠️ 如何重新编译全书？

如果你修改了 `chapters/` 中的任何 Markdown 章节或添加了新内容，只需运行一行命令：

```bash
npm ci                # 首次构建；会下载用于渲染的 Chrome
python3 build.py       # 生成 EPUB / HTML，并检查图表、资源和链接
npm run check         # 单独检查已有成品
npm run check:layout  # 检查手机、平板、横屏下的图表尺寸，输出预览截图
```

若构建机不能启动 Chromium，可使用受限静态构建路径（需要支持 `alerts` 的 Pandoc；本批使用 3.9，及本地完整 Git 历史）：

```bash
.venv/bin/python -m pip install -r requirements-build-static.txt
.venv/bin/python scripts/build_without_browser.py --baseline 66dbeb94dd9935046af0e859a223775b7adf4e0e
```

此路径仅复用与基准原稿**逐字一致**的 Mermaid 预渲染图片，新/修改的静态 SVG 由固定版本 CairoSVG 转 PNG。遇到改动的 Mermaid 会失败，要求回到正常 Chromium 构建，绝不会用旧图冒充新图。它不代替浏览器布局检查；正常构建方式仍是 `python3 build.py`。

Linux 构建机需要中文字体（例如 `fonts-noto-cjk`）。已有 Chrome 时，可以设置
`PUPPETEER_EXECUTABLE_PATH` 指向其可执行文件；安装依赖时可设置 `PUPPETEER_SKIP_DOWNLOAD=true`。
Mermaid 图必须提供 `accTitle` 和 `accDescr`；渲染错误、缺图或内部链接失效会让构建以非零状态退出。

流水线会自动完成合并、生成全新 EPUB 电子书、生成现代化 Web 读物并校验资源完整性！

---

## 🌟 矢量图例资源一览

网页插图采用 SVG；EPUB 使用构建时生成的 PNG，以兼容不同阅读器。原始矢量图：

- [`assets/diagrams/word_vector_space.svg`](assets/diagrams/word_vector_space.svg)：词嵌入特征空间与水果向量坐标系
- [`assets/diagrams/attention_mechanism.svg`](assets/diagrams/attention_mechanism.svg)：缩放点积注意力 QKV 计算完整流程图
- [`assets/diagrams/multi_head_attention.svg`](assets/diagrams/multi_head_attention.svg)：多头注意力空间切分与拼接流程图
- [`assets/diagrams/positional_encoding_waves.svg`](assets/diagrams/positional_encoding_waves.svg)：正弦余弦位置编码波形谱与时钟原理
- [`assets/diagrams/transformer_block.svg`](assets/diagrams/transformer_block.svg)：现代 Pre-LN Transformer Block 积木架构全貌
- [`assets/diagrams/causal_mask.svg`](assets/diagrams/causal_mask.svg)：因果掩码下三角矩阵与防作弊原理
- [`assets/diagrams/transformer_full_architecture.svg`](assets/diagrams/transformer_full_architecture.svg)：经典 Transformer 编码器-解码器宏观架构

## 更新 Release 下载文件

`python3 scripts/package_release.py` 校验并将当前 `dist/` 打包到 `.release/`，包含 EPUB、Web ZIP 和 SHA-256 校验值。
在 GitHub Actions 手动运行 **Refresh book release downloads**，选择已有 Release 标签，即可替换阅读文件。
也可以在推送到 `main` 的提交消息中加入 `[refresh-release]`，显式更新 `v1.0.0` 的下载文件。
工作流不会移动旧标签；Release 说明会标出本次阅读文件对应的提交，并用新的附件链接替换旧 ZIP 链接。

构建环境提示：本书要求 **Pandoc 3.9 或更新版本**，已验证版本为 **3.9**。[官方 3.9 发布记录](https://pandoc.org/releases.html#pandoc-3.9-2026-02-03)说明该版本加入 Pandoc Markdown 的 `+alerts` 支持；3.5 不支持该组合，独立审核也已复现失败。新版本仍须重跑产物/布局检查，不假定任意系统自带版本均兼容。
