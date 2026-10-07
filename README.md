# 写给初学者的 Transformer：从零开始的大模型极简数学与代码之旅

[![Deploy to GitHub Pages](https://github.com/cluster1900/zero2llm/actions/workflows/pages.yml/badge.svg)](https://github.com/cluster1900/zero2llm/actions/workflows/pages.yml) [在线阅读](https://cluster1900.github.io/zero2llm/) · [下载最新 EPUB](https://github.com/cluster1900/zero2llm/releases/latest/download/transformer_for_highschool.epub)

这是一本面向高中数学水平学习者的 Transformer 入门教程：用生活类比、图解、逐行 Python/PyTorch 代码和 Baby-GPT 实验，把词向量、QKV 注意力、位置编码、Transformer Block 与现代大模型串成一条从 0 到 1 的学习路线。

> **面向人群**：零基础或基础薄弱的初学者（仅需掌握初等代数、基础平面向量点积与三角函数，仅能看懂基础 Python 循环与列表）。  
> **核心宗旨**：用生活实例、几何直觉、保姆级逐行代码与全彩图解，带你看懂当今大模型（ChatGPT、Claude、DeepSeek）共同使用的核心架构——**Transformer**。这里的“小模型”是教学实验，不等于真实聊天模型。

![图书封面](assets/cover.jpg)

---

## 📖 全书目录导航

全书共分为 11 个章节。每一章都配有生活类比、数学原理拆解、表格或流程图；能运行的部分同时提供纯 Python 手算版或 PyTorch 实验：

| 章节与标题 | 核心生活类比 / 几何直觉 | 对应高中数学知识点 | 配套可运行代码 |
| :--- | :--- | :--- | :---: |
| [**00. 前言与阅读指南**](chapters/00_preface.md) | 大模型的黑盒魔法与探索路线图 | 初等代数与逻辑 | - |
| [**01. 从词语到空间向量**](chapters/01_word_embedding.md) | 水果特征打分坐标系、“国王 - 男人 + 女人 = 女王” | 平面向量、模长、点积、夹角余弦 | [`code/01_vector_similarity.py`](code/01_vector_similarity.py) |
| [**02. 注意力机制的直觉**](chapters/02_why_attention.md) | 传话游戏、鸡尾酒会聚光灯、一词多义危机 | 矩阵并行 vs 串行路径 | - |
| [**03. 拆解注意力三剑客 Q、K、V**](chapters/03_qkv_self_attention.md) | 图书馆借书标签检索、为什么除以 $\sqrt{d_k}$ | 向量数量积、方差爆炸、Softmax 归一化 | [`code/02_manual_attention.py`](code/02_manual_attention.py)<br>[`code/03_scaled_dot_product.py`](code/03_scaled_dot_product.py) |
| [**04. 多头注意力机制**](chapters/04_multi_head_attention.md) | 三个臭皮匠诸葛亮、低维空间切片魔术 | 空间投影、张量维度切分与拼接 | [`code/04_multi_head_attention.py`](code/04_multi_head_attention.py) |
| [**05. 位置编码与座位号**](chapters/05_positional_encoding.md) | 小猫吃小鱼顺序危机、石英手表三针定乾坤 | 正弦波 $\sin$、余弦波 $\cos$、和角公式 | [`code/05_positional_encoding.py`](code/05_positional_encoding.py) |
| [**06. 积木组件：残差、LN 与 FFN**](chapters/06_residual_layernorm_ffn.md) | 考试保底底分、全科标准化、每个词闭门深思 | 导数恒等 $+1$ 路径、Z-Score 均值与方差 | [`code/06_transformer_block.py`](code/06_transformer_block.py) |
| [**07. 编码器与解码器**](chapters/07_encoder_decoder.md) | 阅卷老师 vs 考场作家、因果掩码闭卷考试、贪吃蛇生成 | 下三角掩码矩阵、自回归循环 | - |
| [**08. 实战：手搭 Baby-GPT**](chapters/08_baby_transformer.md) | 从第一行代码训练一个能背三首短诗的微型模型 | 交叉熵损失、温度采样、梯度下降 | [`code/07_baby_gpt.py`](code/07_baby_gpt.py) |
| [**09. 走向现代大模型 (LLMs)**](chapters/09_modern_llms.md) | RoPE 旋转位置编码、RMSNorm、SwiGLU、DeepSeek MoE | 复数平面旋转、均方根、尺度定律与涌现 | - |
| [**10. 附录与黑话词典**](chapters/10_appendix.md) | 高中数学公式速查卡、深度学习黑话翻译大白话词典 | 全书数理汇总 | - |

---

## 💻 7 个精选保姆级代码实验

代码在 Python 3.12 与 PyTorch 2.14.1 环境中验证过，并带有中文注释与可视化输出。第 1、2 个实验只依赖 Python 标准库；第 3～7 个实验需要 PyTorch：

1. **[`code/01_vector_similarity.py`](code/01_vector_similarity.py)**：纯 Python 手写向量内积与余弦相似度，验证水果空间距离与国王-女王向量平移。
2. **[`code/02_manual_attention.py`](code/02_manual_attention.py)**：纯 Python 零依赖手算 3 个字（“我”、“爱”、“学”）在注意力机制内部经历的每一步数字变化。
3. **[`code/03_scaled_dot_product.py`](code/03_scaled_dot_product.py)**：用数值实验揭开“为什么必须除以 $\sqrt{d_k}$”的方差危机，以及因果掩码如何用 $-1e9$ 阻隔未来。
4. **[`code/04_multi_head_attention.py`](code/04_multi_head_attention.py)**：多头注意力的 PyTorch 完整实现与张量维度变换追踪。
5. **[`code/05_positional_encoding.py`](code/05_positional_encoding.py)**：正弦余弦位置编码矩阵计算与控制台 ASCII 字符波浪热力图打印。
6. **[`code/06_transformer_block.py`](code/06_transformer_block.py)**：Pre-LN Transformer 积木块拼装（LayerNorm + MHA + 残差 + FFN）。
7. **[`code/07_baby_gpt.py`](code/07_baby_gpt.py)**：三首诗分开训练的微型自回归模型。种子 42 时，损失从约 4.13 降到 0.0009，能从真实开头背出三首诗；提示“明月”则不会接回《静夜思》。

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

如果只想先体验高中数学版的两个实验，可以直接运行 `python3 code/01_vector_similarity.py` 和
`python3 code/02_manual_attention.py`，它们不需要安装 PyTorch。重新编译 EPUB/HTML 还需要系统中的
`pandoc` 和 Node.js 22 或更高版本（macOS 可运行 `brew install pandoc node`），并先运行 `npm ci` 安装锁定版本的图表构建工具。阅读成品不需要这些依赖。

---

## 📱 交付成果物与随时阅读方式

项目已全自动编译并生成了三大格式的成品交付物，位于 `dist/` 目录下：

```
dist/
├── transformer_for_highschool.html  # 🌟 现代化响应式 Web 阅读器（图表可离线阅读，公式排版需网络）
├── index.html                       # 🌟 对应 Web 站点入口文件
├── transformer_for_highschool.epub  # 📚 标准 EPUB 电子书 (支持 iPad/Apple Books/Kindle)
├── transformer_for_highschool.md    # 📝 完整合并版 Master Markdown 文档
├── code/                            # 🧪 网页中“打开配套源码”所需的 Python 示例
└── assets/                          # 🎨 高清封面图与矢量架构图 (SVG)
```

### 1. 网页端观看（最佳排版体验）：
- **直接双击打开** [`dist/index.html`](dist/index.html) 或 [`dist/transformer_for_highschool.html`](dist/transformer_for_highschool.html) 即可在任何浏览器（Safari, Chrome, Edge）中畅读！
- 特性：
  - 支持 **深色 (Dark) / 浅色 (Light)** 主题一键切换；
  - 集成 **MathJax 3** 高清数学公式渲染；
  - **预渲染流程图**，断网仍显示，点击可查看原尺寸；
  - 左侧常驻章节目录跳转，顶部实时阅读进度条；
  - 代码块高亮显示与一键下载功能。

  公式排版和代码高亮通过 CDN 加载，需要网络；正文、全部图表和代码可离线阅读。

### 2. 电子书阅读器观看（EPUB）：
- 文件路径：[`dist/transformer_for_highschool.epub`](dist/transformer_for_highschool.epub)
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
