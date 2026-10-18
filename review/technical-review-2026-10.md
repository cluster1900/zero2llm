# dot：zero2llm 全书技术审校（bot-hub #11）

提交身份：dot（OpenAI 助手，不是用户本人）。交付供 Normal Work 审核，所有改动只能通过分支与 PR 合并；没有直接写 main，也没有批准自动合并。

## 基线与范围

- 仓库：https://github.com/cluster1900/zero2llm
- 基线提交：`72c39df44d05032edca7f0509bab6b765c35f48b`（2026-10-08）。
- 本次核查日期：2026-10-18。
- `bot-hub/materials/11/chapters` 的树 SHA 为 `4c60844360870560164b567bbc4d28c4bc76eb2d`，`code` 为 `7156f5aaedf7193a0cdcf7414a47874ddb33a18f`，均与基线相同。
- 已逐章阅读 00–10 共 11 章，审查并运行 code/ 下全部 7 个脚本。未发现仓库 AGENTS.md。
- 保持原章节顺序、教学案例、主要公式和初学者定位；第 09 章只加简短路线介绍，不追逐型号排行榜。

## 优先修复项

1. Q=K 只保证原始点积分数对称，逐行 Softmax 不保证权重对称；列向量转成行向量还须转置权重。
2. 点积不等于余弦，反向不等于反义词；未知词不能凭空用嵌入插值处理；零向量余弦未定义。
3. 正弦位置编码的奇数维最后一列漏算；无位置注意力是置换等变，不是不变，且需排除因果掩码。
4. RMSNorm 补 epsilon，去除统一 10–50% 性能承诺；SwiGLU 明确 SiLU、门控范围和三矩阵隐藏宽度。
5. KV Cache 必须为新 token 计算并追加 K/V，而不仅是 Q；GQA、MLA 和混合线性注意力不能混同。
6. DeepSeek-V3 只有 MoE 层是 256 路由专家选 8，前 3 层仍稠密；参数激活比例不等于总成本比例。
7. ch08 的代码补上 poems 定义，float16 掩码用 -inf，生成张量跟随模型设备，空提示给明确错误。

## 逐项问题与修正

下面位置为修订稿定位行（同一位置多项修订可能使后续行移动）；精确基线删改位置以附带 unified diff 的 @@ 行号为准。每项说明对应修正后的正文或代码，不是仅提出建议。

| 文件 | 修订定位行 | 问题与修正 |
| --- | ---: | --- |
| `chapters/00_preface.md` | 50 | 类比不能写成严格语义等式。详见该处补丁。 |
| `chapters/01_word_embedding.md` | 93 | 点积及反义词关系错误。详见该处补丁。 |
| `chapters/01_word_embedding.md` | 94 | 忽略查表实现。详见该处补丁。 |
| `chapters/01_word_embedding.md` | 95 | OOV 不能自动插值。详见该处补丁。 |
| `chapters/01_word_embedding.md` | 128 | 嵌入维度不可保证可解释。详见该处补丁。 |
| `chapters/01_word_embedding.md` | 151 | 余弦与语义不能等同。详见该处补丁。 |
| `chapters/01_word_embedding.md` | 152 | 正交不等于语义无关。详见该处补丁。 |
| `chapters/01_word_embedding.md` | 153 | 反向不等于反义。详见该处补丁。 |
| `chapters/01_word_embedding.md` | 262 | 点积无界且不是余弦。详见该处补丁。 |
| `chapters/01_word_embedding.md` | 263 | 过度泛化向量类比。详见该处补丁。 |
| `chapters/02_why_attention.md` | 21 | 静态不等于训练中不变。详见该处补丁。 |
| `chapters/02_why_attention.md` | 59 | RNN 遗忘非必然。详见该处补丁。 |
| `chapters/02_why_attention.md` | 63 | GPU 核心数量不能泛化。详见该处补丁。 |
| `chapters/02_why_attention.md` | 114 | 未来线索不适用于因果位置。详见该处补丁。 |
| `chapters/02_why_attention.md` | 155 | 小结与正文并行性冲突。详见该处补丁。 |
| `chapters/03_qkv_self_attention.md` | 24 | 原始打分不是百分比。详见该处补丁。 |
| `chapters/03_qkv_self_attention.md` | 25 | 原始打分不是百分比。详见该处补丁。 |
| `chapters/03_qkv_self_attention.md` | 44 | 对称分数不蕴含对称权重。详见该处补丁。 |
| `chapters/03_qkv_self_attention.md` | 46 | 承接对称性修正。详见该处补丁。 |
| `chapters/03_qkv_self_attention.md` | 52 | 行列向量需要权重转置。详见该处补丁。 |
| `chapters/03_qkv_self_attention.md` | 86 | 四舍五入和不是精确100。详见该处补丁。 |
| `chapters/03_qkv_self_attention.md` | 119 | 不相关不保证乘积方差推导。详见该处补丁。 |
| `chapters/03_qkv_self_attention.md` | 95 | 示例不是维度阈值定理。详见该处补丁。 |
| `chapters/04_multi_head_attention.md` | 46 | 多头计算量与显存区别。详见该处补丁。 |
| `chapters/04_multi_head_attention.md` | 64 | 多头分工不保证。详见该处补丁。 |
| `chapters/04_multi_head_attention.md` | 200 | 单头能力夸大限制。详见该处补丁。 |
| `chapters/04_multi_head_attention.md` | 201 | 复杂度非实测成本。详见该处补丁。 |
| `chapters/04_multi_head_attention.md` | 71 | 补充多头中间张量内存。详见该处补丁。 |
| `chapters/05_positional_encoding.md` | 25 | 置换等变性前提。详见该处补丁。 |
| `chapters/05_positional_encoding.md` | 177 | 随机正交不能证明嵌入与位置无干扰。详见该处补丁。 |
| `chapters/05_positional_encoding.md` | 174 | 依赖说明。详见该处补丁。 |
| `chapters/05_positional_encoding.md` | 211 | 等变不是不变。详见该处补丁。 |
| `chapters/05_positional_encoding.md` | 181 | 奇数维最后正弦列被错误置零。详见该处补丁。 |
| `code/05_positional_encoding.py` | 34 | 奇数维最后正弦列被错误置零。详见该处补丁。 |
| `chapters/06_residual_layernorm_ffn.md` | 42 | 残差非保底保证。详见该处补丁。 |
| `chapters/06_residual_layernorm_ffn.md` | 73 | LayerNorm 仿射参数逐特征。详见该处补丁。 |
| `chapters/06_residual_layernorm_ffn.md` | 92 | 归一化轴避免跨序列歧义。详见该处补丁。 |
| `chapters/06_residual_layernorm_ffn.md` | 95 | RMSNorm 速度无统一百分比。详见该处补丁。 |
| `chapters/06_residual_layernorm_ffn.md` | 106 | Pre-LN 精确位置。详见该处补丁。 |
| `chapters/06_residual_layernorm_ffn.md` | 141 | 加减乘除并非都线性。详见该处补丁。 |
| `chapters/06_residual_layernorm_ffn.md` | 151 | GPT4 细节未公开不能归类。详见该处补丁。 |
| `chapters/07_encoder_decoder.md` | 23 | 表格混淆两类解码器。详见该处补丁。 |
| `chapters/07_encoder_decoder.md` | 22 | 编码器仍可有padding mask。详见该处补丁。 |
| `chapters/07_encoder_decoder.md` | 91 | 有限负数不等于负无穷。详见该处补丁。 |
| `chapters/07_encoder_decoder.md` | 179 | 概率示例标记与EOS边界。详见该处补丁。 |
| `chapters/08_baby_transformer.md` | 12 | 实验输出非跨环境保证。详见该处补丁。 |
| `chapters/08_baby_transformer.md` | 28 | 上下文容量不等于学会长文本。详见该处补丁。 |
| `chapters/08_baby_transformer.md` | 77 | 章节代码漏定义poems。详见该处补丁。 |
| `chapters/08_baby_transformer.md` | 237 | 随机种子复现边界。详见该处补丁。 |
| `chapters/08_baby_transformer.md` | 272 | 闭源模型参数无据。详见该处补丁。 |
| `chapters/10_appendix.md` | 20 | 附录LayerNorm与正文冲突。详见该处补丁。 |
| `chapters/10_appendix.md` | 35 | token单位非字数。详见该处补丁。 |
| `chapters/10_appendix.md` | 36 | 训练Loss非泛化指标。详见该处补丁。 |
| `code/01_vector_similarity.py` | 41 | 代码注释反义误解。详见该处补丁。 |
| `code/01_vector_similarity.py` | 48 | 零向量余弦未定义。详见该处补丁。 |
| `chapters/01_word_embedding.md` | 236 | 零向量余弦未定义。详见该处补丁。 |
| `code/02_manual_attention.py` | 68 | 权重转置注释。详见该处补丁。 |
| `code/02_manual_attention.py` | 122 | 浮点和。详见该处补丁。 |
| `code/03_scaled_dot_product.py` | 133 | 用断言代替无条件成功输出。详见该处补丁。 |
| `code/06_transformer_block.py` | 59 | LayerNorm ReLU 非LLaMA同款。详见该处补丁。 |
| `code/06_transformer_block.py` | 74 | 不同API布尔掩码语义。详见该处补丁。 |
| `code/07_baby_gpt.py` | 152 | 生成设备不匹配及空提示。详见该处补丁。 |
| `code/07_baby_gpt.py` | 240 | 真实聊天模型非仅规模差异。详见该处补丁。 |
| `chapters/09_modern_llms.md` | 1 | 标题避免只锚定旧闭源模型。详见该处补丁。 |
| `chapters/09_modern_llms.md` | 3 | 更新范围与来源边界。详见该处补丁。 |
| `chapters/09_modern_llms.md` | 22 | 淘汰论表头与正文冲突。详见该处补丁。 |
| `chapters/09_modern_llms.md` | 32 | 五构件不是普遍配置。详见该处补丁。 |
| `chapters/09_modern_llms.md` | 59 | 位置可计算不保证长上下文。详见该处补丁。 |
| `chapters/09_modern_llms.md` | 73 | RoPE内容及频率说明。详见该处补丁。 |
| `chapters/09_modern_llms.md` | 81 | RMSNorm适用边界。详见该处补丁。 |
| `chapters/09_modern_llms.md` | 84 | RMSNorm缺epsilon。详见该处补丁。 |
| `chapters/09_modern_llms.md` | 86 | RMSNorm性能无统一保证。详见该处补丁。 |
| `chapters/09_modern_llms.md` | 92 | 激活历史非统一替换。详见该处补丁。 |
| `chapters/09_modern_llms.md` | 94 | 标明含输出投影的完整FFN。详见该处补丁。 |
| `chapters/09_modern_llms.md` | 95 | SwiGLU门值和参数量。详见该处补丁。 |
| `chapters/09_modern_llms.md` | 99 | 中性教学标题。详见该处补丁。 |
| `chapters/09_modern_llms.md` | 123 | DeepSeek并非每层MoE。详见该处补丁。 |
| `chapters/09_modern_llms.md` | 125 | MoE计算非存储同比例下降。详见该处补丁。 |
| `chapters/09_modern_llms.md` | 136 | KV Cache漏新K/V。详见该处补丁。 |
| `chapters/09_modern_llms.md` | 136 | 复杂度前提与教学代码差异。详见该处补丁。 |
| `chapters/09_modern_llms.md` | 140 | 补充2026视角GQA MLA混合路线。详见该处补丁。 |
| `chapters/09_modern_llms.md` | 152 | 版本固定与context单位。详见该处补丁。 |
| `chapters/09_modern_llms.md` | 159 | 归一化类型与位置不可混列。详见该处补丁。 |
| `chapters/09_modern_llms.md` | 160 | 明确DeepSeek位置方案。详见该处补丁。 |
| `chapters/09_modern_llms.md` | 164 | 原版无512固定上限；K表示避免伪精度。详见该处补丁。 |
| `chapters/09_modern_llms.md` | 187 | 能力不只由预训练规模解释。详见该处补丁。 |
| `chapters/09_modern_llms.md` | 194 | 非所有模型五大必选。详见该处补丁。 |
| `chapters/09_modern_llms.md` | 195 | AGI结论无据。详见该处补丁。 |
| `chapters/01_word_embedding.md` | 表 1-2、1.5 | 梨向量模长 12.51 改为 12.50；0.9999 对应 99.99%；展示表是重排过的输出而非脚本原打印顺序。 |
| `code/03_scaled_dot_product.py`、`code/04_multi_head_attention.py`、`code/07_baby_gpt.py` 与第 08 章 | masked_fill | 有限的 -1e9 不是负无穷，转换 float16 会溢出；统一使用 float("-inf")，保持默认实验结果。因果掩码每行至少保留一个可见位置。 |

## 过时或证据不足的表述清单

- 第 06/09 章把 RMSNorm 固定提速 10–50% 写成通用事实：删除统一性能范围。
- 第 06 章把 GPT-4 的归一化结构当公开事实、第 08 章给所有闭源系统统一归到千亿规模：改为可核对的公开实例，不猜未披露结构或参数。
- 第 09 章表头“淘汰/胜出”、五大构件作为所有现代模型固定配方：改为用途与设计取舍。
- 第 09 章止于 2024 模型、GQA/MLA 只列缩写：保留历史版本表，新增共享/压缩/混合线性三条路线，用 Qwen3-Next 官方模型卡作已公开的混合结构实例。
- 第 09 章原版 Transformer 512 固定上限、128K 强行展开为 128000、字数等于 token 数：去除错误精确化与单位混用。
- “涌现证明通用智能”“仅规模变大就能解释聊天能力”：改为经验现象与数据、后训练、推理计算共同影响。

## 覆盖与测试

- 00：定位与词向量类比；01：向量公式、全部水果表数值、语义边界；02：静态嵌入、RNN、因果上下文；03：QKV 约定、方差前提、Softmax、手算输出。
- 04：头拆分、形状、计算/内存；05：位置公式与奇数维、相加解释、等变性；06：残差、归一化轴与仿射参数、FFN、Pre-LN；07：双塔/Decoder-Only、掩码、生成示意；08：词表、编号、参数量、完整训练与输出；09：公开架构及现代路线；10：公式/术语与正文一致性。
- 7 个脚本全部退出码 0。Python 3.12，PyTorch 2.14.1+cpu，CPU，OMP_NUM_THREADS=1、MKL_NUM_THREADS=1。
- Baby-GPT 实跑 80 步：Loss 4.1271 → 0.0196 → 0.0021 → 0.0011 → 0.0009；三个开头续写与书中示例一致。固定种子不承诺跨平台逐字一致。
- 新增 `tests/test_tutorial.py`：7 组测试通过，覆盖公式/SDPA 对照、因果隔离、float16 掩码、奇数与偶数位置编码、LayerNorm 官方对照、词表 57/参数 67872、空提示处理。GPU 不可用，CUDA 分支未实测。
- 未声称运行了完整 requirements.txt 的所有精确依赖组合；安装的是 CPU torch 2.14.1 与其解析依赖、NumPy。
- 完整 EPUB/Web 构建尚未通过：预装 Pandoc 3.1.11.1 不支持 alerts，已用隔离环境的较新 Pandoc 解决解析；npm ci 在可写缓存下通过。随后 Chromium 在此执行环境无法创建 singleton socket，普通构建失败，申请扩大执行权限也因平台封装失败而未运行。没有关闭浏览器安全沙箱。
- 因此提交的是**源码审校 PR**，没有提交不完整的 dist/。合并前须由可正常运行 Chromium 的环境执行 `python build.py`、`python scripts/check_book.py` 和 `npm run check:layout`，并在同一 PR 中加入一致的 dist/ 成品。当前 pages 工作流部署的是已提交 dist/，不会自动重建。

复现：

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -v
for f in code/*.py; do OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python "$f"; done
```

## 核查用一手来源

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：缩放点积、多头、位置编码与原版结构。
- [RoFormer](https://arxiv.org/abs/2104.09864)、[RMSNorm](https://arxiv.org/abs/1910.07467)、[GLU Variants](https://arxiv.org/abs/2002.05202)。
- [GQA](https://arxiv.org/abs/2305.13245)、[DeepSeek-V3](https://arxiv.org/html/2412.19437v2)、[Llama 3](https://arxiv.org/abs/2407.21783)。
- [Qwen3-Next 官方卡](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)：3 层 Gated DeltaNet + 1 层门控注意力。
- [GPT-4 技术报告](https://arxiv.org/abs/2303.08774)：不要推断未公开的架构细节。
- [Compute-optimal training](https://arxiv.org/abs/2203.15556)、[Emergent abilities metric study](https://arxiv.org/abs/2304.15004)：规模与评测边界。

## 交付文件

- `zero2llm-review.patch`：以以上提交为基线的 unified diff；先用 `git apply --check` 检查。
- `zero2llm-reviewed-full.md`：11 章修订后完整 Markdown，供通读，不替代仓库拆分文件。
- PR 中 `tests/test_tutorial.py` 与本报告可用于 Normal Work 复核；无自动合并。
