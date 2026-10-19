# P18–P20：MoE、预训练数据与框架验证记录

日期：2026-10-19。基准 main `66dbeb94dd9935046af0e859a223775b7adf4e0e`，分支 `dot/batch7-p18-p20`。仅原已审核大纲；不引入 #28/#29，不推 main、不自动合并。

## 最终本地验证

| 环境 / 路径 | 实际结果 | 不能推出的结论 |
| --- | --- | --- |
| 原基础 `.venv` | 318 项，310 通过、8 个可选 HF 测试明确跳过 | 不代表在基础环境运行过 HF |
| 隔离 HF + CairoSVG 环境 | 318/318 通过、0 跳过 | 不代表 GPU / 分布式验证 |
| 无 torch 的系统 Python | 318 项，114 通过、204 明确跳过 | 不把跳过算成数值通过 |
| Node | 3/3 通过 | 仅布局选择逻辑，不是真实浏览器布局 |
| 21 个编号脚本的默认 CPU CLI | 全部退出码 0 | code 21 默认只跑标准库账本 |
| code 21 `--hf` | 真实 CPU Trainer、独立 SGD、保存/重载均通过 | HF resume 未测试；P19 另测自己的恢复循环 |
| compileall / diff whitespace | 通过 | 不替代行为回归 |

本批新增 P18 23 项、P19 17 项、P20 18 项，以及 4 项跨章节文本/图/进度回归。所有计数均为最终实际发现的测试数，不沿用早期草稿计数。

- **三张中文 SVG 实际渲染并查看像素**：中文、表格、箭头、边界说明均可读，无缺字、重叠或裁切。P18 首轮发现共享输入箭头穿过均衡偏置框，已绕行修正并重渲染；不是只检查 SVG 文本。
- **静态构建与 check_book 通过**：23 个章节，12 张 Mermaid 与 20 张静态图，共 32 张插图；HTML/EPUB 资源、内部链接和目录锚点完整。EPUB XML 类文件与源码副本另行逐项验证，见下方产物记录。
- **正常 Chromium build 与 check:layout 已实际尝试，均受环境启动限制**：`FATAL:chrome/browser/process_singleton_posix.cc:297 ... socket() failed: Operation not permitted (1)`。未修改 sandbox 或安全设置。本批不声称浏览器布局通过，请审核方对新 head 正常构建并验证 23×3=69 个章节/视口组合。
- 静态路径以以上 main 为基准，仅复用源码完全相同的 12 张 Mermaid 和 17 张既有静态图；3 张新 SVG 用 CairoSVG 2.8.2 渲染。静态构建不代替浏览器检查。

## 实际数值与独立参考

| 实验 | 实际结果 / 计数 |
| --- | --- |
| P18 路由手算 | top-2 权重 4/7、3/7；请求[2,2,1,1]，容量 1 接收[1,1,1,1]；辅助项 16/15 |
| P18 参数预算 | D=4、E=4、k=2、专家宽 2、共享 1：模块总参数 136、每 token 激活预算 88；不是整模型预算 |
| P18 S5 | seed 180 两层负载[4,6,11,3]/[8,6,5,5]；full/cache 误差 1.79e-7；紧凑 KV 1536 字节 |
| P19 数据 | 12 条原始生成记录→10 条唯一；train/validation/test 为 6/2/2 篇，有效 targets 21/7/7 |
| P19 手算 | 8 输入位置、5 有效 targets；loss 总和 9 ln2，均值 1.2476649250；跨文档上下文与标签边界分别核对 |
| P19 训练 | 2040 参数，12 步、84 有效 targets；validation 3.7020121983→3.4868052346；最终 test 未评分 |
| P19 恢复 | step 3 和经过 shuffle 的 step 5 中断，继续后完整状态与逐步 loss 精确一致；限定同环境 CPU 完整更新边界 |
| P20 实际 HF | 随机本地 GPT-2 共 2704 参数，logits [2,6,12]，9 个监督 targets；3 步 SGD loss 2.5129108429→2.3418998718 |
| P20 更新参考 | 独立手写 SGD 参数最大差 3.7253e-9，实际参数改变量 0.0195597243；固定 eager 保存/重载 logits 差 0 |
| P20 内存账本 | P=10、D=4，持久状态最满 rank：Stage 0/1/2/3 为 160/76/62/48 bytes；排除激活/通信/临时聚合/对齐填充 |

浮点末位随平台可变，测试依据明确容差。以上均为教学实验，不是生产模型质量或 GPU 性能成绩。

## 行为变异与审校发现

- P18 六个实际可执行变异：均匀权重、调用所有专家、将均衡偏置混入组合权重、共享输出缩半、丢弃后重归一化、反向派发接受顺序。均与独立标量/字面输出参考对照。另有梯度、空专家、top-1/top-E、请求分叉、缓存签名和未来隔离检查。
- P18 审校发现无偏置 top-k 先计算 Softmax 可能因下溢产生假平局。已改成无偏置/零偏置按 logits 排序，并加入极端 logits 回归；有偏置路径仍遵循明确的概率空间教学契约。
- P19 对错误全局标签 shift、跨文档 causal-only mask、按行均值而非 token 加权做反例；另实际运行四个状态遗漏版本（AdamW、dropout RNG、shuffle RNG、cursor），要求 loss 轨迹与最终权重偏离正确恢复。shuffle 遗漏在 step 5 触发，不能把第一次 shuffle 前等价的随机状态称为有效变异。全忽略批次还检查跳过更新与错误更新的区别。
- P20 八个实际源代码/执行变异：四个分片账本错误、shift 方向、未屏蔽首列 BOS、取消 SGD 更新、损坏重载权重。独立标量 CE、梯度位置与参数更新参考共同验证，不只检查 loss 下降。
- 实际 HF 审校发现版本 4.51.3 Trainer 的外部有效标签计数与内部 shift 契约有细节差异：首列 BOS 不忽略时分母 11 而不是 9。当前数据明确忽略首列，独立 Trainer loss 与 3 步 SGD 对照通过。重载曾默认改变 attention 后端，产生约 5.96e-8 差异；已显式保留 eager，未靠放宽容差掩盖。
- 独立审校分别重跑 P18/P19/P20、原始来源、手算、缓存、packing 隔离与恢复反例；初版问题均已修复。本地独立审校不冒充 Normal Work 对新远端 head 的正式验收。

不将历史审稿方有算术矛盾的变异总数沿用到本批；各项类型分别报告，也不把普通测试数与变异数相加。

## 版本、来源与未运行范围

基础环境为 Python 3.12.14、PyTorch 2.14.1+cpu、Tokenizers 0.21.4、Pandoc 3.9、CairoSVG 2.8.2。可选 HF 在隔离环境固定 Transformers 4.51.3、Accelerate 1.6.0、Tokenizers 0.21.4、huggingface-hub 0.30.2、safetensors 0.5.3；其余基础包复用原环境，包括 CairoSVG 2.8.2。全套零跳过依赖 requirements-build-static.txt 的 CairoSVG；未安装时两项静态渲染测试会明确跳过。没有修改原基础环境，没有下载模型权重。README 的全新隔离安装命令是复跑配方，不声称本机做过第二次全新安装。

- P18：[DeepSeekMoE v1 §3](https://arxiv.org/html/2401.06066v1#S3)、[DeepSeek-V3 v2 §2.1.2](https://arxiv.org/html/2412.19437v2#S2.SS1.SSS2)。教学 Softmax/selected 归一化/batch 辅助项/独立容量示例不冒充 V3 sigmoid 路由与生产 dropless 方案。
- P19：[Chinchilla v1](https://arxiv.org/abs/2203.15556v1)、固定 PyTorch 2.14 与 Transformers 4.51.3 文档。生成语料的去重切分策略不适用于随意重划已冻结外部评估集；相关边界见正文。
- P20：[Transformers 4.51.3](https://huggingface.co/docs/transformers/v4.51.3/en/trainer)、[TRL 0.17.0](https://huggingface.co/docs/trl/v0.17.0/en/index)、[ZeRO v3](https://arxiv.org/abs/1910.02054v3)、[Megatron-Core core_v0.12.0](https://github.com/NVIDIA/Megatron-LM/tree/core_v0.12.0)及正文固定 DeepSpeed 源码。没有安装/运行 TRL、DeepSpeed 或 Megatron，也没有 GPU、多卡、通信吞吐或混合精度实验。

## 复跑命令

```bash
.venv/bin/python -m unittest discover -s tests -v
python3 -m unittest discover -s tests -v
node --test tests/
for f in code/[0-9][0-9]_*.py; do OMP_NUM_THREADS=1 .venv/bin/python "$f"; done
# 可选 HF 环境按 README 创建；路径以你的实际环境为准
.venv-frameworks/bin/python code/21_training_frameworks.py --hf
.venv-frameworks/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q code tests scripts build.py
npm run build
npm run check:layout
.venv/bin/python scripts/check_book.py
git diff --check
```

本机受限静态构建使用：`FONTCONFIG_FILE=/tmp/p06-fonts.conf .venv/bin/python scripts/build_without_browser.py --baseline 66dbeb94dd9935046af0e859a223775b7adf4e0e`。字体配置是本机 Noto Sans CJK SC 设置，不是浏览器安全选项。

### 产物逐项记录

30 份 EPUB XML 类文件（`.xhtml/.opf/.ncx/.xml`）均解析通过；22 个 Python 源文件（21 个编号脚本及 attention_core.py）与 dist/code 逐字节相同；20 个原始 SVG 与 dist/assets/diagrams 逐字节相同。新章节显式锚点和代码链接通过 check_book；EPUB 中的源码链接由构建器转换为仓库链接，不把未打包的测试/依赖文件当成 EPUB 本地资源。

## PR #8 首轮复审后的补强

[Normal Work review 5570682715](https://github.com/cluster1900/zero2llm/pull/8#pullrequestreview-5570682715) 对旧 head `0eacd0f` 独立验证 315 个 HF 测试、21 个脚本、正常构建和 69 layouts 通过，但 61 个审稿方变异中两项非等价变异存活。本轮不修改 code/ 数值实现：

1. 新增容量 None/1/0 的 requested/accepted 手算向量及辅助项字面值 16/15。实际将 route 的 requested 改成 accepted，普通回归在容量 1 和 0 下均断言失败，无运行错误。
2. 新增 decomposed accent→é、CJK canonical 映射、NFC 不等于 NFKC 的边界，以及 NFD/NFC 与 CJK 两组去重和保留文本检查。实际移除 NFC 的错误函数须被普通回归拒绝。
3. 修复 P20 行内数学定界符，增加源码回归并检查构建后 HTML/EPUB 保留正确数学语义；统一 collator 4.51.3 / DDP 2.14 来源。README 先装官方 CPU torch，再安装其余固定依赖和静态渲染依赖。

本轮新 head 的布局不继承旧 head 的 69 layouts；仍需审核方重跑。没有声称运行了审稿方全部 61 个变异，只报告本地内置检查和两个实际错误函数的独立复验。

本次返修重新尝试正常构建与布局，仍在 Chromium 启动阶段遇到上述 socket 权限错误；随后以 `0eacd0f92cb7435b24e9e3f92037b741c1020ac0` 静态重建，复用源码不变的 12 张 Mermaid 和 20 张静态图。没有图像源码变化。重新验证 30 份 XML 与全部源码副本；HTML 行内数学 span 与 EPUB MathML/TeX annotation 均保留 `\mathcal V`，不再是普通文本 `(V)`。

## 合并后的正式复审

[Normal Work review 5571650468](https://github.com/cluster1900/zero2llm/pull/8#pullrequestreview-5571650468) 对 `559dc57` 独立验证 318 个 HF 测试、21 个脚本、正常构建及提交/重建产物各 69 layouts 通过；61/61 个审核侧变异被拒绝（其中 A09 为真实梯度缺失的 backward 错误）。MathJax 花体 V 与 EPUB MathML 均实际核对。PR #8 合并至 `9c4c9e34de9d6d394e45781b4ab115ee9e196f8d`。该浏览器结果只覆盖本批，不覆盖以后新章节。
