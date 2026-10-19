# P18–P20：MoE、预训练数据与框架验证记录

日期：2026-10-19。基准 main `66dbeb94dd9935046af0e859a223775b7adf4e0e`，分支 `dot/batch7-p18-p20`。仅原已审核大纲；不引入 #28/#29，不推 main、不自动合并。

## 最终本地验证

| 环境 / 路径 | 实际结果 | 不能推出的结论 |
| --- | --- | --- |
| 原基础 `.venv` | 315 项，307 通过、8 个可选 HF 测试明确跳过 | 不代表在基础环境运行过 HF |
| 隔离 HF 环境 | 315/315 通过、0 跳过 | 不代表 GPU / 分布式验证 |
| 无 torch 的系统 Python | 315 项，112 通过、203 明确跳过 | 不把跳过算成数值通过 |
| Node | 3/3 通过 | 仅布局选择逻辑，不是真实浏览器布局 |
| 21 个编号脚本的默认 CPU CLI | 全部退出码 0 | code21 默认只跑标准库账本 |
| code21 `--hf` | 真实 CPU Trainer、独立 SGD、保存/重载均通过 | HF resume 未测试；P19 另测自己的恢复循环 |
| compileall / diff whitespace | 通过 | 不替代行为回归 |

本批新增 P18 22 项、P19 16 项、P20 18 项，以及3项跨章节文本/图/进度回归。所有计数均为最终实际发现的测试数，不沿用早期草稿计数。

- **三张中文 SVG 实际渲染并查看像素**：中文、表格、箭头、边界说明均可读，无缺字、重叠或裁切。P18 首轮发现共享输入箭头穿过均衡偏置框，已绕行修正并重渲染；不是只检查 SVG 文本。
- **静态构建与 check_book 通过**：23 个章节，12 张 Mermaid 与20张静态图，共32张插图；HTML/EPUB 资源、内部链接和目录锚点完整。EPUB XML 类文件与源码副本另行逐项验证，见下方产物记录。
- **正常 Chromium build 与 check:layout 已实际尝试，均受环境启动限制**：`FATAL:chrome/browser/process_singleton_posix.cc:297 ... socket() failed: Operation not permitted (1)`。未修改 sandbox 或安全设置。本批不声称浏览器布局通过，请审核方对新 head 正常构建并验证23×3=69个章节/视口组合。
- 静态路径以以上 main 为基准，仅复用源码完全相同的12张 Mermaid 和17张既有静态图；3张新 SVG 用 CairoSVG 2.8.2 渲染。静态构建不代替浏览器检查。

## 实际数值与独立参考

| 实验 | 实际结果 / 计数 |
| --- | --- |
| P18 路由手算 | top-2 权重4/7、3/7；请求[2,2,1,1]，容量1接收[1,1,1,1]；辅助项16/15 |
| P18 参数预算 | D=4、E=4、k=2、专家宽2、共享1：模块总参数136、每 token 激活预算88；不是整模型预算 |
| P18 S5 | seed180两层负载[4,6,11,3]/[8,6,5,5]；full/cache误差1.79e-7；紧凑KV1536字节 |
| P19 数据 | 12条原始生成记录→10条唯一；train/validation/test为6/2/2篇，有效targets21/7/7 |
| P19 手算 | 8输入位置、5有效targets；loss总和9 ln2，均值1.2476649250；跨文档上下文与标签边界分别核对 |
| P19 训练 | 2040参数，12步、84有效targets；validation3.7020121983→3.4868052346；最终test未评分 |
| P19 恢复 | step3和经过shuffle的step5中断，继续后完整状态与逐步loss精确一致；限定同环境CPU完整更新边界 |
| P20 实际 HF | 随机本地GPT-2共2704参数，logits[2,6,12]，9个监督targets；3步SGD loss2.5129108429→2.3418998718 |
| P20 更新参考 | 独立手写SGD参数最大差3.7253e-9，实际参数改变量0.0195597243；固定eager保存/重载logits差0 |
| P20 内存账本 | P=10、D=4，持久状态最满rank：Stage0/1/2/3为160/76/62/48 bytes；排除激活/通信/临时聚合/对齐填充 |

浮点末位随平台可变，测试依据明确容差。以上均为教学实验，不是生产模型质量或 GPU 性能成绩。

## 行为变异与审校发现

- P18 六个实际可执行变异：均匀权重、调用所有专家、将均衡偏置混入组合权重、共享输出缩半、丢弃后重归一化、反向派发接受顺序。均与独立标量/字面输出参考对照。另有梯度、空专家、top-1/top-E、请求分叉、缓存签名和未来隔离检查。
- P18 审校发现无偏置 top-k 先计算 Softmax 可能因下溢产生假平局。已改成无偏置/零偏置按 logits 排序，并加入极端 logits 回归；有偏置路径仍遵循明确的概率空间教学契约。
- P19 对错误全局标签 shift、跨文档 causal-only mask、按行均值而非 token 加权做反例；另实际运行四个状态遗漏版本（AdamW、dropout RNG、shuffle RNG、cursor），要求loss轨迹与最终权重偏离正确恢复。shuffle遗漏在step5触发，不能把第一次shuffle前等价的随机状态称为有效变异。全忽略批次还检查跳过更新与错误更新的区别。
- P20 八个实际源代码/执行变异：四个分片账本错误、shift方向、未屏蔽首列BOS、取消SGD更新、损坏重载权重。独立标量CE、梯度位置与参数更新参考共同验证，不只检查loss下降。
- 实际 HF 审校发现版本4.51.3 Trainer的外部有效标签计数与内部shift契约有细节差异：首列BOS不忽略时分母11而不是9。当前数据明确忽略首列，独立Trainer loss与3步SGD对照通过。重载曾默认改变attention后端，产生约5.96e-8差异；已显式保留eager，未靠放宽容差掩盖。
- 独立审校分别重跑 P18/P19/P20、原始来源、手算、缓存、packing隔离与恢复反例；初版问题均已修复。本地独立审校不冒充 Normal Work 对新远端 head 的正式验收。

不将历史审稿方有算术矛盾的变异总数沿用到本批；各项类型分别报告，也不把普通测试数与变异数相加。

## 版本、来源与未运行范围

基础环境为 Python3.12.14、PyTorch2.14.1+cpu、Tokenizers0.21.4、Pandoc3.9、CairoSVG2.8.2。可选 HF 在隔离环境固定 Transformers4.51.3、Accelerate1.6.0、Tokenizers0.21.4、huggingface-hub0.30.2、safetensors0.5.3；其余基础包复用原环境。没有修改原基础环境，没有下载模型权重。README的全新隔离安装命令是复跑配方，不声称本机做过第二次全新安装。

- P18：[DeepSeekMoE v1 §3](https://arxiv.org/html/2401.06066v1#S3)、[DeepSeek-V3 v2 §2.1.2](https://arxiv.org/html/2412.19437v2#S2.SS1.SSS2)。教学Softmax/selected归一化/batch辅助项/独立容量示例不冒充V3 sigmoid路由与生产dropless方案。
- P19：[Chinchilla v1](https://arxiv.org/abs/2203.15556v1)、固定PyTorch2.14与Transformers4.57.1文档。生成语料的去重切分策略不适用于随意重划已冻结外部评估集；相关边界见正文。
- P20：[Transformers4.51.3](https://huggingface.co/docs/transformers/v4.51.3/en/trainer)、[TRL0.17.0](https://huggingface.co/docs/trl/v0.17.0/en/index)、[ZeRO v3](https://arxiv.org/abs/1910.02054v3)、[Megatron-Core core_v0.12.0](https://github.com/NVIDIA/Megatron-LM/tree/core_v0.12.0)及正文固定DeepSpeed源码。没有安装/运行TRL、DeepSpeed或Megatron，也没有GPU、多卡、通信吞吐或混合精度实验。

## 复跑命令

```bash
.venv/bin/python -m unittest discover -s tests -v
python3 -m unittest discover -s tests -v
node --test tests/
for f in code/[0-9][0-9]_*.py; do OMP_NUM_THREADS=1 .venv/bin/python "$f"; done
# 可选HF环境按README创建；路径以你的实际环境为准
.venv-frameworks/bin/python code/21_training_frameworks.py --hf
.venv-frameworks/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q code tests scripts build.py
npm run build
npm run check:layout
.venv/bin/python scripts/check_book.py
git diff --check
```

本机受限静态构建使用：`FONTCONFIG_FILE=/tmp/p06-fonts.conf .venv/bin/python scripts/build_without_browser.py --baseline 66dbeb94dd9935046af0e859a223775b7adf4e0e`。字体配置是本机Noto Sans CJK SC设置，不是浏览器安全选项。

### 产物逐项记录

30份EPUB XML类文件（`.xhtml/.opf/.ncx/.xml`）均解析通过；22个Python源文件（21个编号脚本及attention_core.py）与dist/code逐字节相同；20个原始SVG与dist/assets/diagrams逐字节相同。新章节显式锚点和代码链接通过check_book；EPUB中的源码链接由构建器转换为仓库链接，不把未打包的测试/依赖文件当成EPUB本地资源。
