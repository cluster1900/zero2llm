# P03–P05 第二批实验与验证记录

日期：2026-10-19。基线为首批 PR #2 合并后的 main `85155216f747a60a3ad4cdef5745a9915d9536cc`，分支 `dot/issue19-batch2-p03-p05`。任务为已批准扩展大纲中的 P03–P05，跟踪 [bot-hub #19](https://github.com/cluster1900/bot-hub/issues/19)。本批没有实现 P06 及以后的新专题。

## 正文与实验

- 第 2 章：上下文表示、RNN 信息路径、训练并行边界；纸笔依赖图、两组玩具权重、加权和与“权重不等于因果解释”。明确区分读完整句的分析和因果模型可见前缀。
- 第 3 章：QKV、缩放条件、稳定 Softmax、padding/causal mask、全禁用行、训练与生成；纠正“Q=K 导致注意力权重必然对称”的旧说法，给数值反例。
- 第 4 章：完整 B/T/H/D/d_h 形状账本、reshape/transpose、拼接、单头对照；参数量、计算量、注意力矩阵存储量分别说明，限定标准 MHA、固定 D 等条件。
- 复用编号示例 02、03、04；新增共享 `code/attention_core.py`，不冒充第十个独立实验。
- 保留第一批正文/代码和其他旧章节；README、大纲进度及阅读产物随本批同步。

## 环境与复现

CPU；Python 3.12.14，PyTorch 2.14.1+cpu，NumPy 2.5.3；选修 Tokenizers 0.21.4、CairoSVG 2.8.2；Pandoc 3.9；Node 24.19.0。PyTorch 使用官方 CPU wheel。没有 GPU、付费 API 或预训练模型下载。测试固定 seed=19，演示脚本保持各自种子。未把小实验当性能基准。

```bash
python3 -m unittest discover -s tests -v
node --test tests/*.test.mjs
python3 code/02_manual_attention.py
python3 code/03_scaled_dot_product.py
python3 code/04_multi_head_attention.py
python3 scripts/check_book.py
```

| 检查 | 本批结果 |
| --- | --- |
| 完整依赖环境 unittest | 35 项通过：原有 20 项 + 本批 15 项 |
| 无 PyTorch/Tokenizers/Cairo 的标准库环境 | 35 项收集，21 项通过、14 项明确跳过；不是完整注意力验收 |
| Node 截图辅助逻辑 | 3 项通过；不代替浏览器布局 |
| 编号脚本 01–09 | CPU 全部退出码 0；包含旧 Baby-GPT 小型训练冒烟检查 |
| 静态构建与 check_book | HTML、EPUB、合并 Markdown 生成；资源、内部链接、目录锚点通过 |
| 正常 Chromium 构建 | 启动失败：容器 `socket() failed: Operation not permitted` |
| check:layout | 同样被浏览器启动限制阻止，第二批逐视口布局未验证 |
| git diff --check、Python 编译 | 通过 |

新测试包括：
1. 稳定 Softmax 与朴素小分数公式一致、大分数不溢出、平移不变、非法输入拒绝。
2. 改动未来 token 不影响过去输出；双向注意力负对照确实变化；每个前缀的最后输出与全序列因果计算一致。
3. padding 键与 query 排除；整个样本全 padding 时输出为零；全禁用行权重及梯度有限，空行权重和为零。
4. 多组 B/T/D/H、每步形状断言、4D² 参数量；单头直接公式、多头逐头独立参考、官方 SDPA 对照。
5. 不兼容尺寸、非布尔 mask、不兼容广播、NaN 分数明确报错。

`masked_softmax` 是教学契约：原始分数必须有限，True=可见，空行返回零。它不是对所有内核行为的承诺。多头最终输出为零的条件是该 query 在所有头都无可见键，且输出投影无 bias。没有声称所有 mask API 使用相同方向。

## 产物与浏览器验证边界

采用首批已合并的静态构建入口，没有变更浏览器安全设置：

```bash
python3 scripts/build_without_browser.py --baseline 85155216f747a60a3ad4cdef5745a9915d9536cc
```

需让 PATH 指向 Pandoc 3.9，并安装 requirements-build-static.txt。构建器仅复用源码逐字相同的基线图；改过 Mermaid 会拒绝，不会悄悄用旧图。此批保留已有图的定义，用正文补足其简化边界；不新增未经检查的图形。共享 Python 模块也同步到 dist/code，下载示例应保留同目录的辅助模块。

已实际尝试正常 build.py 和 check:layout，均被 Chromium 的 socket 限制阻止。请 Normal Work 在本批最终 head 用正常构建及 390/768/844px 布局检查复审。首批在 head 815ce64 的 [36 个章节/视口组合 PASS](https://github.com/cluster1900/zero2llm/pull/2#pullrequestreview-5565985839) 仅属于首批，不算本批的验证结果。

本批仅提交主题分支与 Draft PR；合并、部署、Release 不是本记录宣称已完成的事项。


## 2026-10-19 第3章复审返修

[Normal Work 对 ada40256 的复审](https://github.com/cluster1900/zero2llm/pull/3#pullrequestreview-5566181208) 独立跑过九个脚本、32 项 Python 与3项 Node测试、正常 Chromium 构建，并取得 36 个章节/视口组合 PASS；它还通过8种变异验证原回归有效。该结果属于旧 head，不能代替下述修改后的浏览器复验。

本轮恢复初学者可跟算的完整路径，而不只增加一句说明：
- §3.2.1 恢复“我/爱/学”的输入、三张投影矩阵、Q/K/V三行、缩放分数矩阵、权重百分比、V逐项加权及全部输出，并链接脚本02。解释列向量代码与行向量公式之间要转置矩阵；说明显示权重和100.01%来自舍入。
- §3.2.2 用实际重跑脚本03的seed=42输出对照除以8前后分布，并链接代码，区分“缩放会改变权重”和“减最大值不改变权重”。
- 自测1区分“交换Q/K使S变Sᵀ”和“权重矩阵自身是否对称”：S对称时交换前后相同，尽管权重矩阵本身可能不对称。
- 对齐 any_allowed 变量名、因果与padding取与措辞；修正脚本打印的 True/False、-inf 与浮点舍入说法；第4章补大纲S05的GQA原文链接。

新增3项回归分别锁定脚本02的数值输出、脚本03的缩放输出、对称/非对称S交换Q/K的区别。现有全mask梯度测试加入 anomaly detection；实际临时注入“跳过safe直接softmax”的不安全变异，测试因 SoftmaxBackward0 返回NaN而拒绝，生产源码未保留变异。

本轮最终35项Python测试、3项Node测试、九个编号脚本CPU运行、静态重建及资源/链接检查通过。正常Chromium/布局仍受本容器socket限制；请在修正版head复验，不沿用旧head的PASS。
