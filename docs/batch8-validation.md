# P21–P23：SFT、偏好优化与推理验证记录

日期：2026-10-19。基准 main `9c4c9e34de9d6d394e45781b4ab115ee9e196f8d`，独立分支 `dot/batch8-p21-p23`。仅原已审核大纲，不引入 #28/#29，不推 main、不自动合并。

## 验证范围

本批增加三章正文、三张中文 SVG、code 22–24 与对应行为回归。P21/P22 默认 CLI 是标准库手算，真实 CPU 参数更新分别需要 `--train` / `--torch`；P23 默认执行真实 PyTorch S5/S6 与贪心验证实验。默认命令成功不能代替可选训练运行证据。

| 验证路径 | 最终实际结果 | 不可推断的范围 |
| --- | --- | --- |
| 基础 `.venv` | 396 项：388 通过、8 个可选 HF 测试跳过 | 不代表基础环境运行过 HF |
| 既有隔离 HF 环境 | 396/396 通过，零跳过 | 不代表 GPU / 多卡验证 |
| 本轮无 torch、有 numpy 2.3.5 的系统 Python | 396 项：139 通过、257 明确跳过 | 跳过不算数值通过 |
| Node 布局选择逻辑 | 3/3 通过 | 不是真实浏览器布局 |
| 24 个编号脚本默认 CLI | 全部退出码 0 | P21/P22 默认只跑手算 |
| P21 `--train`、P22 `--torch`、P20 `--hf` | 三条实际 CPU 训练/更新路径通过 | 不代表运行过 TRL / PEFT |
| compileall / diff whitespace | 通过 | 不替代行为测试 |

三张新中文 SVG 均实际栅格化并查看像素；发现的下标字符缺字已改成可读普通字符并重渲染。最终文本、箭头、手算标签均清晰，无重叠、裁切或方框缺字。

正常 Chromium build 与 `check:layout` 已对本批实际尝试，均在启动阶段失败：`FATAL:chrome/browser/process_singleton_posix.cc:297 ... socket() failed: Operation not permitted (1)`。没有修改 sandbox 或安全设置；不声称新 head 浏览器布局通过，请 Normal Work 正常构建并补验 26×3=78 个章节/视口组合。

随后静态构建与 `check_book` 通过：26 个章节、12 张 Mermaid 和 23 张静态图，共 35 张插图。仅复用以上基准中源码完全相同的 12 张 Mermaid 和 20 张既有静态图，3 张新 SVG 使用 CairoSVG 2.8.2。33 份 EPUB XML 类文件解析通过；25 个 Python 源文件（24 个编号脚本及 attention_core.py）和 23 个 SVG 与 dist 副本逐字节核对。HTML/EPUB 资源、代码链接与目录锚点完整。静态检查不替代浏览器检查。

## 实际实验与独立参考

| 实验 | 已实际运行的结果 | 边界 |
| --- | --- | --- |
| P21 assistant-only SFT | 3760 个基础参数，输入 [4,21]、10 个监督目标；24 步 loss 2.83717895 → 0.61146003 | 从零初始化的教学 Transformer，不是预训练大模型微调成绩 |
| P21 LoRA | q/v 两层共 128 个可训练参数；24 步 0.61146003 → 0.61064279；合并 logits 最大差 7.15e-7 | 从上一步 SFT 继续，不是全量/LoRA 同起点公平比较；基础权重保持冻结 |
| P22 六参数策略 | 两题各三个完整答案；SFT 后平均奖励 0.506496，DPO 后 0.608010，GRPO 后 0.560679 | 完整答案作为有限动作，不是 LLM / TRL 训练 |
| P22 精确 KL | DPO 两题各 0.024651；GRPO 为 0.006216 / 0.006595 | 枚举全部三个动作的条件 KL，不冒充论文采样估计 |
| P23 算术预算 | 0/1/2/3/4 个竖式列预算正确 0/1/3/5/6 题（共六题） | 写好的确定性竖式算法，不是语言模型推理能力曲线 |
| P23 贪心生成 | 普通/草稿验证均生成 [5,6,3,1,9,2,5,6,3,1] | 仅贪心相等；随机接受校正只有手算，没有采样器 |
| P23 成本账本 | target forward 10 → 6，但输入 token 11 → 18；草稿额外 4 次完整前缀 forward、24 个输入 token；接受 4/11 | 不把调用次数减少当成吞吐或延迟加速 |

浮点末位随平台可变化，回归使用明确容差。MTP 为 S5 的独立线性辅助 head，S6 可关闭；不冒充 V3 串行 MTP 或 V4.1 DSpark。

## 独立审校与行为变异

- P21：28 项测试；12 个真正执行的错误变体。独立字面矩阵和解析梯度验证 alpha/r、冻结、初始化与合并；审校发现原 rank=1 参考不能抓到漏除 rank，已加入 rank=2、alpha=6 的前向、输入/A/B 梯度和合并参考。assistant-only 标签与 causal attention 是不同屏蔽。
- P22：21 项测试；21 个真正执行的错误变体。DPO 概率比、GRPO 分组/总体标准差/clip/长度归约、精确 KL 均有独立标量或梯度参考。审校者另外用闭式标量梯度复算三个 GRPO 更新，最终 logits 最大差 5.55e-17。极端有限 old logp 导致 exp 溢出时明确拒绝非有限比值，而非让零优势产生 NaN。
- P23：25 项测试；12 个真正执行的错误变体。独立标签/CE/梯度、token-history 目标、真实 S5 完整前缀 logits 和各层 K/V 验证。审校额外运行 300 组模型/草稿/EOS/预算组合及未来输入干预。新增所有 MTP head 的未来隔离回归；全忽略 CE 使用安全的可微空归约，极大有限 float32/float16 输入仍返回零及零梯度。
- 以上 45 个变异是本批源码内实际执行的变体，不是审核方尚未运行的新变异总数；独立扩展探针也不冒充全部已提交回归。

## 依赖、来源与未运行范围

沿用 Python 3.12.14、PyTorch 2.14.1+cpu、Tokenizers 0.21.4、Pandoc 3.9、CairoSVG 2.8.2；未改基础或既有隔离 HF 环境，未下载模型权重。可选 HF 仍固定 Transformers 4.51.3、Accelerate 1.6.0、Tokenizers 0.21.4、huggingface-hub 0.30.2、safetensors 0.5.3；其余包复用基线环境。静态渲染依赖见 requirements-build-static.txt。

- P21：[LoRA v2](https://arxiv.org/abs/2106.09685v2)、[QLoRA v1](https://arxiv.org/abs/2305.14314v1)、Transformers 4.51.3 模板、PEFT 0.15.0 与 TRL 0.18.2 源码。后两者仅机制/接口参照，未安装或运行；QLoRA 未实跑。
- P22：[DPO v3](https://arxiv.org/html/2305.18290v3)、[DeepSeekMath v3](https://arxiv.org/html/2402.03300v3)、InstructGPT v1、PPO v2 与 TRL 0.16.1 固定源码。正文显式区分教学 GRPO 与该 TRL 版本的标准差、epsilon、长度归约及 KL 变体；配置未执行，不承诺与 P20 环境兼容。
- P23：[R1 v1](https://arxiv.org/html/2501.12948v1)、[V3 v2](https://arxiv.org/html/2412.19437v2)、[Speculative Decoding v2](https://arxiv.org/html/2211.17192v2)、[V4.1-Flash v1](https://arxiv.org/html/2609.19969v1)。训练目标、蒸馏与执行方法分开说明；未运行官方权重或 DSpark。

版本不同的 TRL 引用是各章明确固定的源码导读，不是把多个版本装进一个已验证环境。没有 GPU、多卡、量化训练、生产 RLHF、随机投机采样或真实性能实验。

## 复跑

```bash
.venv/bin/python -m unittest discover -s tests -v
python3 -m unittest discover -s tests -v
node --test tests/
for f in code/[0-9][0-9]_*.py; do OMP_NUM_THREADS=1 .venv/bin/python "$f"; done
.venv/bin/python code/22_sft_lora.py --train
.venv/bin/python code/23_preference_optimization.py --torch
# 可选环境按 README 创建；下面路径以实际环境为准
.venv-frameworks/bin/python code/21_training_frameworks.py --hf
.venv-frameworks/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q code tests scripts build.py
npm run build
npm run check:layout
.venv/bin/python scripts/check_book.py
git diff --check
```

本机静态复跑：`FONTCONFIG_FILE=/tmp/p06-fonts.conf .venv/bin/python scripts/build_without_browser.py --baseline 9c4c9e34de9d6d394e45781b4ab115ee9e196f8d`。本机字体配置指定 Noto Sans CJK SC，不修改浏览器安全选项。

## PR #9 首轮独立复审与返修

[Normal Work review 5573077171](https://github.com/cluster1900/zero2llm/pull/9#pullrequestreview-5573077171) 对旧 head `c340f2a` 独立验证 390 个 HF 测试、24 个 CLI 与三条额外训练入口，正常 Chromium 构建及提交/重建产物各 78 layouts 通过。其系统 Python 报告有 numpy，实测为 165 通过 / 225 跳过；本机首次提交记录为 138 / 252。返修时重新检查本机系统环境已可导入 numpy，本轮重新实跑为 139 通过 / 257 跳过。不能仅凭“有 numpy”推断应有相同跳过数量，以各自完整环境及实际日志为准；本次返修未安装或更改依赖。该浏览器结果仅覆盖旧 head，不能自动覆盖本次返修。

审核侧 64 个变异中，60 个被拦下、2 个等价、F19 跳过主 head 未被旧测试识别，F07 删除修正 token 导致无进展循环。这两项需要真实行为回归，不把卡死当作正常断言通过。返修同时在 P21/P22 正文说明三个 TRL 版本各自的来源用途和不可混用边界，而不改装环境或猜测其他版本行为。

本轮补强：

1. `MTPDraft` 顺序未改：完整 prefix 包含 pending，主 head 预测下一位置。新回归独立逐词标量投影得到 `[6,0,9]`，分别检查 limit=1/2/3；真正执行跳过主 head 的 F19，断言失败。另实际运行 CLI，固定 target 6 次/18 个位置、drafter 4 次/24 个位置、接受 4/11、拒绝 4、bonus 0。
2. 每轮提交前检查非空 emitted，缺失修正 token 的 F07 真实源码变异在 prefill + 首次验证两次调用后精确抛出 `RuntimeError('draft/verify made no progress')`。测试有不同类型的有界调用探针，若守卫失效会断言失败而非挂死；探针错误或超时不算成功。合法空草稿仍回退 target 生成并完成。
3. P21/P22 正文解释三处 TRL 版本是独立固定来源快照，不是混装环境；具体 GRPO 行为仅归属于核对过的 0.16.1，不推断其他版本同异。
4. P22 真实 CPU demo 在正常或异常退出时恢复调用方 RNG 与线程数；reference 使用独立更新前快照，新增同时修改 reference/initial 的错误变体。独立审校另注入首次 SGD.step 异常及非法 seed，均正确恢复。正常训练报告与旧 head 逐字相同。

本轮仍实际尝试正常 Chromium build/layout，仍为上述 socket 权限失败；静态重建以 `c340f2acef731096d7c6d2f3e38b6761b439d899` 为基准，复用源码未变的全部 12 张 Mermaid 和 23 张静态图，没有图像改动。旧 head 的 78 layouts 由审稿方确证，但新 head 仍待其重跑，未把历史结果当作新版本通过。
