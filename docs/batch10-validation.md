# P27–P28：综合体检与参考附录验证

核验日期：2026-10-20。基准 main `69e5c87ed4b5762a89127ab395738f688fac6299`；独立分支 `dot/batch10-p27-p28`。本批原定内容已实现并完成以下本地验证，仍待 PR 审核；不表示新 head 已通过浏览器验收。

## 范围与独立检查

- P27：`code/28_minigpt_healthcheck.py`、35 项测试、完整中文正文与体检图。复用已有 S0–S6 实现，核对同权重整段/增量 logits、实际 KV 字节、位置/掩码/请求隔离、受控消融和可信本地 checkpoint。不同架构的输出不作相等或质量排名。
- P27 实际执行 9 个源码变异与 4 类行为故障，共 13 种去重故障：源码变异包括 cache mask、位置、RoPE 符号、RMS 统计、SwiGLU 门、GQA 映射、metadata guard、dtype guard、未来标签移位；另四类是未来可见掩码、错请求缓存、无关消融权重改动和错误 greedy 结果。缓存污染不是源码变异，测试框架自身检查也不计入此数。
- 独立审校发现并修复了 checkpoint 整数权重被 `load_state_dict` 静默转为浮点的问题。现在逐键对照目标 state 的精确 dtype，保留合法 buffer dtype；int/bool/complex 替换和移除防线的真实变异均被回归捕获。`weights_only=True` 不是加载不可信文件的授权；不是训练状态恢复或 HF/GGUF 通用兼容器。
- P28：11 项标准库测试，数学手算、形状、token 加权、梯度步长、内存单位、来源核验流程和 6 道自测逐题对应。负频率周期用 `2π/|ω|`，表格 TeX 使用 `\lvert`/`\rvert` 避免竖线破表。Qwen3-0.6B 固定 revision 的官方配置重新取回，726 字节和 SHA-256 与既有证据一致。
- 携带 PR10 两项非阻塞边界：P24 新增非 32 字节对齐 GGUF 格式探针，实际移除 padding 的变异失败；P25 新增多次 thinking 闭合标签，实际改用最后闭合位置的变异失败。分别为 36 和 28 项测试，各 16 个源码行为变异；没有改动这两章数值实现。
- 独立审校额外做了 S1–S6 的 768 组 float64 分块组合（最大 logits 误差 1.11e-15）、dtype payload 倍增、精确 dtype 拒绝和正文公式/自测/来源检查。不是仅比较两条共享错误的路径。

## 实跑矩阵

全部运行 `python -m unittest discover -s tests -v`：

| 环境 | 总数 | 通过 | 跳过 |
|---|---:|---:|---:|
| 已有隔离 HF 环境，加真实 `P24_LLAMA_BIN` | 531 | 531 | 0 |
| 基线 `.venv` CPU Torch | 531 | 520 | 11 |
| 系统 Python（NumPy 2.3.5，无 Torch） | 531 | 227 | 304 |

11 项基线跳过为可选 HF/外部引擎路径，不算作已经执行。隔离 HF 沿用 P20 的混合安装来源，不宣称干净重装 requirements 已复现。P27 单独在无 Torch 环境为 3 通过、32 明确跳过。

28 个默认编号 CLI 全部通过；另实跑 P21 `--train`、P22 `--torch`、P20 `--hf`、P25 `--torch`、P24 `--llama-bin` 和 P27 `--mtp --json`。Node 3 项回归、`compileall`、whitespace 检查通过。真实 llama.cpp 仍是已固定官方 b5046 CPU 随机 tiny 模型，不下载预训练权重、不推断真实模型质量或 GPU 收益。

## 构建、旧链接与视觉边界

正常 `build.py` 与 `npm run check:layout` 均实际尝试，Chromium 报 `socket() failed: Operation not permitted (1)`。未禁用 sandbox 或降低安全设置。请 Normal Work 在本批新 head 补跑正常构建和 29 章 × 3 宽度（87 项）布局；PR10 的 84 项历史通过不能移用。

备用 `scripts/build_without_browser.py --baseline 69e5c87ed4b5762a89127ab395738f688fac6299` 使用 Pandoc 3.9、CairoSVG 2.8.2 和中文字体；仅复用源码完全相同的 10 张 Mermaid 与 26 张旧 SVG，新 P27 SVG 独立栅格化并查看真实中文像素，无裁切、重叠或缺字。备用构建不等于浏览器渲染。

HTML、EPUB、Markdown 同步；`check_book.py` 通过（10 张流程图、37 张图示）；36 个 EPUB XML/XHTML/OPF/NCX 文件可解析。47 个 code 文件、27 张 SVG 与源逐字相同。15 个旧案例锚点和 8 个旧附录锚点在两份 HTML 及 EPUB 各唯一；P27 小结采用独立 `p27-summary` ID，避免占用旧 `本章小结-1`。附录周期行在两格式均为三列，EPUB MathML 正确保留绝对值。

本批完成原 P00–P28 大纲的内容实现，公开状态采用中性“已实现”，不冒充已审核。GPU、多卡、生产服务收益和完整模型训练均未运行；不纳入 #28/#29 的额外扩展。
