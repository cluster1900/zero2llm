# P27–P28：综合体检与参考附录验证

首稿核验日期：2026-10-20（head `39154e23`；以下首稿记录与末尾返修记录分开）。基准 main `69e5c87ed4b5762a89127ab395738f688fac6299`；独立分支 `dot/batch10-p27-p28`。本批原定内容已实现并完成以下本地验证，仍待 PR 审核；不表示新 head 已通过浏览器验收。

## 范围与独立检查

- P27：`code/28_minigpt_healthcheck.py`、35 项测试、完整中文正文与体检图。复用已有 S0–S6 实现，核对同权重整段/增量 logits、实际 KV 字节、位置/掩码/请求隔离、受控消融和可信本地 checkpoint。不同架构的输出不作相等或质量排名。
- P27 实际执行 9 个源码变异与 4 类行为故障，共 13 种去重故障：源码变异包括 cache mask、位置、RoPE 符号、RMS 统计、SwiGLU 门、GQA 映射、metadata guard、dtype guard、未来标签移位；另四类是未来可见掩码、错请求缓存、无关消融权重改动和错误 greedy 结果。缓存污染不是源码变异，测试框架自身检查也不计入此数。
- 独立审校发现并修复了 checkpoint 整数权重被 `load_state_dict` 静默转为浮点的问题。现在逐键对照目标 state 的精确 dtype，保留合法 buffer dtype；int/bool/complex 替换和移除防线的真实变异均被回归捕获。`weights_only=True` 不是加载不可信文件的授权；不是训练状态恢复或 HF/GGUF 通用兼容器。
- P28：11 项标准库测试，数学手算、形状、token 加权、梯度步长、内存单位、来源核验流程和 6 道自测逐题对应。负频率周期用 `2π/|ω|`，表格 TeX 使用 `\lvert`/`\rvert` 避免竖线破表。Qwen3-0.6B 固定 revision 的官方配置重新取回，726 字节和 SHA-256 与既有证据一致。
- 携带 PR10 两项非阻塞边界：P24 新增非 32 字节对齐 GGUF 格式探针，实际移除 padding 的变异失败；P25 新增多次 thinking 闭合标签，实际改用最后闭合位置的变体偏离本练习首闭合约定，回归将其识别；官方模型卡本就采用最后闭合，不将官方口径称为错误。分别为 36 和 28 项测试，各 16 个源码行为变异；没有改动这两章数值实现。
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


## PR11 两次正式审核与最终返修

[首轮审核 5579713086](https://github.com/cluster1900/zero2llm/pull/11#pullrequestreview-5579713086) 和[补充审核 5579806138](https://github.com/cluster1900/zero2llm/pull/11#pullrequestreview-5579806138) 都针对首稿 `39154e23`。Normal Work 独立通过三环境 531 项测试、28 默认 CLI、正常 Chromium 提交/重建两版各 87 项布局、EPUBCheck 0 error/0 warning，但仍要求修改。补充审核将审核侧总计更新为 103 个变异：72 个行为拦截、1 个等价、30 个存活，替代此前 70 项的当前总计。已有测试通过不代表覆盖充分，亦不是批准合并。

### 声明与正文契约

- P25 首闭合只是本练习记账约定。固定官方模型卡采用最后闭合，模板 reasoning/content 分别取首前/末后；16 项测试变体包含“偏离教学约定”，不把官方口径称为错误。
- README/前言统一为原大纲 29 单元内容已实现，审核及 Release 状态另记；P27 模型卡改为仓库固定 commit / Release，删除“本次 PR”。README 备用构建基准更新为上一合并版本 `69e5c87`。
- P28 现在 16 项标准库测试。六处正文算术（含 z 与 L）按定位段落提取并独立计算，六道自测答案逐条解析，对照独立数学、固定配置与证据状态；六个算术变体、六个答案变体与固定 config URL 改 main 的变体均实际失败。不能靠全文其他位置恰好存在同一个数字通过。附录 D–G 标题统一，同时显式保留原 P28.D–G ID。

### P27 防线与独立变异

P27 现在 71 项测试（CPU Torch 全通过；无 Torch 3 通过、68 跳过），累计 31 个去重实际源码变异。坏输入、梯度、缓存、文件和状态注入按类别描述，不重复计作额外源码变异。纯生产行为入口完全不调用源码变异 helper；外部删除守卫必须因行为失败，不能只靠“源码锚点不存在”算拦截。

- 首/中块错误而末块正确；所有块统一加 0.25；仅首块缩放。四种分块均用真实完整前向作参照，不能拿缓存路径和自己比较。JSON 的四种分块及各误差/字节来自实际调用，另外记录所有 1–7 因果切点。
- 错 cache.length、溢出时静默截断、新请求沿用旧状态、MoE requested 总数错误、NaN loss 各有可观察坏行为和指定守卫删除后的漏检对照。独立审校从外层注入缓存自比及这五种守卫删除，纯行为入口分别产生 4/1/1/1/1/1 个断言失败，全部 0 errors；未用动态源码 inspect 错误冒充检出。
- backward hook 制造主路径零/NaN 梯度、主 head detach，以及旁路零/NaN/detach，有限 loss 不足以通过。旁路 NaN 同时受 general gradient guard 保护，这种冗余不伪装为新故障。独立外层梯度/pickle/payload/最后块变异同样由纯行为入口失败，无 inspect/anchor 附带错误。
- float64 的实际 KV payload 在 MHA 从 4096 变为 8192、GQA/MoE 从 2048 变为 4096。保存端拒绝 BF16、NaN，加载端拒绝额外 envelope 键和 inert dict 子类 pickle；该子类没有自定义 reduce/state 或执行载荷。S1/S2 只改 owner，不借不同权重版本提前拒绝。
- S6 用不同种子初始化再复制 S5，删除复制会真实改变 logits；原 MTP 计数仍为 proposed/accepted=7/3、target calls/tokens=4/12、draft calls/tokens=3/15。额外检查 checkpoint 未被短输入使用的 mask buffer、兼容性拒绝的具体原因、仅末层局部泄漏与全切点遍历。

### 最终完整验证

冻结后的最终完整重跑如下：

| 环境 | 总数 | 通过 | 跳过 |
|---|---:|---:|---:|
| 隔离 HF + 真实 llama.cpp CPU | 572 | 572 | 0 |
| 基础 CPU Torch | 572 | 561 | 11 |
| 系统 Python，无 Torch | 572 | 232 | 340 |

28 个默认 CLI、6 个扩展入口（P20 HF、P21 train、P22 torch、P24 llama、P25 torch、P27 MTP）、Node 3 项、compileall、whitespace 均重跑通过。依赖隔离来源与未运行 GPU/多卡边界仍按首稿说明，不将 skip 当 pass。

正常 Chromium 在返修期间再次实试仍报 socket 权限错误；未改安全设置。以首稿 `39154e23` 为精确源码匹配基准的备用构建通过，`check_book.py`、36 XML、47 code/27 SVG 精确副本、23 个旧案例/附录锚点唯一性通过；附录 D–G 和 P27 小结 5 个额外 ID 在两份 HTML 与 EPUB 亦各唯一。图源未变，首稿实际像素审查不替代返修正文的浏览器布局。请 Normal Work 在本批新 head 重跑正常构建与 87 项布局；历史首稿通过不移用。
