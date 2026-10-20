# P24–P26：量化与模型案例验证记录

首稿核验日期：2026-10-19（head `45ecf45`，以下至返修节为当时本机记录）。基准 main `82d6c84db13a4089f93e06f1b2e8174f5a14ad59`；独立分支 `dot/batch9-p24-p26`。本批已撰写并完成下列本地验证，仍待 PR 审核，不表示已合并或新 head 浏览器验收通过。

## 范围与可复现入口

- P24：`code/25_quantized_serving.py`、23 项测试、7 个真实源码行为变异。标准库整数格点实验与实际 llama.cpp CPU 服务分开记账。
- P25：`code/26_qwen3_case.py`、22 项测试、12 个真实源码行为变异。固定官方配置与本地随机 tiny 模型分开，不下载预训练权重。
- P26：`code/27_deepseek_case.py`、16 项测试、6 个真实源码行为变异。标准库证据矩阵、MLA/MoE/CSA2 计数与视觉布局实验，不声称复现完整模型。
- 三章共 25 个提交的实际行为变异，不能与前批不同审计器的计数简单相加。
- 新增 4 项阅读契约测试；原第 9 章 14 个标题保留 15 个跨格式兼容 ID（HTML 与 EPUB 的旧 H1 不同）。从实际旧整书提取 `本章小结-1`，避免与前章重名。

## 测试与 CLI

实际运行 `python -m unittest discover -s tests -v`：

| 环境 | 总数 | 通过 | 跳过 |
|---|---:|---:|---:|
| 已有隔离 HF 环境，加 `P24_LLAMA_BIN` | 461 | 461 | 0 |
| 基线 `.venv`（CPU Torch，无可选 HF） | 461 | 451 | 10 |
| 系统 Python（NumPy 2.3.5，无 Torch） | 461 | 192 | 269 |

10 项跳过为原 8 项 HF、P25 官方 Transformers 对照 1 项、P24 外部引擎 1 项。依赖缺失不是失败，也不表示这些路径在基线运行过。隔离 HF 环境沿用 P20 的混合安装来源，不宣称全新 requirements 安装复现。

全部 27 个默认 CLI 通过；另实跑 P21 `--train`、P22 `--torch`、P20 `--hf`、P25 `--torch` 和 P24 `--llama-bin`。Node 3 项回归、`compileall` 通过。`git diff --cached --check` 仅报告两份相同 SSE fixture 的结尾空行：这是 `[DONE]` 事件所需的空行边界，保留原始响应，不删除或伪造。排除这两份协议文件后 whitespace 检查通过。

## 实际外部 CPU 对照

llama.cpp 官方 b5046，commit `74d4f5b041ad837153b0e90fc864b8290e01d8d5`：本地生成 18,624 参数随机、未训练 GGUF，F32 81,760 字节变为 Q8_0 27,392 字节。CPU CLI 生成 4 token；临时 loopback 服务的流式/非流式 token 文本相同，prompt/output/total 为 3/4/7。结果存于 `code/fixtures/`，不把控制字符输出当语言质量，不把此旧引擎 smoke 当 Qwen3/V4.1 兼容性证明。二进制不入仓库。

P25 可选 Transformers 4.51.3 eager 对照使用相同随机权重映射，两款 tiny 模型 logits 最大误差为 0 和 2.384e-7；各层 K/V 按 3e-6 容差通过。没有真实权重下载或 GPU 实验。

独立审校额外核对 exact-Fraction 量化格点、随机 SSE 分片、官方源文件 hash、参数公式、135 组视觉展开/CSA2 边界；不是以自实现互相比较代替独立 oracle。三章公式、自测和声明边界均已逐项审阅。

## 构建与视觉边界

正常 `build.py` 与 `npm run check:layout` 都实际尝试，但 Chromium 报 `socket() failed: Operation not permitted (1)`，不能声称正常浏览器通过，未禁用 sandbox 或改低安全设置。请 Normal Work 在新 head 补跑正常构建和 28 章 × 3 宽度布局检查。

备用构建 `scripts/build_without_browser.py --baseline 82d6c84db13a4089f93e06f1b2e8174f5a14ad59` 仅复用源码精确匹配的基准图；10 张 Mermaid 与 23 张旧静态图复用，3 张新 SVG 用 CairoSVG 2.8.2 栅格化。需要 CairoSVG/Pandoc 3.9/中文字体，不能将备用路线称为浏览器渲染。

三张新 SVG 已查看真实中文 PNG 像素，未见乱码、裁切或重叠。HTML、EPUB 与 Markdown 同步；`check_book.py` 通过，35 个 EPUB XML/XHTML/OPF/NCX 容器可解析，46 个 code 数据/脚本文件和 26 张 SVG 与源逐字相同。15 个旧锚点在 HTML 唯一存在、在 EPUB 可定位。顺手修复 P22 历史公式的 `\rm` 为等义 `\mathrm`，避免 EPUB 降级为原始 TeX；没有改动其数值公式。

本报告不把历史 PR9 的 78 项浏览器布局通过记录移用到本批；GPU、分布式、真实模型质量、吞吐和显存收益均未实测。


## PR10 复审返修（2026-10-19）

[Normal Work review 5576213837](https://github.com/cluster1900/zero2llm/pull/10#pullrequestreview-5576213837) 对首稿独立验证了 27 CLI、5 个选修入口、来源/数表、正常构建和 84/84 布局；其 HF 环境实际为 460 pass / 1 fail。失败是 MoE 因果不变性断言把约 2.22e-16 的平台累加差异误判为泄漏，不能用本机原先的 461 pass 覆盖该结果。

本次没有改变任何 `code/` 数值实现：

1. dense 因果断言继续要求精确相等，MoE 只给 1e-12 绝对容差、相对容差 0；正文解释按专家聚合的 GEMM 行数影响末位累加。
2. 测试给每个 RMSNorm gain 独立、固定种子的非单位值；标量参考和官方 Transformers 对照都用这套参数。新增 Q/K gain 混用、block norm 交换、final norm 错用的实际源码变异，不用自一致比较代替独立 oracle。
3. 全负仿射、SSE 明确拒绝原因、同长度 SHA 篡改，以及相邻零 scale/负误差/EOF/多行数据/non-thinking 控制 ID/SWA 来源/Markdown 转义契约均补回归。
4. README/前言/大纲改为中性“已实现”，并明确实现覆盖不等于审核发布；本历史验证记录保留首次审核状态。`.gitattributes` 仅对 SSE 协议文件关闭 `blank-at-eof` 检查，其余 whitespace 规则不变。

返修最终结果：

| 环境 | 总数 | 通过 | 跳过 |
|---|---:|---:|---:|
| 隔离 HF + 官方 llama.cpp CPU | 479 | 479 | 0 |
| 基线 CPU Torch | 479 | 468 | 11 |
| 系统 NumPy / 无 Torch | 479 | 207 | 272 |

P24 为 35 tests / 15 实际变异，P25 为 27 tests / 15 实际变异，P26 为 17 tests / 8 实际变异，共 38 个不同源码变异；P25 同一 norm 变异在 scalar 与 HF、dense 与 MoE 下重复验证，不重复计数。基线 11 项 skip 为原 8 HF + P25 两项官方对照 + P24 实际引擎入口。状态文本回归 4 项通过；27 个默认 CLI、5 个选修入口、3 个 Node tests 与 compileall 全部重新运行通过。

独立审校重新运行三个章节及全部新增行为，另验证 B24/B35/B36 错误结果有限而显著偏离、SHA 校验删除后同长度篡改被错误接受、负区间缩放和 187 个 SSE 分片切点。生产 `code/` 与首稿逐字不变。

正常构建/布局本次再尝试仍报 Chromium `socket() failed: Operation not permitted (1)`，未改安全设置。备用构建以源码精确匹配的 `45ecf45` 为基准，10 Mermaid + 26 SVG 均复用已审图，正文重建，Pandoc 无 warning；check_book 通过，35 XML、46 代码/数据文件、26 SVG、15 兼容 ID 全部核验。没有新增图，也没有把首稿 84 项布局通过结果冒充返修 head 验收。请 Normal Work 在新 head 再跑正常浏览器布局。

`.gitattributes` 后 `git diff --check` 无需人工排除 SSE；不修改原始协议字节，不停用其他 whitespace 检查。


## 已完成的独立复审与合并

Normal Work [复审5578093405](https://github.com/cluster1900/zero2llm/pull/10#pullrequestreview-5578093405) 在固定 head `0648ba5` 重新确认完整479、基线468+11skip、系统207+272skip，提交/重建两版本正常Chromium84/84布局。审核侧100个变异：95行为拦截、5当时fixture等价、0存活。PR10于2026-10-20 00:25:11 UTC合并，merge `69e5c87ed4b5762a89127ab395738f688fac6299`。该结果只属于此head；下一批新增GGUF非对齐及多闭合标签回归会另记测试数字。
