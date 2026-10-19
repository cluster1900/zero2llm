# P15–P17：注意力系统、稀疏/线性机制与 CED 验证记录

核验日期：2026-10-19。基准 main 为 `d9b5ad79e4131730a0794eaf935a52f886c34e4c`，独立分支 `dot/batch6-p15-p17`。仅覆盖已审核原大纲 P15–P17；不引入 #28/#29 的拟扩展内容，不直接推 main、不自动合并。

## 最终验证结果

- **Python 256/256 通过，0 跳过**。P15 26 项、P16 21 项、P17 15 项；另有跨章节/源码/图解/进度回归。测试包含执行故意错误版本并要求语义断言失败的检查，不把这些检查另加到总测试数。
- **无 torch 系统 Python：91 通过、165 明确跳过，共 256 项**。P15 核心标准库路径可运行；P16/P17 缺 torch 时 CLI 给安装提示、退出码 2，没有 traceback。
- **Node 3/3 通过**。全部 18 个编号脚本 CPU CLI 实际执行，退出码均为 0；未执行 GPU kernel。
- **22 个内置行为变异**：P15 12 个、P16 5 个、P17 5 个，均被数值/状态/因果契约拒绝。丢掉旧最大值的变异在合法有限输入上抛 OverflowError，测试将这一明确的稳定性违约转换成断言；不将其他无关异常算作捕获。不是只比较两个共享错误的实现。
- **三张新增中文 SVG 均实际渲染并查看像素**。P16 初次渲染发现 Unicode 转置上标缺字及箭头穿字，改为 ASCII ^T 并移除穿线文字，重渲染后通过；P15/P17 中文、箭头与边界说明无缺字、重叠或裁切。
- **静态构建及 check_book 通过**：20 个章节、12 张 Mermaid 流程图、17 张静态插图，共 29 张插图。HTML/EPUB 资源、内部链接、目录锚点检查通过；27 个 EPUB XML 类文件（.xhtml/.opf/.ncx/.xml）均可解析，code 与 SVG 的 dist 副本逐字节一致。
- **正常 Chromium 构建与布局已实际尝试，但未通过环境启动阶段**：两者均报 `FATAL:chrome/browser/process_singleton_posix.cc:297 ... socket() failed: Operation not permitted (1)`。没有更改 sandbox 或安全设置，不能宣称本批浏览器布局通过。需审核方对本批 head 跑正常构建与 20×3=60 个章节/视口组合。
- 首次实现的静态路径使用固定基准 `d9b5ad79e4131730a0794eaf935a52f886c34e4c`：只复用源码完全相同的 12 张 Mermaid 和 14 张未变静态图，3 张新 SVG 用 CairoSVG 2.8.2 重渲染。它不替代真实浏览器布局验收。

### 可复跑命令

在仓库根目录、已安装 README 所列依赖的环境运行：

```bash
.venv/bin/python -m unittest discover -s tests -v
python3 -m unittest discover -s tests -v
node --test tests/
for f in code/[0-9][0-9]_*.py; do OMP_NUM_THREADS=1 .venv/bin/python "$f"; done
.venv/bin/python -m compileall -q code tests scripts build.py
npm run build
npm run check:layout
.venv/bin/python scripts/check_book.py
git diff --check
```

受限环境的替代构建命令为 `FONTCONFIG_FILE=/tmp/p06-fonts.conf .venv/bin/python scripts/build_without_browser.py --baseline d9b5ad79e4131730a0794eaf935a52f886c34e4c`。其中 FONTCONFIG_FILE 是本机的 Noto Sans CJK SC 字体配置，其他机器须使用实际可用字体配置；它不是浏览器安全开关。

## 独立审校及具体数值

P15/P16 独立审校另做 60 个随机 attention 样本×4 个分块配置、1,000 次分页操作与独立序列历史比对，以及 6 个额外语义变异。P17 独立审校对 encoder/decoder 1–3 层、W=1/2/4/8 共 36 种配置核对标量参考、混合 chunks 与所有未来切点；原始论文/固定源码边界也逐项核对。审校提出的 P16 autocast 契约问题已通过显式拒绝与真实上下文测试修正；P15 HBM/GDDR 措辞已校正。审校没有剩余阻塞，但不替代下一轮 Normal Work 的正式 PR 验收。

| 实验 | 本机输出与解释 |
| --- | --- |
| P15 在线归一化 | 手算输出 6；随机不整齐分块最大误差 1.11e-16，容差 1e-12 |
| P15 COW | A/B 逻辑总长度 7，物理载荷 5，分配槽 6，空槽 1 |
| P15 时间线 | 人工客户端 TTFT 均 100 ms，TPOT 分别45/60 ms；人工观察窗吞吐23.809524 token/s，不是实测性能 |
| P16 递推 | float64 并行/递推最大误差3.33e-16，状态 S/z 形状[2,3,2]/[2,3] |
| P16 检索 | T=1/8/64/256 时线性核输出1/.951219/.684206/.348654；S+z 固定32字节；结论只适用该受控核和干扰设置 |
| P17 手算 | encoder=[2,7/2,11/2]；单层full=[6,9,41/3]，replay=[39/4,41/3] |
| P17 两层反例 | uniform末位full=86/3，replay=697/24，差3/8；不把单层巧合推广为等价 |
| P17 默认路径 | 24/18 block visits，global K+V投影对两者均12；末位差0.014462788；这些计数不是 FLOPs 或测速 |

不同平台的浮点末位可不同；容差见测试。所有数字都来自小实验或明确标注的人工算术，不是生产模型成绩。

## 范围与明确边界

| 单元 | 新增路径 | 证明范围 |
| --- | --- | --- |
| P15 | `chapters/08e_attention_systems.md`、`code/16_attention_systems.py` | CPU 在线 Softmax 数值、分页账本所有权、人工调度时间线 |
| P16 | `chapters/08f_sparse_linear_attention.md`、`code/17_sparse_linear_attention.py` | 窗口可见性、感受野、归一化线性注意力递推及受控检索例 |
| P17 | `chapters/08g_ced_architecture.md`、`code/18_ced_architecture.py` | 独立教学 CED 数据来源、因果性、精确与截断重放的区别 |

三个独立机制实验不更换 S4 主线。CPU 模拟不证明 FlashAttention/PagedAttention GPU 加速，不证明真实服务吞吐，也不复现 DeepSeek-V4.1-Flash 权重或质量。P17 官方 8B prefill / 16B decode 为原始报告的激活参数口径，不是本书计时结果。

## 来源与版本

- CED：[技术报告 v1](https://arxiv.org/html/2609.19969v1)，§2.2、§2.3、§3.2.2；官方模型材料固定 revision `314113f8a7dadb388215396ba5fbee2c610a04d6`。全局 KV 来源与近似重放边界分别核对，不能混为 P/D 服务拆分。
- 其他原始论文、软件文档的固定版本和具体章节见本批各章末尾。文档快照不代表推荐安装版本，不下载大模型权重作为必修。
- 运行环境延续 Python 3.12.14、PyTorch 2.14.1+cpu、Pandoc 3.9、CairoSVG 2.8.2；GPU 选修未运行。最终实际版本随验证命令核对。

## 上批非阻塞建议

README 的 Pandoc 3.9 发布锚点改为 `pandoc-3.9-2026-02-03`；P14.10 多余空格已移除；第五批记录补充审稿方更正的“35/44 旧变异拒绝，44/44 修复后拒绝，另有一项未变异基线”，不将基线计入变异数。上批正常浏览器 51 layouts 的结果只属于 PR #6，不能覆盖本批新增页面。

## PR #7 第一轮复审后的补强

[Normal Work review 5569958426](https://github.com/cluster1900/zero2llm/pull/7#pullrequestreview-5569958426) 对旧 head `46aa14d` 验证253 tests、18脚本、正常build和60 layouts通过，但发现 P15 三个测试缺口。本次没有修改任何数值源码：新增下降极值与 Decimal 独立参考、非零到达的完整指标字典及3个针对性变异。每个变异契约先在原实现上通过。原先“零运行错误”的计数只描述已选变异，未覆盖此种合法输入溢出；不能由该口径推断所有稳定性错误都被发现。

审稿方旧60个变异中55个拒绝、5个存活，其中2个等价/冗余。这里只报告本书22个内置变异及新增三项独立审校，不声称已运行审稿方整套60项。旧head的60 layouts不能覆盖本次重新生成的产物，新head仍请审核方补跑。

XML统计已扩大为包含 `.xml` 的27份；原25份统计仅包括 `.xhtml/.opf/.ncx`，不是声称额外两份已损坏。修正 FlashAttention 论文版本措辞以免误读成 FlashAttention-2，并清理 P17 两处空格、补充路径计数与报告对应。旧章7的 HTML/EPUB 锚点不同，本轮保留普通文字路标，不引入可能断裂的统一链接；P15图块表信息已完整，未为可选箭头建议改变图。

本次返修重新尝试正常 build/layout，仍是上述 socket 权限限制；静态重建以旧 head `46aa14d8ce0e62a585cc82824cbe0faeab8254b9` 为基准，复用源码完全不变的12张Mermaid和17张静态图，无图像改动。重新解析27份XML，源码dist副本一致。独立审校复跑26个P15测试及3个新变异，确认正确基线有限、running-max变异为真实合法输入溢出、两个时间指标变异为字典断言失败。

## 合并后的正式复审

[Normal Work review 5570104663](https://github.com/cluster1900/zero2llm/pull/7#pullrequestreview-5570104663) 对修复 head `579f092` 独立验证256 tests、18脚本、正常构建和60 layouts通过，并确认3个真实测试缺口均已修复。该评论的变异总数与子项算术不一致，本记录不采用其总计作新的统计事实；只引用明确确认的三个修复和测试/构建结果。PR #7 已合并至 `66dbeb94dd9935046af0e859a223775b7adf4e0e`。

后续 [PR #8 review 5570682715](https://github.com/cluster1900/zero2llm/pull/8#pullrequestreview-5570682715) 已更正该历史计数：PR #7 的 60 个变异中，58 个被拒绝（57 个断言失败、1 个 ValueError），2 个等价/冗余变异存活。这是审核方历史结果，不是本批重新执行的统计。
