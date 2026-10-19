# P15–P17：注意力系统、稀疏/线性机制与 CED 验证记录

核验日期：2026-10-19。基准 main 为 `d9b5ad79e4131730a0794eaf935a52f886c34e4c`，独立分支 `dot/batch6-p15-p17`。仅覆盖已审核原大纲 P15–P17；不引入 #28/#29 的拟扩展内容，不直接推 main、不自动合并。

## 最终验证结果

- **Python 253/253 通过，0 跳过**。P15 23 项、P16 21 项、P17 15 项；另有跨章节/源码/图解/进度回归。测试包含执行故意错误版本并要求语义断言失败的检查，不把这些检查另加到总测试数。
- **无 torch 系统 Python：88 通过、165 明确跳过，共 253 项**。P15 核心标准库路径可运行；P16/P17 缺 torch 时 CLI 给安装提示、退出码 2，没有 traceback。
- **Node 3/3 通过**。全部 18 个编号脚本 CPU CLI 实际执行，退出码均为 0；未执行 GPU kernel。
- **19 个内置行为变异**：P15 9 个、P16 5 个、P17 5 个，均被数值/状态/因果断言拒绝，零变异运行错误。不是只比较两个共享错误的实现。
- **三张新增中文 SVG 均实际渲染并查看像素**。P16 初次渲染发现 Unicode 转置上标缺字及箭头穿字，改为 ASCII ^T 并移除穿线文字，重渲染后通过；P15/P17 中文、箭头与边界说明无缺字、重叠或裁切。
- **静态构建及 check_book 通过**：20 个章节、12 张 Mermaid 流程图、17 张静态插图，共 29 张插图。HTML/EPUB 资源、内部链接、目录锚点检查通过；25 个 EPUB XML/XHTML 文档均可解析，code 与 SVG 的 dist 副本逐字节一致。
- **正常 Chromium 构建与布局已实际尝试，但未通过环境启动阶段**：两者均报 `FATAL:chrome/browser/process_singleton_posix.cc:297 ... socket() failed: Operation not permitted (1)`。没有更改 sandbox 或安全设置，不能宣称本批浏览器布局通过。需审核方对本批 head 跑正常构建与 20×3=60 个章节/视口组合。
- 静态路径使用固定基准 `d9b5ad79e4131730a0794eaf935a52f886c34e4c`：只复用源码完全相同的 12 张 Mermaid 和 14 张未变静态图，3 张新 SVG 用 CairoSVG 2.8.2 重渲染。它不替代真实浏览器布局验收。

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
