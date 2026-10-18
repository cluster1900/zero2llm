# P00–P02 首批实验与产物记录

日期：2026-10-18。基于大纲合并后的 main `b946ef1`，独立主题分支 `dot/issue14-batch1-p00-p02`。本批交付完整阅读指南、词嵌入正文、Tokenizer/BPE 插入单元，待批次 PR 审核；不继续下一批、不合并、不发布 Release。

## 实现范围

- P00：四层学习地图、CPU 与选修路线、张量轴直觉、实验记录、复现边界、自测。
- P01：ID / one-hot / embedding 区分、查表与训练、几何计算、尺度与零向量、类比边界、自测。复用原 #11 审校中的数学纠错思路，只改本批相关文件。
- P02：完整字节 BPE 解释与逐行代码、Unicode / emoji、确定性合并与文档边界、真实库对照、模板与特殊 token 的边界。
- 保留旧第 2–10 章文件名；Tokenizer 暂插为第 1A 章。扩展计划 P03–P28 未实施，旧版内容不冒充新版完成。
- 四张新 SVG 图同时生成 PNG，包含在 HTML / EPUB 中。

## 环境与命令

Linux CPU；Python 3.12.14；选修 `tokenizers==0.21.4`；静态图构建 `CairoSVG==2.8.2`；Pandoc 3.9；Node.js 24.19.0（普通 Chromium 路径尝试时使用）。本批未使用 GPU、未下载模型权重、未调用付费 API。BPE 本身是确定性算法；独立随机样例回归固定 seed=42。

| 检查 | 结果 |
| --- | --- |
| `python3 code/01_vector_similarity.py` | 通过，水果值与正文一致 |
| `python3 code/08_byte_bpe.py` | 通过，实际 19 次合并、词表 275，所有示例原样往返 |
| `.venv/bin/python code/09_hf_tokenizer.py` | 通过，tokenizers 0.21.4，词表 275，所有示例原样往返 |
| `.venv/bin/python -m unittest discover -s tests -v` | 19 项通过：向量 5、BPE 11、HF 1、静态缓存安全边界 2 |
| `python3 -m unittest discover -s tests -v` | 标准库路线运行核心 16 项；HF / Cairo 未安装时 3 项明确跳过 |
| `python3 scripts/check_book.py` | 通过：12 张 Mermaid 流程图、共 23 张图，EPUB/Web 资源、内部链接、目录锚点完整 |
| `git diff --check` | 通过 |

向量修正包括：零向量余弦报错、维度不一致报错、点积不等于一般余弦。BPE 回归覆盖空文本、未见中文、emoji / ZWJ、组合重音、空白、NUL、非法 ID、非法 UTF-8、并列频数、重叠候选、训练重置、文档边界以及与独立排名优先参考编码的 100 个随机样例对照。

## 阅读产物如何生成

普通 `python3 build.py` 的 Chromium 启动被当前容器的 socket 限制阻止；提升执行也在环境启动阶段失败。未关闭浏览器沙箱或更改安全设置。

为完成本批阅读产物，增加了一个显式、可重跑的静态构建入口：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-build-static.txt
.venv/bin/python scripts/build_without_browser.py --baseline 72c39df44d05032edca7f0509bab6b765c35f48b
```

前提是 PATH 中有支持 `alerts` 的 Pandoc（本次 3.9）、中文字体，以及含该基准 commit 的本地 Git 历史。正常 Chromium 构建仍保持默认行为。

静态路径用同样的 Pandoc AST 对比基准原稿，**12 张 Mermaid 定义逐字匹配后才复用**对应已提交 SVG/PNG；7 张源码未变的静态图同样复用基准渲染，本批 4 张新 SVG 用 CairoSVG 2.8.2 栅格化。改动的 Mermaid 会直接报错，不会偷偷保留旧图。单元测试覆盖匹配复用和变更拒绝；这些测试使用图像字节夹具，不冒充图像质量测试。

随后同一个构建器生成合并 Markdown、HTML、站点入口、EPUB，运行已有产物检查。新图 PNG 已逐张人工查看，文字、连线和边界可见；EPUB 使用 PNG，Web 使用 SVG。所有示例同步到 `dist/code/`。

## 明确没有完成的验证

- Chromium 响应式布局检查 `npm run check:layout` 未能在此环境运行，不能声称手机/平板浏览器逐视口检查通过。
- 没有重新训练旧 Baby-GPT，也没有宣称对未改动的旧第 2–10 章完成第二轮全面审校。
- 没有 GPU 性能、生产 tokenizer 或真实 LLM 质量基准。小语料 token 数不代表任何服务的计费。
- GitHub Pages 与 Release 只会在获准合并/发布后更新；本批提交的是 PR 中的阅读产物，不是已部署网站。

## 来源版本与审核门槛

本批正文指向 HF Tokenizers 的 v0.21.4 标签、PyTorch 2.14 Embedding 文档与版本化基础论文；核验日期 2026-10-18。下批继续前须完成 Normal Work 对本批 PR 的审核。后续 P17 的 8B prefill / 16B decode 数字只作为官方报告引用，P20 已在大纲明确点名 TRL 与 Transformers 的关系，本批未提前撰写这些章节。
