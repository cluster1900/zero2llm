# P06–P08 第三批：实验、手算与验收记录

日期：2026-10-19。基于第二批合并后的 main `9eeecd4137f7847c1c8c17708b53fa92584439e4`，分支 `dot/issue30-batch3-p06-p08`。对应已批准大纲与 [bot-hub #30](https://github.com/cluster1900/bot-hub/issues/30)，不包含 #28/#29 拟议新专题。上一批[独立复审](https://github.com/cluster1900/zero2llm/pull/3#pullrequestreview-5566284130)的浏览器 PASS 不作为本批结果。

## 完整交付与手算来源

| 单元 | 保留的主线及新增内容 | 数表来源 |
| --- | --- | --- |
| P06 / 第5章 | 8位置×8维PE、位置0余弦回升；置换、三类位置信息、相加/拼接、长度边界、奇数宽度处理 | [code/05_positional_encoding.py](../code/05_positional_encoding.py) |
| P07 / 第6章 | Z-score三步；epsilon、Pre/Post-LN、零支路残差、FFN非线性与偏置参数账 | [code/06_transformer_block.py](../code/06_transformer_block.py) |
| P08 / 第7章 | 四词mask三步；三类架构、cross-attention、标签错位、prefill/decode、温度、EOS/长度停止 | [code/10_generation_loop.py](../code/10_generation_loop.py) |

第7章追踪现在用固定logits脚本生成，不再把示意概率当测量值：EOS在第4步停止；新增预算2在第2步停止。BOS/EOS在正文与表格用字面代码标记，避免Pandoc当HTML标签隐藏。

第3章仅修前轮要求的两处源码逐字一致措辞：`new_vec = [0.0] * len(V[0])` 和 `[s / scale for s in raw_scores]`，不扩写其他主题。README与大纲标记P00–P08正文/示例实现，P09–P28仍未按新大纲实现。当前12章节文件、10编号示例及共享注意力模块，不意味着29单元全部完成。

## 环境与执行命令

Python 3.12.14；PyTorch 2.14.1+cpu；NumPy 2.5.3；Tokenizers 0.21.4；CairoSVG 2.8.2；Pandoc 3.9；Node 24.19.0。CPU小型实验，无GPU、模型权重、付费API。PyTorch测试检查存在性与2.14.x兼容范围；缺失或不匹配明确skip，不能算完整实验通过。

```bash
.venv/bin/python -m unittest discover -s tests -v
python3 -m unittest discover -s tests -v
node --test tests/
.venv/bin/python code/05_positional_encoding.py
.venv/bin/python code/06_transformer_block.py
python3 code/10_generation_loop.py
python3 scripts/check_book.py
```

Node目录入口 `tests/index.js` 仅导入原来的三项截图辅助测试，使本机Node24支持上面精确的目录命令；没有修改测试断言或布局检查脚本。

## 自测与变异证据

- 第5章：公式每元素、奇数末列、余弦回升、置换关系、相加/拼接和参数/长度账，与自测逐项对应。
- 第6章：残差零支路、LayerNorm均值/总体方差与epsilon容差、Pre/Post数值顺序、FFN参数量和非线性、全前后向有限，与自测逐项对应。
- 第7章：标签切片、三步mask表、cross-attention的2×3分数与[0.7,0.8]加权值、温度概率/唯一最大低温、EOS第4步与长度第2步、零预算、固定seed、特殊token字面显示。六道自测分别对应这些数值回归或明确的执行路径解释。

变异仅在内存中或临时模块运行，生产源码未保留错误版本。共9种实测变异均被拒绝：

- 第5章：指数从 `2*k` 改为 `4*k`，独立逐元素断言失败；丢弃奇数宽度最后正弦列，奇数列回归失败。
- 第6章：移除注意力残差、移除FFN残差，均被零支路恒等测试拒绝；交换Pre/Post分支、把注意力归一化移至残差相加后，均被独立顺序公式测试拒绝。每项是断言失败，不是导入错误或skip。
- 第7章：将标签改为输入自身、忽略EOS、最大新增循环多走一步，分别被标签/停止回归拒绝。

最终集成：有固定依赖的CPU环境74项全部通过；系统无torch环境36项通过、38项明确skip；Node三项通过；10个编号示例全部执行成功。P06/P07还分别验证“模块存在但版本不兼容”时只跳过张量实验，依赖门控自身测试仍执行。

## 构建与实际浏览器边界

已实际执行正常Chromium路径：

```bash
PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium .venv/bin/python build.py
PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium npm run check:layout
```

PATH指向Pandoc3.9。两条命令都在浏览器启动阶段报：
`FATAL:chrome/browser/process_singleton_posix.cc:297 ... socket() failed: Operation not permitted (1)`。
没有关闭sandbox、改变系统安全设置或删减布局断言。因此不能声称本批真实36视口通过；请Normal Work在最终head补跑。

受限环境使用已合并的静态构建入口：

```bash
.venv/bin/python scripts/build_without_browser.py --baseline 9eeecd4137f7847c1c8c17708b53fa92584439e4
```

12张Mermaid和10张未变静态图均逐字匹配基准；第5章修改标签的SVG由CairoSVG重新栅格化。新产物通过资源/链接/目录校验，HTML与EPUB的BOS/EOS已核对为可见字面文本。首次构建抓到第6章末尾相对Markdown章节链接不能在合并读物访问，已改为普通章节阅读指引并重建。静态构建不替代浏览器布局实测。

改变的SVG使用Noto Sans CJK SC字体重新渲染并目视核对，避免本机默认Cairo字体中文方框。复现可设置Fontconfig将该字体优先匹配；本次命令附加 `FONTCONFIG_FILE=/tmp/p06-fonts.conf XDG_CACHE_HOME=/tmp/p06-fontcache`，未改系统字体或安全设置。

本批只提交主题分支与Draft PR，不推main、不自行合并、不提前实施下一批。
