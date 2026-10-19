# P09–P11 第四批：训练基线与 KV Cache 验证

日期：2026-10-19。基于已合并第三批 main `d52d084243959a2fad9ffdea82049ee959cf57e9`，主题分支 `dot/batch4-p09-p11`。范围来自已审核的 #14 大纲 P09–P11，不包含 #28/#29 拟议扩展。上一批 [PR #4 独立审核](https://github.com/cluster1900/zero2llm/pull/4#pullrequestreview-5566442375)正常构建/36布局通过，**不视作本批新内容已通过浏览器测试**。

## 交付与阅读顺序

- [P09 训练基础](../chapters/07a_training_loop.md)：文档切分、shift、交叉熵手算、梯度与清零、评估边界、固定种子重复实验；[code/11](../code/11_training_loop.py)。
- [P10 Baby-GPT](../chapters/08_baby_transformer.md)：保留三首诗、全部字符ID表、原80步loss和生成对照、原三张流程图；S0配置与参数账、checkpoint；[code/07](../code/07_baby_gpt.py)。
- [P11 KV Cache](../chapters/08a_kv_cache.md)：同一S0参数的S1，矩形mask手算、位置偏移、逐层KV、容量/重置/批次边界、内存与计时；[code/12](../code/12_kv_cache.py)。

P09/P11通过07a/08a插入，无旧文件重命名。现在14个章节文件、12个编号示例；P12–P28仍未按新大纲实现。新单元用文字流程图，未改Mermaid定义或图像源码。

第三批5项非阻塞意见也已处理：原片段注释逐字对齐、省略明确；P08题答分开/中英文空格/类比边界；README加入标准库生成命令并区分历史9脚本；三组torch测试统一使用 `tests/torch_test_support.py`，skip原因包含实际版本与模块存在性。

## 最终检查

环境：Python3.12.14、torch2.14.1+cpu、NumPy2.5.3、Tokenizers0.21.4、CairoSVG2.8.2、Pandoc3.9、Node24.19.0。核心CPU运行单线程，无GPU、模型下载或付费API。

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
python3 -m unittest discover -s tests -v
node --test tests/
# 分别运行 code/01 到 code/12 的全部编号脚本
python3 code/11_training_loop.py --scalar-only
.venv/bin/python -m compileall -q code tests
python3 scripts/check_book.py
git diff --check
```

- 固定依赖环境：**118项全部通过，无skip**。
- 系统无torch环境：**50项通过、68项明确skip，无失败**；skip不是张量实验通过。
- Node：3/3。12个编号脚本全部退出0；P09标准库模式另跑通过。
- 新测试包括P09的14项、S0的9项、S1的14项（含1项标准库CLI检查）、共用门控2项、阅读/源码/数字账本5项；旧74项继续执行。
- `compileall`、`git diff --check`通过。dist/code与源文件逐字一致，HTML/EPUB资源、链接和目录验证通过。

## 数值与独立审校

P09两类别交叉熵：0.287682072、1.386294361，mean0.836988217；单题梯度[-0.25,0.25]，SGD更新后loss0.275414489。有限差分与autograd另行核对，mean/sum缩放和梯度累加均有测试。

40次CPU更新的整首文档评估：

| seed | 训练 loss 前/后 | 留出 loss 前/后 |
| --- | --- | --- |
| 7 | 4.219313 / 3.175501 | 4.494203 / 4.843005 |
| 7重跑 | 4.219313 / 3.175501 | 4.494203 / 4.843005 |
| 11 | 4.189975 / 3.263426 | 3.923873 / 4.397414 |

留出23个目标中，只有4个标点目标在训练标签中出现，19个目标未见过。共有组loss下降、未见组loss上升；总体上升主要来自标签分布差异，本实验不能证明过拟合。词表包含一个字，不等于训练曾把它作为正确答案。留出只有一首诗，字符词表预先包含全语料，是明确披露的transductive设置，不声称严格独立tokenizer或通用泛化结论。

| seed | 共有组计数 / loss 前→后 | 未见组计数 / loss 前→后 |
| --- | --- | --- |
| 7 | 4 / 4.266185 → 3.506729 | 19 / 4.542207 → 5.124327 |
| 11 | 4 / 4.072877 → 3.441714 | 19 / 3.892503 → 4.598613 |

按实际训练文档的 `ids[1:]` 定义“见过”，不是按整份词表。表中seed11总体终值4.397414采用Normal Work复审参考raw4.397413730；本机raw4.3974132538相差约一个float32 ULP，会舍入为4.397413。固定回归使用绝对容差1e-6以容纳这一环境差异，不能把差异冒充完全一致。

S0仍是67872参数、57字符、三首各24字符，69个独立诗内next-token目标。80步原loss日志4.1271/0.0196/0.0021/0.0011/0.0009及原生成样例全部复现。后来仅收窄“明月”失败案例的解释措辞，不改变数值；一条输出不能判定是否理解。checkpoint验证CPU float32/float64全logits round-trip，不包含完整续训状态。

S1矩形mask P=3/C=2：输出30与36.666667；错误tril输出10与16.666667。默认B1/T48/chunk7完整与增量最大logits误差3.58e-7，KV载荷36864字节；T5为3840字节。测试容差atol=3e-6、rtol=1e-5；比较全部logits而非仅argmax。

发布前独立审校复算了本批18道自测、参数/损失/生成/内存数值及额外B3/H3/L3/D24/T128不规则分块：float32最大误差4.77e-7、float64为8.88e-16。审校发现仅依赖parameter版本counter会漏掉 `load_state_dict(assign=True)` 的参数替换，现已加入identity/version/dtype/device签名及回归；原复现用例明确拒绝旧cache，重置后最大误差0.0。`.data`等绕过版本跟踪的低层篡改不受支持，不引入昂贵权重hash作虚假全覆盖。

## 变异与错误路径

仓库测试含实际负面对照：错误shift/反转标签、训练文档混入留出、遗漏zero_grad造成更新偏差、删除因果mask、错误位置offset、错误矩形mask。不是通过搜源码一行存在就声称正确。

另在内存编译修改后的模块并运行原正常测试（生产文件不改），实测四种变异都产生**断言失败，零错误、零skip**：

1. S1位置改为从0开始 → chunk/full对齐失败（3个subtest）。
2. S1矩形mask改为从本块局部0开始 → 同上（3个subtest）。
3. S0去掉因果mask赋值 → 前缀因果性测试失败。
4. S0交叉熵target改成当前输入idx → 独立loss参考失败。

新增stale-cache回归先构造版本counter相同的两套权重，再assign替换，验证必须重置；同时测不合法层数/pair/rank/head维度、dtype/device和请求batch。每个缓存张量约定只读，frozen容器不深度冻结Tensor。

## 计时，不承诺加速

下表来自最终CPU运行：默认48宽/4头/2层，随机权重同一模型、固定seed42，prompt16+后续32，float32，单线程，warmup2/repeats7，交替顺序。单位ms。first只计提示前向，later为每轮后续前向均值的跨轮统计；不包含分词/采样/网络，不是服务TTFT或尾延迟。

| 路径/阶段 | min | median | max |
| --- | --- | --- | --- |
| S0 first | 0.419469 | 0.509491 | 0.664740 |
| S1 first | 0.613284 | 0.708494 | 0.945494 |
| S0 later | 0.535553 | 0.565541 | 1.082210 |
| S1 later | 0.563479 | 0.598601 | 0.617969 |

本轮S1中位耗时更高。Python校验、拼接和小矩阵开销可抵消省下的计算；不选择最好一次当普遍加速结论，也不设墙钟性能断言。无GPU实测。

## 构建与浏览器边界

实际执行正常Chromium构建与 `npm run check:layout`，两者都在启动时报：
`FATAL:chrome/browser/process_singleton_posix.cc:297 ... socket() failed: Operation not permitted (1)`。
没有关闭sandbox、改安全设置或改布局脚本。新增两单元后应由Normal Work对最终head补跑全部**42个chapter/viewport组合**，不能沿用上批36组合。

使用已有受限构建入口：

```bash
# PATH先指向Pandoc3.9
.venv/bin/python scripts/build_without_browser.py --baseline d52d084243959a2fad9ffdea82049ee959cf57e9
```

12张Mermaid与11张静态图均源码逐字匹配基准，复用既有图，不增加新图；新正文与代码重新编译。静态检查通过12流程图/23插图，所有本地资源/内部链接/目录完整，新章节顺序和特殊token字面显示另核对。此结果不替代真实浏览器布局验收。

来源独立核对：PyTorch2.14 CrossEntropyLoss/zero_grad/autograd/GELU/save/load，Transformers4.57.1缓存与生成文档，Chinchilla及PagedAttention原论文。动态入口明确只作导航；不把它们的API或大模型测量声称为本教学实现结果。

## PR #5 复审修正（同一主题分支）

Normal Work 在 [review 5566607790](https://github.com/cluster1900/zero2llm/pull/5#pullrequestreview-5566607790) 对旧head `2ec4158` 正常构建和42布局全部通过，同时指出P09解释/测试不足。上述最终结果已更新至返修，不用旧head浏览器结果替代新head验收。

- 删去“与过拟合一致”的错误解读，分层实际训练标签：4个共有目标下降、19个未见目标上升。新输出给出count/loss_sum/mean、实际lr，测试核对加权重组；AB/BA小反例确保“见过”依据y标签而非全部输入字符。
- `evaluate`的真实forward pre-hook记录 `training=False`、`is_grad_enabled=False`；不等长1/4个目标的控制模型检查 `(1×1+4×4)/5=3.4`，错误文档平均为2.5。
- 固定seed7/11、40步及lr=.003数值快照，绝对误差1e-6；明确seed11参考4.397414与本机1ULP差，未伪造本地舍入结果。
- 追加5种P09源码内存变异：去eval、去no_grad、文档平均、去显式lr、全输入字符误当训练标签，均运行原行为测试产生断言失败、0errors/skip。
- 缓存生成新增真实前向调用计数：预算0/1/2/5分别0/1/2/5次；多算最终token和删除dtype/deviceguard两种变异均产生断言失败、不是RuntimeError冒充拦截。无torch CLI友好提示并退出2，无traceback；模拟缺依赖的测试也在有torch环境运行。
- 第7章下一单元文字更新为P09；README标题12脚本并补第四批链接；插入单元统一P09.x/P11.x节号；S15补vLLM导航并明确非运行依赖。

原数表、模型架构与缓存计算语义不因这些修正改变。仍需Normal Work对更新head补跑正常构建及42布局后复审。
