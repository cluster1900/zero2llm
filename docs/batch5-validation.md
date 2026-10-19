# 第五批 P12–P14 验证记录

核验日期：2026-10-19。基准 `4a585a660f4c0345b8be59d8b43aef74c83b27b1`，独立分支 `dot/batch5-p12-p14`；不直接推送 main、不自动合并。

## 范围与教学边界

- P12 / S2：在 S1 的模型与显式缓存思路上，移除可学习绝对位置，用 Q/K 的相邻二维旋转实现普通 RoPE。YaRN 为原始论文机制讲解，不宣称完整实现 YaRN 的频率策略、温度或长上下文训练。
- P13 / S3：保留 S2 注意力与缓存，逐项比较 LayerNorm / RMSNorm 和 GELU / SwiGLU。先声明 bias 政策，再核对中间宽度和参数预算；不把玩具 loss 当通用质量排名。
- P14 / S4：真实减少 K/V 投影宽度和持久缓存头数；MHA/MQA 为 GQA 边界。MLA 独立小矩阵演示低秩内容和解耦位置项，不是完整模型、权重转换器或生产 kernel。
- 原 P09 非阻塞回归：`evaluate_strata` 真实 forward hook 检查 `training=False` 与 `is_grad_enabled=False`，并核对原 train/eval 状态恢复。去 eval、去 no_grad 的两个实际变异分别导致断言失败，零 errors。
- 新正文以 `08b/08c/08d` 插入；旧文件链接保留。没有加入未批准的 #28/#29 扩展。

## 环境与命令

本机 Python 3.12.14、PyTorch 2.14.1+cpu、tokenizers 0.21.4、CairoSVG 2.8.2、Pandoc 3.9、Node 24.19.0；CPU 实验单线程，不下载模型权重。系统 Python 无 torch，用于验证依赖跳过。

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
python3 -m unittest discover -s tests -v
node --test tests/
OMP_NUM_THREADS=1 .venv/bin/python code/13_rope.py
OMP_NUM_THREADS=1 .venv/bin/python code/14_modern_block.py
OMP_NUM_THREADS=1 .venv/bin/python code/15_grouped_kv.py
.venv/bin/python build.py
npm run check:layout
# 只有正常浏览器被环境限制时，才使用已有静态渲染路径：
.venv/bin/python scripts/build_without_browser.py --baseline 4a585a660f4c0345b8be59d8b43aef74c83b27b1
.venv/bin/python scripts/check_book.py
git diff --check
```

## 最终验证结果

- PyTorch 环境：**182/182 通过，0 跳过**。其中 P12 17 项、P13 20 项、P14 22 项；包含独立参考公式、训练梯度、全量/分块 logits、紧凑缓存、参数预算与手算回归。
- 系统 Python 无 torch：**182 项 OK，59 通过、123 明确跳过**。修复了测试 helper 默认参数在收集阶段提前引用 torch 的问题；没有把 skip 算作功能通过。
- Node：**3/3 通过**。15 个编号 CPU 脚本全部退出 0；`compileall` 与 `git diff --check` 通过。缺 torch 的新脚本均提示安装并退出 2，无 traceback。
- **14 个额外定向源码变异**在内存中执行，不改生产文件：P09 去 eval / no_grad；P12 旋转符号、频率指数、缓存 offset、缓存因果 mask；P13 错误中心化、epsilon 放根号外、sigmoid 替换 SiLU、门控乘法改加法；P14 错误组映射、offset、矩形 mask、持久缓存错误展开。全部被相应行为测试断言拦下，零 errors。章节测试中另保留负面对照；此处不将其重复计数。
- HTML/EPUB 静态构建与 `check_book.py` **通过**：17 个章节、12 张流程图、26 张插图，资源、内部链接、目录锚点完整；`dist/code` 的 16 个 Python 文件与源码逐字节相同。
- 三张新增 SVG 均实际栅格化查看；中文字体完整、箭头和标签无重叠。P14 曾发现 Unicode 上下标缺字，已改为清楚的 ASCII 标签并重新查看，最终图源与产物匹配。
- 新章节显式 ID 暴露原 HTML 构建器覆盖 ID 的问题：现在保留明确 ID、从目录文字中去除 Pandoc 属性，同时保留旧章节 slug；新增回归与实际 HTML/EPUB 锚点校验均通过。
- **正常浏览器构建/响应式布局未通过，不能声称验收完成**：默认 Puppeteer 缺少 Chrome 131；改用已安装 `/usr/bin/chromium` 的正常启动后，构建与布局均报 `FATAL:chrome/browser/process_singleton_posix.cc:297 ... socket() failed: Operation not permitted (1)`。没有关闭 sandbox、修改安全设置或降低布局断言。
- 使用既有静态构建器：12 张 Mermaid 仅在源码逐字匹配基准时复用，11 张旧静态图复用，3 张新 SVG 用 CairoSVG 2.8.2 + 已安装 Noto Sans CJK SC 栅格化。此路径不是浏览器布局验证；请 Normal Work 在本批新 head 正常构建并检查 **51 个章节/视口组合**。
- 上一批 PR5 的 118 tests / 42 布局通过仅属于上一批，不作为本批结果。独立教学/数值审校已覆盖本批三章，不代替新的浏览器验收。

## 可复跑的数值摘要

### P12：短序列训练后的长度边界

最终 QUERY 位置的检索任务，训练长度 8、400 步、每步 32 个样本、AdamW lr=.003；测试每种长度 512 个样本。不是语言模型每 token perplexity，也不隔离“距离”与“干扰项数量”。所有目标类别在训练中出现，避免未见标签混淆。

| 种子 | T=8 正确数 | T=16 | T=32 | T=64 |
| --- | ---: | ---: | ---: | ---: |
| 2025 | 512 | 351 | 247 | 390 |
| 2026 | 512 | 226 | 170 | 364 |
| 2027 | 512 | 252 | 170 | 203 |

种子 2026 对应 mean CE 为 0.001046、3.318326、3.937189、1.160583。完整表和 `2e-5` CPU 快照容差见正文/测试。外推损失和准确率不是单调关系；没有据此宣称普通 RoPE 永远不能外推或 YaRN 已被验证。相同参数的完整/分块 logits 示例最大差 2.38e-7，缓存 6144 字节。

### P13：先控制变量，再比较部件

A 为独立声明的无 bias LN/GELU 控制，不冒充未改 S2。A→B 仅替换归一化；B→C 替换 FFN 形式及为匹配预算所需的宽度。不变权重显式复制，各自新建同配置 optimizer，同一合成标签 batch。

| 配置 | FFN 宽度 | 每层 FFN 参数 | 总参数 | 8 步前→后 loss |
| --- | ---: | ---: | ---: | --- |
| A LN/GELU | 192 | 18432 | 61008 | 4.429393→2.889954 |
| B RMS/GELU | 192 | 18432 | 61008 | 4.441514→2.896931 |
| C RMS/SwiGLU | 128 | 18432 | 61008 | 4.403015→3.212187 |

三个梯度检查均有限，缓存最大差不超过 5.36e-7。单种子、单重复 batch 只是训练 smoke，不是泛化质量或速度排名。

### P14：减少的是真实 KV 载荷

D=48、Hq=4、d=12、L=2、B=2、T=12、float32：

| Hkv | K/V 投影参数合计 | 实际 K/V 缓存字节 |
| --- | ---: | ---: |
| 4 | 9216 | 18432 |
| 2 | 4608 | 9216 |
| 1 | 2304 | 4608 |

缓存按 Hkv 保存，不把临时 repeat 后的 Hq 张量塞回缓存；完整/分块 logits 最大差不超过 4.77e-7。独立 float64 MLA 小矩阵的 score/output 差分别不超过 1.12e-16 / 4.45e-16；含位置键的压缩/展开载荷为 96 / 240 字节。该恒等式只对构造的低秩权重成立；rank 3→2 反例的 Frobenius 误差为 1，不能声称任意 MHA 可无损压缩。

## 原始出处

核对 RoFormer v5 §3.2、YaRN v2 §3.1–3.4、RMSNorm 原论文、GLU Variants 原论文、GQA 原论文与 DeepSeek-V3 官方仓库/技术报告。完整链接、版本和各项主张见对应章节。动态文档若出现，标明日期与非运行依赖边界。
