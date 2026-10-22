# P27–P28：读者复现与结果解释

固定前置基线：zero2llm `69e5c87ed4b5762a89127ab395738f688fac6299`（P24–P26 案例基线）。

## 复现步骤

在仓库根目录运行：

```bash
OMP_NUM_THREADS=1 python code/28_minigpt_healthcheck.py --json > healthcheck.json
OMP_NUM_THREADS=1 python code/28_minigpt_healthcheck.py --mtp --json > healthcheck-mtp.json
python scripts/plot_p27_healthcheck.py healthcheck-mtp.json --out-dir assets/diagrams/p27
python -m unittest discover -s tests -v
```

两个 JSON 都是对象，顶层字段包含 `environment`、`stages`、`components`、`compatibility`、`negatives`、`timing` 和 `scope`。默认报告有 S0–S5；加 `--mtp` 后增加 S6。每个有缓存的阶段同时包含 `chunk_checks`、`cache_error`、`kv_bytes`；P27 教学配置的 MHA 对照为 4096 bytes，GQA 对照为 2048 bytes。三种负例的 `caught` 必须都是 `true`，并且 `max_logit_error` 是有限正数。

MTP 报告的稳定计数为 `label_counts=[7, 3]`；`greedy_equal=true` 表示 draft/verify 与独立完整前缀贪心结果一致。缓存与因果误差使用脚本写出的 `atol=3e-6`、`rtol=1e-5` 解释，具体末位随 CPU BLAS 与线程设置变化；报告保留每个 chunk 的最大误差，不用单个摘要数字掩盖分块差异。

绘图命令只接受这份 JSON 的结构。它会写入：

- `assets/diagrams/p27/cache_causal.svg`：S0–S5（或 S6）的缓存/因果误差对数柱；
- `assets/diagrams/p27/negative_controls.svg`：三个负例的 before/after 数值柱；
- `assets/diagrams/p27/cpu_timing.svg`：full/cache 与 first/later 的 median，误差线是 min–max。

三张图的 SVG `desc` 都记录 `DATA_SOURCE=code/28_minigpt_healthcheck.py --json`。缺少报告字段时渲染器以非零状态退出并且不写出半成品图。

## P25 口径边界

P25.9 把练习的首闭合记账位置与固定 revision 的模型卡最后闭合示例并列；同一节也给出 chat template 的第一个/最后一个闭合差异。这个对照帮助读者复算标签边界，不把教学选择改写成官方实现结论。

## 可选依赖的结果

没有 PyTorch 时，两个 healthcheck CLI 以状态码 2 输出 `NOT RUN, not passed`；这表示数值路径没有运行。标准库章节测试仍可运行，跳过的官方参考测试会明确显示为 skipped。安装依赖后重新执行上面的命令，不能把旧 JSON 或旧图复制到新代码。

## 结果解释

`environment` 记录 Python、Torch、线程、随机种子、dtype 和 CPU；`timing.milliseconds` 的 min/median/max 只描述当前输入与机器。这个实验验证小型随机模型的形状、缓存、因果和反例检查，不提供预训练质量、GPU 吞吐或生产服务结论。
