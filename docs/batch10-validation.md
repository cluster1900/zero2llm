# P27–P28：综合体检与参考附录验证

## 读者可复现范围

P27 的入口是 `code/28_minigpt_healthcheck.py`；P28 是参考附录与数学自测。本页只保留读者需要的复现步骤、输入约束和结果解释。合并提交、过程时间和发布状态另见 Release notes。

复现步骤：

```bash
python code/28_minigpt_healthcheck.py --json
python code/28_minigpt_healthcheck.py --mtp --json
python -m unittest discover -s tests -v
```

预期输出是 JSON 对象，包含 environment、stages、causal_checks、chunk_checks、components、compatibility、negatives、timing 和 scope。S0–S5 会报告缓存/因果误差与 KV payload；加 `--mtp` 才出现 S6 的标签、梯度和贪心检查。没有 PyTorch 时 CLI 应退出码 2 并写明 NOT RUN，不把跳过当通过。

## P25 口径边界

本练习把首闭合位置作为记账约定，官方模型卡采用最后闭合位置；两种口径必须在同一段中明确区分。

## 复现前提

- 使用仓库固定的代码、模型配置和随机种子；
- 记录 Python、Torch、NumPy、可选 HF/llama.cpp 版本；
- 真实外部引擎不可用时标为跳过；
- 图表输入、参数、输出和命令应一起保存；
- 随机 tiny 模型与 CPU 结果只支持教学正确性，不外推模型质量或 GPU 收益。

## P27 输出检查

1. S0–S6 的整段/增量 logits 误差与全部 causal cut；
2. 实际 KV payload 字节、位置/掩码和请求隔离；
3. 三个负例的 caught 状态与 max_logit_error；
4. P27.10 CPU timing 中 full/cache、first/later 的中位数与 min/max。

每张数据图均由当前 JSON 生成，并在图注中写入输入路径、版本和采样范围。

## P28 速查附录

D–G 每节各有公式与手算例或速查图；旧锚点保留，新锚点唯一。模型卡、自测和章节小结各保留一个权威位置。

## 结果解释

`scope` 字段明确本实验是 CPU toy correctness/timing；`timing` 的毫秒值只适用于当前机器、线程和随机输入。完整报告可用于复核字段，但不能替代真实 GPU、生产服务或完整训练验收。
