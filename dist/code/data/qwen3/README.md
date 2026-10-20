# P25 的小型官方快照

核对日期：2026-10-19。这里只保存两份明确 checkpoint 的 config.json、tokenizer_config.json、generation_config.json 与原始 LICENSE，所有文件保留下载字节，不改写官方内容。没有模型权重、tokenizer.json、vocab.json 或 merges.txt。

- Qwen/Qwen3-0.6B：c1899de289a04d12100db370d81485cdf75e47ca。
- Qwen/Qwen3-30B-A3B：ad44e777bcd18fa416d9da3bd8f70d33ebb85d39。
- 上述官方文件由 Qwen 发布，使用随文件保留的 Apache License 2.0。两个目录各自保留完整 LICENSE。
- manifest.json 记录每个文件的不可变来源 URL、字节数与 SHA-256；26_qwen3_case.py 在使用前校验这些值。文件损坏是错误，不静默联网更新。
- 实现解释核对 Hugging Face Transformers v4.51.3，commit 5f4ecf2d9f867a1255131d2461d75793c0cf1db2。manifest 同时保存两份已核对实现源文件的 URL、字节数与 SHA-256，但没有将该库实现复制进本仓库，也不要求安装 Transformers。
- 这里的模板可用已锁定的 Jinja2 3.1.6 做单轮文本渲染；缺少完整 BPE 文件，所以不提供 Qwen encode/decode，也不能把字符数当作 token 数。

重新取证时应新建版本记录，逐文件从 manifest 中的完整 URL 下载并核对 SHA-256；不能把 URL 的 revision 换成 main 后继续沿用本章的数字。正文和测试中的数值针对当前快照，不代表整个 Qwen3 家族。
