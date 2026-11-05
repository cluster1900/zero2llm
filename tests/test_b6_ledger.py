"""B6（第一组）：正文测试账本移到 docs/testing.md 的回归检查。

被移动段落按原文保存在 MOVED 中：它们必须出现在 docs/testing.md，且不再出现在对应章节。
"""
import ast
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAPTERS = ROOT / "chapters"
TESTING = ROOT / "docs" / "testing.md"
RUN = "运行本章测试文件，末尾应显示 `OK`；测试分类、计数与错误版本清单见 `docs/testing.md`。"
MOVED = {
 "08d_grouped_kv_mla.md": [
  "测试覆盖四类问题：\n\n- **算对了什么**：独立逐头参考、手算映射、MHA/MQA 边界、部分 RoPE、同权重全量/单 token/多种分块对齐、有限梯度。\n- **真的存少了吗**：K/V 投影宽度、缓存头数、tensor 字节与底层存储字节，而不是只检查打印的公式。\n- **哪些输入会被拒绝**：非法头数、空序列、错误 dtype、越界 token、跨模型缓存、长度溢出、坏缓存形状、权重/位置设置改变。\n- **错误实现能被抓住吗**：GQA 检查错误的重复映射、RoPE offset、矩形 mask 和扩展后缓存。MLA 再故意把两条路径的缩放一起改成 √r、一起删除位置打分项，或让位置 key 不旋转；每个变体虽然仍能通过内部两路径对齐，却必须被独立参考判为 assertion failure，且运行错误数为 0。这样不会把坏导入或异常崩溃误报为“抓住数值错误”。源码文件不会被这些 mutation 测试改写。\n\n本章共 23 项测试。没有安装兼容的 torch 2.14.x 时，数值类按共享测试门控明确跳过；缺依赖的 CLI 给出安装提示。**skip 不是通过，不能靠只有 CLI 检查成功来宣称 GQA 数值实现已验证。**"
 ],
 "08e_attention_systems.md": [
  "已经安装本仓库依赖时，测试会额外用 CPU SDPA 做一个独立交叉检查；兼容性按仓库共享规则要求稳定版 torch 2.14.x。不满足条件只跳过这一项，**不会跳过在线 Softmax、分页和时间线的必修测试**。",
  "本章当前 26 项测试：在兼容 torch 环境中 26 项通过；无 torch 环境中 25 项通过、1 项明确 skip。无 torch 的正常 CLI 仍能完整运行，显式请求 GPU 选修时会解释跳过原因。**被跳过的 GPU 或 SDPA 检查不算数值通过。**\n\n除了正例，测试还在内存中构造 12 个错误版本：忘记重缩放、两条路径一起丢缩放分母、忽略 mask、丢最后一块、不做 COW、COW 后不减引用计数、共享时不加计数、按物理块号排序、TPOT 除以 n、丢掉旧块最大值、TTFT/E2E 不减到达时间。每个契约先在正确实现上通过，再要求错误版本被断言拒绝。对于丢旧最大值的版本，合法有限输入（分数1000再到0）会触发 OverflowError；测试明确将这种稳定性违约转成断言失败，不能把它隐瞒成“没有数值异常”。坏导入、语法错误或其他无关异常仍会让变异测试失败。源文件不会被这些 mutation 改写。\n\n分页测试还有 240 步固定种子的随机 create/append/fork/prefix/release，对照一个独立的普通序列模型，并逐步检查引用计数和空闲池。它检验的是这个单线程模拟器，不是生产系统并发安全。"
 ],
 "08f_sparse_linear_attention.md": [
  "21 项测试包含：手算 mask、矩形 offset、W=1/W≥T、层间感受野、全局位置不泄漏未来、手算 S/z、float32/64、T=1/512、非法长度/类型、全遮罩行、下溢/溢出、有限梯度与梯度对齐、改写未来不影响过去、输入不被改写、请求间不残留状态、检索闭式公式及固定状态存储。\n\n另在内存中生成五种错误版本：窗口边界多放进一位；全局位置越过因果限制；线性矩阵漏掉三角 mask；两条路径一起漏掉归一化；递推把当前 value 错换成前一位。每个都必须被相应断言抓住，而不是靠导入失败或异常崩溃假装“检出了错误”。这些测试不改写源码。\n\n无兼容 torch 2.14.x 时，1 项缺依赖 CLI/导入契约测试仍可运行，其余 20 项按共享门控明确跳过。**skip 不是通过**；安装依赖后才能声称数值行为已经验证。"
 ],
 "08m_reasoning_mtp.md": [
  "测试按仓库的兼容版本闸门友好跳过，**跳过不是通过**。",
  "独立测试另有一个“完美草稿”的标量目标：生成 10 个 token，target 调用从 10 次变成 4 次，处理位置都为 11，且有两个 bonus。它证明计数与流程按规则工作，不证明神经模型能够免费拿到完美草稿。",
  "测试不只让两个共享代码的函数互相证明。它使用 Python 标量标签 oracle、显式 log-sum-exp、解析 logits 梯度与权重有限差分、独立 token-history 目标、真实 S5 完整前缀 logits／逐层 K/V 对照，以及不调用 `mtp_logits` 的逐 head 标量投影 oracle；还实际执行十二种错误变体，确认错位标签、错误分母、错误接受、缓存偏一、漏停 EOS、越过预算、跳过主 head、漏掉 correction 等会被拦住。最后一种由 `test_deleted_correction_mutant_fails_fast_without_timeout` 验证精确的 `RuntimeError`，不是把测试超时算作成功。"
 ],
 "08i_pretraining_pipeline.md": [
  "测试的 tensor 部分明确跳过，**skip 不算训练验证通过**。",
  "| 回归测试 | 17 项通过；无 torch 环境为 8 通过、9 跳过 | 标准库与 tensor 边界分别验证 |",
  "### 让坏版本真正跑一下\n\n`tests/test_pretraining_pipeline.py` 不只检查函数是否返回某个标志。每组先验证正确路径，再让有缺陷的计算产生可观察的差异：\n\n| 真实错误变体 | 如何检测 |\n| --- | --- |\n| 将文档 mask 换成普通 causal 下三角 | 改 A 的内容后 B 的 logits 变化；独立标量 attention oracle 不符 |\n| 整箱直接移位标签 | A 的 EOS 开始预测 B 的 BOS；目标数与 loss 改变 |\n| 令 labels 等于当前输入 | 不会预测未来的“抄写器”获得假低 loss |\n| 平均不等长块的均值 | 2 与 5 个目标的独立标量总和不符 |\n| 忽略全 PAD，但仍调用 AdamW step | 已建立的优化器状态推动参数继续变化 |\n| 恢复权重，却遗漏 optimizer | 后续 loss 轨迹与最终参数不同 |\n| 忽略 dropout RNG、shuffle RNG 或 cursor 中任一项 | 各自单独导致后续 loss 与最终参数不同 |\n\n最后一组在第 5 步暂停：已经发生过一次 epoch 洗牌，且 cursor 位于中间。如果在第一次洗牌之前测试“漏掉 shuffle RNG”，新建训练器的 RNG 可能恰好等于保存状态，错误版本和正确版本会暂时等价。**反例必须真的触发它声称检测的错误，不能只让测试名字看起来严格。**",
  "把 mask 故意换成普通下三角后，后一条检查确实会失败。",
  "测试用 2 个目标和 5 个目标的两块检查加权方式，故意平均块均值会给出不同结果。",
  "故意仍然执行 AdamW step 的坏版本确实改变权重。"
 ],
 "09a_minigpt_healthcheck.md": [
  "测试可以因没有受支持 Torch 而跳过数值路径，但**跳过不等于通过**。其他导入错误、错误实现崩溃或输出 NaN 不能被包装成“负例成功”。",
  "测试另外只破坏第一块或中间块的输出，保留正确 K/V 和最后一块，确认完整比较会报警；把检查器改成只看最后一块的真实源码变异则会漏过，因此该错误不能再被正常输出样本掩盖。",
  "正常体检必须拒绝，删掉对应检查的源码变异则会放过它，其他检查保持原样。",
  "相应测试先检查正常路径，再观察坏数据确实被拒绝；移除对应守卫的源码变异会让坏文件被写出或被接受。",
  "“显著超出容差且仍为有限输出”才是测试契约。测试还反向验证：如果变异器根本没改源码，负例检查必须失败；如果缺依赖而抛异常，也必须继续报错，不能算成功捕获数学错误。 数值反例必须产生有限的错误输出；dtype、owner、反序列化与梯度守卫则检查具体拒绝原因，并独立观察删掉守卫后被放过的坏状态。这两类证据不混算。",
  "此外，测试对真实模型取消因果 mask，对旋转符号、RMS 统计、门控、GQA 映射、checkpoint 元数据检查、消融权重与 MTP 标签做实际行为变异。作用是验证这些 oracle 真能区分好坏，不是增加几个永远为真的断言。",
  "测试通过真实 backward hook 分别制造全零／NaN 梯度，或仅 detach 主 head、旁路 head 的权重，验证有限 loss 不会掩盖断开的学习路径；同时保留其他正常路径，避免把无关崩溃当成守卫有效。",
  "测试让缓存路径的每块 logits 都加 0.25，四种分块都应被真正的完整前向抓住；缓存路径与自己比较却会漏掉。另一个测试只把第一块乘 1.5，包括整段 `[8]` 在内也都应报警。这里没有改 K/V，错误只在返回的 logits，不能依靠后续缓存崩溃碰巧抓住它。",
  "兼容性测试还区分拒绝原因：一个文件若先因不支持的 dtype 被拒绝，不能把这次异常算作“架构语义不兼容已验证”。"
 ]
}


def count_tests(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(
        1
        for cls in tree.body if isinstance(cls, ast.ClassDef)
        for f in cls.body
        if isinstance(f, ast.FunctionDef) and f.name.startswith("test_")
    )


class B6LedgerMoveTest(unittest.TestCase):
    def test_moved_text_lives_only_in_docs_testing(self):
        doc = TESTING.read_text(encoding="utf-8")
        for name, paras in MOVED.items():
            body = (CHAPTERS / name).read_text(encoding="utf-8")
            self.assertIn("## " + name, doc)
            for p in paras:
                with self.subTest(chapter=name, para=p[:30]):
                    self.assertIn(p, doc)
                    self.assertNotIn(p, body)

    def test_each_chapter_keeps_run_command_and_expectation(self):
        for name in MOVED:
            body = (CHAPTERS / name).read_text(encoding="utf-8")
            with self.subTest(chapter=name):
                self.assertEqual(body.count(RUN), 1)
                m = re.search(r"unittest discover -s tests -p '?(test_\w+\.py)'?", body)
                self.assertIsNotNone(m)
                self.assertTrue((ROOT / "tests" / m.group(1)).is_file())

    def test_docs_counts_match_test_files(self):
        doc = TESTING.read_text(encoding="utf-8")
        for phrase, fname in [
            ("本章共 {} 项测试", "test_grouped_kv.py"),
            ("本章当前 {} 项测试", "test_attention_systems.py"),
            ("{} 项测试包含", "test_sparse_linear_attention.py"),
        ]:
            with self.subTest(file=fname):
                self.assertIn(phrase.format(count_tests(ROOT / "tests" / fname)), doc)

    def test_ledger_markers_gone_from_edited_chapters(self):
        for name in MOVED:
            body = (CHAPTERS / name).read_text(encoding="utf-8")
            for marker in ("跳过不是通过", "跳过不等于通过", "skip 不是通过", "skip 不算", "项测试", "240 步", "源码变异", "错误变体", "错误版本：", "个错误版本", "让坏版本真正跑一下"):
                with self.subTest(chapter=name, marker=marker):
                    self.assertNotIn(marker, body)


    def test_08i_regression_row_counts_derived_from_test_file(self):
        doc = TESTING.read_text(encoding="utf-8")
        body = (CHAPTERS / "08i_pretraining_pipeline.md").read_text(encoding="utf-8")
        self.assertNotIn("16 项通过；无 torch 环境为 7 通过、9 跳过", body)
        total = count_tests(ROOT / "tests" / "test_pretraining_pipeline.py")
        code = (
            "import sys, unittest; sys.modules['torch'] = None; "
            "s = unittest.defaultTestLoader.discover('tests', pattern='test_pretraining_pipeline.py'); "
            "r = unittest.TextTestRunner(stream=open(__import__('os').devnull, 'w')).run(s); "
            "print(r.testsRun, len(r.skipped), len(r.failures) + len(r.errors))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
            env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True,
        ).stdout.split()
        ran, skipped, bad = map(int, out)
        self.assertEqual((ran, bad), (total, 0))
        self.assertIn(f"| 回归测试 | {total} 项通过；无 torch 环境为 {total - skipped} 通过、{skipped} 跳过 |", doc)
        for piece in ("正文原写「16 项通过；无 torch 环境为 7 通过、9 跳过」", "0eacd0f92", "559dc57dd",
                      "`test_nfc_canonical_text_and_deduplication`", "9c4c9e3",
                      f"现已改为 {total} / {total - skipped} / {skipped}"):
            self.assertIn(piece, doc)


if __name__ == "__main__":
    unittest.main()
