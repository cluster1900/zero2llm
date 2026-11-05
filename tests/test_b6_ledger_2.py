"""B6（第二组）：正文测试账本移到 docs/testing.md 的回归检查。

被移动段落按原文保存在 MOVED 中：它们必须出现在 docs/testing.md，且不再出现在对应章节。
这些检查只锁定确切文本与少量标记词；改写后的同义账本句不会被自动发现。
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
 "08b_rope_yarn.md": [
  "另有 float64、部分旋转、不同 batch、请求重置、缓存分支与因果性测试。",
  "测试沿用仓库的稳定 2.14.x 版本闸门，**被跳过不算通过**。",
  "`tests/test_rope.py` 不只检查形状。它核对旋转范数、独立矩阵参考、相对位置恒等式、未旋转尾部、梯度有限性、float64 精度、缓存位置/因果 mask/内存/重置，以及训练反例的可复现指标。负例还故意重置新块 offset、取消旋转、把旧 K 转第二次，确认会被 logits 比较识别出来。规范正确性与模型质量是两套检查，两套都不能省。"
 ],
 "08c_modern_block.md": [
  "张量测试使用仓库共享的稳定版 torch 2.14.x 检查；未安装或版本不匹配会明确 skip，**skip 不是通过**。",
  "正常通过只能说明当前样例没发现问题。这里还在内存里编译几个错误版本，确认独立参考测试会失败；生产源码不会被改动。\n\n| 故意制造的错误 | 会被什么抓住 |\n| --- | --- |\n| RMS 分母误用居中后的方差 | 非零均值输入和独立均方参考 |\n| ε 加在平方根外 | ε=0.5 的精确手算表 |\n| SwiGLU 使用 sigmoid 而不是 SiLU | gate=-1、2 的符号与幅值 |\n| 门与内容相加而非相乘 | 两路投影的最终输出参考 |\n| 漏换最后的 ln_f，或忽略配置中的 ε | 三种配置下逐个检查 ln1/ln2/ln_f 的类型与 ε=1e-6 |\n| 消融学习率由 0.001 改成 0.002 | 表 P13-4 的八步 loss 快照，绝对容差 2e-5 |\n| 预算宽度平局时误选较大值 | D=2、H=4、有 bias 的等距例子及独立穷举参考 |\n\nFFN 中间宽度，以及表 P13-4 的每层 FFN 参数和整模型参数，也由实际运行结果核对。γ 初值全为 1 时，省略对应复制可能不改变结果，不能把这种等价变异声称为已发现错误；另用非全 1 的 γ 验证复制助手的值与非共享存储约定。\n\n还有 double 精度自动微分数值检查、全零与半精度大输入的有限反向、真实参数计数、显式复制后的数值一致性、S2 兼容模式及缓存对齐。"
 ],
 "08g_ced_architecture.md": [
  "测试按仓库稳定 torch 版本门槛跳过数值部分，**跳过不是通过**。",
  "因此测试同时覆盖：\n\n1. **独立标量 oracle**：用 Python 循环做二维乘法、逐位置 Softmax 和残差，不调用脚本的 attention、projection 或 cache helper；检查 full 和 bounded 两条路径。\n2. **手算结果**：直接断言 `[2,7/2,11/2]`、`[6,9,41/3]` 与 `[39/4,41/3]`。\n3. **未来隔离**：对每一个 prefix cut 改写之后的输入，前缀 encoder / global / 输出都不应变化；replay 也不能提前读 global bank 的未来行。\n4. **缓存等价**：每一种两块切分和逐 token 增量都与独立 full oracle 对齐，包含 offset、W=1、短 prompt、快照分叉和新请求。\n5. **近似差异**：多层 replay 的末位置以及后续 decode 确实可与完整路径不同；不把“不等价”误写成 bug。\n6. **错误变体**：故意把 global 来源接成当前 decoder h、取消局部窗口、放开未来、重置增量位置、把 replay 全局 query 位置重置为 0。必须被实际数值断言抓住，不能把变体运行崩溃当成“成功检测”。",
  "测试覆盖了这个边界。"
 ],
 "08h_moe.md": [
  "测试覆盖的不只是 full/cache 相同：\n\n- 固定路由权重、平局规则、负载、辅助损失和参数预算手算。\n- 独立逐标量 SwiGLU 与路由参考，对比输出、输入梯度和全部参数梯度。\n- 空 token、空专家、所有 token 集中到一个专家、top-1 和 top-E 边界。\n- 容量为零、完全丢弃、部分丢弃，以及共享分支仍保留。\n- 稀疏 dispatch 确实不调用空专家；梯度 None 与零的区别。\n- 多种 chunk 划分、单 token decode、未来 token 改动不影响前缀、batch 独立和请求重置。\n- bias、top-k、权重、dtype、位置参数改变后，旧 cache 被拒绝。\n- 故意改成均匀权重、全专家执行、把 bias 混进组合、共享输出取平均、丢弃后重归一化、后到先收等错误时，独立判据确实失败。",
  "数值测试明确跳过；**跳过不是通过**。",
  "我们的空专家测试只断言本次反向的直接梯度。"
 ],
 "08j_training_frameworks.md": [
  "测试故意恢复首位 BOS，要求独立更新对照报错，而不是只接受一个仍然下降的 loss。",
  "**跳过不等于 HF 通过。**",
  "测试覆盖这些非整除和空分片边界，也检查 D=1 时四种策略完全相同。"
 ],
 "08k_sft_lora.md": [
  "**skipped 不算训练验证成功。**",
  "### 本章的反例不是注释\n\n测试在内存中真正修改并运行 12 种语义错误：监督 user、漏掉 assistant EOS、误监督 role header、截断等号边界错误、错移 logits、错误 loss 分母、丢掉 LoRA 缩放、缩放漏除 rank、合并漏缩放、基础层未冻结、B 初始非零、移除 causal mask。每种都要求独立参考或明确的边界契约拒绝它，且不修改仓库文件。测试因此不是只证明“两个一起写错的实现彼此相同”。\n\n本次有 PyTorch 环境中 28 个测试通过；无 PyTorch 路线执行 13 个标准库测试，另外 15 个明确跳过。未来修改后应重新运行，不应把这些数字当成永远有效的标签。",
  "测试会确认内容中不能凭特殊标记的外观插入新角色。",
  "测试不只看第一轮，还逐轮确认：推理 prompt 必须与训练序列到该轮回答起点为止的前缀完全相同。",
  "测试把一整个 batch 的标签设为忽略值，验证参数未变、没有 backward，也没有 optimizer 更新。",
  "测试直接以这组手算数值为参考，另用 r=2、alpha=6 的独立矩阵检查“漏除 rank”错误，同时检查 A/B 一共 5 个可训练参数、W0 的 6 个参数被冻结。",
  "测试确实执行这两次 backward，验证这一变化。"
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


def count_mutants(path):
    """统计 mutant(...) 语义变异：字面量参数的调用计 1；在 for 循环里以变量为参数的调用，计所遍历列表的长度。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    total = 0
    for func in ast.walk(tree):
        if not isinstance(func, ast.FunctionDef):
            continue
        lists = {
            n.targets[0].id: n.value for n in ast.walk(func)
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.List)
        }
        for node in ast.walk(func):
            if isinstance(node, ast.For):
                it = lists.get(node.iter.id) if isinstance(node.iter, ast.Name) else node.iter
                calls = [c for c in ast.walk(node) if isinstance(c, ast.Call) and getattr(c.func, "id", None) == "mutant"]
                if isinstance(it, ast.List) and calls:
                    total += len(it.elts) * len(calls)
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "mutant" and node.args \
                    and all(isinstance(a, ast.Constant) for a in node.args):
                total += 1
    return total


def no_torch_run(pattern):
    code = (
        "import sys, os, unittest; sys.modules['torch'] = None; "
        f"s = unittest.defaultTestLoader.discover('tests', pattern={pattern!r}); "
        "r = unittest.TextTestRunner(stream=open(os.devnull, 'w')).run(s); "
        "print(r.testsRun, len(r.skipped), len(r.failures) + len(r.errors))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
        env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True,
    ).stdout.split()
    return tuple(map(int, out))


class B6LedgerMoveGroup2Test(unittest.TestCase):
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

    def test_08k_counts_derived_from_test_file(self):
        doc = TESTING.read_text(encoding="utf-8")
        path = ROOT / "tests" / "test_sft_lora.py"
        total = count_tests(path)
        ran, skipped, bad = no_torch_run("test_sft_lora.py")
        self.assertEqual((ran, bad), (total, 0))
        self.assertIn(f"本次有 PyTorch 环境中 {total} 个测试通过；无 PyTorch 路线执行 {total - skipped} 个标准库测试，"
                      f"另外 {skipped} 个明确跳过。", doc)
        self.assertIn(f"测试在内存中真正修改并运行 {count_mutants(path)} 种语义错误", doc)

    def test_ledger_markers_gone_from_edited_chapters(self):
        for name in MOVED:
            body = (CHAPTERS / name).read_text(encoding="utf-8")
            for marker in ("跳过不是通过", "被跳过不算通过", "skip 不是通过", "跳过不等于 HF 通过", "skipped 不算",
                           "个测试通过", "故意制造的错误", "错误变体", "种语义错误", "测试覆盖", "负例还故意",
                           "测试故意", "测试确实执行", "逐轮确认"):
                with self.subTest(chapter=name, marker=marker):
                    self.assertNotIn(marker, body)


if __name__ == "__main__":
    unittest.main()
