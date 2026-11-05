"""Machine-checkable factual corrections (bot-hub#123 B3).

Each test pins the corrected sentence positively and, where possible, ties the
stated fact to the code, CI config or data it describes.
"""
import ast
import pathlib
import re
import unicodedata
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CH = ROOT / "chapters"


def read(name):
    return (CH / name).read_text(encoding="utf-8")


def code(name):
    return (ROOT / "code" / name).read_text(encoding="utf-8")


def func_src(src, name):
    tree = ast.parse(src)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(src, node)


def count_test_files():
    return len(list((ROOT / "tests").glob("test_*.py")))


def count_test_methods():
    n = 0
    for p in (ROOT / "tests").glob("test_*.py"):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                n += sum(1 for b in node.body
                         if isinstance(b, ast.FunctionDef) and b.name.startswith("test"))
    return n


class Preface(unittest.TestCase):
    def setUp(self):
        self.pre = read("00_preface.md")

    def test_discover_sentence(self):
        hundreds = count_test_methods() // 100 * 100
        self.assertGreaterEqual(hundreds, 100)
        self.assertIn(
            f"最后一条运行**全书**的回归测试：`tests/` 下所有测试文件、{hundreds} 多条测试都会执行，"
            "其中大多与上面两个例子无关。", self.pre)
        self.assertGreater(count_test_files(), 1)

    def test_env_sentence_matches_ci_and_p27(self):
        ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        ci_py = re.search(r"python-version: '([\d.]+)'", ci).group(1)
        ci_torch = re.search(r"torch==([\d.]+)", ci).group(1)
        hc = read("09a_minigpt_healthcheck.md")
        unit = re.search(r"^# .*（(P\d+)）", hc, re.M).group(1)
        py, tv = re.search(r"Python (3\.\d+\.\d+)、PyTorch ([\w.+]+)、", hc).groups()
        self.assertIn(
            f"CI 在 Python {ci_py} 加 PyTorch {ci_torch}（CPU 版）上运行这条命令；"
            f"{unit}（09a）报告的实测数字来自 Python {py}、PyTorch {tv}。", self.pre)
        self.assertNotIn("第 27 章", self.pre)


class Chapter01(unittest.TestCase):
    def test_sources(self):
        t = read("01_word_embedding.md")
        self.assertIn(
            "向量类比关系最早见于 Mikolov、Yih 与 Zweig 的 [Linguistic Regularities in Continuous "
            "Space Word Representations（NAACL 2013）](https://aclanthology.org/N13-1090/)；"
            "Mikolov 等人同年稍后的 [Distributed Representations of Words and Phrases and their "
            "Compositionality](https://arxiv.org/abs/1310.4546) 主要讲负采样与短语向量", t)
        self.assertNotIn("同组的", t)


class Chapter05(unittest.TestCase):
    def test_table3_section(self):
        self.assertIn("[来源 1，§3.5；表 3(E) 见 §6.2]", read("05_positional_encoding.md"))


class Chapter06(unittest.TestCase):
    def test_mha_param_reference(self):
        title = read("04_multi_head_attention.md").splitlines()[0]
        num = re.match(r"# 第 ?(\d+) ?章：多头注意力", title).group(1)
        self.assertIn(f"第 {num} 章手写 MHA 使用无 bias 的投影，计数是 $4D^2$",
                      read("06_residual_layernorm_ffn.md"))


class Chapter08b(unittest.TestCase):
    def test_pi_and_yarn_sources(self):
        t = read("08b_rope_yarn.md")
        self.assertIn("- [Position Interpolation 论文（Chen 等），arXiv:2306.15595]"
                      "(https://arxiv.org/abs/2306.15595)：位置插值 PI 的原始来源。", t)
        self.assertIn("- [YaRN 论文 v2（Peng 等），arXiv:2309.00071v2]"
                      "(https://arxiv.org/html/2309.00071v2)：这里固定版本解释分频段插值", t)


class Chapter08c(unittest.TestCase):
    def test_bfloat16_sentence_and_numbers(self):
        t = read("08c_modern_block.md")
        self.assertIn(
            "bfloat16 的情况不同：它的指数位与 float32 相同，最大约 3.4×10³⁸，40000 的平方不会溢出；"
            "源码同样把它升到 float32，是因为 bfloat16 只有 8 位有效精度（7 位尾数），"
            "平方和均值在 float32 中更精确。", t)
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        bf = torch.finfo(torch.bfloat16)
        self.assertEqual(f"{bf.max:.1e}", "3.4e+38")
        # bf16 = 1 sign + 8 exponent + 7 mantissa bits; eps = 2**-7
        self.assertEqual(bf.eps, 2.0 ** -7)
        self.assertEqual(torch.finfo(torch.float32).max.__format__(".1e"), "3.4e+38")
        self.assertTrue(torch.isfinite(torch.tensor(40000.0, dtype=torch.bfloat16).square()))
        self.assertFalse(torch.isfinite(torch.tensor(40000.0, dtype=torch.float16).square()))


class Chapter08e(unittest.TestCase):
    def test_decimal_precision_matches_test(self):
        src = (ROOT / "tests/test_attention_systems.py").read_text(encoding="utf-8")
        prec = re.search(r"context\.prec = (\d+)", src).group(1)
        self.assertIn(f"Decimal 的 **{prec} 位十进制有效数字精度独立参考**",
                      read("08e_attention_systems.md"))


class Chapter08i(unittest.TestCase):
    def setUp(self):
        self.src = code("20_pretraining_pipeline.py")
        self.text = read("08i_pretraining_pipeline.md")

    def test_provenance_sentence_matches_code(self):
        body = func_src(self.src, "clean_deduplicate")
        guard = re.search(r"if (.*)\n\s*raise ValueError\(\"demo accepts only", body).group(1)
        self.assertIn('doc.provenance != "generated-for-this-book"', guard)
        self.assertNotIn("source", guard)
        self.assertIn(
            "程序只检查 `provenance` 字段，值不是 `generated-for-this-book` 就报错，只是限定演示的输入范围；"
            "`source` 字段不做检查，只用于后面的配比账本。", self.text)

    def test_clean_text_order(self):
        ns = {"unicodedata": unicodedata}
        exec(func_src(self.src, "clean_text"), ns)
        clean = ns["clean_text"]
        self.assertEqual(clean(" 春\n风 "), "春风")
        # NFC first, then whitespace removal: a combining accent separated by a
        # space is NOT composed (the other order would give "é").
        self.assertEqual(clean("e \u0301"), "e\u0301")
        self.assertIn("| 先做 Unicode NFC，再删除所有空白字符 |", self.text)


class Chapter09a(unittest.TestCase):
    def test_anchor(self):
        self.assertIn("矩形 mask 的手算见 [P11.4](#p11.4-矩形-mask最容易写错的两行)",
                      read("09a_minigpt_healthcheck.md"))
        self.assertIn("\n## P11.4 矩形 mask：最容易写错的两行\n", read("08a_kv_cache.md"))


class Chapter04(unittest.TestCase):
    def test_labels(self):
        src = code("04_multi_head_attention.py")
        self.assertIn('print(f"\\n--- 头 {h+1} 的注意力权重矩阵 ---")', src)
        self.assertIn("头数（我们要分成几个分队，比如 2）", src)
        self.assertNotIn("专家头", src)
        self.assertIn("1. $B=2,T=5,D=12,H=3$ 时，分头并换轴后的 Q、分数表和拼接结果分别是什么形状？",
                      read("04_multi_head_attention.md"))


if __name__ == "__main__":
    unittest.main()
