"""Follow-up PR-C: F.0 drops A.3/A.4 restatement; keeps independent formulas + pins (bot-hub#123)."""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APPENDIX = ROOT / "chapters" / "10_appendix.md"

# New F.0 body (independent derivation only).
NEW_F0 = (
    # bot-hub#126 F9: F.0 \bar L / L' → \bar{\mathcal L} / \mathcal L' (A.5 loss letter)
    r"$\bar{\mathcal L}_{\mathrm{doc}}=(2+4)/2=3$，而 "
    r"$\bar{\mathcal L}_{\mathrm{token}}=(1\times2+3\times4)/4=3.5$。学习率 $0.1$ 后 "
    r"$w'=1-0.1\times4=0.6$，新损失 $\mathcal L'=(0.6\times2)^2/2=0.72$。这两个分母和更新顺序是核对训练日志的最小账本。"
)

# Old overlapping prose removed from F.0 (still present in A.3/A.4 in other forms).
OLD_F0 = (
    r"文档平均与 token 平均使用不同分母：$\bar L_{\mathrm{doc}}=(2+4)/2=3$，而 "
    r"$\bar L_{\mathrm{token}}=(1\times2+3\times4)/4=3.5$。单参数例子 $z=wx$、$L=(z-y)^2/2$ "
    r"取 $w=1,x=2,y=0$，则 $z=2$，梯度 $(z-y)\cdot x$ 等于 4；学习率 $0.1$ 后 "
    r"$w'=1-0.1\times4=0.6$，新损失 $L'=(0.6\times2)^2/2=0.72$。这两个分母和更新顺序是核对训练日志的最小账本。"
)

PINS = [
    ("### F.0 平均值与一步更新速查", 1),
    (NEW_F0, 1),
    (r"\bar{\mathcal L}_{\mathrm{doc}}=(2+4)/2=3", 1),
    (r"\bar{\mathcal L}_{\mathrm{token}}=(1\times2+3\times4)/4=3.5", 1),
    (r"w'=1-0.1\times4=0.6", 1),
    (r"\mathcal L'=(0.6\times2)^2/2=0.72", 1),
    ("这两个分母和更新顺序是核对训练日志的最小账本", 1),
    (OLD_F0, 0),
    ("文档平均与 token 平均使用不同分母", 0),
    ("单参数例子 $z=wx$", 0),
    (r"(z-y)\cdot x", 0),
]


class FollowCF0Dedupe(unittest.TestCase):
    def test_exact_counts(self):
        text = APPENDIX.read_text(encoding="utf-8")
        for needle, n in PINS:
            with self.subTest(needle=needle[:60], n=n):
                self.assertEqual(text.count(needle), n)

    def test_a3_a4_kept(self):
        text = APPENDIX.read_text(encoding="utf-8")
        self.assertEqual(text.count("### A.3 三个可以手算的校验点"), 1)
        self.assertEqual(text.count("### A.4 别把三种平均混在一起"), 1)
        # A.3 gradient card still holds the hand calc; F.0 no longer restates it.
        self.assertEqual(text.count("(z-y)x=4"), 1)
        self.assertEqual(text.count("$w'=0.6$，新损失 0.72"), 1)
        self.assertEqual(text.count("文档等权平均是 $(2+4)/2=3$"), 1)
        self.assertEqual(text.count("token 等权平均是 $(1\\times2+3\\times4)/4=3.5$"), 1)

    def test_f0_still_after_answers(self):
        text = APPENDIX.read_text(encoding="utf-8")
        answers = text.index("**答案。**")
        f0 = text.index("### F.0 平均值与一步更新速查")
        tools = text.index("仓库根目录的 `python -m unittest discover -s tests -v`")
        self.assertLess(answers, f0)
        self.assertLess(f0, tools)
        between = text[text.index("\n", answers):f0]
        self.assertEqual(between, "\n\n")

    def test_each_positive_pin_mutation_kills(self):
        """Mutating any required needle to a sentinel makes its exact-count pin fail."""
        text = APPENDIX.read_text(encoding="utf-8")
        for needle, n in PINS:
            if n != 1:
                continue
            broken = text.replace(needle, "<<<MUTATED>>>", 1)
            with self.subTest(needle=needle[:60]):
                self.assertEqual(broken.count(needle), 0)
                with self.assertRaises(AssertionError):
                    self.assertEqual(broken.count(needle), 1)

    def test_old_overlap_returning_kills_zero_pin(self):
        text = APPENDIX.read_text(encoding="utf-8")
        self.assertEqual(text.count(OLD_F0), 0)
        revived = text.replace(NEW_F0, OLD_F0, 1)
        self.assertEqual(revived.count(OLD_F0), 1)
        with self.assertRaises(AssertionError):
            self.assertEqual(revived.count(OLD_F0), 0)
        with self.assertRaises(AssertionError):
            self.assertEqual(revived.count("文档平均与 token 平均使用不同分母"), 0)


if __name__ == "__main__":
    unittest.main()
