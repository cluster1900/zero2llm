"""B11 group 4 (08A、08B、08C、08D): readability/P2 polish pins (bot-hub#123)."""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHANGED = [
("chapters/08b_rope_yarn.md","| logits | [2,12,17] | 各 Block 后接归一化和词表分类头 |","| logits | [2,12,17] | 最后一个 Block 之后做一次 `ln_f`，再接词表分类头 |"),
("chapters/08c_modern_block.md","1. LN 在两个输入上都减去各自均值，得到同样的 `[-1,1]`；RMSNorm 不居中，均方从 5 变 17，分子也改变。","1. LN 在两个输入上都减去各自均值，分子都是 `[-1,1]`，方差也都为 1，所以输出同为 `[-0.816497,0.816497]`；RMSNorm 不居中，均方从 5 变 17，分子也改变。"),
]


class B11G4Wording(unittest.TestCase):
    def test_changed(self):
        for rel, old, new in CHANGED:
            text = (ROOT / rel).read_text(encoding="utf-8")
            self.assertEqual(text.count(new), 1, (rel, new))
            if old not in new:
                self.assertEqual(text.count(old), 0, (rel, old))

    def test_08b_head_only_after_last_block(self):
        text = (ROOT / "chapters/08b_rope_yarn.md").read_text(encoding="utf-8")
        self.assertNotIn("各 Block 后接", text)


if __name__ == "__main__":
    unittest.main()
