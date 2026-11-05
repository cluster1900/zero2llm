"""B11 group 5 (08E–08H) plus G4 rulings (08A、08B): wording pins (bot-hub#123)."""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# G4 裁定
G4 = [
("chapters/08a_kv_cache.md","参数更新后旧缓存无效。实现核对参数对象身份、版本、dtype 与 device，包含 `assign=True` 替换参数的情况；不支持通过 `.data` 等方式故意绕过 PyTorch 版本跟踪。","参数更新或替换后，旧缓存会被拒绝。"),
("chapters/08b_rope_yarn.md","源码 `rotation_walkthrough()` 用 float64 生成上表，`test_rotation_matches_independent_float64_matrix_and_preserves_tail` 还用独立的 2×2 矩阵验证实现，","源码 `rotation_walkthrough()` 用 float64 生成上表；本章另用独立的 2×2 矩阵核对实现，"),
("chapters/08b_rope_yarn.md","点积会乘 **a²**，不是 a。","点积会乘 **a²**，不是 a。s>1 时 a>1，即 t<1，logits 被放大，注意力分布变得更尖。"),
]

# G5 机械修正
G5 = [
("chapters/08h_moe.md","router 第一列为 [0,log2,log4]","router 第一列为 [0,ln 2,ln 4]"),
("chapters/08h_moe.md","输入第一维梯度为 78+20log2，","输入第一维梯度为 78+20 ln 2，"),
]

CHANGED = G4 + G5


class B11G5Wording(unittest.TestCase):
    def test_changed(self):
        for rel, old, new in CHANGED:
            body = (ROOT / rel).read_text(encoding="utf-8")
            with self.subTest(file=rel, new=new[:30]):
                if old not in new:
                    self.assertEqual(body.count(old), 0)
                self.assertEqual(body.count(new), 1)

    def test_no_test_name_or_log2_left(self):
        self.assertNotIn("test_rotation_matches", (ROOT / "chapters/08b_rope_yarn.md").read_text(encoding="utf-8"))
        self.assertNotIn("log2", (ROOT / "chapters/08h_moe.md").read_text(encoding="utf-8"))
        self.assertNotIn("assign=True", (ROOT / "chapters/08a_kv_cache.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
