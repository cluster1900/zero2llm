"""B11 G6 (08i-08l) mechanical wording fixes: exact-count pins."""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHANGED = [
    ('chapters/08i_pretraining_pipeline.md', '| 先做 Unicode NFC，再删除所有空白字符 |', '| 先做 Unicode NFC（只统一等价字符的编码写法，不删空白），再删除所有空白字符 |'),
    ('chapters/08j_training_frameworks.md', '本书之前的 miniGPT 是用来逐层解释机制的代码主线', '本书前面从 Baby-GPT 到 S5 的代码是用来逐层解释机制的主线'),
    ('chapters/08j_training_frameworks.md', '梯度会被错误缩小为本例正确值的 9/11。', '梯度会被错误缩小为本例正确值的 9/11。这个计数就是 loss 的分母，所以打印出的 loss 读数也同样变成正确值的 9/11。'),
    ('chapters/08k_sft_lora.md', '我们的 encoder 通常会更早拒绝', '我们的数据函数 `encode_chat` 通常会更早拒绝'),
]


class B11G6WordingTest(unittest.TestCase):
    def test_changed(self):
        for name, old, new in CHANGED:
            with self.subTest(file=name, new=new[:30]):
                text = (ROOT / name).read_text(encoding="utf-8")
                self.assertEqual(text.count(new), 1)
                if old not in new:
                    self.assertEqual(text.count(old), 0)

    def test_old_terms_gone(self):
        self.assertNotIn("miniGPT", (ROOT / "chapters/08j_training_frameworks.md").read_text(encoding="utf-8"))
        self.assertNotIn("encoder 通常", (ROOT / "chapters/08k_sft_lora.md").read_text(encoding="utf-8"))
        self.assertEqual((ROOT / "chapters/08j_training_frameworks.md").read_text(encoding="utf-8").count("loss 读数也同样变成正确值的 9/11"), 1)


if __name__ == "__main__":
    unittest.main()
