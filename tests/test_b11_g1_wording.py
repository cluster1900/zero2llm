"""B11 group 1 (前言、01、01A、02): readability/P2 polish pins (bot-hub#123)."""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHANGED = [
("chapters/00_preface.md","读完基础部分，你应该能够回答：","读完基础篇（P00–P10，见 0.5 节），你应该能够回答："),
("chapters/01_word_embedding.md","这些数说明：在我们选定的特征和尺度下，","图 1-1 只画出甜度和脆度两维；表中的距离和余弦按三维（含价格）计算。按欧氏距离从近到远、按余弦从高到低，五种水果的排名在这里恰好一致；1.4 节会给出两者结论不同的例子。\n\n这些数说明：在我们选定的特征和尺度下，"),
("chapters/01_word_embedding.md","脚本会打印上面的水果数值，并按余弦排序给出结论。","脚本会打印苹果的坐标和上表中各水果的余弦与欧氏距离，并按余弦排序给出结论。"),
("chapters/01a_tokenizer.md","第三行从未出现在训练语料中，仍然可编码和恢复，但几乎没有合并收益。","第三行从未出现在训练语料中，仍然可编码和恢复，但 16 个字节仍是 16 个 token，合并收益为 0。"),
("chapters/01a_tokenizer.md","**填充**需要配合 mask；","**填充**需要配合 mask（第 3 章讲）；"),
("chapters/01a_tokenizer.md","序列长度变化会影响后面的 attention 和 KV cache 成本，","序列长度变化会影响后面注意力计算（第 3 章）和生成缓存（KV cache，P11）的成本，"),
("chapters/01a_tokenizer.md","参考答案：取决于具体 tokenizer；训练生成规则，编码使用已有规则；不能，需要在代表性的未见数据上评估；单个 token 可能不是完整 UTF-8 字符；字符词表可能遇到未收录字符，而完整字节表可表示合法 UTF-8，但并不赋予模型理解能力。",
 "参考答案：\n\n1. 取决于具体 tokenizer。\n2. 训练生成规则，编码使用已有规则。\n3. 不能，需要在代表性的未见数据上评估。\n4. 单个 token 可能不是完整 UTF-8 字符。\n5. 字符词表可能遇到未收录字符，而完整字节表可表示合法 UTF-8，但并不赋予模型理解能力。"),
("chapters/02_why_attention.md","真实向量的各维也不天然叫“甜度”“科技感”。","真实向量的各维也不天然叫“甜度”“脆度”。"),
("chapters/02_why_attention.md","这一步会在因果掩码一章再次派上用场。","第 3 章 §3.4 的因果掩码做的正是这件事：遮住当前位置之后的内容。"),
("chapters/02_why_attention.md","其余五张暂时都设为 $(0,0)$。","其余五张暂时都设为 $(0,0)$，包括“太累了 / 太宽了”。"),
("chapters/02_why_attention.md","于是，相同的内容卡片经过不同权重的混合，得到不同的结果。","于是，相同的内容卡片经过不同权重的混合，得到不同的结果；$c_A$ 与 $c_B$ 的差别完全来自表 2-2 手写的权重，而不是“太累了 / 太宽了”的内容向量。"),
("chapters/02_why_attention.md","5. 如果把表中的权重换一套，聚合结果必然改变吗？请用本章的三个数验证。","5. 如果把一组权重换成另一组，聚合结果必然改变吗？请用 2.5 节的三个数验证。"),
]


class TestB11Group1(unittest.TestCase):
    def test_changed(self):
        for rel, old, new in CHANGED:
            body = (ROOT / rel).read_text(encoding="utf-8")
            with self.subTest(file=rel, new=new[:30]):
                if old not in new:  # pure insertions keep the old text inside new
                    self.assertEqual(body.count(old), 0)
                self.assertEqual(body.count(new), 1)

    def test_no_kejigan_axis(self):
        self.assertNotIn("科技感", (ROOT / "chapters/02_why_attention.md").read_text(encoding="utf-8"))

    def test_01a_answers_one_per_line(self):
        body = (ROOT / "chapters/01a_tokenizer.md").read_text(encoding="utf-8")
        tail = body[body.index("参考答案：\n"):]
        for i in range(1, 6):
            with self.subTest(answer=i):
                self.assertIn(f"\n{i}. ", tail[:600])


if __name__ == "__main__":
    unittest.main()
