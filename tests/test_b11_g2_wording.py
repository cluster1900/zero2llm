"""B11 group 2 (03/04/05) readability/P2 pins plus the G1 boundary rulings (bot-hub#123).

The first two CHANGED entries are the G1 rulings for the preface (L23 MoE pointer, L68 record row).
"""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHANGED = [
("chapters/00_preface.md","“专家”帮助理解 MoE 路由，","“专家”帮助理解 MoE 路由（P18），"),
("chapters/00_preface.md","| 代码版本 | 本次 Git commit；若有未提交修改，写明 |","| 代码版本 | 本书版本/下载日期；若改过代码写明改动 |"),
("chapters/03_qkv_self_attention.md","广播后得到 [B,1,T,T]，可供多个头共享。","广播后得到 [B,1,T,T]；中间那根长度为 1 的轴留给第 4 章的多个头共享。"),
("chapters/03_qkv_self_attention.md","代码中创建下三角时还指定与输入相同的 device。","上面的片段省略了 device；`code/attention_core.py` 中的完整版本创建下三角时指定与输入相同的 device。"),
("chapters/03_qkv_self_attention.md","这是本书实现的契约，不表示所有框架、内核和版本都采用相同处理方式。","这是本书的选择，不表示所有框架、内核和版本都采用相同处理方式。"),
("chapters/04_multi_head_attention.md","共享 KV 头等变体见已实现的 [P14]","共享 KV 头等变体见 [P14]"),
("chapters/04_multi_head_attention.md","| $d_h$ | head dimension，每个头的特征宽度 | 4 |","| $d_h$ | head dimension，每个头的特征宽度；等宽时就是第 3 章的 $d_k$、$d_v$ | 4 |"),
("chapters/04_multi_head_attention.md","可以这样建立一个供所有 batch、所有头共用的 `[T,T]` 掩码：","可以这样建立一个供所有 batch、所有头共用的 `[T,T]` 掩码（`x` 与 `mha` 在 4.6 节创建）："),
("chapters/05_positional_encoding.md","![图 5-1：正弦余弦位置编码的波形频率谱]","![图 5-1：不同频率的正弦波形示意]"),
("chapters/05_positional_encoding.md","以及浮点数或布尔值维度。这是教学函数的明确契约。","以及浮点数或布尔值维度。"),
("chapters/05_positional_encoding.md","原论文采用相加，并在输入词嵌入处使用 $\\sqrt{D}$ 缩放；本例为便于手算，把 $X$ 直接当作待相加的输入。","原论文采用相加。注意：原论文在相加前先把词嵌入乘以 $\\sqrt{D}$。本例为便于手算不做这一步，把 $X$ 直接当作待相加的输入。"),
("code/04_multi_head_attention.py","(批次={batch_size}, 字数={seq_len}, 词向量维度","(批次={batch_size}, 序列长度={seq_len}, 词向量维度"),
("code/04_multi_head_attention.py","(批次, 头数={self.num_heads}, 字数={seq_len},","(批次, 头数={self.num_heads}, 序列长度={seq_len},"),
("code/04_multi_head_attention.py","# 4 个汉字：\"机器\" \"学习\" \"真\" \"酷\"","# 4 个位置，标签为 \"机器\" \"学习\" \"真\" \"酷\""),
]


class TestB11Group2(unittest.TestCase):
    def test_changed(self):
        for rel, old, new in CHANGED:
            body = (ROOT / rel).read_text(encoding="utf-8")
            with self.subTest(file=rel, new=new[:30]):
                if old not in new:
                    self.assertEqual(body.count(old), 0)
                self.assertEqual(body.count(new), 1)

    def test_no_stale_terms(self):
        code = (ROOT / "code/04_multi_head_attention.py").read_text(encoding="utf-8")
        self.assertNotIn("字数=", code)
        self.assertNotIn("4 个汉字", code)
        self.assertNotIn("频率谱", (ROOT / "chapters/05_positional_encoding.md").read_text(encoding="utf-8"))
        self.assertNotIn("Git commit", (ROOT / "chapters/00_preface.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
