"""B10: P1 wording/structure fixes from bot-hub#123."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (file, old, new): old must be gone, new present exactly once.
CHANGED = [
    ("code/01_vector_similarity.py", "大模型世界里的名场面：", "静态词向量里的经典类比："),
    ("chapters/01a_tokenizer.md", None, "这里的 `vocab_size=276` 是请求上限，等于 256 个基础字节加上文最多 20 次合并；上文实测的 275 是 256 + 19，因为合并提前耗尽。两数不同不是错误。"),
    ("chapters/04_multi_head_attention.md", None, "本章各头的 Q、K、V 等宽，所以 $d_h$ 就是第 3 章的 $d_k$ 和 $d_v$，只是换成按头计的名字；图中仍标的 `d_k` 指的是同一个宽度。"),
    ("chapters/08k_sft_lora.md", "这些忽略目标对应的 logits 没有直接 loss 梯度", "预测被忽略 token 的那组 logits 没有直接 loss 梯度"),
    ("chapters/09_modern_llms.md", None, "反过来，[V4.1-Flash 固定配置][V41-config]（本仓库副本 `code/data/deepseek/v41.config.json` 第 29 行）的同名字段是 `num_key_value_heads=1`，也不能按字面断定 V4.1 是 MQA；"),
    ("chapters/09_modern_llms.md", "## P26.10 旧版第 9 章的直觉桥梁：", "## P26.10 架构直觉：保留有用内容，修正大而化之的结论"),
    ("assets/diagrams/transformer_block.svg", "无损直达", None),
    ("assets/diagrams/transformer_block.svg", "任意堆叠", "• 像乐高一样可重复堆叠！"),
    ("chapters/06_residual_layernorm_ffn.md", "图中的“无损直达”“任意堆叠”", "图中的“残差直通”“可重复堆叠”不是总梯度或训练效果的保证"),
    ("chapters/07_encoder_decoder.md", "不是每一轮都重新读完一切", "## 7.6 Prefill 与 decode：本章仍整段重算\n\n前面几节讲的是带编码器的翻译双塔；这一节换成只有 decoder、没有编码器的续写"),
    # bot-hub#126 F5：附录 A.5 统一后 H 只表示头数，FFN 宽度改记 d_ff^GELU / d_ff^SwiGLU（F0 r3 定稿，取代旧的 d_ff'），词表大小记 |V|。
    ("chapters/08c_modern_block.md", "头数 n_head=4", "设置 $\\lvert\\mathcal V\\rvert=57$、D=48、L=2、头数 $H=4$，embedding"),
    ("chapters/08c_modern_block.md", "本章的 H 仍只表示 FFN 宽度", "MHA 投影一行 2×4×48² 中的 4 指 Q、K、V、O 四个 48×48 矩阵，不是头数 $H$；FFN 的中间宽度另记为 $d_{\\mathrm{ff}}^{\\mathrm{GELU}}$、$d_{\\mathrm{ff}}^{\\mathrm{SwiGLU}}$。"),
    ("chapters/08i_pretraining_pipeline.md", None, "2,040 参数的玩具实验只演示数据到训练再到恢复的流程，规模太小，不能用来检验 Chinchilla、6ND 或涌现这类缩放律结论。"),
]

KEPT = [
    ("chapters/08_baby_transformer.md", "`main()` 只训练并打印结果，不会写 checkpoint；"),
    ("chapters/09_modern_llms.md", "[]{#架构选择为什么开放式生成常用-decoder-only}"),
    ("chapters/09_modern_llms.md", "[]{#三种架构各自擅长什么}"),
    ("chapters/09_modern_llms.md", "也不能因为 `num_key_value_heads=128` 就断定实现缓存了 128 份完整 K、V。"),
    ("chapters/01a_tokenizer.md", "    vocab_size=276,\n"),
    ("chapters/07_encoder_decoder.md", "**没有实"),
]


def read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


class TestB10Wording(unittest.TestCase):
    def test_changed(self):
        for rel, old, new in CHANGED:
            body = read(rel)
            with self.subTest(file=rel, old=old, new=(new or "")[:30]):
                if old is not None:
                    self.assertEqual(body.count(old), 0)
                if new is not None:
                    self.assertEqual(body.count(new), 1)

    def test_kept(self):
        for rel, text in KEPT:
            with self.subTest(file=rel, text=text[:30]):
                self.assertEqual(read(rel).count(text), 1)

    def test_svg_residual_labels(self):
        self.assertEqual(read("assets/diagrams/transformer_block.svg").count("(+1 梯度残差直通)"), 2)

    def test_svg_subtitle_no_lossless(self):
        svg = read("assets/diagrams/transformer_block.svg")
        self.assertEqual(svg.count("无损"), 0)
        self.assertEqual(svg.count("保持信号无损直通深层网络"), 0)
        self.assertEqual(svg.count("信号经残差直通深层网络"), 1)

    def test_testing_md_08k_sync(self):
        doc = read("docs/testing.md")
        self.assertEqual(doc.count("代码测试会发现：这些忽略目标对应的 logits"), 0)
        self.assertEqual(doc.count("代码测试会发现：预测被忽略 token 的那组 logits"), 1)

    def test_08c_head_symbol_guard(self):
        # bot-hub#126 F5：H 改为只表示头数（附录 A.5），守卫反过来：08c 里不能再把 H 当 FFN 宽度。
        text = read("chapters/08c_modern_block.md")
        self.assertIsNone(re.search(r"(?<![\w_{])H=(4D|192)|D→H|H→D|2DH|M=(128|192)\b", text))
        self.assertNotIn("n_head", text)

    def test_no_legacy_ch9_title(self):
        self.assertNotIn("旧版第 9 章", read("chapters/09_modern_llms.md"))


if __name__ == "__main__":
    unittest.main()
