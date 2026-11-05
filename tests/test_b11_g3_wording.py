"""B11 group 3 (06/07/07a/08) readability/P2 pins, plus the G1#3/G2 boundary rulings (bot-hub#123).

RULINGS: 02 figure caption arrow direction; 03 scaling node relabel; 04 §4.4 lead sentence.
"""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RULINGS=[
("chapters/02_why_attention.md","accDescr: 它可以直接与小明、在、草地计算相关性，不必逐词传话；生成时仍受因果掩码限制。","accDescr: 它可以直接与小明、在、草地计算相关性，不必逐词传话；生成时仍受因果掩码限制。箭头从当前位置指向被查询的位置。"),
("chapters/03_qkv_self_attention.md","A[\"缩放：Score ÷ √2<br/>√2 ≈ 1.414\"]","A[\"缩放：Score ÷ √d_k（本例 √2）<br/>√2 ≈ 1.414\"]"),
("chapters/04_multi_head_attention.md","## 4.4 多了头，掩码规则没有变\n\n多头把分数","## 4.4 多了头，掩码规则没有变\n\n掩码规则与第 3 章相同，本节只补多头带来的两点：同一个掩码怎样按头广播；以及所有头都被全屏蔽时输出为零，只屏蔽一个头则未必如此。\n\n多头把分数"),
]
G3=[
("chapters/06_residual_layernorm_ffn.md","6.1 的残差把每层输出都以权重 1 相加。","6.1 的残差把每个子层的增量 $F_i$ 都以权重 1 相加。"),
("chapters/06_residual_layernorm_ffn.md","不把它冒充上一批 `attention_core.py`","不把它冒充第 3 章 `attention_core.py`"),
("chapters/06_residual_layernorm_ffn.md","Activation: 过滤杂念 (非线性思维跃迁)","Activation: 过滤杂念 (非线性)"),
("chapters/07_encoder_decoder.md","省略头轴后，Q=[1,2,4]，K=[1,3,4]，V=[1,3,4]；分数为 [1,2,3]，聚合结果为 [1,2,4]。","省略头轴后，Q 形状 (1,2,4)，K 与 V 形状 (1,3,4)；分数形状 (1,2,3)，聚合结果形状 (1,2,4)。"),
("chapters/07_encoder_decoder.md","有源端 padding 时也要屏蔽对应 K/V 位置。这个“跨序列”的连接方式，不应和[P17](#p17-ced) 的 Causal Encoder-Decoder（CED）名称混为一谈；本章没有实现 CED，只学习传统翻译双塔与生成循环。","有源端 padding 时也要屏蔽对应 K/V 位置。"),
("chapters/07a_training_loop.md","本章把三首诗中的前两首作为训练来源，","本章把三首诗中的前两首《静夜思》《登鹳雀楼》作为训练来源，"),
("chapters/07a_training_loop.md","AdamW 学习率 0.003、40 次更新。","AdamW 学习率 0.003（weight_decay 用 PyTorch 默认值 0.01）、40 次更新。"),
("chapters/08_baby_transformer.md","| 春眠 | `春眠落头黄风白望入落欲一日依疑，床` | `春眠不觉晓，处处闻啼鸟。夜来风雨声` |\n","| 春眠 | `春眠落头黄风白望入落欲一日依疑，床` | `春眠不觉晓，处处闻啼鸟。夜来风雨声` |\n\n温度 0 是贪心解码。未训练时“床前”“白日”之后的续写恰好相同，不是表格贴错。\n"),
]
G3_RULINGS=[
("chapters/07a_training_loop.md","显示到六位小数。seed 11 保留后值采用独立审核环境的 4.397413730…","显示到六位小数。测试对表值使用绝对容差 1e-6"),
("chapters/07a_training_loop.md","### 参考答案与可执行测试对应","### 参考答案\n\n1. x=`[7,1,9]`，y=`[1,9,3]`；不能。当前格合法地能看见自己，错误任务变成复制。\n2. 0.836988217。交叉熵是各题负对数的均值，不是概率均值的负对数。\n"),
("chapters/07a_training_loop.md","下一批前调用 `optimizer.zero_grad(set_to_none=True)`。对应","4. 梯度累加成两倍；下一批前调用 `optimizer.zero_grad(set_to_none=True)`。\n5. 两者都不会替代另一方；BabyGPT 无 Dropout/BatchNorm。\n"),
("chapters/07a_training_loop.md","本章未演示过拟合。下一章 S0 更只是同语料记忆。对应","本章未演示过拟合。下一章 S0 更只是同语料记忆。\n"),
("chapters/08_baby_transformer.md","温度 0 是贪心；为兼容旧 API，负温度也走贪心，新的调用建议只用 0 或正数。","温度 0 是贪心；温度应取 0 或正数。"),
("chapters/07_encoder_decoder.md","transformer_full_architecture.svg)\n\n下图只描述","transformer_full_architecture.svg)\n\n图中编码器各位置互相可见，但 padding 仍要屏蔽，见表 7-1。\n\n下图只描述"),
]
CHANGED = RULINGS + G3 + G3_RULINGS


class TestB11Group3(unittest.TestCase):
    def test_changed(self):
        for rel, old, new in CHANGED:
            body = (ROOT / rel).read_text(encoding="utf-8")
            with self.subTest(file=rel, new=new[:30]):
                if old not in new:
                    self.assertEqual(body.count(old), 0)
                self.assertEqual(body.count(new), 1)

    def test_no_stale_terms(self):
        self.assertNotIn("Score ÷ √2<br/>", (ROOT / "chapters/03_qkv_self_attention.md").read_text(encoding="utf-8"))
        self.assertNotIn("思维跃迁", (ROOT / "chapters/06_residual_layernorm_ffn.md").read_text(encoding="utf-8"))
        self.assertNotIn("p17-ced", (ROOT / "chapters/07_encoder_decoder.md").read_text(encoding="utf-8"))
        self.assertNotIn("Q=[1,2,4]", (ROOT / "chapters/07_encoder_decoder.md").read_text(encoding="utf-8"))

    def test_07a_answers_have_no_test_names(self):
        body = (ROOT / "chapters/07a_training_loop.md").read_text(encoding="utf-8")
        tail = body[body.index("### 参考答案\n"):]
        tail = tail[:tail.index("\n## ")] if "\n## " in tail else tail
        self.assertEqual(tail.count("test_"), 0)
        self.assertEqual(tail.count("对应 `"), 0)
        self.assertNotIn("负温度", (ROOT / "chapters/08_baby_transformer.md").read_text(encoding="utf-8"))

    def test_rendered_scaling_node(self):
        svg = (ROOT / "dist/assets/rendered/flow-04.svg").read_text(encoding="utf-8")
        self.assertIn("d_k", svg)


if __name__ == "__main__":
    unittest.main()
