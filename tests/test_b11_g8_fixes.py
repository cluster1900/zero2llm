"""B11 G8 mechanical fixes (09a / 10_appendix): exact-count pins."""
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CH = ROOT / 'chapters'

PINS = {
    '09a_minigpt_healthcheck.md': [
        ('S6 的旁路是独立线性 head，不是 DeepSeek-V3 按深度串联的 MTP 模块', 1),
        ('[V3 报告 v2 §2.2](https://arxiv.org/abs/2412.19437v2)', 1),
        ('只有一对通道，频率 θ=1，位置取 π/2，转角 θ×π/2=π/2', 1),
        ('这里 θ=1 的二维特例', 0),
        ('拿第一条序列（下标 0）的前两个 token 作 prompt（贪心解码器只接受 batch 为 1）', 1),
        ('最后拿前两个 token 作 prompt', 0),
    ],
    '10_appendix.md': [
        ('Q 投影宽度是 $16\\times128=2048$（K/V 各为 $8\\times128=1024$），不能擅自把 Q 投影改成 1024。', 1),
        ('；投影宽度是', 0),
    ],
}


class B11G8Fixes(unittest.TestCase):
    def test_exact_counts(self):
        for name, pins in PINS.items():
            text = (CH / name).read_text(encoding='utf-8')
            for needle, n in pins:
                with self.subTest(file=name, needle=needle):
                    self.assertEqual(text.count(needle), n)

    def test_code_facts_behind_wording(self):
        hc = (ROOT / 'code' / '28_minigpt_healthcheck.py').read_text(encoding='utf-8')
        self.assertEqual(hc.count('tokens[:1, :2]'), 1)
        self.assertEqual(hc.count('torch.tensor([math.pi / 2], dtype=torch.float64)'), 1)
        cfg = (ROOT / 'code' / 'data' / 'qwen3' / 'dense' / 'config.json').read_text(encoding='utf-8')
        for f in ('"num_key_value_heads": 8', '"num_attention_heads": 16', '"head_dim": 128'):
            self.assertEqual(cfg.count(f), 1, f)


if __name__ == '__main__':
    unittest.main()
