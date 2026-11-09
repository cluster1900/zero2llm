"""B11 G8 mechanical fixes (09a / 10_appendix): exact-count pins."""
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CH = ROOT / 'chapters'

PINS = {
    '09a_minigpt_healthcheck.md': [
        ('S6 的旁路是独立线性 head，不是 DeepSeek-V3 按深度串联的 MTP 模块', 1),
        ('[V3 报告 v2 §2.2](https://arxiv.org/abs/2412.19437v2)', 1),
        # bot-hub#126 F8 (F0 r4 A.5): per-pair angular frequency θ→ω_i (same sentence, letter only); old θ wording asserted absent.
        ('只有一对通道，角频率 $\\omega_i=1$，位置取 π/2，转角 $\\omega_i\\times\\pi/2=\\pi/2$', 1),
        ('只有一对通道，频率 θ=1，位置取 π/2，转角 θ×π/2=π/2', 0),
        ('这里 θ=1 的二维特例', 0),
        ('拿第一条序列（下标 0）的前两个 token 作 prompt（贪心解码器只接受 batch 为 1）', 1),
        ('最后拿前两个 token 作 prompt', 0),
    ],
    '09_modern_llms.md': [
        ('源码按 16 通道一组使用 E4M3 scale（固定 revision `inference/model.py` L761：`fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)`；indexer 另用 32 个一组的 E8M0）；', 1),
        ('按从 0 开始的层号，encoder 是下标 0–19 的层，decoder 是下标 20–39 的层；', 1),
        ('这份 global KV 由下标 20 的层（`kv_source_layer_ids` 的最后一项）用本层注意力输入计算，这个输入来自下标 19 层（最后一个 encoder 层）的输出，下标 21–39 的层读取同一份缓存；', 1),
        ('所以选中下标 **2、1**（从 0 计数，即第三个和第二个）的专家。', 1),
        ('所以选中下标 **2、1** 的专家。', 0),
    ],
    '10_appendix.md': [
        ('### F.0 平均值与一步更新速查', 1),
        ('这两个分母和更新顺序是核对训练日志的最小账本', 1),
        ('### A.3 三个可以手算的校验点', 1),
        ('### A.4 别把三种平均混在一起', 1),
        ('不要与 [P17](#p17-ced) 的 CED（Causal Encoder-Decoder，因果编码器—解码器）架构中的两段网络混同', 1),
        ('不要与 CED 架构中的两段网络混同', 0),
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

    def test_appendix_f_starts_with_self_test(self):
        text = (CH / '10_appendix.md').read_text(encoding='utf-8')
        head = '## 附录 F：自测、校验与最后一步 {#p28.f-自测校验与最后一步}\n\n1. A.2 的两头 KV'
        self.assertEqual(text.count(head), 1)
        answers = text.index('**答案。**')
        f0 = text.index('### F.0 平均值与一步更新速查')
        tools = text.index('仓库根目录的 `python -m unittest discover -s tests -v`')
        self.assertLess(text.index(head), answers)
        self.assertLess(answers, f0)
        self.assertLess(f0, tools)
        between = text[text.index('\n', answers):f0]
        self.assertEqual(between, '\n\n')

    def test_p17_anchor_exists(self):
        text = (CH / '08g_ced_architecture.md').read_text(encoding='utf-8')
        self.assertEqual(text.count('{#p17-ced}'), 1)
        self.assertEqual(text.count('Causal Encoder-Decoder（CED，因果编码器—解码器）'), 1)


if __name__ == '__main__':
    unittest.main()
