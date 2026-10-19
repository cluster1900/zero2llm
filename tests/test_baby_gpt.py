"""P10 S0: CPU independent references and boundary regressions."""
from contextlib import redirect_stdout
import io
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from torch_test_support import torch_status, torch_reason

READY, VERSION, PRESENT = torch_status()
if READY:
    import torch
    import torch.nn.functional as F
    spec = importlib.util.spec_from_file_location('s0_test', Path(__file__).resolve().parents[1] / 'code/07_baby_gpt.py')
    s0 = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = s0
    spec.loader.exec_module(s0)


@unittest.skipUnless(READY, torch_reason('P10', VERSION, PRESENT))
class BabyGPTTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.model = s0.BabyGPT(s0.vocab_size, 16, 4, 2, 32).eval()
        self.ids = torch.tensor([s0.encode(p) for p in s0.poems])

    def test_legacy_defaults_and_character_roundtrip(self):
        config = s0.BabyGPT(s0.vocab_size).config
        self.assertEqual((config.d_model, config.num_heads, config.num_layers, config.max_seq_len), (48,4,2,128))
        self.assertEqual(len(s0.poems), 3)
        for poem in s0.poems:
            self.assertEqual(s0.decode(s0.encode(poem)), poem)
        with self.assertRaises(KeyError):
            s0.encode('🦄')

    def test_classic_seed42_eighty_step_demo(self):
        stream = io.StringIO()
        with redirect_stdout(stream):
            s0.main()
        output = stream.getvalue()
        for expected in ('Loss: 4.1271', 'Loss: 0.0196', 'Loss: 0.0021',
                         'Loss: 0.0011', 'Loss: 0.0009',
                         '床前明月光，疑是地上霜。举头望明月',
                         '白日依山尽，黄河入海流。欲穷千里目',
                         '春眠不觉晓，处处闻啼鸟。夜来风雨声',
                         '明月不觉晓，处处闻啼鸟。夜来风雨声'):
            self.assertIn(expected, output)

    def test_chapter_parameter_ledger_and_shift_counts(self):
        model = s0.BabyGPT(s0.vocab_size)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 67872)
        self.assertEqual(model.tok_emb.weight.numel(), 2736)
        self.assertEqual(model.pos_emb.weight.numel(), 6144)
        self.assertEqual(model.head.weight.numel(), 2736)
        self.assertEqual(sum(p.numel() for p in model.blocks[0].parameters()), 28080)
        self.assertEqual(sum(p.numel() for p in model.ln_f.parameters()), 96)
        self.assertIsNot(model.tok_emb.weight, model.head.weight)
        self.assertFalse(any('mask' in name for name,_ in model.named_parameters()))
        self.assertTrue(any('mask' in name for name,_ in model.named_buffers()))
        self.assertEqual(sum(len(s0.encode(p))-1 for p in s0.poems), 69)
        self.assertEqual(len(s0.encode(s0.text_corpus))-1, 71)
        expected = {'。':0, '一':1, '上':2, '不':3, '光':8, '前':10,
                    '地':13, '床':23, '明':27, '月':32, '霜':52, '，':56}
        self.assertEqual({c:s0.char2idx[c] for c in expected},expected)
        self.assertEqual(model(torch.zeros(2,5,dtype=torch.long))[0].shape,(2,5,57))
        # A sentence-ending punctuation token is not EOS.
        self.assertEqual(len(model.generate('。',3,0)),4)

    def test_shape_shift_loss_and_independent_reference(self):
        x, y = self.ids[:, :-1], self.ids[:, 1:]
        self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))
        logits, loss = self.model(x, y)
        self.assertEqual(logits.shape, (3, 23, s0.vocab_size))
        reference = -logits.log_softmax(-1).gather(-1, y.unsqueeze(-1)).mean()
        torch.testing.assert_close(loss, reference)
        self.assertIsNone(self.model(x)[1])
        self.assertTrue(torch.isfinite(loss))

    def test_causality_and_broken_mask_negative_control(self):
        x = self.ids[:, :8]
        future = x.clone()
        future[:, 4:] = (future[:, 4:] + 1) % s0.vocab_size
        expected = self.model(x)[0][:, :4]
        torch.testing.assert_close(expected, self.model(future)[0][:, :4], rtol=0, atol=0)
        for block in self.model.blocks:
            block.attn.mask.fill_(1)
        # Mutation control: this fixture detects a missing causal mask.
        self.assertGreater((self.model(x)[0][:, :4] - self.model(future)[0][:, :4]).abs().max().item(), 1e-4)

    def test_training_gradients_and_update(self):
        self.model.train()
        before = self.model.head.weight.detach().clone()
        opt = torch.optim.AdamW(self.model.parameters(), lr=.01)
        _, loss = self.model(self.ids[:, :-1], self.ids[:, 1:])
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.parameters()))
        opt.step()
        self.assertFalse(torch.equal(before, self.model.head.weight))

    def test_checkpoint_roundtrip_and_rejected_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 's0.pt'
            s0.save_checkpoint(path, self.model)
            restored = s0.load_checkpoint(path)
            self.assertEqual(restored.config, self.model.config)
            self.assertFalse(restored.training)
            torch.testing.assert_close(self.model(self.ids)[0], restored(self.ids)[0], rtol=0, atol=0)
            self.model.double()
            s0.save_checkpoint(path, self.model)
            restored = s0.load_checkpoint(path)
            self.assertEqual(restored.head.weight.dtype, torch.float64)
            torch.testing.assert_close(self.model(self.ids)[0], restored(self.ids)[0], rtol=0, atol=0)
            data = torch.load(path, weights_only=True)
            self.assertEqual(set(data), {'format_version','config','tokenizer_chars','state_dict'})
            data['tokenizer_chars'] = list(reversed(data['tokenizer_chars']))
            torch.save(data, path)
            with self.assertRaisesRegex(ValueError, 'vocabulary'):
                s0.load_checkpoint(path)

    def test_invalid_config_and_input(self):
        for kwargs in ({'d_model': 7}, {'num_heads': 0}, {'num_layers': 0}, {'max_seq_len': -1}, {'d_model': True}):
            with self.assertRaises(ValueError):
                s0.BabyGPT(s0.vocab_size, **kwargs)
        invalid = [torch.empty(1,0,dtype=torch.long), torch.zeros(0,1,dtype=torch.long),
                   torch.zeros(1,33,dtype=torch.long), torch.zeros(3), torch.zeros(1,2),
                   torch.tensor([[-1]]), torch.tensor([[s0.vocab_size]])]
        for x in invalid:
            with self.assertRaises(ValueError):
                self.model(x)
        with self.assertRaises(ValueError):
            self.model(self.ids, self.ids[:, :-1])
        self.assertEqual(self.model(torch.zeros(1,32,dtype=torch.long))[0].shape[1], 32)

    def test_generation_limits_and_preserved_sliding_window(self):
        self.assertEqual(self.model.generate('床前', 0), '床前')
        for n, temp in [(-1,1), (1.5,1), (1,float('nan')), (1,float('inf'))]:
            with self.assertRaises(ValueError):
                self.model.generate('床前', n, temp)
        with self.assertRaises(ValueError):
            self.model.generate('', 1)
        self.assertEqual(self.model.generate('床前', 3, -1), self.model.generate('床前', 3, 0))
        self.assertEqual(self.model.generate('床前', 3, 1e-300), self.model.generate('床前', 3, 0))
        small = s0.BabyGPT(s0.vocab_size, 16,4,1,3).eval()
        prompt = '床前明月'
        expected = s0.decode([small(torch.tensor([s0.encode(prompt[-3:])]))[0][0,-1].argmax().item()])
        self.assertEqual(small.generate(prompt,1,0), prompt + expected)


if __name__ == '__main__':
    unittest.main()
