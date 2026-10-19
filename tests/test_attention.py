"""CPU P03–P05 regression: no GPU, downloads, or model training."""
import importlib.util
from pathlib import Path
import sys
import math
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'code' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


manual = load('02_manual_attention')
HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    import torch.nn.functional as F
    from attention_core import masked_softmax, scaled_attention, self_attention_mask
    MHA = load('04_multi_head_attention').SimpleMultiHeadAttention


class StableSoftmaxTests(unittest.TestCase):
    def test_small_matches_naive(self):
        xs = [-2., .5, 3.]
        expected = [math.exp(x) / sum(map(math.exp, xs)) for x in xs]
        for a, b in zip(manual.softmax(xs), expected):
            self.assertAlmostEqual(a, b)

    def test_large_and_shift(self):
        expected = manual.softmax([0., 1., 2.])
        self.assertEqual(manual.softmax([1000., 1001., 1002.]), expected)
        self.assertTrue(all(math.isfinite(x) for x in expected))
        self.assertAlmostEqual(sum(expected), 1.)

    def test_invalid(self):
        for xs in [[], [math.inf], [-math.inf], [math.nan]]:
            with self.subTest(xs=xs), self.assertRaises(ValueError):
                manual.softmax(xs)


@unittest.skipUnless(HAS_TORCH, 'CPU experiments require torch; install requirements.txt')
class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(1)

    def test_masked_rows_and_gradients(self):
        scores = torch.randn(2, 3, 3, dtype=torch.float64, requires_grad=True)
        allowed = torch.tensor([[True, False, False], [True, True, False], [False, False, False]])
        weights = masked_softmax(scores, allowed)
        torch.testing.assert_close(weights.sum(-1), torch.tensor([[1., 1., 0.]]).double().expand(2, -1))
        self.assertEqual(torch.count_nonzero(weights[..., ~allowed]).item(), 0)
        (weights.square().sum()).backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertEqual(torch.count_nonzero(scores.grad[:, 2]).item(), 0)

    def test_invalid_masks_and_nonfinite(self):
        with self.assertRaises(TypeError):
            masked_softmax(torch.zeros(2, 3), torch.ones(2, 3))
        with self.assertRaises(ValueError):
            masked_softmax(torch.zeros(2, 3), torch.ones(5, dtype=torch.bool))
        with self.assertRaises(ValueError):
            masked_softmax(torch.tensor([[float('nan')]]))

    def test_causality_and_prefix_equivalence(self):
        x = torch.randn(2, 5, 8, dtype=torch.float64)
        model = MHA(8, 2).double()
        allowed = self_attention_mask(torch.ones(2, 5, dtype=torch.bool))
        out, weights = model(x, allowed)
        changed = x.clone()
        changed[:, 3:] += torch.randn_like(changed[:, 3:]) * 20
        changed_out, _ = model(changed, allowed)
        torch.testing.assert_close(out[:, :3], changed_out[:, :3])
        self.assertEqual(torch.count_nonzero(weights.triu(1)).item(), 0)
        for end in range(1, 6):
            prefix, _ = model(x[:, :end], allowed[:, :, :end, :end])
            torch.testing.assert_close(out[:, end-1], prefix[:, -1])
        bidirectional, _ = model(x)
        changed_bidirectional, _ = model(changed)
        self.assertGreater((bidirectional[:, :3] - changed_bidirectional[:, :3]).abs().max().item(), 1e-5)

    def test_padding_queries_and_keys(self):
        x = torch.randn(2, 4, 8, dtype=torch.float64)
        valid = torch.tensor([[True, True, False, False], [False, False, False, False]])
        model = MHA(8, 2).double()
        allowed = self_attention_mask(valid)
        out, weights = model(x, allowed)
        changed = x.clone()
        changed[~valid] += 100
        other, _ = model(changed, allowed)
        torch.testing.assert_close(out, other)
        self.assertEqual(torch.count_nonzero(out[~valid]).item(), 0)
        self.assertTrue(torch.isfinite(weights).all())

    def test_shapes_and_parameter_count(self):
        for b, t, d, h in [(1, 1, 4, 1), (2, 5, 12, 3), (3, 2, 8, 4)]:
            with self.subTest(shape=(b,t,d,h)):
                model = MHA(d, h)
                out, weights = model(torch.randn(b, t, d))
                self.assertEqual(out.shape, (b, t, d))
                self.assertEqual(weights.shape, (b, h, t, t))
                self.assertEqual(sum(p.numel() for p in model.parameters()), 4*d*d)

    def test_one_head_reference(self):
        model = MHA(6, 1).double()
        x = torch.randn(2, 4, 6, dtype=torch.float64)
        mask = torch.ones(4, 4, dtype=torch.bool).tril()
        actual, weights = model(x, mask)
        q, k, v = model.W_q(x), model.W_k(x), model.W_v(x)
        reference_weights = torch.softmax((q @ k.transpose(-1,-2) / math.sqrt(6)).masked_fill(~mask, -torch.inf), -1)
        reference = model.W_o(reference_weights @ v)
        torch.testing.assert_close(actual, reference)
        torch.testing.assert_close(weights[:, 0], reference_weights)

    def test_multiple_heads_reference_and_backward(self):
        model = MHA(8, 2).double()
        x = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
        actual, weights = model(x)
        chunks = []
        for head in range(2):
            sl = slice(head*4,(head+1)*4)
            q, k, v = model.W_q(x)[..., sl], model.W_k(x)[..., sl], model.W_v(x)[..., sl]
            chunks.append(torch.softmax(q @ k.transpose(-1,-2) / 2, -1) @ v)
        torch.testing.assert_close(actual, model.W_o(torch.cat(chunks, -1)))
        actual.square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_official_sdpa(self):
        q, k, v = [torch.randn(2, 3, 4, dtype=torch.float64) for _ in range(3)]
        allowed = torch.ones(3, 3, dtype=torch.bool).tril()
        actual, _ = scaled_attention(q, k, v, allowed)
        reference = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.0)
        torch.testing.assert_close(actual, reference)

    def test_invalid_dimensions(self):
        for d, h in [(0,1),(8,0),(7,2)]:
            with self.assertRaises(ValueError):
                MHA(d,h)
        with self.assertRaises(ValueError):
            MHA(8,2)(torch.randn(2,4,6))


if __name__ == '__main__':
    unittest.main()
