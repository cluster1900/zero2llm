"""P07 CPU regression; tensor checks require an installed stable torch 2.14.x.

Mutation recipe (in-memory only; do not change production files): load this file
with runpy, compile a replaced 06_transformer_block.py into a ModuleType, replace
BlockTests methods' globals()['blocks'] with it, and run the named test. See the
batch validation record for exact replacement strings and observed failures.
"""

from contextlib import redirect_stdout
import importlib.metadata
import importlib.util
import io
from pathlib import Path
import re
import runpy
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'code' / '06_transformer_block.py'


def compatible_torch(version):
    """Allow patch releases/local wheel tags, not another minor or pre-release."""
    return bool(version and re.fullmatch(r'2\.14\.\d+(?:\+[A-Za-z0-9.]+)?', version))


try:
    TORCH_VERSION = importlib.metadata.version('torch')
except importlib.metadata.PackageNotFoundError:
    TORCH_VERSION = None
TORCH_READY = importlib.util.find_spec('torch') is not None and compatible_torch(TORCH_VERSION)
TORCH_REASON = 'P07 CPU numerical checks require installed stable torch 2.14.x; install requirements.txt'

if TORCH_READY:
    import torch
    import torch.nn.functional as F

    spec = importlib.util.spec_from_file_location('chapter06_blocks', CODE)
    blocks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(blocks)


class BlockDependencyTests(unittest.TestCase):
    def test_version_family_gate(self):
        for version in ['2.14.0', '2.14.1+cpu', '2.14.12+cu130']:
            with self.subTest(version=version):
                self.assertTrue(compatible_torch(version))
        for version in [None, '2.13.1', '2.15.0', '2.14.0rc1', '2.14.1.dev0', 'not-a-version']:
            with self.subTest(version=version):
                self.assertFalse(compatible_torch(version))

    def test_missing_or_incompatible_dependency_skips_without_importing_code(self):
        for installed in [None, '2.13.1', '2.15.0', '2.14.0rc1']:
            options = ({'side_effect': importlib.metadata.PackageNotFoundError}
                       if installed is None else {'return_value': installed})
            with self.subTest(installed=installed), patch('importlib.metadata.version', **options):
                namespace = runpy.run_path(str(Path(__file__)))
            self.assertFalse(namespace['TORCH_READY'])
            self.assertNotIn('blocks', namespace)
            result = unittest.TestResult()
            namespace['BlockTests']('test_zero_sublayers_preserve_preln_identity').run(result)
            self.assertEqual(len(result.skipped), 1)
            self.assertEqual(result.errors, [])
            self.assertEqual(result.failures, [])
            self.assertIn('2.14.x', result.skipped[0][1])

    def test_metadata_without_importable_module_also_skips(self):
        find_spec = importlib.util.find_spec

        def without_torch(name, *args, **kwargs):
            return None if name == 'torch' else find_spec(name, *args, **kwargs)

        with patch('importlib.metadata.version', return_value='2.14.1'), \
             patch('importlib.util.find_spec', side_effect=without_torch):
            namespace = runpy.run_path(str(Path(__file__)))
        self.assertFalse(namespace['TORCH_READY'])
        self.assertNotIn('blocks', namespace)


@unittest.skipUnless(TORCH_READY, TORCH_REASON)
class BlockTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        torch.set_num_threads(1)

    def test_zscore_three_steps_and_selftest_answers(self):
        x, mean, centered, squared, variance, denominator, normalized = blocks.layer_norm_walkthrough()
        self.assertEqual(mean.item(), 2.5)
        self.assertEqual(variance.item(), 1.25)
        torch.testing.assert_close(centered, torch.tensor([-1.5, -.5, .5, 1.5]).double())
        torch.testing.assert_close(squared, torch.tensor([2.25, .25, .25, 2.25]).double())
        self.assertAlmostEqual(denominator.item(), (1.25 + 1e-5) ** .5)
        torch.testing.assert_close(normalized, centered / denominator)
        self.assertAlmostEqual(x.var(unbiased=True).item(), 5 / 3)
        self.assertAlmostEqual(normalized.var(unbiased=False).item(), .999992, places=9)

    def test_layernorm_statistics_include_epsilon(self):
        inputs = torch.tensor([[1., 2., 3., 4.], [1., 1.002, 1.004, 1.006],
                               [3., 3., 3., 3.]], dtype=torch.float64)
        for eps in [1e-5, 1.0]:
            with self.subTest(eps=eps):
                out = blocks.ManualLayerNorm(4, eps).double()(inputs)
                variance = inputs.var(-1, unbiased=False)
                expected_variance = variance / (variance + eps)
                torch.testing.assert_close(out.mean(-1), torch.zeros(3).double(), rtol=0, atol=1e-12)
                torch.testing.assert_close(out.var(-1, unbiased=False), expected_variance,
                                           rtol=1e-10, atol=1e-12)
                self.assertTrue(torch.isfinite(out).all())
                self.assertEqual(torch.count_nonzero(out[-1]).item(), 0)
        # A small-variance input is NOT expected to have variance almost one.
        near_constant = blocks.ManualLayerNorm(4).double()(inputs[1])
        self.assertAlmostEqual(near_constant.var(unbiased=False).item(), 1 / 3, places=10)

    def test_layernorm_affine_and_official_forward_backward(self):
        custom = blocks.ManualLayerNorm(4, eps=.01).double()
        official = torch.nn.LayerNorm(4, eps=.01).double()
        with torch.no_grad():
            custom.gamma.copy_(torch.tensor([.5, 1., 1.5, 2.]))
            custom.beta.copy_(torch.tensor([-1., .5, 0., 2.]))
            official.weight.copy_(custom.gamma)
            official.bias.copy_(custom.beta)
        a = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
        b = a.detach().clone().requires_grad_()
        weights = torch.randn_like(a)
        actual, expected = custom(a), official(b)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
        (actual * weights).sum().backward()
        (expected * weights).sum().backward()
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(custom.gamma.grad, official.weight.grad)
        torch.testing.assert_close(custom.beta.grad, official.bias.grad)
        affine = blocks.ManualLayerNorm(4).double()
        with torch.no_grad():
            affine.gamma.fill_(2)
            affine.beta.fill_(1)
        y = affine(torch.tensor([1., 2., 3., 4.]).double())
        self.assertAlmostEqual(y.mean().item(), 1.)
        self.assertAlmostEqual(y.var(unbiased=False).item(), 4 * 1.25 / 1.25001)

    def test_layernorm_is_per_token_not_across_batch_or_time(self):
        ln = blocks.ManualLayerNorm(4).double()
        x = torch.randn(2, 3, 4, dtype=torch.float64)
        changed = x.clone()
        changed[1] *= 100
        changed[0, 1:] += 50
        torch.testing.assert_close(ln(x)[0, 0], ln(changed)[0, 0])
        torch.testing.assert_close(ln(x)[0, 0], ln(x[0, 0]))
        ln.eval()
        before = ln(x)
        ln.train()
        torch.testing.assert_close(ln(x), before)

    @staticmethod
    def zero_sublayers(model):
        with torch.no_grad():
            for sublayer in (model.attn, model.ffn):
                for param in sublayer.parameters():
                    param.zero_()

    def test_zero_sublayers_preserve_preln_identity(self):
        model = blocks.TransformerBlock(8, 2).double()
        self.assertTrue(model.norm_first)
        self.zero_sublayers(model)
        x = (torch.randn(2, 3, 8, dtype=torch.float64) + 5).requires_grad_()
        actual = model(x)
        torch.testing.assert_close(actual, x, rtol=0, atol=0)
        actual.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0, atol=0)

    def test_zero_sublayers_postln_is_not_identity(self):
        model = blocks.TransformerBlock(4, 2, norm_first=False).double()
        self.zero_sublayers(model)
        x = torch.tensor([[[1., 2., 3., 4.]]], dtype=torch.float64)
        expected = F.layer_norm(F.layer_norm(x, (4,), eps=1e-5), (4,), eps=1e-5)
        torch.testing.assert_close(model(x), expected)
        self.assertGreater((model(x) - x).abs().max().item(), 1.)

    def test_preln_and_postln_match_independent_formulas(self):
        x = torch.randn(2, 3, 4, dtype=torch.float64) * 3 + 2
        allowed = torch.ones(3, 3, dtype=torch.bool).tril()
        pre = blocks.TransformerBlock(4, 2).double()
        with torch.no_grad():
            pre.ln1.gamma.copy_(torch.tensor([.5, 1., 1.5, 2.]))
            pre.ln2.beta.copy_(torch.tensor([1., -1., .5, 0.]))
        post = blocks.TransformerBlock(4, 2, norm_first=False).double()
        post.load_state_dict(pre.state_dict())  # only ordering differs

        def norm(values, layer):
            return F.layer_norm(values, (4,), layer.gamma, layer.beta, layer.eps)

        def attention(values, model):
            return model.attn(values, values, values, attn_mask=~allowed, need_weights=False)[0]

        def ffn(values, model):
            hidden = F.linear(values, model.ffn.linear1.weight, model.ffn.linear1.bias)
            return F.linear(F.relu(hidden), model.ffn.linear2.weight, model.ffn.linear2.bias)

        u_pre = x + attention(norm(x, pre.ln1), pre)
        expected_pre = u_pre + ffn(norm(u_pre, pre.ln2), pre)
        u_post = norm(x + attention(x, post), post.ln1)
        expected_post = norm(u_post + ffn(u_post, post), post.ln2)
        torch.testing.assert_close(pre(x, allowed), expected_pre, rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(post(x, allowed), expected_post, rtol=1e-10, atol=1e-12)
        self.assertGreater((expected_pre - expected_post).abs().max().item(), .1)

    def test_single_sublayer_numeric_ordering(self):
        rows = blocks.ordering_walkthrough()
        x = torch.tensor([1., 2., 3., 4.]).double()
        normalized = (x - 2.5) / (1.25 + 1e-5) ** .5
        torch.testing.assert_close(rows[-1][1], x + 2 * normalized)
        post = (3 * x - 7.5) / (11.25 + 1e-5) ** .5
        torch.testing.assert_close(rows[-1][2], post)
        self.assertGreater((post - normalized).abs().max().item(), 1e-6)

    def test_ffn_nonlinearity_and_position_sharing(self):
        ffn = blocks.toy_ffn()
        x = torch.tensor([1., -2.]).double()
        torch.testing.assert_close(ffn.linear1(x), torch.tensor([1., -1., -1., -3.]).double())
        torch.testing.assert_close(ffn(x), torch.tensor([1.5, -.5]).double())
        torch.testing.assert_close(ffn.linear2(ffn.linear1(x)), torch.tensor([1.5, -6.5]).double())
        many = torch.randn(2, 3, 2, dtype=torch.float64)
        many[0, 0] = x
        many[1, 2] = x
        actual = ffn(many)
        torch.testing.assert_close(actual[0, 0], actual[1, 2])
        changed = many.clone()
        changed[1, 0] += 20
        torch.testing.assert_close(actual[0, 0], ffn(changed)[0, 0])
        # Collapse the affine layers without ReLU; validate the algebra too.
        collapsed_weight = ffn.linear2.weight @ ffn.linear1.weight
        collapsed_bias = ffn.linear2.weight @ ffn.linear1.bias + ffn.linear2.bias
        torch.testing.assert_close(F.linear(many, collapsed_weight, collapsed_bias),
                                   ffn.linear2(ffn.linear1(many)))

    def test_ffn_and_block_parameter_budget_includes_bias(self):
        for d_model, d_ff in [(2, 4), (16, 64), (8, 19)]:
            with self.subTest(d_model=d_model, d_ff=d_ff):
                ffn = blocks.FeedForwardNetwork(d_model, d_ff)
                self.assertEqual(sum(p.numel() for p in ffn.parameters()),
                                 2 * d_model * d_ff + d_ff + d_model)
                self.assertEqual(ffn.linear1.bias.numel(), d_ff)
                self.assertEqual(ffn.linear2.bias.numel(), d_model)
        model = blocks.TransformerBlock(16, 4)
        self.assertEqual(sum(p.numel() for p in model.ffn.parameters()), 2128)
        self.assertEqual(sum(p.numel() for p in model.attn.parameters()), 1088)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 3280)
        self.assertEqual(model.ffn(torch.zeros(2, 5, 16)).shape, (2, 5, 16))
        self.assertEqual(model.ffn(torch.zeros(2, 50, 16)).shape, (2, 50, 16))
        self.assertEqual(sum(p.numel() for p in model.ffn.parameters()), 2128)

    def test_forward_backward_finite_and_mask_causality(self):
        for norm_first in [True, False]:
            for b, t, d, heads in [(1, 1, 4, 1), (2, 5, 16, 4)]:
                with self.subTest(norm_first=norm_first, shape=(b, t, d)):
                    model = blocks.TransformerBlock(d, heads, norm_first=norm_first).double()
                    x = torch.randn(b, t, d, dtype=torch.float64, requires_grad=True)
                    allowed = torch.ones(t, t, dtype=torch.bool).tril()
                    out = model(x, allowed)
                    self.assertEqual(out.shape, x.shape)
                    self.assertTrue(torch.isfinite(out).all())
                    (out * torch.randn_like(out)).sum().backward()
                    self.assertTrue(torch.isfinite(x.grad).all())
                    for name, param in model.named_parameters():
                        self.assertIsNotNone(param.grad, name)
                        self.assertTrue(torch.isfinite(param.grad).all(), name)
                    if t > 1:
                        changed = x.detach().clone()
                        changed[:, 2:] += torch.randn_like(changed[:, 2:]) * 10
                        torch.testing.assert_close(out[:, :2], model(changed, allowed)[:, :2])
                        self.assertGreater((model(x.detach())[:, :2] - model(changed)[:, :2]).abs().max().item(), 1e-5)

    def test_generated_tables_and_script_output_match_chapter(self):
        chapter = (ROOT / 'chapters' / '06_residual_layernorm_ffn.md').read_text()
        tables = blocks.chapter_tables()
        self.assertEqual(len(tables), 7)
        for name, table in tables.items():
            with self.subTest(table=name):
                self.assertIn(table, chapter)
        stream = io.StringIO()
        with redirect_stdout(stream):
            blocks.main()
        output = stream.getvalue()
        for line in [
            'Z-score: mean=2.500000, variance=1.250000, denominator=1.118038',
            'normalized variance=0.999992',
            'Pre-LN: shape=[2, 5, 16], forward/backward finite=True',
            'Post-LN: shape=[2, 5, 16], forward/backward finite=True',
            'Pre-LN zero-sublayer identity max error=0.000000',
        ]:
            self.assertIn(line, output)
            self.assertIn(line, chapter)

    def test_invalid_dimensions_and_epsilon(self):
        for d in [0, -1, 2.5, True]:
            with self.assertRaises(ValueError):
                blocks.ManualLayerNorm(d)
        for eps in [0., -1., float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                blocks.ManualLayerNorm(4, eps)
        with self.assertRaises(ValueError):
            blocks.ManualLayerNorm(4)(torch.zeros(2, 3))
        with self.assertRaises(ValueError):
            blocks.FeedForwardNetwork(4, 0)
        for d, h in [(8, 0), (7, 2)]:
            with self.assertRaises(ValueError):
                blocks.TransformerBlock(d, h)
        with self.assertRaises(TypeError):
            blocks.TransformerBlock(4, 2, norm_first='false')


if __name__ == '__main__':
    unittest.main()
