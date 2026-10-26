"""P04/P22 Softmax Collapse and Cancellation (arXiv:2610.31642): cliffs, gradients, mutations."""
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

from torch_test_support import torch_status, torch_reason

READY, VERSION, PRESENT = torch_status()
ROOT = Path(__file__).resolve().parents[1]
P04 = ROOT / 'code/03b_softmax_collapse.py'
P22 = ROOT / 'code/23c_softmax_cancellation.py'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


if READY:
    import torch
    c = load(P04, 'softmax_collapse_test')
    k = load(P22, 'softmax_cancel_test')


@unittest.skipUnless(READY, torch_reason('P04/P22 softmax fix', VERSION, PRESENT))
class SoftmaxCollapseTests(unittest.TestCase):
    def test_measured_cliffs_match_paper_and_unit_roundoff(self):
        # p counts the implicit leading bit: 8 / 24 / 53 (not 7 / 23 / 52).
        for dtype, mant in ((torch.bfloat16, 8), (torch.float32, 24), (torch.float64, 53)):
            measured = c.measured_cliff(dtype)
            self.assertAlmostEqual(measured, c.PAPER_CLIFFS[dtype], delta=0.1)
            self.assertAlmostEqual(measured, mant * math.log(2), delta=0.02)
            # Without the implicit bit the match fails: 7/23/52 * ln2 = 4.85/15.94/36.04.
            self.assertGreater(abs(measured - (mant - 1) * math.log(2)), 0.5)

    def test_margin_by_hand(self):
        # 9 - log(e^-9 + e^-9) = 18 - log 2
        self.assertAlmostEqual(c.margin(torch.tensor([9.0, -9.0, -9.0]), 0), 18 - math.log(2), places=12)

    def test_paper_example_standard_collapses_fixed_does_not(self):
        r = c.compare([9.0, -9.0, -9.0], torch.float32)
        self.assertEqual(r['standard']['grad'][0], 0.0)
        self.assertGreater(r['standard']['sum'], 0.0)
        self.assertAlmostEqual(r['fixed']['loss'], 3.046e-8, delta=1e-11)
        self.assertAlmostEqual(r['fixed']['grad'][0], -3.046e-8, delta=1e-11)
        self.assertEqual(r['fixed']['sum'], 0.0)

    def test_bf16_collapses_at_much_smaller_margin(self):
        r = c.compare([4.0, -4.0, -4.0], torch.bfloat16)
        self.assertEqual(r['standard']['grad'][0], 0.0)
        self.assertLess(r['fixed']['grad'][0], 0.0)
        r32 = c.compare([4.0, -4.0, -4.0], torch.float32)
        self.assertLess(r32['standard']['grad'][0], 0.0)

    def test_fixed_log_softmax_matches_float64_away_from_cliff(self):
        g = torch.Generator().manual_seed(0)
        z = torch.randn(5, 7, generator=g, dtype=torch.float64)
        torch.testing.assert_close(c.fixed_log_softmax(z), torch.log_softmax(z, -1), rtol=0, atol=1e-12)
        z = torch.tensor([9.0, -9.0, -9.0])
        self.assertLess(float(c.fixed_log_softmax(z)[0]), 0.0)
        self.assertEqual(float(torch.log_softmax(z, -1)[0]), 0.0)

    def test_mutant_summing_target_in_repo_helper_reintroduces_collapse(self):
        orig = c._shifted_and_rest

        def mutant(z, idx):  # form 1 + e^{-m} first (target kept in the sum), then remove the 1
            shifted, _ = orig(z, idx)
            return shifted, torch.log(torch.expm1(torch.logsumexp(shifted, -1, keepdim=True)))

        z = torch.tensor([9.0, -9.0, -9.0])
        loss, _ = c.grad_of(c.fixed_cross_entropy, z, 0)
        self.assertGreater(float(loss), 0.0)
        with mock.patch.object(c, '_shifted_and_rest', mutant):
            loss, g = c.grad_of(c.fixed_cross_entropy, z, 0)
        self.assertEqual(float(loss), 0.0)
        self.assertFalse(float(g[0]) < 0.0)  # target gradient lost (0 or nan), no longer -3.046e-8

    def test_cli(self):
        out = subprocess.run([sys.executable, str(P04)], capture_output=True, text=True, check=True).stdout
        self.assertIn('fp32: measured cliff 16.64, paper 16.6', out)
        self.assertIn('fixed    loss=3.046e-08', out)


@unittest.skipUnless(READY, torch_reason('P04/P22 softmax fix', VERSION, PRESENT))
class SoftmaxCancellationTests(unittest.TestCase):
    def test_paper_example_fixed_equals_exact_and_page_values(self):
        z_S, z_T = k.paper_case()
        fixed = k.student_grad(k.soft_target_cross_entropy, z_S, z_T)
        exact = k.exact_grad(z_S, z_T)
        for a, b, p in zip(fixed, exact, k.PAPER_EXAMPLE['fixed']):
            self.assertAlmostEqual(a, b, delta=2e-11)
            self.assertAlmostEqual(a, p, delta=1e-11)

    def test_standard_has_large_relative_error(self):
        z_S, z_T = k.paper_case()
        std = k.student_grad(k.standard_soft_ce, z_S, z_T)
        exact = k.exact_grad(z_S, z_T)
        self.assertGreater(abs(std[2] - exact[2]) / abs(exact[2]), 0.5)

    def test_identical_logits_give_exact_zero_only_when_fixed(self):
        g = torch.Generator().manual_seed(0)
        z = torch.randn(4, 8, generator=g).to(torch.bfloat16) * 4
        std = [k.student_grad(k.standard_soft_ce, r, r) for r in z]
        fix = [k.student_grad(k.soft_target_cross_entropy, r, r) for r in z]
        self.assertGreater(max(abs(x) for row in std for x in row), 0.0)
        self.assertEqual(max(abs(x) for row in fix for x in row), 0.0)

    def test_forward_loss_unchanged_and_gradient_sums_to_zero(self):
        g = torch.Generator().manual_seed(1)
        z_T = torch.randn(6, dtype=torch.float64, generator=g)
        z_S = torch.randn(6, dtype=torch.float64, generator=g)
        self.assertAlmostEqual(float(k.soft_target_cross_entropy(z_S, z_T)), float(k.standard_soft_ce(z_S, z_T)), places=12)
        grad = k.student_grad(k.soft_target_cross_entropy, z_S, z_T)
        for a, b in zip(grad, k.exact_grad(z_S, z_T)):
            self.assertAlmostEqual(a, b, places=12)  # far branch: u = q - p
        self.assertAlmostEqual(sum(grad), 0.0, places=14)

    def test_mutant_without_zero_sum_projection_leaves_larger_residual(self):
        # Patch the real repo helper to a no-op: the final g = u - q*sum(u) step is removed.
        g = torch.Generator().manual_seed(0)
        z_T = torch.randn(256, 8, generator=g) * 3
        z_S = z_T + 1e-4 * torch.randn(256, 8, generator=g)

        def residual():
            z = z_S.clone().requires_grad_(True)
            k.soft_target_cross_entropy(z, z_T).sum().backward()
            return float(z.grad.sum(-1).abs().mean())

        real = residual()
        with mock.patch.object(k, 'remove_leftover_sum', lambda diff, q: diff):
            mutant = residual()
        self.assertLess(real, 0.6 * mutant)  # measured: ~4.6e-13 vs ~1.1e-12 (fp32)

    def test_cli(self):
        out = subprocess.run([sys.executable, str(P22)], capture_output=True, text=True, check=True).stdout
        self.assertIn('fixed    [-4.596e-08, 4.144e-08, 4.513e-09]', out)
        self.assertIn('fixed    max|g| = 0.000e+00', out)


if __name__ == '__main__':
    unittest.main()
