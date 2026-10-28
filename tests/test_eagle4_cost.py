"""P23 extra: EAGLE-4 cost-model k selection, low-rank head, lossless greedy verify."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

from torch_test_support import torch_status, torch_reason

READY, VERSION, PRESENT = torch_status()
SCRIPT = Path(__file__).resolve().parents[1] / 'code/24d_eagle4_cost.py'
if READY:
    import torch
    spec = importlib.util.spec_from_file_location('eagle4_cost_test', SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)


@unittest.skipUnless(READY, torch_reason('P23 EAGLE-4 cost', VERSION, PRESENT))
class Eagle4CostTests(unittest.TestCase):
    def test_expected_accepted_matches_geometric_series(self):
        self.assertEqual(m.expected_accepted(0.5, 0), 1.0)
        self.assertAlmostEqual(m.expected_accepted(0.5, 3), 1 + 0.5 + 0.25 + 0.125)
        self.assertAlmostEqual(m.expected_accepted(1.0, 4), 5.0)
        self.assertAlmostEqual(m.expected_accepted(0.0, 5), 1.0)
        with self.assertRaises(ValueError):
            m.expected_accepted(1.1, 2)

    def test_choose_draft_length_shrinks_when_verify_gets_expensive(self):
        p, c_d = 0.7, 0.15
        k_small = m.choose_draft_length(p, c_d, c0=1.0, c1=0.05)
        k_mid = m.choose_draft_length(p, c_d, c0=1.0, c1=0.25)
        k_large = m.choose_draft_length(p, c_d, c0=1.0, c1=1.20)
        self.assertGreater(k_small, 0)
        self.assertGreaterEqual(k_small, k_mid)
        self.assertGreaterEqual(k_mid, k_large)
        # High concurrent load should stop drafting altogether under these numbers.
        self.assertEqual(k_large, 0)
        rows = m.choose_across_batches(p, c_d, [
            ("small", 1.0, 0.05), ("mid", 1.0, 0.25), ("large", 1.0, 1.20),
        ])
        self.assertEqual([k for _, k, _ in rows], [k_small, k_mid, k_large])

    def test_delta_a_over_delta_c_threshold_stops_depth(self):
        # With tau very large, even the first step fails ΔA/ΔC ≥ τ.
        self.assertEqual(m.choose_draft_length(0.9, 0.1, 1.0, 0.01, tau=100.0), 0)
        # p=0.6, c_d=0.2: p^k/c_d ≥ 1 holds only while p^k ≥ 0.2 (k ≤ 3).
        k = m.choose_draft_length(0.6, 0.2, 1.0, 0.01, k_max=12, tau=1.0)
        self.assertGreaterEqual(k, 1)
        self.assertLessEqual(k, 4)
        # Mutation: forgetting the geometric decay (using p instead of p^k) would
        # keep drafting forever up to k_max when p/c_d ≥ tau.
        fake = 0
        p, c_d, tau = 0.6, 0.2, 1.0
        for depth in range(1, 13):
            if p / c_d < tau:  # wrong: no p**depth
                break
            fake = depth
        self.assertEqual(fake, 12)
        self.assertLess(k, fake)

    def test_low_rank_top1_hit_and_mul_savings(self):
        rate, full_muls, two_muls = m.low_rank_hit_rate(
            vocab=5000, d=64, rank=32, top_m=128, trials=80, seed=7)
        self.assertGreater(rate, 0.80)         # SVD init of W (paper's natural init)
        self.assertLess(two_muls, full_muls)
        # Exact arithmetic of mul counts.
        self.assertEqual(full_muls, 5000 * 64)
        self.assertEqual(two_muls, 5000 * 32 + 32 * 64 + 128 * 64)
        g = torch.Generator().manual_seed(0)
        W = torch.randn(50, 8, generator=g, dtype=torch.float64)
        h = torch.randn(8, generator=g, dtype=torch.float64)
        exact, two, hit, _, _ = m.low_rank_candidate_head(W, h, rank=4, top_m=8)
        self.assertEqual(exact, int((W @ h).argmax().item()))
        self.assertIn(two, range(50))
        self.assertIsInstance(hit, bool)
        # Random init is much worse — catches a regression that swaps SVD for noise.
        bad = 0
        for i in range(40):
            gi = torch.Generator().manual_seed(100 + i)
            Wi = torch.randn(200, 16, generator=gi, dtype=torch.float64)
            hi = torch.randn(16, generator=gi, dtype=torch.float64)
            _, _, hiti, _, _ = m.low_rank_candidate_head(
                Wi, hi, rank=4, top_m=8, generator=gi, init="random")
            bad += int(hiti)
        self.assertLess(bad / 40, 0.5)

    def test_cost_chosen_k_keeps_greedy_output(self):
        torch.set_num_threads(1)
        target = m.train_tiny_target(steps=60, seed=460)
        prompt = torch.tensor([[2, 7]])
        base = m.s6.greedy_decode(target, prompt, 10)
        k = m.choose_draft_length(0.7, 0.15, 1.0, 0.05)
        self.assertGreater(k, 0)
        draft = m.FixedProbDraft(target, base.tokens)
        out = m.s6.greedy_decode(target, prompt, 10, draft=draft, draft_tokens=k)
        self.assertTrue(torch.equal(out.tokens, base.tokens))
        self.assertLess(out.target_calls, base.target_calls)
        # No-draft baseline path (omit draft=) still matches the same tokens.
        plain = m.s6.greedy_decode(target, prompt, 10)
        self.assertTrue(torch.equal(plain.tokens, base.tokens))

    def test_wrong_draft_still_corrected(self):
        torch.set_num_threads(1)
        target = m.train_tiny_target(steps=40, seed=461)
        prompt = torch.tensor([[2, 7]])
        base = m.s6.greedy_decode(target, prompt, 8)

        def bad(prefix, limit):
            return torch.full((1, limit), 3, dtype=torch.long)

        out = m.s6.greedy_decode(target, prompt, 8, draft=bad, draft_tokens=3)
        self.assertTrue(torch.equal(out.tokens, base.tokens))
        self.assertGreater(out.rejected_rounds, 0)

    def test_cli_runs(self):
        out = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, check=True).stdout
        self.assertIn('Cost-model chain length', out)
        self.assertIn('identical to baseline', out)
        self.assertIn('no 27B speedup claimed', out)
        self.assertIn('source-available', out)
