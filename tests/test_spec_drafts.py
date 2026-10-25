"""P23 extra: block/cutoff/ECHO drafts keep greedy output; acceptance algebra; mutations."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

from torch_test_support import torch_status, torch_reason

READY, VERSION, PRESENT = torch_status()
SCRIPT = Path(__file__).resolve().parents[1] / 'code/24b_spec_drafts.py'
if READY:
    import torch
    import torch.nn.functional as F
    spec = importlib.util.spec_from_file_location('spec_drafts_test', SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)


@unittest.skipUnless(READY, torch_reason('P23 drafts', VERSION, PRESENT))
class SpecDraftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.target = m.train_target(steps=60)
        cls.prompt = torch.tensor([[2, 7]])
        cls.base = m.s6.greedy_decode(cls.target, cls.prompt, 10)

    def check_same(self, draft):
        r = m.s6.greedy_decode(self.target, self.prompt, 10, draft=draft, draft_tokens=3)
        self.assertTrue(torch.equal(r.tokens, self.base.tokens))
        return r

    def test_dds_loss_weight_is_not_training_frequency(self):
        """echo_loss averages over 1..D, so the expected weight of horizon k is E[1{D>=k}/D], not P(D>=k)."""
        from fractions import Fraction as Fr
        pi = [Fr(6, 10), Fr(3, 10), Fr(1, 10)]
        w = [sum(p / d for d, p in enumerate(pi, 1) if d >= k) for k in (1, 2, 3)]
        self.assertEqual(w, [Fr(47, 60), Fr(11, 60), Fr(1, 30)])
        for a, b in zip(m.DDS_W, w):
            self.assertAlmostEqual(a, float(b), places=12)
        tg = self.target
        head = m.ECHOHead(tg)
        idx = torch.tensor([[1, 4, 2, 7, 0, 1, 4, 2, 7]])
        with torch.no_grad():
            per = [F.cross_entropy(head.logits(h[:, :-1]).flatten(0, 1), m.horizon_labels(idx, k).flatten())
                   for k, h in enumerate(head.horizons(idx, 3), 1)]
            self.assertTrue(torch.allclose(m.echo_loss(head, idx, 3), torch.stack(per).mean()))

    def test_block_and_cutoff_keep_greedy_output(self):
        self.check_same(m.BlockDraft(self.target))
        full_cut = m.BlockDraft(self.target, tau=1.01)
        r = self.check_same(full_cut)
        self.assertEqual(r.proposed, 0)
        self.assertGreater(full_cut.cut, 0)

    def test_wrong_drafts_are_corrected(self):
        def bad(prefix, limit):
            return torch.full((1, limit), 5, dtype=torch.long)
        r = self.check_same(bad)
        self.assertGreater(r.rejected_rounds, 0)

    def test_echo_draft_and_dds(self):
        head, curve = m.train_echo(self.target, steps=80)
        self.assertLess(curve[-1], curve[0])
        self.check_same(m.ECHODraft(head))
        self.assertAlmostEqual(sum(m.DDS_PI), 1.0)
        self.assertAlmostEqual(sum((i + 1) * p for i, p in enumerate(m.DDS_PI)), 1.5)
        f = [sum(m.DDS_PI[k:]) for k in range(3)]          # paper eq. 4: f_k = P(D >= k)
        for got, want in zip(f, m.DDS_F):
            self.assertAlmostEqual(got, want)

    def test_echo_shares_final_layer_and_only_fusion_trains(self):
        head = m.ECHOHead(self.target)
        self.assertIs(head.target.blocks[-1], self.target.blocks[-1])
        own = [n for n, _ in head.named_parameters() if not n.startswith('target.')]
        self.assertEqual(sorted(own), ['fuse.weight', 'norm_e.weight', 'norm_h.weight'])

    def test_horizon_k_consumes_token_t_plus_k(self):
        """Catches the looped variant: h^(k)_t must read x_{t+k}, and h^(1)_t must not read x_{t+2}."""
        torch.manual_seed(0)
        head = m.ECHOHead(self.target).eval()
        idx = torch.tensor([[1, 4, 2, 7, 0, 1, 4, 2]])
        t = 2
        with torch.no_grad():
            h1, h2 = head.horizons(idx, 2)
            bumped = idx.clone()
            bumped[0, t + 2] = (bumped[0, t + 2] + 3) % 11
            g1, g2 = head.horizons(bumped, 2)
        self.assertEqual(h1.shape[1], idx.shape[1] - 1)
        self.assertEqual(h2.shape[1], idx.shape[1] - 2)
        self.assertTrue(torch.allclose(h1[0, t], g1[0, t], atol=1e-5))          # causal: x_{t+2} unseen at horizon 1
        self.assertFalse(torch.allclose(h2[0, t], g2[0, t]))         # horizon 2 fuses x_{t+2}
        self.assertTrue(torch.allclose(h2[0, :t], g2[0, :t], atol=1e-5))

    def test_horizon_labels_and_wrong_label_mutation(self):
        idx = torch.tensor([[10, 11, 12, 13, 14]])
        self.assertEqual(m.horizon_labels(idx, 1).tolist(), [[12, 13, 14]])
        self.assertEqual(m.horizon_labels(idx, 2).tolist(), [[13, 14]])
        head, _ = m.train_echo(self.target, steps=80)
        held = m.pattern_batch(32, 12, gen=torch.Generator().manual_seed(99))
        self.assertGreater(m.horizon_accuracy(head, held, 2), 0.9)
        with torch.no_grad():
            logits = head.logits(head.horizons(held, 2)[-1][:, :-1])
        same_position = held[:, 2:-1]   # mutation: label x_{t+2} (looped variant) instead of x_{t+3}
        self.assertLess((logits.argmax(-1) == same_position).float().mean().item(), 0.5)

    def test_acceptance_target_matches_monte_carlo(self):
        g = torch.Generator().manual_seed(3)
        p = torch.softmax(torch.randn(9, generator=g, dtype=torch.float64), 0)
        q = torch.softmax(torch.randn(9, generator=g, dtype=torch.float64), 0)
        c = m.acceptance_target(p, q).item()
        self.assertAlmostEqual(c, torch.minimum(p, q).sum().item(), places=12)
        mc = m.speculative_accept_rate(p, q, 100000, g)
        self.assertLess(abs(c - mc), 0.01)
        wrong = 1 - (p - q).abs().sum().item()  # mutation: forgot the 1/2
        self.assertGreater(abs(wrong - mc), 0.05)

    def test_expected_tau(self):
        self.assertAlmostEqual(m.expected_tau((0.8, 0.6, 0.4)), 1 + 0.8 + 0.48 + 0.192)
        self.assertEqual(m.expected_tau(()), 1.0)
        self.assertNotAlmostEqual(m.expected_tau((0.8, 0.6, 0.4)), 1 + 0.8 + 0.6 + 0.4)
        g = torch.Generator().manual_seed(5)
        a = (0.8, 0.6, 0.4)
        draws = torch.rand(50000, 3, generator=g) < torch.tensor(a)
        accepted = draws.int().cumprod(1).sum(1).float() + 1
        self.assertLess(abs(accepted.mean().item() - m.expected_tau(a)), 0.02)

    def test_cli_runs(self):
        out = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, check=True).stdout
        self.assertIn('ECHO-style draft: identical=True', out)
        self.assertIn('no speedup claimed', out)
