"""P23 extra: FlashMap joint-vs-independent teaching toy + lossless verify."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

from torch_test_support import torch_status, torch_reason

READY, VERSION, PRESENT = torch_status()
SCRIPT = Path(__file__).resolve().parents[1] / "code/24e_flashmap_joint.py"
if READY:
    import torch
    spec = importlib.util.spec_from_file_location("flashmap_joint_test", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)


@unittest.skipUnless(READY, torch_reason("P23 FlashMap joint toy", VERSION, PRESENT))
class FlashMapJointTests(unittest.TestCase):
    def test_joint_beats_independent_accept_length(self):
        g_ind = m.mean_accept(m.sample_independent_block, n=30000, seed=11)
        g_joint = m.mean_accept(m.sample_joint_block, n=30000, seed=11)
        self.assertGreater(g_joint, g_ind + 0.15)
        # Exact marginals: P(match slot0)=1/2, and given match P(match slot1)=1
        # for joint vs 1/2 for independent → E[len] joint=1.0, ind=0.75 under
        # independent target/draft seeds; we only assert the inequality.

    def test_shuffle_delta_positive_for_joint(self):
        g_s, g_f, delta = m.shuffle_delta(n=30000, seed=12)
        self.assertGreater(delta, 0.15)
        self.assertGreater(g_s, g_f)
        # Factorized drafter: shared == shuffled (Δ≈0), like DFlash in Table 2.
        g_ind_a = m.mean_accept(m.sample_independent_block, n=20000, seed=13)
        g_ind_b = m.mean_accept(m.sample_shuffled_block, n=20000, seed=13)
        self.assertAlmostEqual(g_ind_a, g_ind_b, places=2)

    def test_accept_prefix_length_stops_at_first_mismatch(self):
        draft = torch.tensor([[0, 1], [0, 3], [2, 3]])
        target = torch.tensor([[0, 1], [0, 1], [0, 3]])
        got = m.accept_prefix_length(draft, target).tolist()
        self.assertEqual(got, [2.0, 1.0, 0.0])

    def test_joint_block_draft_keeps_greedy_output(self):
        torch.set_num_threads(1)
        target = m.train_tiny_target(steps=60, seed=106)
        prompt = torch.tensor([[2, 7]])
        base = m.s6.greedy_decode(target, prompt, 10)
        draft = m.JointBlockDraft(base.tokens, noise=0.4, seed=5)
        out = m.s6.greedy_decode(target, prompt, 10, draft=draft, draft_tokens=m.K)
        self.assertTrue(torch.equal(out.tokens, base.tokens))
        self.assertGreater(draft.calls, 0)
        # Wrong joint still corrected.
        base8 = m.s6.greedy_decode(target, prompt, 8)
        bad = m.JointBlockDraft(
            torch.full_like(base8.tokens, 3), noise=0.0, seed=0
        )
        out_bad = m.s6.greedy_decode(target, prompt, 8, draft=bad, draft_tokens=m.K)
        self.assertTrue(torch.equal(out_bad.tokens, base8.tokens))

    def test_cli_runs(self):
        out = subprocess.run(
            [sys.executable, str(SCRIPT)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        self.assertIn("joint", out.lower())
        self.assertIn("identical to baseline", out)
        self.assertIn("OK", out)


if __name__ == "__main__":
    unittest.main()
