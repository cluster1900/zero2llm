"""P16 MSA-style block top-k: pooling, local block, group sharing, recall, mutations."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

from torch_test_support import torch_status, torch_reason

READY, VERSION, PRESENT = torch_status()
SCRIPT = Path(__file__).resolve().parents[1] / 'code/17b_msa_block_topk.py'
if READY:
    import torch
    spec = importlib.util.spec_from_file_location('msa_test', SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)


@unittest.skipUnless(READY, torch_reason('P16 MSA', VERSION, PRESENT))
class MSATests(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(2)
        self.k = torch.randn(70, 4, dtype=torch.float64, generator=g)
        self.q = torch.randn(4, 4, dtype=torch.float64, generator=g)

    def test_max_pool_by_hand_and_partial_block(self):
        q = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        k = torch.tensor([[1, 0], [3, 0], [-2, 0], [5, 0], [4, 0]], dtype=torch.float64)
        self.assertEqual(m.block_scores(q, k, 2).tolist(), [[3.0, 5.0, 4.0]])

    def test_local_block_always_kept_and_group_shared(self):
        chosen = m.select_blocks(self.q, self.k, 8, 1, 2)
        self.assertEqual(chosen, [[8], [8]])
        chosen = m.select_blocks(self.q, self.k, 8, 3, 2)
        for c in chosen:
            self.assertIn(8, c)
            self.assertEqual(len(c), 3)

    def test_full_budget_gives_full_recall(self):
        self.assertEqual(m.token_recall(self.q, self.k, 8, 9, 2, 5), 1.0)
        self.assertEqual(m.token_recall(self.q, self.k, 1, 70, 4, 5), 1.0)

    def test_mutation_mean_pool_changes_choice(self):
        q = torch.tensor([[1.0]], dtype=torch.float64)
        k = torch.tensor([[9], [-9], [1], [1], [0], [0]], dtype=torch.float64)
        self.assertEqual(m.select_blocks(q, k, 2, 2, 1), [[0, 2]])   # max-pool finds the spike
        mean = (q @ k.T).reshape(1, -1, 2).mean(-1)
        self.assertEqual(int(mean[0, :2].argmax()), 1)                # mean-pool would miss it

    def test_ratio_and_rejects(self):
        self.assertEqual(m.attention_flops_ratio(1 << 20, 128, 16), 512)
        self.assertEqual(m.attention_flops_ratio(100, 128, 16), 1)
        with self.assertRaises(ValueError):
            m.select_blocks(self.q, self.k, 8, 2, 3)
        with self.assertRaises(ValueError):
            m.block_scores(self.q, self.k, 0)

    def test_cli_runs(self):
        out = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, check=True).stdout
        self.assertIn('k=16: top-16 token recall', out)
