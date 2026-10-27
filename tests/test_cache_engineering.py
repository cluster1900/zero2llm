"""P11 extra: cache engineering (arXiv:2610.28577) drop / repos vs re-prefill."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

from torch_test_support import torch_status, torch_reason

READY, VERSION, PRESENT = torch_status()
ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'code/12b_cache_engineering.py'

if READY:
    import torch
    spec = importlib.util.spec_from_file_location('cache_eng_test', SCRIPT)
    ce = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ce
    spec.loader.exec_module(ce)


@unittest.skipUnless(READY, torch_reason('P11 cache engineering', VERSION, PRESENT))
class CacheEngineeringTests(unittest.TestCase):
    def test_append_without_operations_matches_full_forward(self):
        model = ce.make_model(2)
        full, _ = model(torch.tensor([ce.TOKENS + ce.QUERY]))
        _, cache = ce.append(model, None, ce.TOKENS)
        tail, _ = ce.append(model, cache, ce.QUERY)
        torch.testing.assert_close(tail, full[:, -len(ce.QUERY):], rtol=0, atol=1e-12)

    def test_rerotate_equals_rotating_raw_key_at_new_position(self):
        torch.manual_seed(0)
        raw = torch.randn(1, 2, 5, 8, dtype=torch.float64)
        old, new = torch.tensor([3, 6, 7, 8, 9]), torch.tensor([0, 1, 2, 3, 4])
        moved = ce.rerotate(ce.apply_rope(raw, old), old, new)
        torch.testing.assert_close(moved, ce.apply_rope(raw, new), rtol=0, atol=1e-12)

    def test_drop_mask_equals_physically_removing_entries(self):
        model = ce.make_model(2)
        _, cache = ce.append(model, None, ce.TOKENS)
        masked = ce.drop(cache, ce.DROPPED)
        keep = masked.active
        removed = ce.ActiveCache([(k[:, :, keep], v[:, :, keep]) for k, v in cache.layers],
                                 torch.ones(int(keep.sum()), dtype=torch.bool), cache.key_pos[keep], cache.next_pos)
        a, _ = ce.append(model, masked, ce.QUERY)
        b, _ = ce.append(model, removed, ce.QUERY)
        torch.testing.assert_close(a, b, rtol=0, atol=1e-12)
        self.assertEqual(masked.next_pos, len(ce.TOKENS))  # drop does not reset positions

    def test_one_layer_drop_repos_equals_reprefill(self):
        rows, gap = ce.compare(1)
        self.assertLess(max(rows[0]), 1e-12)
        self.assertLess(gap, 1e-12)

    def test_deeper_layers_carry_dropped_context(self):
        for layers in (2, 4):
            rows, gap = ce.compare(layers)
            self.assertLess(max(rows[0]), 1e-12)              # layer 0 depends on token + position only
            for dk, dv in rows[1:]:
                self.assertGreater(min(dk, dv), 1e-2)
            self.assertGreater(gap, 1e-3)

    def test_only_tokens_after_the_dropped_span_differ(self):
        gaps = ce.per_token_gap()
        for i in (0, 1):                                      # never attended to the dropped span
            self.assertLess(gaps[i], 1e-12)
        for i in (6, 7, 8, 9):
            self.assertGreater(gaps[i], 1e-2)

    def test_dropping_a_suffix_is_exact_at_every_layer(self):
        rows, gap = ce.compare(4, dropped=[8, 9])
        self.assertLess(max(max(r) for r in rows), 1e-12)
        self.assertLess(gap, 1e-12)

    def test_without_repos_positions_differ_even_for_one_layer(self):
        rows, gap = ce.compare(1, use_repos=False)
        self.assertGreater(rows[0][0], 1e-1)
        self.assertGreater(gap, 1e-3)

    def test_mutation_noop_repos_breaks_one_layer_match(self):
        with mock.patch.object(ce, 'repos', lambda model, cache: cache):
            _, gap = ce.compare(1)
        self.assertGreater(gap, 1e-3)

    def test_cli_runs(self):
        out = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, check=True,
                             env={'OMP_NUM_THREADS': '1', 'PATH': ''}).stdout
        self.assertIn('query logits max|diff|', out)


if __name__ == '__main__':
    unittest.main()
