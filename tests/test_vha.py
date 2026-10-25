"""P14 VHA: absorption identity, Postmix identity at B=0, cache bytes, and mutations."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

from torch_test_support import torch_status, torch_reason

READY, VERSION, PRESENT = torch_status()
ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'code/15b_vha.py'
if READY:
    import torch
    spec = importlib.util.spec_from_file_location('vha_test', SCRIPT)
    v = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = v
    spec.loader.exec_module(v)


@unittest.skipUnless(READY, torch_reason('P14 VHA', VERSION, PRESENT))
class VHATests(unittest.TestCase):
    def setUp(self):
        self.x = torch.randn(6, 16, dtype=torch.float64, generator=torch.Generator().manual_seed(1))

    def test_training_matches_absorbed_gqa(self):
        for hq, hkv in ((4, 2), (2, 2), (3, 1)):
            p = v.make_params(hq=hq, hkv=hkv, seed=hq)
            err = (v.training_form(self.x, p) - v.gqa_form(self.x, v.absorb(p))).abs().max().item()
            self.assertLess(err, 1e-5)
            self.assertEqual(v.absorb(p)['WK'].shape[0], hkv)

    def test_b_zero_postmix_is_identity(self):
        p = v.make_params(postmix_b_zero=True)
        no_post = dict(p, A=torch.zeros_like(p['A']))
        self.assertTrue(torch.allclose(v.training_form(self.x, p), v.training_form(self.x, no_post), atol=0, rtol=0))
        self.assertTrue(torch.equal(v.absorb(p)['WO'], p['WO']))

    def test_mutations_are_caught(self):
        p = v.make_params()
        ref = v.training_form(self.x, p)
        good = v.absorb(p)
        hv, d = good['WQ'].shape[0], p['d']
        wrong_order = (torch.kron(torch.eye(hv, dtype=torch.float64) + p['B'] @ p['A'].T, torch.eye(d, dtype=torch.float64))
                       @ p['WO'].reshape(hv * d, -1)).reshape(hv, d, -1)
        no_premix = torch.stack([p['WQ'][i] for i in range(p['hq']) for _ in range(p['hkv'])])
        for mutant in (dict(good, WO=wrong_order), dict(good, WQ=no_premix)):
            self.assertGreater((v.gqa_form(self.x, mutant) - ref).abs().max().item(), 1e-4)

    def test_postmix_rank_is_half_physical_query_heads(self):
        for hq, hkv in ((4, 2), (8, 2), (16, 4)):
            p = v.make_params(hq=hq, hkv=hkv)
            self.assertEqual(p['r'], hq // 2)
            self.assertNotEqual(p['r'], hq * hkv // 2)
            self.assertEqual(tuple(p['A'].shape), (hq * hkv, hq // 2))

    def test_cache_bytes_and_rejects(self):
        self.assertEqual(v.kv_cache_bytes(8, 4, 1), 128)
        self.assertEqual(v.kv_cache_bytes(2, 4, 1), 32)
        with self.assertRaises(ValueError):
            v.make_params(hkv=0)

    def test_cli_runs(self):
        out = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, check=True).stdout
        self.assertIn('absorbed GQA-2', out)
        self.assertIn('32 bytes', out)
