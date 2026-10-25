"""P24 mixed-bit: round-trip by hand, bytes, error ordering, allocation, mutations."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

from torch_test_support import torch_status, torch_reason

READY, VERSION, PRESENT = torch_status()
SCRIPT = Path(__file__).resolve().parents[1] / 'code/25b_mixed_bit.py'
if READY:
    import torch
    spec = importlib.util.spec_from_file_location('mixed_bit_test', SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)


@unittest.skipUnless(READY, torch_reason('P24 mixed-bit', VERSION, PRESENT))
class MixedBitTests(unittest.TestCase):
    def test_hand_roundtrip(self):
        w = torch.tensor([[7.0, 3.4, -1.0, 0.0]], dtype=torch.float64)
        self.assertEqual(m.group_quant(w, 4, 4).tolist(), [[7.0, 3.0, -1.0, 0.0]])
        self.assertEqual(m.group_quant(torch.zeros(1, 4, dtype=torch.float64), 4, 2).tolist(), [[0.0] * 4])

    def test_bytes(self):
        self.assertEqual(m.nbytes((64, 256), 4, 32), 8192 + 1024)
        self.assertEqual(m.nbytes((64, 256), 6, 128), 12288 + 256)

    def test_error_ordering(self):
        w = m.make_tensors()['down']
        e = {(b, g): m.rel_err(w, m.group_quant(w, b, g)) for b in (4, 6) for g in (32, 128)}
        self.assertLess(e[6, 32], e[4, 32])
        self.assertLess(e[4, 32], e[4, 128])
        self.assertLess(e[6, 128], e[6, 32] * 3)

    def test_mixed_plan_between_uniform_plans(self):
        t = m.make_tensors()
        lo = m.plan_error(t, {k: (4, 32) for k in t})
        hi = m.plan_error(t, {k: (6, 128) for k in t})
        mid = m.plan_error(t, {'body': (4, 32), 'v': (6, 128), 'down': (4, 32)})
        self.assertTrue(hi[0] < mid[0] < lo[0])
        self.assertTrue(lo[1] < mid[1] < hi[1])

    def test_mutation_unclamped_codes_detected(self):
        w = m.make_tensors()['v']
        qmax = 7
        g = w.reshape(64, -1, 32)
        bad_scale = g.abs().mean(-1, keepdim=True) / qmax     # mutation: mean instead of max
        clipped = (torch.clamp(torch.round(g / bad_scale), -qmax, qmax) * bad_scale).reshape_as(w)
        self.assertGreater(m.rel_err(w, clipped), 2 * m.rel_err(w, m.group_quant(w, 4, 32)))

    def test_rejects_and_cli(self):
        with self.assertRaises(ValueError):
            m.group_quant(torch.zeros(2, 30), 4, 32)
        out = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, check=True).stdout
        self.assertIn('-> upgrade', out)
