"""P16 KDA: recurrence vs chunked form, bounded decay range, mutations."""
import importlib.util
import math
from pathlib import Path
import random
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'code/17c_kda_hybrid.py'
spec = importlib.util.spec_from_file_location('kda_test', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def unit(rnd, n):
    x = [rnd.gauss(0, 1) for _ in range(n)]
    s = math.sqrt(sum(a * a for a in x))
    return [a / s for a in x]


class KDATests(unittest.TestCase):
    def setUp(self):
        rnd = random.Random(7)
        T, dk, dv = 11, 3, 2
        self.qs = [unit(rnd, dk) for _ in range(T)]
        self.ks = [unit(rnd, dk) for _ in range(T)]
        self.vs = [[rnd.gauss(0, 1) for _ in range(dv)] for _ in range(T)]
        self.al = [[math.exp(m.g_bounded(rnd.gauss(0, 2))) for _ in range(dk)] for _ in range(T)]
        self.be = [rnd.random() for _ in range(T)]

    def test_one_step_by_hand(self):
        # dk=2, dv=1, S0=[[1],[2]], alpha=(0.5,1), k=(1,0), beta=0.5, v=(4)
        S = m.kda_step([[1.0], [2.0]], [1.0, 0.0], [4.0], [0.5, 1.0], 0.5)
        # Diag(alpha)S=[[0.5],[2]]; (I-0.5kk^T) -> [[0.25],[2]]; +0.5*k*v -> [[2.25],[2]]
        self.assertEqual(S, [[2.25], [2.0]])

    def test_chunked_matches_recurrent_for_several_chunk_sizes(self):
        ref, Sref = m.kda_recurrent(self.qs, self.ks, self.vs, self.al, self.be)
        for c in (1, 2, 4, 5, 11):
            out, S = m.kda_chunked(self.qs, self.ks, self.vs, self.al, self.be, c)
            err = max(abs(a - b) for x, y in zip(ref, out) for a, b in zip(x, y))
            self.assertLess(err, 1e-12, c)
            self.assertLess(max(abs(a - b) for r1, r2 in zip(S, Sref) for a, b in zip(r1, r2)), 1e-12)

    def test_beta_one_overwrites_value_along_unit_key(self):
        # With alpha=1 and beta=1, reading S with the same unit key returns v exactly.
        k = [0.6, 0.8]
        S = m.kda_step([[3.0], [-1.0]], k, [5.0], [1.0, 1.0], 1.0)
        self.assertAlmostEqual(m.readout(S, k)[0], 5.0, places=12)

    def test_bounded_decay_range(self):
        for z in (-50.0, -1.0, 0.0, 3.0, 50.0):
            g = m.g_bounded(z)
            self.assertTrue(-5.0 < g < 0.0 or g == -5.0 or g == 0.0)
            self.assertGreater(math.exp(g), math.exp(-5.0) - 1e-15)
        self.assertAlmostEqual(math.exp(-5.0), 6.7e-3, delta=5e-5)

    def test_tile_reciprocal_stays_in_bf16_only_when_bounded(self):
        lim = math.log(m.BF16_MAX)
        zs = [8.0] * 16
        self.assertLess(m.max_log_reciprocal([m.g_bounded(z) for z in zs]), 80.0 + 1e-9)
        self.assertLess(80.0, lim)
        self.assertGreater(m.max_log_reciprocal([m.g_softplus(z) for z in zs]), lim)

    def test_mutation_dropping_decay_breaks_equivalence(self):
        ref, _ = m.kda_recurrent(self.qs, self.ks, self.vs, self.al, self.be)
        ones = [[1.0] * 3 for _ in self.al]
        bad, _ = m.kda_chunked(self.qs, self.ks, self.vs, ones, self.be, 4)
        err = max(abs(a - b) for x, y in zip(ref, bad) for a, b in zip(x, y))
        self.assertGreater(err, 1e-3)


if __name__ == '__main__':
    unittest.main()
