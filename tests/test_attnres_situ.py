"""P07/P13 AttnRes weights and SiTU-GLU bound."""
import importlib.util
import math
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'code/14b_attnres_situ.py'
spec = importlib.util.spec_from_file_location('attnres_test', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

SRC = [[1.0, 0.0, 2.0], [0.5, -1.0, 0.0], [3.0, 3.0, -3.0], [0.1, 0.2, 0.3]]


class AttnResTests(unittest.TestCase):
    def test_weights_sum_to_one(self):
        for w in ([0, 0, 0], [1, -2, 0.5], [10, 10, -10]):
            a = m.attnres_weights(w, SRC)
            self.assertAlmostEqual(sum(a), 1.0, places=12)
            self.assertTrue(all(x > 0 for x in a))

    def test_uniform_weights_equal_plain_residual_divided_by_count(self):
        h, a = m.attnres_mix([0.0, 0.0, 0.0], SRC)
        self.assertTrue(all(abs(x - 0.25) < 1e-12 for x in a))
        plain = m.plain_residual(SRC)
        for x, y in zip(h, plain):
            self.assertAlmostEqual(x * len(SRC), y, places=12)

    def test_rmsnorm_keeps_large_source_from_dominating(self):
        big = [[100.0 * x for x in SRC[2]]] + SRC[:2]
        small = [SRC[2]] + SRC[:2]
        w = [0.3, 0.1, -0.2]
        for x, y in zip(m.attnres_weights(w, big), m.attnres_weights(w, small)):
            self.assertAlmostEqual(x, y, places=6)


class SiTUTests(unittest.TestCase):
    def test_bound_is_beta1_times_beta2(self):
        for g in (-100.0, -3.0, 0.5, 10.0, 1e4):
            for u in (-1e4, -2.0, 0.0, 7.0, 1e4):
                self.assertLessEqual(abs(m.situ_glu(g, u)), 100.0)
        self.assertGreater(abs(m.swiglu(1e3, 1e3)), 1e5)

    def test_close_to_swiglu_near_origin(self):
        for x in (-0.1, 0.05, 0.1):
            self.assertAlmostEqual(m.situ_glu(x, x), m.swiglu(x, x), delta=1e-4)

    def test_mutation_without_up_cap_is_unbounded(self):
        uncapped = 4.0 * math.tanh(1e4 / 4.0) * m.sigmoid(1e4) * 1e4
        self.assertGreater(uncapped, 100.0)


if __name__ == '__main__':
    unittest.main()
