"""P24 FP4 E2M1 rounding, MXFP4/NVFP4 block scales vs per-tensor."""
import importlib.util
import math
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'code/25c_fp4_block.py'
spec = importlib.util.spec_from_file_location('fp4_test', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class FP4Tests(unittest.TestCase):
    def test_grid_is_exact_and_saturates(self):
        for g in m.E2M1:
            self.assertEqual(m.round_e2m1(g), g)
            self.assertEqual(m.round_e2m1(-g), -g)
        self.assertEqual(m.round_e2m1(100.0), 6.0)
        self.assertEqual(m.round_e2m1(2.4), 2.0)
        self.assertEqual(m.round_e2m1(2.6), 3.0)

    def test_e4m3_rounding(self):
        self.assertEqual(m.round_e4m3(1.0), 1.0)
        self.assertEqual(m.round_e4m3(1.06), 1.0)
        self.assertEqual(m.round_e4m3(1.07), 1.125)
        self.assertEqual(m.round_e4m3(1000.0), 448.0)

    def test_mx_scale_is_power_of_two(self):
        blk = [0.3, -5.0, 1.2] + [0.0] * 29
        q = m.quant_mxfp4(blk)
        X = 2.0 ** (math.floor(math.log2(5.0)) - 2)  # = 1
        self.assertEqual(X, 1.0)
        self.assertEqual(q[:3], [0.5, -4.0, 1.0])  # 5 is a tie between 4 and 6

    def test_block_scales_beat_per_tensor_on_outlier_data(self):
        xs = m.make_data()
        e_t = m.rel_mse(xs, m.quant_per_tensor(xs))
        e_mx = m.rel_mse(xs, m.quant_mxfp4(xs))
        e_nv = m.rel_mse(xs, m.quant_nvfp4(xs))
        self.assertLess(e_mx, e_t)
        self.assertLess(e_nv, e_mx)

    def test_mutation_one_block_equals_per_tensor_amax(self):
        xs = m.make_data(n=64, seed=3)
        whole = m.quant_nvfp4(xs, block=len(xs))
        # one block: s_block*s_global is amax/6 up to E4M3 rounding of 448 (exact)
        self.assertAlmostEqual(m.rel_mse(xs, whole), m.rel_mse(xs, m.quant_per_tensor(xs)), places=12)


if __name__ == '__main__':
    unittest.main()
