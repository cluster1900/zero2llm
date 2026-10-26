"""P22 MOPD per-token reward: clip, -KL identity, teacher routing."""
import importlib.util
import math
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'code/23b_mopd_reward.py'
spec = importlib.util.spec_from_file_location('mopd_test', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

T = [0.70, 0.20, 0.05, 0.05]
S = [0.25, 0.25, 0.25, 0.25]


class MOPDTests(unittest.TestCase):
    def test_hand_values_and_clip(self):
        self.assertAlmostEqual(m.opd_reward(0.70, 0.25, 10.0), math.log(2.8), places=12)
        self.assertEqual(m.opd_reward(0.70, 0.25, 1.0), 1.0)
        self.assertEqual(m.opd_reward(0.05, 0.25, 1.0), -1.0)
        self.assertAlmostEqual(m.opd_reward(0.20, 0.25, 1.0), math.log(0.8), places=12)

    def test_unclipped_expectation_is_minus_reverse_kl(self):
        self.assertAlmostEqual(m.expected_reward(T, S), -m.kl(S, T), places=12)
        self.assertLess(m.expected_reward(T, S), 0.0)

    def test_matching_student_gets_zero_reward(self):
        for y in range(4):
            self.assertEqual(m.opd_reward(T[y], T[y], 1.0), 0.0)

    def test_self_anchor_routing(self):
        anchor = 'frozen-start'
        teachers = {'code': 'code-teacher', 'math': 'math-teacher'}
        self.assertEqual(m.pick_teacher('code', teachers, anchor), 'code-teacher')
        self.assertEqual(m.pick_teacher('general', teachers, anchor), anchor)

    def test_rejects_nonpositive_rmax(self):
        with self.assertRaises(ValueError):
            m.opd_reward(0.5, 0.5, 0.0)

    def test_mutation_swapped_ratio_flips_sign(self):
        self.assertAlmostEqual(m.opd_reward(0.25, 0.70, 10.0), -m.opd_reward(0.70, 0.25, 10.0))


if __name__ == '__main__':
    unittest.main()
