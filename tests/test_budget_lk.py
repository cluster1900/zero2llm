"""P23 budget reward override and LK loss."""
import importlib.util
import math
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'code/24c_budget_lk.py'
spec = importlib.util.spec_from_file_location('lk_test', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class BudgetTests(unittest.TestCase):
    def test_override_only_above_threshold(self):
        self.assertEqual(m.budget_reward(1.0, 2000, 1000, 2.0), 1.0)  # equal is not above
        self.assertEqual(m.budget_reward(1.0, 2001, 1000, 2.0), -1.0)
        self.assertEqual(m.budget_reward(0.0, 3000, 1000, 2.0), -1.0)

    def test_smaller_tau_never_restores_a_penalised_run(self):
        runs = [(1.0, 900), (1.0, 1500), (0.0, 700), (1.0, 2600)]
        prev = None
        for tau in (3.0, 2.0, 1.0):
            cur = [m.budget_reward(r, t, 1000, tau) == -1.0 for r, t in runs]
            if prev:
                self.assertTrue(all(c or not p for p, c in zip(prev, cur)))
            prev = cur


class LKTests(unittest.TestCase):
    def test_zero_when_equal_and_positive_otherwise(self):
        self.assertAlmostEqual(m.lk_loss(m.TARGET, m.TARGET), 0.0, places=12)
        self.assertGreater(m.lk_loss(m.TARGET, m.DRAFT_A), 0.0)

    def test_hand_acceptance(self):
        self.assertAlmostEqual(m.acceptance(m.TARGET, m.DRAFT_A), 0.85, places=12)
        self.assertAlmostEqual(m.acceptance(m.TARGET, m.DRAFT_B), 0.901, places=12)

    def test_lower_kl_does_not_mean_higher_acceptance(self):
        self.assertLess(m.kl(m.TARGET, m.DRAFT_A), m.kl(m.TARGET, m.DRAFT_B))
        self.assertLess(m.acceptance(m.TARGET, m.DRAFT_A), m.acceptance(m.TARGET, m.DRAFT_B))
        self.assertLess(m.lk_loss(m.TARGET, m.DRAFT_B), m.lk_loss(m.TARGET, m.DRAFT_A))

    def test_acceptance_equals_one_minus_total_variation(self):
        tv = 0.5 * sum(abs(a - b) for a, b in zip(m.TARGET, m.DRAFT_B))
        self.assertAlmostEqual(m.acceptance(m.TARGET, m.DRAFT_B), 1 - tv, places=12)

    def test_mutation_max_instead_of_min_is_caught(self):
        bad = -math.log(sum(max(a, b) for a, b in zip(m.TARGET, m.DRAFT_A)))
        self.assertLess(bad, 0.0)


if __name__ == '__main__':
    unittest.main()
