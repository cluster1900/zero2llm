"""P18 Quantile Balancing toy and LatentMoE ledger."""
import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'code/19b_latent_moe_qb.py'
spec = importlib.util.spec_from_file_location('qb_test', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class QBTests(unittest.TestCase):
    def test_both_toys_start_at_figure5_loads(self):
        for s in (m.SCORES_BALANCES, m.SCORES_STALLS):
            self.assertEqual(m.loads(m.route(s, [0.0] * 4, 1), 4), [4, 3, 1, 0])

    def test_one_step_balances_first_toy(self):
        b = m.quantile_balance(m.SCORES_BALANCES, [0.0] * 4, 1)
        self.assertEqual(m.loads(m.route(m.SCORES_BALANCES, b, 1), 4), [2, 2, 2, 2])

    def test_second_toy_does_not_balance(self):
        b = [0.0] * 4
        for _ in range(5):
            b = m.quantile_balance(m.SCORES_STALLS, b, 1)
        self.assertEqual(m.loads(m.route(m.SCORES_STALLS, b, 1), 4), [3, 3, 1, 1])

    def test_bias_is_mean_zero(self):
        b = m.quantile_balance(m.SCORES_BALANCES, [0.0] * 4, 1)
        self.assertAlmostEqual(sum(b), 0.0, places=12)

    def test_common_offset_does_not_change_routing(self):
        b = m.quantile_balance(m.SCORES_BALANCES, [0.0] * 4, 1)
        c = [x + 0.37 for x in b]
        self.assertEqual(m.route(m.SCORES_BALANCES, b, 1), m.route(m.SCORES_BALANCES, c, 1))

    def test_mixture_weights_ignore_bias(self):
        row = m.SCORES_BALANCES[1]
        w = m.routed_weights(row, [0, 2])
        self.assertAlmostEqual(w[0], 0.5)
        self.assertAlmostEqual(sum(w.values()), 1.0)

    def test_ledger_arithmetic(self):
        led = m.latent_moe_ledger()
        self.assertEqual(led['sparsity'], 56)
        self.assertEqual(led['latent_ratio'], 0.5)


if __name__ == '__main__':
    unittest.main()
