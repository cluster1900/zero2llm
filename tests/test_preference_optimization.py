"""P22 independent scalar/gradient oracles and genuine behavioral mutants.

No torch: arithmetic, alignment, mutation and CLI tests still run; torch tests
are honestly SKIPPED. No model or framework downloads occur.
"""
import dataclasses
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import types
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "code/23_preference_optimization.py"
spec = importlib.util.spec_from_file_location("preference_lesson", SCRIPT)
lesson = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = lesson
spec.loader.exec_module(lesson)
try:
    import torch
except ImportError:
    torch = None


def mutant(old, new):
    source = SCRIPT.read_text(encoding="utf-8")
    if source.count(old) != 1:
        raise AssertionError(f"mutation must match once: {old!r}")
    module = types.ModuleType("preference_mutant")
    module.__file__ = str(SCRIPT)
    sys.modules[module.__name__] = module
    exec(compile(source.replace(old, new), str(SCRIPT), "exec"), module.__dict__)
    return module


def logsoftmax(row):
    # Independent scalar implementation; no lesson helper or torch loss.
    m = max(row)
    z = m + math.log(sum(math.exp(v - m) for v in row))
    return [v - z for v in row]


def scalar_kl(p_logits, q_logits):
    lp, lq = logsoftmax(p_logits), logsoftmax(q_logits)
    return sum(math.exp(p) * (p - q) for p, q in zip(lp, lq))


def scalar_grpo(logits, targets, old, reference, mask, advantages, beta=.13, clip=.2):
    total = 0.0
    for i, row in enumerate(logits):
        sequence = []
        for t, token_logits in enumerate(row):
            if not mask[i][t]:
                continue
            probability_ratio = math.exp(logsoftmax(token_logits)[targets[i][t]] - old[i][t])
            a = advantages[i]
            # Independent piecewise form, not torch.minimum/clamp expression.
            if a >= 0:
                gain = a * min(probability_ratio, 1 + clip)
            else:
                gain = a * max(probability_ratio, 1 - clip)
            sequence.append(-gain + beta * scalar_kl(token_logits, reference[i][t]))
        total += sum(sequence) / len(sequence)
    return total / len(logits)


class ArithmeticTests(unittest.TestCase):
    def assert_dpo_oracle(self, module):
        for beta in (.1, .5, 2):
            for pc, pr, qc, qr in ((.6, .2, .4, .4), (.2, .5, .1, .5), (.1, .7, .6, .2)):
                expected = math.log(1 + ((pr / pc) / (qr / qc)) ** beta)
                actual = module.dpo_pair_loss(*(math.log(p) for p in (pc, pr, qc, qr)), beta=beta)
                self.assertAlmostEqual(actual, expected, places=13)

    def test_hand_dpo_batch_and_stability(self):
        self.assert_dpo_oracle(lesson)
        report = lesson.hand_demo()
        self.assertAlmostEqual(report["dpo_batch_mean"], .49527319557394833)
        self.assertAlmostEqual(lesson.dpo_pair_loss(-1, -10001, -1, -1, 1), 0)
        self.assertAlmostEqual(lesson.dpo_pair_loss(-10001, -1, -1, -1, 1), 10000)
        self.assertAlmostEqual(lesson.dpo_pair_loss(-2, -5, -2, -5), math.log(2))

    def assert_groups(self, module):
        # Interleaved groups with different scales: batch/global norm is wrong.
        actual = module.group_advantages([0, 10, 1, 12, 1, 14, 2, 12], [7, 9] * 4)
        a = 1 / (math.sqrt(.5) + 1e-8)
        b = 2 / (math.sqrt(2) + 1e-8)
        expected = [-a, -b, 0, 0, 0, b, a, 0]
        for observed, wanted in zip(actual, expected):
            self.assertAlmostEqual(observed, wanted, places=12)
        self.assertEqual(module.group_advantages([7] * 4, [0] * 4), [0] * 4)

    def test_group_population_std_interleaving_and_equal_rewards(self):
        self.assert_groups(lesson)
        self.assertEqual(lesson.group_advantages([1, 1], [0, 0]), [0, 0])
        a = lesson.group_advantages([0, 1], [0, 0], epsilon=1)
        self.assertEqual(a, [-1/3, 1/3])
        self.assertNotEqual(a, lesson.group_advantages([0, 1], [0, 0]))

    def test_kl_is_distribution_weighted_directional_and_full(self):
        self.assertAlmostEqual(lesson.exact_kl([.6, .3, .1], [.2, .5, .3]), .39605845720425753)
        self.assertNotAlmostEqual(lesson.exact_kl([.2, .5, .3], [.6, .3, .1]), .39605845720425753)
        self.assertEqual(lesson.exact_kl([.4, .6], [.4, .6]), 0)
        self.assertGreaterEqual(lesson.exact_kl([.8, .2], [.1, .9]), 0)

    def test_alignment_joins_the_right_prompt_sample_action_and_reward(self):
        rows = [lesson.Rollout("a", 0, 0, "2"), lesson.Rollout("b", 1, 2, "3")]
        rewards = lesson.score_rollouts(rows)
        self.assertEqual(lesson.aligned_rewards(rows, rewards), ([1, 1], [0, 1]))
        for broken in (rewards[::-1], rewards[:-1],
                       [dataclasses.replace(rewards[0], prompt_id=1), rewards[1]],
                       [dataclasses.replace(rewards[0], value=float("nan")), rewards[1]]):
            with self.assertRaises(ValueError):
                lesson.aligned_rewards(rows, broken)
        for broken in ([rows[0], rows[0]], [dataclasses.replace(rows[0], completion="3"), rows[1]],
                       [dataclasses.replace(rows[0], action_id=9), rows[1]]):
            with self.assertRaises(ValueError):
                lesson.aligned_rewards(broken, rewards)
        # Same completion needs different reward under another prompt.
        other = [lesson.Rollout("a", 1, 0, "2")]
        self.assertEqual(lesson.score_rollouts(other)[0].value, 0)

    def test_invalid_inputs_are_rejected(self):
        for bad in (0, -1, True, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                lesson.dpo_pair_loss(-1, -2, -1, -2, beta=bad)
            with self.assertRaises(ValueError):
                lesson.group_advantages([0, 1], [0, 0], epsilon=bad)
        for r, ids in (([], []), ([1], [0]), ([1, 2], [0]), ([1, float("nan")], [0, 0])):
            with self.assertRaises(ValueError):
                lesson.group_advantages(r, ids)
        for p, q in (([], []), ([.2, .2], [.5, .5]), ([0, 1], [.4, .6]), ([1], [.5, .5])):
            with self.assertRaises(ValueError):
                lesson.exact_kl(p, q)

    def test_behavioral_arithmetic_mutants_are_detected(self):
        for old, new in (("z = beta * margin", "z = -beta * margin"),
                         ("z = beta * margin", "z = margin / beta"),
                         ("margin = (chosen - rejected) - (ref_chosen - ref_rejected)",
                          "margin = chosen - rejected")):
            with self.subTest(mutation=new), self.assertRaises(AssertionError):
                self.assert_dpo_oracle(mutant(old, new))
        for old, new in (("groups.setdefault(prompt_id, []).append(index)", "groups.setdefault(0, []).append(index)"),
                         ("** 2 for i in indices) / len(indices)", "** 2 for i in indices) / (len(indices) - 1)"),
                         ("math.sqrt(variance) + epsilon", "math.sqrt(variance + epsilon)")):
            with self.subTest(mutation=new), self.assertRaises(AssertionError):
                self.assert_groups(mutant(old, new))
        broken = mutant("if row.sample_id != reward.sample_id or row.prompt_id != reward.prompt_id:", "if False:")
        # The original contract must reject, mutant silently accepts wrong scores.
        def alignment_contract(module):
            rows = [module.Rollout("a", 0, 0, "2"), module.Rollout("b", 1, 2, "3")]
            scores = module.score_rollouts(rows)[::-1]
            with self.assertRaises(ValueError):
                module.aligned_rewards(rows, scores)
        alignment_contract(lesson)
        with self.assertRaises(AssertionError):
            alignment_contract(broken)

    def test_standard_library_cli_and_no_torch_skip(self):
        result = subprocess.run([sys.executable, "-S", str(SCRIPT), "--torch"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        default = subprocess.run([sys.executable, "-S", str(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(default.returncode, 0)
        self.assertIn('"dpo_batch_mean"', result.stdout)
        self.assertIn("P22 torch SKIP", result.stdout)
        self.assertNotIn('"status": "EXECUTED', result.stdout)


@unittest.skipIf(torch is None, "optional torch unavailable: no CPU tensor/gradient checks ran")
class TensorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def token_fixture(self):
        values = [[[.1 + .17*b - .11*t + .09*j*j + .03*t*j for j in range(4)]
                   for t in range(5)] for b in range(2)]
        logits = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        ids = torch.tensor([[1, 2, 3, 1, 0], [1, 3, 2, 1, 2]])
        mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
        return logits, ids, mask, [2, 1]

    def assert_completion_oracle(self, module):
        logits, ids, attention, starts = self.token_fixture()
        actual, mask = module.completion_log_probs(logits, ids, attention, starts)
        expected = []
        for b in range(2):
            expected.append(sum(logsoftmax(logits[b, t-1].tolist())[int(ids[b, t])]
                                for t in range(starts[b], int(attention[b].sum()))))
        torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float64), atol=1e-12, rtol=0)
        self.assertEqual(mask.tolist(), [[False, True, True, False], [True] * 4])

    def test_completion_sum_shift_prompt_padding_eos_and_gradients(self):
        self.assert_completion_oracle(lesson)
        logits, ids, attention, starts = self.token_fixture()
        sums, mask = lesson.completion_log_probs(logits, ids, attention, starts)
        sums.sum().backward()
        self.assertEqual(logits.grad[0, 0].abs().sum().item(), 0)  # prompt target
        self.assertEqual(logits.grad[0, 3:].abs().sum().item(), 0)  # PAD / no next target
        self.assertGreater(logits.grad[0, 2].abs().sum().item(), 0)  # real EOS target
        changed = logits.detach().clone()
        changed[0, 0] = torch.tensor([50., -20., 2., -1.])
        changed[0, 3:] = 100
        newer, _ = lesson.completion_log_probs(changed, ids, attention, starts)
        torch.testing.assert_close(newer, sums.detach())
        # Direct scalar derivative: 1(target=j)-p_j for each valid position.
        for b in range(2):
            for t in range(4):
                for j in range(4):
                    expected = ((float(int(ids[b, t+1]) == j) - math.exp(logsoftmax(logits[b, t].tolist())[j]))
                                if mask[b, t] else 0)
                    self.assertAlmostEqual(logits.grad[b, t, j].item(), expected, places=12)

    def test_token_validation(self):
        logits, ids, mask, starts = self.token_fixture()
        for bad_starts in ([0, 1], [4, 1], [2], [True, 1]):
            with self.assertRaises(ValueError):
                lesson.completion_log_probs(logits, ids, mask, bad_starts)
        for bad_mask in (torch.tensor([[1, 1, 0, 1, 0], [1]*5]), mask.float() + .2):
            with self.assertRaises(ValueError):
                lesson.completion_log_probs(logits, ids, bad_mask, starts)
        with self.assertRaises(ValueError):
            lesson.completion_log_probs(logits, ids.float(), mask, starts)

    def assert_dpo_gradient(self, module):
        c = torch.tensor([-.4, -2.2], dtype=torch.float64, requires_grad=True)
        r = torch.tensor([-1.2, -.7], dtype=torch.float64, requires_grad=True)
        qc = torch.tensor([-.7, -3.], dtype=torch.float64, requires_grad=True)
        qr = torch.tensor([-.9, -.2], dtype=torch.float64, requires_grad=True)
        beta = .37
        loss = module.dpo_loss(c, r, qc, qr, beta=beta)
        loss.backward()
        expected_loss = 0
        for i in range(2):
            z = beta * ((c[i].item()-r[i].item()) - (qc[i].item()-qr[i].item()))
            expected_loss += math.log(1+math.exp(-z)) / 2
            derivative = -beta / (2 * (1 + math.exp(z)))
            self.assertAlmostEqual(c.grad[i].item(), derivative, places=12)
            self.assertAlmostEqual(r.grad[i].item(), -derivative, places=12)
        self.assertAlmostEqual(loss.item(), expected_loss, places=12)
        self.assertIsNone(qc.grad)
        self.assertIsNone(qr.grad)

    def test_dpo_closed_form_gradient_and_reference_is_frozen(self):
        self.assert_dpo_gradient(lesson)

    def assert_kl_gradient(self, module):
        p = torch.tensor([[.4, -1., .8], [-.3, 2., .7]], dtype=torch.float64, requires_grad=True)
        q = torch.tensor([[-.5, .3, .2], [1., -.1, .4]], dtype=torch.float64, requires_grad=True)
        values = module.categorical_kl(p, q)
        values.sum().backward()
        for i in range(2):
            expected = scalar_kl(p[i].tolist(), q[i].tolist())
            self.assertAlmostEqual(values[i].item(), expected, places=12)
            lp, lq = logsoftmax(p[i].tolist()), logsoftmax(q[i].tolist())
            for j in range(3):
                gradient = math.exp(lp[j]) * (lp[j] - lq[j] - expected)
                self.assertAlmostEqual(p.grad[i, j].item(), gradient, places=12)
        self.assertIsNone(q.grad)

    def test_exact_kl_values_and_closed_form_gradient(self):
        self.assert_kl_gradient(lesson)

    def grpo_fixture(self):
        logits = torch.tensor([[[.3, -.2, .7], [-.2, .1, -.8], [.8, -.3, .4]],
                               [[-.4, .6, .2], [.9, -.1, .4], [.2, -.7, .3]]], dtype=torch.float64, requires_grad=True)
        reference = torch.tensor([[[.2, -.1, .5]] * 3, [[-.2, .1, .2]] * 3], dtype=torch.float64, requires_grad=True)
        targets = torch.tensor([[0, 2, 1], [2, 0, 1]])
        lp = logits.detach().log_softmax(-1).gather(-1, targets[:, :, None]).squeeze(-1)
        # Ratios exercise clipping AND non-clipping, both advantage signs.
        ratios = torch.tensor([[1.6, .7, 1.1], [.5, 1.4, 1.0]], dtype=torch.float64)
        old = (lp-ratios.log()).requires_grad_()
        mask = torch.tensor([[True, True, False], [True, True, True]])
        advantages = torch.tensor([.8, -1.2], dtype=torch.float64, requires_grad=True)
        return logits, targets, old, reference, mask, advantages

    def assert_grpo_oracle(self, module, gradients=False):
        args = self.grpo_fixture()
        actual = module.grpo_loss(*args, beta=.13)
        data = [x.detach().tolist() for x in args]
        expected = scalar_grpo(*data)
        self.assertAlmostEqual(actual.item(), expected, places=12)
        if gradients:
            actual.backward()
            h = 1e-5
            for i in range(2):
                for t in range(3):
                    for j in range(3):
                        data[0][i][t][j] += h
                        plus = scalar_grpo(*data)
                        data[0][i][t][j] -= 2*h
                        minus = scalar_grpo(*data)
                        data[0][i][t][j] += h
                        self.assertAlmostEqual(args[0].grad[i, t, j].item(), (plus-minus)/(2*h), places=8)
            for frozen in (args[2], args[3], args[5]):
                self.assertIsNone(frozen.grad)

    def test_grpo_clipping_mask_equal_sequence_weight_and_finite_difference_gradient(self):
        self.assert_grpo_oracle(lesson, gradients=True)

    def test_all_equal_rewards_policy_gradient_zero_but_kl_can_act(self):
        args = list(self.grpo_fixture())
        args[5] = torch.tensor(lesson.group_advantages([9, 9], [0, 0]), dtype=torch.float64)
        zero_loss = lesson.grpo_loss(*args, beta=0)
        zero_loss.backward()
        self.assertEqual(zero_loss.item(), 0)
        self.assertEqual(args[0].grad.abs().sum().item(), 0)
        args[0].grad.zero_()
        regularized = lesson.grpo_loss(*args, beta=.1)
        regularized.backward()
        self.assertGreater(regularized.item(), 0)
        self.assertGreater(args[0].grad.abs().sum().item(), 0)

    def test_sampling_reproducible_with_replacement_not_greedy(self):
        logits = torch.zeros(2, 3, dtype=torch.float64)
        rows = lesson.sample_rollouts(logits, group_size=2000, seed=22)
        self.assertEqual(rows, lesson.sample_rollouts(logits, group_size=2000, seed=22))
        self.assertNotEqual(rows, lesson.sample_rollouts(logits, group_size=2000, seed=23))
        for prompt in (0, 1):
            group = [r for r in rows if r.prompt_id == prompt]
            self.assertEqual(len(group), 2000)
            for action in (0, 1, 2):
                self.assertTrue(.29 < sum(r.action_id == action for r in group)/2000 < .38)
        values, ids = lesson.aligned_rewards(rows, lesson.score_rollouts(rows))
        self.assertEqual(len(values), len(ids))
        self.assertEqual(len({r.sample_id for r in rows}), 4000)

    def test_real_training_changes_logits_and_exact_distribution_kl(self):
        report = lesson.torch_demo()
        self.assertTrue(report["reference_unchanged"])
        self.assertEqual(report["exact_kl_to_sft_before"], [0, 0])
        self.assertLess(report["dpo_loss_before_after"][1], report["dpo_loss_before_after"][0])
        self.assertEqual(report["parameters"], 6)
        baseline = report["expected_reward_before_after_dpo_grpo"][0]
        for stage in ("dpo", "grpo"):
            self.assertNotEqual(report[f"{stage}_logits"], report["sft_logits"])
            self.assertGreater(report["expected_reward_before_after_dpo_grpo"][1 if stage == "dpo" else 2], baseline)
            for p in range(2):
                independent = scalar_kl(report[f"{stage}_logits"][p], report["sft_logits"][p])
                self.assertAlmostEqual(report[f"exact_kl_to_sft_after_{stage}"][p], independent, places=12)
                self.assertGreater(independent, 0)
        # Independently reproduce every SFT/DPO SGD scalar update, not just direction.
        weights = [[0., 0., 0.], [0., 0., 0.]]
        for _ in range(4):
            for p, chosen in enumerate((0, 2)):
                probabilities = [math.exp(v) for v in logsoftmax(weights[p])]
                weights[p] = [v - .4*(probabilities[j]-float(j == chosen))/2 for j, v in enumerate(weights[p])]
        reference = [row[:] for row in weights]
        for _ in range(6):
            for p, chosen in enumerate((0, 2)):
                z = .5*((weights[p][chosen]-weights[p][1])-(reference[p][chosen]-reference[p][1]))
                derivative = -.5/(2*(1+math.exp(z)))
                weights[p][chosen] -= .4*derivative
                weights[p][1] += .4*derivative
        for p in range(2):
            for j in range(3):
                self.assertAlmostEqual(report["sft_logits"][p][j], reference[p][j], places=12)
                self.assertAlmostEqual(report["dpo_logits"][p][j], weights[p][j], places=12)

    def test_full_grpo_updates_against_independent_scalar_finite_difference(self):
        report = lesson.torch_demo()
        weights = [row[:] for row in report["sft_logits"]]
        reference = [row[:] for row in weights]
        actions = report["sampled_action_ids"]
        rewards = [float(action == (0 if i < 8 else 2)) for i, action in enumerate(actions)]
        advantages = []
        for start in (0, 8):
            group = rewards[start:start+8]
            mean = sum(group)/8
            sigma = math.sqrt(sum((v-mean)**2 for v in group)/8)
            advantages += [(v-mean)/(sigma+1e-8) for v in group]
        old = [[logsoftmax(reference[i//8])[action]] for i, action in enumerate(actions)]
        targets = [[a] for a in actions]
        ref_rows = [[reference[i//8][:]] for i in range(16)]
        def objective():
            values = [[weights[i//8][:]] for i in range(16)]
            return scalar_grpo(values, targets, old, ref_rows, [[True]]*16, advantages, beta=.04)
        for _ in range(3):
            gradient = [[0.]*3 for _ in range(2)]
            for p in range(2):
                for j in range(3):
                    h = 1e-5
                    weights[p][j] += h
                    plus = objective()
                    weights[p][j] -= 2*h
                    minus = objective()
                    weights[p][j] += h
                    gradient[p][j] = (plus-minus)/(2*h)
            weights = [[value-.2*gradient[p][j] for j, value in enumerate(row)]
                       for p, row in enumerate(weights)]
        for p in range(2):
            for j in range(3):
                self.assertAlmostEqual(report["grpo_logits"][p][j], weights[p][j], places=9)

    def test_extreme_finite_old_logps_rejected_instead_of_nan(self):
        logits = torch.zeros(1, 1, 2, dtype=torch.float64, requires_grad=True)
        for advantage in (0., 1., -1.):
            with self.subTest(advantage=advantage), self.assertRaisesRegex(ValueError, "ratio overflow"):
                lesson.grpo_loss(logits, torch.tensor([[0]]),
                                 torch.tensor([[-1000.]], dtype=torch.float64),
                                 logits.detach().clone(), torch.tensor([[True]]),
                                 torch.tensor([advantage], dtype=torch.float64), beta=0)
        # Finite but huge advantage may overflow the objective even with finite ratio.
        with self.assertRaisesRegex(ValueError, "finite numerical range"):
            lesson.grpo_loss(logits, torch.tensor([[0]]),
                             torch.tensor([[-3.]], dtype=torch.float64),
                             logits.detach().clone(), torch.tensor([[True]]),
                             torch.tensor([-1e308], dtype=torch.float64), beta=0)

    def test_grpo_rejects_empty_mask_or_bad_shapes(self):
        args = list(self.grpo_fixture())
        for index, broken in ((4, torch.zeros(2, 3, dtype=torch.bool)),
                              (4, torch.ones(2, 3)), (5, torch.ones(2, 1)), (1, args[1].float())):
            changed = args[:]
            changed[index] = broken
            with self.assertRaises(ValueError):
                lesson.grpo_loss(*changed)
        for beta in (-1, float("nan")):
            with self.assertRaises(ValueError):
                lesson.grpo_loss(*args, beta=beta)

    def test_behavioral_tensor_mutants_are_detected(self):
        mutations = [
            ("positions >= starts", "positions >= 1", self.assert_completion_oracle),
            ("logits[:, :-1].log_softmax", "logits[:, 1:].log_softmax", self.assert_completion_oracle),
            ("(selected * mask).sum(dim=-1), mask", "(selected * mask).sum(dim=-1) / mask.sum(dim=-1), mask", self.assert_completion_oracle),
            ("softplus(-beta * margin)", "softplus(beta * margin)", self.assert_dpo_gradient),
            ("softplus(-beta * margin)", "softplus(-margin / beta)", self.assert_dpo_gradient),
            ("(ref_chosen.detach() - ref_rejected.detach())", "(ref_chosen * 0 - ref_rejected * 0)", self.assert_dpo_gradient),
            ("logp.exp() * (logp - logq)", "logq.exp() * (logq - logp)", self.assert_kl_gradient),
            ("logp.exp() * (logp - logq)", "(logp - logq)", self.assert_kl_gradient),
            ("selected - old_logps.detach()", "selected - selected.detach()", self.assert_grpo_oracle),
            ("torch.minimum(unclipped, clipped)", "torch.maximum(unclipped, clipped)", self.assert_grpo_oracle),
            ("torch.minimum(unclipped, clipped)", "unclipped", self.assert_grpo_oracle),
            ("((token_loss * mask).sum(dim=-1) / mask.sum(dim=-1)).mean()", "(token_loss * mask).sum() / mask.sum()", self.assert_grpo_oracle),
            ("token_loss * mask", "token_loss", self.assert_grpo_oracle),
        ]
        for old, new, oracle in mutations:
            with self.subTest(mutation=new), self.assertRaises(AssertionError):
                oracle(mutant(old, new))


if __name__ == "__main__":
    unittest.main()
