"""P09 regression: scalar checks always run; tensors require stable torch 2.14.x."""
import importlib.util
import math
from pathlib import Path
import unittest
from torch_test_support import torch_status, torch_reason

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('training_loop', ROOT / 'code/11_training_loop.py')
loop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loop)
READY, VERSION, PRESENT = torch_status()
if READY:
    import torch


class ScalarTrainingTests(unittest.TestCase):
    def test_hand_calculated_cross_entropy(self):
        result = loop.hand_calculation()
        self.assertAlmostEqual(result['losses'][0], -math.log(.75), places=14)
        self.assertAlmostEqual(result['losses'][1], -math.log(.25), places=14)
        self.assertAlmostEqual(result['mean_loss'], .8369882167858357, places=14)
        self.assertAlmostEqual(loop.cross_entropy([1001., 1000.], 0), math.log1p(math.exp(-1)), places=12)

    def test_gradient_finite_difference_and_update(self):
        result = loop.hand_calculation()
        logits = result['logits'][0]
        for index in range(2):
            plus, minus = logits[:], logits[:]
            plus[index] += 1e-6
            minus[index] -= 1e-6
            numeric = (loop.cross_entropy(plus, 0) - loop.cross_entropy(minus, 0)) / 2e-6
            self.assertAlmostEqual(numeric, result['gradient'][index], places=8)
        self.assertAlmostEqual(result['updated_loss'], .27541448852675354, places=14)
        self.assertLess(result['updated_loss'], result['losses'][0])

    def test_shift_and_wrong_shift_mutation(self):
        ids = [4, 7, 1, 9, 3]
        x, y = loop.shifted_pair(ids, 1, 3)
        self.assertEqual((x, y), ([7, 1, 9], [1, 9, 3]))
        loop.assert_shift(ids, 1, 3, x, y)
        for mutant in [x, [9, 3, 4], list(reversed(y))]:
            with self.assertRaises(ValueError):
                loop.assert_shift(ids, 1, 3, x, mutant)
        with self.assertRaises(ValueError):
            loop.shifted_pair(ids, 3, 2)

    def test_split_and_leakage_mutation(self):
        train, heldout = loop.split_documents(['ABCD', 'EFGH', 'IJKL'])
        self.assertEqual((train, heldout), (['ABCD', 'EFGH'], ['IJKL']))
        with self.assertRaises(ValueError):
            loop.assert_disjoint(train, heldout + train[:1])
        with self.assertRaises(ValueError):
            loop.split_documents(['ABCD', 'EFGH', 'ABCD'])

    def test_copy_cheat_is_not_next_token_skill(self):
        x = [0, 1, 2]
        self.assertLess(loop.copy_cheat_loss(x, x, 4), .00002)
        self.assertGreater(loop.copy_cheat_loss(x, [1, 2, 3], 4), 12)

    def test_dependency_gate(self):
        for v in ['2.14.0', '2.14.1+cpu']:
            self.assertTrue(loop.torch_eligible(True, v))
            self.assertFalse(loop.torch_eligible(False, v))
        for v in ['2.13.1', '2.15.0', '2.14.1rc1', 'not installed']:
            self.assertFalse(loop.torch_eligible(True, v))


@unittest.skipUnless(READY, torch_reason('P09 tensor checks', VERSION, PRESENT))
class TensorTrainingTests(unittest.TestCase):
    def test_reduction_and_autograd_match_handcalc(self):
        ref = loop.hand_calculation()
        scores = torch.tensor(ref['logits'], dtype=torch.float64, requires_grad=True)
        labels = torch.tensor(ref['targets'])
        losses = torch.nn.functional.cross_entropy(scores, labels, reduction='none')
        torch.testing.assert_close(losses, torch.tensor(ref['losses'], dtype=torch.float64))
        mean = torch.nn.functional.cross_entropy(scores, labels)
        self.assertAlmostEqual(float(mean.detach()), ref['mean_loss'])
        mean.backward()
        # Mean across two positions divides each per-position gradient by two.
        torch.testing.assert_close(scores.grad, torch.tensor([[-.125, .125], [-.375, .375]], dtype=torch.float64))

    def test_gradient_accumulation_and_clear(self):
        p = torch.nn.Parameter(torch.tensor([math.log(3), 0.], dtype=torch.float64))
        opt = torch.optim.SGD([p], lr=.1)
        def backward():
            torch.nn.functional.cross_entropy(p[None], torch.tensor([0])).backward()
        backward()
        first = p.grad.clone()
        backward()
        torch.testing.assert_close(p.grad, first * 2)
        opt.zero_grad(set_to_none=True)
        self.assertIsNone(p.grad)
        backward()
        torch.testing.assert_close(p.grad, first)
        opt.step()
        torch.testing.assert_close(p.detach(), torch.tensor(loop.hand_calculation()['updated_logits'], dtype=torch.float64))

    def test_train_step_rejects_zero_grad_omission_numerically(self):
        base = loop.load_baseline()
        torch.manual_seed(17)
        model = base.BabyGPT(5, d_model=8, num_heads=2, num_layers=1, max_seq_len=8)
        import copy
        reference, mutant = copy.deepcopy(model), copy.deepcopy(model)
        x, y = torch.tensor([[0, 1, 2]]), torch.tensor([[1, 2, 3]])
        optimizers = [torch.optim.SGD(m.parameters(), lr=.02) for m in [model, reference, mutant]]
        for _ in range(2):
            loop.train_step(model, optimizers[0], x, y)
            optimizers[1].zero_grad(set_to_none=True)
            reference(x, y)[1].backward()
            optimizers[1].step()
            # Deliberately broken control: second backward accumulates stale gradients.
            mutant(x, y)[1].backward()
            optimizers[2].step()
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected)
        self.assertTrue(any(not torch.allclose(a, b) for a, b in zip(model.parameters(), mutant.parameters())))

    def test_eval_no_grad_and_restore_mode(self):
        base = loop.load_baseline()
        model = base.BabyGPT(base.vocab_size, d_model=8, num_heads=2, num_layers=1, max_seq_len=32)
        snapshot = [p.detach().clone() for p in model.parameters()]
        model.train()
        observations = []
        handle = model.register_forward_pre_hook(
            lambda module, args: observations.append((module.training, torch.is_grad_enabled())))
        try:
            self.assertTrue(math.isfinite(loop.evaluate(model, base.poems, base.encode)))
        finally:
            handle.remove()
        self.assertEqual(observations, [(False, False)] * len(base.poems))
        self.assertTrue(model.training)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        for p, old in zip(model.parameters(), snapshot):
            torch.testing.assert_close(p, old)
        # eval() does NOT disable autograd. no_grad() does NOT set eval().
        model.eval()
        logits, _ = model(torch.tensor([[0, 1]]))
        self.assertTrue(logits.requires_grad)
        model.train()
        with torch.no_grad():
            logits, _ = model(torch.tensor([[0, 1]]))
        self.assertTrue(model.training)
        self.assertFalse(logits.requires_grad)
        # BabyGPT has no dropout; a controlled module illustrates mode-sensitive behavior.
        dropout = torch.nn.Dropout(p=1.)
        self.assertEqual(float(dropout(torch.ones(4)).sum()), 0.)
        dropout.eval()
        self.assertEqual(float(dropout(torch.ones(4)).sum()), 4.)

    def test_strata_forward_eval_no_grad_and_restore_mode(self):
        base = loop.load_baseline()
        model = base.BabyGPT(base.vocab_size, d_model=8, num_heads=2,
                             num_layers=1, max_seq_len=32)
        observations = []
        handle = model.register_forward_pre_hook(
            lambda module, args: observations.append((module.training, torch.is_grad_enabled())))
        try:
            for initial_training in (True, False):
                observations.clear()
                model.train(initial_training)
                groups = loop.evaluate_strata(model, base.poems[1:], base.encode, base.poems[:1])
                self.assertEqual(observations, [(False, False)] * 2)
                self.assertEqual(model.training, initial_training)
                self.assertEqual(sum(group['count'] for group in groups.values()), 46)
                self.assertTrue(all(p.grad is None for p in model.parameters()))
        finally:
            handle.remove()

    def test_evaluate_unequal_documents_token_weighted(self):
        # A controlled forward emits loss 1 for one target and 4 for four targets.
        # Correct token mean=(1*1+4*4)/5=3.4; incorrect document mean=2.5.
        class LengthLoss(torch.nn.Module):
            def forward(self, x, y):
                return None, torch.tensor(float(y.numel()))
        actual = loop.evaluate(LengthLoss(), ['ab', 'abcde'], lambda doc: list(range(len(doc))))
        self.assertAlmostEqual(actual, 17 / 5, places=12)
        self.assertNotAlmostEqual(actual, (1 + 4) / 2, places=6)

    def test_strata_use_labels_not_all_input_characters(self):
        class UniformScores(torch.nn.Module):
            def forward(self, x):
                return torch.zeros((*x.shape, 2)), None
        groups = loop.evaluate_strata(UniformScores(), ['BA'],
                                     lambda doc: [ord(ch) - ord('A') for ch in doc], ['AB'])
        # A occurs in training INPUT, but only B is a training LABEL.
        self.assertEqual(groups['seen'], {'count': 0, 'loss_sum': 0., 'loss': None})
        self.assertEqual(groups['unseen']['count'], 1)
        self.assertAlmostEqual(groups['unseen']['loss'], math.log(2), places=7)

    def test_40_step_snapshots_and_label_strata(self):
        # Independent review reference, six decimals. 1e-6 allows a float32 ULP
        # plus rounding across CPU environments; default AdamW lr=.001 is far away.
        expected = {
            7: (4.280819, 4.219313, 3.175501, 4.494203, 4.843005,
                4.266185, 3.506729, 4.542207, 5.124327),
            11: (4.219005, 4.189975, 3.263426, 3.923873, 4.397414,
                 4.072877, 3.441714, 3.892503, 4.598613),
        }
        base = loop.load_baseline()
        train_labels = set(''.join(doc[1:] for doc in base.poems[:2]))
        heldout_labels = base.poems[2][1:]
        shared = [ch for ch in heldout_labels if ch in train_labels]
        self.assertEqual(sorted(shared), sorted('，，。。'))
        self.assertEqual(len(heldout_labels), 23)
        self.assertEqual(sum(ch not in train_labels for ch in heldout_labels), 19)
        for seed, reference in expected.items():
            with self.subTest(seed=seed):
                result = loop.run_cpu(seed, 40)
                self.assertEqual(result['lr'], .003)
                before, after = result['before'], result['after']
                b, a = before['heldout_strata'], after['heldout_strata']
                values = (result['first_loss'], before['train'], after['train'],
                          before['heldout'], after['heldout'], b['seen']['loss'],
                          a['seen']['loss'], b['unseen']['loss'], a['unseen']['loss'])
                for value, wanted in zip(values, reference):
                    self.assertAlmostEqual(value, wanted, delta=1e-6)
                self.assertLess(a['seen']['loss'], b['seen']['loss'])
                self.assertGreater(a['unseen']['loss'], b['unseen']['loss'])
                for report in (before, after):
                    groups = report['heldout_strata']
                    self.assertEqual(groups['seen']['count'], 4)
                    self.assertEqual(groups['unseen']['count'], 19)
                    recombined = sum(g['count'] * g['loss'] for g in groups.values()) / 23
                    self.assertAlmostEqual(recombined, report['heldout'], delta=1e-6)
                    for group in groups.values():
                        self.assertAlmostEqual(group['count'] * group['loss'], group['loss_sum'], places=10)

    def test_fixed_seed_repeat_and_heldout_report(self):
        first, repeated = loop.run_cpu(7, 4), loop.run_cpu(7, 4)
        self.assertEqual(first, repeated)
        self.assertEqual(first['logits_shape'][:2], (4, 8))
        self.assertEqual(first['heldout_documents'], 1)
        self.assertTrue(all(math.isfinite(first['after'][key]) for key in ('train', 'heldout')))
        other = loop.run_cpu(11, 4)
        self.assertNotEqual(first['first_loss'], other['first_loss'])


if __name__ == '__main__':
    unittest.main()
