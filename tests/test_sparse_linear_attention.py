"""P16: independent scalar references, causal boundaries and semantic mutants."""
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import types
import unittest

from torch_test_support import torch_status, torch_reason

READY, VERSION, PRESENT = torch_status()
SCRIPT = Path(__file__).resolve().parents[1] / 'code/17_sparse_linear_attention.py'
if READY:
    import torch
    spec = importlib.util.spec_from_file_location('sparse_linear_test', SCRIPT)
    lesson = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lesson)


class SparseLinearCLITests(unittest.TestCase):
    def test_missing_torch_cli_and_import(self):
        blocker = """
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'torch':
        raise ModuleNotFoundError("No module named 'torch'", name='torch')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
"""
        cli = subprocess.run([sys.executable, '-c', blocker +
            "runpy.run_path(sys.argv[1], run_name='__main__')", str(SCRIPT)],
            capture_output=True, text=True)
        self.assertEqual(cli.returncode, 2)
        self.assertIn('python -m pip install -r requirements.txt', cli.stderr)
        self.assertNotIn('Traceback', cli.stderr)
        imported = subprocess.run([sys.executable, '-c', blocker + """
try:
    runpy.run_path(sys.argv[1], run_name='missing_torch_test')
except ModuleNotFoundError as exc:
    assert exc.name == 'torch'
else:
    raise AssertionError('missing dependency was swallowed')
""", str(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(imported.returncode, 0, imported.stderr)


@unittest.skipUnless(READY, torch_reason('P16', VERSION, PRESENT))
class SparseLinearTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(16)
        self.q, self.k = [torch.randn(2, 7, 3, dtype=torch.float64) for _ in range(2)]
        self.v = torch.randn(2, 7, 2, dtype=torch.float64)

    def scalar_reference(self, q, k, v, kind='linear', visible=None):
        # No lesson masks, feature helper, matmul, cumsum or recurrent state.
        result = torch.zeros(q.shape[0], q.shape[1], v.shape[2], dtype=q.dtype)
        for b in range(q.shape[0]):
            for t in range(q.shape[1]):
                positions = list(range(t + 1)) if visible is None else visible[t]
                if kind == 'linear':
                    def phi(x):
                        return x + 1 if x >= 0 else math.exp(x)
                    scores = [sum(phi(q[b, t, d].item()) * phi(k[b, j, d].item())
                                  for d in range(q.shape[2])) for j in positions]
                else:
                    logits = [sum(q[b, t, d].item() * k[b, j, d].item()
                                  for d in range(q.shape[2])) / math.sqrt(q.shape[2]) for j in positions]
                    maximum = max(logits)
                    scores = [math.exp(score - maximum) for score in logits]
                total = sum(scores)
                for dv in range(v.shape[2]):
                    result[b, t, dv] = sum(score * v[b, j, dv].item()
                                          for score, j in zip(scores, positions)) / total
        return result

    def test_window_hand_table_and_chunk_offset(self):
        expected = [[1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0], [0, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 0], [0, 0, 0, 1, 1, 1]]
        self.assertEqual(lesson.causal_window_mask(6, 3).int().tolist(), expected)
        self.assertEqual(lesson.causal_window_mask(2, 3, offset=4).int().tolist(), expected[4:])
        full = lesson.masked_softmax_attention(self.q, self.k, self.v, lesson.causal_window_mask(7, 3))
        chunk = lesson.masked_softmax_attention(self.q[:, 5:], self.k, self.v,
                                                  lesson.causal_window_mask(2, 3, offset=5))
        torch.testing.assert_close(full[:, 5:], chunk, atol=2e-14, rtol=2e-14)
        self.assertEqual(lesson.causal_window_mask(4, 1).int().tolist(), torch.eye(4).int().tolist())
        self.assertTrue(torch.equal(lesson.causal_window_mask(4, 9), torch.ones(4, 4).bool().tril()))

    def test_receptive_field_by_layer(self):
        self.assertEqual([lesson.receptive_positions(6, 3, n)[5] for n in (1, 2, 3)],
                         [(3, 4, 5), (1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)])
        for window in (1, 2, 3, 9):
            for depth in (0, 1, 2, 3):
                reach = lesson.receptive_positions(8, window, depth)
                self.assertEqual(reach, tuple(tuple(range(max(0, i-depth*(window-1)), i+1)) for i in range(8)))

    def test_global_local_never_sees_future(self):
        mask = lesson.global_local_mask(6, 3, (0, 4))
        self.assertEqual(mask[4].int().tolist(), [1, 1, 1, 1, 1, 0])
        self.assertEqual(mask[5].int().tolist(), [1, 0, 0, 1, 1, 1])
        self.assertEqual(mask[0].int().tolist(), [1, 0, 0, 0, 0, 0])
        self.assertFalse(mask.triu(1).any().item())
        self.assertTrue(torch.equal(lesson.global_local_mask(6, 3), lesson.causal_window_mask(6, 3)))

    def test_softmax_independent_window_and_global_oracle(self):
        for mask in (lesson.causal_window_mask(7, 3), lesson.global_local_mask(7, 2, (0, 4))):
            # Construct expected positions from the declared inequalities, not from mask.
            if torch.equal(mask, lesson.causal_window_mask(7, 3)):
                positions = [list(range(max(0, i-2), i+1)) for i in range(7)]
            else:
                positions = [[j for j in range(i+1) if j >= i-1 or j in (0, 4) or i in (0, 4)] for i in range(7)]
            expected = self.scalar_reference(self.q, self.k, self.v, 'softmax', positions)
            torch.testing.assert_close(lesson.masked_softmax_attention(self.q, self.k, self.v, mask), expected,
                                       atol=2e-14, rtol=2e-14)

    def test_linear_both_paths_independent_scalar_oracle(self):
        expected = self.scalar_reference(self.q, self.k, self.v)
        for actual in (lesson.linear_parallel(self.q, self.k, self.v),
                       lesson.linear_recurrent(self.q, self.k, self.v)[0]):
            torch.testing.assert_close(actual, expected, atol=2e-14, rtol=2e-14)

        _, state, normalizer = lesson.linear_recurrent(self.q, self.k, self.v)
        self.assertEqual(sum(x.numel()*x.element_size() for x in (state, normalizer)), 144)

    def test_hand_state_and_normalizer(self):
        q = torch.tensor([[[0., 0.], [0., 1.]]], dtype=torch.float64)
        k = torch.tensor([[[0., 0.], [1., 0.]]], dtype=torch.float64)
        v = torch.tensor([[[2.], [4.]]], dtype=torch.float64)
        out, state, normalizer = lesson.linear_recurrent(q, k, v)
        torch.testing.assert_close(out, torch.tensor([[[2.], [22/7]]], dtype=torch.float64))
        self.assertEqual(state.tolist(), [[[10.], [6.]]])
        self.assertEqual(normalizer.tolist(), [[3., 2.]])

    def test_single_token_and_float32(self):
        for dtype, atol in ((torch.float32, 1e-6), (torch.float64, 1e-14)):
            q, k, v = [x[:, :1].to(dtype) for x in (self.q, self.k, self.v)]
            for actual in (lesson.linear_parallel(q, k, v), lesson.linear_recurrent(q, k, v)[0],
                           lesson.masked_softmax_attention(q, k, v, lesson.causal_window_mask(1, 1))):
                torch.testing.assert_close(actual, v, atol=atol, rtol=atol)
                self.assertEqual(actual.dtype, dtype)
            q, k, v = [x.to(dtype) for x in (self.q, self.k, self.v)]
            torch.testing.assert_close(lesson.linear_parallel(q, k, v), lesson.linear_recurrent(q, k, v)[0],
                                       atol=atol, rtol=atol)

    def test_future_isolation_and_input_immutability(self):
        before = [x.clone() for x in (self.q, self.k, self.v)]
        altered = [x.clone() for x in before]
        for x in altered:
            x[:, 4:] = x[:, 4:] * 7 + 11
        paths = [lesson.linear_parallel, lambda q, k, v: lesson.linear_recurrent(q, k, v)[0],
                 lambda q, k, v: lesson.masked_softmax_attention(q, k, v, lesson.global_local_mask(7, 3, (0, 5)))]
        for path in paths:
            torch.testing.assert_close(path(*before)[:, :4], path(*altered)[:, :4], atol=0, rtol=0)
            torch.testing.assert_close(path(*before), path(*before), atol=0, rtol=0)
        for actual, original in zip((self.q, self.k, self.v), before):
            self.assertTrue(torch.equal(actual, original))

    def test_backward_matches_and_is_finite(self):
        gradients = []
        for path in (lesson.linear_parallel, lambda q, k, v: lesson.linear_recurrent(q, k, v)[0]):
            q, k, v = [x.clone().requires_grad_() for x in (self.q, self.k, self.v)]
            path(q, k, v).square().sum().backward()
            self.assertTrue(all(torch.isfinite(x.grad).all().item() for x in (q, k, v)))
            gradients.append([x.grad for x in (q, k, v)])
        for left, right in zip(*gradients):
            torch.testing.assert_close(left, right, atol=2e-14, rtol=2e-14)

    def test_all_masked_row_and_bad_masks_rejected(self):
        mask = lesson.causal_window_mask(7, 3)
        mask[4] = False
        for bad in (mask, mask.float(), torch.ones(7, 8, dtype=torch.bool), [[True]]):
            with self.assertRaises(ValueError):
                lesson.masked_softmax_attention(self.q, self.k, self.v, bad)

    def test_lengths_windows_offsets_and_global_bounds(self):
        for length in (0, -1, 513, True, 1.5):
            with self.assertRaises(ValueError):
                lesson.causal_window_mask(length, 3)
        for window in (0, -1, True, 2.5):
            with self.assertRaises(ValueError):
                lesson.causal_window_mask(3, window)
        for offset in (-1, True, 0.5, 511):
            with self.assertRaises(ValueError):
                lesson.causal_window_mask(2, 3, offset)
        for positions in ((-1,), (7,), (True,), (1.5,), {0}):
            with self.assertRaises(ValueError):
                lesson.global_local_mask(7, 3, positions)
        for layers in (-1, True, 0.5):
            with self.assertRaises(ValueError):
                lesson.receptive_positions(7, 3, layers)
        self.assertEqual(tuple(lesson.causal_window_mask(1, 1, offset=511).shape), (1, 512))

    def test_tensor_contract(self):
        bad_triplets = [(self.q[:, :0], self.k[:, :0], self.v[:, :0]),
                        (self.q.int(), self.k.int(), self.v.int()),
                        (self.q.half(), self.k.half(), self.v.half()),
                        (self.q.bfloat16(), self.k.bfloat16(), self.v.bfloat16()),
                        (self.q.float(), self.k, self.v),
                        (self.q[0], self.k, self.v), (self.q, self.k[:1], self.v),
                        (self.q, self.k[:, :, :2], self.v), (self.q, self.k, self.v[:, :6]),
                        (self.q[:, :6], self.k, self.v),
                        (self.q, self.k, torch.zeros(2, 7, 0, dtype=torch.float64))]
        for values in bad_triplets:
            for path in (lesson.linear_parallel, lesson.linear_recurrent):
                with self.assertRaises(ValueError):
                    path(*values)
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            for path in (lesson.linear_parallel, lesson.linear_recurrent):
                with self.assertRaisesRegex(ValueError, 'autocast'):
                    path(self.q, self.k, self.v)
        for bad in (float('nan'), float('inf'), -float('inf')):
            q = self.q.clone()
            q[0, 0, 0] = bad
            with self.assertRaises(ValueError):
                lesson.linear_parallel(q, self.k, self.v)

    def test_maximum_length_and_overflow_length(self):
        q = torch.zeros(1, 512, 1, dtype=torch.float64)
        v = torch.ones_like(q)
        for output in (lesson.linear_parallel(q, q, v), lesson.linear_recurrent(q, q, v)[0]):
            self.assertTrue(torch.equal(output, v))
        too_long = torch.zeros(1, 513, 1, dtype=torch.float64)
        with self.assertRaises(ValueError):
            lesson.linear_recurrent(too_long, too_long, too_long)

    def test_underflow_overflow_and_no_cancellation(self):
        for dtype in (torch.float32, torch.float64):
            q = torch.full((1, 2, 1), -40., dtype=dtype)
            v = torch.full_like(q, 3)
            for path in (lesson.linear_parallel, lambda q, k, v: lesson.linear_recurrent(q, k, v)[0]):
                torch.testing.assert_close(path(q, q, v), v)
                with self.assertRaisesRegex(ValueError, 'denominator'):
                    path(q * 100, q * 100, v)
                large = torch.full_like(q, 1e30 if dtype == torch.float32 else 1e200)
                with self.assertRaises(ValueError):
                    path(large, large, v)
            with self.assertRaises(ValueError):
                lesson.masked_softmax_attention(large, large, v, lesson.causal_window_mask(2, 2))

    def test_retrieval_controls_formula_and_fixed_storage(self):
        rows = lesson.retrieval_experiment()
        a = 25 + math.exp(-8)
        b = 10 * math.exp(-4)
        for row in rows:
            n = row['length']
            self.assertAlmostEqual(row['linear'], a/(a+(n-1)*b), places=14)
            self.assertAlmostEqual(row['softmax'], 1/(1+(n-1)*math.exp(-64/math.sqrt(2))), places=14)
            self.assertEqual(row['state_bytes'], 32)
            self.assertEqual(row['window4'], 1. if n == 1 else 0.)
        self.assertGreaterEqual(rows[1]['linear'], .9)
        self.assertLess(rows[2]['linear'], .9)
        for n in (1, 64):
            q = torch.zeros(1, n, 2, dtype=torch.float64)
            v = torch.ones(1, n, 1, dtype=torch.float64)
            _, state, normalizer = lesson.linear_recurrent(q, q, v)
            self.assertEqual(state.untyped_storage().nbytes()+normalizer.untyped_storage().nbytes(), 32)

    def mutant(self, old, new):
        source = SCRIPT.read_text()
        self.assertEqual(source.count(old), 1)
        mod = types.ModuleType('p16_mutant')
        exec(compile(source.replace(old, new), str(SCRIPT), 'exec'), mod.__dict__)
        return mod

    def test_mutation_window_off_by_one_is_detected(self):
        mutant = self.mutant('(key_pos > query_pos - window)', '(key_pos >= query_pos - window)')
        with self.assertRaises(AssertionError):
            self.assertEqual(mutant.causal_window_mask(4, 3)[3].tolist(), [False, True, True, True])

    def test_mutation_future_global_edge_is_detected(self):
        mutant = self.mutant('return (local | global_flags[:, None] | global_flags[None, :]) & causal',
                             'return local | global_flags[:, None] | global_flags[None, :]')
        with self.assertRaises(AssertionError):
            self.assertFalse(mutant.global_local_mask(7, 3, (5,)).triu(1).any().item())

    def test_mutation_noncausal_linear_is_detected(self):
        mutant = self.mutant('torch.ones(q.shape[1], k.shape[1], dtype=torch.bool).tril()',
                             'torch.ones(q.shape[1], k.shape[1], dtype=torch.bool)')
        expected = self.scalar_reference(self.q, self.k, self.v)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(mutant.linear_parallel(self.q, self.k, self.v), expected)

    def test_mutation_shared_missing_normalizer_is_detected(self):
        mutant = self.mutant('output = numerator / denominator', 'output = numerator')
        parallel = mutant.linear_parallel(self.q, self.k, self.v)
        recurrent = mutant.linear_recurrent(self.q, self.k, self.v)[0]
        torch.testing.assert_close(parallel, recurrent, atol=2e-14, rtol=2e-14)
        expected = self.scalar_reference(self.q, self.k, self.v)
        for wrong in (parallel, recurrent):
            with self.assertRaises(AssertionError):
                torch.testing.assert_close(wrong, expected)

    def test_mutation_omitted_current_update_is_detected(self):
        mutant = self.mutant('v[:, t, None, :]', 'v[:, max(t-1, 0), None, :]')
        expected = self.scalar_reference(self.q, self.k, self.v)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(mutant.linear_recurrent(self.q, self.k, self.v)[0], expected)


if __name__ == '__main__':
    unittest.main()
