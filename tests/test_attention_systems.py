"""P15: independent Decimal oracle, paged state model, and assertion-only mutations."""
from decimal import Decimal, localcontext
import importlib.util
import math
from pathlib import Path
import random
import subprocess
import sys
import types
import unittest

from torch_test_support import torch_reason, torch_status

SCRIPT = Path(__file__).resolve().parents[1] / 'code/16_attention_systems.py'
spec = importlib.util.spec_from_file_location('attention_systems_test', SCRIPT)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
READY, VERSION, PRESENT = torch_status()


def decimal_oracle(q, k, v, allowed=None):
    """No production helpers, Python math.exp, dense_attention, or block recurrence."""
    with localcontext() as context:
        context.prec = 65
        convert = lambda x: Decimal(str(x))
        output = []
        for i in range(len(q)):
            visible = [j for j in range(len(k)) if allowed is None or allowed[i][j]]
            if not visible:
                output.append([0.0 for _ in v[0]])
                continue
            scores = [sum(convert(q[i][c]) * convert(k[j][c]) for c in range(len(q[i])))
                      / Decimal(len(q[i])).sqrt() for j in visible]
            peak = max(scores)
            weights = [(score - peak).exp() for score in scores]
            denom = sum(weights)
            output.append([float(sum(w * convert(v[j][c]) for w, j in zip(weights, visible)) / denom)
                           for c in range(len(v[0]))])
        return output


def assert_matrix(case, actual, expected, tolerance=2e-12):
    case.assertEqual(len(actual), len(expected))
    for actual_row, expected_row in zip(actual, expected):
        case.assertEqual(len(actual_row), len(expected_row))
        for got, want in zip(actual_row, expected_row):
            case.assertTrue(math.isfinite(got))
            case.assertAlmostEqual(got, want, delta=tolerance)


class OnlineAttentionTests(unittest.TestCase):
    def test_hand_computed_two_uneven_blocks(self):
        q, k, v = [[1]], [[0], [math.log(2)], [math.log(4)]], [[2], [4], [8]]
        for block in (1, 2, 3, 8):
            assert_matrix(self, a.blocked_attention(q, k, v, key_block_size=block), [[6]])
        # Before second block: m=log(2), ell=1.5, acc=5. After: ell=1.75, acc=10.5.
        self.assertAlmostEqual(math.exp(math.log(2) - math.log(4)), 0.5)
        self.assertEqual(0.5 * 1.5 + 1, 1.75)
        self.assertEqual(0.5 * 5 + 8, 10.5)
        self.assertEqual(10.5 / 1.75, 6)

    def test_decimal_oracle_uneven_blocks_and_value_width(self):
        rng = random.Random(2026)
        for nq, nk, d, dv in ((1, 1, 1, 1), (5, 7, 3, 2), (4, 9, 4, 1)):
            q = [[rng.uniform(-3, 3) for _ in range(d)] for _ in range(nq)]
            k = [[rng.uniform(-3, 3) for _ in range(d)] for _ in range(nk)]
            v = [[rng.uniform(-3, 3) for _ in range(dv)] for _ in range(nk)]
            allowed = [[rng.random() > .35 for _ in range(nk)] for _ in range(nq)]
            for mask in (None, allowed):
                expected = decimal_oracle(q, k, v, mask)
                assert_matrix(self, a.dense_attention(q, k, v, mask), expected)
                for qb in (1, 2, 9):
                    for kb in (1, 2, 3, 4, 16):
                        with self.subTest(nq=nq, nk=nk, qb=qb, kb=kb, masked=mask is not None):
                            assert_matrix(self, a.blocked_attention(q, k, v, mask, qb, kb), expected)

    def test_all_masked_first_block_and_later_maximum_increase(self):
        q, k, v = [[1], [1], [1]], [[1000], [1001], [1002], [1005], [1009]], [[1], [2], [4], [8], [16]]
        allowed = [[False] * 5, [False, False, True, True, True], [True, False, False, False, True]]
        expected = decimal_oracle(q, k, v, allowed)
        actual = a.blocked_attention(q, k, v, allowed, 2, 2)
        assert_matrix(self, actual, expected)
        self.assertEqual(actual[0], [0.0])
        assert_matrix(self, a.dense_attention(q, k, v, allowed), expected)

    def test_large_positive_and_negative_scores_and_shift_invariance(self):
        q, v = [[1]], [[-5], [4], [8]]
        baseline = decimal_oracle(q, [[0], [1], [2]], v)
        for shift in (-10000, 0, 10000):
            for block in (1, 2, 3):
                assert_matrix(self, a.blocked_attention(q, [[shift], [shift + 1], [shift + 2]], v,
                                                       key_block_size=block), baseline)

    def test_descending_extreme_scores_preserve_running_maximum(self):
        # The maximum can be in an EARLIER block, not only the newest block.
        for shift in (-10000, 0, 10000):
            for keys, values in (([[shift + 1000], [shift]], [[1], [2]]),
                                 ([[shift + 1000], [shift], [shift - 5]], [[1], [2], [3]])):
                q = [[1]]
                expected = decimal_oracle(q, keys, values)
                for block in (1, 2):
                    with self.subTest(shift=shift, keys=keys, block=block):
                        actual = a.blocked_attention(q, keys, values, key_block_size=block)
                        self.assertTrue(all(math.isfinite(x) for row in actual for x in row))
                        assert_matrix(self, actual, expected)
                        self.assertEqual(actual, [[1.0]])

    def test_absolute_causal_offset_and_future_mutation(self):
        mask = a.causal_mask(2, 5, 3)
        self.assertEqual(mask, [[True, True, True, True, False], [True] * 5])
        q, k, v = [[1, 0], [0, 1]], [[1, 2]] * 5, [[j, -j] for j in range(5)]
        actual = a.blocked_attention(q, k, v, mask)
        changed = a.blocked_attention(q, k[:-1] + [[9999, 9999]], v[:-1] + [[999, -999]], mask)
        assert_matrix(self, [actual[0]], [[1.5, -1.5]])
        self.assertEqual(actual[0], changed[0])
        self.assertNotEqual(actual[1], changed[1])

    def test_zero_queries_mean_visible_values(self):
        actual = a.blocked_attention([[0, 0]], [[1, 2], [3, 4], [5, 6]], [[1, 4], [5, 8], [9, 12]])
        assert_matrix(self, actual, [[5, 8]])

    def test_validation_shapes_masks_and_nonfinite(self):
        invalid = [([], [[1]], [[1]], None), ([[1]], [], [], None),
                   ([[1, 2]], [[1]], [[1]], None), ([[1]], [[1], [2]], [[1]], None),
                   ([[1], [2, 3]], [[1]], [[1]], None), ([[True]], [[1]], [[1]], None),
                   ([[math.inf]], [[1]], [[1]], None), ([[1]], [[math.nan]], [[1]], None),
                   ([[1]], [[1]], [[math.inf]], None), ([[1]], [[1]], [[1]], [[1]]),
                   ([[1]], [[1]], [[1]], []), ([[1]], [[1]], [[1]], [[True, False]])]
        for args in invalid:
            for implementation in (a.dense_attention, a.blocked_attention):
                with self.subTest(args=args, implementation=implementation.__name__), self.assertRaises(ValueError):
                    implementation(*args)
        for block in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                a.blocked_attention([[1]], [[1]], [[1]], key_block_size=block)
            with self.assertRaises(ValueError):
                a.blocked_attention([[1]], [[1]], [[1]], query_block_size=block)
        for args in ((0, 1, 0), (1, 0, 0), (2, 2, 1), (1, 1, True), (1, 1, -1)):
            with self.assertRaises(ValueError):
                a.causal_mask(*args)
        for implementation in (a.dense_attention, a.blocked_attention):
            with self.assertRaisesRegex(ValueError, 'overflowed'):
                implementation([[1e308]], [[1e308]], [[1]])


class PagedKVTests(unittest.TestCase):
    def fill(self, pool, request, tokens):
        pool.create(request)
        for token in tokens:
            pool.append(request, [token], [token * 10])

    def test_hand_block_tables_refs_and_copy_on_write(self):
        pool = a.PagedKVCache()
        self.fill(pool, 'A', [10, 20, 30])
        pool.fork('A', 'B')
        self.assertEqual(pool.block_table('A'), (0, 1))
        self.assertEqual(pool.block_table('B'), (0, 1))
        self.assertEqual(pool.refcounts(), {0: 2, 1: 2})
        pool.append('B', [40], [400])
        self.assertEqual(pool.block_table('B'), (0, 2))
        self.assertEqual(pool.refcounts(), {0: 2, 1: 1, 2: 1})
        self.assertEqual(pool.read('A')[0], [[10], [20], [30]])
        self.assertEqual(pool.read('B')[0], [[10], [20], [30], [40]])
        self.assertEqual(pool.stats(), {'logical_tokens': 7, 'physical_tokens': 5, 'allocated_slots': 6,
                                       'unused_slots': 1, 'physical_blocks': 3})
        pool.append('A', [99], [990])
        self.assertEqual(pool.read('B')[0][-1], [40])
        self.assertEqual(pool.read('A')[0][-1], [99])
        self.assertTrue(pool.check_invariants())

    def test_full_block_boundary_allocates_without_copying_full_prefix(self):
        pool = a.PagedKVCache()
        self.fill(pool, 'A', [1, 2])
        pool.fork('A', 'B')
        pool.append('B', [3], [30])
        self.assertEqual(pool.block_table('A'), (0,))
        self.assertEqual(pool.block_table('B'), (0, 1))
        self.assertEqual(pool.refcounts(), {0: 2, 1: 1})
        self.assertEqual(pool.stats()['physical_tokens'], 3)
        pool.release('A')
        self.assertEqual(pool.read('B')[0], [[1], [2], [3]])
        pool.release('B')
        self.assertEqual(pool.stats()['physical_blocks'], 0)
        self.assertTrue(pool.check_invariants())

    def test_whole_block_prefix_and_invalid_partial_prefix(self):
        pool = a.PagedKVCache()
        self.fill(pool, 'A', [1, 2, 3, 4, 5])
        pool.share_prefix('A', 'B', 4)
        self.assertEqual(pool.block_table('B'), (0, 1))
        self.assertEqual(pool.refcounts(), {0: 2, 1: 2, 2: 1})
        pool.append('B', [9], [90])
        self.assertEqual(pool.read('A')[0], [[1], [2], [3], [4], [5]])
        self.assertEqual(pool.read('B')[0], [[1], [2], [3], [4], [9]])
        for count in (-1, 1, 3, 6, True, 2.0):
            with self.assertRaises(ValueError):
                pool.share_prefix('A', 'invalid', count)
        pool.share_prefix('A', 'empty', 0)
        self.assertEqual(pool.read('empty'), ([], []))
        pool.fork('empty', 'empty2')
        self.assertTrue(pool.check_invariants())

    def test_logical_not_physical_order_and_reused_block_cleanliness(self):
        pool = a.PagedKVCache()
        self.fill(pool, 'temporary', [777])
        self.fill(pool, 'A', [1, 2, 3, 4])
        pool.release('temporary')
        pool.append('A', [5], [50])
        self.assertEqual(pool.block_table('A'), (1, 2, 0))
        self.assertEqual(pool.read('A'), ([[1], [2], [3], [4], [5]], [[10], [20], [30], [40], [50]]))
        keys, values = pool.read('A')
        assert_matrix(self, a.blocked_attention([[0]], keys, values), [[30]])
        self.assertTrue(pool.check_invariants())

    def test_capacity_failure_atomic_for_new_block_and_copy_on_write(self):
        for full in (False, True):
            pool = a.PagedKVCache(max_blocks=1)
            self.fill(pool, 'A', [1, 2] if full else [1])
            pool.fork('A', 'B')
            before = (pool.read('A'), pool.read('B'), pool.refcounts(), pool.stats(), pool.block_table('B'))
            with self.assertRaises(MemoryError):
                pool.append('B', [3], [30])
            self.assertEqual(before, (pool.read('A'), pool.read('B'), pool.refcounts(), pool.stats(), pool.block_table('B')))
            self.assertTrue(pool.check_invariants())
        pool.release('A')  # Unshared partial tail can append with a full physical pool.
        # The loop ended with full=True, so this still requires a new block.
        with self.assertRaises(MemoryError):
            pool.append('B', [3], [30])
        partial = a.PagedKVCache(max_blocks=1)
        self.fill(partial, 'A', [1])
        partial.fork('A', 'B')
        partial.release('A')
        partial.append('B', [2], [20])
        self.assertTrue(partial.check_invariants())

    def test_payload_snapshots_and_argument_validation(self):
        pool = a.PagedKVCache()
        pool.create('A')
        key, value = [2], [3]
        pool.append('A', key, value)
        key[0], value[0] = 99, 99
        read_keys, read_values = pool.read('A')
        read_keys[0][0] = 999
        pool.refcounts().clear()
        self.assertEqual(pool.read('A'), ([[2]], [[3]]))
        for call in (lambda: pool.create('A'), lambda: pool.create(''), lambda: pool.create(1),
                     lambda: pool.append('missing', [1], [1]), lambda: pool.release('missing'),
                     lambda: pool.fork('A', 'A'), lambda: pool.append('A', [1, 2], [1]),
                     lambda: pool.append('A', [math.nan], [1]), lambda: pool.read([])):
            with self.assertRaises(ValueError):
                call()
        for kwargs in ({'block_size': 0}, {'key_dim': True}, {'value_dim': -1}, {'max_blocks': 1.5}):
            with self.assertRaises(ValueError):
                a.PagedKVCache(**kwargs)
        self.assertTrue(pool.check_invariants())

    def test_randomized_operations_against_independent_sequence_model(self):
        rng = random.Random(42)
        pool = a.PagedKVCache(block_size=3, max_blocks=256)
        model = {}
        counter = 0
        for _ in range(240):
            action = rng.choice(('new', 'append', 'append', 'fork', 'prefix', 'release')) if model else 'new'
            if action == 'new':
                name = f'r{counter}'
                counter += 1
                pool.create(name)
                model[name] = []
            else:
                source = rng.choice(list(model))
                if action == 'append':
                    value = rng.randrange(1000)
                    pool.append(source, [value], [-value])
                    model[source].append(value)
                elif action in ('fork', 'prefix'):
                    target = f'r{counter}'
                    counter += 1
                    if action == 'fork':
                        pool.fork(source, target)
                        model[target] = list(model[source])
                    else:
                        count = rng.randrange(len(model[source]) // 3 + 1) * 3
                        pool.share_prefix(source, target, count)
                        model[target] = model[source][:count]
                else:
                    pool.release(source)
                    del model[source]
            for name, values in model.items():
                self.assertEqual(pool.read(name), ([[x] for x in values], [[-x] for x in values]))
            self.assertEqual(pool.stats()['logical_tokens'], sum(map(len, model.values())))
            self.assertTrue(pool.check_invariants())


class ServingArithmeticTests(unittest.TestCase):
    def test_chapter_memory_slots_address_and_hand_counterexample(self):
        self.assertEqual(4096 ** 2 * 2 // 2 ** 20, 32)
        self.assertEqual(8192 ** 2 * 2 // 2 ** 20, 128)
        blocks = (9 + 4 - 1) // 4
        self.assertEqual((blocks, blocks * 4 - 9), (3, 3))
        table = [9, 2, 7]
        self.assertEqual((table[7 // 4], 7 % 4), (2, 3))
        self.assertEqual((5 * 2 * 4, 6 * 2 * 4), (40, 48))
        self.assertEqual(3 * 1 * 1 * 8 * 16 * 2, 768)
        self.assertAlmostEqual((5 / 1.5 + 8) / 2, 17 / 3)
        continuous = a.scheduling_demo(True)
        self.assertEqual(continuous['token_times_ms']['C'][0] - 10, 20)

    def test_timestamps_ttft_tpot_and_throughput(self):
        self.assertEqual(a.request_metrics(0, [100, 140, 190]),
                         {'ttft_ms': 100, 'tpot_ms': 45, 'e2e_ms': 190, 'output_tokens': 3})
        self.assertEqual(a.request_metrics(50, [150, 210]),
                         {'ttft_ms': 100, 'tpot_ms': 60, 'e2e_ms': 160, 'output_tokens': 2})
        self.assertAlmostEqual(a.output_throughput(5, 0, 210), 23.80952380952381)
        self.assertEqual((45 + 60) / 2, 52.5)  # Per-request mean, not pooled intervals.
        self.assertEqual((90 + 60) / 3, 50)
        self.assertIsNone(a.request_metrics(10, [25])['tpot_ms'])
        self.assertEqual(a.request_metrics(10, [10, 10])['tpot_ms'], 0)

    def test_continuous_batching_concrete_timeline(self):
        static, continuous = a.scheduling_demo(False), a.scheduling_demo(True)
        self.assertEqual(static['timeline'], [('A', 'B'), ('A', 'B'), (None, 'B'), (None, 'B'), ('C', None), ('C', None)])
        self.assertEqual(continuous['timeline'], [('A', 'B'), ('A', 'B'), ('C', 'B'), ('C', 'B')])
        self.assertEqual(static['duration_ms'], 60)
        self.assertEqual(continuous['duration_ms'], 40)
        self.assertEqual(static['token_times_ms']['C'], [50, 60])
        self.assertEqual(continuous['token_times_ms']['C'], [30, 40])
        self.assertEqual(sum(map(len, continuous['token_times_ms'].values())), 8)
        self.assertEqual(a.output_throughput(8, 0, 40), 200)
        self.assertAlmostEqual(a.output_throughput(8, 0, 60), 133.33333333333334)

    def test_metrics_invalid_and_empty_cases(self):
        for arrival, times in ((0, []), (-1, [1]), (10, [9]), (0, [3, 2]), (0, [math.inf]), (True, [1]), (0, [True])):
            with self.assertRaises(ValueError):
                a.request_metrics(arrival, times)
        for args in ((1, 0, 0), (-1, 0, 1), (True, 0, 1), (1, 0, math.inf), (1, -1, 2)):
            with self.assertRaises(ValueError):
                a.output_throughput(*args)
        with self.assertRaises(ValueError):
            a.scheduling_demo(1)


class MutationTests(unittest.TestCase):
    def rejected_by_assertion(self, old, new, checker):
        checker(self, a)  # The same contract must pass on the unmodified implementation.
        source = SCRIPT.read_text()
        self.assertIn(old, source)
        mutated = types.ModuleType('p15_mutant')
        exec(compile(source.replace(old, new), str(SCRIPT), 'exec'), mutated.__dict__)
        class Contract(unittest.TestCase):
            def runTest(case):
                checker(case, mutated)
        result = unittest.TestResult()
        Contract().run(result)
        self.assertEqual(len(result.errors), 0, result.errors)
        self.assertEqual(len(result.failures), 1, 'Mutant must fail a semantic assertion, not crash')

    def test_lost_running_maximum_overflow_breaks_valid_input_contract(self):
        def descending(case, module):
            q, k, v = [[1]], [[1000], [0]], [[1], [2]]
            expected = decimal_oracle(q, k, v)
            try:
                actual = module.blocked_attention(q, k, v, key_block_size=1)
            except OverflowError as exc:
                # A legitimate numeric failure on valid input is the defect.
                # Do not swallow import errors, syntax errors or unrelated exceptions.
                case.fail(f"Valid finite descending scores must not overflow: {exc}")
            case.assertTrue(all(math.isfinite(x) for row in actual for x in row))
            assert_matrix(case, actual, expected)
        self.rejected_by_assertion('next_m = max(m, max(scores))',
                                   'next_m = max(scores)', descending)

    def test_nonzero_arrival_ttft_and_e2e_mutations(self):
        def timestamps(case, module):
            case.assertEqual(module.request_metrics(50, [150, 210]),
                             {'ttft_ms': 100, 'tpot_ms': 60, 'e2e_ms': 160, 'output_tokens': 2})
        self.rejected_by_assertion('token_times_ms[0] - arrival_ms',
                                   'token_times_ms[0]', timestamps)
        self.rejected_by_assertion('token_times_ms[-1] - arrival_ms',
                                   'token_times_ms[-1]', timestamps)

    def test_rescale_scale_mask_and_remainder_mutations(self):
        def online(case, module):
            q, k, v = [[1, 2]], [[0, 1], [2, 3], [4, 1], [5, 4], [8, 8]], [[2], [4], [8], [16], [32]]
            mask = [[False, True, True, True, False]]
            assert_matrix(case, module.blocked_attention(q, k, v, mask, 2, 2), decimal_oracle(q, k, v, mask))
        self.rejected_by_assertion('math.exp(m - next_m)', '1.0', online)
        # Both dense and blocked paths share this wrong scale; independent oracle catches it.
        self.rejected_by_assertion('scale = math.sqrt(len(q[0]))', 'scale = 1.0', online)
        self.rejected_by_assertion('if allowed is None or allowed[i][j]', 'if True', online)
        def remainder(case, module):
            assert_matrix(case, module.blocked_attention([[0]], [[1], [1], [1]], [[2], [4], [9]],
                                                        key_block_size=2), [[5]])
        self.rejected_by_assertion('range(0, len(k), key_block_size)',
                                   'range(0, len(k) - len(k) % key_block_size, key_block_size)', remainder)

    def test_copy_on_write_refcount_and_order_mutations(self):
        def cow(case, module):
            pool = module.PagedKVCache()
            pool.create('A')
            for x in (1, 2, 3):
                pool.append('A', [x], [x])
            pool.fork('A', 'B')
            pool.append('B', [4], [4])
            case.assertEqual(pool.block_table('A'), (0, 1))
            case.assertEqual(pool.block_table('B'), (0, 2))
            case.assertEqual(pool.refcounts(), {0: 2, 1: 1, 2: 1})
        self.rejected_by_assertion('elif self._refs[table[-1]] > 1:', 'elif False:', cow)
        self.rejected_by_assertion('self._refs[old] -= 1', 'self._refs[old] -= 0', cow)
        self.rejected_by_assertion('self._refs[block] += 1', 'self._refs[block] += 0', cow)
        def order(case, module):
            pool = module.PagedKVCache()
            pool.create('temporary')
            pool.append('temporary', [9], [9])
            pool.create('A')
            pool.append('A', [1], [1])
            pool.append('A', [2], [2])
            pool.release('temporary')
            pool.append('A', [3], [3])
            case.assertEqual(pool.block_table('A'), (1, 0))
            case.assertEqual(pool.read('A')[0], [[1], [2], [3]])
        self.rejected_by_assertion('payload = [item for block in self._tables[request]',
                                   'payload = [item for block in sorted(self._tables[request])', order)

    def test_wrong_tpot_denominator_mutation(self):
        self.rejected_by_assertion('/ (count - 1) if count > 1', '/ count if count > 1',
                                   lambda case, module: case.assertEqual(module.request_metrics(0, [100, 140, 190])['tpot_ms'], 45))


class CLITests(unittest.TestCase):
    def test_no_torch_core_runs_and_optional_gpu_skips_explicitly(self):
        blocker = """
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise ModuleNotFoundError("No module named 'torch'", name='torch')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name='__main__')
"""
        for extra in ([], ['--gpu-sdpa']):
            result = subprocess.run([sys.executable, '-c', blocker, str(SCRIPT), *extra], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn('Traceback', result.stderr)
            self.assertIn('output=6.000000000000', result.stdout)
            self.assertIn('SKIP GPU SDPA: PyTorch is not installed' if extra else 'GPU SDPA: not requested, not run', result.stdout)


@unittest.skipUnless(READY, torch_reason('P15 optional SDPA cross-check', VERSION, PRESENT))
class OptionalTorchTests(unittest.TestCase):
    def test_cpu_sdpa_against_independent_decimal_oracle(self):
        import torch
        torch.set_num_threads(1)
        q, k, v = [[1, 2], [2, -1]], [[1, 0], [0, 1], [2, 3]], [[1, 2], [4, 8], [16, 32]]
        mask = [[True, True, False], [False, False, False]]
        tensors = [torch.tensor(x, dtype=torch.float64)[None, None] for x in (q, k, v)]
        output = torch.nn.functional.scaled_dot_product_attention(*tensors, attn_mask=torch.tensor(mask), dropout_p=0.0)
        assert_matrix(self, output[0, 0].tolist(), decimal_oracle(q, k, v, mask))


if __name__ == '__main__':
    unittest.main()
