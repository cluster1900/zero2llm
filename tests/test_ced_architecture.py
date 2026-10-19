"""P17: independent scalar oracle, cache contracts and runnable wrong-path mutants."""
from dataclasses import replace
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import types
import unittest

from torch_test_support import torch_status, torch_reason

READY, VERSION, PRESENT = torch_status()
SCRIPT = Path(__file__).resolve().parents[1] / 'code/18_ced_architecture.py'
if READY:
    import torch
    spec = importlib.util.spec_from_file_location('ced_test', SCRIPT)
    c = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = c
    spec.loader.exec_module(c)


def scalar_oracle(model, inputs, replay_start=0):
    """Python scalars, no production attention/mask/project/cache helpers.

    Row-vector multiplication and softmax are separately looped. Only the
    declared weights/config are shared; scalar positions are absolute.
    """
    def multiply(row, matrix):
        return [sum(row[a] * matrix[a][j] for a in range(2)) for j in range(2)]

    def attend(query, keys, values, visible):
        scores = [sum(query[d] * keys[j][d] for d in range(2)) / math.sqrt(2) for j in visible]
        exps = [math.exp(s - max(scores)) for s in scores]
        total = sum(exps)
        return [sum(w * values[j][d] for w, j in zip(exps, visible)) / total for d in range(2)]

    outputs, encoders = [], []
    win = model.config.window
    for batch in inputs.tolist():
        h = [[row[0] + t * model.config.position_scale,
              row[1] - t * model.config.position_scale] for t, row in enumerate(batch)]
        for layer in model.enc:
            q, k, v = [getattr(layer, name).tolist() for name in ('q', 'k', 'v')]
            keys, values = [multiply(row, k) for row in h], [multiply(row, v) for row in h]
            result = []
            for t, row in enumerate(h):
                value = attend(multiply(row, q), keys, values, range(max(0, t-win+1), t+1))
                result.append([row[d] + value[d] for d in range(2)])
            h = result
        encoder = [row[:] for row in h]
        encoders.append(encoder)
        h = h[replay_start:]
        for layer in model.dec:
            q, k, v, gk, gv = [getattr(layer, name).tolist() for name in ('q', 'k', 'v', 'global_k', 'global_v')]
            keys, values = [multiply(row, k) for row in h], [multiply(row, v) for row in h]
            global_keys = [multiply(row, gk) for row in encoder]
            global_values = [multiply(row, gv) for row in encoder]
            result = []
            for row_id, row in enumerate(h):
                absolute = replay_start + row_id
                query = multiply(row, q)
                global_out = attend(query, global_keys, global_values, range(absolute+1))
                local_out = attend(query, keys, values, range(max(0, row_id-win+1), row_id+1))
                result.append([row[d] + global_out[d] + local_out[d] for d in range(2)])
            h = result
        outputs.append(h)
    return torch.tensor(outputs, dtype=torch.float64), torch.tensor(encoders, dtype=torch.float64)


class CEDCLITests(unittest.TestCase):
    def test_absent_torch_has_friendly_cli_but_import_raises(self):
        blocker = """
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'torch':
        raise ModuleNotFoundError("No module named 'torch'", name='torch')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
"""
        result = subprocess.run([sys.executable, '-c', blocker +
            "runpy.run_path(sys.argv[1], run_name='__main__')", str(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('P17 requires PyTorch', result.stderr)
        self.assertNotIn('Traceback', result.stderr)
        result = subprocess.run([sys.executable, '-c', blocker +
            "runpy.run_path(sys.argv[1], run_name='imported_ced')", str(SCRIPT)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('ModuleNotFoundError', result.stderr)


@unittest.skipUnless(READY, torch_reason('P17 CED tests', VERSION, PRESENT))
class CEDTests(unittest.TestCase):
    def setUp(self):
        self.model = c.ToyCED()
        self.x = torch.tensor([[[1., .5], [-1., 2.], [3., -.5], [.5, 1.], [2., -2.], [1., 3.]],
                               [[-2., 1.], [3., -.5], [1., 2.], [2., -1.], [-3., 2.], [4., .5]]], dtype=torch.float64)

    def close(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def oracle_assertion(self, model):
        expected, encoder = scalar_oracle(model, self.x)
        actual = model(self.x)
        self.close(actual.output, expected)
        self.close(actual.encoder, encoder)

    def test_full_and_bounded_against_independent_scalar_oracle(self):
        self.oracle_assertion(self.model)
        for n in (1, 2, 3, 4, 6):
            x = self.x[:, :n]
            start = max(0, n-self.model.config.window)
            expected, encoder = scalar_oracle(self.model, x, start)
            actual = self.model.bounded_prefill(x)
            self.close(actual.output, expected)
            self.close(actual.encoder, encoder)
            self.assertEqual(actual.output_start, start)

    def test_hand_numbers_are_not_just_full_cache_agreement(self):
        full, bounded = c.hand_walkthrough()
        self.close(full.encoder[0, :, 0], torch.tensor([2., 7/2, 11/2], dtype=torch.float64))
        self.close(full.output[0, :, 0], torch.tensor([6., 9., 41/3], dtype=torch.float64))
        self.close(bounded.output[0, :, 0], torch.tensor([39/4, 41/3], dtype=torch.float64))
        two = c.ToyCED(c.Config(1, 2, 2, 0.0, True))
        x = torch.tensor([[[1., 0.], [2., 0.], [3., 0.]]], dtype=torch.float64)
        exact_two, replay_two = two(x), two.bounded_prefill(x)
        self.close(exact_two.output[0, -1, 0], torch.tensor(86/3, dtype=torch.float64))
        self.close(replay_two.output[0, -1, 0], torch.tensor(697/24, dtype=torch.float64))
        self.close(replay_two.output[0, -1, 0] - exact_two.output[0, -1, 0], torch.tensor(3/8, dtype=torch.float64))
        # A one-layer last row matching does NOT establish multilayer replay equivalence.
        exact = self.model(self.x)
        replay = self.model.bounded_prefill(self.x)
        self.assertGreater((exact.output[:, -1] - replay.output[:, -1]).abs().max().item(), 1e-6)

    def test_encoder_global_edge_and_layer_local_edge(self):
        result = self.model(self.x)
        for index, layer in enumerate(self.model.dec):
            self.close(result.cache.decoder_global[index].k, result.encoder @ layer.global_k)
            self.close(result.cache.decoder_global[index].v, result.encoder @ layer.global_v)
        original_global = tuple(kv.v.clone() for kv in result.cache.decoder_global)
        # Intervene on lower decoder's LOCAL values, leaving encoder/global projections intact.
        self.model.dec[0].v.mul_(3)
        changed = self.model(self.x)
        for kv, before in zip(changed.cache.decoder_global, original_global):
            self.close(kv.v, before)
        self.assertGreater((changed.cache.decoder_local[1].v - result.cache.decoder_local[1].v).abs().max().item(), .001)
        self.assertGreater((changed.output - result.output).abs().max().item(), .001)

    def test_future_intervention_at_every_prefix_cut(self):
        reference = self.model(self.x)
        for cut in range(1, self.x.shape[1]):
            changed = self.x.clone()
            changed[:, cut:] = changed[:, cut:] * -7 + 17
            result = self.model(changed)
            self.close(result.encoder[:, :cut], reference.encoder[:, :cut])
            self.close(result.output[:, :cut], reference.output[:, :cut])
            for new, old in zip(result.cache.decoder_global, reference.cache.decoder_global):
                self.close(new.k[:, :cut], old.k[:, :cut])
                self.close(new.v[:, :cut], old.v[:, :cut])
        # A precomputed global bank must not leak its later rows into early replay queries.
        replay = self.model.bounded_prefill(self.x)
        changed = self.x.clone(); changed[:, -1] += 30
        result = self.model.bounded_prefill(changed)
        self.close(result.output[:, :-1], replay.output[:, :-1])

    def test_all_two_chunk_splits_and_single_token_decode(self):
        expected, _ = scalar_oracle(self.model, self.x)
        for cut in range(1, self.x.shape[1]):
            first = self.model(self.x[:, :cut])
            second = self.model(self.x[:, cut:], first.cache)
            self.close(torch.cat((first.output, second.output), 1), expected)
            self.assertEqual(second.output_start, cut)
            self.assertEqual(second.cache.next_position, 6)
        cache, parts = None, []
        for t in range(6):
            result = self.model(self.x[:, t:t+1], cache)
            cache = result.cache; parts.append(result.output)
        self.close(torch.cat(parts, 1), expected)
        for kv in cache.encoder_local + cache.decoder_local:
            self.assertEqual((kv.start, kv.k.shape[1]), (4, 2))
        for kv in cache.decoder_global:
            self.assertEqual((kv.start, kv.k.shape[1]), (0, 6))

    def test_snapshot_reset_and_branch_are_isolated(self):
        first = self.model(self.x[:, :3])
        saved = [(kv.k.clone(), kv.v.clone()) for kv in first.cache.encoder_local + first.cache.decoder_global + first.cache.decoder_local]
        self.model(self.x[:, 3:], first.cache)
        for kv, (k, v) in zip(first.cache.encoder_local + first.cache.decoder_global + first.cache.decoder_local, saved):
            self.close(kv.k, k); self.close(kv.v, v)
        branch = self.model(self.x[:, 3:4] + 2, first.cache)
        self.close(branch.output, self.model(torch.cat((self.x[:, :3], self.x[:, 3:4]+2), 1)).output[:, -1:])
        self.close(self.model(self.x.flip(1)).output, scalar_oracle(self.model, self.x.flip(1))[0])

    def test_replay_cache_keeps_global_prefix_and_approximate_provenance(self):
        exact, replay = self.model(self.x[:, :5]), self.model.bounded_prefill(self.x[:, :5])
        for a, b in zip(exact.cache.decoder_global, replay.cache.decoder_global):
            self.close(a.k, b.k); self.close(a.v, b.v)
        continued = self.model(self.x[:, 5:], replay.cache)
        self.assertTrue(continued.cache.approximate)
        self.assertEqual(continued.output_start, 5)
        self.assertEqual(continued.cache.next_position, 6)
        self.assertGreater((continued.output - self.model(self.x).output[:, -1:]).abs().max().item(), 1e-6)
        self.assertFalse(self.model(self.x).cache.approximate)

    def test_window_one_and_prompt_shorter_than_window(self):
        for win in (1, 2, 8):
            model = c.ToyCED(c.Config(window=win))
            exact, replay = model(self.x), model.bounded_prefill(self.x)
            self.close(exact.output, scalar_oracle(model, self.x)[0])
            if win >= 6:
                self.close(exact.output, replay.output)
                self.assertFalse(replay.cache.approximate)
            if win == 1:
                for kv in exact.cache.encoder_local + exact.cache.decoder_local:
                    self.assertEqual((kv.start, kv.k.shape[1]), (6, 0))
                self.close(model(self.x[:, :1], exact.cache).output,
                           model(torch.cat((self.x, self.x[:, :1]), 1)).output[:, -1:])

    def test_actual_work_counts_and_transparent_budget(self):
        for n in (1, 2, 3, 6):
            exact, replay = self.model(self.x[:, :n]), self.model.bounded_prefill(self.x[:, :n])
            budget = c.path_budget(n, 2, 2, 3, batch=2)
            self.assertEqual(exact.work.block_visits, 2*n*4)
            self.assertEqual(replay.work.block_visits, 2*(n*2 + min(n, 3)*2))
            self.assertEqual(exact.work.block_visits, budget['full_blocks'])
            self.assertEqual(replay.work.block_visits, budget['bounded_blocks'])
            self.assertEqual(exact.work.global_kv_token_layers, 2*n*2)
            self.assertEqual(replay.work.global_kv_token_layers, 2*n*2)
        self.assertEqual(c.path_budget(1000, 20, 20, 128), {
            'full_blocks': 40000, 'bounded_blocks': 22560,
            'global_projection_pairs_each_path': 20000, 'replay_tokens': 128})

    def test_invalid_inputs_config_and_cache_are_rejected(self):
        for kwargs in ({'window': 0}, {'encoder_layers': True}, {'decoder_layers': 1.5}, {'position_scale': float('nan')}):
            with self.assertRaises(ValueError): c.Config(**kwargs)
        for x in (self.x.float(), self.x[:, :0], self.x[0], torch.full_like(self.x, float('nan'))):
            with self.assertRaises(ValueError): self.model(x)
        first = self.model(self.x[:, :3])
        with self.assertRaises(ValueError): c.ToyCED()(self.x[:, 3:], first.cache)
        with self.assertRaises(ValueError): self.model(self.x[:1, 3:], first.cache)
        for bad in (replace(first.cache, next_position=4), replace(first.cache, encoder_local=()),
                    replace(first.cache, decoder_global=(replace(first.cache.decoder_global[0], start=1),) + first.cache.decoder_global[1:])):
            with self.assertRaises(ValueError): self.model(self.x[:, 3:], bad)
        self.model.dec[0].q.add_(.1)
        with self.assertRaisesRegex(ValueError, 'weights/config changed'): self.model(self.x[:, 3:], first.cache)
        for args in ((0, 2, 2, 3), (3, 2, 2, True)):
            with self.assertRaises(ValueError): c.path_budget(*args)

    def mutant(self, old, new, label):
        source = SCRIPT.read_text()
        self.assertIn(old, source)
        name = 'ced_mutant_' + label
        module = types.ModuleType(name); module.__file__ = str(SCRIPT)
        sys.modules[name] = module
        self.addCleanup(sys.modules.pop, name, None)
        exec(compile(source.replace(old, new), str(SCRIPT), 'exec'), module.__dict__)
        return module

    def assert_detected(self, assertion):
        """Mutation evidence must be an assertion FAILURE, not a runtime ERROR."""
        report = unittest.TestResult()
        unittest.FunctionTestCase(assertion).run(report)
        self.assertEqual(report.testsRun, 1)
        self.assertEqual(len(report.errors), 0, report.errors)
        self.assertEqual(len(report.failures), 1, report.failures)

    def test_wrong_global_source_and_missing_local_window_fail_oracle(self):
        mutations = [
            ('global_out = attention(q, global_kv[i], offset)',
             'global_out = attention(q, KV(h @ layer.global_k, h @ layer.global_v, offset), offset)', 'decoder_source'),
            ('local_out = attention(q, kv, offset, self.config.window)',
             'local_out = attention(q, kv, offset)', 'no_window'),
            ('allowed = key_positions <= query_positions',
             'allowed = torch.ones_like(key_positions + query_positions, dtype=torch.bool)', 'future')]
        for old, new, name in mutations:
            with self.subTest(name=name):
                mutant = self.mutant(old, new, name)
                model = mutant.ToyCED()
                self.assert_detected(lambda: self.oracle_assertion(model))

    def test_wrong_offset_fails_incremental_oracle(self):
        mutant = self.mutant('positions = torch.arange(offset, offset + x.shape[1], dtype=x.dtype)',
                             'positions = torch.arange(x.shape[1], dtype=x.dtype)', 'position')
        model = mutant.ToyCED()
        first = model(self.x[:, :3]); second = model(self.x[:, 3:], first.cache)
        expected = scalar_oracle(model, self.x)[0]
        self.assert_detected(lambda: self.close(torch.cat((first.output, second.output), 1), expected))

    def test_wrong_replay_global_position_fails_bounded_oracle(self):
        mutant = self.mutant('global_out = attention(q, global_kv[i], offset)',
                             'global_out = attention(q, global_kv[i], 0)', 'replay_position')
        model = mutant.ToyCED()
        result = model.bounded_prefill(self.x)
        expected = scalar_oracle(model, self.x, 3)[0]
        self.assert_detected(lambda: self.close(result.output, expected))

    def test_builtin_selftest(self):
        c.self_test()


if __name__ == '__main__':
    unittest.main()
