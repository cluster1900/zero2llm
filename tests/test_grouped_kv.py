"""P14: independent head mapping, compact cache, MLA algebra, and mutations."""
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
SCRIPT = Path(__file__).resolve().parents[1] / 'code/15_grouped_kv.py'
if READY:
    import torch
    spec = importlib.util.spec_from_file_location('grouped_kv_test', SCRIPT)
    g = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = g
    spec.loader.exec_module(g)


class GroupedCLITests(unittest.TestCase):
    def test_missing_torch_cli_and_import_contract(self):
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
            "runpy.run_path(sys.argv[1], run_name='__main__')", str(SCRIPT)],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('python -m pip install -r requirements.txt', result.stderr)
        self.assertNotIn('Traceback', result.stderr)
        imported = subprocess.run([sys.executable, '-c', blocker + """
try:
    runpy.run_path(sys.argv[1], run_name='grouped_without_torch')
except ModuleNotFoundError as exc:
    assert exc.name == 'torch'
else:
    raise AssertionError('import should report missing dependency')
""", str(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(imported.returncode, 0, imported.stderr)


@unittest.skipUnless(READY, torch_reason('P14', VERSION, PRESENT))
class GroupedKVTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(57)
        self.config = g.GroupedConfig(vocab_size=17, d_model=16, num_heads=4,
                                      num_layers=2, max_seq_len=16, num_kv_heads=2)
        self.model = g.GroupedGPT(self.config).eval()
        self.tokens = torch.randint(17, (2, 12))

    def loop_reference(self, q, k, v, offset=0):
        # Deliberately no repeat/repeat_interleave, shared helper, or head_mapping.
        result = torch.zeros_like(q)
        for batch in range(q.shape[0]):
            for head in range(q.shape[1]):
                kv_head = head // (q.shape[1] // k.shape[1])
                for t in range(q.shape[2]):
                    end = offset + t + 1
                    scores = k[batch, kv_head, :end] @ q[batch, head, t] / math.sqrt(q.shape[-1])
                    result[batch, head, t] = scores.softmax(0) @ v[batch, kv_head, :end]
        return result

    def chunks(self, model, tokens, sizes, module=None):
        module = g if module is None else module
        start, cache, outputs = 0, None, []
        for size in sizes:
            out, cache = module.cached_forward(model, tokens[:, start:start + size], cache)
            outputs.append(out)
            start += size
        self.assertEqual(start, tokens.shape[1])
        return torch.cat(outputs, 1), cache

    def test_head_mapping_hand_values_not_tiled(self):
        self.assertEqual(g.head_mapping(4, 2), (0, 0, 1, 1))
        self.assertEqual(g.head_mapping(4, 4), (0, 1, 2, 3))
        self.assertEqual(g.head_mapping(4, 1), (0, 0, 0, 0))
        self.assertEqual(g.head_mapping(1, 1), (0,))
        self.assertEqual(g.head_mapping(8, 2)[5], 1)
        self.assertEqual(g.mapping_walkthrough().tolist(), [[10, 11], [10, 11], [20, 21], [20, 21]])

    def test_independent_reference_all_head_boundaries_and_offsets(self):
        for Hq, Hkv in ((1, 1), (4, 4), (4, 2), (4, 1), (6, 3)):
            for offset in (0, 3):
                with self.subTest(Hq=Hq, Hkv=Hkv, offset=offset):
                    q = torch.randn(2, Hq, 3, 4, dtype=torch.float64)
                    k = torch.randn(2, Hkv, offset + 3, 4, dtype=torch.float64)
                    v = torch.randn_like(k)
                    torch.testing.assert_close(g.grouped_attention(q, k, v, offset),
                                               self.loop_reference(q, k, v, offset), rtol=1e-12, atol=1e-12)

    def test_invalid_head_and_configuration_values(self):
        for Hq, Hkv in ((4, 3), (4, 8), (0, 1), (4, 0), (4, -1), (4, True), (4, 2.0)):
            with self.subTest(Hq=Hq, Hkv=Hkv), self.assertRaises(ValueError):
                g.head_mapping(Hq, Hkv)
        for Hkv in (0, 3, 8, True, 2.0):
            with self.assertRaises(ValueError):
                replace(self.config, num_kv_heads=Hkv)
        with self.assertRaises(ValueError):
            replace(self.config, d_model=15)
        with self.assertRaises(ValueError):
            g.GroupedGPT(g.s3.ModernConfig(vocab_size=17))

    def test_grouped_attention_validation(self):
        q, k, v = torch.randn(2, 4, 2, 4), torch.randn(2, 2, 3, 4), torch.randn(2, 2, 3, 4)
        cases = ((q[0], k, v, 1), (q, k, v[..., :2], 1), (q, k[:1], v[:1], 1),
                 (q.long(), k.long(), v.long(), 1), (q, k, v, 0), (q, k, v, -1), (q, k, v, True), (q, k.double(), v.double(), 1),
                 (q[:, :, :0], k, v, 3), (q, k[:, :0], v[:, :0], 1))
        for args in cases:
            with self.subTest(shapes=[getattr(x, 'shape', x) for x in args]), self.assertRaises(ValueError):
                g.grouped_attention(*args)

    def test_actual_projection_shapes_and_parameter_counts(self):
        for Hkv in (4, 2, 1):
            model = g.GroupedGPT(replace(self.config, num_kv_heads=Hkv))
            for block in model.blocks:
                self.assertEqual(block.attn.W_q.weight.shape, (16, 16))
                self.assertEqual(block.attn.W_k.weight.shape, (Hkv * 4, 16))
                self.assertEqual(block.attn.W_v.weight.shape, (Hkv * 4, 16))
                q, k, v = block.attn.project(torch.randn(2, 3, 16))
                self.assertEqual(q.shape, (2, 4, 3, 4))
                self.assertEqual(k.shape, (2, Hkv, 3, 4))
                self.assertEqual(v.shape, k.shape)

    def test_mha_boundary_exactly_matches_s3_same_weights(self):
        base = g.s3.ModernGPT(g.s3.ModernConfig(vocab_size=17, d_model=16, num_heads=4,
                                              num_layers=2, max_seq_len=16)).eval()
        model = g.GroupedGPT(replace(self.config, num_kv_heads=4)).eval()
        model.load_state_dict(base.state_dict(), strict=True)
        torch.testing.assert_close(model(self.tokens)[0], base(self.tokens)[0], rtol=0, atol=0)
        _, cache = g.cached_forward(model, self.tokens)
        _, reference = g.s3.cached_forward(base, self.tokens)
        for actual_pair, reference_pair in zip(cache.layers, reference.layers):
            for actual, expected in zip(actual_pair, reference_pair):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_same_seed_mha_boundary_keeps_s3_initial_weights(self):
        torch.manual_seed(123)
        base = g.s3.ModernGPT(g.s3.ModernConfig(vocab_size=17))
        torch.manual_seed(123)
        model = g.GroupedGPT(g.GroupedConfig(vocab_size=17, num_kv_heads=4))
        for key, value in base.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)

    def test_full_cache_logits_chunks_single_tokens_and_partial_rope(self):
        for Hkv in (4, 2, 1):
            for rotary_dim in (None, 2):
                model = g.GroupedGPT(replace(self.config, num_kv_heads=Hkv, rotary_dim=rotary_dim)).eval()
                full = model(self.tokens)[0]
                for sizes in ([12], [1] * 12, [3, 4, 1, 4], [11, 1]):
                    with self.subTest(Hkv=Hkv, rotary_dim=rotary_dim, sizes=sizes):
                        actual, cache = self.chunks(model, self.tokens, sizes)
                        torch.testing.assert_close(actual, full, rtol=1e-5, atol=3e-6)
                        self.assertEqual(cache.layers[0][0].shape, (2, Hkv, 12, 4))
        one, cache = g.cached_forward(self.model, self.tokens[:, :1])
        torch.testing.assert_close(one, self.model(self.tokens[:, :1])[0])
        self.assertEqual(cache.length, 1)

    def test_future_token_mutation_does_not_change_prefix(self):
        changed = self.tokens.clone()
        changed[:, 6:] = (changed[:, 6:] + 1) % 17
        torch.testing.assert_close(self.model(self.tokens)[0][:, :6], self.model(changed)[0][:, :6])
        old_logits, _ = self.chunks(self.model, self.tokens, [3, 9])
        new_logits, _ = self.chunks(self.model, changed, [3, 9])
        torch.testing.assert_close(old_logits[:, :6], new_logits[:, :6])

    def test_float64_cache_memory_is_actual_compact_payload(self):
        for dtype in (torch.float32, torch.float64):
            self.model.to(dtype)
            _, cache = g.cached_forward(self.model, self.tokens)
            expected = 2 * 2 * 2 * 12 * 2 * 4 * torch.empty((), dtype=dtype).element_size()
            self.assertEqual(g.cache_bytes(cache), expected)
            self.assertEqual(g.cache_bytes(cache), g.expected_cache_bytes(self.config, 2, 12, dtype))
            for pair in cache.layers:
                for tensor in pair:
                    self.assertEqual(tensor.shape, (2, 2, 12, 4))
                    self.assertEqual(tensor.untyped_storage().nbytes(), tensor.numel() * tensor.element_size())
            torch.testing.assert_close(self.model(self.tokens)[0], self.chunks(self.model, self.tokens, [3, 9])[0])
        self.assertEqual(g.expected_cache_bytes(self.config, 1, 0), 0)
        for b, t in ((0, 1), (1, -1), (1, 17), (True, 1), (1, 1.5)):
            with self.assertRaises(ValueError):
                g.expected_cache_bytes(self.config, b, t)

    def test_chapter_memory_and_projection_table(self):
        for Hkv, params, payload in ((4, 9216, 18432), (2, 4608, 9216), (1, 2304, 4608)):
            config = g.GroupedConfig(vocab_size=17, num_kv_heads=Hkv)
            model = g.GroupedGPT(config).eval()
            actual = sum(b.attn.W_k.weight.numel() + b.attn.W_v.weight.numel() for b in model.blocks)
            self.assertEqual(actual, params)
            _, cache = g.cached_forward(model, self.tokens)
            self.assertEqual(g.cache_bytes(cache), payload)
            self.assertEqual(g.expected_cache_bytes(config, 2, 12), payload)

    def test_cache_snapshot_branch_and_reset_isolation(self):
        _, first = g.cached_forward(self.model, self.tokens[:, :4])
        snapshots = tuple(tuple(t.clone() for t in pair) for pair in first.layers)
        _, extended = g.cached_forward(self.model, self.tokens[:, 4:7], first)
        self.assertEqual(first.length, 4)
        self.assertEqual(extended.length, 7)
        for original, snapshot in zip(first.layers, snapshots):
            for tensor, expected in zip(original, snapshot):
                torch.testing.assert_close(tensor, expected, rtol=0, atol=0)
        other = self.tokens.flip(1)
        fresh, _ = g.cached_forward(self.model, other, None)
        torch.testing.assert_close(fresh, self.model(other)[0])
        branch, _ = g.cached_forward(self.model, self.tokens[:, 4:5], first)
        torch.testing.assert_close(branch, self.model(self.tokens[:, :5])[0][:, 4:5])

    def test_foreign_cache_training_and_capacity_rejected(self):
        _, cache = g.cached_forward(self.model, self.tokens)
        other = g.GroupedGPT(self.config).eval()
        with self.assertRaisesRegex(ValueError, 'different'):
            g.cached_forward(other, self.tokens[:, :1], cache)
        with self.assertRaisesRegex(ValueError, 'max_seq_len'):
            g.cached_forward(self.model, self.tokens[:, :5], cache)
        _, limit = g.cached_forward(self.model, self.tokens[:, :4], cache)
        self.assertEqual(limit.length, 16)
        with self.assertRaisesRegex(ValueError, 'eval'):
            g.cached_forward(self.model.train(), self.tokens)
        self.model.eval()
        for tokens in (self.tokens.float(), self.tokens[:, :0], torch.full((1, 1), 17), torch.full((1, 1), -1)):
            with self.assertRaises(ValueError):
                g.cached_forward(self.model, tokens)
        with self.assertRaises(ValueError):
            g.cached_forward(g.s3.ModernGPT(g.s3.ModernConfig(vocab_size=17)).eval(), self.tokens)

    def test_malformed_cache_structure_shape_dtype(self):
        _, cache = g.cached_forward(self.model, self.tokens[:, :4])
        k, v = cache.layers[0]
        cases = [(None, 'layer count'), ((), 'layer count'), ((cache.layers[0],), 'layer count'),
                 (((k,), cache.layers[1]), 'K/V pair'), (((None, v), cache.layers[1]), 'shape'),
                 (((k[0], v[0]), cache.layers[1]), 'shape'),
                 (((k.repeat_interleave(2, 1), v.repeat_interleave(2, 1)), cache.layers[1]), 'shape'),
                 (((k[..., :2], v[..., :2]), cache.layers[1]), 'shape'),
                 (((k, v[:, :, :3]), cache.layers[1]), 'shape'),
                 ((cache.layers[0], (k[:, :, :3], v[:, :, :3])), 'shape'),
                 (((k.double(), v.double()), cache.layers[1]), 'dtype/device'),
                 (((k.to('meta'), v.to('meta')), cache.layers[1]), 'dtype/device')]
        for layers, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                g.cached_forward(self.model, self.tokens[:, 4:5], g.GroupedCache(layers, self.model, cache.versions))
        with self.assertRaisesRegex(ValueError, 'shape'):
            g.cached_forward(self.model, self.tokens[:1, :1], cache)

    def test_weight_position_norm_and_head_setting_mutations_invalidate_cache(self):
        mutations = [lambda m: m.head.weight.add_(.01), lambda m: m.double(),
                     lambda m: setattr(m.blocks[0].attn, 'rope_base', 20.),
                     lambda m: setattr(m.blocks[0].attn, 'rotary_dim', 2),
                     lambda m: setattr(m.blocks[0].attn, 'num_kv_heads', 1),
                     lambda m: setattr(m.ln_f, 'eps', .1),
                     lambda m: setattr(m, 'max_seq_len', 32),
                     lambda m: setattr(m, 'config', replace(m.config, rope_base=20.)),
                     lambda m: m.load_state_dict(g.GroupedGPT(self.config).state_dict(), assign=True)]
        for mutation in mutations:
            model = g.GroupedGPT(self.config).eval()
            _, cache = g.cached_forward(model, self.tokens[:, :4])
            with torch.no_grad():
                mutation(model)
            with self.assertRaisesRegex(ValueError, 'changed'):
                g.cached_forward(model, self.tokens[:, 4:5], cache)

    def test_training_all_heads_have_finite_gradients(self):
        for Hkv in (4, 2, 1):
            model = g.GroupedGPT(replace(self.config, num_kv_heads=Hkv)).train()
            logits, loss = model(self.tokens, self.tokens.roll(-1, 1))
            self.assertTrue(torch.isfinite(logits).all())
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_checkpoint_mean_pool_initialization_is_not_equivalence(self):
        weights = torch.arange(8, dtype=torch.float64).reshape(4, 2)
        pooled = g.mean_pool_kv_weight(weights, 4, 2)
        self.assertEqual(pooled.tolist(), [[1., 2.], [5., 6.]])
        torch.testing.assert_close(g.mean_pool_kv_weight(weights, 4, 4), weights)
        self.assertEqual(g.mean_pool_kv_weight(weights, 4, 1).tolist(), [[3., 4.]])
        # A concrete query-weighted output changes after pooling, not just shape.
        q = torch.ones(1, 4, 1, 1, dtype=torch.float64)
        k = torch.zeros(1, 4, 1, 1, dtype=torch.float64)
        values = torch.tensor([1., 3., 5., 7.], dtype=torch.float64).reshape(1, 4, 1, 1)
        original = g.grouped_attention(q, k, values)
        shared_values = values.reshape(1, 2, 2, 1, 1).mean(2)
        converted = g.grouped_attention(q, k[:, :2], shared_values)
        self.assertFalse(torch.allclose(original, converted))
        self.assertEqual(converted.flatten().tolist(), [2., 2., 6., 6.])
        for bad in (torch.ones(3, 2), torch.ones(4), torch.ones(4, 2, dtype=torch.long), torch.empty(0, 2)):
            with self.assertRaises(ValueError):
                g.mean_pool_kv_weight(bad, 4, 2)

    @staticmethod
    def mla_independent_reference():
        """Scalar-loop oracle from the documented fixture, not demo intermediates.

        Deliberately restate x/weights here so editing the demo cannot silently
        edit its expected answer. No demo helper, einsum, apply_rope, latent
        absorption path, or production scores enter this calculation.
        """
        x = [[1., 0., 1., 0.], [0., 1., 0., 1.], [1., 1., 0., 0.]]
        down = [[1., 0.], [0., 1.], [.5, 0.], [0., .5]]
        up_k = [[[1., 0.], [0., 1.]], [[1., .5], [-.5, 1.]]]
        up_v = [[[1., 2.], [0., 1.]], [[-1., 0.], [1., 2.]]]
        query_content = [[[1., .2], [.3, 1.], [.5, -.5]],
                         [[.1, 1.], [1., -.2], [.4, .8]]]

        def row_times_matrix(row, matrix):
            return [sum(row[j] * matrix[j][i] for j in range(len(row)))
                    for i in range(len(matrix[0]))]

        def manual_rotate(pair, position):
            # Explicit positions 0, 1, 2 radians, with the fixture frequency 1.
            cosine, sine = math.cos(position), math.sin(position)
            return [pair[0] * cosine - pair[1] * sine,
                    pair[0] * sine + pair[1] * cosine]

        latent = [row_times_matrix(row, down) for row in x]
        key_rope = [manual_rotate(x[t][:2], t) for t in (0, 1, 2)]
        scores, outputs, positional_scores = [], [], []
        for head in range(2):
            # Construct literal four-dimensional Q/K, instead of splitting
            # and adding the two dot products as both demo paths do.
            queries = [query_content[head][t] + manual_rotate(
                [value * .5 for value in query_content[head][t]], t)
                for t in (0, 1, 2)]
            keys = [row_times_matrix(latent[t], up_k[head]) + key_rope[t]
                    for t in (0, 1, 2)]
            values = [row_times_matrix(row, up_v[head]) for row in latent]
            head_scores = [[sum(query[j] * key[j] for j in range(4)) / 2.
                            for key in keys] for query in queries]
            head_positions = [[sum(query[j] * key[j] for j in (2, 3))
                               for key in keys] for query in queries]
            head_outputs = []
            for position in (0, 1, 2):
                # Causality by selecting only the visible prefix, not by
                # reusing the demo's mask or any attention implementation.
                visible = head_scores[position][:position + 1]
                exponentials = [math.exp(score - max(visible)) for score in visible]
                normalizer = sum(exponentials)
                probabilities = [value / normalizer for value in exponentials]
                head_outputs.append([sum(probabilities[t] * values[t][feature]
                                         for t in range(position + 1))
                                     for feature in range(2)])
            scores.append(head_scores)
            outputs.append(head_outputs)
            positional_scores.append(head_positions)
        return {name: torch.tensor(value, dtype=torch.float64) for name, value in
                {'latent': latent, 'k_rope': key_rope, 'scores': scores,
                 'output': outputs, 'positional_scores': positional_scores}.items()}

    def assert_mla_matches_independent_reference(self, result):
        expected = self.mla_independent_reference()
        for name in ('latent', 'k_rope', 'positional_scores'):
            torch.testing.assert_close(result[name], expected[name], atol=1e-12, rtol=1e-12)
        for name in ('full_scores', 'latent_scores'):
            torch.testing.assert_close(result[name], expected['scores'], atol=1e-12, rtol=1e-12)
        for name in ('full_output', 'latent_output'):
            torch.testing.assert_close(result[name], expected['output'], atol=1e-12, rtol=1e-12)

    def test_mla_absorption_position_payload_and_rank_loss(self):
        result = g.mla_matrix_demo()
        self.assert_mla_matches_independent_reference(result)
        self.assertEqual(result['compact_bytes'], 96)
        self.assertEqual(result['expanded_bytes'], 240)
        self.assertEqual((result['rank_target'], result['rank_compressed'], result['rank_loss']), (3, 2, 1.))
        self.assertLess(result['score_error'], 1e-12)
        self.assertLess(result['output_error'], 1e-12)

    def test_mla_shared_wrong_scale_omitted_position_and_unrotated_key_mutants_fail(self):
        changes = [
            ('scale = math.sqrt(q_content.shape[-1] + q_rope.shape[-1])',
             'scale = math.sqrt(c.shape[-1])', 1, 'mla_scale'),
            (' + positional_scores) / scale', ') / scale', 2, 'mla_position'),
            ('k_rope = rotate2(x[:, :2])', 'k_rope = x[:, :2]', 1, 'mla_key_rotation'),
        ]
        for old, new, occurrences, name in changes:
            with self.subTest(mutation=name):
                self.assertEqual(SCRIPT.read_text().count(old), occurrences)
                module = self.mutant(old, new, name)
                mutated = module.mla_matrix_demo()
                # All three bugs still pass the old two-self-path comparison.
                self.assertLess(mutated['score_error'], 1e-12)
                self.assertLess(mutated['output_error'], 1e-12)
                # Execute the real oracle assertion: require a test FAILURE,
                # never count a broken mutant/import/runtime ERROR as detection.
                probe = unittest.FunctionTestCase(
                    lambda: self.assert_mla_matches_independent_reference(mutated))
                report = unittest.TestResult()
                probe.run(report)
                self.assertEqual(report.testsRun, 1)
                self.assertEqual(len(report.errors), 0, report.errors)
                self.assertEqual(len(report.failures), 1, report.failures)

    def mutant(self, old, new, suffix):
        source = SCRIPT.read_text()
        self.assertIn(old, source)
        name = 'grouped_mutant_' + suffix
        module = types.ModuleType(name)
        module.__file__ = str(SCRIPT)
        sys.modules[name] = module
        self.addCleanup(sys.modules.pop, name, None)
        exec(compile(source.replace(old, new), str(SCRIPT), 'exec'), module.__dict__)
        return module

    def test_wrong_repeat_mapping_is_caught_by_hand_reference(self):
        mutant = self.mutant('v.repeat_interleave(repeats, dim=1)', 'v.repeat(1, repeats, 1, 1)', 'mapping')
        self.assertNotEqual(mutant.mapping_walkthrough().tolist(), g.mapping_walkthrough().tolist())

    def test_wrong_position_offset_and_rectangular_mask_are_detected(self):
        changes = [('positions = torch.arange(offset, offset + T, device=x.device)',
                    'positions = torch.arange(T, device=x.device)'),
                   ('allowed = key_positions <= query_positions',
                    'allowed = key_positions <= torch.arange(C, device=q.device)[:, None]')]
        for number, (old, new) in enumerate(changes):
            mutant = self.mutant(old, new, str(number))
            model = mutant.GroupedGPT(mutant.GroupedConfig(vocab_size=17, d_model=16,
                                      num_heads=4, num_layers=2, max_seq_len=16, num_kv_heads=2)).eval()
            actual, _ = self.chunks(model, self.tokens, [3, 4, 5], module=mutant)
            self.assertGreater((actual - model(self.tokens)[0]).abs().max().item(), 1e-5)

    def test_expanded_cache_mutation_fails_memory_contract(self):
        mutant = self.mutant('layers.append((k, v))',
                            'layers.append((k.repeat_interleave(2, 1), v.repeat_interleave(2, 1)))', 'cache')
        model = mutant.GroupedGPT(mutant.GroupedConfig(vocab_size=17)).eval()
        _, cache = mutant.cached_forward(model, self.tokens)
        self.assertNotEqual(mutant.cache_bytes(cache), mutant.expected_cache_bytes(model.config, 2, 12))
        self.assertNotEqual(cache.layers[0][0].shape[1], model.config.num_kv_heads)


if __name__ == '__main__':
    unittest.main()
