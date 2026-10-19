"""P12: numeric rotation reference, causal/full-cache equivalence, failure demo."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
from torch_test_support import torch_status, torch_reason

READY, VERSION, PRESENT = torch_status()
if READY:
    import torch
    spec = importlib.util.spec_from_file_location('rope_test', Path(__file__).resolve().parents[1] / 'code/13_rope.py')
    rope = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = rope
    spec.loader.exec_module(rope)


class RopeCLITests(unittest.TestCase):
    def test_missing_torch_cli_is_concise_and_import_still_raises(self):
        script = Path(__file__).resolve().parents[1] / 'code/13_rope.py'
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
            "runpy.run_path(sys.argv[1], run_name='__main__')", str(script)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('python -m pip install -r requirements.txt', result.stderr)
        self.assertNotIn('Traceback', result.stderr)
        imported = subprocess.run([sys.executable, '-c', blocker + """
try:
    runpy.run_path(sys.argv[1], run_name='rope_without_torch')
except ModuleNotFoundError as exc:
    assert exc.name == 'torch'
else:
    raise AssertionError('import should report missing dependency')
""", str(script)], capture_output=True, text=True)
        self.assertEqual(imported.returncode, 0, imported.stderr)


@unittest.skipUnless(READY, torch_reason('P12', VERSION, PRESENT))
class RopeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(47)
        self.model = rope.RopeGPT(rope.RopeConfig(17, 16, 4, 2, 32)).eval()
        self.tokens = torch.randint(17, (2, 12))

    def test_hand_rotation_and_relative_phase_table(self):
        rotated, norms, rows = rope.rotation_walkthrough()
        torch.testing.assert_close(rotated, torch.tensor([[3., 4.], [-4., 3.], [-3., -4.]], dtype=torch.float64), atol=1e-14, rtol=0)
        torch.testing.assert_close(norms, torch.full((3,), 5., dtype=torch.float64), atol=1e-14, rtol=0)
        self.assertAlmostEqual(rows[0][2], 7.62626733416533, places=12)
        self.assertAlmostEqual(rows[1][2], rows[0][2], places=12)
        self.assertAlmostEqual(rows[2][2], -2.7590203483672, places=12)

    def test_rotation_matches_independent_float64_matrix_and_preserves_tail(self):
        # Explicit 2x2 matrices catch pair/sign/frequency mistakes even if both
        # full and cache paths happen to share the same wrong apply_rope.
        x = torch.randn(2, 3, 5, 10, dtype=torch.float64)
        positions = torch.tensor([0., 1., 100., 1000001., 1000002.], dtype=torch.float64)
        actual = rope.apply_rope(x, positions, base=100., rotary_dim=6)
        expected = x.clone()
        for t, p in enumerate(positions.tolist()):
            for i in range(3):
                angle = p * 100. ** (-2 * i / 6)
                c, s = rope.math.cos(angle), rope.math.sin(angle)
                matrix = torch.tensor([[c, -s], [s, c]], dtype=torch.float64)
                expected[:, :, t, 2*i:2*i+2] = x[:, :, t, 2*i:2*i+2] @ matrix.T
        torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-13)
        torch.testing.assert_close(actual[..., 6:], x[..., 6:], atol=0, rtol=0)
        torch.testing.assert_close(actual.norm(dim=-1), x.norm(dim=-1), atol=1e-13, rtol=1e-13)
        self.assertEqual(actual.dtype, torch.float64)

    def test_frequency_interpolation_tables_and_norm_runtime_cache_guard(self):
        rows = rope.frequency_walkthrough()
        self.assertEqual([r[1] for r in rows], [1., .1, .01, .001])
        for i, theta, angle, wavelength in rows:
            self.assertAlmostEqual(angle,10*theta)
            self.assertAlmostEqual(wavelength*theta,2*rope.math.pi)
        self.assertEqual(rope.interpolation_walkthrough(), [(0,0.,0.),(7,7.,1.75),(8,8.,2.),(31,31.,7.75)])
        _, cache = rope.cached_forward(self.model, self.tokens[:, :3])
        self.model.blocks[0].ln1.eps *= 10
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            rope.cached_forward(self.model,self.tokens[:,3:4],cache)

    def test_common_position_shift_and_relative_rotation_identity(self):
        q, k = torch.randn(2, 4, 5, 8, dtype=torch.float64), torch.randn(2, 4, 5, 8, dtype=torch.float64)
        m = torch.tensor([0., 1., 3., 7., 11.], dtype=torch.float64)
        n = m + torch.tensor([1., 2., 5., 1., 4.], dtype=torch.float64)
        dot = lambda a, b: (a * b).sum(-1)
        expected = dot(q, rope.apply_rope(k, n-m))
        actual = dot(rope.apply_rope(q, m), rope.apply_rope(k, n))
        shifted = dot(rope.apply_rope(q, m+1341), rope.apply_rope(k, n+1341))
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(shifted, expected, atol=1e-12, rtol=1e-12)
        self.assertFalse(torch.allclose(actual, dot(q, k)))

    def test_half_dtype_and_rotation_gradients(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            x = torch.randn(2, 5, 6, dtype=dtype, requires_grad=True)
            out = rope.apply_rope(x, torch.arange(5), rotary_dim=4)
            self.assertEqual(out.dtype, dtype)
            self.assertTrue(torch.isfinite(out).all())
            (out.float().square().mean()).backward()
            self.assertTrue(torch.isfinite(x.grad).all())

    def compare_chunks(self, sizes, dtype=None):
        dtype = torch.float32 if dtype is None else dtype
        self.model.to(dtype)
        with torch.no_grad():
            expected = self.model(self.tokens)[0]
            cache, outputs, start = None, [], 0
            for size in sizes:
                logits, cache = rope.cached_forward(self.model, self.tokens[:, start:start+size], cache)
                outputs.append(logits)
                start += size
            self.assertEqual(start, self.tokens.shape[1])
            tol = 1e-12 if dtype == torch.float64 else 3e-6
            torch.testing.assert_close(torch.cat(outputs, 1), expected, atol=tol, rtol=tol)
            return cache

    def test_full_token_chunk_and_float64_cache_equivalence(self):
        for dtype in (torch.float32, torch.float64):
            for sizes in ([1]*12, [3, 4, 2, 3], [12], [11, 1]):
                with self.subTest(dtype=dtype, sizes=sizes):
                    self.compare_chunks(sizes, dtype)
        logits, cache = rope.cached_forward(self.model, self.tokens[:, :1])
        torch.testing.assert_close(logits, self.model(self.tokens[:, :1])[0])
        self.assertEqual(cache.length, 1)

    def test_rotated_key_unrotated_value_cache_and_memory(self):
        for dtype in (torch.float32, torch.float64):
            self.model.to(dtype)
            _, cache = rope.cached_forward(self.model, self.tokens)
            attn = self.model.blocks[0].attn
            x = self.model.blocks[0].ln1(self.model.tok_emb(self.tokens))
            k = attn.W_k(x).view(2, 12, 4, 4).transpose(1, 2)
            v = attn.W_v(x).view(2, 12, 4, 4).transpose(1, 2)
            torch.testing.assert_close(cache.layers[0][0], rope.apply_rope(k, torch.arange(12)))
            torch.testing.assert_close(cache.layers[0][1], v)
            self.assertEqual(rope.cache_bytes(cache), rope.expected_cache_bytes(self.model.config, 2, 12, dtype))
        self.assertEqual(rope.expected_cache_bytes(self.model.config, 2, 12), 6144)

    def test_future_change_does_not_change_full_or_chunk_past(self):
        changed = self.tokens.clone()
        changed[:, 7:] = (changed[:, 7:] + 1) % 17
        torch.testing.assert_close(self.model(self.tokens)[0][:, :7], self.model(changed)[0][:, :7])
        _, prefix = rope.cached_forward(self.model, self.tokens[:, :3])
        a, _ = rope.cached_forward(self.model, self.tokens[:, 3:], prefix)
        b, _ = rope.cached_forward(self.model, changed[:, 3:], prefix)
        torch.testing.assert_close(a[:, :4], b[:, :4])
        self.assertGreater((a[:, 4:] - b[:, 4:]).abs().max().item(), 1e-4)

    def test_cache_branch_reset_and_model_weight_position_changes(self):
        _, prefix = rope.cached_forward(self.model, self.tokens[:, :3])
        snapshots = tuple((k.clone(), v.clone()) for k, v in prefix.layers)
        logits, extended = rope.cached_forward(self.model, self.tokens[:, 3:5], prefix)
        self.assertEqual(prefix.length, 3)
        self.assertEqual(extended.length, 5)
        for pair, original in zip(prefix.layers, snapshots):
            for actual, expected in zip(pair, original):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        fresh, _ = rope.cached_forward(self.model, self.tokens.flip(1))
        torch.testing.assert_close(fresh, self.model(self.tokens.flip(1))[0])
        other = rope.RopeGPT(self.model.config).eval()
        with self.assertRaisesRegex(ValueError, 'different model'):
            rope.cached_forward(other, self.tokens[:, :1], prefix)
        with torch.no_grad():
            self.model.head.weight.add_(.1)
        with self.assertRaisesRegex(ValueError, 'weights or configuration changed'):
            rope.cached_forward(self.model, self.tokens[:, :1], prefix)
        _, prefix = rope.cached_forward(self.model, self.tokens[:, :3])
        self.model.blocks[0].attn.rope_base = 100.
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            rope.cached_forward(self.model, self.tokens[:, :1], prefix)

    def test_parameter_replacement_and_dtype_change_reject_old_cache(self):
        for change in ('assign', 'double', 'rotary'):
            self.setUp()
            _, cache = rope.cached_forward(self.model, self.tokens[:, :3])
            if change == 'assign':
                self.model.load_state_dict(rope.RopeGPT(self.model.config).state_dict(), assign=True)
            elif change == 'double':
                self.model.double()
            else:
                self.model.blocks[0].attn.rotary_dim = 2
            with self.assertRaisesRegex(ValueError, 'configuration changed'):
                rope.cached_forward(self.model, self.tokens[:, 3:5], cache)
            out, _ = rope.cached_forward(self.model, self.tokens)
            torch.testing.assert_close(out, self.model(self.tokens)[0])

    def test_mutations_reset_offset_and_skip_rotation_are_detected(self):
        reference = self.model(self.tokens)[0]
        original = rope.apply_rope
        _, cache = rope.cached_forward(self.model, self.tokens[:, :3])
        # Wrong only on new chunks: resetting phase cannot be caught by norms.
        with patch.object(rope, 'apply_rope', side_effect=lambda x, p, *args: original(x, p-p[0], *args)):
            broken, _ = rope.cached_forward(self.model, self.tokens[:, 3:], cache)
        self.assertGreater((broken - reference[:, 3:]).abs().max().item(), 1e-4)
        with patch.object(rope, 'apply_rope', side_effect=lambda x, *args: x):
            broken, _ = rope.cached_forward(self.model, self.tokens)
        self.assertGreater((broken-reference).abs().max().item(), 1e-4)
        # Re-rotating already rotated history is also a genuine observable bug.
        twice = tuple((rope.apply_rope(k, torch.arange(cache.length)), v) for k, v in cache.layers)
        broken_cache = rope.KVCache(twice, self.model, cache.versions)
        broken, _ = rope.cached_forward(self.model, self.tokens[:, 3:], broken_cache)
        self.assertGreater((broken-reference[:, 3:]).abs().max().item(), 1e-4)

    def test_shapes_training_and_no_absolute_embeddings(self):
        self.assertFalse(hasattr(self.model, 'pos_emb'))
        for block in self.model.blocks:
            self.assertIsInstance(block.ln1, torch.nn.LayerNorm)
            self.assertIsInstance(block.mlp[1], torch.nn.GELU)
        logits, loss = self.model(self.tokens[:, :-1], self.tokens[:, 1:])
        self.assertEqual(logits.shape, (2, 11, 17))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.parameters()))
        self.model.train()
        with self.assertRaisesRegex(ValueError, 'model.eval'):
            rope.cached_forward(self.model, self.tokens)

    def test_validation_and_malformed_cache(self):
        for kwargs in ({'d_model':15}, {'rotary_dim':3}, {'rotary_dim':6}, {'rope_base':0}, {'rope_base':float('inf')}, {'num_layers':True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                rope.RopeConfig(17, d_model=kwargs.get('d_model', 16), **{k:v for k,v in kwargs.items() if k!='d_model'})
        for x, pos, kwargs in ((torch.ones(3,4), torch.arange(2), {}), (torch.ones(3,4), torch.tensor([-1.,0.,1.]), {}),
                              (torch.ones(3,4), torch.tensor([0.,1.,float('nan')]), {}),
                              (torch.ones(3,4), torch.arange(3), {'rotary_dim':3})):
            with self.assertRaises(ValueError):
                rope.apply_rope(x, pos, **kwargs)
        for tokens in (self.tokens[:, :0], self.tokens.float(), torch.full((1,2),17), torch.zeros(1,33,dtype=torch.long)):
            with self.assertRaises(ValueError):
                rope.cached_forward(self.model, tokens)
        _, cache = rope.cached_forward(self.model, self.tokens[:, :3])
        k, v = cache.layers[0]
        for layers, reason in (((), 'layer count'), (((k,), cache.layers[1]), 'K/V pair'),
                               (((k[0],v),cache.layers[1]), 'shape'),
                               (((k[:,:2],v[:,:2]),cache.layers[1]), 'shape'),
                               (((k.double(),v.double()),cache.layers[1]), 'dtype/device')):
            with self.assertRaisesRegex(ValueError, reason):
                rope.cached_forward(self.model, self.tokens[:, 3:4], rope.KVCache(layers,self.model,cache.versions))
        _, full = rope.cached_forward(self.model, torch.zeros(2,32,dtype=torch.long))
        with self.assertRaisesRegex(ValueError, 'exceeds max_seq_len'):
            rope.cached_forward(self.model, self.tokens[:, :1], full)

    def test_partial_rotation_model_cache_equivalence(self):
        self.model = rope.RopeGPT(rope.RopeConfig(17, 24, 4, 2, 32, rotary_dim=4)).eval()
        self.compare_chunks([2, 3, 7])

    def test_retrieval_dataset_and_short_reproducibility(self):
        for length in (8,16,32,64):
            x, target = rope.retrieval_batch(torch.Generator().manual_seed(100),128,length)
            self.assertTrue((x[:,0] == 9).all() and (x[:,-1] == 8).all())
            self.assertTrue(torch.equal(x[:,1],target))
            self.assertTrue((x[:,2:-1] < 8).all())
            self.assertEqual(set(target.tolist()),{0,1,2,3})
        before = torch.random.get_rng_state().clone()
        a = rope.length_experiment(steps=2, eval_lengths=(8,16), eval_batches=1)
        b = rope.length_experiment(steps=2, eval_lengths=(8,16), eval_batches=1)
        self.assertEqual(a,b)
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        for row in a['rows']:
            self.assertAlmostEqual(row['mean_ce']*row['count'],row['loss_sum'])
            self.assertAlmostEqual(row['accuracy']*row['count'],row['correct'])

    def test_default_seed_failure_is_reproducible_not_monotonic_claim(self):
        # Regression snapshot for the documented, pinned CPU setup. These loose
        # bounds distinguish the teaching result, not universal model quality.
        result = rope.length_experiment()
        rows = result['rows']
        self.assertGreater(rows[0]['accuracy'],.98)
        self.assertLess(rows[1]['accuracy'],.60)
        self.assertLess(rows[2]['accuracy'],.50)
        self.assertGreater(rows[3]['accuracy'],rows[2]['accuracy'])
        self.assertEqual([row['count'] for row in rows],[512]*4)
        self.assertLess(result['last_train_ce'],.01)
        self.assertAlmostEqual(result['first_train_ce'], 2.338030, delta=2e-5)
        self.assertAlmostEqual(result['last_train_ce'], .001039, delta=2e-5)
        for row, ce, correct in zip(rows, [.001046,3.318326,3.937189,1.160583], [512,226,170,364]):
            self.assertEqual(row['correct'], correct)
            self.assertAlmostEqual(row['mean_ce'], ce, delta=2e-5)
            self.assertAlmostEqual(row['accuracy'], correct/512, delta=1e-12)



if __name__ == '__main__':
    unittest.main()
