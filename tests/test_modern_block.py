"""P13: independent formulas, actual budgets, controlled changes, S2 cache reuse."""
import importlib.metadata
import importlib.util
from pathlib import Path
import runpy
import subprocess
import sys
import types
import unittest
from unittest.mock import patch
from torch_test_support import torch_status, torch_reason

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'code/14_modern_block.py'
READY, VERSION, PRESENT = torch_status()
if READY:
    import torch
    spec = importlib.util.spec_from_file_location('modern_block_test', CODE)
    modern = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = modern
    spec.loader.exec_module(modern)


class ModernDependencyTests(unittest.TestCase):
    def test_missing_torch_cli_is_concise_and_import_raises(self):
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
            "runpy.run_path(sys.argv[1], run_name='__main__')", str(CODE)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('python -m pip install -r requirements.txt', result.stderr)
        self.assertNotIn('Traceback', result.stderr)
        result = subprocess.run([sys.executable, '-c', blocker + """
try:
    runpy.run_path(sys.argv[1], run_name='modern_without_torch')
except ModuleNotFoundError as exc:
    assert exc.name == 'torch'
else:
    raise AssertionError('expected missing dependency')
""", str(CODE)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_incompatible_torch_skips_without_importing_lesson(self):
        for version in ('2.13.1', '2.15.0', '2.14.1rc1'):
            with self.subTest(version=version), patch('importlib.metadata.version', return_value=version):
                namespace = runpy.run_path(str(Path(__file__)))
            self.assertFalse(namespace['READY'])
            self.assertNotIn('modern', namespace)
            result = unittest.TestResult()
            namespace['ModernTests']('test_rms_reference_and_learned_gamma').run(result)
            self.assertEqual(len(result.skipped), 1)
            self.assertFalse(result.failures or result.errors)
            self.assertIn('skipped is not passed', result.skipped[0][1])

    def test_chapter_source_links_diagram_and_hand_values(self):
        body = (ROOT / 'chapters/08c_modern_block.md').read_text()
        self.assertIn('{#p13-modern-block}', body)
        self.assertIn('参考答案', body)
        self.assertIn('1910.07467', body)
        self.assertIn('2002.05202', body)
        for value in ('0.426401', '1.279204', '-0.806824', '7.046377', '18432', '61008'):
            self.assertIn(value, body)
        self.assertIn('assets/diagrams/p13-modern-block.svg', body)
        import xml.etree.ElementTree as ET
        root = ET.parse(ROOT / 'assets/diagrams/p13-modern-block.svg').getroot()
        self.assertTrue(root.attrib.get('viewBox'))
        self.assertTrue(root.findall('.//{http://www.w3.org/2000/svg}text'))


@unittest.skipUnless(READY, torch_reason('P13', VERSION, PRESENT))
class ModernTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def config(self, **kwargs):
        return modern.ModernConfig(vocab_size=19, d_model=24, num_heads=4,
                                   num_layers=2, max_seq_len=16, **kwargs)

    def test_rms_reference_and_learned_gamma(self):
        layer = modern.RMSNorm(3, eps=.25).double()
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([2., .5, -1.]))
        x = torch.tensor([[[1., 2., 3.], [2., 4., 6.]], [[-3., 4., 0.], [0., 0., 0.]]], dtype=torch.float64)
        # Independent explicit sums, not the implementation's mean/rsqrt expression.
        ref = x / torch.sqrt((x[..., 0:1]**2 + x[..., 1:2]**2 + x[..., 2:3]**2) / 3 + .25)
        ref = ref * torch.tensor([2., .5, -1.], dtype=torch.float64)
        torch.testing.assert_close(layer(x), ref, atol=1e-14, rtol=1e-14)
        builtin = torch.nn.RMSNorm(3, eps=.25).double()
        builtin.load_state_dict(layer.state_dict())
        torch.testing.assert_close(layer(x), builtin(x), atol=1e-14, rtol=1e-14)
        self.assertIsNone(getattr(layer, 'bias', None))
        self.assertEqual(modern.parameter_count(layer), 3)

    def test_norm_hand_table_shift_and_epsilon(self):
        rows = modern.norm_walkthrough()
        torch.testing.assert_close(rows['mean'], torch.tensor([2., 4.], dtype=torch.float64))
        torch.testing.assert_close(rows['variance'], torch.ones(2, dtype=torch.float64))
        torch.testing.assert_close(rows['mean_square'], torch.tensor([5., 17.], dtype=torch.float64))
        ln_ref = torch.tensor([[-1., 1.], [-1., 1.]], dtype=torch.float64) / (1.5 ** .5)
        rms_ref = rows['x'] / torch.tensor([[5.5 ** .5], [17.5 ** .5]], dtype=torch.float64)
        torch.testing.assert_close(rows['ln'], ln_ref)
        torch.testing.assert_close(rows['rms'], rms_ref)
        self.assertFalse(torch.allclose(rows['ln'], rows['rms']))
        self.assertFalse(torch.allclose(rows['rms'][0], rows['rms'][1]))
        wrong_outside = rows['x'] / (rows['mean_square'].sqrt()[:, None] + .5)
        self.assertGreater((wrong_outside - rows['rms']).abs().max().item(), .1)

    def test_rms_is_per_token_not_batch_or_sequence(self):
        layer = modern.RMSNorm(4).double()
        x = torch.randn(2, 3, 4, dtype=torch.float64)
        expected = layer(x).clone()
        x[1, 1] *= 50
        self.assertTrue(torch.equal(layer(x)[0], expected[0]))
        self.assertTrue(torch.equal(layer(x)[1, 0], expected[1, 0]))
        torch.testing.assert_close(layer(x.reshape(6, 4)).reshape(2, 3, 4), layer(x))

    def test_low_precision_accumulates_in_fp32_and_keeps_dtype(self):
        for dtype in (torch.float16, torch.bfloat16):
            layer = modern.RMSNorm(4, eps=1e-5).to(dtype)
            x = torch.tensor([[40000., -40000., 20000., -20000.],
                              [0., 0., 0., 0.]], dtype=dtype, requires_grad=True)
            output = layer(x)
            ref = (x.float() / (x.float().square().mean(-1, keepdim=True) + 1e-5).sqrt()).to(dtype)
            self.assertEqual(output.dtype, dtype)
            torch.testing.assert_close(output, ref, atol=0, rtol=0)
            self.assertTrue(torch.isfinite(output).all())
            (output.float() * torch.tensor([1., 2., 3., 4.])).sum().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertTrue(torch.isfinite(layer.weight.grad).all())
        # A half-precision square would overflow before normalization.
        self.assertFalse(torch.isfinite(torch.tensor([40000.], dtype=torch.float16).square()).all())

    def test_norm_and_gate_double_gradcheck(self):
        x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(modern.RMSNorm(3).double(), (x,)))
        self.assertTrue(torch.autograd.gradcheck(modern.SwiGLU(3, 4).double(), (x,)))

    def test_gate_hand_table_and_multiplication_not_addition(self):
        rows = modern.gate_walkthrough()
        expected_gate = torch.tensor([-1., 2.], dtype=torch.float64)
        expected_up = torch.tensor([3., 4.], dtype=torch.float64)
        expected = torch.tensor([-3 / (1 + __import__('math').e),
                                 8 / (1 + __import__('math').exp(-2))], dtype=torch.float64)
        torch.testing.assert_close(rows['gate'], expected_gate)
        torch.testing.assert_close(rows['up'], expected_up)
        torch.testing.assert_close(rows['output'], expected, atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(rows['output'], rows['product'])
        self.assertFalse(torch.allclose(rows['output'], rows['silu'] + rows['up']))
        self.assertFalse(torch.allclose(rows['output'], rows['sigmoid'] * rows['up']))
        self.assertLess(rows['silu'][0].item(), 0)
        self.assertGreater(rows['silu'][1].item(), 1)

    def test_swiglu_reference_and_all_parameters_have_finite_gradients(self):
        for bias in (False, True):
            layer = modern.SwiGLU(4, 7, bias=bias).double()
            x = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
            gate = x @ layer.gate.weight.T
            up = x @ layer.up.weight.T
            if bias:
                gate = gate + layer.gate.bias
                up = up + layer.up.bias
            ref = (gate / (1 + (-gate).exp()) * up) @ layer.down.weight.T
            if bias:
                ref = ref + layer.down.bias
            torch.testing.assert_close(layer(x), ref, atol=1e-14, rtol=1e-14)
            layer(x).square().sum().backward()
            for param in [x, *layer.parameters()]:
                self.assertIsNotNone(param.grad)
                self.assertTrue(torch.isfinite(param.grad).all())

    def test_parameter_formulas_and_closest_width(self):
        for d in (2, 16, 48):
            for bias in (False, True):
                width = modern.matched_swiglu_width(d, bias=bias)
                layer = modern.SwiGLU(d, width, bias)
                self.assertEqual(modern.parameter_count(layer), modern.ffn_parameter_count(d, width, 'swiglu', bias))
                gelu = torch.nn.Sequential(torch.nn.Linear(d, 4*d, bias=bias), torch.nn.GELU(),
                                           torch.nn.Linear(4*d, d, bias=bias))
                target = modern.parameter_count(gelu)
                self.assertEqual(target, modern.ffn_parameter_count(d, 4*d, 'gelu', bias))
                for alternative in range(1, 4*d+3):
                    self.assertLessEqual(abs(modern.parameter_count(layer)-target),
                        abs(modern.ffn_parameter_count(d, alternative, 'swiglu', bias)-target))
        self.assertEqual(modern.matched_swiglu_width(48), 128)
        self.assertEqual(modern.ffn_parameter_count(48, 128, 'swiglu'), 18432)
        self.assertEqual(modern.ffn_parameter_count(48, 192, 'swiglu'), 27648)
        self.assertEqual(modern.ffn_parameter_count(48, 192, 'gelu', True), 18672)
        self.assertEqual(modern.ffn_parameter_count(48, 128, 'swiglu', True), 18736)
        self.assertEqual(modern.matched_swiglu_width(16, multiple_of=8), 40)

    def test_width_ties_choose_smaller_with_exhaustive_independent_reference(self):
        # D=2, H=4, bias=True: GELU has 22 parameters, SwiGLU M=2/3
        # has 18/26. Both miss by 4, so the documented smaller-width tie wins.
        self.assertEqual(modern.matched_swiglu_width(2, gelu_width=4, bias=True), 2)
        for d in range(1, 9):
            for h in range(1, 13):
                for bias in (False, True):
                    for multiple in (1, 2, 4, 8):
                        target = 2*d*h + (h+d if bias else 0)
                        # Enumerate in ascending order using independent arithmetic.
                        # The last width is >=2H+multiple, past the positive optimum.
                        candidates = range(multiple, 2*h + 2*multiple, multiple)
                        error = lambda m: abs(3*d*m + (2*m+d if bias else 0) - target)
                        best_error = min(error(m) for m in candidates)
                        expected = next(m for m in candidates if error(m) == best_error)
                        with self.subTest(d=d, h=h, bias=bias, multiple=multiple):
                            self.assertEqual(modern.matched_swiglu_width(
                                d, gelu_width=h, bias=bias, multiple_of=multiple), expected)

    def test_invalid_config_and_module_contracts(self):
        for kwargs in ({'norm_kind':'batchnorm'}, {'ffn_kind':'relu'}, {'norm_eps':0},
                       {'norm_eps':float('nan')}, {'norm_eps':True}, {'intermediate_size':0},
                       {'intermediate_size':False}, {'norm_bias':True}, {'ffn_bias':1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.config(**kwargs)
        for bad in (0, -1, float('inf'), True):
            with self.assertRaises(ValueError):
                modern.RMSNorm(4, bad)
        for x in (torch.ones(2, 3), torch.ones(4, dtype=torch.long), torch.tensor(1.)):
            with self.assertRaises(ValueError):
                modern.RMSNorm(4)(x)
        with self.assertRaises(ValueError):
            modern.SwiGLU(4, 0)
        with self.assertRaises(ValueError):
            modern.ModernGPT(modern.s2.RopeConfig(19))
        with self.assertRaises(ValueError):
            modern.ModernConfig(19, d_model=15, num_heads=4)

    def test_compatibility_setting_is_exactly_s2_without_reinitializing(self):
        cfg = self.config(norm_kind='layernorm', ffn_kind='gelu', norm_bias=True, ffn_bias=True)
        torch.manual_seed(77)
        reference = modern.s2.RopeGPT(modern.s2.RopeConfig(19, 24, 4, 2, 16)).eval()
        torch.manual_seed(77)
        model = modern.ModernGPT(cfg).eval()
        self.assertEqual(reference.state_dict().keys(), model.state_dict().keys())
        for key, tensor in reference.state_dict().items():
            self.assertTrue(torch.equal(tensor, model.state_dict()[key]), key)
        ids = torch.randint(19, (2, 9))
        torch.testing.assert_close(model(ids)[0], reference(ids)[0], atol=0, rtol=0)
        self.assertIsInstance(reference.blocks[0].ln1, torch.nn.LayerNorm)
        self.assertIsInstance(reference.blocks[0].mlp[1], torch.nn.GELU)

    def test_all_configured_norm_types_and_nondefault_epsilon(self):
        models = modern.make_ablation_models(self.config(norm_eps=1e-6))
        expected_types = {
            'A_LN_GELU': torch.nn.LayerNorm,
            'B_RMS_GELU': modern.RMSNorm,
            'C_RMS_SwiGLU': modern.RMSNorm,
        }
        self.assertEqual(set(models), set(expected_types))
        for label, model in models.items():
            norms = [('ln_f', model.ln_f)]
            for i, block in enumerate(model.blocks):
                norms.extend(((f'blocks.{i}.ln1', block.ln1),
                              (f'blocks.{i}.ln2', block.ln2)))
            self.assertEqual(len(norms), 2*model.config.num_layers + 1)
            for name, norm in norms:
                with self.subTest(model=label, norm=name):
                    self.assertIs(type(norm), expected_types[label])
                    self.assertEqual(norm.eps, 1e-6)
                    self.assertIsNone(getattr(norm, 'bias', None))
                    self.assertEqual(tuple(norm.weight.shape), (model.config.d_model,))

    def test_norm_copy_helper_preserves_nontrivial_gamma_without_aliasing(self):
        # Initial gamma is all ones: deleting its copy is equivalent for that
        # initial state. Exercise the helper's actual contract with other values.
        source, target, _ = modern.make_ablation_models(self.config()).values()
        source_norms = [(name, module) for name, module in source.named_modules()
                        if isinstance(module, torch.nn.LayerNorm)]
        target_modules = dict(target.named_modules())
        self.assertEqual(len(source_norms), 2*source.config.num_layers + 1)
        with torch.no_grad():
            for i, (name, norm) in enumerate(source_norms):
                norm.weight.copy_(torch.linspace(-.7, 1.9, source.config.d_model) + i)
                target_modules[name].weight.fill_(-9)
        modern.copy_unchanged(source, target, 'norm')
        for name, norm in source_norms:
            copied = target_modules[name].weight
            with self.subTest(norm=name):
                torch.testing.assert_close(copied, norm.weight, atol=0, rtol=0)
                self.assertNotEqual(copied.data_ptr(), norm.weight.data_ptr())
        with torch.no_grad():
            source_norms[0][1].weight.add_(1)
        self.assertFalse(torch.equal(source_norms[0][1].weight,
                                     target_modules[source_norms[0][0]].weight))

    def test_ablation_changes_only_declared_modules_and_copies_common_weights(self):
        models = list(modern.make_ablation_models(self.config()).values())
        a, b, c = models
        self.assertEqual(a.config.ffn_width, 96)
        self.assertEqual(c.config.ffn_width, 64)
        self.assertEqual(modern.parameter_count(a), modern.parameter_count(b))
        self.assertEqual(modern.parameter_count(b), modern.parameter_count(c))
        for left, right, changed in ((a,b,'norm'), (b,c,'ffn')):
            excluded = modern._is_norm if changed == 'norm' else modern._is_ffn
            for name, tensor in left.state_dict().items():
                if not excluded(name):
                    self.assertTrue(torch.equal(tensor, right.state_dict()[name]), name)
        # Prove copying itself works, not merely same-seed constructors.
        with torch.no_grad():
            c.head.weight.add_(7)
            c.blocks[1].attn.W_k.weight.zero_()
            b.blocks[0].ln1.weight.fill_(2)
        self.assertFalse(torch.equal(b.head.weight, c.head.weight))
        modern.copy_unchanged(b, c, 'ffn')
        self.assertTrue(torch.equal(b.head.weight, c.head.weight))
        self.assertTrue(torch.equal(b.blocks[0].ln1.weight, c.blocks[0].ln1.weight))
        self.assertFalse(b.head.weight.data_ptr() == c.head.weight.data_ptr())
        with self.assertRaises(ValueError):
            modern.copy_unchanged(a, c, 'norm')

    def test_ablation_factory_preserves_caller_rng_and_is_reproducible(self):
        before = torch.random.get_rng_state().clone()
        first = modern.make_ablation_models(self.config(), seed=8)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        second = modern.make_ablation_models(self.config(), seed=8)
        for name in first:
            for key, value in first[name].state_dict().items():
                self.assertTrue(torch.equal(value, second[name].state_dict()[key]))

    def test_full_cached_logits_all_variants_batch_chunks_and_dtype(self):
        tokens = torch.randint(19, (2, 12))
        for model in modern.make_ablation_models(self.config()).values():
            for dtype in (torch.float32, torch.float64):
                model.to(dtype).eval()
                full = model(tokens)[0]
                for sizes in ([12], [1]*12, [3, 5, 4], [11,1]):
                    outputs, cache, offset = [], None, 0
                    for size in sizes:
                        out, cache = modern.cached_forward(model, tokens[:, offset:offset+size], cache)
                        outputs.append(out)
                        offset += size
                    torch.testing.assert_close(torch.cat(outputs,1), full,
                                               atol=3e-6 if dtype == torch.float32 else 1e-12, rtol=1e-5 if dtype == torch.float32 else 1e-12)
                    self.assertEqual(cache.length, 12)

    def test_cached_path_uses_modern_norm_and_ffn_not_stale_s2(self):
        model = modern.ModernGPT(self.config()).eval()
        tokens = torch.randint(19, (2, 9))
        with torch.no_grad():
            model.blocks[0].ln1.weight.copy_(torch.linspace(.5, 2, 24))
            model.blocks[1].mlp.gate.weight.mul_(3)
        full = model(tokens)[0]
        prefix, cache = modern.cached_forward(model, tokens[:,:4])
        suffix, _ = modern.cached_forward(model, tokens[:,4:], cache)
        torch.testing.assert_close(torch.cat((prefix,suffix),1), full, atol=3e-6, rtol=1e-5)
        with torch.no_grad():
            model.blocks[0].ln1.weight.add_(.1)
        with self.assertRaisesRegex(ValueError, 'changed'):
            modern.cached_forward(model, tokens[:,4:5], cache)

    def test_cache_rejects_norm_epsilon_and_gelu_configuration_changes(self):
        tokens = torch.randint(19, (2, 9))
        for target in ('ln1', 'ln2', 'ln_f'):
            model = modern.ModernGPT(self.config()).eval()
            _, cache = modern.cached_forward(model, tokens[:, :4])
            norm = model.ln_f if target == 'ln_f' else getattr(model.blocks[0], target)
            norm.eps = .5
            with self.assertRaisesRegex(ValueError, 'configuration changed'):
                modern.cached_forward(model, tokens[:, 4:], cache)
            # A fresh request after changing a numerical setting is still valid.
            fresh, _ = modern.cached_forward(model, tokens)
            torch.testing.assert_close(fresh, model(tokens)[0], atol=3e-6, rtol=1e-5)
        model = modern.ModernGPT(self.config(ffn_kind='gelu')).eval()
        _, cache = modern.cached_forward(model, tokens[:, :4])
        model.blocks[0].mlp[1].approximate = 'tanh'
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            modern.cached_forward(model, tokens[:, 4:], cache)

    def test_ablation_one_batch_same_optimizer_and_measured_finite_results(self):
        records = modern.run_ablation(self.config(), steps=2)
        repeat = modern.run_ablation(self.config(), steps=2)
        self.assertEqual(records, repeat)
        self.assertEqual(len(records), 3)
        for row in records:
            self.assertTrue(row['finite_backward'])
            self.assertLess(row['after'], row['before'])
            self.assertLess(row['cache_error'], 3e-6)

    def test_default_eight_step_ablation_matches_chapter_snapshot(self):
        expected = (
            ('A_LN_GELU', 'A：LN + GELU', 192, 4.429393, 2.889954),
            ('B_RMS_GELU', 'B：RMSNorm + GELU', 192, 4.441514, 2.896931),
            ('C_RMS_SwiGLU', 'C：RMSNorm + SwiGLU', 128, 4.403015, 3.212187),
        )
        records = modern.run_ablation(steps=8)
        self.assertEqual(len(records), len(expected))
        chapter = (ROOT / 'chapters/08c_modern_block.md').read_text()
        for record, (name, label, width, before, after) in zip(records, expected):
            with self.subTest(model=name):
                self.assertEqual(record['name'], name)
                self.assertEqual(record['width'], width)
                self.assertEqual(record['ffn_params'], 18432)
                self.assertEqual(record['params'], 61008)
                self.assertAlmostEqual(record['before'], before, delta=2e-5)
                self.assertAlmostEqual(record['after'], after, delta=2e-5)
                self.assertTrue(record['finite_backward'])
                # Cache rounding is hardware-dependent, not a decimal snapshot.
                self.assertLess(record['cache_error'], 3e-6)
                self.assertIn(f'| {label} | 18432 | 61008 | {before:.6f} | '
                              f'{after:.6f} | True |', chapter)

    def test_norm_budget_and_optimizer_contract_mutations_are_detected(self):
        mutations = (
            ('        self.ln_f = norm(self.ln_f)',
             '        # Mutant: omit final normalization replacement.',
             'test_all_configured_norm_types_and_nondefault_epsilon'),
            ('result = RMSNorm(config.d_model, eps=config.norm_eps)',
             'result = RMSNorm(config.d_model)',
             'test_all_configured_norm_types_and_nondefault_epsilon'),
            ('result = nn.LayerNorm(config.d_model, eps=config.norm_eps, bias=config.norm_bias)',
             'result = nn.LayerNorm(config.d_model, bias=config.norm_bias)',
             'test_all_configured_norm_types_and_nondefault_epsilon'),
            ('lr=1e-3,', 'lr=2e-3,',
             'test_default_eight_step_ablation_matches_chapter_snapshot'),
            ('d_model, width, "swiglu", bias) - target), width))',
             'd_model, width, "swiglu", bias) - target), -width))',
             'test_width_ties_choose_smaller_with_exhaustive_independent_reference'),
            ('                new[name].copy_(old[name])',
             '                pass  # Mutant: omit corresponding gamma copy.',
             'test_norm_copy_helper_preserves_nontrivial_gamma_without_aliasing'),
        )
        self.assert_mutations_fail(mutations)

    def test_wrong_rms_and_gate_mutations_are_detected(self):
        # Compile mutants only in memory; production files never change.
        mutations = (
            ('work.square().mean(dim=-1, keepdim=True) + self.eps',
             '(work - work.mean(dim=-1, keepdim=True)).square().mean(dim=-1, keepdim=True) + self.eps',
             'test_rms_reference_and_learned_gamma'),
            ('torch.rsqrt(work.square().mean(dim=-1, keepdim=True) + self.eps)',
             '1 / (work.square().mean(dim=-1, keepdim=True).sqrt() + self.eps)',
             'test_norm_hand_table_shift_and_epsilon'),
            ('self.down(F.silu(self.gate(x)) * self.up(x))',
             'self.down(torch.sigmoid(self.gate(x)) * self.up(x))',
             'test_gate_hand_table_and_multiplication_not_addition'),
            ('self.down(F.silu(self.gate(x)) * self.up(x))',
             'self.down(F.silu(self.gate(x)) + self.up(x))',
             'test_gate_hand_table_and_multiplication_not_addition'),
        )
        self.assert_mutations_fail(mutations)

    def assert_mutations_fail(self, mutations):
        source = CODE.read_text()
        for i, (old, new, test) in enumerate(mutations):
            self.assertEqual(source.count(old), 1, f'mutation target must be unique: {old}')
            mutant = types.ModuleType(f'p13_mutant_{i}')
            mutant.__file__ = str(CODE)
            sys.modules[mutant.__name__] = mutant
            try:
                exec(compile(source.replace(old, new, 1), str(CODE), 'exec'), mutant.__dict__)
                with patch.dict(globals(), modern=mutant):
                    outcome = unittest.TestResult()
                    ModernTests(test).run(outcome)
                self.assertTrue(outcome.failures, f'mutation {i} survived or only errored: {outcome.errors}')
            finally:
                sys.modules.pop(mutant.__name__, None)


if __name__ == '__main__':
    unittest.main()
