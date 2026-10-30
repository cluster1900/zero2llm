"""P27 integration contracts, literal/scalar oracles and actual behavior mutants.

Skipped Torch paths are explicitly NOT passing numerical evidence. The report is
not accepted merely because it prints PASS; tests exercise components and guards.
"""
from contextlib import contextmanager
from dataclasses import replace
import importlib.util
import io
import json
import math
import pickle
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from torch_test_support import torch_status, torch_reason

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'code/28_minigpt_healthcheck.py'
READY, VERSION, PRESENT = torch_status()
if READY:
    import torch
    spec = importlib.util.spec_from_file_location('p27_healthcheck_test', SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)


class InertCheckpointEnvelope(dict):
    """No methods or reducers: inert container used only to test the allowlist.

    Never encode functions, imports, shell commands, or other execution payloads.
    The unrestricted-load mutant merely reconstructs this ordinary dict subclass.
    """


@contextmanager
def gradient_fault(model, kind):
    """Change a real model's backward/forward behavior; restore every hook."""
    handles = []
    layer = None
    if kind == 'all_zero':
        handles = [parameter.register_hook(lambda grad: torch.zeros_like(grad))
                   for parameter in model.parameters()]
    elif kind == 'embedding_nan':
        handles = [model.tok_emb.weight.register_hook(lambda grad: torch.full_like(grad, float('nan')))]
    elif kind == 'head_detached':
        layer = model.head
    elif kind == 'side_detached':
        layer = model.mtp_heads[0]
    elif kind in ('side_zero', 'side_nan'):
        fill = 0. if kind == 'side_zero' else float('nan')
        handles = [model.mtp_heads[0].weight.register_hook(lambda grad: torch.full_like(grad, fill))]
    else:
        raise ValueError('unknown test gradient fault')
    try:
        if layer is None:
            yield
        else:
            # Values are unchanged. Only the selected parameter leaves autograd;
            # gradients still flow through inputs into the backbone.
            with patch.object(layer, 'forward', lambda inputs: torch.nn.functional.linear(
                    inputs, layer.weight.detach())):
                yield
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)


class HealthcheckCLITests(unittest.TestCase):
    def test_missing_torch_cli_is_not_a_pass_and_import_raises(self):
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
        self.assertIn('NOT RUN, not passed', result.stderr)
        self.assertNotIn('Traceback', result.stderr)
        self.assertNotIn('PASS:', result.stdout)
        imported = subprocess.run([sys.executable, '-c', blocker + """
try:
    runpy.run_path(sys.argv[1], run_name='no_torch_import')
except ModuleNotFoundError as exc:
    assert exc.name == 'torch'
else:
    raise AssertionError('dependency errors must propagate')
""", str(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(imported.returncode, 0, imported.stderr)

    def test_unrelated_import_failure_is_not_treated_as_missing_torch(self):
        code = """
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'torch':
        raise ModuleNotFoundError('broken transitive dependency', name='broken_dependency')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
runpy.run_path(sys.argv[1], run_name='__main__')
"""
        result = subprocess.run([sys.executable, '-c', code, str(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn('broken transitive dependency', result.stderr)
        self.assertNotIn('PASS:', result.stdout)

    @unittest.skipUnless(READY, torch_reason('P27 CLI', VERSION, PRESENT))
    def test_default_and_optional_json_cli(self):
        for options in ([], ['--mtp']):
            result = subprocess.run([sys.executable, str(SCRIPT), '--json', *options],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stderr, '')
            report = json.loads(result.stdout)
            self.assertEqual([x['stage'] for x in report['stages']],
                             ['S0', 'S1', 'S2', 'S3', 'S4', 'S5'] + (['S6'] if options else []))
            self.assertEqual(report['environment']['threads'], 1)
            self.assertEqual(report['environment']['device'], 'cpu')
            self.assertEqual(report['components']['GQA_mapping'], [0, 0, 1, 1])
            self.assertEqual(report['environment']['s6_initialization_seed'], 271 if options else None)
            for row in report['stages']:
                self.assertEqual([check['cut'] for check in row['causal_checks']], list(range(1, 8)))
                self.assertEqual(row['causal_error'], max(check['max_logit_error'] for check in row['causal_checks']))
                if row['stage'] == 'S0':
                    self.assertNotIn('chunk_checks', row)
                    continue
                checks = row['chunk_checks']
                self.assertEqual([check['sizes'] for check in checks],
                                 [[3, 1, 4], [1] * 8, [8], [7, 1]])
                self.assertTrue(all(math.isfinite(check['max_logit_error']) for check in checks))
                self.assertEqual(row['cache_error'], max(check['max_logit_error'] for check in checks))
                self.assertTrue(all(check['kv_bytes'] == row['kv_bytes'] for check in checks))
            self.assertEqual([x['name'] for x in report['negatives']],
                             ['wrong_mask', 'wrong_position', 'cache_pollution'])
            self.assertTrue(all(x['caught'] and x['max_logit_error'] > .1 for x in report['negatives']))
            for row in report['negatives']:
                self.assertTrue(math.isfinite(row['before_error']))
                self.assertLess(row['before_error'], row['max_logit_error'])
                self.assertLessEqual(row['before_error'], m.ATOL)
            if options:
                self.assertEqual(report['mtp']['label_counts'], [14, 12, 10])
                self.assertTrue(report['mtp']['greedy_equal'])
            else:
                self.assertIn('not requested', report['mtp']['status'])
            for value in report['timing']['milliseconds'].values():
                self.assertGreater(value['min'], 0)
                self.assertLessEqual(value['min'], value['median'])
                self.assertLessEqual(value['median'], value['max'])

    def test_reading_contract_and_chinese_svg(self):
        body = (ROOT / 'chapters/09a_minigpt_healthcheck.md').read_text()
        for text in ('{#p27-healthcheck}', '../code/28_minigpt_healthcheck.py', '自测', '模型卡',
                     '--mtp', '--json', '2026-10-20', '不是', '4096', '2048'):
            self.assertIn(text, body)
        self.assertNotIn('](../tests/', body)
        svg = ET.parse(ROOT / 'assets/diagrams/p27-healthcheck.svg')
        ns = {'s': 'http://www.w3.org/2000/svg'}
        for tag in ('title', 'desc'):
            self.assertRegex(''.join(''.join(e.itertext()) for e in svg.findall('.//s:' + tag, ns)),
                             r'[\u4e00-\u9fff]')
        self.assertGreaterEqual(len(svg.findall('.//s:text', ns)), 8)


@unittest.skipUnless(READY, torch_reason('P27', VERSION, PRESENT))
class HealthcheckTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.stages = m.make_stages(include_mtp=True)
        self.by = {s.name: s for s in self.stages}
        self.tokens, self.targets = m.fixture()

    def checkpoint(self, name='S2'):
        stage = self.by[name]
        out = io.BytesIO()
        m.save_checkpoint(out, name, stage.model, m.VOCAB)
        return out.getvalue()

    def test_stage_identity_and_no_duplicate_transformer(self):
        self.assertIs(self.by['S0'].model, self.by['S1'].model)
        self.assertIs(type(self.by['S0'].model), m.s0.BabyGPT)
        self.assertIs(type(self.by['S2'].model), m.s2.RopeGPT)
        self.assertIs(type(self.by['S5'].model), m.s5.MoEGPT)
        self.assertIs(type(self.by['S6'].model), m.s6.MTPGPT)
        self.assertEqual([m.s3.parameter_count(x.model) for x in self.stages],
                         [7264, 7264, 7008, 6800, 6288, 5456, 6000])

    def test_fixture_is_next_token_without_wraparound(self):
        self.assertEqual(self.tokens[0].tolist(), [1, 2, 3, 2, 6, 7, 4, 8])
        self.assertEqual(self.targets[0].tolist(), [2, 3, 2, 6, 7, 4, 8, 9])
        self.assertEqual(self.tokens[1].tolist(), [4, 0, 2, 1, 4, 3, 8, 6])
        self.assertEqual(self.targets[1, -1].item(), 5)

    def test_all_stages_have_finite_loss_and_backward(self):
        for stage in self.stages:
            with self.subTest(stage=stage.name):
                loss = m.finite_backward(stage.model, self.tokens, self.targets)
                self.assertTrue(math.isfinite(loss))
                self.assertGreater(loss, 0)

    def test_future_interventions_every_cut_float64(self):
        for stage in self.stages:
            stage.model.double()
            for cut in range(1, self.tokens.shape[1]):
                changed = self.tokens.clone()
                changed[:, cut:] = (changed[:, cut:] + 3) % 17
                original = stage.model(self.tokens)[0][:, :cut]
                actual = stage.model(changed)[0][:, :cut]
                # Dense identical arithmetic, MoE may change GEMM row grouping.
                m.check_close(actual, original, atol=1e-12 if stage.name in ('S5', 'S6') else 0, rtol=0)

    def test_same_weight_cache_chunktopologies_and_literal_payloads(self):
        for stage in self.stages[1:]:
            for sizes in ((1,) * 8, (3, 1, 4), (2, 5, 1), (8,)):
                with self.subTest(stage=stage.name, sizes=sizes):
                    error, payload = m.cache_equivalence(stage, self.tokens, sizes)
                    self.assertLess(error, 3e-6)
                    self.assertEqual(payload, 4096 if stage.name in ('S1', 'S2', 'S3') else 2048)

    def test_chunk_sizes_and_nonfinite_or_shape_errors_do_not_pass(self):
        for sizes in ((3, 1), (0, 8), (-1, 9), (True, 7)):
            with self.assertRaisesRegex(ValueError, 'positive chunks'):
                m.cache_equivalence(self.by['S1'], self.tokens, sizes)
        for actual in (torch.zeros(1), torch.tensor([float('nan'), 0]), torch.tensor([float('inf'), 0])):
            with self.assertRaisesRegex(AssertionError, 'shape/nonfinite'):
                m.check_close(actual, torch.zeros(2))

    def test_checkpoint_roundtrip_every_stage_both_dtypes(self):
        for stage in self.stages:
            for dtype in (torch.float32, torch.float64):
                with self.subTest(stage=stage.name, dtype=dtype):
                    stage.model.to(dtype)
                    self.assertEqual(m.checkpoint_error(stage, self.tokens), 0)

    def test_compatibility_checks_reject_semantics_not_just_shapes(self):
        result = m.compatibility_checks(self.stages)
        self.assertEqual(result['rejected'], ['S0_to_S2', 'vocabulary_order', 'rope_base_same_shapes',
                                             'GQA_head_count', 'S4_to_S5', 'MTP_depth'])
        # norm epsilon changes no tensor shape, but changes computation.
        data = self.checkpoint('S3')
        with self.assertRaisesRegex(ValueError, 'architecture/config/vocabulary/depth'):
            m.load_checkpoint(io.BytesIO(data), 'S3', replace(self.by['S3'].model.config, norm_eps=.01), m.VOCAB)

    def test_checkpoint_strict_state_and_finite_dtype_validation(self):
        original = torch.load(io.BytesIO(self.checkpoint()), weights_only=True)
        config = self.by['S2'].model.config
        for kind in ('missing', 'unexpected', 'shape', 'nonfinite', 'dtype'):
            with self.subTest(kind=kind):
                state = {key: value.clone() for key, value in original['state_dict'].items()}
                key = 'tok_emb.weight'
                if kind == 'missing': del state[key]
                if kind == 'unexpected': state['not_a_weight'] = torch.zeros(1)
                if kind == 'shape': state[key] = state[key][:1]
                if kind == 'nonfinite': state[key][0, 0] = float('nan')
                if kind == 'dtype': state[key] = state[key].double()
                buffer = io.BytesIO()
                torch.save({'metadata': original['metadata'], 'state_dict': state}, buffer)
                buffer.seek(0)
                with self.assertRaises((RuntimeError, ValueError)):
                    m.load_checkpoint(buffer, 'S2', config, m.VOCAB)

    def test_checkpoint_rejects_integer_bool_complex_weights_and_buffer_dtypes(self):
        for name, key in (('S2', 'tok_emb.weight'), ('S0', 'blocks.0.attn.mask'),
                          ('S5', 'blocks.0.mlp.balance_bias')):
            for dtype in (torch.int64, torch.bool, torch.complex64, torch.float64):
                with self.subTest(stage=name, key=key, dtype=dtype):
                    state = torch.load(io.BytesIO(self.checkpoint(name)), weights_only=True)
                    state['state_dict'][key] = state['state_dict'][key].to(dtype)
                    buffer = io.BytesIO()
                    torch.save(state, buffer)
                    buffer.seek(0)
                    with self.assertRaisesRegex(ValueError, 'tensor dtype mismatch'):
                        m.load_checkpoint(buffer, name, self.by[name].model.config, m.VOCAB)

    def test_mutant_checkpoint_dtype_guard_reinstates_silent_cast_bug(self):
        mutant = m.mutate_function(m.load_checkpoint,
            'key in expected_state and tensor.dtype != expected_state[key].dtype',
            'False')
        for name, key in (('S2', 'tok_emb.weight'), ('S0', 'blocks.0.attn.mask'),
                          ('S5', 'blocks.0.mlp.balance_bias')):
            for dtype in (torch.int64, torch.bool, torch.complex64):
                state = torch.load(io.BytesIO(self.checkpoint(name)), weights_only=True)
                state['state_dict'][key] = state['state_dict'][key].to(dtype)
                data = io.BytesIO()
                torch.save(state, data)
                def contract(loader):
                    try:
                        loader(io.BytesIO(data.getvalue()), name, self.by[name].model.config, m.VOCAB)
                    except ValueError as exc:
                        self.assertIn('tensor dtype mismatch', str(exc))
                    else:
                        raise AssertionError('silent dtype cast accepted')
                contract(m.load_checkpoint)
                with self.assertRaisesRegex(AssertionError, 'silent dtype cast accepted'):
                    contract(mutant)

    def test_checkpoint_rejects_wrong_model_and_vocabulary(self):
        for stage, model, vocabulary in [('S2', self.by['S3'].model, m.VOCAB),
                                        ('S2', self.by['S2'].model, ['same'] * 17),
                                        ('S2', self.by['S2'].model, list(range(17)))]:
            with self.assertRaises(ValueError):
                m.save_checkpoint(io.BytesIO(), stage, model, vocabulary)
        with self.assertRaisesRegex(ValueError, 'stage/config'):
            m.construct('S4', self.by['S2'].model.config)
        with self.assertRaisesRegex(ValueError, 'MTP depth'):
            m.construct('S0', self.by['S0'].model.config, 1)

    def test_cache_is_not_checkpoint_state_and_foreign_owner_rejected(self):
        for name in ('S1', 'S2', 'S4'):
            with self.subTest(stage=name):
                stage = self.by[name]
                _, cache = stage.cached_forward(stage.model, self.tokens[:, :3])
                loaded = m.load_checkpoint(io.BytesIO(self.checkpoint(name)), name, stage.model.config, m.VOCAB)
                with self.assertRaisesRegex(ValueError, 'different model/request'):
                    stage.cached_forward(loaded, self.tokens[:, 3:4], cache)
                fresh, _ = stage.cached_forward(loaded, self.tokens, None)
                m.check_close(fresh, loaded(self.tokens)[0])

    def test_ablation_copies_all_unaffected_tensors_and_records_budget(self):
        rows = m.controlled_ablation(self.tokens, self.targets)
        self.assertEqual([x['width'] for x in rows], [64, 64, 43])
        self.assertEqual([x['ffn_parameters'] for x in rows], [2048, 2048, 2064])
        self.assertEqual([x['parameters'] for x in rows], [6768, 6768, 6800])
        self.assertEqual([x['unchanged_tensors'] for x in rows], [0, 14, 15])

    def test_component_hand_oracles(self):
        self.assertEqual(m.component_oracles()['MoE_load'], [2, 2, 1, 1])
        # Fraction arithmetic in prose: expected payload ratio is exactly 1/2.
        self.assertEqual(2 * 2 * 2 * 8 * 2 * 4 * 4, 2048)
        self.assertEqual(2 * 16 * 64, 2048)
        self.assertEqual(3 * 16 * 43, 2064)

    def test_moe_zero_router_selects_lowest_ids_and_unused_experts(self):
        model = self.by['S5'].model
        with torch.no_grad():
            for block in model.blocks:
                block.mlp.router.weight.zero_()
        model.zero_grad(set_to_none=True)
        _, loss = model(self.tokens, self.targets)
        loss.backward()
        for block in model.blocks:
            info = block.mlp.last_info
            self.assertEqual(info.requested.tolist(), [16, 16, 0, 0])
            self.assertEqual(info.accepted.tolist(), [16, 16, 0, 0])
            self.assertEqual(info.weights.tolist(), [[.5, .5]] * 16)
            for expert in block.mlp.experts[2:]:
                self.assertTrue(all(p.grad is None for p in expert.parameters()))
            self.assertTrue(torch.isfinite(block.mlp.router.weight.grad).all())

    def test_s6_main_logits_equal_s5_and_auxiliary_heads_receive_gradients(self):
        m.check_close(self.by['S6'].model(self.tokens)[0], self.by['S5'].model(self.tokens)[0], atol=0, rtol=0)
        result = m.mtp_check(self.by['S6'].model, self.tokens)
        self.assertEqual(result['label_counts'], [14, 12, 10])
        self.assertTrue(result['greedy_equal'])
        self.assertGreater(result['proposed'], 0)
        self.assertGreater(result['target_tokens'], result['target_calls'])

    def test_s6_depth_zero_keeps_s5_state_and_ordinary_logits(self):
        zero = m.construct('S6', self.by['S5'].model.config, 0).eval()
        zero.load_state_dict(self.by['S5'].model.state_dict(), strict=True)
        self.assertEqual(set(zero.state_dict()), set(self.by['S5'].model.state_dict()))
        self.assertEqual(m.s3.parameter_count(zero), 5456)
        m.check_close(zero(self.tokens)[0], self.by['S5'].model(self.tokens)[0], atol=0, rtol=0)

    def test_three_negatives_have_actual_finite_wrong_outputs(self):
        result = m.negative_controls(self.by['S0'].model, self.tokens)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(row['caught'] and math.isfinite(row['max_logit_error']) and
                            row['max_logit_error'] > .1 for row in result))
        for row in result:
            self.assertTrue(math.isfinite(row['before_error']))
            self.assertLess(row['before_error'], row['max_logit_error'])
            self.assertLessEqual(row['before_error'], m.ATOL)

    def test_mutant_bad_causal_mask_is_detected(self):
        model = self.by['S0'].model
        with torch.no_grad():
            for block in model.blocks:
                block.attn.mask.fill_(1)  # Actual model behavior, not expected data.
        with self.assertRaises(AssertionError):
            m.causal_error(model, self.tokens)

    def test_mutant_wrong_cache_mask_and_position_are_detected(self):
        changes = [('allowed = key_positions <= query_positions',
                    'allowed = torch.ones(T, offset + T, dtype=torch.bool, device=x.device).tril()'),
                   ('pos = torch.arange(offset, offset + T, device=idx.device)',
                    'pos = torch.arange(T, device=idx.device)')]
        for old, new in changes:
            with self.subTest(change=old):
                mutant = m.mutate_function(m.s1.cached_forward, old, new)
                with self.assertRaises(AssertionError):
                    m.cache_equivalence(m.Stage('S1', self.by['S0'].model, mutant), self.tokens)

    def test_mutant_rope_rotation_sign_is_killed_by_literal_oracle(self):
        mutant = m.mutate_function(m.s2.apply_rope, 'even * c - odd * s', 'even * c + odd * s')
        with patch.object(m.s2, 'apply_rope', mutant):
            with self.assertRaises(AssertionError):
                m.component_oracles()

    def test_mutant_rmsnorm_uses_mean_instead_of_mean_square(self):
        mutant = m.mutate_function(m.s3.RMSNorm.forward, 'work.square().mean', 'work.mean')
        with patch.object(m.s3.RMSNorm, 'forward', mutant):
            with self.assertRaises(AssertionError):
                m.component_oracles()

    def test_mutant_swiglu_drops_gate_is_killed_by_literal_oracle(self):
        mutant = m.mutate_function(m.s3.SwiGLU.forward, 'F.silu(self.gate(x)) * self.up(x)', 'self.up(x)')
        with patch.object(m.s3.SwiGLU, 'forward', mutant):
            with self.assertRaises(AssertionError):
                m.component_oracles()

    def test_mutant_gqa_interleaves_wrong_group_mapping(self):
        mutant = m.mutate_function(m.s4.grouped_attention,
                                  'values = v.repeat_interleave(repeats, dim=1)',
                                  'values = v.repeat(1, repeats, 1, 1)')
        with patch.object(m.s4, 'grouped_attention', mutant):
            with self.assertRaises(AssertionError):
                m.component_oracles()

    def test_mutant_checkpoint_metadata_guard_is_killed(self):
        mutant = m.mutate_function(m.load_checkpoint, 'if meta != _metadata(stage, model, vocabulary):',
                                  'if False:')
        data = self.checkpoint('S2')
        # Same tensor shapes but a semantically different position base.
        other = replace(self.by['S2'].model.config, rope_base=20000.)
        def contract(loader):
            try:
                loader(io.BytesIO(data), 'S2', other, m.VOCAB)
            except ValueError as exc:
                self.assertIn('architecture/config/vocabulary/depth', str(exc))
            else:
                raise AssertionError('semantic mismatch accepted')
        contract(m.load_checkpoint)
        with self.assertRaisesRegex(AssertionError, 'semantic mismatch accepted'):
            contract(mutant)

    def test_mutant_ablation_changes_unrelated_tensor_is_detected(self):
        original = m.s3.make_ablation_models
        def corrupted(*args, **kwargs):
            models = original(*args, **kwargs)
            with torch.no_grad():
                models['B_RMS_GELU'].tok_emb.weight.add_(.1)
            return models
        with patch.object(m.s3, 'make_ablation_models', corrupted):
            with self.assertRaisesRegex(AssertionError, 'unrelated weights'):
                m.controlled_ablation(self.tokens, self.targets)

    def test_mutant_future_label_shift_is_killed_by_scalar_oracle(self):
        mutant = m.mutate_function(m.s6.future_labels, 'tokens[:, horizon:]', 'tokens[:, horizon - 1:-1]')
        with patch.object(m.s6, 'future_labels', mutant):
            with self.assertRaisesRegex(AssertionError, 'future labels misaligned'):
                m.mtp_check(self.by['S6'].model, self.tokens)

    def test_mutant_s6_greedy_result_is_killed_by_uncached_oracle(self):
        # A real returned token is changed; checker recomputes using uncached model.
        original = m.s6.greedy_decode
        def corrupted(*args, **kwargs):
            result = original(*args, **kwargs)
            tokens = result.tokens.clone()
            tokens[0, -1] = (tokens[0, -1] + 1) % 17
            return replace(result, tokens=tokens)
        with patch.object(m.s6, 'greedy_decode', corrupted):
            with self.assertRaisesRegex(AssertionError, 'independent full-prefix greedy'):
                m.mtp_check(self.by['S6'].model, self.tokens)

    # Ordinary contracts never construct test-side source mutants. They invoke
    # production entry points (including the CLI's normal negative controls).
    # External audit mutants must fail on behavior, not a self-injection anchor.
    def test_plain_behavior_checks_first_and_middle_chunks(self):
        stage = self.by['S1']
        sizes = (3, 1, 4)
        m.cache_equivalence(stage, self.tokens, sizes)
        for bad_chunk in (0, 1):
            with self.subTest(bad_chunk=bad_chunk):
                calls, bad_outputs = [], []
                def forward(model, chunk, cache=None):
                    output, new_cache = stage.cached_forward(model, chunk, cache)
                    call = len(calls)
                    calls.append(chunk.shape[1])
                    if call == bad_chunk:
                        correct = output.clone()
                        output = output.clone() + .25
                        self.assertTrue(torch.isfinite(output).all())
                        self.assertGreater(float((output - correct).abs().max()), .24)
                        bad_outputs.append(call)
                    return output, new_cache
                with self.assertRaises(AssertionError):
                    m.cache_equivalence(m.Stage('S1', stage.model, forward), self.tokens, sizes)
                self.assertEqual(calls, [3, 1, 4])
                self.assertEqual(bad_outputs, [bad_chunk])

    def test_plain_behavior_backward_zero_nan_and_detached_head(self):
        model = self.by['S0'].model
        self.assertTrue(math.isfinite(m.finite_backward(model, self.tokens, self.targets)))
        for kind in ('all_zero', 'embedding_nan', 'head_detached'):
            with self.subTest(fault=kind), gradient_fault(model, kind):
                model.zero_grad(set_to_none=True)
                _, loss = model(self.tokens, self.targets)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                if kind == 'all_zero':
                    gradients = [p.grad for p in model.parameters() if p.grad is not None]
                    self.assertTrue(gradients)
                    self.assertTrue(all(torch.isfinite(g).all() and not torch.count_nonzero(g) for g in gradients))
                elif kind == 'embedding_nan':
                    self.assertTrue(torch.isnan(model.tok_emb.weight.grad).all())
                    self.assertTrue(torch.isfinite(model.head.weight.grad).all())
                else:
                    self.assertIsNone(model.head.weight.grad)
                    self.assertGreater(float(model.tok_emb.weight.grad.abs().sum()), 0)
                expected = 'main head detached' if kind == 'head_detached' else 'missing/nonfinite/zero gradients'
                with self.assertRaisesRegex(AssertionError, expected):
                    m.finite_backward(model, self.tokens, self.targets)

    def test_plain_behavior_mtp_side_head_gradient_contract(self):
        model = self.by['S6'].model
        self.assertTrue(m.mtp_check(model, self.tokens)['greedy_equal'])
        for kind in ('side_zero', 'side_detached', 'side_nan'):
            with self.subTest(fault=kind), gradient_fault(model, kind):
                result = model.forward_mtp(self.tokens, weight=.2)
                self.assertTrue(torch.isfinite(result.loss))
                model.zero_grad(set_to_none=True)
                result.loss.backward()
                first = model.mtp_heads[0].weight.grad
                if kind == 'side_zero':
                    self.assertTrue(torch.isfinite(first).all())
                    self.assertEqual(torch.count_nonzero(first), 0)
                elif kind == 'side_detached':
                    self.assertIsNone(first)
                else:
                    self.assertTrue(torch.isnan(first).all())
                self.assertTrue(torch.isfinite(model.head.weight.grad).all())
                self.assertGreater(float(model.mtp_heads[1].weight.grad.abs().sum()), 0)
                with self.assertRaisesRegex(AssertionError, 'MTP heads must receive finite nonzero gradients'):
                    m.mtp_check(model, self.tokens)

    def test_plain_behavior_float64_payload_contract(self):
        for stage in self.stages[1:]:
            with self.subTest(stage=stage.name):
                float32_bytes = 4096 if stage.name in ('S1', 'S2', 'S3') else 2048
                self.assertEqual(m.cache_equivalence(stage, self.tokens)[1], float32_bytes)
                stage.model.double()
                _, cache = stage.cached_forward(stage.model, self.tokens)
                tensors = [t for pair in cache.layers for t in pair]
                self.assertTrue(all(t.dtype == torch.float64 and t.element_size() == 8 for t in tensors))
                self.assertEqual(sum(t.numel() for t in tensors), float32_bytes // 4)
                self.assertEqual(m.cache_equivalence(stage, self.tokens)[1], float32_bytes * 2)

    def test_plain_behavior_weights_only_inert_container_refusal(self):
        normal = self.checkpoint('S2')
        config = self.by['S2'].model.config
        m.load_checkpoint(io.BytesIO(normal), 'S2', config, m.VOCAB)
        raw = torch.load(io.BytesIO(normal), weights_only=True)
        data = io.BytesIO()
        torch.save(InertCheckpointEnvelope(raw), data)
        with self.assertRaises(pickle.UnpicklingError) as caught:
            m.load_checkpoint(io.BytesIO(data.getvalue()), 'S2', config, m.VOCAB)
        self.assertIn('Unsupported global', str(caught.exception))
        self.assertIn('InertCheckpointEnvelope', str(caught.exception))

    def test_plain_behavior_bfloat16_and_nan_save_refusal(self):
        for kind in ('bfloat16', 'nan'):
            with self.subTest(kind=kind):
                model = m.make_stages()[2].model
                m.save_checkpoint(io.BytesIO(), 'S2', model, m.VOCAB)
                if kind == 'bfloat16':
                    model.bfloat16()
                    self.assertEqual(model.tok_emb.weight.dtype, torch.bfloat16)
                    message = 'support float32/float64 only'
                else:
                    with torch.no_grad():
                        model.tok_emb.weight[0, 0] = float('nan')
                    self.assertTrue(torch.isnan(model.tok_emb.weight[0, 0]))
                    message = 'checkpoint tensors must be finite'
                destination = io.BytesIO()
                with self.assertRaisesRegex(ValueError, message):
                    m.save_checkpoint(destination, 'S2', model, m.VOCAB)
                self.assertEqual(destination.getvalue(), b'')

    def test_plain_behavior_extra_envelope_key_refusal(self):
        normal = self.checkpoint('S2')
        config = self.by['S2'].model.config
        m.load_checkpoint(io.BytesIO(normal), 'S2', config, m.VOCAB)
        raw = torch.load(io.BytesIO(normal), weights_only=True)
        raw['unexpected_resume_state'] = {'step': 123}
        self.assertEqual(set(raw), {'metadata', 'state_dict', 'unexpected_resume_state'})
        data = io.BytesIO()
        torch.save(raw, data)
        with self.assertRaisesRegex(ValueError, 'checkpoint envelope mismatch'):
            m.load_checkpoint(io.BytesIO(data.getvalue()), 'S2', config, m.VOCAB)

    def test_plain_behavior_s1_s2_owner_only_refusal(self):
        for name in ('S1', 'S2'):
            with self.subTest(stage=name):
                stage = self.by[name]
                _, cache = stage.cached_forward(stage.model, self.tokens[:, :3])
                correct, _ = stage.cached_forward(stage.model, self.tokens[:, 3:], cache)
                m.check_close(correct, stage.model(self.tokens)[0][:, 3:])
                other = m.construct(name, stage.model.config).eval()
                wrong_owner = replace(cache, owner=other)
                self.assertIsNot(wrong_owner.owner, stage.model)
                self.assertEqual(wrong_owner.versions, cache.versions)
                self.assertIs(wrong_owner.layers, cache.layers)
                with self.assertRaisesRegex(ValueError, 'different model/request'):
                    stage.cached_forward(stage.model, self.tokens[:, 3:], wrong_owner)

    def test_every_chunk_is_checked_and_last_only_source_mutant_is_killed(self):
        stage = self.by['S1']
        sizes = (3, 1, 4)
        reference = stage.model(self.tokens)[0].detach()
        m.cache_equivalence(stage, self.tokens, sizes)  # Normal implementation/input.
        for bad_chunk in (0, 1):
            def make_forward():
                calls = 0
                def forward(model, chunk, cache=None):
                    nonlocal calls
                    output, new_cache = stage.cached_forward(model, chunk, cache)
                    if calls == bad_chunk:
                        output = output.clone() + .25
                    calls += 1
                    return output, new_cache
                return forward
            with self.subTest(bad_chunk=bad_chunk):
                # Independent evidence: only the nominated earlier chunk changes;
                # K/V and the final output chunk remain numerically correct.
                forward, cache, pieces, start = make_forward(), None, [], 0
                for size in sizes:
                    output, cache = forward(stage.model, self.tokens[:, start:start + size], cache)
                    pieces.append(output)
                    start += size
                actual = torch.cat(pieces, 1)
                self.assertTrue(torch.isfinite(actual).all())
                first = sum(sizes[:bad_chunk])
                last = first + sizes[bad_chunk]
                self.assertGreater(float((actual[:, first:last] - reference[:, first:last]).abs().max()), .24)
                m.check_close(actual[:, -sizes[-1]:], reference[:, -sizes[-1]:])
                with self.assertRaises(AssertionError):
                    m.cache_equivalence(m.Stage('S1', stage.model, make_forward()), self.tokens, sizes)
                # Prove the ordinary behavioral assertion first; an external
                # source mutant must fail above, not just lose a source anchor.
                last_only = m.mutate_function(m.cache_equivalence,
                    'error = check_close(torch.cat(pieces, 1), expected)',
                    'error = check_close(pieces[-1], expected[:, -sizes[-1]:])')
                error, payload = last_only(m.Stage('S1', stage.model, make_forward()), self.tokens, sizes)
                self.assertLess(error, 3e-6)  # The faulty checker really misses it.
                self.assertEqual(payload, 4096)

    def test_finite_backward_gradient_faults_and_guard_source_mutants(self):
        model = self.by['S0'].model
        self.assertTrue(math.isfinite(m.finite_backward(model, self.tokens, self.targets)))
        for kind in ('all_zero', 'embedding_nan', 'head_detached'):
            with self.subTest(fault=kind), gradient_fault(model, kind):
                model.zero_grad(set_to_none=True)
                _, loss = model(self.tokens, self.targets)
                self.assertTrue(torch.isfinite(loss))  # Not a forward/loss NaN.
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                if kind == 'all_zero':
                    self.assertTrue(grads)
                    self.assertTrue(all(torch.isfinite(g).all() and not torch.count_nonzero(g) for g in grads))
                elif kind == 'embedding_nan':
                    self.assertTrue(torch.isnan(model.tok_emb.weight.grad).all())
                    self.assertTrue(torch.isfinite(model.head.weight.grad).all())
                else:
                    self.assertIsNone(model.head.weight.grad)
                    self.assertGreater(float(model.tok_emb.weight.grad.abs().sum()), 0)
                message = 'main head detached' if kind == 'head_detached' else 'missing/nonfinite/zero gradients'
                with self.assertRaisesRegex(AssertionError, message):
                    m.finite_backward(model, self.tokens, self.targets)
                old = ('if model.tok_emb.weight.grad is None or model.head.weight.grad is None:'
                       if kind == 'head_detached' else
                       'if not grads or not all(torch.isfinite(g).all() for g in grads) or not any(g.abs().sum() > 0 for g in grads):')
                mutant = m.mutate_function(m.finite_backward, old, 'if False:')
                escaped_loss = mutant(model, self.tokens, self.targets)
                self.assertTrue(math.isfinite(escaped_loss))
                self.assertAlmostEqual(escaped_loss, float(loss.detach()), places=6)

    def test_mtp_side_head_real_gradients_and_guard_source_mutant(self):
        model = self.by['S6'].model
        self.assertTrue(m.mtp_check(model, self.tokens)['greedy_equal'])
        for kind in ('side_zero', 'side_detached', 'side_nan'):
            with self.subTest(fault=kind), gradient_fault(model, kind):
                result = model.forward_mtp(self.tokens, weight=.2)
                self.assertTrue(torch.isfinite(result.loss))
                model.zero_grad(set_to_none=True)
                result.loss.backward()
                grad = model.mtp_heads[0].weight.grad
                if kind == 'side_zero':
                    self.assertTrue(torch.isfinite(grad).all())
                    self.assertEqual(torch.count_nonzero(grad), 0)
                elif kind == 'side_detached':
                    self.assertIsNone(grad)
                else:
                    self.assertTrue(torch.isnan(grad).all())
                # A healthy main head/other side head prevents an all-loss failure
                # from masquerading as this targeted auxiliary-gradient regression.
                for layer in (model.head, model.mtp_heads[1]):
                    self.assertTrue(torch.isfinite(layer.weight.grad).all())
                    self.assertGreater(float(layer.weight.grad.abs().sum()), 0)
                with self.assertRaisesRegex(AssertionError, 'MTP heads must receive finite nonzero gradients'):
                    m.mtp_check(model, self.tokens)
                mutant = m.mutate_function(m.mtp_check,
                    'if any(head.weight.grad is None or not torch.isfinite(head.weight.grad).all() or\n'
                    '           not head.weight.grad.abs().sum() > 0 for head in model.mtp_heads):', 'if False:')
                if kind == 'side_nan':
                    # This bad case is also guarded by the general NaN check;
                    # do not misreport redundant protection as a surviving mutant.
                    with self.assertRaisesRegex(AssertionError, 'MTP nonfinite backbone gradients'):
                        mutant(model, self.tokens)
                else:
                    escaped = mutant(model, self.tokens)
                    self.assertTrue(escaped['greedy_equal'])
                    self.assertEqual(escaped['label_counts'], [14, 12, 10])

    def test_float64_payload_doubles_and_fixed_four_byte_source_mutant_fails(self):
        for stage in self.stages[1:]:
            with self.subTest(stage=stage.name):
                expected32 = 4096 if stage.name in ('S1', 'S2', 'S3') else 2048
                self.assertEqual(m.cache_equivalence(stage, self.tokens)[1], expected32)
                stage.model.double()
                _, cache = stage.cached_forward(stage.model, self.tokens)
                # Directly inspect physical dtype and scalar element count first.
                tensors = [tensor for pair in cache.layers for tensor in pair]
                self.assertTrue(all(t.dtype == torch.float64 and t.element_size() == 8 for t in tensors))
                self.assertEqual(sum(t.numel() for t in tensors), expected32 // 4)
                self.assertEqual(sum(t.numel() * 8 for t in tensors), expected32 * 2)
                self.assertEqual(m.cache_equivalence(stage, self.tokens)[1], expected32 * 2)
                mutant = m.mutate_function(m.cache_equivalence,
                    'next(stage.model.parameters()).element_size()', '4')
                with self.assertRaisesRegex(AssertionError, 'KV tensor payload differs from independent ledger'):
                    mutant(stage, self.tokens)

    def test_weights_only_refuses_inert_pickle_container_and_mutant_accepts(self):
        normal = self.checkpoint('S2')
        original = m.load_checkpoint(io.BytesIO(normal), 'S2', self.by['S2'].model.config, m.VOCAB)
        raw = torch.load(io.BytesIO(normal), weights_only=True)
        inert = InertCheckpointEnvelope(raw)
        data = io.BytesIO()
        torch.save(inert, data)
        with self.assertRaises(pickle.UnpicklingError) as caught:
            m.load_checkpoint(io.BytesIO(data.getvalue()), 'S2', original.config, m.VOCAB)
        self.assertIn('Unsupported global', str(caught.exception))
        self.assertIn('InertCheckpointEnvelope', str(caught.exception))
        mutant = m.mutate_function(m.load_checkpoint, 'weights_only=True)', 'weights_only=False)')
        # Deliberately NO custom reduce/state hooks or executable payload. The
        # mutant reconstructs only this inert dict, demonstrating the lost gate.
        restored = mutant(io.BytesIO(data.getvalue()), 'S2', original.config, m.VOCAB)
        for key, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, restored.state_dict()[key]))
        m.check_close(restored(self.tokens)[0], original(self.tokens)[0], atol=0, rtol=0)

    def test_bfloat16_save_refusal_and_dtype_guard_source_mutant(self):
        model = self.by['S2'].model
        normal = io.BytesIO()
        m.save_checkpoint(normal, 'S2', model, m.VOCAB)
        self.assertGreater(len(normal.getvalue()), 0)
        model.bfloat16()
        self.assertTrue(all(p.dtype == torch.bfloat16 for p in model.parameters()))
        destination = io.BytesIO()
        with self.assertRaisesRegex(ValueError, 'support float32/float64 only'):
            m.save_checkpoint(destination, 'S2', model, m.VOCAB)
        self.assertEqual(destination.getvalue(), b'')
        mutant = m.mutate_function(m._metadata, 'if dtype not in (torch.float32, torch.float64):', 'if False:')
        with patch.object(m, '_metadata', mutant):
            m.save_checkpoint(destination, 'S2', model, m.VOCAB)
        raw = torch.load(io.BytesIO(destination.getvalue()), weights_only=True)
        self.assertEqual(raw['metadata']['dtype'], 'torch.bfloat16')
        self.assertEqual(raw['state_dict']['tok_emb.weight'].dtype, torch.bfloat16)

    def test_nan_save_refusal_and_finite_guard_source_mutant(self):
        model = self.by['S2'].model
        m.save_checkpoint(io.BytesIO(), 'S2', model, m.VOCAB)
        with torch.no_grad():
            model.tok_emb.weight[0, 0] = float('nan')
        self.assertTrue(torch.isnan(model.tok_emb.weight[0, 0]))
        destination = io.BytesIO()
        with self.assertRaisesRegex(ValueError, 'checkpoint tensors must be finite'):
            m.save_checkpoint(destination, 'S2', model, m.VOCAB)
        self.assertEqual(destination.getvalue(), b'')
        mutant = m.mutate_function(m.save_checkpoint,
            'if any(t.is_floating_point() and (t.dtype != next(model.parameters()).dtype or\n'
            '            not torch.isfinite(t).all()) for t in state.values()):', 'if False:')
        mutant(destination, 'S2', model, m.VOCAB)
        raw = torch.load(io.BytesIO(destination.getvalue()), weights_only=True)
        self.assertTrue(torch.isnan(raw['state_dict']['tok_emb.weight'][0, 0]))

    def test_extra_envelope_key_refusal_and_source_mutant(self):
        normal = self.checkpoint('S2')
        config = self.by['S2'].model.config
        m.load_checkpoint(io.BytesIO(normal), 'S2', config, m.VOCAB)
        raw = torch.load(io.BytesIO(normal), weights_only=True)
        raw['unexpected_resume_state'] = {'step': 123}
        data = io.BytesIO()
        torch.save(raw, data)
        self.assertEqual(set(raw), {'metadata', 'state_dict', 'unexpected_resume_state'})
        with self.assertRaisesRegex(ValueError, 'checkpoint envelope mismatch'):
            m.load_checkpoint(io.BytesIO(data.getvalue()), 'S2', config, m.VOCAB)
        mutant = m.mutate_function(m.load_checkpoint,
            " or set(checkpoint) != {'metadata', 'state_dict'}", '')
        restored = mutant(io.BytesIO(data.getvalue()), 'S2', config, m.VOCAB)
        m.check_close(restored(self.tokens)[0], self.by['S2'].model(self.tokens)[0], atol=0, rtol=0)

    def test_s1_s2_owner_guards_are_independent_of_weight_signature(self):
        for name in ('S1', 'S2'):
            with self.subTest(stage=name):
                stage = self.by[name]
                _, cache = stage.cached_forward(stage.model, self.tokens[:, :3])
                expected, _ = stage.cached_forward(stage.model, self.tokens[:, 3:], cache)
                other = m.construct(name, stage.model.config).eval()
                # Deliberately wrong owner ONLY. Do not let a different weight
                # signature catch the fault before this independent gate is tested.
                wrong_owner = replace(cache, owner=other)
                self.assertIsNot(wrong_owner.owner, stage.model)
                self.assertEqual(wrong_owner.versions, cache.versions)
                self.assertIs(wrong_owner.layers, cache.layers)
                with self.assertRaisesRegex(ValueError, 'different model/request'):
                    stage.cached_forward(stage.model, self.tokens[:, 3:], wrong_owner)
                function = m.s1.cached_forward if name == 'S1' else m.s2._validate_cache
                mutant = m.mutate_function(function, ' or cache.owner is not model', '')
                if name == 'S1':
                    actual, escaped_cache = mutant(stage.model, self.tokens[:, 3:], wrong_owner)
                else:
                    with patch.object(m.s2, '_validate_cache', mutant):
                        actual, escaped_cache = stage.cached_forward(stage.model, self.tokens[:, 3:], wrong_owner)
                m.check_close(actual, expected, atol=0, rtol=0)
                self.assertEqual(escaped_cache.length, 8)
                # Correct numerical payload does not excuse wrong owner metadata.

    def test_s6_copy_is_observable_after_distinct_seed_source_mutant(self):
        original = m.make_stages(include_mtp=True)
        s5, s6 = original[-2].model, original[-1].model
        self.assertTrue(all(torch.equal(value, s6.state_dict()[key]) for key, value in s5.state_dict().items()))
        m.check_close(s5(self.tokens)[0], s6(self.tokens)[0], atol=0, rtol=0)
        mutant = m.mutate_function(m.make_stages,
            'model.load_state_dict({**model.state_dict(), **result[-1].model.state_dict()}, strict=True)',
            'pass')
        changed = mutant(include_mtp=True)
        old, new = changed[-2].model, changed[-1].model
        self.assertFalse(torch.equal(old.tok_emb.weight, new.tok_emb.weight))
        actual, expected = new(self.tokens)[0], old(self.tokens)[0]
        self.assertTrue(torch.isfinite(actual).all())
        self.assertGreater(float((actual - expected).detach().abs().max()), .1)
        with self.assertRaises(AssertionError):
            m.check_close(actual, expected, atol=0, rtol=0)

    def test_json_topologies_come_from_actual_calls_and_missing_topology_mutant(self):
        observed = []
        real = m.cache_equivalence
        def record(stage, tokens, sizes=(3, 1, 4)):
            error, payload = real(stage, tokens, sizes)
            observed.append((stage.name, list(sizes), error, payload))
            return error, payload
        with patch.object(m, 'cache_equivalence', record):
            report = m.run_healthcheck()
        measured = [(row['stage'], check['sizes'], check['max_logit_error'], check['kv_bytes'])
                    for row in report['stages'][1:] for check in row['chunk_checks']]
        self.assertEqual(len(measured), 20)
        self.assertEqual(observed[:20], measured)
        self.assertEqual(len(observed), 23)  # Plus three explicit S3 controls.
        mutant = m.mutate_function(m.run_healthcheck,
            'for sizes in ((3, 1, 4), (1,) * 8, (8,), (7, 1)):', 'for sizes in ((3, 1, 4),):')
        incomplete = mutant()
        for row in incomplete['stages'][1:]:
            self.assertEqual([check['sizes'] for check in row['chunk_checks']], [[3, 1, 4]])
            self.assertNotEqual([check['sizes'] for check in row['chunk_checks']],
                                [[3, 1, 4], [1] * 8, [8], [7, 1]])

    def _cache_output_fault(self, kind):
        stage = self.by['S1']
        observations = []
        def forward(model, chunk, cache=None):
            output, state = stage.cached_forward(model, chunk, cache)
            if kind == 'all_offset' or cache is None:
                correct = output
                output = output + .25 if kind == 'all_offset' else output * 1.5
                observations.append(float((output - correct).abs().max()))
                self.assertTrue(torch.isfinite(output).all())
            return output, state
        return m.Stage('S1', stage.model, forward), observations

    @contextmanager
    def _pipeline_fault(self, kind):
        """Real faulty output/state, with every unrelated healthcheck intact."""
        stages = m.make_stages()
        stage = stages[1]
        observations = []
        if kind == 'overflow':
            def forward(model, chunk, cache=None):
                if cache is not None and cache.length + chunk.shape[1] > model.max_seq_len:
                    before = cache.length
                    # A plausible but invalid sliding-window fallback: drop old
                    # K/V instead of honoring this lesson's overflow refusal.
                    drop = before + chunk.shape[1] - model.max_seq_len
                    cache = replace(cache, layers=tuple((k[:, :, drop:], v[:, :, drop:])
                                                        for k, v in cache.layers))
                    output, result = stage.cached_forward(model, chunk, cache)
                    observations.append((before, chunk.shape[1], result.length))
                    self.assertTrue(torch.isfinite(output).all())
                    return output, result
                return stage.cached_forward(model, chunk, cache)
            stages[1] = m.Stage('S1', stage.model, forward)
        elif kind == 'reset':
            other = self.tokens.flip(1)
            _, stale = stage.cached_forward(stage.model, self.tokens[:, :3], None)
            def forward(model, chunk, cache=None):
                if cache is None and torch.equal(chunk, other):
                    output, result = stage.cached_forward(model, chunk, stale)
                    delta = float((output - model(chunk)[0]).detach().abs().max())
                    observations.append((stale.length, result.length, delta))
                    self.assertTrue(torch.isfinite(output).all())
                    return output, result
                return stage.cached_forward(model, chunk, cache)
            stages[1] = m.Stage('S1', stage.model, forward)
        elif kind == 'moe_count':
            layer = stages[-1].model.blocks[0].mlp
            original = layer.forward
            def forward(inputs):
                output = original(inputs)
                info = layer.last_info
                bad = info.requested.clone()
                bad[0] += 1
                layer.last_info = replace(info, requested=bad)
                observations.append((int(info.requested.sum()), int(bad.sum())))
                return output
        else:
            raise ValueError('unknown pipeline fault')
        with patch.object(m, 'make_stages', return_value=stages):
            if kind == 'moe_count':
                with patch.object(layer, 'forward', forward):
                    yield stages, observations
            else:
                yield stages, observations

    @contextmanager
    def _nan_loss_fault(self):
        model = self.by['S0'].model
        original = model.forward
        def forward(tokens, targets=None):
            logits, loss = original(tokens, targets)
            # NaN in the value, but derivative 1 with respect to the real loss:
            # deleting the loss guard cannot be rescued by gradient checks.
            return logits, loss + loss.new_tensor(float('nan')) if loss is not None else None
        with patch.object(model, 'forward', forward):
            yield model
        model.zero_grad(set_to_none=True)

    @contextmanager
    def _unused_mask_checkpoint_fault(self):
        original = m.load_checkpoint
        observations = []
        def load(*args, **kwargs):
            model = original(*args, **kwargs)
            mask = model.blocks[0].attn.mask
            self.assertEqual(tuple(mask.shape), (16, 16))
            with torch.no_grad():
                mask[15, 15] = 0
            observations.append(model)
            return model
        with patch.object(m, 'load_checkpoint', load):
            yield observations

    @contextmanager
    def _wrong_rejection_fault(self):
        original = m.load_checkpoint
        observations = []
        def load(source, name, config, vocabulary, depth=0):
            raw = torch.load(io.BytesIO(source.getvalue()), weights_only=True)
            if name == 'S2' and raw['metadata']['family'] == 'baby':
                raw['metadata']['dtype'] = 'torch.bfloat16'
                source = io.BytesIO()
                torch.save(raw, source)
                source.seek(0)
                try:
                    return original(source, name, config, vocabulary, depth)
                except ValueError as exc:
                    observations.append(str(exc))
                    raise
            return original(source, name, config, vocabulary, depth)
        with patch.object(m, 'load_checkpoint', load):
            yield observations

    def test_plain_behavior_full_forward_reference_all_chunks_offset(self):
        for sizes in ((3, 1, 4), (1,) * 8, (8,), (7, 1)):
            with self.subTest(sizes=sizes):
                m.cache_equivalence(self.by['S1'], self.tokens, sizes)
                bad, observations = self._cache_output_fault('all_offset')
                with self.assertRaises(AssertionError):
                    m.cache_equivalence(bad, self.tokens, sizes)
                self.assertEqual(len(observations), len(sizes))
                self.assertTrue(all(delta > .24 for delta in observations))

    def test_plain_behavior_first_chunk_scaled_all_topologies(self):
        for sizes in ((3, 1, 4), (1,) * 8, (8,), (7, 1)):
            with self.subTest(sizes=sizes):
                m.cache_equivalence(self.by['S1'], self.tokens, sizes)
                bad, observations = self._cache_output_fault('first_scaled')
                with self.assertRaises(AssertionError):
                    m.cache_equivalence(bad, self.tokens, sizes)
                self.assertEqual(len(observations), 1)
                self.assertGreater(observations[0], .1)

    def _wrong_cache_length(self, delta=1):
        from types import SimpleNamespace
        stage = self.by['S1']
        observations = []
        def forward(model, chunk, cache=None):
            output, result = stage.cached_forward(model, chunk, cache)
            if result.length == self.tokens.shape[1]:
                observations.append((result.length, result.length + delta))
                # Return a malformed cache-interface object at the final chunk;
                # tensors/logits stay intact, so only the length guard detects it.
                result = SimpleNamespace(length=result.length + delta, layers=result.layers)
            return output, result
        return m.Stage('S1', stage.model, forward), observations

    def test_plain_behavior_wrong_cache_length_is_refused(self):
        m.cache_equivalence(self.by['S1'], self.tokens)
        for delta in (-1, 1):
            with self.subTest(delta=delta):
                stage, observations = self._wrong_cache_length(delta)
                with self.assertRaisesRegex(AssertionError, 'cache length differs from consumed prefix'):
                    m.cache_equivalence(stage, self.tokens)
                self.assertEqual(observations, [(8, 8 + delta)])

    def test_plain_behavior_capacity_overflow_cannot_silently_truncate(self):
        self.assertTrue(m.run_healthcheck()['stages'][1]['capacity_rejected'])
        with self._pipeline_fault('overflow') as (_, observations):
            with self.assertRaisesRegex(AssertionError, 'cache capacity overflow accepted'):
                m.run_healthcheck()
            self.assertEqual(observations, [(16, 1, 16)])

    def test_plain_behavior_none_resets_independent_request(self):
        self.assertEqual(len(m.run_healthcheck()['stages']), 6)
        with self._pipeline_fault('reset') as (_, observations):
            with self.assertRaises(AssertionError):
                m.run_healthcheck()
            self.assertEqual(len(observations), 1)
            self.assertEqual(observations[0][:2], (3, 11))
            self.assertGreater(observations[0][2], .1)

    def test_plain_behavior_moe_requested_sum_is_independent_ledger(self):
        normal = m.run_healthcheck()['stages'][-1]['expert_loads']
        self.assertTrue(all(sum(load) == 32 for load in normal))
        with self._pipeline_fault('moe_count') as (_, observations):
            with self.assertRaisesRegex(AssertionError, 'MoE assignment ledger mismatch'):
                m.run_healthcheck()
            self.assertEqual(observations[-1], (32, 33))

    def test_plain_behavior_nan_loss_is_refused_even_with_finite_gradients(self):
        self.assertTrue(math.isfinite(m.finite_backward(self.by['S0'].model, self.tokens, self.targets)))
        with self._nan_loss_fault() as model:
            logits, loss = model(self.tokens, self.targets)
            self.assertTrue(torch.isfinite(logits).all())
            self.assertTrue(torch.isnan(loss))
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(float(model.head.weight.grad.abs().sum()), 0)
            with self.assertRaisesRegex(AssertionError, 'nonfinite loss'):
                m.finite_backward(model, self.tokens, self.targets)

    def test_plain_behavior_checkpoint_unused_mask_still_must_match(self):
        stage = self.by['S0']
        self.assertEqual(m.checkpoint_error(stage, self.tokens), 0)
        with self._unused_mask_checkpoint_fault() as observations:
            with self.assertRaisesRegex(AssertionError, 'checkpoint changed state tensor'):
                m.checkpoint_error(stage, self.tokens)
            changed = observations[-1]
            self.assertNotEqual(changed.blocks[0].attn.mask[15, 15], stage.model.blocks[0].attn.mask[15, 15])
            m.check_close(changed(self.tokens)[0], stage.model(self.tokens)[0], atol=0, rtol=0)

    def test_plain_behavior_wrong_rejection_reason_is_not_compatibility_pass(self):
        self.assertIn('S0_to_S2', m.compatibility_checks(self.stages)['rejected'])
        with self._wrong_rejection_fault() as observations:
            with self.assertRaisesRegex(ValueError, 'unsupported checkpoint dtype/metadata'):
                m.compatibility_checks(self.stages)
            self.assertEqual(observations, ['unsupported checkpoint dtype/metadata'])

    def test_plain_behavior_cli_causal_sweep_catches_last_layer_local_leak(self):
        report = m.run_healthcheck()
        for row in report['stages']:
            self.assertEqual([check['cut'] for check in row['causal_checks']], list(range(1, 8)))
            self.assertEqual(row['causal_error'], max(check['max_logit_error'] for check in row['causal_checks']))
        model = self.by['S0'].model
        with torch.no_grad():
            model.blocks[-1].attn.mask[5, 6] = 1
        # Only the last attention layer leaks at position 5. The old cut=3
        # probe remains exactly unchanged, but cut=6 exposes a future dependency.
        self.assertEqual(m.causal_error(model, self.tokens, cut=3), 0)
        with self.assertRaises(AssertionError):
            m.causal_error(model, self.tokens, cut=6)
        with patch.object(m, 'make_stages', return_value=self.stages), \
             patch.object(m, 'cache_equivalence', wraps=m.cache_equivalence) as cache_check:
            with self.assertRaises(AssertionError):
                m.run_healthcheck()
            cache_check.assert_not_called()  # Causal probe, not a later cache mismatch.

    def test_source_mutant_cached_path_reference_misses_shared_output_bias(self):
        self.test_plain_behavior_full_forward_reference_all_chunks_offset()
        mutant = m.mutate_function(m.cache_equivalence,
            'expected = stage.model(tokens)[0]',
            'expected = stage.cached_forward(stage.model, tokens, None)[0]')
        for sizes in ((3, 1, 4), (1,) * 8, (8,), (7, 1)):
            stage, observations = self._cache_output_fault('all_offset')
            error, payload = mutant(stage, self.tokens, sizes)
            self.assertLess(error, 3e-6)
            self.assertEqual(payload, 4096)
            self.assertEqual(len(observations), len(sizes) + 1)
            self.assertTrue(all(delta > .24 for delta in observations))

    def test_source_mutant_wrong_cache_length_guard_accepts_malformed_state(self):
        self.test_plain_behavior_wrong_cache_length_is_refused()
        mutant = m.mutate_function(m.cache_equivalence, 'if cache.length != start:', 'if False:')
        for delta in (-1, 1):
            with self.subTest(delta=delta):
                stage, observations = self._wrong_cache_length(delta)
                error, payload = mutant(stage, self.tokens)
                self.assertLess(error, 3e-6)
                self.assertEqual(payload, 4096)
                self.assertEqual(observations, [(8, 8 + delta)])
        # A one-sided replacement still rejects oversized lengths but silently
        # accepts undersized ones. The plain behavior test above must catch it.
        one_sided = m.mutate_function(m.cache_equivalence, 'if cache.length != start:',
                                     'if cache.length > start:')
        shorter, observations = self._wrong_cache_length(-1)
        error, payload = one_sided(shorter, self.tokens)
        self.assertLess(error, 3e-6)
        self.assertEqual(payload, 4096)
        self.assertEqual(observations, [(8, 7)])

    def test_source_mutants_pipeline_guards_accept_specific_bad_behaviors(self):
        cases = [('overflow', "raise AssertionError('cache capacity overflow accepted')", 'pass'),
                 ('reset', 'check_close(fresh, model(other)[0])', 'pass'),
                 ('moe_count', 'if any(sum(load) != tokens.numel() * model.config.top_k for load in loads):', 'if False:')]
        # Pure production checks all complete before any source mutation.
        self.test_plain_behavior_capacity_overflow_cannot_silently_truncate()
        self.test_plain_behavior_none_resets_independent_request()
        self.test_plain_behavior_moe_requested_sum_is_independent_ledger()
        for kind, old, new in cases:
            with self.subTest(kind=kind):
                mutant = m.mutate_function(m.run_healthcheck, old, new)
                # A compiled function has its own globals mapping. Patch its
                # stage factory explicitly; otherwise we'd test healthy models.
                with self._pipeline_fault(kind) as (stages, observations):
                    mutant.__globals__['make_stages'] = lambda include_mtp=False: stages
                    report = mutant()
                    self.assertEqual(len(report['stages']), 6)
                    if kind == 'overflow':
                        self.assertEqual(observations, [(16, 1, 16)])
                        self.assertTrue(report['stages'][1]['capacity_rejected'])
                    elif kind == 'reset':
                        self.assertEqual(len(observations), 1)
                        self.assertEqual(observations[0][:2], (3, 11))
                        self.assertGreater(observations[0][2], .1)
                    else:
                        self.assertEqual(sum(report['stages'][-1]['expert_loads'][0]), 33)
                        self.assertEqual(observations[-1], (32, 33))

    def test_source_mutant_loss_guard_returns_nan_with_finite_gradients(self):
        self.test_plain_behavior_nan_loss_is_refused_even_with_finite_gradients()
        mutant = m.mutate_function(m.finite_backward, 'if not torch.isfinite(loss):', 'if False:')
        with self._nan_loss_fault() as model:
            self.assertTrue(math.isnan(mutant(model, self.tokens, self.targets)))

    def test_source_mutant_checkpoint_tensor_guard_misses_unused_mask_change(self):
        self.test_plain_behavior_checkpoint_unused_mask_still_must_match()
        mutant = m.mutate_function(m.checkpoint_error,
            'if not torch.equal(tensor, loaded.state_dict()[key]):', 'if False:')
        with self._unused_mask_checkpoint_fault() as observations:
            mutant.__globals__['load_checkpoint'] = m.load_checkpoint
            self.assertEqual(mutant(self.by['S0'], self.tokens), 0)
            self.assertEqual(observations[-1].blocks[0].attn.mask[15, 15], 0)

    def test_source_mutant_compatibility_reason_guard_mislabels_dtype_refusal(self):
        self.test_plain_behavior_wrong_rejection_reason_is_not_compatibility_pass()
        mutant = m.mutate_function(m.compatibility_checks,
            "if 'architecture/config/vocabulary/depth mismatch' not in str(exc):", 'if False:')
        with self._wrong_rejection_fault() as observations:
            mutant.__globals__['load_checkpoint'] = m.load_checkpoint
            report = mutant(self.stages)
            self.assertIn('S0_to_S2', report['rejected'])
            self.assertEqual(observations, ['unsupported checkpoint dtype/metadata'])

    def test_source_mutant_single_causal_cut_cannot_match_json_contract(self):
        report = m.run_healthcheck()
        self.assertTrue(all([item['cut'] for item in row['causal_checks']] == list(range(1, 8))
                            for row in report['stages']))
        mutant = m.mutate_function(m.run_healthcheck,
            'for cut in range(1, tokens.shape[1])', 'for cut in (3,)')
        incomplete = mutant()
        self.assertTrue(all([item['cut'] for item in row['causal_checks']] == [3]
                            for row in incomplete['stages']))
        self.assertTrue(all([item['cut'] for item in row['causal_checks']] != list(range(1, 8))
                            for row in incomplete['stages']))

    def test_negative_controls_cannot_pass_when_mutants_are_unchanged(self):
        with patch.object(m, 'mutate_function', side_effect=lambda function, old, new: function):
            with self.assertRaisesRegex(AssertionError, 'survived the equivalence oracle'):
                m.negative_controls(self.by['S0'].model, self.tokens)

    def test_negative_dependency_exception_propagates_not_caught_as_success(self):
        with patch.object(m, 'mutate_function', side_effect=ModuleNotFoundError('missing helper')):
            with self.assertRaises(ModuleNotFoundError):
                m.negative_controls(self.by['S0'].model, self.tokens)
        with self.assertRaisesRegex(AssertionError, 'anchor'):
            m.mutate_function(m.s1.cached_forward, 'nonexistent anchor', 'x')


if __name__ == '__main__':
    unittest.main()
