"""P27 integration contracts, literal/scalar oracles and actual behavior mutants.

Skipped Torch paths are explicitly NOT passing numerical evidence. The report is
not accepted merely because it prints PASS; tests exercise components and guards.
"""
from dataclasses import replace
import importlib.util
import io
import json
import math
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
            self.assertEqual([x['name'] for x in report['negatives']],
                             ['wrong_mask', 'wrong_position', 'cache_pollution'])
            self.assertTrue(all(x['caught'] and x['max_logit_error'] > .1 for x in report['negatives']))
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
        stage = self.by['S4']
        _, cache = stage.cached_forward(stage.model, self.tokens[:, :3])
        loaded = m.load_checkpoint(io.BytesIO(self.checkpoint('S4')), 'S4', stage.model.config, m.VOCAB)
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
