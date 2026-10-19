"""P26: offline evidence contracts, independent scalar/coordinate oracles, mutants."""
import copy
from fractions import Fraction
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'code/27_deepseek_case.py'
spec = importlib.util.spec_from_file_location('deepseek_case', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def coordinate_oracle(grid, ratio):
    """Destination-index inversion, not the production source-block loops."""
    h, w, channels = len(grid), len(grid[0]), len(grid[0][0])
    rows, cols = math.ceil(h/ratio), math.ceil(w/ratio)
    result = []
    for token in range(rows*cols):
        out = []
        for feature in range(channels*ratio*ratio):
            channel, offset = divmod(feature, ratio*ratio)
            y = (token//cols)*ratio + offset//ratio
            x = (token % cols)*ratio + offset % ratio
            out.append(grid[y][x][channel] if y < h and x < w else 0)
        result.append(out)
    return result


def mutant(old, new):
    """Compile a valid changed program; assertions must reject its returned values."""
    source = SCRIPT.read_text(encoding='utf-8')
    if source.count(old) != 1:
        raise AssertionError('mutation must target exactly one site')
    namespace = {'__name__': 'p26_mutant', '__file__': str(SCRIPT)}
    exec(compile(source.replace(old, new), str(SCRIPT), 'exec'), namespace)
    return types.SimpleNamespace(**namespace)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.manifest, self.configs = m.load_sources()

    def test_official_identity_pins_and_raw_bytes(self):
        expected = {
            'v3': 'e815299b0bcbac849fa540c768ef21845365c9eb',
            'r1': '56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad',
            'distill': 'ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562',
            'v41': '314113f8a7dadb388215396ba5fbee2c610a04d6',
        }
        self.assertEqual({k: v['revision'] for k, v in self.manifest['models'].items()}, expected)
        self.assertEqual(self.manifest['checked_date'], '2026-10-19')
        self.assertEqual(len(self.configs), 5)
        self.assertEqual(self.manifest['files']['v41.config.json']['sha256'],
                         '8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879')
        for row in self.manifest['files'].values():
            self.assertNotIn('/main/', row['url'])

    def test_fixture_corruption_and_unpinned_urls_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for path in m.DATA.glob('*.json'):
                shutil.copy(path, root/path.name)
            config = root/'v3.config.json'
            config.write_bytes(config.read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                m.load_sources(root)
            shutil.copy(m.DATA/'v3.config.json', config)
            manifest = copy.deepcopy(self.manifest)
            manifest['files']['v3.config.json']['url'] = 'https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/main/config.json'
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'pinned model revision'):
                m.load_sources(root)
            manifest = copy.deepcopy(self.manifest)
            manifest['models']['v3']['revision'] = 'main'
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'immutable'):
                m.load_sources(root)

    def test_file_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest = copy.deepcopy(self.manifest)
            manifest['files'] = {'../outside.json': next(iter(manifest['files'].values()))}
            (Path(folder)/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'basename'):
                m.load_sources(folder)

    def test_missing_field_is_unknown_and_existing_false_stays_false(self):
        fields, missing = m.extract_fields({'a': {'b': False}}, ('a.b', 'a.c'))
        self.assertIs(fields['a.b'], False)
        self.assertIsNone(fields['a.c'])
        self.assertEqual(missing, ['a.c'])
        configs = copy.deepcopy(self.configs)
        del configs['v3.config.json']['kv_lora_rank']
        rows = m.evidence_matrix(self.manifest, configs)
        self.assertEqual(rows[0]['status'], 'unknown')
        self.assertEqual(rows[0]['missing'], ['kv_lora_rank'])

    def test_matrix_separates_config_and_reviewed_training_evidence(self):
        rows = m.evidence_matrix(self.manifest, self.configs)
        self.assertEqual(len(rows), 19)
        self.assertEqual(sum(r['status'] == 'extracted' for r in rows), 11)
        self.assertEqual(sum(r['status'] == 'reviewed' for r in rows), 8)
        config_row = next(r for r in rows if r['component'] == 'V4.1-Flash / DSpark 字段')
        self.assertEqual(config_row['values']['n_mtp_layers'], 3)
        # No generated inference that MTP was used during backbone pretraining.
        self.assertNotIn('backbone_mtp', json.dumps(config_row))
        paper_row = next(r for r in rows if r['component'] == 'V4.1-Flash / 训练时序')
        self.assertEqual(paper_row['support'], 'paper')
        self.assertIn('省略 MTP', paper_row['statement'])
        self.assertIn('2609.19969v1', paper_row['source'])

    def test_student_and_teacher_cannot_share_an_architecture_label(self):
        student = self.configs['distill.config.json']
        teacher = self.configs['r1.config.json']
        self.assertEqual(student['model_type'], 'qwen2')
        self.assertEqual(teacher['model_type'], 'deepseek_v3')
        self.assertEqual((student['num_attention_heads'], student['num_key_value_heads']), (12, 2))
        self.assertIs(student['use_sliding_window'], False)
        # A scalar window exists even though the corresponding path is disabled.
        self.assertEqual(student['sliding_window'], 4096)

    def test_official_config_and_reference_config_aliases_agree(self):
        text = self.configs['v41.config.json']['text_config']
        ref = self.configs['v41.inference_config.json']
        for left, right in [('num_hidden_layers', 'n_layers'), ('sliding_window', 'window_size'),
                            ('kv_source_layer_ids', 'kv_source_layers'),
                            ('index_source_layer_ids', 'index_source_layers'),
                            ('num_nextn_predict_layers', 'n_mtp_layers'),
                            ('compress_ratios', 'compress_ratios')]:
            self.assertEqual(text[left], ref[right])
        self.assertEqual(len(ref['compress_ratios']), 43)
        self.assertEqual(ref['n_layers'], 40)

    def test_cli_json_is_offline_portable_and_no_performance_claim(self):
        with tempfile.TemporaryDirectory() as folder:
            # Running from elsewhere must still locate sibling fixture files.
            proc = subprocess.run([sys.executable, str(SCRIPT), '--json'], cwd=folder,
                                  capture_output=True, text=True, check=True)
            report = json.loads(proc.stdout)
            self.assertEqual(len(report['matrix']), 19)
            self.assertIn('official weights', report['not_run'])
            self.assertEqual(report['experiments']['csa2']['shared_sources']['entries'], 25)
        proc = subprocess.run([sys.executable, str(SCRIPT), '--matrix-only'],
                              capture_output=True, text=True, check=True)
        self.assertTrue(proc.stdout.startswith('| 组件 |'))
        self.assertNotIn('CPU 教学实验', proc.stdout)
        # No Torch-dependent skip can make this suite appear to pass.
        self.assertNotIn('import torch', SCRIPT.read_text())
        self.assertNotIn('urllib', SCRIPT.read_text())


class ShapeTests(unittest.TestCase):
    def setUp(self):
        _, configs = m.load_sources()
        self.config = configs['v41.inference_config.json']

    def assert_mla_oracle(self, module):
        result = module.mla_probe()
        self.assertEqual(result['expanded_logits'], [-2., -1., 0.])
        self.assertEqual(result['latent_logits'], [-2., -1., 0.])
        a, b = math.exp(-2), math.exp(-1)
        expected = [(4*a-b+4)/(a+b+1), (6*a+3*b)/(a+b+1)]
        for actual in (result['expanded_output'], result['absorbed_output']):
            for x, y in zip(actual, expected):
                self.assertAlmostEqual(x, y, places=13)
        self.assertLess(result['max_error'], 1e-14)

    def test_mla_hand_logits_and_independent_weighted_values(self):
        self.assert_mla_oracle(m)

    def test_moe_hand_fraction_oracle_and_bias_changes_only_selection(self):
        result = m.route_experts([.75, .5, .25], [-1., 0., 1.],
                                [[2., 0.], [0., 4.], [6., 2.]], [1., -1.], 2)
        self.assertEqual(result['selected'], [2, 1])
        for actual, expected in zip(result['weights'], (Fraction(1, 3), Fraction(2, 3))):
            self.assertAlmostEqual(actual, float(expected))
        self.assertEqual(result['output'][0], 3.)
        self.assertAlmostEqual(result['output'][1], float(Fraction(7, 3)))
        changed = m.route_experts([.75, .5, .25], [-4., 0., 2.],
                                 [[2., 0.], [0., 4.], [6., 2.]], [1., -1.], 2)
        self.assertEqual(changed, result)  # ranking unchanged, despite changed biases
        scaled = m.route_experts([.75, .5, .25], [-1., 0., 1.],
                                [[2., 0.], [0., 4.], [6., 2.]], [1., -1.], 2, 2.5)
        self.assertEqual(scaled['output'][0], 6.)
        self.assertAlmostEqual(scaled['output'][1], float(Fraction(22, 3)))

    def test_tie_and_invalid_inputs(self):
        self.assertEqual(m.route_experts([.5, .5], [0, 0], [[1], [2]], [0], 1)['selected'], [0])
        for k in (0, True, 3):
            with self.assertRaises(ValueError):
                m.route_experts([.5, .5], [0, 0], [[1], [2]], [0], k)
        with self.assertRaises(ValueError):
            m.route_experts([math.nan, .5], [0, 0], [[1], [2]], [0], 1)
        for a, b in (([[1, 2]], [[1]]), ([[math.inf]], [[1]]), ([], [[1]])):
            with self.assertRaises(ValueError):
                m.matmul(a, b)
        self.assertEqual(m.matmul([[1, 2], [3, 4]], [[2, -1], [1, 2]]), [[4, 3], [10, 5]])

    def test_schedule_independent_interval_oracle(self):
        schedule = m.csa2_schedule(self.config)
        self.assertEqual(len(schedule), 40)
        for layer, row in enumerate(schedule):
            self.assertEqual(row['local_source'], layer)
            if layer < 2:
                self.assertEqual(row['mode'], 'SWA')
                self.assertIsNone(row['kv_source'])
                continue
            expected_kv = 2 if layer < 8 else 8 if layer < 14 else 14 if layer < 20 else 20
            expected_index = expected_kv if layer < 20 else 20+4*((layer-20)//4)
            expected_mode = 'Full' if layer in (2, 8, 14, 20) else 'Reindex' if layer in (24, 28, 32, 36) else 'Reuse'
            self.assertEqual((row['kv_source'], row['index_source'], row['mode']),
                             (expected_kv, expected_index, expected_mode))
        ledger = m.cache_ledger(self.config)
        self.assertEqual(ledger['modes'], {'SWA': 2, 'Full': 4, 'Reuse': 30, 'Reindex': 4})
        self.assertEqual(ledger['shared_sources'], {'banks': 4, 'entries': 25, 'payload_bytes': 50})
        self.assertEqual(ledger['independent_per_layer'], {'banks': 38, 'entries': 290, 'payload_bytes': 580})

    def test_completed_group_causality_and_bank_byte_rounding(self):
        for ratio in (1, 2, 3, 5):
            for query in range(15):
                # Enumerate original token positions, not the production formula.
                ends = list(range(ratio-1, query+1, ratio))
                self.assertEqual(m.completed_entries(query, ratio), len(ends))
                self.assertTrue(all(end <= query for end in ends))
        empty = m.cache_ledger(self.config, tokens=0)
        self.assertEqual(empty['shared_sources']['payload_bytes'], 0)
        tiny = m.cache_ledger(self.config, tokens=1, channels=1, bits=4)
        self.assertEqual(tiny['shared_sources']['payload_bytes'], 1)
        self.assertEqual(tiny['independent_per_layer']['payload_bytes'], 20)
        for query, ratio in ((-1, 2), (True, 2), (0, 0), (0, False)):
            with self.assertRaises(ValueError):
                m.completed_entries(query, ratio)

    def test_bad_schedule_contracts(self):
        for field, value in [('kv_source_layers', [8, 14, 20]),
                             ('kv_source_layers', [2, 8, 8, 20]),
                             ('index_source_layers', [2, 8, 14]),
                             ('kv_source_layers', [0, 2, 8, 14, 20]),
                             ('n_layers', True), ('compress_ratios', [0, 0])]:
            bad = copy.deepcopy(self.config); bad[field] = value
            with self.assertRaises(ValueError):
                m.csa2_schedule(bad)
        bad = copy.deepcopy(self.config); bad['compress_ratios'][3] = 1
        with self.assertRaisesRegex(ValueError, 'same compression ratio'):
            m.csa2_schedule(bad)

    def test_unshuffle_coordinate_oracle_padding_channels_and_conservation(self):
        for height, width, channels, ratio in ((6, 6, 2, 3), (4, 5, 2, 3),
                                               (1, 1, 3, 3), (3, 4, 2, 1), (4, 4, 1, 2)):
            grid = [[[1+100*y+10*x+c for c in range(channels)] for x in range(width)]
                    for y in range(height)]
            result = m.unshuffle_tokens(grid, ratio)
            self.assertEqual(result, coordinate_oracle(grid, ratio))
            original = sorted(v for row in grid for pixel in row for v in pixel)
            self.assertEqual(sorted(v for row in result for v in row if v != 0), original)
        grid = [[[100*y+10*x+c for c in range(2)] for x in range(6)] for y in range(6)]
        self.assertEqual(m.unshuffle_tokens(grid)[0],
                         [0, 10, 20, 100, 110, 120, 200, 210, 220,
                          1, 11, 21, 101, 111, 121, 201, 211, 221])
        for grid in ([], [[]], [[[1], [2]], [[3]]], [[[math.nan]]]):
            with self.assertRaises(ValueError):
                m.unshuffle_tokens(grid)

    def test_numeric_mutants_are_rejected_without_crashing(self):
        transpose = mutant('list(map(list, zip(*key_up)))', 'key_up')
        with self.assertRaises(AssertionError):
            self.assert_mla_oracle(transpose)
        position = mutant('sum(q*k for q, k in zip(query_rope, pos))) / 2\n                     for row',
                          '0) / 2\n                     for row')
        with self.assertRaises(AssertionError):
            self.assert_mla_oracle(position)
        biased = mutant('weights = [affinities[i]/denominator for i in selected]',
                        'weights = [(affinities[i]+biases[i])/sum(affinities[j]+biases[j] for j in selected) for i in selected]')
        result = biased.route_experts([.75, .5, .25], [-1., 0., 1.],
                                      [[2., 0.], [0., 4.], [6., 2.]], [1., -1.], 2)
        with self.assertRaises(AssertionError):
            self.assertAlmostEqual(result['output'][0], 3.)
        repeated = mutant("banks = [r for r in global_rows if r['mode'] == 'Full']", 'banks = global_rows')
        with self.assertRaises(AssertionError):
            self.assertEqual(repeated.cache_ledger(self.config)['shared_sources']['entries'], 25)
        future = mutant('return (query_position + 1) // ratio',
                        'return (query_position + ratio) // ratio')
        with self.assertRaises(AssertionError):
            self.assertEqual(future.completed_entries(0, 2), 0)
        channel = mutant('for c in range(channels)\n                           for dy in range(ratio) for dx in range(ratio)',
                         'for dy in range(ratio) for dx in range(ratio)\n                           for c in range(channels)')
        grid = [[[100*y+10*x+c for c in range(2)] for x in range(3)] for y in range(3)]
        changed = channel.unshuffle_tokens(grid)
        with self.assertRaises(AssertionError):
            self.assertEqual(changed, coordinate_oracle(grid, 3))


if __name__ == '__main__':
    unittest.main()
