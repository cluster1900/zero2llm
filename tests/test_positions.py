"""P06 CPU regression: require a present, stable torch 2.14.x for tensor checks.

The dependency gate is tested without importing torch. The numerical reference
is scalar math, independent of the implementation's pairing loop. No downloads.
"""
import importlib.util
from torch_test_support import compatible_torch, torch_status, torch_reason
import io
import math
from pathlib import Path
import re
import unittest
from contextlib import redirect_stdout
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
CHAPTER = (ROOT / 'chapters/05_positional_encoding.md').read_text()


def torch_eligible(present, installed_version):
    return bool(present and compatible_torch(installed_version))


TORCH_READY, TORCH_VERSION, HAS_TORCH = torch_status()

if TORCH_READY:
    import torch

    spec = importlib.util.spec_from_file_location('positions', ROOT / 'code/05_positional_encoding.py')
    positions = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(positions)


def scalar_reference(pos, column, width):
    """Choose the paired even column from the individual column index."""
    exponent = (column - column % 2) / width
    angle = pos / (10000.0 ** exponent)
    return math.cos(angle) if column % 2 else math.sin(angle)


class PositionDependencyTests(unittest.TestCase):
    def test_presence_and_version_gate(self):
        for value in ['2.14.0', '2.14.1', '2.14.1+cpu', '2.14.10+cu130']:
            with self.subTest(version=value):
                self.assertTrue(torch_eligible(True, value))
                self.assertFalse(torch_eligible(False, value))
        for value in ['not installed', '2.13.9', '2.15.0', '12.14.1', '2.14',
                      '2.14.1rc1', '2.14.1.dev1', '2.14.1-extra']:
            with self.subTest(version=value):
                self.assertFalse(torch_eligible(True, value))

    def test_illustration_is_actual_sine_with_distinct_frequencies(self):
        # Retain the existing geometry, but verify all plotted points, not just labels.
        root = ET.parse(ROOT / 'assets/diagrams/positional_encoding_waves.svg').getroot()
        paths = root.findall('.//{http://www.w3.org/2000/svg}path')
        self.assertEqual(len(paths), 3)
        for path, denominator in zip(paths, [1, 8, 80]):
            points = re.findall(r'[ML] ([\d.]+) ([\d.]+)', path.get('d'))
            self.assertEqual(len(points), 121)
            for index, (x, y) in enumerate(points):
                pos = index / 5
                self.assertAlmostEqual(float(x), pos * 640 / 24, delta=.051)
                self.assertAlmostEqual(float(y), 40 - 34 * math.sin(pos / denominator), delta=.051)


@unittest.skipUnless(
    TORCH_READY,
    torch_reason('P06 tensor experiments', TORCH_VERSION, HAS_TORCH),
)
class PositionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(6)

    def test_formula_elementwise(self):
        # Multiple widths, including odd widths; checking D=8 alone misses odd-column bugs.
        for length, width in [(1, 1), (5, 2), (9, 5), (16, 8), (19, 16)]:
            actual = positions.generate_positional_encoding(length, width)
            self.assertEqual(tuple(actual.shape), (length, width))
            self.assertEqual(actual.dtype, torch.float32)
            self.assertEqual(actual.device.type, 'cpu')
            for pos in range(length):
                for col in range(width):
                    with self.subTest(length=length, width=width, pos=pos, col=col):
                        self.assertAlmostEqual(actual[pos, col].item(),
                                               scalar_reference(pos, col, width), delta=6e-8)

    def test_odd_last_column_and_even_norm_boundary(self):
        odd = positions.generate_positional_encoding(8, 5)
        self.assertAlmostEqual(odd[1, 4].item(), .0006309573026154199, delta=1e-10)
        self.assertGreater(torch.count_nonzero(odd[:, 4]).item(), 0)
        self.assertEqual(positions.generate_positional_encoding(3, 1)[1, 0].item(),
                         torch.tensor(math.sin(1)).item())
        self.assertFalse(torch.allclose(odd.norm(dim=-1), torch.full((8,), math.sqrt(5/2))))
        even = positions.generate_positional_encoding(8, 8)
        torch.testing.assert_close(even.norm(dim=-1), torch.full((8,), 2.))

    def test_dimension_contract(self):
        self.assertEqual(tuple(positions.generate_positional_encoding(0, 8).shape), (0, 8))
        for length, width in [(-1, 8), (2, 0), (2, -4)]:
            with self.subTest(length=length, width=width), self.assertRaises(ValueError):
                positions.generate_positional_encoding(length, width)
        for length, width in [(True, 8), (2, False), (2., 8), (2, 8.), ('2', 8)]:
            with self.subTest(length=length, width=width), self.assertRaises(TypeError):
                positions.generate_positional_encoding(length, width)

    def test_full_pe_table_matches_formula_and_script(self):
        rows = re.findall(r'^\| \*\*位置 (\d)\*\* \| (.+) \|$', CHAPTER, re.M)
        self.assertEqual(len(rows), 8, 'Keep all eight rows of table 5-2')
        generated = positions.generate_positional_encoding(8, 8)
        for pos_string, values in rows:
            pos = int(pos_string)
            cells = [value.strip().replace('**', '') for value in values.split('|')]
            self.assertEqual(len(cells), 8, 'Keep all eight columns of table 5-2')
            for col, value in enumerate(cells):
                with self.subTest(pos=pos, col=col):
                    self.assertEqual(value, f'{scalar_reference(pos, col, 8):.2f}')
                    self.assertEqual(value, f'{generated[pos, col].item():.2f}')
        # Verify the separate frequency ledger, including the finite-width endpoint.
        frequency_rows = re.findall(r'^\| d(\d)、d(\d) \| (\d) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|$', CHAPTER, re.M)
        self.assertEqual(len(frequency_rows), 4)
        for first, second, pair, denominator, frequency, period in frequency_rows:
            pair = int(pair)
            expected_denominator = 10000 ** (2 * pair / 8)
            self.assertEqual((int(first), int(second)), (2 * pair, 2 * pair + 1))
            self.assertEqual(float(denominator), expected_denominator)
            self.assertEqual(float(frequency), 1 / expected_denominator)
            self.assertEqual(period, f'{2 * math.pi * expected_denominator:.2f}')
        # Row identities must not silently repeat or skip a position.
        self.assertEqual([int(row[0]) for row in rows], list(range(8)))

    def test_cosine_table_and_nonmonotonic_counterexample(self):
        pe = positions.generate_positional_encoding(8, 8)
        actual = positions.position_cosines(pe)
        # Independent identity for even D: mean of cosines of the position differences.
        reference = [sum(math.cos(p / (10 ** i)) for i in range(4)) / 4 for p in range(8)]
        torch.testing.assert_close(actual, torch.tensor(reference), atol=1e-7, rtol=1e-6)
        rows = re.findall(r'^\| \*\*位置 0 与 位置 (\d)\*\* \| \*\*([\d.]+)\*\* \| `(█*)` \|',
                          CHAPTER, re.M)
        self.assertEqual(len(rows), 8, 'Keep all eight comparisons of table 5-3')
        self.assertEqual([int(row[0]) for row in rows], list(range(8)))
        for pos_string, value, bar in rows:
            pos = int(pos_string)
            self.assertEqual(value, f'{reference[pos]:.3f}')
            self.assertEqual(value, f'{actual[pos].item():.3f}')
            self.assertEqual(len(bar), int(reference[pos] * 25))
        self.assertGreater(actual[0].item(), actual[3].item())
        self.assertGreater(actual[6].item(), actual[3].item())  # Greater distance, higher similarity.
        self.assertFalse(torch.allclose(pe[0], pe[6]))

    def test_cosine_rejects_undefined_inputs(self):
        for matrix in [torch.empty(0, 8), torch.zeros(1, 8), torch.zeros(8),
                       torch.tensor([[float('nan'), 1.]]), torch.tensor([[float('inf'), 1.]])]:
            with self.subTest(shape=tuple(matrix.shape)), self.assertRaises(ValueError):
                positions.position_cosines(matrix)

    def test_fixed_offset_linear_relation(self):
        pe = positions.generate_positional_encoding(32, 8).double()
        for offset in [1, 3, 7]:
            for pair in range(4):
                angle = offset / (10000 ** (2 * pair / 8))
                # Column-vector convention [sin(theta), cos(theta)].
                transform = torch.tensor([[math.cos(angle), math.sin(angle)],
                                          [-math.sin(angle), math.cos(angle)]], dtype=torch.float64)
                old = pe[:-offset, 2*pair:2*pair+2]
                shifted = old @ transform.T
                torch.testing.assert_close(shifted, pe[offset:, 2*pair:2*pair+2], atol=1e-7, rtol=1e-6)

    def test_permutation_equivariance_and_fixed_position_counterexample(self):
        result = positions.permutation_example()
        order = result['order']
        torch.testing.assert_close(result['plain_swapped'], result['plain'][order])
        torch.testing.assert_close(result['plain'][1], torch.tensor([.5988879, .8022242]),
                                   atol=1e-6, rtol=1e-6)
        self.assertGreater((result['positioned_swapped'] - result['positioned'][order]).abs().max().item(), .9)
        # Fixed causal visibility breaks the unrestricted permutation equivalence.
        x = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
        mask = torch.ones(3, 3, dtype=torch.bool).tril()
        def causal(a):
            return torch.softmax((a @ a.T / math.sqrt(2)).masked_fill(~mask, -torch.inf), -1) @ a
        self.assertFalse(torch.allclose(causal(x[order]), causal(x)[order]))

    def test_addition_concat_shapes_values_and_parameter_count(self):
        x, pe, added, concatenated = positions.addition_concat_example()
        self.assertEqual(tuple(added.shape), (1, 2, 4))
        self.assertEqual(tuple(concatenated.shape), (1, 2, 8))
        expected = torch.tensor([[[.2, .9, .4, 1.],
                                  [.5+math.sin(1), .3+math.cos(1), -.2+math.sin(.01), .1+math.cos(.01)]]])
        torch.testing.assert_close(added, expected)
        torch.testing.assert_close(concatenated[..., :4], x)
        torch.testing.assert_close(concatenated[..., 4:], pe)
        self.assertEqual(torch.nn.Linear(4, 4, bias=False).weight.numel(), 16)
        self.assertEqual(torch.nn.Linear(8, 4, bias=False).weight.numel(), 32)
        # Same position rows broadcast across multiple batch elements.
        batch = x.repeat(3, 1, 1)
        torch.testing.assert_close(batch + pe, added.repeat(3, 1, 1))

    def test_generation_length_not_training_quality(self):
        result = positions.length_example()
        self.assertEqual(result['learned_parameters'], 64)
        self.assertTrue(result['table_rejects_position_8'])
        self.assertTrue(result['prefix_equal'])
        self.assertEqual(result['longer_shape'], (12, 8))
        # Formula changes with D, not with total generated sequence length.
        narrow = positions.generate_positional_encoding(8, 8)
        wider = positions.generate_positional_encoding(8, 16)
        self.assertFalse(torch.allclose(narrow[:, 2:], wider[:, 2:8]))

    def test_numeric_selftest_answers_and_printed_walkthrough(self):
        output = io.StringIO()
        with redirect_stdout(output):
            positions.main()
        output = output.getvalue()
        for snippet in [
            'd0/d1: 分母=1，周期=6.28', 'd2/d3: 分母=10，周期=62.83',
            'd4/d5: 分母=100，周期=628.32', 'd6/d7: 分母=1000，周期=6283.19',
            'cos(PE0, PE3)=0.491 < cos(PE0, PE6)=0.946',
            '固定座位编码，只换词，换序误差=0.971712',
            'Add 位置 0: [0.200, 0.900, 0.400, 1.000]',
            'Add 位置 1: [1.341, 0.840, -0.190, 1.100]',
            'Add=(1, 2, 4)；Concat=(1, 2, 8)',
            '可学习表参数=64；位置 8 越界=True',
            '正弦扩展形状=(12, 8)；前 8 行不变=True',
            '奇数宽度 d_model=5：PE[1,4]=0.000631',
        ]:
            self.assertIn(snippet, output)
        answers = CHAPTER.split('### 参考答案', 1)[1].split('## 5.10', 1)[0]
        # Verify textual answers, in addition to the numerical computations above.
        for snippet in [r'\mathbf{0.100}', '0.946 > 0.491', '[0.200, 0.900, 0.400, 1.000]',
                        '[1,2,4]', '[1,2,8]', '**16**', '**32**', r'\mathbf{0.000631}', '**64**']:
            self.assertIn(snippet, answers)
        self.assertEqual(f'{positions.generate_positional_encoding(2, 8)[1, 2].item():.3f}', '0.100')
        # ASCII is a different quantization from the rounded table; check actual rendering.
        self.assertIn('Pos  0    |  =  @  =  @  =  @  =  @ ', output)
        self.assertIn('Pos  7    |  %  %  %  %  +  @  +  @ ', output)


if __name__ == '__main__':
    unittest.main()
