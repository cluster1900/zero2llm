"""Keep the new lessons' executable ledgers and literal snippets connected."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Batch4TextTests(unittest.TestCase):
    def test_reading_order_and_no_broken_chapter_file_links(self):
        chapters = sorted(p.name for p in (ROOT/'chapters').glob('*.md'))
        self.assertLess(chapters.index('07_encoder_decoder.md'), chapters.index('07a_training_loop.md'))
        self.assertLess(chapters.index('07a_training_loop.md'), chapters.index('08_baby_transformer.md'))
        self.assertLess(chapters.index('08_baby_transformer.md'), chapters.index('08a_kv_cache.md'))
        for file in ('07a_training_loop.md','08_baby_transformer.md','08a_kv_cache.md'):
            body = (ROOT/'chapters'/file).read_text()
            self.assertNotRegex(body, r'\]\(\d\w*[^)]*\.md\)')
            self.assertIn('参考答案',body)
            self.assertIn('../code/',body)

    def test_progress_and_inserted_unit_numbering(self):
        previous = (ROOT/'chapters/07_encoder_decoder.md').read_text()
        self.assertIn('下一单元 P09', previous)
        self.assertNotIn('训练闭环与评估章节将另批完成', previous)
        readme = (ROOT/'README.md').read_text()
        self.assertIn('12 个配套代码实验', readme)
        self.assertIn('[第四批记录](docs/batch4-validation.md)', readme)
        for filename, prefix in [('07a_training_loop.md', 'P09'), ('08a_kv_cache.md', 'P11')]:
            headings = re.findall(r'^## (.+)$', (ROOT/'chapters'/filename).read_text(), re.M)
            numbered = [h for h in headings if re.match(r'(?:P\d+\.\d+|\d+\.)\s', h)]
            self.assertTrue(numbered)
            self.assertTrue(all(h.startswith(prefix + '.') for h in numbered))

    def test_p10_parameter_ledger_arithmetic(self):
        rows = [57*48,128*48,2*(4*48**2+(48*192+192+192*48+48)+4*48),2*48,48*57]
        self.assertEqual(sum(rows),67872)
        body = (ROOT/'chapters/08_baby_transformer.md').read_text()
        for value in (2736,6144,9216,18672,192,56160,96,67872):
            self.assertIn(str(value),body)
        self.assertIn('3×23=69',body)
        self.assertIn('71',body)

    def test_p11_memory_ledger_arithmetic(self):
        body = (ROOT/'chapters/08a_kv_cache.md').read_text()
        for length, expected in ((5,3840),(48,36864),(128,98304)):
            self.assertEqual(2*2*1*4*length*12*4,expected)
            self.assertIn(f'| {length} |',body)
            self.assertIn(str(expected),body)
        self.assertIn('缓存不保证',body)
        self.assertIn('不是端到端服务 TTFT',body)

    def test_core_snippet_statements_are_in_source(self):
        pairs = [('07a_training_loop.md','11_training_loop.py'),
                 ('08_baby_transformer.md','07_baby_gpt.py'),
                 ('08a_kv_cache.md','12_kv_cache.py')]
        # Explicit selected source extracts; illustrative invocation blocks are not extracts.
        selected = ('optimizer.zero_grad(set_to_none=True)',
                    'x = x + self.attn(self.ln1(x))',
                    'x = x + self.mlp(self.ln2(x))',
                    'pos = torch.arange(offset, offset + T, device=idx.device)',
                    'allowed = key_positions <= query_positions')
        for chapter,source in pairs:
            body = (ROOT/'chapters'/chapter).read_text()
            code = (ROOT/'code'/source).read_text()
            for line in selected:
                if line in body:
                    self.assertIn(line,code)
