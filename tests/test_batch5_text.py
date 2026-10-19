"""Reading order and packaging metadata; numerical behavior lives in lesson tests."""
from pathlib import Path
import re
import unittest
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import build
LESSONS = [('08b_rope_yarn.md', '13_rope.py', 'p12-rope'),
           ('08c_modern_block.md', '14_modern_block.py', 'p13-modern-block'),
           ('08d_grouped_kv_mla.md', '15_grouped_kv.py', 'p14-grouped-kv')]


class Batch5TextTests(unittest.TestCase):
    def test_completed_lesson_forward_references(self):
        for filename, anchor, stale in (
            ('04_multi_head_attention.md', '#p14-grouped-kv', '变体留到后续章节'),
            ('05_positional_encoding.md', '#p12-rope', '详细实现留给大纲 P12'),
            ('06_residual_layernorm_ffn.md', '#p13-modern-block', '后续规划 P13'),
            ('10_appendix.md', '#p12-rope', '位置核心 | 第 9 章'),
            ('08a_kv_cache.md', '#p12-rope', '下一单元 P12 尚未实现')):
            body = (ROOT / 'chapters' / filename).read_text()
            self.assertIn(anchor, body)
            self.assertNotIn(stale, body)
        self.assertIn('Pandoc 3.9', (ROOT / 'README.md').read_text())

    def test_explicit_heading_ids_and_legacy_slugs(self):
        self.assertEqual(build.chapter_title_id('# 旋转位置（P12）：RoPE {#p12-rope}\n'),
                         ('旋转位置（P12）：RoPE', 'p12-rope'))
        title = '第1章：词嵌入'
        self.assertEqual(build.chapter_title_id('# ' + title), (title, build.slugify(title)))

    def test_inserted_order_and_explicit_navigation(self):
        names = sorted(p.name for p in (ROOT / 'chapters').glob('*.md'))
        chain = ['08a_kv_cache.md', *(r[0] for r in LESSONS), '09_modern_llms.md']
        positions = [names.index(name) for name in chain]
        self.assertEqual(positions, sorted(positions))
        for chapter, code, anchor in LESSONS:
            body = (ROOT / 'chapters' / chapter).read_text()
            self.assertIn('{#' + anchor + '}', body.splitlines()[0])
            self.assertIn('../code/' + code, body)
            self.assertIn('参考答案', body)
            self.assertNotRegex(body, r'\]\(\d\w*[^)]*\.md\)')
            for asset in re.findall(r'!\[[^\]]*\]\((assets/[^)]+)\)', body):
                self.assertTrue((ROOT / asset).is_file(), asset)

    def test_readme_counts_and_local_links(self):
        readme = (ROOT / 'README.md').read_text()
        chapter_count = len(list((ROOT / 'chapters').glob('*.md')))
        code_count = len(list((ROOT / 'code').glob('[0-9][0-9]_*.py')))
        self.assertIn(f'{chapter_count} 个章节文件', readme)
        self.assertIn(f'{code_count} 个配套代码实验', readme)
        for link in re.findall(r'\]\(((?:chapters|code|docs)/[^)#]+)(?:#[^)]*)?\)', readme):
            self.assertTrue((ROOT / link).is_file(), link)
        self.assertIn('P24–P28', readme)
        self.assertIn('第五批验证', readme)

    def test_scope_and_source_links(self):
        outline = (ROOT / 'docs/expansion-outline.md').read_text()
        self.assertIn('P24–P28 尚未', outline)
        for chapter, _, _ in LESSONS:
            text = (ROOT / 'chapters' / chapter).read_text()
            self.assertIn('https://', text)
            self.assertIn('2026-10-19', text)
        # This batch's planned teaching variants are not production kernels.
        self.assertIn('YaRN', (ROOT/'chapters/08b_rope_yarn.md').read_text())
        self.assertIn('MLA', (ROOT/'chapters/08d_grouped_kv_mla.md').read_text())
