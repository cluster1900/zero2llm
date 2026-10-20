"""Final approved outline integration: truthful coverage and stable links."""
from pathlib import Path
import json
import re
import unittest

ROOT=Path(__file__).resolve().parents[1]

class FinalBatchTextTests(unittest.TestCase):
    def test_reading_order_and_readme_links(self):
        names=sorted(p.name for p in (ROOT/'chapters').glob('*.md'))
        self.assertEqual(names[-3:], ['09_modern_llms.md','09a_minigpt_healthcheck.md','10_appendix.md'])
        self.assertEqual(len(names),29)
        self.assertEqual(len(list((ROOT/'code').glob('[0-9][0-9]_*.py'))),28)
        readme=(ROOT/'README.md').read_text()
        for marker in ('29 个章节文件','28 个配套代码实验','P27–P28','第十批验证'):
            self.assertIn(marker,readme)
        for link in re.findall(r'\]\(((?:chapters|code|docs)/[^)#]+)(?:#[^)]*)?\)',readme):
            self.assertTrue((ROOT/link).is_file(),link)

    def test_neutral_completion_not_review_or_production_claim(self):
        for path in ('README.md','chapters/00_preface.md','docs/expansion-outline.md'):
            body=(ROOT/path).read_text()
            self.assertIn('已实现',body)
            self.assertNotIn('P27–P28 尚未',body)
            self.assertNotIn('P28 新附录仍按批次推进',body)
            for stale in ('尚未完成的扩展', '尚未全部按新大纲改写',
                          '不是已完成全部 29 个单元的新书', '每批审核后才继续'):
                self.assertNotIn(stale,body)
        for path in ('README.md','chapters/00_preface.md'):
            self.assertIn('原大纲 29 个单元的内容已实现',(ROOT/path).read_text())
        self.assertNotIn('本次 PR', (ROOT/'chapters/09a_minigpt_healthcheck.md').read_text())
        self.assertIn('不等于所有生产技术', (ROOT/'README.md').read_text())

    def test_old_appendix_heading_aliases_not_lost(self):
        body=(ROOT/'chapters/10_appendix.md').read_text()
        aliases=json.loads((ROOT/'tests/fixtures/p28_legacy_anchors.json').read_text())
        for alias in aliases:self.assertIn('{#'+alias+'}',body)
        self.assertIn('{#p28-reference}',body)

    def test_exact_expected_commands_and_pinned_baseline(self):
        readme=(ROOT/'README.md').read_text()
        self.assertIn('python code/28_minigpt_healthcheck.py --json',readme)
        self.assertIn('python code/28_minigpt_healthcheck.py --mtp --json',readme)
        self.assertIn('69e5c87',(ROOT/'docs/batch10-validation.md').read_text())
        for file in ('test_minigpt_healthcheck.py','test_reference_appendix.py'):
            self.assertTrue((ROOT/'tests'/file).is_file())

if __name__=='__main__':unittest.main()
