import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAPTER = ROOT / 'chapters/09a_minigpt_healthcheck.md'
SOURCE = ROOT / 'code/28_minigpt_healthcheck.py'
MARKER = re.compile(r'<!-- code-excerpt: (\S+) L(\d+)-L(\d+) -->\n```python\n(.*?)\n```', re.S)


def sections(text):
    parts = re.split(r'^## (P27\.\d+)', text, flags=re.M)
    return dict(zip(parts[1::2], parts[2::2]))


class P27CodeExcerptTests(unittest.TestCase):
    def setUp(self):
        self.text = CHAPTER.read_text(encoding='utf-8')
        self.source = SOURCE.read_text(encoding='utf-8').splitlines()

    def test_excerpts_are_verbatim_source_lines(self):
        found = MARKER.findall(self.text)
        self.assertGreaterEqual(len(found), 5)
        for path, start, end, body in found:
            self.assertEqual(path, 'code/28_minigpt_healthcheck.py')
            self.assertEqual(body, '\n'.join(self.source[int(start) - 1:int(end)]), (start, end))

    def test_each_required_section_has_10_to_20_real_lines_explained(self):
        for name, anchor in (('P27.5', 'def cache_equivalence'), ('P27.6', 'torch.load(source'),
                             ('P27.8', 'source.count(old) != 1')):
            body = sections(self.text)[name]
            blocks = MARKER.findall(body)
            self.assertTrue(blocks, name)
            longest = max(int(e) - int(s) + 1 for _, s, e, _ in blocks)
            self.assertTrue(10 <= longest <= 20, (name, longest))
            self.assertIn(anchor, ''.join(b for *_, b in blocks))
            for _, s, e, _ in blocks:
                # every excerpt is followed by explanations citing its own line numbers
                tail = body[body.index(f'L{s}-L{e} -->'):].split('```', 2)[2]
                bullets = re.match(r'\s*(?:逐行读：\s*)?((?:- 第 .*\n)+)', tail)
                self.assertIsNotNone(bullets, (name, s))
                cited = [int(n) for n in re.findall(r'(\d+)(?=[–\s]*行|–)', bullets.group(1))]
                self.assertGreaterEqual(len(re.findall(r'^- 第', bullets.group(1), re.M)), 3, (name, s))
                self.assertTrue(cited and all(int(s) <= n <= int(e) for n in cited), (name, s, cited))

    def test_sync_script_reports_no_drift_and_detects_drift(self):
        ok = subprocess.run([sys.executable, 'scripts/sync_code_excerpts.py', '--check'], cwd=ROOT)
        self.assertEqual(ok.returncode, 0)
        sys.path.insert(0, str(ROOT / 'scripts'))
        import sync_code_excerpts
        stale = self.text.replace('def cache_equivalence', 'def cache_equivalenc', 1)
        self.assertNotEqual(sync_code_excerpts.render(stale), stale)
        self.assertEqual(sync_code_excerpts.render(self.text), self.text)

    def test_self_test_drops_questions_repeated_from_earlier_chapters(self):
        quiz = sections(self.text)['P27.13']
        self.assertNotIn('直接 tril 错了却仍可能通过因果测试', quiz)
        self.assertNotIn('为何负载总数应为 32', quiz)
        self.assertNotIn('都调用了错误的 RMSNorm', quiz)
        questions, answers = quiz.split('### 参考答案')
        count = lambda s: len(re.findall(r'^\d+\. ', s, re.M))
        self.assertEqual(count(questions), count(answers))
        self.assertEqual(re.findall(r'^(\d+)\. ', questions, re.M), [str(i) for i in range(1, count(questions) + 1)])


if __name__ == '__main__':
    unittest.main()
