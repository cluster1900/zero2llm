"""B8 follow-up (Wu ruling): 01a also runs the optional tokenizer test; appendix L183/L200 point to 附录 H."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CH = ROOT / "chapters"

CHANGED = {
    "10_appendix.md": [
        ('维护记录至少包含：日期、对象/revision、主张、证据路径、验证命令、环境、通过/失败/跳过、限制、对应 PR。既往正常浏览器检查属于既往 head，不能复制到新 head。源代码、README、章节状态、HTML/EPUB 与验证报告应同批更新；旧的来源与验证记录作为历史保留。',
         '维护记录至少包含：日期、对象/revision、主张、证据路径、验证命令、环境、通过/失败/跳过、限制、对应 PR。既往正常浏览器检查属于既往 head，不能复制到新 head。源代码、README、章节状态、HTML/EPUB 与验证报告应同批更新；旧的来源与版本信息集中保留在[附录 H](#p28.h-版本与来源)。'),
        ('仓库根目录的 `python -m unittest discover -s tests -v` 检查示例与契约，`python scripts/check_book.py` 检查已生成制品的资源/链接，`npm run check:layout` 检查正常浏览器视口布局。公式正确、链接存在、图片可读与代码通过互不替代。具体通过与跳过数以当批验证记录为准，不在这里保留会迅速过时的全书总数。',
         '仓库根目录的 `python -m unittest discover -s tests -v` 检查示例与契约，`python scripts/check_book.py` 检查已生成制品的资源/链接，`npm run check:layout` 检查正常浏览器视口布局。公式正确、链接存在、图片可读与代码通过互不替代。具体通过与跳过数以当前 CI 运行结果为准，运行环境版本见[附录 H](#p28.h-版本与来源)，不在这里保留会迅速过时的全书总数。'),
    ],
}
REMOVED = {"10_appendix.md": ["当批验证记录", "旧的来源与验证记录作为历史保留"]}
CMDS_01A = [
    "python3 -m unittest discover -s tests -p 'test_foundations.py' -v",
    "python3 -m unittest discover -s tests -p 'test_optional_tokenizer.py' -v",
]


class B8FollowupTest(unittest.TestCase):
    def test_changed_whole_lines(self):
        for name, pairs in CHANGED.items():
            lines = (CH / name).read_text(encoding="utf-8").splitlines()
            for old, new in pairs:
                with self.subTest(new=new[:30]):
                    self.assertEqual(lines.count(old), 0)
                    self.assertEqual(lines.count(new), 1)

    def test_removed_phrases(self):
        for name, phrases in REMOVED.items():
            body = (CH / name).read_text(encoding="utf-8")
            for p in phrases:
                self.assertNotIn(p, body)

    def test_appendix_h_anchor_exists(self):
        body = (CH / "10_appendix.md").read_text(encoding="utf-8")
        self.assertEqual(body.count("{#p28.h-版本与来源}"), 1)
        self.assertEqual(body.count("[附录 H](#p28.h-版本与来源)"), 2)

    def test_01a_runs_both_test_files_in_order(self):
        lines = (CH / "01a_tokenizer.md").read_text(encoding="utf-8").splitlines()
        idx = [lines.index(c) for c in CMDS_01A]
        for c in CMDS_01A:
            self.assertEqual(lines.count(c), 1)
            f = re.search(r"-p '(test_\w+\.py)'", c).group(1)
            self.assertTrue((ROOT / "tests" / f).is_file())
        self.assertEqual(idx[1], idx[0] + 1)


if __name__ == "__main__":
    unittest.main()
