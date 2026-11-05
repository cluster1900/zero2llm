import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

# README 面向读者：不出现交付账本与编校元信息。
LEDGER_PATTERNS = (
    r"本批",
    r"首批",
    r"第[一二三四五六七八九十]+批",
    r"bot-hub",
    r"/pull/\d+",
    r"PR #\d+",
    r"审核",
    r"扩展大纲",
    r"旧文件名",
)


class ReadmeWordingTests(unittest.TestCase):
    def test_no_delivery_ledger_wording(self):
        text = README.read_text(encoding="utf-8")
        for pattern in LEDGER_PATTERNS:
            with self.subTest(pattern=pattern):
                self.assertIsNone(re.search(pattern, text), pattern)

    def test_reader_sections_present_in_order(self):
        text = README.read_text(encoding="utf-8")
        heads = re.findall(r"^## (.+)$", text, re.M)
        self.assertEqual(heads[:3], ["这本书讲什么", "怎么读", "怎么跑"])

    def test_validation_record_links_cover_every_unit(self):
        text = README.read_text(encoding="utf-8")
        links = re.findall(r"^- (P\d\d–P\d\d)：\[\1 验证记录\]\(docs/batch(\d+)-validation\.md\)$", text, re.M)
        self.assertEqual([int(n) for _, n in links], list(range(1, 11)))
        spans = [tuple(int(x) for x in re.findall(r"\d+", r)) for r, _ in links]
        self.assertEqual(spans[0][0], 0)
        self.assertEqual(spans[-1][1], 28)
        for (_, end), (start, _) in zip(spans, spans[1:]):
            self.assertEqual(start, end + 1)


if __name__ == "__main__":
    unittest.main()
