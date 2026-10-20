"""Semantic regression checks for P25 wording and validation docs."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]

class FinalBatchTextTests(unittest.TestCase):
    def test_p25_records_the_two_perspectives(self):
        body = (ROOT / "chapters/08o_qwen3_case.md").read_text()
        paragraph = re.search(r"(?is)(?:P25|闭合).{0,1800}", body)
        self.assertIsNotNone(paragraph)
        text = paragraph.group(0)
        self.assertIn("官方做法", text)
        self.assertIn("本练习", text)
        self.assertIn("并非官方做法错误", text)

    def test_validation_doc_is_reader_facing(self):
        body = (ROOT / "docs/batch10-validation.md").read_text()
        self.assertIn("复现步骤", body)
        self.assertNotIn("合并与发布记录", body)
        self.assertNotIn("tree", body)
        self.assertNotIn("首稿核验日期", body)

if __name__ == "__main__":
    unittest.main()
