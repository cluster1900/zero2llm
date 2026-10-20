"""P24–P26 reading contracts; runtime correctness lives in lesson tests."""
from pathlib import Path
import json
import re
import unittest
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[1]
LESSONS = (("08n_quantized_serving.md", "25_quantized_serving.py", "p24-quantization"),
           ("08o_qwen3_case.md", "26_qwen3_case.py", "p25-qwen3"),
           ("09_modern_llms.md", "27_deepseek_case.py", "p26-deepseek"))

class Batch9TextTests(unittest.TestCase):
    def test_reading_order_sources_and_math(self):
        names = sorted(p.name for p in (ROOT / "chapters").glob("*.md"))
        chain = ["08m_reasoning_mtp.md", *(x[0] for x in LESSONS), "10_appendix.md"]
        self.assertEqual([names.index(x) for x in chain], sorted(names.index(x) for x in chain))
        for name, code, anchor in LESSONS:
            body = (ROOT / "chapters" / name).read_text()
            self.assertIn("{#" + anchor + "}", body)
            self.assertIn("../code/" + code, body)
            self.assertIn("自测", body)
            self.assertIn("2026-10-19", body)
            self.assertIn("https://", body)
            self.assertNotIn("](../tests/", body)
            self.assertNotIn(r"\(", body)
            self.assertNotIn(r"\[", body)

    def test_chinese_svg_metadata_and_labels(self):
        ns = {"s": "http://www.w3.org/2000/svg"}
        for _, _, anchor in LESSONS:
            svg = ET.parse(ROOT / "assets/diagrams" / (anchor + ".svg"))
            for tag in ("title", "desc"):
                self.assertRegex("".join("".join(e.itertext()) for e in svg.findall(".//s:"+tag, ns)), r"[\u4e00-\u9fff]")
            self.assertGreaterEqual(sum(bool(re.search(r"[\u4e00-\u9fff]", "".join(e.itertext())))
                                       for e in svg.findall(".//s:text", ns)), 4)

    def test_progress_counts_match_files(self):
        readme = (ROOT / "README.md").read_text()
        self.assertEqual(len(list((ROOT / "chapters").glob("*.md"))), 29)
        self.assertEqual(len(list((ROOT / "code").glob("[0-9][0-9]_*.py"))), 28)
        self.assertIn("29 个章节文件", readme)
        self.assertIn("28 个配套代码实验", readme)
        self.assertIn("P27–P28", readme)
        self.assertIn("第九批验证", readme)
        for path in ("README.md", "chapters/00_preface.md", "docs/expansion-outline.md"):
            body = (ROOT / path).read_text()
            self.assertIn("已实现", body)
            self.assertNotIn("P24–P26 本分支", body)
            self.assertNotIn("本分支继续撰写 P24–P26", body)
            self.assertNotIn("P24–P26 为本分支已撰写的待审批次", body)

    def test_old_chapter_nine_anchors_remain_explicit(self):
        body = (ROOT / "chapters/09_modern_llms.md").read_text()
        anchors = json.loads((ROOT / "tests/fixtures/p26_legacy_anchors.json").read_text())
        self.assertEqual(len(anchors), 15)
        for anchor in anchors:
            self.assertIn("{#" + anchor + "}", body)
