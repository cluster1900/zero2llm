"""Reading artifacts and explicit implementation boundaries for P21–P23."""
from pathlib import Path
import re
import unittest
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[1]
LESSONS = (
    ("08k_sft_lora.md", "22_sft_lora.py", "p21-sft"),
    ("08l_preference_optimization.md", "23_preference_optimization.py", "p22-preference"),
    ("08m_reasoning_mtp.md", "24_reasoning_mtp.py", "p23-mtp"),
)

class Batch8TextTests(unittest.TestCase):
    def test_reading_order_and_sources(self):
        names = sorted(p.name for p in (ROOT / "chapters").glob("*.md"))
        chain = ["08j_training_frameworks.md", *(x[0] for x in LESSONS), "09_modern_llms.md"]
        positions = [names.index(p) for p in chain]
        self.assertEqual(positions, sorted(positions))
        for name, script, anchor in LESSONS:
            body = (ROOT / "chapters" / name).read_text()
            self.assertIn("{#" + anchor + "}", body.splitlines()[0])
            self.assertIn("../code/" + script, body)
            self.assertIn("自测", body)
            self.assertIn("2026-10-19", body)
            self.assertIn("https://", body)
            self.assertTrue((ROOT / "code" / script).is_file())
            self.assertNotRegex(body, r"\]\(\d\w*[^)]*\.md\)")
            # Tests are not packaged as relative EPUB files.
            self.assertNotIn("](../tests/", body)

    def test_actual_chinese_svg_labels(self):
        ns = {"s": "http://www.w3.org/2000/svg"}
        for _, _, anchor in LESSONS:
            svg = ET.parse(ROOT / "assets/diagrams" / (anchor + ".svg"))
            for tag in ("title", "desc"):
                parts = svg.findall(".//s:" + tag, ns)
                self.assertTrue(parts)
                self.assertRegex("".join("".join(p.itertext()) for p in parts), r"[\u4e00-\u9fff]")
            labels = ["".join(x.itertext()) for x in svg.findall(".//s:text", ns)]
            self.assertGreaterEqual(sum(bool(re.search(r"[\u4e00-\u9fff]", x)) for x in labels), 4)

    def test_separate_trl_source_snapshots_are_explained(self):
        for chapter in ("08k_sft_lora.md", "08l_preference_optimization.md"):
            text = (ROOT / "chapters" / chapter).read_text()
            for version in ("0.16.1", "0.17.0", "0.18.2"):
                self.assertIn(version, text)
            self.assertIn("环境", text)
            self.assertIn("版本", text)

    def test_progress_and_supported_math_delimiters(self):
        readme = (ROOT / "README.md").read_text()
        self.assertIn("26 个章节文件", readme)
        self.assertIn("24 个配套代码实验", readme)
        self.assertIn("P24–P28", readme)
        self.assertIn("第八批验证", readme)
        for name, _, _ in LESSONS:
            body = (ROOT / "chapters" / name).read_text()
            self.assertNotIn(r"\(", body)
            self.assertNotIn(r"\[", body)
