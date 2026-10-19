"""Reading artifacts and explicit implementation boundaries for P18–P20."""
from pathlib import Path
import re
import unittest
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[1]
LESSONS = (
    ("08h_moe.md", "19_moe.py", "p18-moe"),
    ("08i_pretraining_pipeline.md", "20_pretraining_pipeline.py", "p19-pretraining"),
    ("08j_training_frameworks.md", "21_training_frameworks.py", "p20-frameworks"),
)

class Batch7TextTests(unittest.TestCase):
    def test_reading_order_and_sources(self):
        names = sorted(p.name for p in (ROOT / "chapters").glob("*.md"))
        chain = ["08g_ced_architecture.md", *(x[0] for x in LESSONS), "09_modern_llms.md"]
        positions = [names.index(p) for p in chain]
        self.assertEqual(positions, sorted(positions))
        for name, script, anchor in LESSONS:
            body = (ROOT / "chapters" / name).read_text()
            self.assertIn("{#" + anchor + "}", body.splitlines()[0])
            self.assertIn("../code/" + script, body)
            self.assertIn("参考答案", body)
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

    def test_progress_and_optional_framework_contract(self):
        readme = (ROOT / "README.md").read_text()
        self.assertIn("28 个章节文件", readme)
        self.assertIn("27 个配套代码实验", readme)
        self.assertIn("P27–P28", readme)
        self.assertIn("第七批验证", readme)
        self.assertTrue((ROOT / "requirements-frameworks.txt").is_file())
        chapter = (ROOT / "chapters/08j_training_frameworks.md").read_text()
        for name in ("Transformers", "TRL", "DeepSpeed", "Megatron"):
            self.assertIn(name, chapter)

    def test_math_uses_pandoc_supported_delimiters(self):
        chapter = (ROOT / "chapters/08j_training_frameworks.md").read_text()
        self.assertIn(r"其中有效集合 $\mathcal V$", chapter)
        self.assertNotIn(r"\(\mathcal V\)", chapter)
