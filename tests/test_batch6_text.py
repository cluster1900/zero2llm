"""P15–P17 navigation, packaging and source boundaries."""
from pathlib import Path
import re
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
LESSONS = (
    ("08e_attention_systems.md", "16_attention_systems.py", "p15-attention-systems"),
    ("08f_sparse_linear_attention.md", "17_sparse_linear_attention.py", "p16-sparse-linear"),
    ("08g_ced_architecture.md", "18_ced_architecture.py", "p17-ced"),
)

class Batch6TextTests(unittest.TestCase):
    def test_reading_order_and_source_links(self):
        names = sorted(p.name for p in (ROOT / "chapters").glob("*.md"))
        chain = ["08d_grouped_kv_mla.md", *(x[0] for x in LESSONS), "09_modern_llms.md"]
        self.assertEqual([names.index(p) for p in chain], sorted(names.index(p) for p in chain))
        for filename, code, anchor in LESSONS:
            body = (ROOT / "chapters" / filename).read_text()
            self.assertIn("{#" + anchor + "}", body.splitlines()[0])
            self.assertIn("../code/" + code, body)
            self.assertIn("参考答案", body)
            self.assertIn("2026-10-19", body)
            self.assertIn("https://", body)
            self.assertTrue((ROOT / "code" / code).is_file())
            self.assertNotRegex(body, r"\]\(\d\w*[^)]*\.md\)")

    def test_new_diagram_explanations_are_chinese(self):
        ns = {"svg": "http://www.w3.org/2000/svg"}
        for _, _, anchor in LESSONS:
            # Diagram naming uses the same stable chapter ID.
            filename = "p17-ced" if anchor == "p17-ced" else anchor
            tree = ET.parse(ROOT / "assets" / "diagrams" / (filename + ".svg"))
            for tag in ("title", "desc"):
                elements = tree.findall(".//svg:" + tag, ns)
                self.assertTrue(elements, filename + ": " + tag)
                self.assertRegex(" ".join("".join(e.itertext()) for e in elements), r"[\u4e00-\u9fff]")
            labels = ["".join(e.itertext()) for e in tree.findall(".//svg:text", ns)]
            self.assertGreaterEqual(sum(bool(re.search(r"[\u4e00-\u9fff]", s)) for s in labels), 4)

    def test_review_followups_and_current_progress(self):
        readme = (ROOT / "README.md").read_text()
        self.assertIn("pandoc-3.9-2026-02-03", readme)
        self.assertIn("20 个章节文件", readme)
        self.assertIn("18 个配套代码实验", readme)
        self.assertIn("P18–P28", readme)
        self.assertIn("第六批验证", readme)
        self.assertNotIn("固定 query 投影。 原始论文", (ROOT / "chapters/08d_grouped_kv_mla.md").read_text())
        record = (ROOT / "docs/batch5-validation.md").read_text()
        self.assertIn("35/44", record)
        self.assertIn("44/44", record)
        self.assertIn("基线", record)
