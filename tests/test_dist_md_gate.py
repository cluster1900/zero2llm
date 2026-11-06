"""Gate: dist/transformer_from_scratch.md must match chapters/ merge (bot-hub#123)."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def expected_dist_md_body():
    chapters = sorted((ROOT / "chapters").glob("*.md"))
    return "\n\n---\n\n".join(p.read_text(encoding="utf-8") for p in chapters) + "\n\n---\n\n"


class DistMdGate(unittest.TestCase):
    def test_dist_md_matches_chapters(self):
        text = (ROOT / "dist/transformer_from_scratch.md").read_text(encoding="utf-8")
        header = re.match(r"(?:%.*\n){3}\n", text)
        self.assertIsNotNone(header, "dist md missing pandoc % header")
        self.assertEqual(text[header.end():], expected_dist_md_body())

    def test_check_book_reports_dist_md_drift(self):
        from scripts.check_book import check_dist_md_matches_chapters

        self.assertIsNone(check_dist_md_matches_chapters(ROOT / "dist"))


if __name__ == "__main__":
    unittest.main()
