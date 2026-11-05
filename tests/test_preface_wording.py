import pathlib
import re
import unittest

PREFACE = pathlib.Path(__file__).resolve().parents[1] / "chapters" / "00_preface.md"

# bot-hub #122：前言 0.5 节两处过时措辞，按字面断言不再出现。
OLD_PHRASES = (
    "这些 P 编号是扩展计划的编号，不会立即替换旧版全部章号。",
    "#28/#29 前沿更新是后续范围，不混入本轮交付。",
)


class PrefaceWordingTests(unittest.TestCase):
    def test_outdated_preface_phrases_removed(self):
        text = PREFACE.read_text(encoding="utf-8")
        for phrase in OLD_PHRASES:
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, text)


    def test_route_table_matches_chapter_titles(self):
        # bot-hub #123 B1：「P 编号 → 文件 → 章名」表逐行对照真实文件与大标题。
        # 章名规则：取文件首行，去掉行首「# 」和行尾 {#…} 锚点，其余逐字一致。
        chapters = PREFACE.parent
        rows = re.findall(r"^\| (P\d\d) \| (\S+\.md) \| (.+?) \|$", PREFACE.read_text(encoding="utf-8"), re.M)
        self.assertEqual([r[0] for r in rows], [f"P{i:02d}" for i in range(29)])
        self.assertEqual(sorted(r[1] for r in rows), sorted(p.name for p in chapters.glob("*.md")))
        for pnum, name, title in rows:
            with self.subTest(pnum=pnum, file=name):
                path = chapters / name
                self.assertTrue(path.is_file())
                first = path.read_text(encoding="utf-8").splitlines()[0]
                self.assertTrue(first.startswith("# "))
                expected = re.sub(r"\s*\{#[^}]*\}\s*$", "", first[2:]).strip()
                self.assertEqual(title, expected)
                for found in re.findall(r"（(P\d\d)）", expected):
                    self.assertEqual(found, pnum)


if __name__ == "__main__":
    unittest.main()
