import pathlib
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


if __name__ == "__main__":
    unittest.main()
