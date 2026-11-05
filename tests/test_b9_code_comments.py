"""B9 follow-up: 0-based ordinals in code comments use 下标 wording."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
CHAPTERS = ROOT / "chapters"

CHANGED = {
    "code/02_manual_attention.py": [
        (
            "# 代码把词向量竖着放。新向量的第 i 个数 = 矩阵第 i 行 · 词向量。",
            "# 代码把词向量竖着放。新向量下标 i 的数 = 矩阵下标 i 的行 · 词向量。",
        ),
        (
            "# 点积计算：Q 的第 i 个向量与 K 的第 j 个向量做点积",
            "# 点积计算：Q 下标 i 的向量与 K 下标 j 的向量做点积",
        ),
    ],
    "code/05_positional_encoding.py": [
        (
            "i 是第几对维度，不是已经乘过 2 的列号：",
            "i 是维度对的下标，不是已经乘过 2 的列号：",
        ),
        (
            "d_model=5 时填充 (0,1)、(2,3)、最后的第 4 列 sine。",
            "d_model=5 时填充 (0,1)、(2,3)、最后下标 4 的列（d4）只填 sine。",
        ),
        (
            '# 向上取整，奇数宽度的最后一列也要填；k 是“第几对”。',
            "# 向上取整，奇数宽度的最后一列也要填；k 是维度对的下标（从 0 起）。",
        ),
        (
            '"""本章表格：第 0 行与各行的 cosine；输入须为非零行的二维矩阵。"""',
            '"""本章表格：下标 0 的行与各行的 cosine；输入须为非零行的二维矩阵。"""',
        ),
        (
            "table(torch.tensor([8]))  # 表只有第 0 到第 7 行。",
            "table(torch.tensor([8]))  # 表只有下标 0 到 7 的行。",
        ),
    ],
    "code/07_baby_gpt.py": [
        (
            '# 如果粘成一整条，模型会偷懒：它只记住“第 i 个座位的下一个字”，',
            '# 如果粘成一整条，模型会偷懒：它只记住“下标 i 的座位的下一个字”，',
        ),
        (
            '# 提示词“白日”一旦被放到第 0 号座位，它仍会接着背《静夜思》。',
            '# 提示词“白日”一旦被放到下标 0 的座位，它仍会接着背《静夜思》。',
        ),
    ],
    "chapters/04_multi_head_attention.md": [
        (
            "print(heads[0, 0])  # 第 0 条序列，第 0 个头",
            "print(heads[0, 0])  # 下标 0 的序列，下标 0 的头",
        ),
        (
            "print(heads[0, 1])  # 同一条序列，第 1 个头",
            "print(heads[0, 1])  # 同一条序列，下标 1 的头",
        ),
    ],
    "chapters/05_positional_encoding.md": [
        (
            '# 向上取整，奇数宽度的最后一列也要填；k 是“第几对”。',
            "# 向上取整，奇数宽度的最后一列也要填；k 是维度对的下标（从 0 起）。",
        ),
        (
            "- 外层遍历位置 0 到 $T-1$，内层遍历“第几对”。`2*k` 已经是偶数列索引，不要在指数里再多乘一次 2。",
            "- 外层遍历位置 0 到 $T-1$，内层遍历维度对的下标 k。`2*k` 已经是偶数列索引，不要在指数里再多乘一次 2。",
        ),
    ],
}


# 「第 N 列/行/对/个/维」 in code/*.py reads as an ordinal; this book's code
# indexes from 0, so such text is only allowed where N is a genuine 1-based
# count. Each allowlisted line is pinned whole and must appear exactly once.
ORDINAL_RE = re.compile(r"第\s*\d+\s*(?:列|行|对|个|维)")
ORDINAL_ALLOWLIST = {
    # [B, H, T, Dh]: num_heads is the 2nd axis counting from 1 (axis index 1).
    "code/04_multi_head_attention.py": [
        "# 为什么要换轴？因为把 num_heads 放到第 2 维后，每个头就可以独立看作一个传统的注意力！",
    ],
    # Everyday analogy: writing the 2nd character, must not peek at the 3rd.
    "code/03_scaled_dot_product.py": [
        'print("生活比喻：闭卷做题。你正在写第 2 个字，绝对不能偷看第 3 个字！")',
    ],
}


def stripped_lines(path):
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]


class TestB9CodeComments(unittest.TestCase):
    def test_changed_lines(self):
        for rel, pairs in CHANGED.items():
            lines = stripped_lines(ROOT / rel)
            for old, new in pairs:
                with self.subTest(file=rel, new=new[:40]):
                    self.assertEqual(lines.count(old), 0)
                    self.assertEqual(lines.count(new), 1)

    def test_edited_code_comments_match_chapter_copies(self):
        chapter_lines = []
        for path in sorted(CHAPTERS.glob("*.md")):
            for line in stripped_lines(path):
                chapter_lines.append((path.name, line))
        for rel, pairs in CHANGED.items():
            if not rel.startswith("code/"):
                continue
            for _old, new in pairs:
                new_s = new.strip()
                hash_at = new_s.find("#")
                if hash_at < 0:
                    continue
                comment = new_s[hash_at:]
                for name, line in chapter_lines:
                    if comment not in line:
                        continue
                    with self.subTest(code=rel, chapter=name, comment=comment[:40]):
                        self.assertEqual(line, new_s)

    def test_no_di_zero_substring_in_code(self):
        for path in sorted(CODE.glob("*.py")):
            text = path.read_text(encoding="utf-8")
            with self.subTest(file=path.name):
                self.assertNotIn("第 0 ", text)

    def test_no_unlisted_numeric_ordinals_in_code(self):
        hits = []
        for path in sorted(CODE.glob("*.py")):
            rel = str(path.relative_to(ROOT))
            allowed = ORDINAL_ALLOWLIST.get(rel, [])
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if ORDINAL_RE.search(line) and line.strip() not in allowed:
                    hits.append(f"{rel}:{lineno}:{line.strip()}")
        self.assertEqual(hits, [])

    def test_ordinal_allowlist_entries_exist(self):
        for rel, lines in ORDINAL_ALLOWLIST.items():
            stripped = stripped_lines(ROOT / rel)
            for line in lines:
                with self.subTest(file=rel, line=line[:40]):
                    self.assertEqual(stripped.count(line), 1)
                    self.assertIsNotNone(ORDINAL_RE.search(line))

    def test_no_dijidui_in_code_or_chapters(self):
        hits = []
        for folder in (CODE, CHAPTERS):
            for path in sorted(folder.glob("*")):
                if path.suffix not in {".py", ".md"}:
                    continue
                for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    if "第几对" in line:
                        hits.append(f"{path.relative_to(ROOT)}:{lineno}:{line.strip()}")
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
