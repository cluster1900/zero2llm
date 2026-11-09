"""F9 review r_0333 B1: A.1 其中 pin + no tab/control chars in chapters/*.md."""
from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAPTERS = ROOT / "chapters"
APPENDIX = CHAPTERS / "10_appendix.md"

# bot-hub#126 F9 r_0333 B1: exact 其中 line after tab fix + symbol coverage.
A1_QIZHONG = (
    "**其中（表 A-1 记号）：** $\\mathcal L$ 是损失；$z_i$ 是第 $i$ 个 logit，"
    "$z_{\\max}$ 是同一行里最大的 logit（减它只为数值稳定，不改概率）；"
    "$\\omega$ 是局部角频率，$\\lambda=2\\pi/\\lvert\\omega\\rvert$ 是对应波长；"
    "$Q$、$K$ 是查询、键矩阵，$^\\top$ 表示转置；$M$ 是加性 mask；"
    "$y$ 是正确目标的编号，$p_y$ 是模型给它的概率；"
    "$d_k$ 是单头 Q/K 宽度；$d$ 是向量长度，对 LayerNorm/RMSNorm 即模型宽度 $D$。"
    "**白话：** 这张卡是翻书时的速查，字母已与 A.5 对齐；细节推导仍回各章。"
)

# No NUL–BS, TAB, or VT–US (allow LF/CR via splitlines; no form-feed etc.).
CTRL = re.compile(r"[\x00-\x08\x09\x0b-\x1f]")


class F9AppendixControlChars(unittest.TestCase):
    def test_a1_qizhong_line_exact(self):
        text = APPENDIX.read_text(encoding="utf-8")
        self.assertEqual(text.count(A1_QIZHONG), 1)
        self.assertNotIn("	", text)

    def test_chapters_md_have_no_tab_or_control_chars(self):
        bad = []
        for path in sorted(CHAPTERS.glob("*.md")):
            raw = path.read_text(encoding="utf-8")
            for i, line in enumerate(raw.splitlines(), 1):
                for m in CTRL.finditer(line):
                    bad.append((path.name, i, hex(ord(m.group())), line[max(0, m.start() - 12) : m.start() + 12]))
        self.assertEqual(bad, [], msg=f"control chars in chapters: {bad[:20]}")


if __name__ == "__main__":
    unittest.main()
