"""Follow-up PR-B rulings (bot-hub#123): 09 empty 本章小结 anchor; 08g cache nuances; no mHC/Engram in 08g."""
import re
import unittest
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CACHE_NEW = (
    "缓存是调用者持有的快照，新请求传 None；增量调用不覆盖旧快照，可以从同一个前缀分叉实验；"
    "改权重、换配置或换模型后旧缓存会被拒绝。不要编辑缓存张量，也不要用 `.data` 或底层共享存储写入绕过版本跟踪；"
    "这里不承诺检测这些破坏。"
)
CACHE_OLD = "缓存是调用者持有的快照，新请求传 None；增量调用不覆盖旧快照，改权重、换配置或换模型后旧缓存会被拒绝。"
CLOSING = "本章完成的不是“复现一个品牌”，而是搭起一条可检查的链：**固定对象 → 找到证据 → 说明边界 → 做小实验 → 用独立 oracle 与错误变体检查**。V3、R1、蒸馏学生和 V4.1-Flash 可以放在同一张地图上，但不能被拼接成一个现实中不存在的超级架构。"


class IdCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.ids.append(attrs["id"])
        if attrs.get("href"):
            self.hrefs.append(attrs["href"])


class FollowBRulings(unittest.TestCase):
    def test_09_empty_anchor_gone_closing_kept(self):
        import json
        body = (ROOT / "chapters/09_modern_llms.md").read_text(encoding="utf-8")
        self.assertEqual(body.count("[]{#本章小结-1}"), 0)
        anchors = json.loads((ROOT / "tests/fixtures/p26_legacy_anchors.json").read_text(encoding="utf-8"))
        self.assertNotIn("本章小结-1", anchors)
        self.assertEqual(body.count(CLOSING), 1)
        # Closing still sits just before P26.12
        i = body.index(CLOSING)
        after = body[i + len(CLOSING): i + len(CLOSING) + 40]
        self.assertIn("## P26.12", after)

    def test_08g_cache_paragraph(self):
        body = (ROOT / "chapters/08g_ced_architecture.md").read_text(encoding="utf-8")
        self.assertEqual(body.count(CACHE_NEW), 1)
        self.assertEqual(body.count(CACHE_OLD), 0)
        self.assertEqual(body.count("可以从同一个前缀分叉实验"), 1)
        self.assertEqual(body.count("不承诺检测这些破坏"), 1)
        self.assertEqual(body.count("`.data`"), 1)

    def test_08g_no_mhc_engram(self):
        body = (ROOT / "chapters/08g_ced_architecture.md").read_text(encoding="utf-8")
        self.assertEqual(body.count("mHC"), 0)
        self.assertEqual(body.count("Engram"), 0)

    def test_dist_html_summary_id_unique_and_links_ok(self):
        html = (ROOT / "dist/index.html").read_text(encoding="utf-8")
        parser = IdCollector()
        parser.feed(html)
        # The auto-id from 08l's ## 本章小结 must appear exactly once.
        self.assertEqual(parser.ids.count("本章小结-1"), 1)
        # No empty-span duplicate left from 09.
        self.assertEqual(html.count('<span id="本章小结-1"></span>'), 0)
        # Internal fragment links that target this id must resolve.
        targets = {h.split("#", 1)[1] for h in parser.hrefs if "#" in h and not h.startswith("http")}
        if "本章小结-1" in targets:
            self.assertIn("本章小结-1", parser.ids)
        # All local fragment-only links resolve.
        id_set = set(parser.ids)
        for href in parser.hrefs:
            if href.startswith("#"):
                frag = href[1:]
                if frag:
                    self.assertIn(frag, id_set, href)

    def test_dist_md_no_empty_anchor(self):
        dist = (ROOT / "dist/transformer_from_scratch.md").read_text(encoding="utf-8")
        self.assertEqual(dist.count("[]{#本章小结-1}"), 0)
        self.assertEqual(dist.count(CACHE_NEW), 1)
        self.assertEqual(dist.count(CLOSING), 1)


if __name__ == "__main__":
    unittest.main()
