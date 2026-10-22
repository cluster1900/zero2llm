"""Regression checks for the reader-facing P25-P28 integration."""
from pathlib import Path
import json
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FinalBatchTextTests(unittest.TestCase):
    def test_reading_order_and_readme_links(self):
        names = sorted(p.name for p in (ROOT / "chapters").glob("*.md"))
        self.assertEqual(names[-3:], ["09_modern_llms.md", "09a_minigpt_healthcheck.md", "10_appendix.md"])
        self.assertEqual(len(names), 29)
        self.assertEqual(len(list((ROOT / "code").glob("[0-9][0-9]_*.py"))), 28)
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for marker in ("29 个章节文件", "28 个配套代码实验", "P27–P28", "第十批验证"):
            self.assertIn(marker, readme)
        for link in re.findall(r"\]\(((?:chapters|code|docs)/[^)#]+)(?:#[^)]*)?\)", readme):
            self.assertTrue((ROOT / link).is_file(), link)

    def test_p25_closure_contract_is_semantic_and_mutation_sensitive(self):
        body = (ROOT / "chapters" / "08o_qwen3_case.md").read_text(encoding="utf-8")
        match = re.search(r"(?ms)^## P25\.9 .*?\{#p25\.9-thinking-budget\}.*?(?=^## P25\.10 )", body)
        self.assertIsNotNone(match)
        section = match.group(0)

        def assert_contract(text):
            self.assertIn("首闭合", text)
            self.assertIn("最后一个闭合", text)
            self.assertIn("模型卡", text)
            self.assertIn("chat template", text)
            self.assertIn("https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/README.md", text)
            self.assertIn("https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer_config.json", text)
            for pattern in (
                r"官方[^。\n]{0,18}(错误|不正确|有误|错了)",
                r"官方[^。\n]{0,18}做法[^。\n]{0,8}(错|错误)",
            ):
                self.assertNotRegex(text, pattern)

        assert_contract(section)
        removed_comparison = section.replace("模型卡", "", 1).replace("chat template", "", 1)
        with self.assertRaises(AssertionError):
            assert_contract(removed_comparison)
        official_error = section.replace("官方模型卡采用最后一个闭合位置", "官方模型卡采用错误位置", 1)
        with self.assertRaises(AssertionError):
            assert_contract(official_error)

    def test_neutral_completion_not_review_or_production_claim(self):
        for path in ("README.md", "chapters/00_preface.md", "docs/expansion-outline.md"):
            body = (ROOT / path).read_text(encoding="utf-8")
            self.assertIn("已实现", body)
            self.assertNotIn("P27–P28 尚未", body)
            self.assertNotIn("P28 新附录仍按批次推进", body)
            for stale in ("尚未完成的扩展", "尚未全部按新大纲改写",
                          "不是已完成全部 29 个单元的新书", "每批审核后才继续"):
                self.assertNotIn(stale, body)
        for path in ("README.md", "chapters/00_preface.md"):
            self.assertIn("原大纲 29 个单元的内容已实现", (ROOT / path).read_text(encoding="utf-8"))
        self.assertNotIn("本次 PR", (ROOT / "chapters/09a_minigpt_healthcheck.md").read_text(encoding="utf-8"))
        self.assertIn("不等于所有生产技术", (ROOT / "README.md").read_text(encoding="utf-8"))

    def test_old_appendix_heading_aliases_not_lost(self):
        body = (ROOT / "chapters/10_appendix.md").read_text(encoding="utf-8")
        aliases = json.loads((ROOT / "tests/fixtures/p28_legacy_anchors.json").read_text(encoding="utf-8"))
        for alias in aliases:
            self.assertIn("{#" + alias + "}", body)
        self.assertIn("{#p28-reference}", body)

    def test_validation_doc_is_reader_facing_and_has_pinned_values(self):
        body = (ROOT / "docs/batch10-validation.md").read_text(encoding="utf-8")
        self.assertIn("复现步骤", body)
        self.assertIn("S0–S5", body)
        self.assertIn("2048 bytes", body)
        self.assertIn("4096 bytes", body)
        self.assertIn("label_counts=[7, 3]", body)
        self.assertIn("cache_causal.svg", body)
        self.assertIn("DATA_SOURCE=code/28_minigpt_healthcheck.py --json", body)
        for process_word in ("Normal Work", "本次 PR", "审核 head", "等待审核"):
            self.assertNotIn(process_word, body)

    def test_exact_expected_commands_and_pinned_baseline(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("python code/28_minigpt_healthcheck.py --json", readme)
        self.assertIn("python code/28_minigpt_healthcheck.py --mtp --json", readme)
        self.assertIn("69e5c87", (ROOT / "docs/batch10-validation.md").read_text(encoding="utf-8"))
        for file in ("test_minigpt_healthcheck.py", "test_reference_appendix.py"):
            self.assertTrue((ROOT / "tests" / file).is_file())


if __name__ == "__main__":
    unittest.main()
