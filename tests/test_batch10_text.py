"""Regression checks for the reader-facing P25–P28 integration."""
from pathlib import Path
import json
import math
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
GOLD_MTP = ROOT / "docs" / "gold" / "p27_healthcheck_mtp.json"

# P25.9 语义锚点：同一段内的官方/练习对照，而不是冻住某次措辞。
_P25_OFFICIAL = re.compile(r"官方.{0,12}(模型卡|模板|template|解析|做法|口径)")
_P25_EXERCISE = re.compile(r"(练习|教学|本章|本书).{0,16}(选择|约定|选|口径)")
_P25_FIRST_CLOSE = re.compile(r"(首|第一个).{0,12}(闭合|</think>)")
_P25_LAST_CLOSE = re.compile(r"(最后|末|last).{0,12}(闭合|</think>)")
# 错误谓词含 paraphrase（缺陷 / 搞错 / 错的 / bug …）；bug 不区分大小写。
_P25_WRONG_PRED = re.compile(
    r"不正确|有缺陷|有问题|不可靠|有误|不对|搞错了?|出错|错误|错了|错的|错|bug",
    re.I,
)
# 只有直接修饰错误谓词的否定才豁免，例如「并非错误」「不是错的」「不算错」。
_P25_DIRECT_NEG = re.compile(r"(?:并非|不是|不算|不代表)(?:是)?\s*$")

_P25_PINNED_REV = "c1899de289a04d12100db370d81485cdf75e47ca"
_P25_PINNED_README = (
    f"https://huggingface.co/Qwen/Qwen3-0.6B/blob/{_P25_PINNED_REV}/README.md"
)
_P25_PINNED_TOKENIZER = (
    f"https://huggingface.co/Qwen/Qwen3-0.6B/blob/{_P25_PINNED_REV}/tokenizer_config.json"
)
_P25_TOKENIZER_MD_LINK = f"；[固定模板]({_P25_PINNED_TOKENIZER})"

_P25_WRONG_PARAPHRASES = (
    "官方口径是个 bug。",
    "官方模板有缺陷。",
    "官方做法搞错了。",
    "官方解析有误。",
    "官方模型卡不正确。",
    "官方的写法是错的。",
    "官方做法错。",
)
_P25_ALLOWED_NEGATIONS = (
    "官方模型卡的做法并非错误。",
    "官方口径不是错的。",
    "官方做法不算错。",
)


def _md_paragraphs(text):
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _iter_sentences(text):
    """整节每个段落、每个以。切出的句子，以及段末没有。的残句。"""
    for para in _md_paragraphs(text):
        last = 0
        for m in re.finditer(r"[^。]*。", para):
            yield m.group(0)
            last = m.end()
        tail = para[last:].strip()
        if tail:
            yield tail


def _drop_sentences_matching(text, pattern):
    """按。去掉匹配句，避免 replace(精确子串) 静默空操作。"""
    kept, last = [], 0
    for m in re.finditer(r"[^。]*。", text):
        if not pattern.search(m.group(0)):
            kept.append(m.group(0))
        last = m.end()
    tail = text[last:]
    if tail and not pattern.search(tail):
        kept.append(tail)
    return "".join(kept)


def _unnegated_official_wrong_sentences(text):
    """含「官方」且错误谓词未被直接否定的句子（扫描整节每个段落/句子）。"""
    hits = []
    for sent in _iter_sentences(text):
        if "官方" not in sent:
            continue
        for m in _P25_WRONG_PRED.finditer(sent):
            if _P25_DIRECT_NEG.search(sent[: m.start()]):
                continue
            hits.append(sent)
            break
    return hits


def _contrast_paragraphs(text):
    return [
        p for p in _md_paragraphs(text)
        if _P25_OFFICIAL.search(p) and _P25_EXERCISE.search(p)
    ]


def _comparison_paragraph(text):
    """对照官方解析与练习选择、并同时点名首闭合与最后闭合的那段。"""
    for para in _contrast_paragraphs(text):
        if _P25_FIRST_CLOSE.search(para) and _P25_LAST_CLOSE.search(para):
            return para
    return None


def _first_body_paragraph(text):
    for para in _md_paragraphs(text):
        if not para.lstrip().startswith("#"):
            return para
    return None


def _append_to_paragraph(section, paragraph, extra):
    return section.replace(paragraph, paragraph + extra, 1)


class FinalBatchTextTests(unittest.TestCase):
    def test_reading_order_and_readme_links(self):
        names = sorted(p.name for p in (ROOT / "chapters").glob("*.md"))
        self.assertEqual(names[-3:], ["09_modern_llms.md", "09a_minigpt_healthcheck.md", "10_appendix.md"])
        self.assertEqual(len(names), 29)
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        canonical_scripts = set(re.findall(r"\]\((code/[0-9][0-9]_[^)]+\.py)\)", readme))
        self.assertEqual(len(canonical_scripts), 28)
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
            paras = _contrast_paragraphs(text)
            self.assertTrue(paras, "P25.9 缺少同一段内的官方/练习对照")
            hit = _comparison_paragraph(text)
            self.assertIsNotNone(hit, "对照段未同时提到首闭合与最后闭合")
            self.assertIn("chat template", hit)
            self.assertIn("模型卡", hit)
            self.assertIn(_P25_PINNED_README, text)
            self.assertIn(_P25_PINNED_TOKENIZER, text)
            leftover = _unnegated_official_wrong_sentences(text)
            self.assertFalse(leftover, leftover)

        assert_contract(section)

        contrast = _comparison_paragraph(section)
        first_body = _first_body_paragraph(section)
        last_para = _md_paragraphs(section)[-1]
        self.assertIsNotNone(contrast)
        self.assertIsNotNone(first_body)
        # 最后一段应是 generation-config 段，而不是对照段。
        self.assertNotEqual(last_para, contrast)

        mutations = (
            (
                "A_wrong_claim_outside_contrast_paragraph",
                _append_to_paragraph(section, last_para, "官方模型卡的做法是错误的。"),
            ),
            (
                "B_unrelated_negation_same_sentence",
                _append_to_paragraph(
                    section, contrast, "这不是小问题：官方模型卡的做法是错误的。"
                ),
            ),
            (
                "C_paraphrase_bug",
                _append_to_paragraph(section, first_body, "官方口径是个 bug。"),
            ),
            (
                "D_drop_official_sentences",
                _drop_sentences_matching(section, re.compile(r"官方")),
            ),
            (
                "E_no_chat_template_in_contrast",
                section.replace(contrast, contrast.replace("chat template", "模板"), 1),
            ),
            (
                "F_drop_exercise_sentences",
                _drop_sentences_matching(section, re.compile(r"练习|教学")),
            ),
            (
                "G_drop_tokenizer_config_link",
                section.replace(_P25_TOKENIZER_MD_LINK, "", 1),
            ),
            (
                "H_unpinned_revision",
                section.replace(_P25_PINNED_REV, "main"),
            ),
        )
        for name, mutated in mutations:
            with self.subTest(mutation=name):
                self.assertNotEqual(mutated, section)
                with self.assertRaises(AssertionError):
                    assert_contract(mutated)

        for phrase in _P25_WRONG_PARAPHRASES:
            with self.subTest(paraphrase=phrase):
                mutated = section + phrase
                self.assertNotEqual(mutated, section)
                with self.assertRaises(AssertionError):
                    assert_contract(mutated)

        for phrase in _P25_ALLOWED_NEGATIONS:
            with self.subTest(allowed=phrase):
                assert_contract(section + phrase)

    def test_neutral_completion_not_review_or_production_claim(self):
        for path in ("README.md", "chapters/00_preface.md", "docs/expansion-outline.md"):
            body = (ROOT / path).read_text(encoding="utf-8")
            self.assertIn("已实现", body)
            self.assertNotIn("P27–P28 尚未", body)
            self.assertNotIn("P28 新附录仍按批次推进", body)
            for stale in (
                "尚未完成的扩展",
                "尚未全部按新大纲改写",
                "不是已完成全部 29 个单元的新书",
                "每批审核后才继续",
            ):
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

    def test_validation_doc_binds_gold_magnitudes(self):
        body = (ROOT / "docs/batch10-validation.md").read_text(encoding="utf-8")
        report = json.loads(GOLD_MTP.read_text(encoding="utf-8"))
        self.assertIn("复现步骤", body)
        self.assertIn("docs/gold/p27_healthcheck_mtp.json", body)
        self.assertIn("cache_causal.svg", body)
        self.assertIn("DATA_SOURCE=code/28_minigpt_healthcheck.py --json", body)
        self.assertIn("proposed=7", body)
        self.assertIn("accepted=3", body)
        self.assertIn("label_counts=[14, 12, 10]", body)
        self.assertNotIn("label_counts=[7, 3]", body)
        for process_word in ("Normal Work", "本次 PR", "审核 head", "等待审核"):
            self.assertNotIn(process_word, body)
        def assert_all_magnitudes(text):
            for row in report["stages"]:
                if "cache_error" in row:
                    self.assertIn(f"{row['cache_error']}", text)
                self.assertIn(f"{row['causal_error']}", text)
            for row in report["negatives"]:
                self.assertIn(row["name"], text)
                self.assertIn(f"{row['max_logit_error']}", text)

        assert_all_magnitudes(body)
        # Unique S2 cache_error — remove every occurrence and re-assert.
        unique = f"{report['stages'][2]['cache_error']}"
        self.assertEqual(body.count(unique), 1)
        missing = body.replace(unique, "MISSING")
        with self.assertRaises(AssertionError):
            assert_all_magnitudes(missing)

    def test_gold_json_matches_validation_table_contract(self):
        report = json.loads(GOLD_MTP.read_text(encoding="utf-8"))
        self.assertEqual([row["stage"] for row in report["stages"]], [f"S{i}" for i in range(7)])
        self.assertEqual(report["mtp"]["proposed"], 7)
        self.assertEqual(report["mtp"]["accepted"], 3)
        self.assertEqual(report["mtp"]["label_counts"], [14, 12, 10])
        self.assertIs(report["mtp"]["greedy_equal"], True)
        self.assertEqual(report["environment"]["atol"], 3e-6)
        self.assertEqual(report["environment"]["rtol"], 1e-5)
        for row in report["stages"]:
            self.assertTrue(math.isfinite(float(row["causal_error"])))
            self.assertLessEqual(float(row["causal_error"]), 3e-6)
            if "cache_error" in row:
                self.assertLessEqual(float(row["cache_error"]), 3e-6)

    def test_exact_expected_commands_and_pinned_baseline(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("python code/28_minigpt_healthcheck.py --json", readme)
        self.assertIn("python code/28_minigpt_healthcheck.py --mtp --json", readme)
        self.assertIn("69e5c87", (ROOT / "docs/batch10-validation.md").read_text(encoding="utf-8"))
        for file in ("test_minigpt_healthcheck.py", "test_reference_appendix.py"):
            self.assertTrue((ROOT / "tests" / file).is_file())


if __name__ == "__main__":
    unittest.main()
