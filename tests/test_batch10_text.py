"""Regression checks for the reader-facing P25–P28 integration."""
from pathlib import Path
import json
import math
import re
import unicodedata
import unittest

ROOT = Path(__file__).resolve().parents[1]
GOLD_MTP = ROOT / "docs" / "gold" / "p27_healthcheck_mtp.json"

# P25.9 语义锚点：同一段内的官方/练习对照，而不是冻住某次措辞。
_P25_OFFICIAL = re.compile(r"官方.{0,12}(模型卡|模板|template|解析|做法|口径)")
_P25_EXERCISE = re.compile(r"(练习|教学|本章|本书).{0,16}(选择|约定|选|口径)")
_P25_FIRST_CLOSE = re.compile(r"(首|第一个).{0,12}(闭合|</think>)")
_P25_LAST_CLOSE = re.compile(r"(最后|末|last).{0,12}(闭合|</think>)")
# 主语必须是官方做法本身；后面可叠模型卡/模板/口径等，谓词要直接贴着。
_P25_SUBJ = r"官方(?:的)?(?:(?:模型卡|chat template|模板|template|解析|做法|口径|写法|实现|规则|设计|约定|方案|示例)(?:的)?)*"
_P25_LINK = r"(?:本身|其实|确实|实际上|显然|明显|根本|完全|就|也|都|并|是|算|属于|一个|一种|个|\s)*"
_P25_NEG_INFIX = r"(?:并非|不是|不算|不代表|没有|没|无|不存在)"
# 错误谓词含 paraphrase（缺陷 / 不妥 / 失误 / 谬误 / bug …）；bug 不区分大小写。
# 「不错」「交错」「错开」「错落」不当作错误谓词。
_P25_WRONG_PRED = re.compile(
    r"存在缺陷|有缺陷|缺陷|不正确|不准确|不妥|有问题|不可靠|有误|不对|搞错了?|出错|失误|谬误|错误|错了|错的|(?<![不交])错(?![开落])|bug",
    re.I,
)
# 分句内：官方主语 + 可选连接 + 可选中缀否定 + 连接 + 错误谓词。
# 谓词必须成组，否则交替项会脱离主语单独匹配。
_P25_OFFICIAL_WRONG_CLAUSE = re.compile(
    _P25_SUBJ
    + _P25_LINK
    + r"(?P<neg>"
    + _P25_NEG_INFIX
    + r")?"
    + _P25_LINK
    + r"(?:"
    + _P25_WRONG_PRED.pattern
    + r")",
    re.I,
)
# 「官方」之前的前缀否定才豁免；「无」必须紧贴「官方」，故「毫无疑问官方…」不豁免。
_P25_PREFIX_NEG = re.compile(
    r"(?:并非|不是|不算|不代表|没有|没|无须|无需|无|不存在|不能说|不能认为|并不意味着|不意味着)(?:说|认为|是)?\s*$"
)
# NFKC 后全角逗号/分号等会变 ASCII，分句仍同时接受两种字形。
_P25_CLAUSE_SEP = re.compile(r"[，,；;：:、！!？?\n]")

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
    "官方口径是个 ＢＵＧ。",
    "官方做法存在缺陷。",
    "官方解析不准确。",
    "官方做法不妥。",
    "官方模板是一种失误。",
    "官方口径属于谬误。",
    "毫无疑问官方做法是错的。",
)
_P25_ALLOWED_NEGATIONS = (
    "官方模型卡的做法并非错误。",
    "官方口径不是错的。",
    "官方做法不算错。",
    "官方没有错。",
    "不能说官方是错的。",
    "口径不同并不意味着官方解析出错。",
    "官方模型卡写得不错。",
    "官方两处口径彼此错开。",
    "无须认为官方有问题。",
    "按官方口径实现时，手写解析容易出错。",
    "这不代表官方做法有误。",
)


def _md_paragraphs(text):
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _iter_sentences(text):
    """整节每个段落、每个以。！!？?切出的句子，以及段末没有句末标点的残句。"""
    text = unicodedata.normalize("NFKC", text)
    for para in _md_paragraphs(text):
        last = 0
        for m in re.finditer(r"[^。！!?？]*[。！!?？]", para):
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


# 已知局限：
# - 跨句绕过，如「官方做法如下。它是错的。」主语与谓词不在同一句。
# - 不写「官方」的绕过，如「模型卡的做法是错的。」
# - 主语形式超出 SUBJ，如「官方给出的做法是错的」。
# 主线同款正则也抓不住前两条。
def _unnegated_official_wrong_sentences(text):
    """各分句上：主语是官方做法本身、错误谓词未被中缀或前缀否定。"""
    text = unicodedata.normalize("NFKC", text)
    hits = []
    for sent in _iter_sentences(text):
        if "官方" not in sent:
            continue
        flagged = False
        for clause in _P25_CLAUSE_SEP.split(sent):
            for m in _P25_OFFICIAL_WRONG_CLAUSE.finditer(clause):
                if m.group("neg"):
                    continue
                # 匹配从「官方」起；其前的前缀否定才豁免。
                if _P25_PREFIX_NEG.search(clause[: m.start()]):
                    continue
                flagged = True
                break
            if flagged:
                break
        if flagged:
            hits.append(sent)
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

        # 评审字母 / 定义 / 本测试用例 / 此前 (fed6bdf) 名称：
        # A  把「官方模型卡的做法是错误的。」插入另一段（接到最后一段）  A  A
        # B  同句无关否定：对照段追加「这不是小问题：官方…错误的。」    B  B
        # C  bug 改写：首个正文段追加「官方口径是个 bug。」              C  C
        # D  去掉含「官方」的句子                                        D  D（额外）
        # E  对照段后另起一段「官方模型卡的做法错了。」                  E  新增（旧 E 现为 X1）
        # F  去掉练习句                                                  F  F（额外）
        # G  删掉对照段里含 chat template 的整句                         G  新增（旧 G 现为 H）
        # H  删掉 tokenizer_config 固定链接                              H  旧 G
        # X1 对照段 chat template → 模板                                 X1 旧 E
        # X2 固定 revision → main                                        X2 旧 H
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
                "E_standalone_wrong_paragraph",
                section.replace(contrast, contrast + "\n\n官方模型卡的做法错了。\n\n", 1),
            ),
            (
                "F_drop_exercise_sentences",
                _drop_sentences_matching(section, re.compile(r"练习|教学")),
            ),
            (
                "G_drop_chat_template_sentence",
                section.replace(
                    contrast,
                    _drop_sentences_matching(contrast, re.compile(r"chat template")),
                    1,
                ),
            ),
            (
                "H_drop_tokenizer_config_link",
                section.replace(_P25_TOKENIZER_MD_LINK, "", 1),
            ),
            (
                "X1_no_chat_template_in_contrast",
                section.replace(contrast, contrast.replace("chat template", "模板"), 1),
            ),
            (
                "X2_unpinned_revision",
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
