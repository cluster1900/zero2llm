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
# 错误谓词按最长优先；裸「错」在匹配后再排除「不错/交错/错开/错落」。
_P25_WRONG_PRED = re.compile(
    r"存在缺陷|有缺陷|缺陷|不成立|不正确|不准确|不妥|有问题|不可靠|有误|不对|搞错了?|出错|失误|谬误|错误|错了|错的|错|bug",
    re.I,
)
# 紧贴谓词的中缀否定，中间可夹 _P25_NEG_FILL。「无疑」以「疑」收尾，不会误中「无$」。
_P25_NEG_WORDS = r"并非|不是|不算|并没有|也没有|没有|没|不存在|无|不会|从不|从未|未曾|很少|并无|毫无|未必|不一定"
_P25_NEG_FILL = r"任何|什么|会|过|是"
_P25_INFIX_NEG = re.compile(rf"(?:{_P25_NEG_WORDS})(?:{_P25_NEG_FILL})?$")
# 「官方解析没有被证明是错的」：被动结构里否定不贴邻谓词，仍豁免。
_P25_PASSIVE_NEG = re.compile(r"(?:没有|并没有|未)被(?:证明|证实|认为)是?$")
# 双重否定：中缀否定之前还以否定/前缀否定收尾（「官方做法不是没有错」「官方做法不能说没有错误」）→ 双重否定，命中。
_P25_DOUBLE_NEG = re.compile(r"(?:不是|并非|并不是|不能说|不能认为|不代表|并不意味着|不意味着|不是说|不说|不认为)$")
# 「没错才怪 / 错了才怪 / 错的才怪」：谓词后紧跟才怪，仍算在说官方错。
_P25_CAI_GUAI = re.compile(r"^(?:的|了)?才怪")
# 「官方」正前方的前缀否定才豁免（不认为 覆盖 并不认为/本书不认为）。
_P25_PREFIX_NEG = re.compile(
    r"(?:不能说|不能认为|无须认为|无需认为|不代表|并不意味着|不意味着|不是说|并非说|不认为|并不是|不说)$"
)
# 「不能不认为/不能不说」是双重否定的肯定，不得当前缀豁免。
_P25_PREFIX_AFFIRM = re.compile(r"不(?:能|得)不(?:说|认为)$")
# 出错且描述读者自己的实现/计数时：谓词前的情态（就会/也会/将会 以「会」收尾）。
_P25_CHUCUO_MODAL = re.compile(r"(?:容易|会)$")
# 前缀豁免失效的分句标点（NFKC 后全角已折半角，两边都查以防漏）。
_P25_CLAUSE_BREAK = frozenset("，,；;：:")
# 只按句末标点切分；半角 '.' 仅在英文句末（字母/数字/右括号后接空格+大写）。
_P25_SENT_END = re.compile(r"[。！？!?]|(?<=[A-Za-z0-9)])\.(?=\s+[A-Z])")
_P25_BARE_CUO_LEFT = frozenset({"不错", "交错"})
_P25_BARE_CUO_RIGHT = frozenset({"错开", "错落"})

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
    "官方的最后闭合解析其实错了。",
    "官方模型卡的说明有误。",
    "官方给出的做法是错的。",
    "官方的最后闭合口径错了。",
    "官方模型卡采用最后闭合是错误的。",
    "官方的做法实在是错的。",
    "官方做法不是小事，它是错的。",
    "官方口径；错误。",
    "官方做法，其实是错的。",
    "官方做法. 它是错的.",
    "官方做法无疑是错误的。",
    "官方做法无疑有误。",
    "官方做法不是没有错。",
    "官方做法没错才怪。",
    "官方做法不是不对，而是完全错误。",
    "官方解析交错误。",
    "不能说官方做法没错。",
    "不能认为官方做法没有错误。",
    "谁说官方做法没错？",
    "官方做法难道没错？",
    "不能说官方做法完全没问题，它其实错了。",
    "不代表官方做法正确，它其实是错误的。",
    "不能说官方解析正确；它错了。",
    "官方公式不成立。",
    "官方解析并非没有错误。",
    "官方做法并非没错。",
    "按官方口径，官方解析会出错。",
    "官方解析会出错。",
    "不能不认为官方做法是错的。",
    "官方做法不能说没有错误。",
    "官方做法不是没有任何错误。",
    "这不代表官方做法没有错。",
    "不能不说官方做法是错的。",
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
    "官方做法也没有错。",
    "官方口径并没有错。",
    "不能认为官方模板有误。",
    "官方口径与本练习不同，但并非错误。",
    "官方做法没有问题。",
    "若把官方口径套到本练习，计数会出错。",
    "官方模型卡的示例不存在错误。",
    "官方解析并非不可靠。",
    "官方解析不会出错。",
    "官方解析很少出错。",
    "官方解析从未出错。",
    "官方模板不会有问题。",
    "我们并不认为官方解析是错的。",
    "官方解析不存在任何缺陷。",
    "官方口径与本练习口径不同，读者自己实现时容易出错。",
    "官方解析没有任何错误。",
    "官方做法并没有什么错误。",
    "本书不认为官方做法有误。",
    "并不是官方错了。",
    "这里不说官方做法错，只说口径不同。",
    "官方做法未必错。",
    "官方做法不一定是错的。",
    "官方解析没有被证明是错的。",
    "官方模板并无错误。",
    "官方做法不会是错的。",
)


def _md_paragraphs(text):
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _iter_sentences(text):
    """整节每个段落、每个以句末标点切出的句子，以及段末没有句末标点的残句。

    只按 。！？!? 以及英文句末半角 '.' 切分，不按逗号/分号/冒号。
    半角点须满足「字母/数字/右括号后接空格+大写」，因此 Qwen3-0.6B、URL、
    「官方做法. 它是错的.」都不会被切开。空行分段也分隔句子。
    """
    text = unicodedata.normalize("NFKC", text)
    for para in _md_paragraphs(text):
        last = 0
        for m in _P25_SENT_END.finditer(para):
            sent = para[last:m.end()].strip()
            if sent:
                yield sent
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
# - 跨句：「官方做法如下。它是错的。」主语与谓词跨句号不被关联；
# - 不写「官方」：「模型卡的做法是错的。」；
# - 英文 wrong/incorrect 不在谓词表（仅 bug）；
# - 否定正面词表达错误不被识别：「官方模型卡并非正确」「不是对的」（main 同样漏判）；
# - 分句只按句末标点，谓词必须在「官方」之后：「错的是官方做法」不被识别；同句多主语时就近取「官方」：「官方是对的，我们的改写错了」会误报（main 同样误报）；
# - 否定豁免只看紧贴谓词的否定词（可夹 任何/什么/会/过/是），不做句法分析：「官方做法也没有错到哪里去」类口语会被豁免、需靠同句其他谓词抓；
# - 反问只识别「谁说/难道」+ 紧贴否定；「谁说官方做法错了？」这类反问会误报（main 同样误报，保守方向）；
# - 「出错」不同主语规则：「官方…，…容易/会出错」逗号后的出错视为读者实现，若作者真想说官方在逗号后会出错（「官方解析，在多 </think> 时会出错」）会漏判；main 不检查「出错」，不弱于 main；
# - 引述/假设语境不识别：「说官方做法错了是一种误读」「如果你认为官方做法错了，…」会误报（main 同样误报，保守方向）；
# - main 的正则同样抓不住前两条。
def _unnegated_official_wrong_sentences(text):
    """各句上：谓词前最近的「官方」未被前缀/中缀否定，且不是「官方…，…容易/会出错」。"""
    text = unicodedata.normalize("NFKC", text)
    hits = []
    for sent in _iter_sentences(text):
        if "官方" not in sent:
            continue
        flagged = False
        for m in _P25_WRONG_PRED.finditer(sent):
            pred = m.group(0)
            i = m.start()
            # 裸「错」：整词「不错/交错/错开/错落」不当错误谓词；更长谓词从不跳过。
            if pred == "错":
                if (
                    sent[i - 1 : i + 1] in _P25_BARE_CUO_LEFT
                    or sent[i : i + 2] in _P25_BARE_CUO_RIGHT
                ):
                    continue
            pos = sent.rfind("官方", 0, i)
            if pos < 0:
                continue
            before = sent[:i]
            between = sent[pos + 2 : i]
            prefix = sent[:pos]
            # 前缀贴在最近「官方」之前，且官方到谓词之间无分句标点；「不能不」是肯定。
            prefix_binds = bool(
                _P25_PREFIX_NEG.search(prefix)
                and not _P25_PREFIX_AFFIRM.search(prefix)
                and not any(ch in between for ch in _P25_CLAUSE_BREAK)
            )
            # 中缀否定先判定（含被动「没有被证明是」）。
            # 「并非不可靠」：谓词是「不可靠」，before 以「并非」结尾，按中缀豁免，不是双重否定。
            infix_m = _P25_INFIX_NEG.search(before) or _P25_PASSIVE_NEG.search(before)
            if infix_m:
                # a 双重否定：中缀否定之前还以 不是/并非/不能说/不代表 等否定收尾 → 命中
                # b 谓词后紧跟「才怪」→ 命中
                # c 前缀否定直接贴在「官方」前 → 命中
                # d 句中有「谁说/难道」→ 命中
                # 命中时不再走前缀豁免；否则跳过该谓词。
                rhetorical = (
                    _P25_DOUBLE_NEG.search(before[: infix_m.start()])
                    or _P25_CAI_GUAI.match(sent[m.end() :])
                    or prefix_binds
                    or ("谁说" in sent)
                    or ("难道" in sent)
                )
                if rhetorical:
                    flagged = True
                    break
                continue
            # 「官方」正前方的前缀否定才豁免；分句标点后的谓词不跟前缀走。
            if prefix_binds:
                continue
            # 出错的不同主语：「官方…，…容易/会出错」→ 出错说的是读者实现/计数。
            # 例：「按官方口径实现时，手写解析容易出错」。
            # 「按官方口径，官方解析会出错」就近「官方」与出错之间无逗号，仍算命中。
            if pred == "出错" and _P25_CHUCUO_MODAL.search(before):
                if "，" in between or "," in between:
                    continue
            flagged = True
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
        # E  对照段后另起一段「官方的最后闭合解析其实错了。」            E  评审原 E
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
                section.replace(contrast, contrast + "\n\n官方的最后闭合解析其实错了。\n\n", 1),
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
