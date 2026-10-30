"""Regression checks for the reader-facing P25–P28 integration."""
from pathlib import Path
import json
import math
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
GOLD_MTP = ROOT / "docs" / "gold" / "p27_healthcheck_mtp.json"

# P25.9 语义锚点：同一段内的官方/练习对照，而不是冻住某次措辞。
_P25_OFFICIAL = re.compile(r"官方.{0,12}(模型卡|模板|template|解析|做法|口径)")
_P25_EXERCISE = re.compile(r"(练习|教学|本章|本书).{0,16}(选择|约定|选|口径)")
_P25_FIRST_CLOSE = re.compile(r"(首|第一个).{0,12}(闭合|</think>)")
_P25_LAST_CLOSE = re.compile(r"(最后|末|last).{0,12}(闭合|</think>)")

# 已知局限（与 main 完全相同，本 PR 不再尝试修补）
# - 误报（main 会拒绝的正当表述）：带否定的 「官方做法并非错误」「官方做法不算错」
#   「这不代表官方做法有误」「官方做法也没有错」「官方口径与本练习不同，但并非错误」
#   「官方做法未必错」「官方解析从未被证明有误」；引述/假设/反问「“官方做法有误”是一种误解」
#   「如果你认为官方做法错了，…」「谁说官方做法错了？」；换主语「官方是对的，我们的改写错了」；
#   跨逗号的「官方做法，其实，没有错」；「官方做法不错」；以及中性叙述「官方文档列出了常见的错误用法」。
# - 漏报（main 不会抓的错误断言）：bug/缺陷/不成立/不准确/不妥/不对/失误/谬误/会出错/是错的
#   （不含「错了/错误/有误/不正确」时，且不含「做法」），如「官方口径是个 bug」「官方模板有缺陷」
#   「官方公式不成立」「官方解析会出错」「官方的写法是错的」；全角 ＢＵＧ；「官方」与谓词隔开超过
#   18 字或被 。/换行 隔开（「官方做法\n是错的」、跨句「官方做法如下。它是错的。」）；谓词前置
#   「错的是官方做法」；不写「官方」；英文 wrong/incorrect。
# - 第 1–6 轮曾用否定词表/豁免/补充规则去修这些，review 证明每次都引入新的漏报或误报，
#   第 7 轮起删除，按 #13 第 3 项允许的方式「靠人工审核把关」。

# mirrors main: origin/main tests/test_batch10_text.py assert_contract 的两条 assertNotRegex，逐字复制，勿改。
_P25_MAIN_PATTERNS = (
    r"官方[^。\n]{0,18}(错误|不正确|有误|错了)",
    r"官方[^。\n]{0,18}做法[^。\n]{0,8}(错|错误)",
)


def _p25_segments(text):
    """main 的切分：两条正则都只用 [^。\\n]，所以命中只可能落在 re.split(r"[。\\n]", text) 的一个片段内。"""
    return re.split(r"[。\n]", text)


def _p25_wrong_claim_segments(text):
    """与 main 的 assertNotRegex 完全等价：返回命中 main 任一正则的片段；没有豁免、没有补充规则。"""
    return [seg for seg in _p25_segments(text) if any(re.search(p, seg) for p in _P25_MAIN_PATTERNS)]


_P25_PINNED_REV = "c1899de289a04d12100db370d81485cdf75e47ca"
_P25_PINNED_README = (
    f"https://huggingface.co/Qwen/Qwen3-0.6B/blob/{_P25_PINNED_REV}/README.md"
)
_P25_PINNED_TOKENIZER = (
    f"https://huggingface.co/Qwen/Qwen3-0.6B/blob/{_P25_PINNED_REV}/tokenizer_config.json"
)

# review 给出的必须失败句中、main 本身就能抓住的代表（第 4–6 轮回归 + 各豁免类一例）；全部由 main 的正则拦截
_P25_MUST_FAIL = (
    "官方解析有误。",
    "官方模型卡不正确。",
    "官方做法搞错了。",
    "毫无疑问官方做法是错的。",
    "官方的最后闭合解析其实错了。",
    "不能说官方做法没错。",
    "谁说官方做法没错？",
    "官方做法没错。\n\n才怪。",
    "官方做法哪里没错？",
    "官方解析哪里没错了？",
    "官方做法怎么可能没错？",
    "官方做法不可能没错。",
    "官方解析不可能没有错误。",
    "很难说官方做法没错。",
    "很难说官方解析没有错误。",
    "官方做法未必没错。",
    "没有人不认为官方做法是错的。",
    "很难不认为官方做法有误。",
    "我们无法不认为官方做法错了。",
    "官方做法没错？才怪。",
    "官方做法也没有错到哪里去。",
    "官方解析不是错误，是漏洞。",
    "官方做法不算错，简直荒唐。",
    "这不代表官方有误？我们不同意，它确实漏掉了最后一段。",
    "官方做法没有错，这是假话。",
    "官方口径与本练习不同，但并非错误——而是在多闭合输入上直接失效。",
    "官方做法未必错，但一定会漏掉最后一段思考。",
    "官方解析从未被证明有误？现在被证明了。",
)


def _md_paragraphs(text):
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


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


def _drop_sentences_overlapping(text, pattern):
    """在全文上找 pattern 的匹配区间，删掉与任一区间重叠的句子（按 。/\n 切分，含结尾符）。

    匹配可以跨 。（如 _P25_EXERCISE 的 .{0,16}），逐句判断会漏删；重复直到不再匹配（至多 20 轮）。
    """
    for _ in range(20):
        matches = []
        for m in pattern.finditer(text):
            ms, me = m.start(), m.end()
            if me == ms:
                me = ms + 1
            matches.append((ms, me))
        if not matches:
            return text
        pieces = []
        start = 0
        for i, ch in enumerate(text):
            if ch in "。\n":
                pieces.append((start, i + 1))
                start = i + 1
        if start < len(text):
            pieces.append((start, len(text)))
        kept = []
        for ps, pe in pieces:
            if any(ps < me and ms < pe for ms, me in matches):
                continue
            kept.append(text[ps:pe])
        new_text = "".join(kept)
        if new_text == text:
            return text
        text = new_text
    return text


def _drop_exercise_sentences(text):
    return _drop_sentences_overlapping(text, _P25_EXERCISE)


_P25_TOKENIZER_LINK_RE = re.compile(
    r"\[[^\]]*\]\(" + re.escape(_P25_PINNED_TOKENIZER) + r"\)"
)


def _drop_tokenizer_link(text):
    # 按 URL 删：先删 markdown 链接（任意链接文字），再删残留的裸 URL
    return _P25_TOKENIZER_LINK_RE.sub("", text).replace(_P25_PINNED_TOKENIZER, "")


def _contrast_paragraphs(text):
    return [
        p for p in _md_paragraphs(text)
        if _P25_OFFICIAL.search(p) and _P25_EXERCISE.search(p)
    ]


def _is_comparison_paragraph(p):
    return bool(
        _P25_OFFICIAL.search(p)
        and _P25_EXERCISE.search(p)
        and _P25_FIRST_CLOSE.search(p)
        and _P25_LAST_CLOSE.search(p)
        and "chat template" in p
        and "模型卡" in p
    )


def _comparison_paragraphs(text):
    return [p for p in _md_paragraphs(text) if _is_comparison_paragraph(p)]


def _comparison_paragraph(text):
    """满足官方/练习对照、首闭合、最后闭合、chat template 与模型卡全部同段条件的第一段。"""
    paras = _comparison_paragraphs(text)
    return paras[0] if paras else None


def _map_comparison_paragraphs(section, fn):
    """对每个满足对照段条件的段落按位置单独应用 fn，重复段落各处理一次。"""
    parts = re.split(r"(\n\s*\n)", section)
    out = []
    for i, c in enumerate(parts):
        if i % 2 == 1:
            out.append(c)
            continue
        core = c.strip()
        if core and _is_comparison_paragraph(core):
            out.append(c.replace(core, fn(core), 1))
        else:
            out.append(c)
    return "".join(out)


def _first_body_paragraph(text):
    for para in _md_paragraphs(text):
        if not para.lstrip().startswith("#"):
            return para
    return None


def _append_to_paragraph(section, paragraph, extra):
    return section.replace(paragraph, paragraph + extra, 1)


# 评审字母 / 定义 / 本测试用例 / 此前 (fed6bdf) 名称：
# A  把「官方模型卡的做法是错误的。」插入另一段（接到最后一段）  A  A
# B  同句无关否定：对照段追加「这不是小问题：官方…错误的。」    B  B
# C  已删：依赖 extra 规则抓「官方口径是个 bug。」；现为文档化漏报
# D  去掉含「官方」的句子                                        D  D（额外）
# E  对照段后另起一段「官方的最后闭合解析其实错了。」            E  评审原 E
# F  删掉与任一 _P25_EXERCISE 匹配区间重叠的句子（匹配可跨 。）   F  F（额外）
# G  删掉每个满足条件的对照段里含 chat template 的整句             G  新增（旧 G 现为 H）
# H  按 URL 正则删掉 tokenizer_config 链接，不依赖链接文字/分隔符  H  旧 G
# X1 每个满足条件的对照段 chat template → 模板                     X1 旧 E
# X2 固定 revision → main                                        X2 旧 H
def _p25_mutations(section):
    contrast = _comparison_paragraph(section)
    last_para = _md_paragraphs(section)[-1]
    return (
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
            "D_drop_official_sentences",
            _drop_sentences_matching(section, re.compile(r"官方")),
        ),
        (
            "E_standalone_wrong_paragraph",
            section.replace(contrast, contrast + "\n\n官方的最后闭合解析其实错了。\n\n", 1),
        ),
        (
            "F_drop_exercise_sentences",
            _drop_exercise_sentences(section),
        ),
        (
            "G_drop_chat_template_sentence",
            _map_comparison_paragraphs(
                section,
                lambda p: _drop_sentences_matching(p, re.compile(r"chat template")),
            ),
        ),
        (
            "H_drop_tokenizer_link",
            _drop_tokenizer_link(section),
        ),
        (
            "X1_no_chat_template_in_contrast",
            _map_comparison_paragraphs(
                section,
                lambda p: p.replace("chat template", "模板"),
            ),
        ),
        (
            "X2_unpinned_revision",
            section.replace(_P25_PINNED_REV, "main"),
        ),
    )


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

        # mirrors main，逐字保留
        def main_contract(text):
            self.assertIn("首闭合", text)
            self.assertIn("最后一个闭合", text)
            self.assertIn("模型卡", text)
            self.assertIn("chat template", text)
            self.assertIn(
                "https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/README.md",
                text,
            )
            self.assertIn(
                "https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer_config.json",
                text,
            )
            for pattern in (
                r"官方[^。\n]{0,18}(错误|不正确|有误|错了)",
                r"官方[^。\n]{0,18}做法[^。\n]{0,8}(错|错误)",
            ):
                self.assertNotRegex(text, pattern)

        main_contract(section)
        removed_comparison = section.replace("模型卡", "", 1).replace("chat template", "", 1)
        with self.assertRaises(AssertionError):
            main_contract(removed_comparison)
        official_error = section.replace("官方模型卡采用最后一个闭合位置", "官方模型卡采用错误位置", 1)
        with self.assertRaises(AssertionError):
            main_contract(official_error)

        def assert_contract(text):
            main_contract(text)
            paras = _contrast_paragraphs(text)
            self.assertTrue(paras, "P25.9 缺少同一段内的官方/练习对照")
            hit = _comparison_paragraph(text)
            self.assertIsNotNone(hit, "没有一段同时含官方/练习对照、首闭合、最后闭合、chat template 与模型卡")
            self.assertIn("chat template", hit)
            self.assertIn("模型卡", hit)
            self.assertIn(_P25_PINNED_README, text)
            self.assertIn(_P25_PINNED_TOKENIZER, text)
            leftover = _p25_wrong_claim_segments(text)
            self.assertFalse(leftover, leftover)

        assert_contract(section)

        contrast = _comparison_paragraph(section)
        first_body = _first_body_paragraph(section)
        last_para = _md_paragraphs(section)[-1]
        self.assertIsNotNone(contrast)
        self.assertIsNotNone(first_body)
        # 最后一段应是 generation-config 段，而不是对照段。
        self.assertNotEqual(last_para, contrast)

        def check_mutations(doc, label):
            for name, mutated in _p25_mutations(doc):
                with self.subTest(doc=label, mutation=name):
                    self.assertNotEqual(mutated, doc)
                    with self.assertRaises(AssertionError):
                        assert_contract(mutated)

        check_mutations(section, "real")

        # variants go through the same mutation table as the real section
        # (r8 review: helper-only checks missed table reverts).
        def _insert_before_trailing_period(text, extra):
            if text.endswith("。"):
                return text[:-1] + extra + "。"
            return text + extra

        B = (
            "上述固定 revision 的官方模型卡采用最后一个闭合位置（反向查找 151668）；"
            "同 revision 的 chat template 对 reasoning 取第一个 `</think>` 之前，对 content 取最后一个之后。"
        )
        for name, new_contrast in (
            ("本章_wording", re.sub(r"本?(?:教学)?练习|教学", "本章", contrast)),
            ("本书_wording", re.sub(r"本?(?:教学)?练习|教学", "本书", contrast)),
            (
                "mixed_教学_本章",
                "注意教学约定和官方做法的区别。官方模型卡采用最后一个闭合位置切分；"
                "chat template 则 reasoning 取首个 `</think>` 前、content 取最后一个后。"
                "本章为了在一条序列里记账，约定使用首闭合。"
                + f"[固定模型卡]({_P25_PINNED_README})；[固定模板]({_P25_PINNED_TOKENIZER})。",
            ),
            (
                "cross_period",
                "下面是教学实现。这里选择首闭合记账："
                + B
                + "多个闭合时口径不同。"
                + f"[固定模型卡]({_P25_PINNED_README})；[固定模板]({_P25_PINNED_TOKENIZER})。",
            ),
        ):
            with self.subTest(variant=name):
                # 不断言 variant != section：正文若已采用该措辞，变体即为原文
                variant_section = section.replace(contrast, new_contrast, 1)
                assert_contract(variant_section)
                if name == "cross_period":
                    # 逐句匹配在这段上删不掉任何句子（锚点跨 。），只看这段本身，
                    # 不看整节，免得正文别处的句子影响这一断言
                    self.assertEqual(
                        _drop_sentences_matching(new_contrast, _P25_EXERCISE),
                        new_contrast,
                    )
                check_mutations(variant_section, name)
                self.assertFalse(
                    _P25_EXERCISE.search(
                        dict(_p25_mutations(variant_section))["F_drop_exercise_sentences"]
                    )
                )

        # 在整节上删链接（链接不一定在对照段里），再把新形式的链接放回对照段末
        # 只为构造变体：链接文字若带对照段锚点（如「固定 chat template」），保留为纯文字
        section_wo = re.sub(
            r"\[([^\]]*)\]\(" + re.escape(_P25_PINNED_TOKENIZER) + r"\)",
            lambda m: m.group(1) if ("chat template" in m.group(1) or "模型卡" in m.group(1)) else "",
            section,
        ).replace(_P25_PINNED_TOKENIZER, "")
        contrast_wo = _comparison_paragraph(section_wo)
        self.assertIsNotNone(contrast_wo)
        # 去掉链接后留下的分隔符（如「；。」），使变体与 review 的改写一致
        base = re.sub(r"[；，、\s]+。$", "。", contrast_wo)
        for name, extra in (
            ("link_label", f"；[固定 tokenizer 配置]({_P25_PINNED_TOKENIZER})"),
            ("link_sep", f"，[固定模板]({_P25_PINNED_TOKENIZER})"),
            ("link_and", f" 与 [固定模板]({_P25_PINNED_TOKENIZER})"),
            ("bare_url", f"；<{_P25_PINNED_TOKENIZER}>"),
        ):
            with self.subTest(variant=name):
                new_contrast = _insert_before_trailing_period(base, extra)
                variant_section = section_wo.replace(contrast_wo, new_contrast, 1)
                assert_contract(variant_section)
                check_mutations(variant_section, name)
                self.assertNotIn(
                    _P25_PINNED_TOKENIZER,
                    dict(_p25_mutations(variant_section))["H_drop_tokenizer_link"],
                )

        # r11 N05：对照段正文写「模板」，chat template 只出现在 tokenizer 链接文字里；
        # 正文没有这两处原措辞时不改写（变体即原文），避免绑定章节措辞
        if "同 revision 的 chat template" in contrast and "[固定模板]" in contrast:
            n05_contrast = contrast.replace(
                "同 revision 的 chat template", "同 revision 的模板", 1
            ).replace("[固定模板]", "[固定 chat template]", 1)
        else:
            n05_contrast = contrast
        with self.subTest(variant="ct_in_link_label"):
            variant_section = section.replace(contrast, n05_contrast, 1)
            assert_contract(variant_section)
            self.assertIn(n05_contrast, _comparison_paragraph(variant_section))
            check_mutations(variant_section, "ct_in_link_label")

        # 整节变换：前置小结/回顾、首段点名、重复对照段；不断言 variant != section
        for name, variant_section in (
            (
                "summary_before",
                section.replace(
                    contrast,
                    "小结：本章选择首闭合，官方模板与模型卡取最后一个闭合。\n\n" + contrast,
                    1,
                ),
            ),
            (
                "recap_before_no_card",
                section.replace(
                    contrast,
                    "回顾：本章选择首闭合，与官方模板取最后一个闭合不同。\n\n" + contrast,
                    1,
                ),
            ),
            (
                "first_para_mentions_all",
                section.replace(
                    first_body,
                    first_body + "本章约定首闭合，官方做法取最后一个闭合。",
                    1,
                ),
            ),
            (
                "decoy_no_card",
                section.replace(contrast, "回顾：本章选择首闭合，官方 chat template 取最后一个闭合。\n\n" + contrast, 1),
            ),
            (
                "decoy_no_official",
                section.replace(contrast, "回顾：本章选择首闭合，模型卡与 chat template 取最后一个闭合。\n\n" + contrast, 1),
            ),
            (
                "decoy_no_exercise",
                section.replace(contrast, "回顾：官方模型卡与 chat template 取最后一个闭合，而不是首闭合。\n\n" + contrast, 1),
            ),
            (
                "decoy_no_first",
                section.replace(contrast, "回顾：本章选择的口径与官方模型卡、chat template 取最后一个闭合不同。\n\n" + contrast, 1),
            ),
            (
                "decoy_no_last",
                section.replace(contrast, "回顾：本章选择首闭合，与官方模型卡、chat template 的口径不同。\n\n" + contrast, 1),
            ),
            (
                "dup_comparison",
                section.replace(contrast, contrast + "\n\n" + contrast, 1),
            ),
        ):
            with self.subTest(variant=name):
                assert_contract(variant_section)
                if name == "dup_comparison":
                    self.assertEqual(
                        len(_comparison_paragraphs(variant_section)),
                        len(_comparison_paragraphs(section)) + 1,
                    )
                else:
                    # 选中的仍是原对照段（若首段本身就是对照段，则是追加后的它）
                    self.assertIn(contrast, _comparison_paragraph(variant_section))
                check_mutations(variant_section, name)

        for phrase in _P25_MUST_FAIL:
            with self.subTest(must_fail=phrase):
                mutated = section + phrase
                with self.assertRaises(AssertionError):
                    assert_contract(mutated)
                self.assertTrue(_p25_wrong_claim_segments(phrase))

    def test_p25_wrong_claim_check_is_exactly_main(self):
        self.assertEqual(
            _P25_MAIN_PATTERNS,
            (
                r"官方[^。\n]{0,18}(错误|不正确|有误|错了)",
                r"官方[^。\n]{0,18}做法[^。\n]{0,8}(错|错误)",
            ),
        )
        sample = "官方做法错了？官方解析有误！它很好。第二行\n官方模型卡不正确"
        self.assertEqual(_p25_segments(sample), re.split(r"[。\n]", sample))
        self.assertEqual(
            _p25_segments(sample),
            ["官方做法错了？官方解析有误！它很好", "第二行", "官方模型卡不正确"],
        )

        negated_fps = (
            "官方做法并非错误",
            "官方做法不算错",
            "这不代表官方做法有误",
            "官方做法也没有错",
            "官方口径与本练习不同，但并非错误",
            "官方做法未必错",
            "官方解析从未被证明有误",
        )
        misses = (
            "官方口径是个 bug。",
            "官方模板有缺陷。",
            "官方公式不成立。",
            "官方解析会出错。",
            "官方做法\n是错的。",
        )
        neutral = "官方在 v2 中修复了这个 bug。"
        probes = list(_P25_MUST_FAIL) + list(negated_fps) + list(misses) + [neutral]
        for p in probes:
            with self.subTest(probe=p):
                self.assertEqual(
                    bool(_p25_wrong_claim_segments(p)),
                    any(re.search(pat, p) for pat in _P25_MAIN_PATTERNS),
                )
        for p in negated_fps:
            with self.subTest(negated_fp=p):
                self.assertTrue(_p25_wrong_claim_segments(p), p)
        for p in misses + (neutral,):
            with self.subTest(known_miss=p):
                self.assertFalse(_p25_wrong_claim_segments(p), p)

        for name in (
            "_p25_exemption",
            "_p25_extra_hits",
            "_P25_EXEMPTIONS",
            "_P25_EXTRA",
            "_unnegated_official_wrong_sentences",
        ):
            self.assertFalse(hasattr(sys.modules[__name__], name))

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
