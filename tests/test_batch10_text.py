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

# mirrors main: origin/main tests/test_batch10_text.py assert_contract 的 assertNotRegex 两条，逐字复制，勿改
_P25_MAIN_PATTERNS = (
    r"官方[^。\n]{0,18}(错误|不正确|有误|错了)",
    r"官方[^。\n]{0,18}做法[^。\n]{0,8}(错|错误)",
)
_P25_NEG = re.compile(r"[不没无非未别莫否勿毋]|难道|谁说|哪里|哪儿|怎么|岂|很难|才怪|很少|极少|鲜少")
_P25_WRONG_TOKEN = re.compile(r"[错误谬]|bug|缺陷|问题|不对|不妥", re.I)
_P25_BENIGN = ("不同",)
_P25_GAP = r"[^，,；;：:。！？!?\n]{0,12}?"
_P25_EXEMPTIONS = (
    # main 把「官方…并非/不是错误」收成命中（官方到「错误」之间可夹否定）。
    # 例：官方模型卡的做法并非错误。 / 官方做法绝不是错的。
    ("E1 并非/不是 错误", re.compile(r"官方" + _P25_GAP + r"(?P<neg>并非|绝非|绝不是|不是|不会是)(?P<pred>错误|错的)")),
    # main 第二条「官方…做法…错」会吃到「不算错 / 谈不上错」。
    # 例：官方做法不算错。 / 官方做法谈不上错，只是与本练习的约定不同。
    ("E2 不算/谈不上 错", re.compile(r"官方" + _P25_GAP + r"(?P<neg>不算|谈不上)(?P<pred>错)")),
    # 否定在「官方」之前，main 仍从「官方」扫到「有误/错误/错了/错」。
    # 例：这不代表官方做法有误。 / 不能认为官方模板有误。
    ("E3 不代表/不能认为/不认为/不说/没有说…官方…有误", re.compile(r"(?P<neg>不代表|不能认为|不认为|并不是|不说|没有说)官方" + _P25_GAP + r"(?P<pred>有误|错误|错了|错)")),
    # main 把「官方…没有错 / 不存在错误」里的「错/错误」当谓词。
    # 例：官方做法也没有错。 / 官方模型卡的示例不存在错误。
    ("E4 也没有错/不存在错误", re.compile(r"官方" + _P25_GAP + r"(?P<neg>没有|没|并无|不存在)(?:任何|什么)?(?P<pred>错误|错)")),
    # main 的 [^。\n] 能越过逗号，把「官方…，但并非错误」收成命中。
    # 例：官方口径与本练习不同，但并非错误。
    ("E5 …，但并非错误", re.compile(r"官方[^。\n]{0,18}?[，,]但(?P<neg>并非)(?P<pred>错误)")),
    # main 第二条会吃到「官方做法未必错 / 不一定是错的」。
    # 例：官方做法未必错。
    ("E6 未必/不一定 错", re.compile(r"官方" + _P25_GAP + r"(?P<neg>未必|不一定)(?:是)?(?P<pred>错的|错)")),
    # main 第一条会吃到「官方…从未/尚未被证明有误」。
    # 例：官方解析从未被证明有误。
    ("E7 从未被证明有误", re.compile(r"官方" + _P25_GAP + r"(?P<neg>从未|尚未|未)被(?:证明|证实)(?:是)?(?P<pred>有误|错误|错的)")),
)
# extra 只在分句（按 ，,；;：:。！？!?\n 切开）以「官方」起头时开火；宁可漏、不误报。
_P25_EXTRA_CLAUSE = re.compile(r"[，,；;：:。！？!?\n]")
_P25_EXTRA = re.compile(r"^[\s\"'“”「」『』（(*]*官方(?P<span>[^，,；;：:。！？!?\n]{0,24}?)(?P<pred>bug|缺陷|存在问题|有问题|不成立|不准确|不妥|不对|不可靠|失误|谬误|搞错|算错|写错|出错|是错的|是个错|错的|有误|错误|错了)", re.I)
_P25_EXTRA_ASIDE = re.compile(r"[，,]\s*(?:坦白说|说实话|老实说|实话说|说到底|事实上|实际上|其实|显然)\s*[，,]")
_P25_EXTRA_SUBJECT = re.compile(r"我们|我|本练习|练习|本章|本书|读者|手写|实现|你|他们|用户|改写")
_P25_NEXT_STRIP = " \"'“”「」『』（）()*"

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
    # 第五轮评审 fresh.json / lim.json：
    "官方做法哪里没错？",  # 第五轮回归（main 能抓，999b3f5 漏判）
    "官方解析哪里没错了？",  # 第五轮回归（main 能抓，999b3f5 漏判）
    "官方做法怎么可能没错？",  # 第五轮回归（main 能抓，999b3f5 漏判）
    "官方做法怎么可能是对的，它明明错了。",
    "官方做法不可能没错。",  # 第五轮回归（main 能抓，999b3f5 漏判）
    "官方解析不可能没有错误。",  # 第五轮回归（main 能抓，999b3f5 漏判）
    "很难说官方做法没错。",  # 第五轮回归（main 能抓，999b3f5 漏判）
    "很难说官方解析没有错误。",  # 第五轮回归（main 能抓，999b3f5 漏判）
    "官方做法未必没错。",  # 第五轮回归（main 能抓，999b3f5 漏判）
    "没有人不认为官方做法是错的。",  # 第五轮回归（main 能抓，999b3f5 漏判）
    "很难不认为官方做法有误。",  # 第五轮回归（main 能抓，999b3f5 漏判）
    "我们无法不认为官方做法错了。",  # 第五轮回归（main 能抓，999b3f5 漏判）
    "与其说官方做法是对的，不如说它错了。",
    "官方解析并非毫无问题，而是错误的。",
    "官方做法不是错了，是大错特错。",
    "「官方解析」是错的。",
    "“官方做法”其实错了。",
    "“官方做法没错”这种说法本身就错了。",
    "本练习（以及官方模型卡）的闭合规则都是错误的。",
    "（官方模型卡的做法）错误。",
    "官方解析没有被证明是错的，但它的确有误。",
    "官方解析——说实话——不对。",
    "Honestly, 官方 parser 的做法 is 错的.",
    "官方模板 (chat template) 有误!",
    "官方模板没有问题，官方解析有误。",
    "官方的说法没错，官方的做法错了。",
    "说官方做法没错，那是自欺欺人；它错了。",
    "官方做法 显然 错了。",
    "官方做法并不是没有错。",
    "官方做法并非不是错的。",
    "我们不认为官方做法是对的，它有误。",
    "官方的“最后闭合”解析是错误的。",
    "官方做法实际上是一个严重的错误。",
    "官方做法从头到尾都错了。",
    "官方解析这里搞错了闭合位置。",
    "官方做法；说到底；错了。",
    "官方做法没错？才怪。",  # 第五轮回归（main 能抓，999b3f5 漏判）
    "官方做法也没有错到哪里去。",  # 第五轮回归（main 能抓，999b3f5 漏判）
    "官方做法没错。\n\n才怪。",
    # 封闭插入语（_P25_EXTRA_ASIDE）：main 只因 removed-comparison 变异偶然失败，本检测器由 extra 抓。
    "官方的 chat template 写法，坦白说，也确实不对。",
    "官方解析，说到底，是错的。",
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
    # 第五轮评审 fresh.json / lim.json：
    "官方做法绝非错误。",
    "官方做法绝不是错的。",
    "官方解析没有错，问题出在我们的实现有 bug。",  # 第五轮误报
    "官方模板没问题，我们的手写解析有缺陷。",  # 第五轮误报
    "官方解析是对的，是本练习的计数口径不对。",  # 第五轮误报
    "与官方口径不同的手写解析容易出错。",  # 第五轮误报
    "不了解官方口径的读者很容易出错。",  # 第五轮误报
    "本练习（与官方模型卡不同）的计数容易出错。",  # 第五轮误报
    "官方解析没被证明是错的。",  # 第五轮误报
    "官方解析从未被证明有误。",
    "官方解析尚未被证实有问题。",
    "官方做法谈不上错，只是与本练习的约定不同。",
    "官方模型卡的做法完全没有问题。",
    "官方解析一点也没错。",
    "官方解析没有哪里不对。",  # 第五轮误报
    "官方做法本身并没有错。",
    "官方做法不是错误，本章只是选了首闭合。",
    "官方口径和本章口径彼此交错，读者要分清。",
    "官方与本练习的两套标签错落排列。",
    "官方的模型卡写得相当不错。",
    "官方解析不存在问题。",
    "官方解析并无不妥。",
    "官方解析毫无不妥之处。",
    "按官方口径实现时，若漏掉最后一个闭合，就容易出错。",
    "官方的 chat template（同 revision）取最后一个闭合，这并没有错。",
    "官方解析从不出错。",
    "官方做法根本没有错。",
    "官方做法当然没错。",
    "我们没有说官方做法有误。",
    "官方做法和本练习各有道理，谁也没有错。",
    "官方解析即使在多个 </think> 时也不会出错。",
    # 封闭插入语不切分句，但否定照旧挡住 extra。
    "官方做法，说实话，并没有不对。",
)
# 评审会放行、但 main 同样拒绝；本检测器刻意与 main 一样严。
_P25_MAIN_SHARED_FALSE_POSITIVES = (
    "“官方做法有误”是一种误解。",
    "官方做法不错，本练习的做法也不错。",
    "谁说官方做法错了？",
    "官方是对的，我们的改写错了。",
    "说官方做法错了是一种误读。",
    "如果你认为官方做法错了，请再看一遍模型卡。",
)
# 官方错判、但 main 与本检测器都抓不住（记录极限；改进时再挪走）。
_P25_KNOWN_MISSES = (
    "官方做法不是不对，而是根本不对。",
    "官方解析绝不是没错。",
    "官方解析没有被证明是对的，它就是错的。",
    "官方做法在我们反复、仔细、逐 token 地核对之后，依然、确实、毫无疑问地是错的。",
    "官方做法也许没有大问题，但确实不对。",
    "官方解析的闭合逻辑也并非没有缺陷。",
    "官方做法不能说是对的，它有问题。",
    "官方解析，在多 </think> 时会出错。",
    "错的是官方做法。",
    "官方模型卡并非正确。",
    "官方解析不是没有问题。",
    "官方口径没有道理，是错的。",
    "不得不说官方是错的。",
    "官方做法\n是错的。",
    "官方做法没有对的地方，全是错的。",
    "官方给出的闭合解，在多个 </think> 的情况下，其实是错误的。",
    "官方解析没有被正确实现，它是错的。",
)
_P25_EXEMPTION_CASES = (
    ("E1", "官方模型卡的做法并非错误。"),
    ("E1", "官方做法绝不是错的。"),
    ("E2", "官方做法不算错。"),
    ("E2", "官方做法谈不上错，只是与本练习的约定不同。"),
    ("E3", "这不代表官方做法有误。"),
    ("E3", "不能认为官方模板有误。"),
    ("E4", "官方做法也没有错。"),
    ("E4", "官方模型卡的示例不存在错误。"),
    ("E5", "官方口径与本练习不同，但并非错误。"),
    ("E6", "官方做法未必错。"),
    ("E7", "官方解析从未被证明有误。"),
)
_P25_EXEMPTION_KILLERS = (
    "官方做法哪里没错？",
    "官方做法也没有错到哪里去。",
    "很难说官方做法没错。",
    "官方做法未必没错。",
    "很难不认为官方做法有误。",
    "我们无法不认为官方做法错了。",
    "官方做法没错？才怪。",
    "官方做法并非错误，它错了。",
    "官方做法不算错，但官方解析有误。",
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


# 已知局限（verdict = (main_hit ∧ ¬exemption) ∨ extra_hit，故不会弱于 main）：
# (a) 与 main 共享的误报（无否定、main 本身就会抓）：引述/假设/反问「谁说官方做法错了？」、
#     「不错」里的裸「错」、「官方是对的，我们的改写错了」；见 _P25_MAIN_SHARED_FALSE_POSITIVES。
# (b) 已知漏判（main 同样漏）列在 _P25_KNOWN_MISSES：双重否定、谓词落在逗号后的下一分句、
#     跨句、主语倒置「错的是官方做法」、「官方」距谓词超过 24 字。
# (c) _P25_BENIGN 把「不同」从 rest 里抹掉，以免「口径不同」误杀豁免。
# (d) extra_hit 只在分句以「官方」起头时开火（可先有引号/括号）；逗号/别的主语/否定都不开火。
def _p25_main_segments(text):
    """按 [。\\n] 切 RAW text（不做 NFKC）。

    main 的两条正则用 [^。\\n]，因此任何 main 命中都落在恰好一个这样的片段里；
    「main 匹配整段 text」⇔「main 匹配其中某个 segment」。
    """
    return re.split(r"[。\n]", text)


def _p25_main_hit(seg):
    return any(re.search(p, seg) for p in _P25_MAIN_PATTERNS)


def _p25_next_nonempty_segment(segs, i):
    """下一段去掉空白/引号/括号/* 之后仍非空的 raw segment；没有则空串。"""
    for nxt in segs[i + 1 :]:
        if unicodedata.normalize("NFKC", nxt).lstrip(_P25_NEXT_STRIP):
            return nxt
    return ""


def _p25_exemption(seg, next_seg):
    """显式豁免：该片段因此恰好含一处否定（pattern 的）和一处错误词（pattern 的谓词）。

    任何第二处否定/反问词（不/没/无/非/未/别/莫/难道/谁说/哪里/怎么/很难/才怪/很少 …）
    或第二处错误断言都会杀死豁免。next_seg 应是下一段非空 fragment
    （去掉空白/引号/括号/* 之后），以便「官方做法没错。\\n\\n才怪。」仍能取消豁免。
    """
    s = unicodedata.normalize("NFKC", seg)
    if unicodedata.normalize("NFKC", next_seg).lstrip(_P25_NEXT_STRIP).startswith("才怪"):
        return None
    for name, rx in _P25_EXEMPTIONS:
        for m in rx.finditer(s):
            rest = s[: m.start("neg")] + "\x00" + s[m.end("neg") : m.start("pred")] + "\x00" + s[m.end("pred") :]
            for b in _P25_BENIGN:
                rest = rest.replace(b, "")
            if _P25_NEG.search(rest) or _P25_WRONG_TOKEN.search(rest):
                continue
            return name
    return None


def _p25_extra_hits(text):
    """保守补漏：分句（按 ，,；;：:。！？!?\\n 切开）必须以「官方」起头（可先有引号/括号），
    且「官方」到谓词的 span 里没有否定、也没有别的主语（我们/本练习/读者/手写/实现…）；
    逗号/别的主语/否定 ⇒ 不开火。宁可漏、不误报。
    markdown 链接的 `)` / `）` 后若直接接「官方」，插入分句边界（mutation C 把
    「官方口径是个 bug。」接到以 README.md) 结尾的段上）；不放宽谓词、不跨逗号。
    """
    t = unicodedata.normalize("NFKC", text)
    t = re.sub(r"([)）])(?=官方)", r"\1\n", t)
    # 封闭列表的插入语（「，坦白说，」「，说到底，」…）不算分句边界：去掉后再分句，
    # 使「官方…写法，坦白说，也确实不对」与「官方解析，说到底，是错的」仍归同一分句；
    # span 的否定/主语检查照旧，所以「官方做法，说实话，并没有不对」不开火。
    t = _P25_EXTRA_ASIDE.sub("", t)
    out = []
    for cl in _P25_EXTRA_CLAUSE.split(t):
        m = _P25_EXTRA.search(cl)
        if m and not _P25_NEG.search(m.group("span")) and not _P25_EXTRA_SUBJECT.search(m.group("span")):
            out.append(cl)
    return out


def _unnegated_official_wrong_sentences(text):
    """命中列表：非豁免的 main 片段 + extra 分句。verdict = (main ∧ ¬exemption) ∨ extra。"""
    segs = _p25_main_segments(text)
    out = []
    for i, seg in enumerate(segs):
        if _p25_main_hit(seg) and not _p25_exemption(seg, _p25_next_nonempty_segment(segs, i)):
            out.append(("main", seg))
    out += [("extra", c) for c in _p25_extra_hits(text)]
    return out


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

        for phrase in _P25_MAIN_SHARED_FALSE_POSITIVES:
            with self.subTest(shared_fp=phrase):
                self.assertTrue(
                    any(_p25_main_hit(seg) for seg in _p25_main_segments(phrase)),
                    phrase,
                )
                mutated = section + phrase
                self.assertNotEqual(mutated, section)
                with self.assertRaises(AssertionError):
                    assert_contract(mutated)

        for phrase in _P25_KNOWN_MISSES:
            with self.subTest(known_miss=phrase):
                self.assertFalse(
                    any(_p25_main_hit(seg) for seg in _p25_main_segments(phrase)),
                    phrase,
                )
                assert_contract(section + phrase)

        for name, sentence in _P25_EXEMPTION_CASES:
            with self.subTest(exemption=name, sentence=sentence):
                segs = _p25_main_segments(sentence)
                matched = False
                for i, seg in enumerate(segs):
                    if not _p25_main_hit(seg):
                        continue
                    got = _p25_exemption(seg, _p25_next_nonempty_segment(segs, i))
                    self.assertIsNotNone(got, sentence)
                    self.assertTrue(got.startswith(name), (got, name, sentence))
                    matched = True
                self.assertTrue(matched, sentence)
                assert_contract(section + sentence)

        for phrase in _P25_EXEMPTION_KILLERS:
            with self.subTest(killer=phrase):
                segs = _p25_main_segments(phrase)
                for i, seg in enumerate(segs):
                    if _p25_main_hit(seg):
                        self.assertIsNone(
                            _p25_exemption(seg, _p25_next_nonempty_segment(segs, i)),
                            phrase,
                        )
                mutated = section + phrase
                self.assertNotEqual(mutated, section)
                with self.assertRaises(AssertionError):
                    assert_contract(mutated)

    def test_p25_detector_is_never_weaker_than_main(self):
        self.assertEqual(
            _P25_MAIN_PATTERNS,
            (
                r"官方[^。\n]{0,18}(错误|不正确|有误|错了)",
                r"官方[^。\n]{0,18}做法[^。\n]{0,8}(错|错误)",
            ),
        )
        phrases = []
        phrases.extend(_P25_WRONG_PARAPHRASES)
        phrases.extend(_P25_ALLOWED_NEGATIONS)
        phrases.extend(_P25_MAIN_SHARED_FALSE_POSITIVES)
        phrases.extend(_P25_KNOWN_MISSES)
        phrases.extend(sentence for _, sentence in _P25_EXEMPTION_CASES)
        phrases.extend(_P25_EXEMPTION_KILLERS)
        for phrase in phrases:
            with self.subTest(phrase=phrase):
                segs = _p25_main_segments(phrase)
                for i, seg in enumerate(segs):
                    if (
                        _p25_main_hit(seg)
                        and _p25_exemption(seg, _p25_next_nonempty_segment(segs, i)) is None
                    ):
                        self.assertTrue(
                            _unnegated_official_wrong_sentences(phrase),
                            phrase,
                        )
                        break

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
