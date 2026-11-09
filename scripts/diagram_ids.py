"""Stable names and source fingerprints for every pre-rendered book figure.

Rendered files live in dist/assets/rendered/<id>.svg|png. The id no longer
depends on where a diagram appears in the book, so adding one diagram never
renames the others:

* Mermaid: ``flow-<slug>`` when the block has one ``%% id: <slug>`` line,
  otherwise ``flow-<first 12 hex of sha256(definition)>``.
* Static SVG (``![..](assets/diagrams/x.svg)``): the file stem, e.g. ``p12-rope``.

manifest.json records the sha256 of each figure's source, so a build (and
scripts/check_book.py) can tell a reusable render from a stale one.
"""
import hashlib
import re

MANIFEST = 'manifest.json'
DIMENSIONS = 'dimensions.json'
MERMAID_PREFIX = 'flow-'
LEGACY_RENDERER = 'legacy: committed before D0 (bot-hub#126)'

_ID_LINE = re.compile(r'^[ \t]*%%[ \t]*id:[ \t]*(.*?)[ \t]*$', re.M)
_SLUG = re.compile(r'^[a-z0-9]+(?:-[a-z0-9]+)*$')
_HASH_LIKE = re.compile(r'^[0-9a-f]{12}$')
# Mermaid's own accTitle/accDescr lines, or the same text as a %% comment for
# diagram types whose grammar has no accessibility syntax (mindmap, block-beta).
_TITLE = re.compile(r'^[ \t]*(?:%%[ \t]*)?accTitle:[ \t]*(.+?)[ \t]*$', re.M)
_DESCR = re.compile(r'^[ \t]*(?:%%[ \t]*)?accDescr:[ \t]*(.+?)[ \t]*$', re.M)
# Same block boundaries Pandoc uses for the book's ```mermaid fences.
_FENCE = re.compile(r'^```mermaid[ \t]*\n(.*?)\n?^```[ \t]*$', re.M | re.S)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_text(text):
    return sha256_bytes(text.encode('utf-8'))


def mermaid_id(definition):
    """Return the stable asset id of one Mermaid definition."""
    ids = _ID_LINE.findall(definition)
    if len(ids) > 1:
        raise ValueError(f'Mermaid 图只能有一行 %% id:，找到 {len(ids)} 行：{ids}')
    if ids:
        slug = ids[0]
        if not _SLUG.match(slug):
            raise ValueError(f'%% id: {slug!r} 只能用小写字母、数字和单个短横线，例如 p11-prefill-decode')
        if _HASH_LIKE.match(slug):
            raise ValueError(f'%% id: {slug!r} 像内容哈希，会与未写 id 的图混淆；请换一个有含义的名字')
        return MERMAID_PREFIX + slug
    return MERMAID_PREFIX + sha256_text(definition)[:12]


def mermaid_labels(definition):
    """(title, description) from accTitle/accDescr lines; ValueError if missing."""
    title = _TITLE.search(definition)
    description = _DESCR.search(definition)
    if not title or not description:
        raise ValueError('每张 Mermaid 图都需要 accTitle 和 accDescr，方便离线阅读和无障碍访问。'
                         'mindmap、block-beta 不支持这两行语法时，写成 %% accTitle: … / %% accDescr: … 注释行。')
    return title[1], description[1]


def mermaid_blocks(markdown):
    """Definitions of all ```mermaid blocks, in book order (no Pandoc needed)."""
    return [match[1] for match in _FENCE.finditer(markdown)]
