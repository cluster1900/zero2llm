#!/usr/bin/env python3
"""Validate the actual EPUB/Web artifacts, including images and local links."""
import json
import posixpath
import re
import struct
import sys
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

try:
    from scripts.diagram_ids import MANIFEST, mermaid_blocks, mermaid_id, sha256_bytes, sha256_text
except ImportError:  # run as `python scripts/check_book.py`
    from diagram_ids import MANIFEST, mermaid_blocks, mermaid_id, sha256_bytes, sha256_text

# Mermaid source must never reach a reader: an EPUB <pre>/<code> element that
# starts with a diagram header (after optional front matter and %% comment
# lines such as `%% id:` / `%% accTitle:`) means a block escaped pre-rendering.
MERMAID_SOURCE = re.compile(
    r'^\s*(?:---\n.*?\n---\s*\n\s*)?(?:%%[^\n]*\n\s*)*(?:flowchart\s|sequenceDiagram\b|mindmap\b'
    # Other diagram keywords are ordinary words too, so require a following line.
    r'|(?:graph\s+(?:TB|TD|BT|RL|LR)|stateDiagram(?:-v2)?|block-beta|gantt|classDiagram|architecture-beta'
    r'|erDiagram|journey|timeline|pie|quadrantChart|xychart-beta|sankey-beta|gitGraph)\b[^\n]*\n)', re.S)


class WebDocument(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = set()
        self.images = []
        self.mermaid = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get('id'):
            self.ids.add(attrs['id'])
        for key in ('src', 'href'):
            if attrs.get(key):
                self.links.append(attrs[key])
        if 'mermaid' in attrs.get('class', '').split():
            self.mermaid = True
        if tag == 'img':
            self.images.append(attrs)


def check_dist_md_matches_chapters(dist):
    """Fail when dist md drifts from chapters/ (catches dist-only edits)."""
    root = dist.parent
    chapters = sorted((root / 'chapters').glob('*.md'))
    if not chapters:
        return 'chapters/ 下没有 Markdown 章节'
    expected = '\n\n---\n\n'.join(p.read_text(encoding='utf-8') for p in chapters) + '\n\n---\n\n'
    text = (dist / 'transformer_from_scratch.md').read_text(encoding='utf-8')
    header = re.match(r'(?:%.*\n){3}\n', text)
    if not header:
        return 'dist Markdown 缺少 Pandoc 元数据头'
    if text[header.end():] != expected:
        return 'dist Markdown 与 chapters/ 不一致'
    return None


def check_render_manifest(dist, markdown, sizes):
    """Every rendered figure must come from the current source (stable id + sha256)."""
    errors = []
    path = dist / 'assets/rendered' / MANIFEST
    if not path.is_file():
        return [f'缺少渲染清单：assets/rendered/{MANIFEST}']
    manifest = json.loads(path.read_text())['diagrams']
    if list(manifest) != list(sizes):
        errors.append('渲染清单与 dimensions.json 的图表 id 不一致')
    wanted = {}
    for definition in mermaid_blocks(markdown):
        try:
            diagram_id = mermaid_id(definition)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if wanted.setdefault(diagram_id, sha256_text(definition)) != sha256_text(definition):
            errors.append(f'两张不同的 Mermaid 图用了同一个 id：{diagram_id}')
    for diagram_id, digest in wanted.items():
        entry = manifest.get(diagram_id)
        if not entry or entry.get('kind') != 'mermaid':
            errors.append(f'Mermaid 图没有渲染：{diagram_id}')
        elif entry.get('sha256') != digest:
            errors.append(f'Mermaid 图的渲染已过期（源码改了但没重新构建）：{diagram_id}')
    for diagram_id, entry in manifest.items():
        if entry.get('kind') == 'mermaid':
            if diagram_id not in wanted:
                errors.append(f'渲染清单里有书稿中不存在的 Mermaid 图：{diagram_id}')
        elif entry.get('kind') == 'static':
            source = dist / entry.get('source', '')
            if not source.is_file():
                errors.append(f'静态图源文件缺失：{diagram_id}')
            elif sha256_bytes(source.read_bytes()) != entry.get('sha256'):
                errors.append(f'静态图的渲染已过期（SVG 改了但没重新构建）：{diagram_id}')
        else:
            errors.append(f'渲染清单条目类型未知：{diagram_id}')
    return errors


def check_book(dist_dir):
    dist = Path(dist_dir)
    errors = []
    markdown = (dist / 'transformer_from_scratch.md').read_text()
    expected_flows = len(re.findall(r'^```mermaid\s*$', markdown, re.M))
    sizes = json.loads((dist / 'assets/rendered/dimensions.json').read_text())
    errors.extend(check_render_manifest(dist, markdown, sizes))
    if len([key for key in sizes if key.startswith('flow-')]) != expected_flows:
        errors.append('渲染图表数量与书稿不一致')
    for name, size in sizes.items():
        # Catch accidentally restored long chains before they become unreadable thumbnails.
        if name.startswith('flow-') and not (0.45 <= size['width'] / size['height'] <= 2.4):
            errors.append(f'图表比例过长或过宽：{name}')
        for extension in ('png', 'svg'):
            if not (dist / f'assets/rendered/{name}.{extension}').is_file():
                errors.append(f'缺少图像：{name}.{extension}')

    with zipfile.ZipFile(dist / 'transformer_from_scratch.epub') as book:
        if book.testzip():
            errors.append('EPUB ZIP 数据损坏')
        if book.namelist()[0] != 'mimetype' or book.read('mimetype') != b'application/epub+zip':
            errors.append('EPUB mimetype 不正确')
        names = set(book.namelist())
        roots = {name: ET.fromstring(book.read(name)) for name in names if name.endswith(('.xhtml', '.opf', '.ncx'))}
        ids = {name: {el.get('id') for el in root.iter() if el.get('id')} for name, root in roots.items()}
        for name, root in roots.items():
            if name.endswith('.opf'):
                for item in root.findall('.//{*}manifest/{*}item'):
                    target = posixpath.normpath(posixpath.join(posixpath.dirname(name), item.get('href')))
                    if target in roots and roots[target].find('.//{http://www.w3.org/1998/Math/MathML}math') is not None:
                        if 'mathml' not in item.get('properties', '').split():
                            errors.append(f'缺少 MathML 格式声明：{target}')
        epub_images = 0
        for name, root in roots.items():
            for el in root.iter():
                tag = el.tag.rsplit('}', 1)[-1]
                if 'mermaid' in el.get('class', '').split() or tag == 'script':
                    errors.append(f'EPUB 遗留未渲染图表或脚本：{name}')
                if tag in ('pre', 'code') and MERMAID_SOURCE.match(''.join(el.itertext())):
                    errors.append(f'EPUB 出现流程图源码：{name}')
                if tag == 'img':
                    epub_images += 1
                    if not el.get('alt'):
                        errors.append(f'图片缺少替代文字：{name}')
                    if el.get('src', '').endswith('.svg'):
                        errors.append(f'EPUB 图表未转换成兼容图像：{name}')
                for attr in ('href', 'src'):
                    link = el.get(attr)
                    if not link:
                        continue
                    parsed = urlsplit(link)
                    if parsed.scheme or parsed.netloc:
                        continue
                    target = posixpath.normpath(posixpath.join(posixpath.dirname(name), unquote(parsed.path))) if parsed.path else name
                    if target not in names:
                        errors.append(f'EPUB 链接缺失：{name} -> {link}')
                    elif parsed.fragment and target in ids and unquote(parsed.fragment) not in ids[target]:
                        errors.append(f'EPUB 锚点缺失：{name} -> {link}')
        if epub_images != len(sizes):
            errors.append(f'EPUB 插图数量不符：{epub_images} / {len(sizes)}')
        for name in names:
            if name.endswith('.png'):
                data = book.read(name)
                if data[:8] != b'\x89PNG\r\n\x1a\n' or min(struct.unpack('>II', data[16:24])) < 100:
                    errors.append(f'无效或过小的 PNG：{name}')

    web = WebDocument()
    web.feed((dist / 'index.html').read_text())
    if web.mermaid:
        errors.append('HTML 遗留 Mermaid 运行时图表')
    if len(web.images) != len(sizes):
        errors.append(f'网页插图数量不符：{len(web.images)} / {len(sizes)}')
    for link in web.links:
        parsed = urlsplit(link)
        if parsed.scheme or parsed.netloc:
            continue
        if parsed.path and not (dist / unquote(parsed.path)).is_file():
            errors.append(f'网页资源缺失：{link}')
        elif not parsed.path and parsed.fragment and unquote(parsed.fragment) not in web.ids:
            errors.append(f'网页锚点缺失：{link}')
    if (dist / 'index.html').read_bytes() != (dist / 'transformer_from_scratch.html').read_bytes():
        errors.append('两个网页入口内容不一致')
    md_err = check_dist_md_matches_chapters(dist)
    if md_err:
        errors.append(md_err)
    if errors:
        raise ValueError('\n'.join(errors))
    print(f'✓ 产物校验通过：{expected_flows} 张流程图，{len(sizes)} 张插图；EPUB/Web 资源、内部链接、目录锚点完整')


if __name__ == '__main__':
    try:
        check_book(Path(__file__).resolve().parents[1] / 'dist')
    except (ValueError, OSError, zipfile.BadZipFile, ET.ParseError) as exc:
        print(f'❌ {exc}', file=sys.stderr)
        sys.exit(1)
