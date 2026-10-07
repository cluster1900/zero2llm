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


def check_book(dist_dir):
    dist = Path(dist_dir)
    errors = []
    expected_flows = len(re.findall(r'^```mermaid\s*$', (dist / 'transformer_for_highschool.md').read_text(), re.M))
    sizes = json.loads((dist / 'assets/rendered/dimensions.json').read_text())
    if len([key for key in sizes if key.startswith('flow-')]) != expected_flows:
        errors.append('渲染图表数量与书稿不一致')
    for name, size in sizes.items():
        # Catch accidentally restored long chains before they become unreadable thumbnails.
        if name.startswith('flow-') and not (0.45 <= size['width'] / size['height'] <= 2.4):
            errors.append(f'图表比例过长或过宽：{name}')
        for extension in ('png', 'svg'):
            if not (dist / f'assets/rendered/{name}.{extension}').is_file():
                errors.append(f'缺少图像：{name}.{extension}')

    with zipfile.ZipFile(dist / 'transformer_for_highschool.epub') as book:
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
                if tag in ('pre', 'code') and re.match(r'\s*(flowchart |sequenceDiagram\b|mindmap\b)', ''.join(el.itertext())):
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
    if (dist / 'index.html').read_bytes() != (dist / 'transformer_for_highschool.html').read_bytes():
        errors.append('两个网页入口内容不一致')
    if errors:
        raise ValueError('\n'.join(errors))
    print(f'✓ 产物校验通过：{expected_flows} 张流程图，{len(sizes)} 张插图；EPUB/Web 资源、内部链接、目录锚点完整')


if __name__ == '__main__':
    try:
        check_book(Path(__file__).resolve().parents[1] / 'dist')
    except (ValueError, OSError, zipfile.BadZipFile, ET.ParseError) as exc:
        print(f'❌ {exc}', file=sys.stderr)
        sys.exit(1)
