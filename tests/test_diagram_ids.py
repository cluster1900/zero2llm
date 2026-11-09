"""D0 (bot-hub#126): stable figure ids, render reuse, manifest gate, Mermaid leak check."""
import json
import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from book_diagrams import collect_jobs, render_all  # noqa: E402
from check_book import MERMAID_SOURCE, check_render_manifest  # noqa: E402
from diagram_ids import (mermaid_blocks, mermaid_id, mermaid_labels,  # noqa: E402
                         sha256_bytes, sha256_text)

DIST = ROOT / 'dist'
FLOW = 'flowchart TD\n    accTitle: 甲\n    accDescr: 乙\n    A-->B'
MINDMAP = 'mindmap\n  %% id: p17-codec-names\n  %% accTitle: 四种编码—解码\n  %% accDescr: 中心分四支。\n  root((编码—解码))\n    seq2seq'


def fake_renderer(calls):
    """Stands in for render_diagrams.mjs: records which jobs it was asked to draw."""
    def render(jobs, output):
        calls.extend(job['id'] for job in jobs)
        for job in jobs:
            (output / f'{job["id"]}.svg').write_text(f'<svg>{job["id"]}</svg>')
            (output / f'{job["id"]}.png').write_bytes(b'png:' + job['id'].encode())
        (output / 'dimensions.json').write_text(json.dumps({job['id']: {'width': 300, 'height': 200} for job in jobs}))
    return render


def job(definition):
    return {'id': mermaid_id(definition), 'kind': 'mermaid', 'definition': definition,
            'sha256': sha256_text(definition)}


class IdTests(unittest.TestCase):
    def test_hash_id_depends_only_on_source(self):
        self.assertEqual(mermaid_id(FLOW), 'flow-' + sha256_text(FLOW)[:12])
        book = f'```mermaid\n{FLOW}\n```\n'
        earlier = f'```mermaid\n{MINDMAP}\n```\n\n' + book
        self.assertEqual([mermaid_id(d) for d in mermaid_blocks(earlier)][1], mermaid_id(FLOW))

    def test_explicit_id(self):
        self.assertEqual(mermaid_id(MINDMAP), 'flow-p17-codec-names')
        self.assertEqual(mermaid_id('%% id: a-b\n' + FLOW), mermaid_id('%% id: a-b\n' + FLOW + '\n    B-->C'))

    def test_bad_explicit_ids_are_rejected(self):
        for bad, message in [('%% id: A_B\n', '小写字母'), ('%% id: 0123456789ab\n', '内容哈希'),
                             ('%% id: a\n%% id: b\n', '只能有一行')]:
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, message):
                mermaid_id(bad + FLOW)

    def test_labels_accept_comment_form_only_as_documented(self):
        self.assertEqual(mermaid_labels(FLOW), ('甲', '乙'))
        self.assertEqual(mermaid_labels(MINDMAP), ('四种编码—解码', '中心分四支。'))
        with self.assertRaisesRegex(ValueError, 'accTitle 和 accDescr'):
            mermaid_labels('mindmap\n  %% accTitle: 只有标题\n  root')

    def test_regex_extraction_matches_pandoc(self):
        pandoc = shutil.which('pandoc')
        if not pandoc:
            self.skipTest('选修：未安装 Pandoc')
        sys.path.insert(0, str(ROOT))
        import build
        parsed = subprocess.run([pandoc, str(DIST / 'transformer_from_scratch.md'), '-f', build.MD_FORMAT, '-t', 'json'],
                                check=True, capture_output=True, text=True)
        found = []

        def walk(value):
            if isinstance(value, list):
                for item in value:
                    walk(item)
            elif isinstance(value, dict):
                if value.get('t') == 'CodeBlock' and 'mermaid' in value['c'][0][1]:
                    found.append(value['c'][1])
                for item in value.values():
                    walk(item)
        walk(json.loads(parsed.stdout))
        self.assertEqual(mermaid_blocks((DIST / 'transformer_from_scratch.md').read_text()), found)

    def test_collect_rejects_two_sources_with_one_id(self):
        doc = [{'t': 'CodeBlock', 'c': [['', ['mermaid'], []], '%% id: twin\n' + FLOW]},
               {'t': 'CodeBlock', 'c': [['', ['mermaid'], []], '%% id: twin\n' + FLOW + '\n    B-->C']}]
        with self.assertRaisesRegex(ValueError, '同一个 id'):
            collect_jobs(doc, ROOT)


class ReuseTests(unittest.TestCase):
    def build_twice(self, first, second, force=False):
        with TemporaryDirectory() as folder:
            cache, out = Path(folder) / 'cache', Path(folder) / 'out'
            calls = []
            render_all(first, cache, renderer=fake_renderer(calls), renderer_label='first')
            calls.clear()
            reused, rendered = render_all(second, out, cache_dir=cache, renderer=fake_renderer(calls),
                                          renderer_label='second', force=force)
            manifest = json.loads((out / 'manifest.json').read_text())['diagrams']
            dimensions = json.loads((out / 'dimensions.json').read_text())
            return reused, rendered, calls, manifest, dimensions

    def test_only_new_or_changed_figures_are_rendered(self):
        a, b, c = job(FLOW), job(MINDMAP), job('%% id: p11-x\n' + FLOW)
        changed = job('%% id: p11-x\n' + FLOW + '\n    B-->C')
        reused, rendered, calls, manifest, dimensions = self.build_twice([a, c], [b, a, changed])
        self.assertEqual(reused, [a['id']])
        self.assertEqual(calls, [b['id'], changed['id']])
        self.assertEqual(rendered, calls)
        self.assertEqual(list(manifest), [b['id'], a['id'], changed['id']])
        self.assertEqual(list(dimensions), list(manifest))
        self.assertEqual(manifest[a['id']]['renderer'], 'first')
        self.assertEqual(manifest[changed['id']], {'kind': 'mermaid', 'sha256': changed['sha256'], 'renderer': 'second'})

    def test_nothing_changed_needs_no_renderer(self):
        a = job(FLOW)
        reused, rendered, calls, _, _ = self.build_twice([a], [a])
        self.assertEqual((reused, rendered, calls), ([a['id']], [], []))

    def test_force_rerenders_everything(self):
        a = job(FLOW)
        _, rendered, calls, _, _ = self.build_twice([a], [a], force=True)
        self.assertEqual((rendered, calls), ([a['id']], [a['id']]))


class ManifestGateTests(unittest.TestCase):
    def setUp(self):
        self.markdown = (DIST / 'transformer_from_scratch.md').read_text()
        self.sizes = json.loads((DIST / 'assets/rendered/dimensions.json').read_text())

    def test_committed_book_passes_and_uses_stable_names(self):
        self.assertEqual(check_render_manifest(DIST, self.markdown, self.sizes), [])
        self.assertFalse([k for k in self.sizes if re.fullmatch(r'flow-\d{2}', k)], '不应再有按顺序编号的 flow-NN')
        manifest = json.loads((DIST / 'assets/rendered/manifest.json').read_text())['diagrams']
        expected = {mermaid_id(d) for d in mermaid_blocks(self.markdown)}
        self.assertEqual({k for k, v in manifest.items() if v['kind'] == 'mermaid'}, expected)

    def test_stale_or_missing_mermaid_render_is_caught(self):
        stale = self.markdown.replace('accTitle: 注意力下半程：按比例收集信息', 'accTitle: 注意力下半程：按比例收集信息（改）')
        errors = check_render_manifest(DIST, stale, self.sizes)
        self.assertTrue(any('没有渲染' in e for e in errors), errors)
        self.assertTrue(any('书稿中不存在' in e for e in errors), errors)
        added = self.markdown + f'\n```mermaid\n%% id: p99-new\n{FLOW}\n```\n'
        self.assertIn('Mermaid 图没有渲染：flow-p99-new', check_render_manifest(DIST, added, self.sizes))

    def test_same_explicit_id_with_new_source_is_stale(self):
        with TemporaryDirectory() as folder:
            dist = Path(folder) / 'dist'
            shutil.copytree(DIST / 'assets', dist / 'assets')
            definition = '%% id: p99-new\n' + FLOW
            sizes = dict(self.sizes, **{'flow-p99-new': {'width': 300, 'height': 200}})
            path = dist / 'assets/rendered/manifest.json'
            manifest = json.loads(path.read_text())
            manifest['diagrams']['flow-p99-new'] = {'kind': 'mermaid', 'sha256': sha256_text(definition)}
            path.write_text(json.dumps(manifest))
            book = self.markdown + f'\n```mermaid\n{definition}\n```\n'
            self.assertEqual(check_render_manifest(dist, book, sizes), [])
            edited = book.replace('A-->B\n```\n', 'A-->C\n```\n')
            self.assertIn('Mermaid 图的渲染已过期（源码改了但没重新构建）：flow-p99-new',
                          check_render_manifest(dist, edited, sizes))

    def test_stale_static_svg_is_caught(self):
        with TemporaryDirectory() as folder:
            dist = Path(folder) / 'dist'
            shutil.copytree(DIST / 'assets', dist / 'assets')
            source = dist / 'assets/diagrams/p12-rope.svg'
            source.write_bytes(source.read_bytes() + b'\n<!-- edited -->\n')
            self.assertNotEqual(sha256_bytes(source.read_bytes()),
                                json.loads((DIST / 'assets/rendered/manifest.json').read_text())['diagrams']['p12-rope']['sha256'])
            self.assertIn('静态图的渲染已过期（SVG 改了但没重新构建）：p12-rope',
                          check_render_manifest(dist, self.markdown, self.sizes))


class ReferencedImagesTests(unittest.TestCase):
    """After the one-time rename, no page may point at a missing rendered image."""

    def test_every_rendered_image_referenced_by_the_site_exists(self):
        html = (DIST / 'index.html').read_text()
        refs = set(re.findall(r'(?:src|href)="(assets/rendered/[^"#?]+)"', html))
        manifest = json.loads((DIST / 'assets/rendered/manifest.json').read_text())['diagrams']
        self.assertEqual(refs, {f'assets/rendered/{i}.svg' for i in manifest})
        for ref in sorted(refs):
            with self.subTest(ref=ref):
                self.assertTrue((DIST / ref).is_file())

    def test_rendered_directory_has_no_orphans_or_order_names(self):
        manifest = json.loads((DIST / 'assets/rendered/manifest.json').read_text())['diagrams']
        files = {p.name for p in (DIST / 'assets/rendered').iterdir()}
        self.assertEqual(files, {'manifest.json', 'dimensions.json'}
                         | {f'{i}.{ext}' for i in manifest for ext in ('svg', 'png')})
        tracked = subprocess.run(['git', 'grep', '-n', '-I', '-E', r'assets/rendered/flow-[0-9]{2}\b', '--', '.'],
                                 cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(tracked.stdout, '', '仍有文件引用旧的按顺序编号的图片路径')

    def test_every_epub_image_exists(self):
        import posixpath
        import zipfile
        import xml.etree.ElementTree as ET
        with zipfile.ZipFile(DIST / 'transformer_from_scratch.epub') as book:
            names = set(book.namelist())
            count = 0
            for name in names:
                if not name.endswith('.xhtml'):
                    continue
                for el in ET.fromstring(book.read(name)).iter('{http://www.w3.org/1999/xhtml}img'):
                    count += 1
                    target = posixpath.normpath(posixpath.join(posixpath.dirname(name), el.get('src')))
                    self.assertIn(target, names, f'{name}: {el.get("src")}')
        self.assertEqual(count, len(json.loads((DIST / 'assets/rendered/dimensions.json').read_text())))


class LeakCheckTests(unittest.TestCase):
    def test_all_planned_diagram_types_are_detected(self):
        for source in [FLOW, 'sequenceDiagram\n  A->>B: x', 'stateDiagram-v2\n  [*] --> A', MINDMAP,
                       '%% id: p16-x\n%% accTitle: a\n%% accDescr: b\nblock-beta\n  a b', 'gantt\n  x :0, 1',
                       'graph LR\n  A-->B', '---\nconfig:\n  theme: base\n---\nclassDiagram\n  A <|-- B',
                       'architecture-beta\n  service a(server)[A]']:
            with self.subTest(source=source[:24]):
                self.assertTrue(MERMAID_SOURCE.match(source))

    def test_ordinary_code_is_not_flagged(self):
        for text in ['timeline', 'pie', 'gantt', 'x = 1\nflowchart TD', 'graph = build()\nprint(graph)']:
            with self.subTest(text=text):
                self.assertFalse(MERMAID_SOURCE.match(text))


if __name__ == '__main__':
    unittest.main()
