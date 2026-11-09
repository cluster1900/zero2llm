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
from check_book import (MERMAID_KEYWORDS, MERMAID_KEYWORDS_ANY_CASE, MERMAID_SOURCE,  # noqa: E402
                        check_book, check_render_manifest, scan_mermaid_js_detectors)
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
            shutil.copytree(ROOT / 'assets', Path(folder) / 'assets')
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

    def _copy_book(self, folder):
        """A self-contained repo copy (sources + committed dist) to mutate."""
        root = Path(folder)
        for name in ('chapters', 'assets'):
            shutil.copytree(ROOT / name, root / name)
        shutil.copytree(DIST, root / 'dist')
        return root

    def test_edited_repo_svg_source_fails_check_book(self):
        # People edit assets/diagrams/ (docs/diagrams.md); the build hashes that file,
        # so check_book must hash the same file, not the dist/assets/diagrams copy.
        with TemporaryDirectory() as folder:
            root = self._copy_book(folder)
            check_book(root / 'dist')
            source = root / 'assets/diagrams/p12-rope.svg'
            source.write_bytes(source.read_bytes() + b'\n<!-- edited -->\n')
            self.assertNotEqual(sha256_bytes(source.read_bytes()),
                                json.loads((DIST / 'assets/rendered/manifest.json').read_text())['diagrams']['p12-rope']['sha256'])
            self.assertIn('静态图的渲染已过期（SVG 改了但没重新构建）：p12-rope',
                          check_render_manifest(root / 'dist', self.markdown, self.sizes))
            with self.assertRaisesRegex(ValueError, '静态图的渲染已过期.*p12-rope'):
                check_book(root / 'dist')

    def test_edited_dist_svg_copy_fails(self):
        with TemporaryDirectory() as folder:
            root = self._copy_book(folder)
            copy = root / 'dist/assets/diagrams/p12-rope.svg'
            copy.write_bytes(copy.read_bytes() + b'\n')
            self.assertIn('dist 里的静态图副本和仓库源文件不一致（请重新构建）：p12-rope',
                          check_render_manifest(root / 'dist', self.markdown, self.sizes))

    def test_missing_repo_svg_source_fails(self):
        with TemporaryDirectory() as folder:
            root = self._copy_book(folder)
            (root / 'assets/diagrams/p12-rope.svg').unlink()
            self.assertIn('静态图源文件缺失：p12-rope',
                          check_render_manifest(root / 'dist', self.markdown, self.sizes))

    def test_orphan_rendered_file_fails_check_book(self):
        with TemporaryDirectory() as folder:
            root = self._copy_book(folder)
            for orphan in ('orphan.svg', 'flow-01.png', 'notes.txt'):
                with self.subTest(orphan=orphan):
                    path = root / 'dist/assets/rendered' / orphan
                    path.write_text('x')
                    self.assertIn(f'渲染目录里有清单之外的孤儿文件：assets/rendered/{orphan}',
                                  check_render_manifest(root / 'dist', self.markdown, self.sizes))
                    with self.assertRaisesRegex(ValueError, f'孤儿文件：assets/rendered/{re.escape(orphan)}'):
                        check_book(root / 'dist')
                    path.unlink()
            nested = root / 'dist/assets/rendered' / 'tmp' / 'nested.png'
            nested.parent.mkdir()
            nested.write_text('x')
            self.assertIn('渲染目录里有清单之外的孤儿文件：assets/rendered/tmp/nested.png',
                          check_render_manifest(root / 'dist', self.markdown, self.sizes))
            with self.assertRaisesRegex(ValueError, r'孤儿文件：assets/rendered/tmp/nested\.png'):
                check_book(root / 'dist')
            nested.unlink()
            nested.parent.rmdir()
            check_book(root / 'dist')


class ReferencedImagesTests(unittest.TestCase):
    """After the one-time rename, no page may point at a missing rendered image."""

    def test_every_rendered_image_referenced_by_the_site_exists(self):
        html = (DIST / 'index.html').read_text()
        refs = set(re.findall(r'(?:src|href)="(assets/rendered/[^"#?]+)"', html))
        manifest = json.loads((DIST / 'assets/rendered/manifest.json').read_text())['diagrams']
        # Mermaid figures whose source contains U+1D4B1 use PNG on the website;
        # everything else stays on SVG (see book_diagrams.diagram_asset_ext).
        from book_diagrams import SCRIPT_V, diagram_asset_ext
        script_v_ids = set()
        for path in sorted((ROOT / 'chapters').glob('*.md')):
            text = path.read_text(encoding='utf-8')
            for block in re.findall(r'```mermaid\n(.*?)```', text, re.S):
                if SCRIPT_V in block:
                    mid = re.search(r'%% id:\s*(\S+)', block)
                    self.assertIsNotNone(mid, f'missing %% id in {path}')
                    script_v_ids.add(f'flow-{mid.group(1)}')
        expected = set()
        for diagram_id in manifest:
            ext = 'png' if diagram_id in script_v_ids else 'svg'
            self.assertEqual(ext, diagram_asset_ext(
                SCRIPT_V if diagram_id in script_v_ids else 'no-script-v', False))
            expected.add(f'assets/rendered/{diagram_id}.{ext}')
        self.assertEqual(refs, expected)
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


# Every diagram header the pinned mermaid@11.17.2 recognises, plus zenuml (external plugin).
ALL_KEYWORDS = [
    'flowchart', 'flowchart-elk', 'graph', 'sequenceDiagram', 'classDiagram', 'classDiagram-v2',
    'stateDiagram', 'stateDiagram-v2', 'erDiagram', 'journey', 'gantt', 'pie', 'quadrantChart',
    'requirement', 'requirementDiagram', 'gitGraph', 'C4Context', 'C4Container', 'C4Component',
    'C4Dynamic', 'C4Deployment', 'mindmap', 'timeline', 'zenuml', 'sankey', 'sankey-beta',
    'xychart', 'xychart-beta', 'block', 'block-beta', 'packet', 'packet-beta', 'kanban',
    'architecture', 'architecture-beta', 'radar-beta', 'treemap', 'treemap-beta', 'treeView-beta',
    'swimlane-beta', 'eventmodeling', 'venn-beta', 'cynefin-beta', 'info',
    'ishikawa', 'ishikawa-beta', 'wardley-beta', 'railroad-beta', 'railroad-ebnf-beta',
    'railroad-abnf-beta', 'railroad-peg-beta',
]
MERMAID_JS = ROOT / 'node_modules/mermaid/dist/mermaid.js'


class LeakCheckTests(unittest.TestCase):
    def test_keyword_list_is_exactly_the_documented_list(self):
        self.assertEqual(sorted(MERMAID_KEYWORDS + MERMAID_KEYWORDS_ANY_CASE), sorted(ALL_KEYWORDS))
        docs = (ROOT / 'docs/diagrams.md').read_text()
        for keyword in ALL_KEYWORDS:
            with self.subTest(keyword=keyword):
                self.assertIn(f'`{keyword}`', docs)

    def test_every_diagram_keyword_is_detected(self):
        for keyword in ALL_KEYWORDS:
            for source in (f'{keyword}\n  a', f'  {keyword}\n    a\n    b',
                           f'%% id: p99-x\n%% accTitle: 甲\n%% accDescr: 乙\n{keyword}\n  a',
                           f'---\nconfig:\n  theme: base\n---\n{keyword}\n  a',
                           f'%%{{init: {{"theme": "base"}}}}%%\n{keyword}\n  a'):
                with self.subTest(source=source):
                    self.assertTrue(MERMAID_SOURCE.match(source))
        for keyword in MERMAID_KEYWORDS_ANY_CASE:
            with self.subTest(keyword=keyword.upper()):
                self.assertTrue(MERMAID_SOURCE.match(keyword.upper() + '\n  a'))

    def test_headers_with_arguments_are_detected(self):
        for source in [FLOW, 'graph LR\n  A-->B', 'graph TD;\n  A-->B',
                       'graph LR; A-->B\n', 'graph TD;A-->B\n',
                       'pie title 宠物\n  "猫" : 3',
                       'pie showData\n  "a" : 1', 'gitGraph LR:\n  commit', 'xychart-beta horizontal\n  x-axis [a]',
                       'info showInfo\n', 'sequenceDiagram\n  A->>B: x', 'stateDiagram-v2\n  [*] --> A', MINDMAP,
                       '%% id: p16-x\n%% accTitle: a\n%% accDescr: b\nblock-beta\n  a b', 'gantt\n  x :0, 1',
                       '---\nconfig:\n  theme: base\n---\nclassDiagram\n  A <|-- B',
                       'architecture-beta\n  service a(server)[A]', 'packet-beta\n  0-15: "Source Port"',
                       'requirementDiagram\n  requirement r {\n  }', 'C4Context\n  title 系统',
                       'zenuml\n  A.method()', 'kanban\n  todo[待办]', 'radar-beta\n  axis a, b',
                       'treemap-beta\n  "a": 1', 'sankey-beta\n  a,b,1', 'quadrantChart\n  x-axis a --> b',
                       'timeline\n  2017 : Transformer', 'journey\n  title x', 'erDiagram\n  A ||--o{ B : has',
                       'cynefin-beta:\n  a', 'cynefin-beta:\n', 'info\n']:
            with self.subTest(source=source[:30]):
                self.assertTrue(MERMAID_SOURCE.match(source))

    def test_ordinary_code_is_not_flagged(self):
        for text in ['timeline', 'pie', 'gantt', 'x = 1\nflowchart TD', 'graph = build()\nprint(graph)',
                     'info = load()\nprint(info)', 'block: int = 4\n', 'packet.send()\n', 'graphs\n  x',
                     'timeline(events)\n', 'pie, cake = 1, 2\n', 'requirements.txt\n', 'kanban_board = []\n',
                     'architecture: str\n', 'blocks = [a, b]\n', 'C4ContextLoader()\n', 'flowchart-v3\n  x',
                     '[0.1, 0.2]\n', 'python build.py\n',
                     'graph of the function\n', 'journey home tomorrow\n', 'pie chart data\n',
                     'info about config\n', 'timeline of events\n']:
            with self.subTest(text=text):
                self.assertFalse(MERMAID_SOURCE.match(text))

    @unittest.skipUnless(MERMAID_JS.is_file(), 'needs node_modules/mermaid (see docs/diagrams.md)')
    def test_covers_every_detector_in_installed_mermaid(self):
        version = json.loads((ROOT / 'node_modules/mermaid/package.json').read_text())['version']
        self.assertEqual(version, '11.17.2')
        source = MERMAID_JS.read_text()
        from_return, from_if = scan_mermaid_js_detectors(source)
        detectors = from_return + from_if
        self.assertGreaterEqual(len(detectors), 30)
        self.assertTrue(any('flowchart-elk' in d for d in detectors), 'should see flowchart-elk from if (/…/)')
        self.assertTrue(any('stateDiagram-v2' in d for d in detectors), 'should see stateDiagram-v2 from if (/…/)')
        for pattern in detectors:
            depth, alternatives = 0, ['']
            for char in pattern:   # split on top-level `|` only (C4Context|C4Container|...)
                depth += {'(': 1, ')': -1}.get(char, 0)
                if char == '|' and depth == 0:
                    alternatives.append('')
                else:
                    alternatives[-1] += char
            for alternative in alternatives:
                head = re.match(r'([A-Za-z0-9-]+)', alternative).group(1)
                with self.subTest(detector=pattern, header=head):
                    self.assertTrue(MERMAID_SOURCE.match(head + '\n  a'))


class RenderBackupTests(unittest.TestCase):
    """build.py keeps the last render outside dist/ until the build succeeds."""

    def setUp(self):
        import build
        self.build = build
        self.folder = TemporaryDirectory()
        root = Path(self.folder.name)
        self.rendered = root / 'dist/assets/rendered'
        self.backup = str(root / '.render-backup')
        self.rendered.mkdir(parents=True)
        for name, text in (('manifest.json', '{}'), ('flow-a.svg', 'old')):
            (self.rendered / name).write_text(text)

    def tearDown(self):
        self.folder.cleanup()

    def test_backup_is_outside_dist_and_gitignored(self):
        self.assertEqual(Path(self.build.RENDER_BACKUP).parent, ROOT)
        ignored = subprocess.run(['git', 'check-ignore', '-q', '.render-backup/x'], cwd=ROOT)
        self.assertEqual(ignored.returncode, 0)

    def test_restore_puts_back_the_old_render(self):
        cache = self.build.backup_render(str(self.rendered), self.backup)
        self.assertEqual(cache, self.backup)
        shutil.rmtree(self.rendered)          # build.py wipes dist/assets
        self.rendered.mkdir()
        (self.rendered / 'flow-a.svg').write_text('half-written')   # crash mid-render
        self.build.restore_render(self.backup, str(self.rendered))
        self.assertEqual((self.rendered / 'flow-a.svg').read_text(), 'old')
        self.assertTrue((self.rendered / 'manifest.json').is_file())
        self.assertFalse(Path(self.backup).exists())

    def test_crash_during_restore_keeps_backup_and_next_build_reuses_it(self):
        self.build.backup_render(str(self.rendered), self.backup)
        shutil.rmtree(self.rendered)
        real_copytree = shutil.copytree

        def crash(*args, **kwargs):
            raise OSError('disk full')
        shutil.copytree = crash
        try:
            with self.assertRaises(OSError):
                self.build.restore_render(self.backup, str(self.rendered))
        finally:
            shutil.copytree = real_copytree
        self.assertEqual((Path(self.backup) / 'flow-a.svg').read_text(), 'old')
        # Next build: dist has no complete render, so the leftover backup is the cache.
        self.assertEqual(self.build.backup_render(str(self.rendered), self.backup), self.backup)
        self.assertEqual((Path(self.backup) / 'flow-a.svg').read_text(), 'old')

    def test_complete_dist_render_replaces_stale_leftover_backup(self):
        Path(self.backup).mkdir()
        (Path(self.backup) / 'flow-a.svg').write_text('older')
        Path(self.backup + '.tmp').mkdir()   # half-copied backup from an earlier crash
        self.build.backup_render(str(self.rendered), self.backup)
        self.assertEqual((Path(self.backup) / 'flow-a.svg').read_text(), 'old')
        self.assertFalse(Path(self.backup + '.tmp').exists())

    def test_no_render_no_backup(self):
        shutil.rmtree(self.rendered)
        self.assertIsNone(self.build.backup_render(str(self.rendered), self.backup))



class D0FollowUpMutationTests(unittest.TestCase):
    """Each assertion fails if the corresponding D0 follow-up fix is reverted."""

    def test_docs_and_comment_do_not_require_a_body_line(self):
        docs = (ROOT / 'docs/diagrams.md').read_text()
        comment = (ROOT / 'scripts/check_book.py').read_text()
        self.assertNotIn('并且后面还有正文行', docs)
        self.assertNotIn('needs a body line after it', comment)
        self.assertIn('不要求', docs)
        self.assertIn('body line after it is not required', comment)

    def test_ambiguous_keyword_plus_ordinary_word_is_not_a_diagram(self):
        for text in ('graph of the function\n', 'journey home tomorrow\n', 'pie chart data\n',
                     'info about config\n', 'timeline of events\n'):
            with self.subTest(text=text):
                self.assertFalse(MERMAID_SOURCE.match(text))

    def test_cynefin_beta_colon_header_is_detected(self):
        self.assertTrue(MERMAID_SOURCE.match('cynefin-beta:\n'))
        self.assertTrue(MERMAID_SOURCE.match('cynefin-beta:\n  a'))

    def test_installed_mermaid_scan_includes_if_slash_detectors(self):
        if not MERMAID_JS.is_file():
            self.skipTest('needs node_modules/mermaid')
        source = MERMAID_JS.read_text()
        from_return, from_if = scan_mermaid_js_detectors(source)
        self.assertTrue(from_if, 'mermaid.js still uses if (/…/) detectors')
        self.assertFalse(any('flowchart-elk' in d for d in from_return))
        self.assertTrue(any('flowchart-elk' in d for d in from_if))

    def test_same_line_direction_args_are_detected(self):
        for source in ('graph LR; A-->B\n', 'graph TD;A-->B\n', 'graph RL;\n', 'graph BT A-->B\n'):
            with self.subTest(source=source):
                self.assertTrue(MERMAID_SOURCE.match(source))

    def test_nested_orphan_reports_file_path_not_just_directory(self):
        with TemporaryDirectory() as folder:
            root = ManifestGateTests('test_orphan_rendered_file_fails_check_book')._copy_book(folder)
            nested = root / 'dist/assets/rendered/tmp/nested.png'
            nested.parent.mkdir()
            nested.write_text('x')
            markdown = (root / 'dist/transformer_from_scratch.md').read_text()
            sizes = json.loads((root / 'dist/assets/rendered/dimensions.json').read_text())
            errors = check_render_manifest(root / 'dist', markdown, sizes)
            self.assertTrue(any(e.endswith('assets/rendered/tmp/nested.png') for e in errors), errors)
            self.assertFalse(any(e.rstrip('/').endswith('assets/rendered/tmp') for e in errors), errors)

    def test_docs_backup_wording_matches_restore_path(self):
        docs = (ROOT / 'docs/diagrams.md').read_text()
        self.assertNotIn('备份只在构建成功后删除', docs)
        self.assertIn('构建**成功**后才删除备份', docs)
        self.assertIn('换好之后**才删备份', docs)

    def test_dist_copy_and_missing_source_are_separate_tests(self):
        source = (ROOT / 'tests/test_diagram_ids.py').read_text()
        self.assertIn('def test_edited_dist_svg_copy_fails(self):', source)
        self.assertIn('def test_missing_repo_svg_source_fails(self):', source)
        self.assertNotIn('def test_edited_dist_svg_copy_or_' + 'missing_source_fails(self):', source)


class ScriptVWebAssetTests(unittest.TestCase):
    """HTML must use PNG for any Mermaid figure whose source contains U+1D4B1."""

    def test_diagram_asset_ext_png_when_script_v_or_epub(self):
        from book_diagrams import SCRIPT_V, diagram_asset_ext
        self.assertEqual(diagram_asset_ext('logits [B,T,|%s|]' % SCRIPT_V, False), 'png')
        self.assertEqual(diagram_asset_ext('no special glyph', False), 'svg')
        self.assertEqual(diagram_asset_ext('no special glyph', True), 'png')
        self.assertEqual(diagram_asset_ext('', False), 'svg')

    def test_site_uses_png_for_script_v_diagrams(self):
        from book_diagrams import SCRIPT_V
        html = (DIST / 'index.html').read_text(encoding='utf-8')
        self.assertIn(SCRIPT_V, (ROOT / 'chapters' / '08l_preference_optimization.md').read_text(encoding='utf-8'))
        # Both img src and 原尺寸 href for the logp-pair figure must be PNG.
        self.assertIn('assets/rendered/flow-p22-logp-pair.png', html)
        self.assertNotIn('href="assets/rendered/flow-p22-logp-pair.svg"', html)
        self.assertNotIn('src="assets/rendered/flow-p22-logp-pair.svg"', html)


class ScriptVFontTests(unittest.TestCase):
    """D7 (bot-hub#126): |𝒱| must not render as tofu; CI checks config + SVG."""

    SCRIPT_V = '\U0001D4B1'
    RENDERER = ROOT / 'scripts' / 'render_diagrams.mjs'
    FONT_CANDIDATES = (
        Path('/workspace/tools/fonts/NotoSansMath-Regular.ttf'),
        Path.home() / '.local/share/fonts/NotoSansMath-Regular.ttf',
    )

    def test_renderer_font_family_includes_noto_sans_math(self):
        src = self.RENDERER.read_text(encoding='utf-8')
        self.assertIn('Noto Sans Math', src)
        # Keep CJK ahead so Chinese stays on CJK; Math is the fallback for U+1D4B1.
        cjk = src.find('Noto Sans CJK SC')
        math = src.find('Noto Sans Math')
        self.assertGreater(cjk, 0)
        self.assertGreater(math, cjk)

    def test_committed_svgs_keep_script_v_when_source_uses_it(self):
        """If a Mermaid block contains 𝒱, its committed SVG must too (not U+FFFD)."""
        md = (DIST / 'transformer_from_scratch.md').read_text(encoding='utf-8')
        checked = 0
        for definition in mermaid_blocks(md):
            if self.SCRIPT_V not in definition:
                continue
            diagram_id = mermaid_id(definition)
            svg_path = DIST / 'assets' / 'rendered' / f'{diagram_id}.svg'
            self.assertTrue(svg_path.is_file(), f'missing {svg_path}')
            svg = svg_path.read_text(encoding='utf-8')
            self.assertIn(self.SCRIPT_V, svg, f'{diagram_id}.svg lost 𝒱')
            self.assertNotIn('\uFFFD', svg)
            checked += 1
        self.assertGreaterEqual(checked, 1, 'expected at least one diagram using 𝒱')

    def test_installed_noto_sans_math_covers_script_v(self):
        font_path = next((p for p in self.FONT_CANDIDATES if p.is_file()), None)
        if font_path is None:
            self.skipTest('Noto Sans Math not installed on this machine (OK for CI)')
        try:
            from fontTools.ttLib import TTFont
        except ImportError:
            self.skipTest('fontTools not installed')
        font = TTFont(str(font_path))
        has = any(0x1D4B1 in table.cmap for table in font['cmap'].tables)
        self.assertTrue(has, f'{font_path} cmap missing U+1D4B1')

if __name__ == '__main__':
    unittest.main()
