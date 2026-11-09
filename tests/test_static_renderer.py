"""静态构建器（build_without_browser.py）的缓存安全边界；不启动浏览器。"""
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import build_without_browser as static  # noqa: E402
from book_diagrams import render_all  # noqa: E402
from diagram_ids import mermaid_id, sha256_bytes, sha256_text  # noqa: E402

FLOW = 'flowchart TD\n    accTitle: 甲\n    accDescr: 乙\n    A-->B'
SVG = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 200"><rect width="300" height="200"/></svg>'


def cache_with(folder, job, svg=b'cached-svg', png=b'cached-png'):
    """A committed render directory holding exactly one figure."""
    cache = Path(folder) / 'cache'
    cache.mkdir()
    (cache / f'{job["id"]}.svg').write_bytes(svg)
    (cache / f'{job["id"]}.png').write_bytes(png)
    (cache / 'dimensions.json').write_text(json.dumps({job['id']: {'width': 300, 'height': 200}}))
    (cache / 'manifest.json').write_text(json.dumps({'diagrams': {job['id']: {
        'kind': job['kind'], 'sha256': job['sha256'], 'renderer': 'chromium'}}}))
    return cache


def mermaid_job(definition):
    return {'id': mermaid_id(definition), 'kind': 'mermaid', 'definition': definition,
            'sha256': sha256_text(definition)}


class StaticRendererTests(unittest.TestCase):
    def test_changed_mermaid_is_rejected(self):
        with TemporaryDirectory() as folder:
            cache = cache_with(folder, mermaid_job(FLOW))
            changed = mermaid_job(FLOW.replace('A-->B', 'A-->C'))
            with self.assertRaisesRegex(ValueError, 'Mermaid 源码有变化'):
                render_all([changed], Path(folder) / 'out', cache_dir=cache, renderer=static.render)

    def test_changed_mermaid_with_same_explicit_id_is_rejected(self):
        old = '%% id: same-name\n' + FLOW
        with TemporaryDirectory() as folder:
            cache = cache_with(folder, mermaid_job(old))
            new = mermaid_job(old.replace('A-->B', 'A-->C'))
            self.assertEqual(new['id'], 'flow-same-name')
            with self.assertRaisesRegex(ValueError, 'Mermaid 源码有变化'):
                render_all([new], Path(folder) / 'out', cache_dir=cache, renderer=static.render)

    def test_matching_mermaid_reuses_exact_bytes(self):
        job = mermaid_job(FLOW)
        with TemporaryDirectory() as folder:
            cache = cache_with(folder, job, svg=b'cached-image', png=b'cached-image')
            output = Path(folder) / 'out'
            reused, rendered = render_all([job], output, cache_dir=cache, renderer=static.render)
            self.assertEqual((reused, rendered), ([job['id']], []))
            self.assertEqual((output / f'{job["id"]}.svg').read_bytes(), b'cached-image')
            self.assertEqual((output / f'{job["id"]}.png').read_bytes(), b'cached-image')
            self.assertEqual(json.loads((output / 'dimensions.json').read_text())[job['id']]['width'], 300)

    def test_new_mermaid_with_committed_render_is_accepted(self):
        # A diagram that is new relative to any older baseline, but whose render was
        # already committed (made where Chromium runs), builds without a browser.
        job = mermaid_job('%% id: p99-new-figure\n' + FLOW)
        with TemporaryDirectory() as folder:
            cache = cache_with(folder, job)
            output = Path(folder) / 'out'
            render_all([job], output, cache_dir=cache, renderer=static.render)
            manifest = json.loads((output / 'manifest.json').read_text())['diagrams']
            self.assertEqual(manifest['flow-p99-new-figure']['sha256'], job['sha256'])
            self.assertEqual(manifest['flow-p99-new-figure']['renderer'], 'chromium')

    @unittest.skipUnless(importlib.util.find_spec('cairosvg'), '选修：未安装 CairoSVG')
    def test_changed_static_svg_is_rasterized(self):
        with TemporaryDirectory() as folder:
            source = Path(folder) / 'demo.svg'
            source.write_bytes(SVG)
            job = {'id': 'demo', 'kind': 'static', 'path': 'assets/diagrams/demo.svg', 'source': str(source),
                   'sha256': sha256_bytes(SVG)}
            output = Path(folder) / 'out'
            reused, rendered = render_all([job], output, renderer=static.render,
                                          renderer_label=static.RENDERER_LABEL)
            self.assertEqual((reused, rendered), ([], ['demo']))
            self.assertEqual((output / 'demo.png').read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
            entry = json.loads((output / 'manifest.json').read_text())['diagrams']['demo']
            self.assertEqual(entry, {'kind': 'static', 'sha256': sha256_bytes(SVG),
                                     'renderer': static.RENDERER_LABEL, 'source': 'assets/diagrams/demo.svg'})


if __name__ == '__main__':
    unittest.main()
