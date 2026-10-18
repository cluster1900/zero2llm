"""静态构建器的缓存安全边界；不启动浏览器。"""
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import build_without_browser as static


@unittest.skipUnless(importlib.util.find_spec('cairosvg'), '选修：未安装 CairoSVG')
class StaticRendererTests(unittest.TestCase):
    def renderer(self):
        ast = {'t': 'CodeBlock', 'c': [['', ['mermaid'], []], 'flowchart TD\nA-->B']}
        def baseline(commit, path):
            if path.endswith('dimensions.json'):
                return json.dumps({'flow-01': {'width': 300, 'height': 200}}).encode()
            return b'fixture'
        with patch.object(static, 'baseline_bytes', side_effect=baseline), patch.object(
                static.subprocess, 'run', return_value=SimpleNamespace(stdout=json.dumps(ast).encode())):
            return static.make_renderer('fixed-commit')

    def test_changed_mermaid_is_rejected(self):
        renderer = self.renderer()
        with TemporaryDirectory() as folder, self.assertRaisesRegex(ValueError, 'Mermaid 源码有变化'):
            renderer([{'id': 'flow-01', 'definition': 'flowchart TD\nA-->C'}], Path(folder))

    def test_matching_mermaid_reuses_exact_bytes(self):
        renderer = self.renderer()
        with TemporaryDirectory() as folder, patch.object(static, 'baseline_bytes', return_value=b'cached-image'):
            output = Path(folder)
            renderer([{'id': 'flow-02', 'definition': 'flowchart TD\nA-->B'}], output)
            self.assertEqual((output / 'flow-02.svg').read_bytes(), b'cached-image')
            self.assertEqual((output / 'flow-02.png').read_bytes(), b'cached-image')
            self.assertEqual(json.loads((output / 'dimensions.json').read_text())['flow-02']['width'], 300)
