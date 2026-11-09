#!/usr/bin/env python3
"""受限构建机的静态路径：不启动浏览器，只复用已提交的渲染，新 SVG 用 CairoSVG 转 PNG。

图表按稳定 id 和源码 sha256 复用（见 docs/diagrams.md）：
- Mermaid 图只要 dist/assets/rendered/manifest.json 里有同一 id、同一 sha256 的渲染，
  就直接复用，新增的图也一样（例如别人已在能启动 Chromium 的机器上渲染并提交）；
  没有可复用的渲染就报错，请在能启动 Chromium 的环境运行 build.py。
- 静态 SVG 源文件未变时复用；变了或新增时用 CairoSVG 2.8.2 栅格化。
不更改浏览器安全配置。

用法：
  python scripts/build_without_browser.py                    # 复用工作区 dist/assets/rendered
  python scripts/build_without_browser.py --baseline <commit> # 复用某个 commit 里已提交的渲染
"""
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import build  # noqa: E402

CAIROSVG_VERSION = '2.8.2'
RENDERER_LABEL = f'scripts/build_without_browser.py CairoSVG=={CAIROSVG_VERSION}'


def export_baseline(commit, target):
    """Write <commit>:dist/assets/rendered/* into `target` (a render cache directory)."""
    listing = subprocess.check_output(
        ['git', 'ls-tree', '--name-only', f'{commit}:dist/assets/rendered'], cwd=ROOT, text=True)
    target.mkdir(parents=True, exist_ok=True)
    for name in listing.split():
        data = subprocess.check_output(['git', 'show', f'{commit}:dist/assets/rendered/{name}'], cwd=ROOT)
        (target / name).write_bytes(data)
    if not (target / 'manifest.json').is_file():
        raise ValueError(f'{commit} 的 dist/assets/rendered 没有 manifest.json（早于 D0），不能按 id 复用；'
                         '请换用 D0 之后的 commit，或不加 --baseline 直接复用工作区')


def render(jobs, output):
    """Only called for figures with no reusable render."""
    mermaid = [job['id'] for job in jobs if job['kind'] == 'mermaid']
    if mermaid:
        raise ValueError(f'{", ".join(mermaid)}：Mermaid 源码有变化或是新图，且没有已提交的同源码渲染，'
                         '不能复用旧图；请在可用 Chromium 的环境运行 build.py（见 docs/diagrams.md）')
    import cairosvg
    from importlib.metadata import version
    if version('CairoSVG') != CAIROSVG_VERSION:
        raise ValueError(f'静态构建验证版本为 CairoSVG=={CAIROSVG_VERSION}')
    import json
    dimensions = {}
    for job in jobs:
        data = re.sub(rb'[\t ]+$', b'', Path(job['source']).read_bytes(), flags=re.M)
        match = re.search(rb'viewBox="([^"]+)"', data)
        if not match:
            raise ValueError(f'{job["id"]}: SVG 缺少 viewBox')
        _, _, width, height = map(float, match[1].split())
        dimensions[job['id']] = {'width': width, 'height': height}
        (output / f'{job["id"]}.svg').write_bytes(data)
        cairosvg.svg2png(bytestring=data, write_to=str(output / f'{job["id"]}.png'), scale=2)
    (output / 'dimensions.json').write_text(json.dumps(dimensions, ensure_ascii=False, indent=2) + '\n')
    print(f'静态渲染：{len(jobs)} 张 SVG 由 CairoSVG {CAIROSVG_VERSION} 栅格化')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--baseline', help='从这个 Git commit 的 dist/assets/rendered 复用渲染（需 D0 之后的 commit）；'
                                           '默认复用工作区里的 dist/assets/rendered')
    args = parser.parse_args()
    if not args.baseline:
        return build.main(renderer=render, renderer_label=RENDERER_LABEL)
    commit = subprocess.check_output(['git', 'rev-parse', '--verify', f'{args.baseline}^{{commit}}'],
                                     cwd=ROOT, text=True).strip()
    with tempfile.TemporaryDirectory(prefix='zero2llm-baseline-') as folder:
        cache = Path(folder) / 'rendered'
        export_baseline(commit, cache)
        return build.main(renderer=render, cache_dir=str(cache), renderer_label=RENDERER_LABEL)


if __name__ == '__main__':
    sys.exit(main())
