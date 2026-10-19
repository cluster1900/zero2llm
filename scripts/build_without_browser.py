#!/usr/bin/env python3
"""受限构建机的静态路径：只复用完全相同的已提交 Mermaid 图，SVG 用 CairoSVG 转 PNG。

不启动浏览器，不更改浏览器安全配置。新/改动 Mermaid 必须走正常 build.py。
用法：python scripts/build_without_browser.py --baseline <已有成品的 Git commit>
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import build  # noqa: E402


def baseline_bytes(commit, path):
    return subprocess.check_output(['git', 'show', f'{commit}:{path}'], cwd=ROOT)


def make_renderer(commit):
    # 先解析基准原稿，使用与正常构建相同的 Pandoc AST 提取定义。
    markdown = baseline_bytes(commit, 'dist/transformer_from_scratch.md')
    parsed = subprocess.run(['pandoc', '-f', build.MD_FORMAT, '-t', 'json'],
                            input=markdown, capture_output=True, check=True)
    definitions = []

    def collect(value):
        if isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            if value.get('t') == 'CodeBlock' and 'mermaid' in value['c'][0][1]:
                definitions.append(value['c'][1])
            for item in value.values():
                collect(item)

    collect(json.loads(parsed.stdout))
    cache = {definition: f'flow-{index:02d}' for index, definition in enumerate(definitions, 1)}
    sizes = json.loads(baseline_bytes(commit, 'dist/assets/rendered/dimensions.json'))

    def render(jobs, output):
        import cairosvg
        from importlib.metadata import version
        if version('CairoSVG') != '2.8.2':
            raise ValueError('静态构建验证版本为 CairoSVG==2.8.2')
        output.mkdir(parents=True, exist_ok=True)
        dimensions = {}
        reused, static_reused, rasterized = 0, 0, 0
        for job in jobs:
            name = job['id']
            if 'definition' in job:
                if job['definition'] not in cache:
                    raise ValueError(f'{name}: Mermaid 源码有变化，不能复用旧图；请在可用 Chromium 的环境运行 build.py')
                old_id = cache[job['definition']]
                for ext in ('svg', 'png'):
                    (output / f'{name}.{ext}').write_bytes(baseline_bytes(commit, f'dist/assets/rendered/{old_id}.{ext}'))
                dimensions[name] = sizes[old_id]
                reused += 1
            else:
                source = Path(job['source'])
                # 未改变的静态图也保留原渲染，避免无意义的像素差异。
                if name in sizes and source.read_bytes() == baseline_bytes(commit, source.relative_to(ROOT).as_posix()):
                    for ext in ('svg', 'png'):
                        (output / f'{name}.{ext}').write_bytes(baseline_bytes(commit, f'dist/assets/rendered/{name}.{ext}'))
                    dimensions[name] = sizes[name]
                    static_reused += 1
                    continue
                data = re.sub(rb"[\t ]+$", b"", source.read_bytes(), flags=re.M)
                match = re.search(rb'viewBox="([^"]+)"', data)
                if not match:
                    raise ValueError(f'{name}: SVG 缺少 viewBox')
                _, _, width, height = map(float, match[1].split())
                dimensions[name] = {'width': width, 'height': height}
                (output / f'{name}.svg').write_bytes(data)
                cairosvg.svg2png(bytestring=data, write_to=str(output / f'{name}.png'), scale=2)
                rasterized += 1
        (output / 'dimensions.json').write_text(json.dumps(dimensions, ensure_ascii=False, indent=2) + '\n')
        print(f'静态渲染：{reused} 张源码完全匹配的 Mermaid 复用基准 {commit}；{static_reused} 张未变静态图复用；{rasterized} 张 SVG 由 CairoSVG 2.8.2 栅格化')

    return render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, help='含完整已验证成品的 Git commit；需要本地 Git 历史')
    args = parser.parse_args()
    commit = subprocess.check_output(['git', 'rev-parse', '--verify', f'{args.baseline}^{{commit}}'], cwd=ROOT, text=True).strip()
    return build.main(renderer=make_renderer(commit))


if __name__ == '__main__':
    sys.exit(main())
