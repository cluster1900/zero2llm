"""Replace Mermaid code blocks with packaged images in Pandoc's document tree.

Every figure gets a stable asset id (see scripts/diagram_ids.py). A figure whose
source is unchanged since the last build is copied from the previous render
cache instead of being rendered again, so a build only launches the browser
renderer for new or changed figures, and the bytes of untouched figures never
drift with fonts or Chromium versions.
"""
import copy
import json
import os
import shutil
import subprocess
from pathlib import Path

try:
    from scripts.diagram_ids import (DIMENSIONS, LEGACY_RENDERER, MANIFEST, MERMAID_PREFIX,
                                     mermaid_id, mermaid_labels, sha256_bytes, sha256_text)
except ImportError:  # scripts/ on sys.path (tests, build_without_browser.py)
    from diagram_ids import (DIMENSIONS, LEGACY_RENDERER, MANIFEST, MERMAID_PREFIX,  # noqa: F401
                             mermaid_id, mermaid_labels, sha256_bytes, sha256_text)

# Set ZERO2LLM_RERENDER=1 to ignore the cache, e.g. after changing render_diagrams.mjs.
FORCE_ENV = 'ZERO2LLM_RERENDER'


def node_renderer_label(base):
    """Provenance string recorded for figures rendered by render_diagrams.mjs."""
    versions = {}
    try:
        packages = json.loads((Path(base) / 'package-lock.json').read_text())['packages']
        for name in ('mermaid', '@mermaid-js/mermaid-cli', 'puppeteer'):
            versions[name] = packages[f'node_modules/{name}']['version']
    except (OSError, KeyError, ValueError):
        pass
    pinned = ' '.join(f'{name}@{version}' for name, version in versions.items())
    return f'scripts/render_diagrams.mjs {pinned}'.strip()


def collect_jobs(document, base):
    """Walk the Pandoc AST once; return (jobs, mermaid_labels_by_definition, static_ids)."""
    jobs = []
    labels = {}
    static_ids = {}
    seen = {}

    def add(job):
        previous = seen.get(job['id'])
        if previous is None:
            seen[job['id']] = job['sha256']
            jobs.append(job)
        elif previous != job['sha256']:
            raise ValueError(f'两张不同的图用了同一个 id：{job["id"]}；请改其中一张的 %% id: 或文件名')

    def walk(value):
        if isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            if value.get('t') == 'CodeBlock' and 'mermaid' in value['c'][0][1]:
                definition = value['c'][1]
                title, description = mermaid_labels(definition)
                diagram_id = mermaid_id(definition)
                labels[definition] = (diagram_id, title, description)
                add({'id': diagram_id, 'kind': 'mermaid', 'definition': definition,
                     'sha256': sha256_text(definition)})
            elif value.get('t') == 'Image':
                source = value['c'][2][0]
                if source.endswith('.svg') and source not in static_ids:
                    diagram_id = Path(source).stem
                    if diagram_id.startswith(MERMAID_PREFIX):
                        raise ValueError(f'静态图文件名不能以 {MERMAID_PREFIX} 开头（留给 Mermaid）：{source}')
                    static_ids[source] = diagram_id
                    add({'id': diagram_id, 'kind': 'static', 'path': source,
                         'source': str(Path(base) / source),
                         'sha256': sha256_bytes((Path(base) / source).read_bytes())})
            for item in value.values():
                walk(item)

    walk(document)
    return jobs, labels, static_ids


def read_cache(cache_dir):
    """(manifest entries, dimensions) of a previous render directory, or empty."""
    if not cache_dir:
        return {}, {}
    cache = Path(cache_dir)
    try:
        manifest = json.loads((cache / MANIFEST).read_text())['diagrams']
        sizes = json.loads((cache / DIMENSIONS).read_text())
    except (OSError, KeyError, ValueError):
        return {}, {}
    return manifest, sizes


def reusable(job, cache_dir, manifest, sizes):
    entry = manifest.get(job['id'])
    return bool(entry and entry.get('sha256') == job['sha256'] and job['id'] in sizes
                and all((Path(cache_dir) / f'{job["id"]}.{ext}').is_file() for ext in ('svg', 'png')))


def render_all(jobs, output, cache_dir=None, renderer=None, renderer_label=None, force=False):
    """Fill `output` with <id>.svg/.png, dimensions.json and manifest.json for `jobs`.

    Unchanged figures are copied from `cache_dir`; only the rest go to `renderer`
    (a callable taking (jobs, output_dir) that writes <id>.svg/.png and a
    dimensions.json for the jobs it was given). Returns (reused_ids, rendered_ids).
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest, sizes = ({}, {}) if force else read_cache(cache_dir)
    reused, pending = [], []
    for job in jobs:
        (reused if reusable(job, cache_dir, manifest, sizes) else pending).append(job)
    for job in reused:
        for ext in ('svg', 'png'):
            shutil.copyfile(Path(cache_dir) / f'{job["id"]}.{ext}', output / f'{job["id"]}.{ext}')
    fresh = {}
    if pending:
        work = output / '.render-tmp'
        if work.exists():
            shutil.rmtree(work)
        work.mkdir()
        try:
            renderer(pending, work)
            fresh = json.loads((work / DIMENSIONS).read_text())
            for job in pending:
                if job['id'] not in fresh:
                    raise ValueError(f'渲染器没有输出 {job["id"]} 的尺寸')
                for ext in ('svg', 'png'):
                    shutil.move(str(work / f'{job["id"]}.{ext}'), output / f'{job["id"]}.{ext}')
        finally:
            shutil.rmtree(work, ignore_errors=True)
    dimensions, entries = {}, {}
    for job in jobs:
        if job['id'] in fresh:
            dimensions[job['id']] = fresh[job['id']]
            provenance = renderer_label or 'unknown renderer'
        else:
            dimensions[job['id']] = sizes[job['id']]
            provenance = manifest[job['id']].get('renderer', LEGACY_RENDERER)
        entry = {'kind': job['kind'], 'sha256': job['sha256'], 'renderer': provenance}
        if job['kind'] == 'static':
            entry['source'] = job['path']
        entries[job['id']] = entry
    (output / DIMENSIONS).write_text(json.dumps(dimensions, ensure_ascii=False, indent=2) + '\n')
    (output / MANIFEST).write_text(json.dumps({'diagrams': entries}, ensure_ascii=False, indent=2) + '\n')
    return [job['id'] for job in reused], [job['id'] for job in pending]


def node_render(base):
    def render(jobs, output):
        if not (Path(base) / 'node_modules' / '@mermaid-js' / 'mermaid-cli').is_dir():
            raise ValueError('有新增或改动的图需要渲染，但缺少图表构建依赖：请先运行 npm ci（见 docs/diagrams.md）')
        jobs_path = Path(output) / 'jobs.json'
        jobs_path.write_text(json.dumps(
            [{key: job[key] for key in ('id', 'definition', 'source') if key in job} for job in jobs],
            ensure_ascii=False), encoding='utf-8')
        subprocess.run(['node', str(Path(base) / 'scripts/render_diagrams.mjs'), str(jobs_path), str(output)],
                       cwd=base, check=True)
    return render


def prepare_documents(merged_md, temp_dir, base_dir, markdown_format, renderer=None, cache_dir=None,
                      renderer_label=None):
    base = Path(base_dir)
    tmp = Path(temp_dir)
    parsed = subprocess.run(
        ['pandoc', str(merged_md), '-f', markdown_format, '-t', 'json'],
        cwd=base, check=True, capture_output=True, text=True,
    )
    document = json.loads(parsed.stdout)
    jobs, definitions, static_images = collect_jobs(document, base)
    output = base / 'dist/assets/rendered'
    if renderer is None:
        renderer, renderer_label = node_render(base), node_renderer_label(base)
    force = os.environ.get(FORCE_ENV) == '1'
    reused, rendered = render_all(jobs, output, cache_dir=cache_dir, renderer=renderer,
                                  renderer_label=renderer_label, force=force)
    sizes = json.loads((output / DIMENSIONS).read_text())

    def image(diagram_id, title, description, epub):
        size = sizes[diagram_id]
        # Fit within a compact reading panel, with proportional dimensions and no upscaling.
        # CSS additionally handles narrower screens and landscape reader viewports.
        max_width, max_height = (560, 480) if epub else (720, 560)
        width = round(min(size['width'], max_width, max_height * size['width'] / size['height']))
        ext = 'png' if epub else 'svg'
        src = f'assets/rendered/{diagram_id}.{ext}'
        return {'t': 'Image', 'c': [
            ['', ['book-diagram'], [['width', f'{width}px']]],
            [{'t': 'Str', 'c': f'{title}。{description}'}], [src, ''],
        ]}

    def convert(value, epub):
        if isinstance(value, list):
            return [convert(item, epub) for item in value]
        if not isinstance(value, dict):
            return value
        if value.get('t') == 'CodeBlock' and 'mermaid' in value['c'][0][1]:
            diagram_id, title, description = definitions[value['c'][1]]
            img = image(diagram_id, title, description, epub)
            if not epub:
                img = {'t': 'Link', 'c': [
                    ['', [], []], [img], [f'assets/rendered/{diagram_id}.svg', '点击查看原尺寸图'],
                ]}
            return {'t': 'Div', 'c': [['', ['diagram-figure'], []], [
                {'t': 'Plain', 'c': [img]},
                {'t': 'Div', 'c': [['', ['diagram-caption'], []], [{'t': 'Plain', 'c': [{'t': 'Str', 'c': title}]}]]},
            ]]}
        if value.get('t') == 'Image' and value['c'][2][0] in static_images:
            original = value['c']
            diagram_id = static_images[original[2][0]]
            replacement = image(diagram_id, '', '', epub)
            replacement['c'][1] = original[1]
            if not epub:
                return {'t': 'Link', 'c': [
                    ['', [], []], [replacement],
                    [f'assets/rendered/{diagram_id}.svg', '点击查看原尺寸图'],
                ]}
            return replacement
        if epub and value.get('t') == 'Link' and value['c'][2][0].startswith('../code/'):
            value['c'][2][0] = 'https://github.com/cluster1900/zero2llm/blob/main/code/' + value['c'][2][0][8:]
        return {key: convert(item, epub) for key, item in value.items()}

    paths = []
    for kind in ('epub', 'html'):
        target = tmp / f'{kind}.json'
        target.write_text(json.dumps(convert(copy.deepcopy(document), kind == 'epub'), ensure_ascii=False), encoding='utf-8')
        paths.append(str(target))
    mermaid_count = sum(1 for job in jobs if job['kind'] == 'mermaid')
    print(f'✓ 图表：{mermaid_count} 张 Mermaid 流程图与 {len(static_images)} 张原有插图；'
          f'复用未改动的渲染 {len(reused)} 张，新渲染 {len(rendered)} 张'
          + (f'（{", ".join(rendered)}）' if rendered else ''))
    return paths
