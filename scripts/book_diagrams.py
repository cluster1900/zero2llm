"""Replace Mermaid code blocks with packaged images in Pandoc's document tree."""
import copy
import json
import re
import subprocess
from pathlib import Path


def prepare_documents(merged_md, temp_dir, base_dir, markdown_format):
    base = Path(base_dir)
    tmp = Path(temp_dir)
    parsed = subprocess.run(
        ['pandoc', str(merged_md), '-f', markdown_format, '-t', 'json'],
        cwd=base, check=True, capture_output=True, text=True,
    )
    document = json.loads(parsed.stdout)
    jobs = []
    definitions = {}
    static_images = {}

    def collect(value):
        if isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            if value.get('t') == 'CodeBlock' and 'mermaid' in value['c'][0][1]:
                definition = value['c'][1]
                title = re.search(r'^\s*accTitle:\s*(.+)$', definition, re.M)
                description = re.search(r'^\s*accDescr:\s*(.+)$', definition, re.M)
                if not title or not description:
                    raise ValueError('每张 Mermaid 图都需要 accTitle 和 accDescr，方便离线阅读和无障碍访问。')
                diagram_id = f'flow-{len(definitions) + 1:02d}'
                definitions[definition] = (diagram_id, title[1], description[1])
                jobs.append({'id': diagram_id, 'definition': definition})
            elif value.get('t') == 'Image':
                source = value['c'][2][0]
                if source.endswith('.svg') and source not in static_images:
                    diagram_id = Path(source).stem
                    static_images[source] = diagram_id
                    jobs.append({'id': diagram_id, 'source': str(base / source)})
            for item in value.values():
                collect(item)

    collect(document)
    jobs_path = tmp / 'diagrams.json'
    jobs_path.write_text(json.dumps(jobs, ensure_ascii=False), encoding='utf-8')
    output = base / 'dist/assets/rendered'
    subprocess.run(['node', str(base / 'scripts/render_diagrams.mjs'), str(jobs_path), str(output)], cwd=base, check=True)
    sizes = json.loads((output / 'dimensions.json').read_text())

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
    print(f'✓ 已预渲染 {len(definitions)} 张流程图与 {len(static_images)} 张原有插图')
    return paths
