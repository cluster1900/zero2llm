#!/usr/bin/env python3
"""Regenerate chapter code excerpts verbatim from the source files they cite.

Each excerpt is a marker line followed by a fenced python block:

    <!-- code-excerpt: code/28_minigpt_healthcheck.py L121-L133 -->
    ```python
    ...exact source lines...
    ```

Run without arguments to rewrite; run with --check to fail on drift.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = re.compile(
    r'(<!-- code-excerpt: (?P<path>\S+) L(?P<start>\d+)-L(?P<end>\d+) -->\n)```python\n.*?\n```',
    re.S)


def source_lines(path, start, end):
    lines = (ROOT / path).read_text(encoding='utf-8').splitlines()
    if not 1 <= start <= end <= len(lines):
        raise ValueError(f'{path} L{start}-L{end} out of range')
    return '\n'.join(lines[start - 1:end])


def render(text):
    return MARKER.sub(lambda m: m.group(1) + '```python\n' + source_lines(
        m.group('path'), int(m.group('start')), int(m.group('end'))) + '\n```', text)


def main(argv):
    check = '--check' in argv
    drift = []
    for chapter in sorted((ROOT / 'chapters').glob('*.md')):
        text = chapter.read_text(encoding='utf-8')
        new = render(text)
        if new != text:
            drift.append(chapter.name)
            if not check:
                chapter.write_text(new, encoding='utf-8')
    if check and drift:
        print('code excerpts out of date: ' + ', '.join(drift), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
