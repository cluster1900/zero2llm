#!/usr/bin/env python3
"""Package the committed, validated book; do not rebuild differently in CI."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import zipfile
from check_book import check_book

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--tag', default='v1.0.0')
args = parser.parse_args()
dist = root / 'dist'
check_book(dist)
out = root / '.release'
out.mkdir(exist_ok=True)
shutil.copyfile(dist / 'transformer_for_highschool.epub', out / 'transformer_for_highschool.epub')
# Stable file order and timestamps make a downloaded ZIP verifiable across CI/local packaging.
with zipfile.ZipFile(out / 'zero2llm-web.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for file in sorted(dist.rglob('*')):
        if file.is_file() and file.name != '.DS_Store' and '__pycache__' not in file.parts:
            info = zipfile.ZipInfo(file.relative_to(dist).as_posix(), (2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, file.read_bytes())
assets = ['transformer_for_highschool.epub', 'zero2llm-web.zip']
(out / 'SHA256SUMS.txt').write_text(''.join(
    f'{hashlib.sha256((out / name).read_bytes()).hexdigest()}  {name}\n' for name in assets
))
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
base = f'https://github.com/cluster1900/zero2llm/releases/download/{args.tag}'
notes = f'''## Transformer 从零到一学习包

面向高中数学水平、具备少量 Python 阅读经验的学习者：11 章讲解、7 个代码实验与 Baby-GPT 实战。

## 图表与 EPUB 修订

- Mermaid 在构建时预渲染，EPUB 无需 JavaScript 或网络即可显示图表。
- 长流程拆成 16 张紧凑图，加上原有 7 张插图，共 23 张；按比例限制宽高，附图题与替代文字。
- 修复 EPUB 配套代码链接，代码块允许换行。
- 网页图表支持离线显示；网页版公式排版和代码高亮仍需要网络。

## 阅读与下载

- [在线阅读](https://cluster1900.github.io/zero2llm/)
- [下载 EPUB 电子书]({base}/transformer_for_highschool.epub)
- [下载 Web ZIP]({base}/zero2llm-web.zip)：解压后打开 `index.html`。
- [下载 SHA-256 校验文件]({base}/SHA256SUMS.txt)

如果阅读器仍显示旧图，请移除已导入的旧书，再重新导入本次下载的 EPUB。

本次阅读文件来自提交 [{commit[:7]}](https://github.com/cluster1900/zero2llm/commit/{commit})。
按要求替换现有 Release 的阅读文件；`{args.tag}` 标签和 GitHub 自动生成的 Source code 包保留首次发布记录，修订后的源码见上述提交。
'''
(out / 'release-notes.md').write_text(notes, encoding='utf-8')
print(f'✓ 发布文件与 SHA-256 校验值：{out}')
