#!/usr/bin/env python3
"""
build.py - Transformer 初学者启蒙书自动化编译流水线
任务：
1. 合并 chapters/ 下所有章节 Markdown 为 dist/transformer_from_scratch.md
2. 调用 Pandoc 编译生成标准 EPUB 电子书：dist/transformer_from_scratch.epub
3. 编译生成现代化响应式单页 Web 读物：dist/transformer_from_scratch.html
4. 验证与生成质量审计报告
"""

import os
import re
import sys
import shutil
import subprocess
import html
import tempfile
from scripts.book_diagrams import prepare_documents
from scripts.check_book import check_book
from scripts.epub_metadata import declare_mathml
from datetime import date

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAPTERS_DIR = os.path.join(BASE_DIR, "chapters")
DIST_DIR = os.path.join(BASE_DIR, "dist")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
# Pandoc 默认要求列表和引用前面必须空一行。书稿里大量列表紧挨着上一段，
# 不打开这两个扩展时，网页和电子书会把清单挤成一行。
# +alerts 把 > [!NOTE] 收成提示框，而不是把 [!NOTE] 这几个字印出来。
MD_FORMAT = (
    "markdown-blank_before_blockquote"
    "+lists_without_preceding_blankline"
    "+alerts"
)
ALERTS_LUA = os.path.join(ASSETS_DIR, "callout_titles.lua")

def ensure_dirs():
    os.makedirs(DIST_DIR, exist_ok=True)


def require_command(command, purpose):
    """在开始编译前给出可操作的依赖提示，而不是抛出长 traceback。"""
    if shutil.which(command) is None:
        print(f"❌ 找不到 {command}（{purpose}）。")
        print(f"   macOS 可用 Homebrew 安装：brew install {command}")
        print("   安装后重新运行：python3 build.py")
        return False
    return True

def get_sorted_chapters():
    files = sorted([f for f in os.listdir(CHAPTERS_DIR) if f.endswith(".md")])
    return [os.path.join(CHAPTERS_DIR, f) for f in files]


def slugify(text):
    """生成目录和正文共用的中文锚点，避免依赖 Pandoc 的方言差异。"""
    clean = re.sub(r"<[^>]+>", "", html.unescape(text))
    clean = re.sub(r"[*_`]", "", clean).strip()
    return re.sub(r"[^\w\u4e00-\u9fa5]+", "-", clean).strip("-")

def merge_markdown():
    print("▶ 正在合并 Markdown 章节文件...")
    chapter_files = get_sorted_chapters()
    output_path = os.path.join(DIST_DIR, "transformer_from_scratch.md")
    
    with open(output_path, "w", encoding="utf-8") as outfile:
        # 写入元数据前言
        outfile.write("% 写给初学者的 Transformer：从零开始的大模型极简数学与代码之旅\n")
        outfile.write("% Zero2Transformer 教程\n")
        # 不把日期写死，避免第二天重新编译仍显示旧日期。
        outfile.write(f"% {date.today().isoformat()}\n\n")

        for idx, cpath in enumerate(chapter_files):
            with open(cpath, "r", encoding="utf-8") as infile:
                content = infile.read()
                # 确保两章之间有分页符或换行
                outfile.write(content)
                outfile.write("\n\n---\n\n")
            print(f"  + 已合并: {os.path.basename(cpath)}")

    print(f"✓ 完整 Markdown 已输出至: {output_path}")
    return output_path

def build_epub(document_path):
    print("\n▶ 正在使用 Pandoc 编译 EPUB 电子书...")
    epub_out = os.path.join(DIST_DIR, "transformer_from_scratch.epub")
    cover_image = os.path.join(ASSETS_DIR, "cover.jpg")
    epub_css = os.path.join(ASSETS_DIR, "epub.css")
    
    cmd = [
        "pandoc",
        document_path,
        "-f", "json",
        "--lua-filter", ALERTS_LUA,
        "--resource-path", os.pathsep.join([DIST_DIR, BASE_DIR]),
        "-o", epub_out,
        "--toc",
        "--toc-depth=3",
        "--mathml",
        f"--epub-cover-image={cover_image}",
        f"--css={epub_css}",
        "--metadata", "title=写给初学者的 Transformer：从零开始的大模型极简数学与代码之旅",
        "--metadata", "author=Zero2Transformer 教程",
        "--metadata", "language=zh-CN",
        "--metadata", "rights=Creative Commons CC-BY-NC 4.0"
    ]
    
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE_DIR)
    except OSError as exc:
        print(f"❌ EPUB 编译失败：无法启动 pandoc（{exc}）")
        return False
    if res.returncode != 0:
        print(f"❌ EPUB 编译失败:\n{res.stderr}")
        return False
    if res.stderr:
        print(res.stderr.strip())
    
    declare_mathml(epub_out)
    epub_size_mb = os.path.getsize(epub_out) / (1024 * 1024)
    print(f"✓ EPUB 电子书生成成功: {epub_out} ({epub_size_mb:.2f} MB)")
    return True

def build_html(document_path):
    print("\n▶ 正在编译现代化 Web HTML 读物...")
    # 1. 先用 pandoc 将 markdown 编译为 html 片段
    temp_fragment_path = os.path.join(DIST_DIR, "_fragment.html")
    cmd = [
        "pandoc",
        document_path,
        "-f", "json",
        "--lua-filter", ALERTS_LUA,
        "-t", "html5",
        "--mathjax",
        "-o", temp_fragment_path
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE_DIR)
    except OSError as exc:
        print(f"❌ Pandoc 转换 HTML 片段失败：无法启动 pandoc（{exc}）")
        return False
    if res.returncode != 0:
        print(f"❌ Pandoc 转换 HTML 片段失败:\n{res.stderr}")
        return False

    with open(temp_fragment_path, "r", encoding="utf-8") as f:
        html_body = f.read()

    # dist/ 可以被单独部署为静态站点，因此源码链接改为指向同目录下的 code/。
    # Markdown 原文仍保留 ../code/，从项目根目录阅读时路径更自然。
    html_body = html_body.replace('href="../code/', 'href="code/')

    # Pandoc 对中文标题的默认 id 会去掉部分标点，而目录需要保留可读的短横线。
    # 在服务端直接补 id，这样即使 CDN 脚本加载失败，目录仍然能跳转。
    def add_heading_id(match):
        attrs, inner = match.group(1), match.group(2)
        attrs = re.sub(r'\s+id="[^"]*"', "", attrs)
        return f'<h1 id="{slugify(inner)}"{attrs}>{inner}</h1>'

    html_body = re.sub(
        r"<h1\b([^>]*)>(.*?)</h1>", add_heading_id, html_body, flags=re.DOTALL
    )

    # 宽表在手机上横向滚动。Pandoc 有的表是 <table>，有的带 style，两种都要包住。
    # 只替换 </table>、不替换带属性的 <table> 时，多出来的 </div> 会把后面的章节拆出正文栏。
    html_body = re.sub(
        r"<table\b([^>]*)>",
        r'<div class="table-wrapper"><table\1>',
        html_body,
    )
    html_body = html_body.replace("</table>", "</table></div>")
    # 读取 html_style.css
    css_path = os.path.join(ASSETS_DIR, "html_style.css")
    with open(css_path, "r", encoding="utf-8") as f:
        custom_css = f.read()

    # 从章节文件中提取目录结构
    chapter_files = get_sorted_chapters()
    toc_items = []
    for cpath in chapter_files:
        with open(cpath, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("# "):
                    title = line.replace("# ", "").strip()
                    # 提取纯文字
                    clean_title = re.sub(r'[*_`]', '', title)
                    slug = slugify(clean_title)
                    toc_items.append((clean_title, slug))
                    break

    toc_html = "<ul>\n"
    for title, slug in toc_items:
        toc_html += f'  <li><a href="#{slug}">{title}</a></li>\n'
    toc_html += "</ul>"

    # 包装完整的单页现代化 HTML5 模板
    html_template = f"""<!DOCTYPE html>
<html lang="zh-CN" data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>写给初学者的 Transformer：从零开始的大模型极简数学与代码之旅</title>
  <meta name="description" content="面向数学与编程初学者的从零到一 Transformer 入门精美读物。深入浅出图解词向量、QKV 注意力、多头机制、位置编码与 GPT 搭建。">
  
  <!-- MathJax for rendering math formulas -->
  <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
        processEscapes: true
      }},
      options: {{
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
      }}
    }};
  </script>
  <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

  <!-- Highlight.js for code syntax highlighting -->
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark-dimmed.min.css">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/python.min.js"></script>
  <script>document.addEventListener('DOMContentLoaded', () => {{ if (window.hljs) hljs.highlightAll(); }});</script>

  <style>
{custom_css}
  </style>
</head>
<body>
  <div id="progress-bar"></div>

  <header>
    <a href="#" class="brand">
      <span>🚀 Transformer 从零到一</span>
      <span class="brand-badge">初学者极简版</span>
    </a>
    <div class="header-controls">
      <button class="btn-icon" id="theme-toggle" title="切换深色/浅色主题">
        🌓 <span>主题</span>
      </button>
      <a class="btn-icon" href="transformer_from_scratch.epub" download title="下载 EPUB 电子书">
        📖 <span>下载 EPUB</span>
      </a>
      <a class="btn-icon" href="transformer_from_scratch.md" download title="下载 Markdown 原文">
        📝 <span>下载 Markdown</span>
      </a>
    </div>
  </header>

  <div class="container">
    <aside class="sidebar">
      <h3>📚 章节快速导航</h3>
      {toc_html}
    </aside>

    <main class="content">
      {html_body}
    </main>
  </div>

  <script>
    // 图表已在构建时渲染，切换主题无需重新计算布局。
    const toggleBtn = document.getElementById('theme-toggle');
    const htmlElem = document.documentElement;
    toggleBtn.addEventListener('click', () => {{
      const current = htmlElem.getAttribute('data-theme');
      const next = current === 'dark' ? 'light' : 'dark';
      htmlElem.setAttribute('data-theme', next);
    }});

    // 顶部阅读进度条
    window.addEventListener('scroll', () => {{
      const winScroll = document.documentElement.scrollTop;
      const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
      const scrolled = (winScroll / height) * 100;
      document.getElementById('progress-bar').style.width = scrolled + '%';
    }});

    // H1 的目录锚点已在构建时写入；这里仅为没有 id 的小标题补上锚点。
    document.querySelectorAll('h1, h2, h3').forEach(heading => {{
      if (heading.id) return;
      const slug = heading.innerText.trim().replace(/[^\\w\\u4e00-\\u9fa5]+/g, '-').replace(/^-|-$/g, '');
      heading.id = slug;
    }});
  </script>
</body>
</html>
"""

    html_out = os.path.join(DIST_DIR, "transformer_from_scratch.html")
    with open(html_out, "w", encoding="utf-8") as f:
        f.write(html_template)

    # 复制一份作为 index.html 方便直接开启本地服务器
    index_out = os.path.join(DIST_DIR, "index.html")
    shutil.copyfile(html_out, index_out)

    # 清理临时文件
    if os.path.exists(temp_fragment_path):
        os.remove(temp_fragment_path)

    html_size_kb = os.path.getsize(html_out) / 1024
    print(f"✓ HTML 交互读物生成成功: {html_out} ({html_size_kb:.1f} KB)")
    print(f"✓ 同步生成入口: {index_out}")
    return True

def main():
    print("=" * 65)
    print("【Transformer 初学者启蒙书：全自动编译交付流水线】")
    print("=" * 65)
    if not require_command("pandoc", "EPUB 和 HTML 都需要它"):
        return 1
    if not require_command("node", "预渲染全书图表需要 Node.js；安装后先运行 npm ci"):
        return 1
    if not os.path.isdir(os.path.join(BASE_DIR, "node_modules", "@mermaid-js", "mermaid-cli")):
        print("❌ 缺少图表构建依赖，请先运行 npm ci")
        return 1
    ensure_dirs()
    # GitHub Pages 以静态文件方式发布 dist/，避免 Jekyll 改写资源路径。
    with open(os.path.join(DIST_DIR, ".nojekyll"), "w", encoding="utf-8") as f:
        f.write("")
    
    # 1. 复制资源文件到 dist/ 以便相对路径正常解析
    dist_assets = os.path.join(DIST_DIR, "assets")
    if os.path.exists(dist_assets):
        shutil.rmtree(dist_assets)
    shutil.copytree(ASSETS_DIR, dist_assets)
    print(f"✓ 静态图片与矢量图资源已同步到 dist/assets/")

    dist_code = os.path.join(DIST_DIR, "code")
    if os.path.exists(dist_code):
        shutil.rmtree(dist_code)
    shutil.copytree(
        os.path.join(BASE_DIR, "code"),
        dist_code,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    print("✓ 示例源码已同步到 dist/code/")

    # 2. 合并 Markdown
    merged_md = merge_markdown()

    # 3. 在临时目录准备两种输出的文档树；任何图表失败都会终止构建。
    try:
        with tempfile.TemporaryDirectory(prefix="zero2llm-") as temp_dir:
            epub_doc, html_doc = prepare_documents(merged_md, temp_dir, BASE_DIR, MD_FORMAT)
            epub_ok = build_epub(epub_doc)
            html_ok = build_html(html_doc)
        if epub_ok and html_ok:
            check_book(DIST_DIR)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"❌ 构建或产物校验失败：{exc}")
        return 1

    print("\n" + "=" * 65)
    if epub_ok and html_ok:
        print("🎉 恭喜！全流程编译已 100% 成功完成！")
        print(f"  - Markdown 原文: dist/transformer_from_scratch.md")
        print(f"  - EPUB 电子书  : dist/transformer_from_scratch.epub")
        print(f"  - 交互 Web HTML: dist/transformer_from_scratch.html")
    else:
        print("⚠️ 编译过程中出现警告，请检查上方日志。")
    print("=" * 65)
    return 0 if epub_ok and html_ok else 1

if __name__ == "__main__":
    sys.exit(main())
