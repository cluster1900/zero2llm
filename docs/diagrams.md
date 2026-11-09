# 怎样给书加一张图（D0，bot-hub#126）

读者看到的图都是**构建时预渲染**的：网站用 SVG，EPUB 用 PNG，dist md 原样保留 Mermaid 源码。阅读时不需要 JavaScript，也不需要 Mermaid。**CI 不渲染图**：`ci.yml` 跑 `python -m unittest discover -s tests` 和 `scripts/check_book.py`，`pages.yml` 只跑 `scripts/check_book.py`，然后发布已提交的 `dist/`。两者都不启动浏览器，也不重新生成 `dist/`。因此加图或改图的 PR 必须**在本地渲染**，再把 `dist/` 一起提交。

## 1. 图的名字：稳定 id，不随位置变

渲染文件是 `dist/assets/rendered/<id>.svg` 和 `<id>.png`。id 的规则在 `scripts/diagram_ids.py`：

| 图 | id |
|---|---|
| Mermaid 块里有一行 `%% id: <slug>` | `flow-<slug>`，例如 `%% id: p11-prefill-decode` 对应 `flow-p11-prefill-decode` |
| Mermaid 块没有 `%% id:` | `flow-<sha256(源码) 前 12 位十六进制>`，例如 `flow-e59ba8fcee2e` |
| 手绘 SVG `![…](assets/diagrams/x.svg)` | 文件名 `x` |

- slug 只能用小写字母、数字和单个短横线，不能是 12 位十六进制（会和哈希名混淆）。一张图只能有一行 `%% id:`。
- **新图请写 `%% id:`**，按「章节编号-内容」命名，如 `p15-online-softmax`、`ch03-mask-broadcast`。这样以后改图时，文件名不变，diff 一眼能看出是哪张图。
- 在别处插入或删除一张图，其他图的 id 都不变。这一点和 D0 之前不同：以前按出现顺序编号 `flow-01…flow-10`，插一张图就会让后面所有图改名。

- 没有 `%% id:` 时，哈希按源码**原样**计算，不做任何规范化：改一个空格、缩进或换行，id 就会变。想让名字固定，就写 `%% id:`。

`dist/assets/rendered/manifest.json` 记下每张图源码的 sha256 和它是由什么渲染的（`renderer`）。D0 之前提交的 40 张图标为 `legacy: committed before D0 (bot-hub#126)`。

## 2. 只渲染新增或改动的图

`build.py` 开始时，先把上一次的 `dist/assets/rendered/` 备份到 `dist/` 之外的仓库根目录 `.render-backup/`（已 gitignore；先复制到 `.render-backup.tmp/`，复制完整后再改名），然后按 id 和 sha256 判断：

- id 相同、源码 sha256 相同、svg/png 都在：**直接复制旧文件**，字节不变，不启动浏览器；
- 新图或源码改过的图：交给 `scripts/render_diagrams.mjs`（puppeteer + Chromium）渲染。

如果所有图都没变，构建根本不需要 Node.js 和 `npm ci`。

构建**成功**后才删除备份。渲染中途失败时，`build.py` 先把备份复制到 `dist/assets/rendered.restore-tmp/`，再用改名换回 `dist/assets/rendered/`，**换好之后**才删备份——所以失败路径上备份会多留一会儿，并不是“一开始就删”。就算恢复时崩溃，备份也还在：下次 `build.py` 发现 `dist/assets/rendered/manifest.json` 不在、`.render-backup/` 还在，就自动拿备份当复用缓存。要强制全部重渲（例如改了 `render_diagrams.mjs` 的主题或字体），运行 `ZERO2LLM_RERENDER=1 python3 build.py`；这会让 40 多张图全部出现字节差异，请单独开一个 PR。

`scripts/check_book.py`（CI 会跑）会核对以下几点：
- dist md 里每个 Mermaid 块都有同 id、同 sha256 的渲染，否则报「没有渲染」或「渲染已过期（源码改了但没重新构建）」；
- 每张手绘 SVG 的 sha256 都与清单一致。哈希的是**仓库里的源文件** `assets/diagrams/x.svg`（也就是你要改的那个文件，和 `build.py` 一致），不是 `dist/assets/diagrams/` 里的副本；副本和源文件不一致也会报错；
- `dist/assets/rendered/` 里除了 `manifest.json`、`dimensions.json` 和清单中每个 id 的 `.svg/.png` 之外，没有任何别的文件（孤儿文件，例如旧的 `flow-01.png` 或 `orphan.svg`，会报错）；
- 清单里没有书稿中已不存在的 Mermaid 图；
- 清单与 `dimensions.json` 一致。

所以「改了 chapters 或 assets/diagrams 却忘了重新构建」会在 CI 里失败。

### EPUB 源码泄漏检查覆盖的图型

`check_book.py` 会检查 EPUB 里每个 `<pre>`/`<code>`：去掉开头的 front matter（`---…---`）和 `%%` 注释行以后，如果第一行是 Mermaid 图型关键字，就说明有一张图没被预渲染、源码漏给了读者。关键字列表取自锁定版本 mermaid 11.17.2 的 `dist/mermaid.js` 中注册的全部 detector，再加上外部插件 @mermaid-js/mermaid-zenuml 的 `zenuml`：

`flowchart` `flowchart-elk` `graph` `sequenceDiagram` `classDiagram` `classDiagram-v2` `stateDiagram` `stateDiagram-v2` `erDiagram` `journey` `gantt` `pie` `quadrantChart` `requirement` `requirementDiagram` `gitGraph` `C4Context` `C4Container` `C4Component` `C4Dynamic` `C4Deployment` `mindmap` `timeline` `zenuml` `sankey` `sankey-beta` `xychart` `xychart-beta` `block` `block-beta` `packet` `packet-beta` `kanban` `architecture` `architecture-beta` `radar-beta` `treemap` `treemap-beta` `treeView-beta` `swimlane-beta` `eventmodeling` `venn-beta` `cynefin-beta` `info`，以及不区分大小写的 `ishikawa` `ishikawa-beta` `wardley-beta` `railroad-beta` `railroad-ebnf-beta` `railroad-abnf-beta` `railroad-peg-beta`。

其中很多关键字也是普通单词（`graph`、`pie`、`info`、`block`、`timeline`、`journey`）。判定时关键字后面要满足 mermaid 常见的边界 `(?:[\s:]|$)`（所以 `cynefin-beta:` 算图头，`graphs` / `flowchart-v3` 不算）；整行要么只有关键字（可带结尾 `:`），要么后面跟参数。对容易误伤的 `graph` / `journey` / `pie` / `info` / `timeline`，参数只放行方向词（`LR`/`TD`/…）和 `title` / `showData` / `showInfo`，避免 `graph of the function` 这类普通句子被当成图；其他关键字仍拒绝以 `=`、`(`、`.`、`,`、`[`、`{` 开头的代码尾巴。正则里的换行只结束图头这一行，**不要求**后面再跟正文行（`info\n` 单独一行就算）。`tests/test_diagram_ids.py` 逐个测试这些关键字。装了 node_modules 时，它还会把这个列表和 `node_modules/mermaid` 实际注册的 detector 对一遍（同时解析 `return /…/` 与 `if (/…/)` 两种写法）。升级 mermaid 时，同步更新 `scripts/check_book.py` 的 `MERMAID_KEYWORDS`、这份列表和测试。

## 3. 渲染工具链（实际版本）

以下版本取自 `package-lock.json`，以及 2026-11-10 在 box 上实际运行的结果：

| 组件 | 版本 | 来源 |
|---|---|---|
| mermaid | 11.17.2 | package-lock.json |
| @mermaid-js/mermaid-cli | 11.12.0 | package.json / package-lock.json |
| puppeteer（及 puppeteer-core） | 23.11.1 | package.json / package-lock.json |
| Chrome（puppeteer 自动下载并启动） | 131.0.6778.204（`chrome/linux-131.0.6778.204/chrome-linux64/chrome`；同版本 chrome-headless-shell 也会下载，但不使用） | puppeteer-core `revisions.js`；`browser.version()` 实测为 `Chrome/131.0.6778.204` |
| Node.js / npm | v24.22.1 / 11.21.1（package.json 要求 Node ≥ 22） | `node -v`、`npm -v` |
| Pandoc | 3.9 | `pandoc --version` |
| Python | 3.13.5 | `python3 --version` |
| 中文字体 | Noto Sans CJK SC（`render_diagrams.mjs` 依次尝试 PingFang SC → Noto Sans CJK SC → **Noto Sans Math** → Microsoft YaHei） | `fc-match "Noto Sans CJK SC"` |
| 数学字母字体 | **Noto Sans Math Regular v3.000**（覆盖 U+1D4B1 `𝒱` 等 Mathematical Alphanumeric Symbols；CJK 缺字形时回退到它） | 见下文安装 |

字体或 Chrome 不同，文字宽度会差几个像素。实测重渲 D0 之前的 10 张图，2 张字节相同，其余宽度相差 < 2.5%，内容一致。复用机制保证这种漂移只出现在新增或改动的图上。

### 数学字体：Noto Sans Math（D7，bot-hub#126）

附录 A.5 的词表大小写作 $\lvert\mathcal V\rvert$，图里对应明文 `|𝒱|`（U+1D4B1 MATHEMATICAL SCRIPT CAPITAL V）。Noto Sans CJK SC **没有**这个码位，缺字体时会画成空白方框。因此 `scripts/render_diagrams.mjs` 的 `fontFamily` 在 CJK 之后加入 **Noto Sans Math**：中文仍走 CJK，script V 回退到数学字体。

| 项 | 值 |
|---|---|
| 字体名 | Noto Sans Math |
| 样式 / 版本 | Regular / **3.000**（`fc-query` 的 `fontversion` 为 196608 = 3.0.0） |
| 发布包 | [NotoSansMath-v3.000.zip](https://github.com/notofonts/math/releases/download/NotoSansMath-v3.000/NotoSansMath-v3.000.zip)（notofonts/math） |
| 包 sha256 | `ac351837b41f8a897f020b97fb0f075ad574c1e9669fb5839ada1f92fd748356` |
| 选用文件 | `NotoSansMath/full/ttf/NotoSansMath-Regular.ttf` |
| 文件 sha256 | `7283c396e9b22699bb542d9631030dc804a7e5b954f193d8f8f5b5f1162fbc61` |
| 已核验 | `full/ttf` 的 cmap **含 U+1D4B1**（本机用 fontTools 查过） |

安装到本机（构建机 / box；CI 不渲染图，不必装）：

```bash
curl -fsSL -o /tmp/NotoSansMath-v3.000.zip \
  https://github.com/notofonts/math/releases/download/NotoSansMath-v3.000/NotoSansMath-v3.000.zip
echo ac351837b41f8a897f020b97fb0f075ad574c1e9669fb5839ada1f92fd748356  /tmp/NotoSansMath-v3.000.zip | sha256sum -c
unzip -o /tmp/NotoSansMath-v3.000.zip -d /tmp/noto-math
mkdir -p /workspace/tools/fonts "$HOME/.local/share/fonts"
cp /tmp/noto-math/NotoSansMath/full/ttf/NotoSansMath-Regular.ttf /workspace/tools/fonts/
cp /tmp/noto-math/NotoSansMath/full/ttf/NotoSansMath-Regular.ttf "$HOME/.local/share/fonts/"
fc-cache -f "$HOME/.local/share/fonts"
fc-list "Noto Sans Math"
fc-list :charset=1D4B1 family   # 应列出 Noto Sans Math
```

`tests/test_diagram_ids.py` 会检查：`render_diagrams.mjs` 的 `fontFamily` 含 `Noto Sans Math`；若某张已提交的 Mermaid SVG 源码含 `𝒱`，则 SVG 正文也必须含 `𝒱`（不是 `U+FFFD`）；本机装了该 ttf 时再断言 cmap 覆盖 U+1D4B1，CI 未装字体则跳过字形检查。

## 4. 复现命令

```bash
# 一次性：在仓库外的草稿目录按 lockfile 精确安装，把 Chrome 下载到指定缓存
mkdir -p /workspace/tools/z2l-mermaid
cp package.json package-lock.json /workspace/tools/z2l-mermaid/
(cd /workspace/tools/z2l-mermaid && \
  PUPPETEER_CACHE_DIR=/workspace/tools/puppeteer-cache npm ci --no-audit --no-fund)

# 每个工作区：链接 node_modules（链接文件不会被 .gitignore 的 node_modules/ 规则忽略，
# 所以加进本地 exclude，绝不提交）
ln -sfn /workspace/tools/z2l-mermaid/node_modules node_modules
echo node_modules >> .git/info/exclude
export PATH=/path/to/pandoc-3.9/bin:$PATH          # 让 pandoc 3.9 排在 PATH 最前
export PUPPETEER_CACHE_DIR=/workspace/tools/puppeteer-cache

python3 build.py                 # 输出里会列出「新渲染 N 张（id…）」
python3 scripts/check_book.py
python3 -m unittest discover -s tests
```

只想单独预览一张图、不跑整本书：

```bash
cat > /tmp/jobs.json <<'JSON'
[{"id": "preview", "definition": "flowchart TD\n    accTitle: 标题\n    accDescr: 说明\n    A-->B"}]
JSON
node scripts/render_diagrams.mjs /tmp/jobs.json /tmp/preview   # 得到 preview.svg / preview.png / dimensions.json
```

不能启动浏览器的机器可以用 `python3 scripts/build_without_browser.py`。它会复用工作区里已提交的渲染（`--baseline <commit>` 则复用该 commit 的渲染，要求 D0 之后的 commit）：新增的 Mermaid 图只要已有同 id、同源码的渲染就能复用，否则报错；改过的手绘 SVG 用 CairoSVG 2.8.2 栅格化。

## 5. 加一张 Mermaid 图的步骤

1. 在章节里相应小节（正文引出这张图的句子之后）写：

   ````markdown
   ```mermaid
   flowchart TD
       %% id: p15-online-softmax
       accTitle: 在线 Softmax 每读一块都把旧账换到新尺度
       accDescr: 初始 m=−∞、ℓ=0、acc=0；读一块后更新最大值，ℓ 和 acc 先乘 α 再加新块，读完输出 acc/ℓ。
       A["读下一块分数"] --> B["next_m = max(m, 块内最大)"]
   ```
   ````

   - `accTitle:` 写**一句完整的中文说明**，显示为图下的图注；`accDescr:` 写一两句替代文字，给 EPUB 和读屏使用。两行都必须有，否则构建失败。
   - **mindmap、block-beta** 的语法里没有这两行，要写成注释：`%% accTitle: …`、`%% accDescr: …`（`%% id:` 同理）。例：

     ```
     mindmap
       %% id: p17-codec-names
       %% accTitle: 四种“编码—解码”同名不同义
       %% accDescr: 中心是编码—解码，分出 seq2seq、CED、推理阶段和 tokenizer 四支。
       root((编码—解码))
         经典 seq2seq
     ```
   - 图中符号遵守附录 A.5（F0）：`D`、`H`、`H_kv`、`d_h`、`T`（`T_q`/`T_kv`）、`B`、`|𝒱|`（词表大小；需 Noto Sans Math）、`N_E`、`τ`、`κ`、`T_win`，形状写 `[B,T,D]`。标签不支持 LaTeX（`htmlLabels:false`），用纯文本或 Unicode 下标。
   - 图型：流程、形状、判断用 `flowchart`；多角色随时间用 `sequenceDiagram`；生命周期、停止条件用 `stateDiagram-v2`；`mindmap`、`block-beta` 也可以用。**不要用 `gantt`**：默认宽高比约 4.85。
2. 运行 `python3 build.py`，确认输出里「新渲染」恰好是你加的那几张。
3. 打开 `dist/assets/rendered/<id>.png` 看一眼。`check_book.py` 要求宽高比在 **0.45–2.4** 之间（例如一行并排三块的 block-beta 是 566×53，会失败，要改成 `columns 1`）。过长的链拆成两张，或改用 `LR`。
4. 运行 `python3 scripts/check_book.py` 和 `python3 -m unittest discover -s tests`。
5. 提交 `chapters/…`，以及 `dist/` 下全部变化：新的 `<id>.svg/.png`、`dimensions.json`、`manifest.json`、`transformer_from_scratch.{md,html,epub}`、`index.html`。dist md 第 3 行的日期每次构建都会变，这是正常的。
6. 与其他 PR 冲突时：`git merge main`，然后重新运行 `python3 build.py`。已渲染的图会被复用，不会重渲。不要手工合并 `dimensions.json`、`manifest.json`。

## 6. 手绘 SVG

放进 `assets/diagrams/`，正文用 `![说明](assets/diagrams/x.svg)` 引用；文件名不能以 `flow-` 开头。源文件改了以后，`build.py` 只重新栅格化这一张。
