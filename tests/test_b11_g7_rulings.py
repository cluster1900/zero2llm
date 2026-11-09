"""B11 G7 rulings (08m–08o, bot-hub#123): exact pins for every edit."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAPTERS = ROOT / "chapters"

# Edited lines: each new line appears exactly once; each old line zero times.
PINS = [('chapters/08m_reasoning_mtp.md', '设有 $n_{\\mathrm{mtp}}$ 个额外 head，总共 $n_{\\mathrm{mtp}}+1$ 个预测距离，记为 horizon $j$：'), ('chapters/08m_reasoning_mtp.md', 'z_t^{(j)}=h_tW_j^\\top\\quad(j=2,\\ldots,n_{\\mathrm{mtp}}+1).'), ('chapters/08m_reasoning_mtp.md', '| 主 head：j=1 | 2 | 3 | 4 | 5 | 忽略 |'), ('chapters/08m_reasoning_mtp.md', '| 旁路下标 0 的 head：j=2 | 3 | 4 | 5 | 忽略 | 忽略 |'), ('chapters/08m_reasoning_mtp.md', '| 旁路下标 1 的 head：j=3 | 4 | 5 | 忽略 | 忽略 | 忽略 |'), ('chapters/08m_reasoning_mtp.md', '对 horizon $j$ 的有效位置集合 $\\Omega_j$，本章定义：'), ('chapters/08m_reasoning_mtp.md', '\\mathcal L_j=-\\frac{1}{|\\Omega_j|}\\sum_{(b,t)\\in \\Omega_j}'), ('chapters/08m_reasoning_mtp.md', '\\log\\operatorname{softmax}(z_{b,t}^{(j)})[x_{b,t+j}].'), ('chapters/08m_reasoning_mtp.md', '若 $\\Omega_j$ 为空，令 $\\mathcal L_j=0$，并保留可反向传播的零值。否则“对空集合取均值”可能得到 NaN。因为不同 horizon 的有效标签数不同，分母不能固定写成 B×T。'), ('chapters/08m_reasoning_mtp.md', '\\mathcal L=\\mathcal L_1+\\frac{w_{\\mathrm{mtp}}}{n_{\\mathrm{mtp}}}\\sum_{j=2}^{n_{\\mathrm{mtp}}+1}\\mathcal L_j+\\mathcal L_{\\mathrm{MoE}}\\quad(n_{\\mathrm{mtp}}>0).'), ('chapters/08m_reasoning_mtp.md', '- `valid`：这个 token 是否真实存在，而不是 padding。跨度 t 到 t+j 中的**每个位置**都必须有效。'), ('chapters/08m_reasoning_mtp.md', '- `supervise`：哪些目标位置应计入损失，例如只监督 assistant token。检查的是 t+j，**不是 t**。最后一个 prompt token 预测第一个 assistant token 正是合法监督。'), ('chapters/08o_qwen3_case.md', '另一个容易混淆的字段是 BOS：模型配置含 `bos_token_id=151643`（固定 tokenizer 配置中这个 ID 就是 `<|endoftext|>`），但 tokenizer 配置的 `bos_token` 为 null、`add_bos_token=false`。不能看见前者就手工往每段对话前重复加 BOS。具体输入应遵循所选 tokenizer 与模板的处理规则。[固定 tokenizer 配置](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer_config.json)'), ('chapters/08o_qwen3_case.md', '最后，**配置长度不等于能力保证**。两个 config 的 `max_position_embeddings` 都是 40960；对应模型卡描述的原生上下文是 32768，两者相差 40960 − 32768 = 8192。30B-A3B 的卡另外说明借助 YaRN 达到 131072，并解释 40960 默认值的预算安排。两个快照的 tokenizer_config.json 里 `model_max_length` 是 131072，这是 tokenizer 截断和长度警告用的上限，不是模型配置，也不证明模型能处理这么长。本地快照的 `rope_scaling=null`，本章没有启用或验证 YaRN，更不能从 40960 的字段宣布 131072 已经可用。[30B-A3B 长文本说明](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/README.md#processing-long-texts)')]
GONE = [('chapters/08m_reasoning_mtp.md', '设有 D 个额外 head，总共 D+1 个预测距离，记为 horizon h：'), ('chapters/08m_reasoning_mtp.md', 'z_t^{(h)}=W_hh_t\\quad(h=2,\\ldots,D+1).'), ('chapters/08m_reasoning_mtp.md', '| 主 head：h=1 | 2 | 3 | 4 | 5 | 忽略 |'), ('chapters/08m_reasoning_mtp.md', '| 旁路下标 0 的 head：h=2 | 3 | 4 | 5 | 忽略 | 忽略 |'), ('chapters/08m_reasoning_mtp.md', '| 旁路下标 1 的 head：h=3 | 4 | 5 | 忽略 | 忽略 | 忽略 |'), ('chapters/08m_reasoning_mtp.md', '对 horizon h 的有效位置集合 M_h，本章定义：'), ('chapters/08m_reasoning_mtp.md', 'L_h=-\\frac{1}{|M_h|}\\sum_{(b,t)\\in M_h}'), ('chapters/08m_reasoning_mtp.md', '\\log\\operatorname{softmax}(z_{b,t}^{(h)})[x_{b,t+h}].'), ('chapters/08m_reasoning_mtp.md', '若 M_h 为空，令 L_h=0，并保留可反向传播的零值。否则“对空集合取均值”可能得到 NaN。因为不同 horizon 的有效标签数不同，分母不能固定写成 B×T。'), ('chapters/08m_reasoning_mtp.md', 'L=L_1+\\frac{\\lambda}{D}\\sum_{h=2}^{D+1}L_h+L_{MoE}\\quad(D>0).'), ('chapters/08m_reasoning_mtp.md', '- `valid`：这个 token 是否真实存在，而不是 padding。跨度 t 到 t+h 中的**每个位置**都必须有效。'), ('chapters/08m_reasoning_mtp.md', '- `supervise`：哪些目标位置应计入损失，例如只监督 assistant token。检查的是 t+h，**不是 t**。最后一个 prompt token 预测第一个 assistant token 正是合法监督。'), ('chapters/08o_qwen3_case.md', '另一个容易混淆的字段是 BOS：模型配置含 `bos_token_id=151643`，但 tokenizer 配置的 `bos_token` 为 null、`add_bos_token=false`。不能看见前者就手工往每段对话前重复加 BOS。具体输入应遵循所选 tokenizer 与模板的处理规则。[固定 tokenizer 配置](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer_config.json)'), ('chapters/08o_qwen3_case.md', '最后，**配置长度不等于能力保证**。两个 config 的 `max_position_embeddings` 都是 40960；对应模型卡描述的原生上下文是 32768。30B-A3B 的卡另外说明借助 YaRN 达到 131072，并解释 40960 默认值的预算安排。两个快照的 tokenizer_config.json 里 `model_max_length` 是 131072，这是 tokenizer 截断和长度警告用的上限，不是模型配置，也不证明模型能处理这么长。本地快照的 `rope_scaling=null`，本章没有启用或验证 YaRN，更不能从 40960 的字段宣布 131072 已经可用。[30B-A3B 长文本说明](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/README.md#processing-long-texts)')]

# Ruling 1: the closing sentence is the sole body of a final 「## 本章小结」.
CLOSING = {'08m_reasoning_mtp.md': '读完本章，应能清楚说出：**推理能力要靠任务评估，MTP 是训练监督，投机解码是保留目标生成规则的执行方法，测试时计算是需要付账的预算。** 把它们分别定义、分别验证，比把所有改进都叫“多想一步”更重要。', '08n_quantized_serving.md': '你现在得到的不是一个“装上就万能”的工具名单，而是一条可检查的链：**数值规则 → 格式与架构 → 设备内核 → 服务边界 → 同条件评估**。后面读真实模型配置时，继续沿这条链检查，不要让模型名替代证据。', '08o_qwen3_case.md': '读完本章，应能把“Qwen3 用了 GQA 和 MoE”改写为更有信息量的句子：**这份完整 revision 的 dense／MoE checkpoint，按这些配置和实现证据使用了这些组件；我们只在这个小型、可复算的范围内验证过它们。** 这也是阅读下一个模型案例时应保留的习惯。'}


def read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


# bot-hub#126 F8: the B11 j-lines were re-lettered to the A.5 convention
# (MTP depth D -> n_mtp, W_jh_t -> h_tW_j^T, M_j -> Omega_j, lambda -> w_mtp,
# L -> \mathcal L). The B11 j-wording must not come back either.
GONE += [
    ('chapters/08m_reasoning_mtp.md', '设有 D 个额外 head，总共 D+1 个预测距离，记为 horizon j：'),
    ('chapters/08m_reasoning_mtp.md', 'z_t^{(j)}=W_jh_t\\quad(j=2,\\ldots,D+1).'),
    ('chapters/08m_reasoning_mtp.md', '对 horizon j 的有效位置集合 M_j，本章定义：'),
    ('chapters/08m_reasoning_mtp.md', 'L_j=-\\frac{1}{|M_j|}\\sum_{(b,t)\\in M_j}'),
    ('chapters/08m_reasoning_mtp.md', '若 M_j 为空，令 L_j=0，并保留可反向传播的零值。否则“对空集合取均值”可能得到 NaN。因为不同 horizon 的有效标签数不同，分母不能固定写成 B×T。'),
    ('chapters/08m_reasoning_mtp.md', 'L=L_1+\\frac{\\lambda}{D}\\sum_{j=2}^{D+1}L_j+L_{MoE}\\quad(D>0).'),
    ('chapters/08m_reasoning_mtp.md', 'z_t^{(j)}=x_tW_j\\quad(j=2,\\ldots,n_{\\mathrm{mtp}}+1).'),
]

class B11G7Rulings(unittest.TestCase):
    def test_new_lines_once(self):
        for rel, new in PINS:
            with self.subTest(file=rel, new=new[:30]):
                self.assertEqual(read(rel).count(new), 1)

    def test_old_lines_gone(self):
        for rel, old in GONE:
            with self.subTest(file=rel, old=old[:30]):
                self.assertEqual(read(rel).count(old), 0)

    def test_closing_is_last_section(self):
        for name, sentence in CLOSING.items():
            body = (CHAPTERS / name).read_text(encoding="utf-8")
            with self.subTest(chapter=name):
                self.assertEqual(body.count(sentence), 1)
                self.assertEqual(body.count("## 本章小结"), 1)
                headings = re.findall(r"^## .*$", body, flags=re.M)
                self.assertEqual(headings[-1], "## 本章小结")
                self.assertTrue(body.endswith("\n\n## 本章小结\n\n" + sentence + "\n"))

    def test_08m_last_two_sections_order(self):
        headings = re.findall(r"^## .*$", read("chapters/08m_reasoning_mtp.md"), flags=re.M)
        self.assertEqual(headings[-2:], ["## 资料版本与复现边界", "## 本章小结"])

    def test_08m_horizon_letter_j(self):
        body = read("chapters/08m_reasoning_mtp.md")
        for old in ("horizon h", "M_h", "L_h", "^{(h)}", "t+h", "h=1 |", "h=2 |", "h=3 |", "\\sum_{h=2}"):
            with self.subTest(old=old):
                self.assertEqual(body.count(old), 0)
        # bot-hub#126 F8 (F0 r3/r4 final A.5): P23 keeps the chapter-local hidden
        # vector h_t (r2's x_t is reverted) and multiplies the [|V|,D]-stored head
        # as h_tW_j^T; the r2 x_t spelling and the interim X^{(L)}_t must not return.
        self.assertEqual(body.count("h_t"), 9)
        self.assertEqual(body.count("$h_t$"), 6)
        self.assertEqual(body.count("h_tW_"), 3)
        self.assertEqual(body.count("X^{(L)}_t"), 0)
        self.assertEqual(body.count("$x_t$"), 0)
        self.assertEqual(body.count("x_tW_"), 0)

    def test_8192_arithmetic(self):
        body = read("chapters/08o_qwen3_case.md")
        needle = "两者相差 40960 − 32768 = 8192"
        self.assertEqual(body.count(needle), 1)
        self.assertEqual(40960 - 32768, 8192)
        # Dist must carry the same hand check (caught by the dist-md gate too).
        dist = read("dist/transformer_from_scratch.md")
        self.assertEqual(dist.count(needle), 1)


if __name__ == "__main__":
    unittest.main()
