"""B5 (bot-hub#123): frontier extension sections compressed to short pointers.

Each pointer is pinned as a whole paragraph under its heading, so rewording or
re-expanding it fails; removed scripts, tests and appendix records must stay gone.
"""
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CH = ROOT / "chapters"


def read(name):
    return (CH / name).read_text(encoding="utf-8")


def section(text, heading):
    """Body between `heading` and the next heading of the same or higher level."""
    level = len(heading.split()[0])
    start = text.index(heading + "\n")
    rest = text[start + len(heading) + 1:]
    m = re.search(r"^#{1,%d} " % level, rest, re.M)
    return (rest[:m.start()] if m else rest).strip()


# bot-hub#126 F4: the 06 §6.9 pointer now carries an inline 「其中：」/白话 note for its formula symbol
# (Wu: every formula explained); F_i -> F_l because A.5 reserves i for query positions. Kept on one
# line and within 3 sentences so the pointer-shape assertions stay unchanged.
POINTERS = [('03_qkv_self_attention.md', '### 3.3a 延伸阅读：Softmax Collapse {#p04-softmax-collapse}', '减最大值解决了上溢，但没有解决另一种精度问题：正确类的 logit 间隔很大时，交叉熵对目标 logit 的梯度会在浮点里舍入成 0（论文给出的 fp32 阈值约为 16.6）。现象和修法见 [A Fix for Softmax and Cross-Entropy，arXiv:2610.31642](https://arxiv.org/abs/2610.31642) §1–§2，本书不展开。'), ('06_residual_layernorm_ffn.md', '## 6.9 延伸阅读：AttnRes {#p07-attnres}', '6.1 的残差把每个子层的增量 $F_l$ 都以权重 1 相加。**其中：** $F_l$ 是第 $l$ 个子层（注意力或 FFN）算出的增量，$l$ 只是子层编号；**白话：** 每一层的“批注”都原样记到同一条主干上，谁也不比谁分量重。[Attention Residuals，arXiv:2603.15031](https://arxiv.org/abs/2603.15031) 把它换成沿深度方向的注意力：每层用一个可学习的 pseudo-query，对 embedding 和此前各层的输出做加权平均；[Kimi K3 技术报告，arXiv:2607.24653](https://arxiv.org/abs/2607.24653)采用了这个做法。'), ('08a_kv_cache.md', '## P11.10 延伸阅读：直接改 KV（cache engineering） {#p11.10-cache-engineering}', '[Cache Engineering for Agents，arXiv:2610.28577](https://arxiv.org/abs/2610.28577)（代码 [thu-nics/AKV](https://github.com/thu-nics/AKV)）不改文本，直接在 KV 缓存里丢掉旧的工具结果，再用 RoPE 旋转把保留的 key 挪到新位置。要注意，保留 token 的 K/V 当初看过被删内容，所以这样做与只用保留文本重新 prefill 并不相同。'), ('08b_rope_yarn.md', '## P12.13 延伸阅读：NoPE {#p12.13-nope}', 'NoPE 指某些层的 Q、K 不加任何显式位置编码，只靠因果 mask 间接感知顺序。[Kimi K3 技术报告，arXiv:2607.24653](https://arxiv.org/abs/2607.24653)让全部 MLA 层都用 NoPE，位置信息交给线性注意力层（见 [P16.14](#p16.14-production-hybrid)）。'), ('08c_modern_block.md', '## P13.13 延伸阅读：SiTU-GLU {#p13.13-situ-glu}', 'P13.4 的 SwiGLU 两个因子都没有上界，两个大坐标相乘就是离群值。[Kimi K3 技术报告，arXiv:2607.24653](https://arxiv.org/abs/2607.24653)的 SiTU-GLU 用软上限 `β tanh(x/β)` 同时限制两个因子。'), ('08d_grouped_kv_mla.md', '## P14.13 延伸阅读：VHA', '[Virtual Head Attention，arXiv:2610.17959](https://arxiv.org/abs/2610.17959) 的思路是：缓存仍按 GQA 只存少数 KV 头，训练时额外加两次可学习的头间混合，推理前把它们吸收进普通权重，部署形态仍是 GQA。论文报告的质量对比本书没有复现。'), ('08f_sparse_linear_attention.md', '## P16.13 延伸阅读：MSA', 'P16.4 的固定窗口是事先定好看哪里。[MiniMax Sparse Attention，arXiv:2606.13392](https://arxiv.org/abs/2606.13392) 改为让每个 query 给历史分块打分，只读分数最高的几块，并总是保留当前所在的局部块。'), ('08f_sparse_linear_attention.md', '## P16.14 延伸阅读：生产级混合 {#p16.14-production-hybrid}', '生产模型通常不把注意力全换成线性层，而是保留少量全局注意力层做精确检索。例如[Kimi K3 技术报告，arXiv:2607.24653](https://arxiv.org/abs/2607.24653)每 3 层 KDA 线性注意力配 1 层 Gated MLA；KDA 的逐通道遗忘门来自 [Kimi Linear，arXiv:2510.26692](https://arxiv.org/abs/2510.26692)。'), ('08h_moe.md', '## P18.13 延伸阅读：Stable LatentMoE 与 Quantile Balancing {#p18.13-latent-moe-qb}', '[LatentMoE，arXiv:2601.18089](https://arxiv.org/abs/2601.18089) 把路由专家放到更窄的 latent 宽度上计算，少搬数据。[Kimi K3 技术报告，arXiv:2607.24653](https://arxiv.org/abs/2607.24653)在此基础上用 Quantile Balancing 做负载均衡：不再按固定步长调偏置，而是直接算出让每个专家接到目标 token 数的偏置。'), ('08l_preference_optimization.md', '## P22.11 延伸阅读：on-policy 蒸馏与多教师合并（MOPD） {#p22.11-mopd}', 'On-policy 蒸馏让学生自己采样，再请老师给学生写下的每个 token 打分，公开讲解见 [Thinking Machines Lab，On-Policy Distillation（2025）](https://thinkingmachines.ai/blog/on-policy-distillation/)。[Kimi K3 技术报告，arXiv:2607.24653](https://arxiv.org/abs/2607.24653)用这种做法把多个领域的 RL 专家合并进一个模型，称为 Multi-Teacher On-Policy Distillation（MOPD）。'), ('08l_preference_optimization.md', '## P22.12 延伸阅读：Softmax Cancellation {#p22.12-softmax-cancellation}', '这篇论文已在 [P04 3.3a](#p04-softmax-collapse) 介绍；其 §3 讨论软目标交叉熵梯度 $q-p$ 的抵消与修法，本书不展开。'), ('08m_reasoning_mtp.md', '## P23.13 延伸阅读：更便宜的草稿', 'P23.7 的验收规则不关心草稿从哪来：只要每轮都由目标模型贪心验收，输出就等于普通贪心。所以草稿可以换成更便宜或更准的方法，例如一次并行写出整块草稿的 [DFlash，arXiv:2602.06036](https://arxiv.org/abs/2602.06036)、按逐位置置信度截断草稿的 [DSpark，arXiv:2607.05147](https://arxiv.org/abs/2607.05147)，以及沿预测距离递归、复用主干最后一层的 [ECHO，arXiv:2610.24766](https://arxiv.org/abs/2610.24766)。'), ('08m_reasoning_mtp.md', '## P23.14 延伸阅读：推理强度预算与 LK loss {#p23.14-budget-lk}', '[Kimi K3 技术报告，arXiv:2607.24653](https://arxiv.org/abs/2607.24653)在 RL 中给每道题估一个 token 预算，轨迹超出预算的若干倍就把任务奖励改成 −1，以此训练出不同推理强度的档位。[LK Losses，arXiv:2602.23881](https://arxiv.org/abs/2602.23881) 则让草稿直接优化接受率 `Σ_x min(p(x), q(x))` 的负对数，而不是 KL。'), ('08m_reasoning_mtp.md', '## P23.15 延伸阅读：batch 变大时的投机解码 {#p23.15-eagle4}', 'batch 小时解码受内存带宽限制，多验证几个草稿位置几乎不增加耗时；batch 变大后矩阵乘吃满算力，验证就要付真实的算力账。EAGLE-4（[论文 PDF](https://github.com/eagle-hq/EAGLE-4/blob/3b96ef6e1c0b2ef0c698fab74c8e383d8bf8053d/EAGLE4_Paper.pdf)）按实测的起草和验证代价，每轮决定起草多少、用树还是用链。'), ('08m_reasoning_mtp.md', '## P23.16 延伸阅读：FlashMap {#p23.16-flashmap}', '[FlashMap，arXiv:2610.24365](https://arxiv.org/abs/2610.24365) 用 flow map 从噪声一步跳到整块草稿，在单次前向里保留块内各位置的依赖，再交给目标模型并行验证。'), ('08n_quantized_serving.md', '## P24.13 延伸阅读：混合位宽 GGUF', 'P24.7 说过 GGUF 只是容器，同一个文件里不同张量可以用不同量化类型。常见做法是给对量化更敏感的张量多分几个 bit；哪些张量敏感，要靠困惑度或任务评估来定。'), ('08n_quantized_serving.md', '## P24.14 延伸阅读：量化感知训练（QAT）', '前面的方法都是训练后量化。QAT 在训练中就让前向经过量化，让模型适应舍入误差，早期做法见 [Jacob 等，arXiv:1712.05877](https://arxiv.org/abs/1712.05877)。'), ('08n_quantized_serving.md', '## P24.15 延伸阅读：FP4 块缩放 {#p24.15-fp4-block}', 'FP4 块格式让每一小块元素共享一个 scale，块越小，异常值影响的范围越小：MXFP4 每 32 个元素共享一个 E8M0 scale（[MX 格式论文，arXiv:2310.10537](https://arxiv.org/abs/2310.10537) Table 1），NVFP4 每 16 个元素共享一个 FP8 E4M3 scale（[Transformer Engine NVFP4 文档](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html)）。'), ('08o_qwen3_case.md', '## P25.14 延伸阅读：读其他模型的配置 {#p25.14-config-compare}', '读其他模型时，可以照本章的方法逐字段对照官方配置或报告，例如[Kimi K3 技术报告，arXiv:2607.24653](https://arxiv.org/abs/2607.24653)的 Table 1。'), ('09_modern_llms.md', '### mHC、Engram、DSpark', 'DeepSeek-V4.1-Flash [报告的 §2.4.1–2.4.3][V41-paper] 还介绍了 mHC、Engram 和 DSpark：一个改 residual stream 的混合，一个提供条件查表记忆，一个服务于草稿生成。本章只读配置字段，不实现它们。')]

REMOVED_CODE = ['03b_softmax_collapse.py', '14b_attnres_situ.py', '12b_cache_engineering.py', '15b_vha.py', '17b_msa_block_topk.py', '17c_kda_hybrid.py', '19b_latent_moe_qb.py', '23b_mopd_reward.py', '23c_softmax_cancellation.py', '24b_spec_drafts.py', '24c_budget_lk.py', '24d_eagle4_cost.py', '24e_flashmap_joint.py', '25b_mixed_bit.py', '25c_fp4_block.py']
REMOVED_TESTS = ['test_softmax_fix.py', 'test_attnres_situ.py', 'test_cache_engineering.py', 'test_vha.py', 'test_msa_block_topk.py', 'test_kda_hybrid.py', 'test_latent_moe_qb.py', 'test_mopd_reward.py', 'test_spec_drafts.py', 'test_budget_lk.py', 'test_eagle4_cost.py', 'test_flashmap_joint.py', 'test_mixed_bit.py', 'test_fp4_block.py']


class Pointers(unittest.TestCase):
    def test_each_section_is_exactly_its_pointer(self):
        for fn, head, body in POINTERS:
            with self.subTest(heading=head):
                self.assertEqual(section(read(fn), head), body)

    def test_pointers_are_short(self):
        for fn, head, body in POINTERS:
            with self.subTest(heading=head):
                self.assertNotIn("\n", body)
                self.assertLessEqual(len(re.findall("[。！？]", body)), 3)

    def test_kept_anchors_and_links_to_them(self):
        kept = {m.group(1) for _, head, _ in POINTERS for m in [re.search(r"\{#([^}]+)\}", head)] if m}
        self.assertEqual(len(kept), 14)
        linked = set()
        for p in CH.glob("*.md"):
            linked.update(a for a in re.findall(r"\]\(#([^)\s]+)\)", p.read_text(encoding="utf-8")) if a in kept)
        self.assertIn("p16.14-production-hybrid", linked)
        self.assertIn("p04-softmax-collapse", linked)

    def test_training_loop_note(self):
        text = read("07a_training_loop.md")
        self.assertIn("> [!NOTE]\n> 稳定版也有精度极限：正确答案的 logit 间隔很大时，目标梯度会舍入成 0，"
                      "详见 [P04 的 3.3a](#p04-softmax-collapse)。\n", text)
        self.assertNotIn("Slingshot", text)

    def test_quantized_intro(self):
        text = read("08n_quantized_serving.md")
        self.assertIn("`25_quantized_serving.py` 的默认实验只用 Python 标准库、CPU、小数组，不下载权重。可选实战", text)


class Removed(unittest.TestCase):
    def test_scripts_and_tests_gone(self):
        for name in REMOVED_CODE:
            for base in (ROOT / "code", ROOT / "dist/code"):
                self.assertFalse((base / name).exists(), base / name)
        for name in REMOVED_TESTS:
            self.assertFalse((ROOT / "tests" / name).exists(), name)
        self.assertFalse((ROOT / "assets/diagrams/p23/echo_loss.svg").exists())

    def test_no_chapter_links_missing_code(self):
        for p in CH.glob("*.md"):
            for rel in re.findall(r"\(\.\./(code/[^)#\s]+\.py)\)", p.read_text(encoding="utf-8")):
                with self.subTest(file=p.name, target=rel):
                    self.assertTrue((ROOT / rel).exists())

    def test_no_chapter_mentions_removed_code(self):
        for p in CH.glob("*.md"):
            text = p.read_text(encoding="utf-8")
            for name in REMOVED_CODE:
                with self.subTest(file=p.name, script=name):
                    self.assertNotIn(name.removesuffix(".py"), text)

    def test_readme_and_dist_do_not_name_removed_code(self):
        files = [ROOT / "README.md", ROOT / "dist/transformer_from_scratch.md",
                 ROOT / "dist/transformer_from_scratch.html", ROOT / "dist/index.html"]
        for f in files:
            text = f.read_text(encoding="utf-8")
            for name in REMOVED_CODE:
                with self.subTest(file=f.name, script=name):
                    self.assertNotIn(name.removesuffix(".py"), text)

    def test_echo_diagram_gone_everywhere(self):
        self.assertEqual(sorted(str(p.relative_to(ROOT)) for p in ROOT.rglob("echo_loss*")
                                if ".git" not in p.parts), [])

    def test_softmax_fix_only_in_pointer_in_ch03(self):
        text = read("03_qkv_self_attention.md")
        self.assertEqual(text.count("arxiv.org/abs/2610.31642"), 1)
        self.assertNotIn("jonbarron.info/softmaxfix", text)

    def test_ch08f_verification_sentence(self):
        text = read("08f_sparse_linear_attention.md")
        self.assertIn("上述原始来源与官方入口的核验日期见附录 H。CPU 实验只验证数学表达", text)
        self.assertNotIn("MSA 除外", text)

    def test_ch08b_cache_edit_sentence(self):
        self.assertIn("P11.10 的 [KV 编辑](#p11.10-cache-engineering)正是这样用 RoPE 旋转把保留的 key 挪到新位置，压紧被删内容留下的空位；但它只改位置，改不掉 K/V 内容里已经混入的上下文。", read("08b_rope_yarn.md"))
        self.assertNotIn("`repos`", read("08b_rope_yarn.md"))

    def test_appendix_records_gone(self):
        text = read("10_appendix.md")
        for gone in ("### E.4 前沿跟踪记录", "ECHO", "[softmax-fix]", "[akv]", "Slingshot"):
            self.assertNotIn(gone, text)

    def test_mtp_sources_pruned(self):
        srcs = read("08m_reasoning_mtp.md").split("## 资料版本与复现边界", 1)[1]
        for gone in ("DFlash", "ECHO", "EAGLE-4", "FlashMap", "LK losses", "Nemotron"):
            self.assertNotIn(gone, srcs)
        self.assertIn("核验日期见附录 H。以下固定论文版本", srcs)


if __name__ == "__main__":
    unittest.main()
