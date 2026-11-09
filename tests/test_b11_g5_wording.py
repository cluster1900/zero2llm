"""B11 group 5 (08E–08H) plus G4 rulings (08A、08B): wording pins (bot-hub#123)."""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# G4 裁定
G4 = [
("chapters/08a_kv_cache.md","参数更新后旧缓存无效。实现核对参数对象身份、版本、dtype 与 device，包含 `assign=True` 替换参数的情况；不支持通过 `.data` 等方式故意绕过 PyTorch 版本跟踪。","参数更新或替换后，旧缓存会被拒绝。"),
("chapters/08b_rope_yarn.md","源码 `rotation_walkthrough()` 用 float64 生成上表，`test_rotation_matches_independent_float64_matrix_and_preserves_tail` 还用独立的 2×2 矩阵验证实现，","源码 `rotation_walkthrough()` 用 float64 生成上表；本章另用独立的 2×2 矩阵核对实现，"),
# bot-hub#126 F5：扩展倍数 s→κ、温度 t→τ（附录 A.5），句意不变。
("chapters/08b_rope_yarn.md","s>1 时 a>1，即 t<1","点积会乘 **a²**，不是 a。$\\kappa>1$ 时 $a>1$，即 $\\tau<1$，logits 被放大，注意力分布变得更尖。"),
]

# G5 机械修正
G5 = [
("chapters/08h_moe.md","router 第一列为 [0,log2,log4]","router 第一列为 [0,ln 2,ln 4]"),
("chapters/08h_moe.md","输入第一维梯度为 78+20log2，","输入第一维梯度为 78+20 ln 2，"),
]


# G5 裁定（review 2026-11-05）
G5R = [
('chapters/08e_attention_systems.md','不是“连续 batch 一定快 1.5 倍”的结论','注意两个窗口不一样长：各自从 t=0 算到自己最后一个输出 token，静态 60 ms，连续 40 ms。若同用 0–60 ms 窗口，两者都完成 8 个 token，都是 `8/0.06≈133.333` tokens/s；200 只是来自更短的 40 ms 窗口。'),
('chapters/08e_attention_systems.md','PagedAttention 有配合分页布局读取 KV 的注意力计算方式；','本章的「块」有三种意思，分别出现在三层：FlashAttention 的计算分块（一次注意力内部的 query/key 块）、PagedAttention 的 KV 物理块（固定 token 数的存储页）、调度器的 chunked prefill 分块（把一个 prompt 拆成多轮）。后文按上下文写明是哪一种。\n\nPagedAttention 有配合分页布局读取 KV 的注意力计算方式；'),
('chapters/08e_attention_systems.md','不能用本脚本的运行时间或进程内存证明','本章只实现其中容易手算的**前向归一化递推**。'),
('chapters/08e_attention_systems.md','固定版本 vLLM 文档说明了块级缓存查找的设计。[vLLM 0.6.4','固定版本 vLLM 文档说明了块级缓存查找的设计（版本见本章末资料）。'),
('chapters/08e_attention_systems.md','这里引用 0.6.4 是为了固定机制说明的文档快照','物理块就释放了。\n'),
('chapters/08e_attention_systems.md','此固定源码链接用于接口背景，不是安装建议。','“调用了 SDPA”本身不能证明走了期望后端。[PyTorch 固定源码 v2.5.1：SDPA 后端选择接口](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/nn/attention/__init__.py)\n\n本章另检查了'),
('chapters/08e_attention_systems.md','这是选修实验的记录清单','要做真正的性能对比，需要另外安排可复现基准：固定模型 revision、dtype 和工作负载（长度分布、到达方式、并发、前缀命中率），预热后重复多轮并正确同步计时；同时报 TTFT、TPOT、E2E、输出吞吐和失败请求，分项记录显存，并用 profiler 核验实际后端。'),
('chapters/08f_sparse_linear_attention.md','或者把线性层与普通注意力层交替。','或者让线性层与普通注意力层按比例混合。'),
('chapters/08g_ced_architecture.md','20 个 decoder 层的压缩比是 1。配置','20 个 decoder 层的压缩比是 1（报告 §2.3：压缩比为 1 即不压缩的 main KV）。'),
('chapters/08g_ced_architecture.md','本章核验了技术报告 v1（核验日期见附录 H），以及官方 Hugging Face 仓库的固定 revision','核验所用的报告版本与仓库固定 revision 见本章末「资料与实现边界」。\n'),
('chapters/08g_ced_architecture.md','`tensor[:, -0:]` 会意外保留全部历史','缓存是调用者持有的快照，新请求传 None；增量调用不覆盖旧快照，可以从同一个前缀分叉实验；改权重、换配置或换模型后旧缓存会被拒绝。不要编辑缓存张量，也不要用 `.data` 或底层共享存储写入绕过版本跟踪；这里不承诺检测这些破坏。'),
('chapters/08g_ced_architecture.md','mHC、Engram','、RMSNorm、FFN、量化、视觉输入、其他生产组件与 GPU kernel。'),
('chapters/08g_ced_architecture.md','Full 产生 global KV 并选择索引；','把 CSA2 分成三种模式：\n\n- **Full**：产生 global KV，并选择索引。\n- **Reindex**：复用前面层的 global KV，重新选择索引。\n- **Reuse**：global KV 和选择结果都复用。\n\n三种模式都保留'),
# bot-hub#126 F6 (F0 r4 A.5): window length W→T_win (W 只指权重); same sentence, letter only; old W sentence still asserted absent.
('chapters/08g_ced_architecture.md','因此“最后 W 行还在”不等于“能从这 W 个原始输入精确重建所有深层状态”。','因此“最后 $T_{\\mathrm{win}}$ 行还在”不等于“能从这 $T_{\\mathrm{win}}$ 个原始输入精确重建所有深层状态”。[报告 §1](https://arxiv.org/html/2609.19969v1#S1) 给出的精确重建长度是最近 $L\\cdot T_{\\mathrm{win}}$ 个 token（$L$ 为层数；报告记作 L × n_win）。'),
('chapters/08h_moe.md','得到 78x，而不是 74x。','得到 78x，而不是 74x（74x 是把两个共享分支取平均的错误算法：70+(3+5)/2）。'),
]

G5R_TEST_NAMES = ['见 `test_hand_route_weights_counts_aux_and_ties`。', '见 `test_parameter_active_budget_and_fine_grained_example`。', '见 `test_capacity_fixed_order_no_renormalization_shared_survives`。', '见 `test_sparse_execution_empty_experts_and_top1_router_gradient`。', '见 `test_mutations_fail_independent_oracle_not_only_cache_agreement`。', '见 `test_full_cache_alignment_each_mode_and_extreme_k`。', 'test_runtime_and_buffer_mutations_invalidate_cache']

CHANGED = G4 + G5 + G5R


class B11G5Wording(unittest.TestCase):
    def test_changed(self):
        for rel, old, new in CHANGED:
            body = (ROOT / rel).read_text(encoding="utf-8")
            with self.subTest(file=rel, new=new[:30]):
                if old not in new:
                    self.assertEqual(body.count(old), 0)
                self.assertEqual(body.count(new), 1)

    def test_no_test_name_or_log2_left(self):
        self.assertNotIn("test_rotation_matches", (ROOT / "chapters/08b_rope_yarn.md").read_text(encoding="utf-8"))
        self.assertNotIn("log2", (ROOT / "chapters/08h_moe.md").read_text(encoding="utf-8"))
        self.assertNotIn("assign=True", (ROOT / "chapters/08a_kv_cache.md").read_text(encoding="utf-8"))
        g = (ROOT / "chapters/08g_ced_architecture.md").read_text(encoding="utf-8")
        self.assertEqual(g.count("每 m 个 token 压缩成一条 main KV"), 0)
        self.assertEqual(g.count("（报告 §2.3：压缩比为 1 即不压缩的 main KV）"), 1)

    def test_08h_answers_no_test_names(self):
        body = (ROOT / "chapters/08h_moe.md").read_text(encoding="utf-8")
        for name in G5R_TEST_NAMES:
            self.assertEqual(body.count(name), 0, name)
        self.assertEqual(body.count("见本章测试。"), 7)
        answers = body[body.index("## P18.12"):body.index("## P18.13")]
        self.assertNotIn("`test_", answers)

    def test_08g_sources_section_holds_revision(self):
        body = (ROOT / "chapters/08g_ced_architecture.md").read_text(encoding="utf-8")
        tail = body[body.index("## 资料与实现边界"):]
        self.assertIn("314113f8a7dadb388215396ba5fbee2c610a04d6", tail)
        self.assertIn("核验日期见附录 H", tail)
        self.assertEqual(body.count("```text\n314113f8"), 0)

    def test_08e_single_version_disclaimer(self):
        body = (ROOT / "chapters/08e_attention_systems.md").read_text(encoding="utf-8")
        head = body[:body.index("## 资料与实现边界")]
        self.assertEqual(head.count("](https://docs.vllm.ai/en/v0.6.4/"), 1)
        self.assertEqual(head.count("](https://github.com/pytorch/pytorch/blob/v2.5.1/"), 1)
        self.assertNotIn("不是说它是当前最新版或建议安装的版本", head)
        self.assertEqual(body.count("本章的「块」有三种意思"), 1)


if __name__ == "__main__":
    unittest.main()
