"""Machine-checkable factual corrections (bot-hub#123 B3)."""
import json
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CH = ROOT / "chapters"


def read(name):
    return (CH / name).read_text(encoding="utf-8")


class FactualCorrections(unittest.TestCase):
    def test_preface_env_matches_ci_and_09a(self):
        pre = read("00_preface.md")
        self.assertNotIn("本批在 Python 3.12 上验证", pre)
        ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        ci_py = re.search(r"python-version: '([\d.]+)'", ci).group(1)
        ci_torch = re.search(r"torch==([\d.]+)", ci).group(1)
        self.assertIn(f"CI 在 Python {ci_py} 加 PyTorch {ci_torch}", pre)
        hc = read("09a_minigpt_healthcheck.md")
        m = re.search(r"Python (3\.\d+\.\d+)、PyTorch ([\w.+]+)、", hc)
        self.assertIn(f"Python {m.group(1)}、PyTorch {m.group(2)}", pre)

    def test_preface_says_discover_runs_whole_suite(self):
        pre = read("00_preface.md")
        self.assertIn("最后一条运行**全书**的回归测试", pre)
        n = sum(1 for _ in (ROOT / "tests").glob("test_*.py"))
        self.assertGreater(n, 1)

    def test_08f_local_block_matches_code(self):
        code = (ROOT / "code/17b_msa_block_topk.py").read_text(encoding="utf-8")
        self.assertIn("local = n - 1", code)
        self.assertNotIn("causal", code.split("def select_blocks")[1].split("def token_recall")[0])
        text = read("08f_sparse_linear_attention.md")
        self.assertIn("下标 n−1", text)
        self.assertIn("没有因果掩码", text)
        self.assertNotIn("top-(k−1) 个远处块", text)

    def test_08i_only_provenance_checked(self):
        code = (ROOT / "code/20_pretraining_pipeline.py").read_text(encoding="utf-8")
        body = code.split("def clean_deduplicate")[1].split("\ndef ")[0]
        self.assertIn('doc.provenance != "generated-for-this-book"', body)
        self.assertNotIn("doc.source", body.split("raise ValueError")[0])
        text = read("08i_pretraining_pipeline.md")
        self.assertNotIn("程序拒绝其他来源标签", text)
        self.assertIn("`source` 字段不做检查", text)

    def test_09a_points_to_rectangular_mask_section(self):
        self.assertIn("](#p11.4-矩形-mask最容易写错的两行)", read("09a_minigpt_healthcheck.md"))
        self.assertIn("## P11.4 矩形 mask：最容易写错的两行", read("08a_kv_cache.md"))

    def test_06_cross_reference_names_chapter_4(self):
        self.assertNotIn("上一章手写 MHA", read("06_residual_layernorm_ffn.md"))
        self.assertIn("4D^2", read("06_residual_layernorm_ffn.md"))

    def test_08b_pi_and_yarn_separate_sources(self):
        t = read("08b_rope_yarn.md")
        self.assertIn("arXiv:2306.15595", t)
        self.assertNotIn("这里固定版本解释 PI", t)

    def test_04_no_expert_heads(self):
        code = (ROOT / "code/04_multi_head_attention.py").read_text(encoding="utf-8")
        self.assertNotIn("专家头", code)
        self.assertNotIn("专家分队", code)
        self.assertIn("分头并换轴后的 Q", read("04_multi_head_attention.md"))

    def test_08c_bfloat16_range_claim(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        info = torch.finfo(torch.bfloat16)
        self.assertEqual(f"{info.max:.1e}", "3.4e+38")
        self.assertEqual(torch.finfo(torch.float32).bits - info.bits, 16)
        self.assertTrue(torch.isfinite(torch.tensor(40000.0, dtype=torch.bfloat16).square()))
        self.assertFalse(torch.isfinite(torch.tensor(40000.0, dtype=torch.float16).square()))
        t = read("08c_modern_block.md")
        self.assertIn("最大约 3.4×10³⁸，40000 的平方不会溢出", t)
        self.assertIn("只有 8 位有效尾数", t)


if __name__ == "__main__":
    unittest.main()
