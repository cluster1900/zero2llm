"""Verified 待核实 items (bot-hub#123 B4).

Each test pins the corrected (or verified-unchanged) sentence positively and,
where the fact lives in this repository, derives it from the code or data.
"""
import ast
import json
import os
import pathlib
import re
import runpy
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CH = ROOT / "chapters"
CODE = ROOT / "code"


def read(name):
    return (CH / name).read_text(encoding="utf-8")


def imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def has_torch():
    try:
        import torch  # noqa: F401
        return True
    except ModuleNotFoundError:
        return False


class EncoderDecoderPrefixes(unittest.TestCase):
    def test_two_prefix_snapshots_are_explained(self):
        svg = (ROOT / "assets/diagrams/transformer_full_architecture.svg").read_text(encoding="utf-8")
        text = read("07_encoder_decoder.md")
        svg_prefix = re.search(r"前缀提示 \(例如: &quot;&lt;BOS&gt; ([^&]+)&quot;\)|前缀提示 \(例如: \"&lt;BOS&gt; ([^\"]+)\"\)", svg)
        self.assertIsNotNone(svg_prefix)
        svg_words = next(g for g in svg_prefix.groups() if g)
        mer = re.search(r'A\["前缀：起始符 \+ ([^<]+)<br/>', text).group(1)
        target = re.search(r"预测下一个字：(.)", text).group(1)
        extra = svg_words[len(mer):]
        self.assertTrue(svg_words.startswith(mer))
        self.assertEqual(extra[0], target)
        self.assertIn(f"图 7-2 的前缀「`<BOS> {svg_words}`」与下面解码器简图的前缀「起始符 + {mer}」"
                      f"是同一句译文在不同生成步的快照：前者已多生成了「{extra}」两个字，后者正在预测「{target}」。", text)
        self.assertEqual(len(extra), 2)


class BabyGptCheckpointSnippet(unittest.TestCase):
    def snippet(self):
        text = read("08_baby_transformer.md")
        sec = text.split("## 8.7 保存再加载", 1)[1].split("## 8.8", 1)[0]
        self.assertIn("`main()` 只训练并打印结果，不会写 checkpoint；文件名以数字开头，也不能直接 `import 07_baby_gpt`。", sec)
        return re.search(r"```python\n(.*?)```", sec, re.S).group(1)

    def test_main_does_not_save(self):
        src = (CODE / "07_baby_gpt.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        called = {n.func.id for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertNotIn("save_checkpoint", called)
        # load_checkpoint already returns eval(); the snippet still shows it explicitly.
        self.assertTrue(self.snippet().rstrip().endswith("restored.eval()"))
        self.assertFalse("07_baby_gpt".isidentifier())
        self.snippet()

    @unittest.skipUnless(has_torch(), "torch required")
    def test_snippet_runs_from_repo_root(self):
        code = self.snippet()
        old = os.getcwd()
        with tempfile.TemporaryDirectory() as d:
            os.symlink(CODE, pathlib.Path(d) / "code")
            os.chdir(d)
            try:
                import io, contextlib
                ns, buf = {}, io.StringIO()
                with contextlib.redirect_stdout(buf):
                    exec(compile(code, "snippet", "exec"), ns)
                self.assertEqual(buf.getvalue(), "", "run_path must not execute main()")
                self.assertTrue((pathlib.Path(d) / "baby_gpt.pt").exists())
                self.assertFalse(ns["restored"].training)
            finally:
                os.chdir(old)


class AkvMessageIndices(unittest.TestCase):
    def test_zero_based_indices(self):
        text = read("08a_kv_cache.md")
        self.assertIn('`"drop_message": {"4": [2]}` 表示处理完下标 4 的消息后丢掉下标 2 的消息的 KV，'
                      '`"reposition": [4]` 表示处理完下标 4 的消息后压紧位置。'
                      'README 的消息下标从 0 开始数：示例里下标 2 是第一条工具结果 Record A，下标 4 是第二条工具结果 Record B。', text)
        # README Quick Start payload (pinned commit c567a42): roles by index.
        roles = ["user", "assistant", "tool", "assistant", "tool"]
        tools = [i for i, r in enumerate(roles) if r == "tool"]
        self.assertEqual(tools, [2, 4])
        self.assertIn("c567a429a058f1abce3b2879e69978b09e0fda4c", text)


class RopeQueryTable(unittest.TestCase):
    @unittest.skipUnless(has_torch(), "torch required")
    def test_t64_row_matches_rerun(self):
        import torch
        torch.set_num_threads(1)
        ns = runpy.run_path(str(CODE / "13_rope.py"))
        result = ns["length_experiment"](seed=2026)
        text = read("08b_rope_yarn.md")
        table = text.split("### 表 P12-6", 1)[1].split("\n\n", 2)[1]
        stated = {}
        for line in table.splitlines()[2:]:
            cells = [c.strip() for c in line.strip("|").split("|")]
            stated[int(cells[0])] = cells
        self.assertEqual(sorted(stated), [row["length"] for row in result["rows"]])
        for row in result["rows"]:
            cells = stated[row["length"]]
            with self.subTest(length=row["length"]):
                # Exact: length, count and loss_sum (3 decimals).
                self.assertEqual(cells[:3], [str(row["length"]), str(row["count"]), f"{row['loss_sum']:.3f}"])
                # mean_ce: last (6th) decimal may differ by ±1 (Wu, 2026-11-05; CI gave
                # 1.160583 vs 1.160584 on #44, cause unknown). ±2 or more fails.
                self.assertLessEqual(abs(float(f"{row['mean_ce']:.6f}") - float(cells[3])), 1e-6 + 1e-12)
        self.assertIn("| 64 | 512 | 594.219 | 1.160584 |", table)


class ThroughputWindows(unittest.TestCase):
    def test_windows_stated(self):
        text = read("08e_attention_systems.md")
        static_end = max(int(x) for x in re.findall(r"静态 batch 则是 (\d+)、(\d+) ms", text)[0])
        cont_end = max(int(x) for x in re.findall(r"两个输出时间为 (\d+)、(\d+) ms", text)[0])
        self.assertEqual((static_end, cont_end), (60, 40))
        self.assertIn(f"`8/0.{static_end // 10:02d}≈133.333`", text)
        self.assertIn(f"注意两个窗口不一样长：各自从 t=0 算到自己最后一个输出 token，静态 {static_end} ms，连续 {cont_end} ms。", text)


class MuseSparkLayerTypes(unittest.TestCase):
    def test_default_pattern(self):
        text = read("08f_sparse_linear_attention.md")
        n = 62  # MuseSparkTextConfig.num_hidden_layers default at d4804f9
        # Rule from configuration_muse_spark.py __post_init__ at that commit.
        types = ["full_attention" if (n - 1 - i) % 4 == 0 else "sliding_attention" for i in range(n)]
        full = [i for i, t in enumerate(types) if t == "full_attention"]
        self.assertIn("它没有线性层，而是局部与全局注意力混排。", text)
        self.assertIn(f"默认 {n} 层时，全局层是第 {full[0]}、{full[1]}、…、{full[-1]} 层（从 0 数），"
                      f"共 {len(full)} 层，局部 {n - len(full)} 层。所以它不是一层局部、一层全局地严格交替，而是 3 局部 : 1 全局。", text)
        self.assertIn("d4804f9d76b8c071b993d813e9e87067b780cbee", text)
        self.assertNotIn("局部与全局注意力交替", text)
        # Derive the wording of the rule from the rule itself.
        period = full[1] - full[0]
        self.assertTrue(all(b - a == period for a, b in zip(full, full[1:])))
        self.assertEqual(full[-1], n - 1)          # anchored at the LAST layer
        self.assertNotEqual(full[0], 0)             # so not counted from the first
        self.assertEqual(period - 1, 3)             # matches "3 局部 : 1 全局"
        self.assertIn(f"给出的默认规则是：从最后一层往前数，每 {period} 层有 1 层全局（`full_attention`），"
                      "其余是局部（`sliding_attention`）。", text)
        para = text.split("**Muse Spark Open（S33）。**", 1)[1].split("\n\n", 1)[0]
        self.assertNotIn("未验证", para)


class EagleVocab(unittest.TestCase):
    def test_full_vocab_not_draft(self):
        text = read("08m_reasoning_mtp.md")
        m = re.search(r"目标 Qwen3.8-27B 的完整词表 (\d+)，TP4 下每片 (\d+)；不是截断词表", text)
        self.assertIsNotNone(m)
        vocab, shard = int(m.group(1)), int(m.group(2))
        # bench_kernels.py L67 @3b96ef6: "vocab 248320, TP4 shard 62080" -> shard is vocab/4.
        self.assertEqual(vocab, 4 * shard)
        self.assertIn(f"248320 也是 Qwen3.8-27B config.json 里 text_config 的 `vocab_size`", text)
        self.assertIn("Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json", text)
        self.assertNotIn("草稿词表 248320", text)


class FrameworkInstallHint(unittest.TestCase):
    def test_hint_matches_chapter_install(self):
        text = read("08j_training_frameworks.md")
        url = re.search(r"--index-url (\S+) torch==([\d.]+)", text)
        src = (CODE / "21_training_frameworks.py").read_text(encoding="utf-8")
        fn = src.split("def framework_status", 1)[1].split("\ndef ", 1)[0]
        hint = "".join(ast.literal_eval(m) for m in re.findall(r'"(?:[^"\\]|\\.)*"', fn.split(".join(issues) +", 1)[1].split("if issues", 1)[0]))
        self.assertEqual(hint, f". In a separate environment, install torch=={url.group(2)} from "
                               f"{url.group(1)}, then requirements-frameworks.txt; skipped is not passed.")
        self.assertNotIn("requirements.txt and", hint)


class DeepSeekV3BiasRule(unittest.TestCase):
    def test_attribution(self):
        text = read("08h_moe.md")
        self.assertIn("那条规则来自 DeepSeek-V3（[arXiv:2412.19437](https://arxiv.org/abs/2412.19437) §2.1.2），"
                      "原文用文字描述：每步结束时，过载专家的偏置减 γ，欠载专家的偏置加 γ。"
                      "写成公式 `b_j ← b_j + γ sign(ℓ̄ − ℓ_j)` 是 S29 的转写，不是 V3 原文的公式。", text)


class QuantizedDependencies(unittest.TestCase):
    def test_default_vs_25b(self):
        text = read("08n_quantized_serving.md")
        self.assertNotIn("torch", imports(CODE / "25_quantized_serving.py"))
        self.assertIn("torch", imports(CODE / "25b_mixed_bit.py"))
        self.assertIn("`25_quantized_serving.py` 的默认实验只用 Python 标准库、CPU、小数组，不下载权重；"
                      "P24.13 的 `25b_mixed_bit.py` 另需 PyTorch（CPU 即可）。", text)
        self.assertIn("## P24.13 混合位宽 GGUF", text)


class QwenTokenizerLength(unittest.TestCase):
    def test_model_max_length_mentioned(self):
        text = read("08o_qwen3_case.md")
        vals = set()
        for kind in ("dense", "moe"):
            tok = json.loads((CODE / "data/qwen3" / kind / "tokenizer_config.json").read_text(encoding="utf-8"))
            cfg = json.loads((CODE / "data/qwen3" / kind / "config.json").read_text(encoding="utf-8"))
            vals.add((tok["model_max_length"], cfg["max_position_embeddings"]))
        self.assertEqual(len(vals), 1)
        mml, mpe = vals.pop()
        self.assertIn(f"两个 config 的 `max_position_embeddings` 都是 {mpe}", text)
        self.assertIn(f"两个快照的 tokenizer_config.json 里 `model_max_length` 是 {mml}，"
                      "这是 tokenizer 截断和长度警告用的上限，不是模型配置，也不证明模型能处理这么长。", text)


class CachePollutionNegative(unittest.TestCase):
    def test_shapes_from_code(self):
        src = (CODE / "28_minigpt_healthcheck.py").read_text(encoding="utf-8")
        neg = src.split("def negative_controls", 1)[1].split("\ndef ", 1)[0]
        a_len = int(re.search(r"_, stale = good\(base, tokens\[:, :(\d+)\], None\)", neg).group(1))
        b_len = int(re.search(r"other = \(tokens\[:, :(\d+)\]", neg).group(1))
        tree = ast.parse(src)
        fx = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "fixture")
        rows = next(n for n in ast.walk(fx) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "tensor").args[0]
        batch = len(rows.elts)  # rows of the fixture() tensor literal
        self.assertIn("# Same model/weight signature and batch size (lengths differ), WRONG request history", src)
        text = read("09a_minigpt_healthcheck.md")
        self.assertIn("### 负例三：后一个请求接上了前一个请求的历史", text)
        self.assertIn(f"请求 A 是一批 {batch} 条序列的前三个 token（形状 [{batch},{a_len}]），已经产生缓存。"
                      f"请求 B 是另一批 {batch} 条、各 {b_len} 个 token 的输入（形状 [{batch},{b_len}]），", text)
        self.assertIn(f"batch 大小也相同（都是 {batch}，只是长度不同）", text)
        self.assertNotEqual(a_len, b_len)


class VerifiedUnchanged(unittest.TestCase):
    def test_slingshot_sentence(self):
        # arXiv:2610.31642v1 §4: "These Slingshot spikes are a direct consequence of Softmax Collapse".
        text = read("03_qkv_self_attention.md")
        self.assertIn("记住训练集之后，Slingshot 损失尖峰由它引起", text)
        # §8: "extreme margins ... are uncommon in standard large-scale training".
        self.assertIn("论文 §8 也指出，这么大的间隔在标准大规模训练中并不常见", text)

    def test_v41_kv_scale_sentence(self):
        # DeepSeek-V4.1-Flash inference/model.py@314113f _compress_kv:
        # fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)
        text = read("09_modern_llms.md")
        self.assertIn("源码按 16 通道一组使用 E4M3 scale", text)
        self.assertIn("DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/model.py", text)

    @unittest.skipUnless(has_torch(), "torch required")
    def test_echo_heldout_accuracy(self):
        import io, contextlib, torch
        torch.set_num_threads(1)
        ns = runpy.run_path(str(CODE / "24b_spec_drafts.py"))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ns["main"]([])
        acc = re.search(r"held-out accuracy at horizons 1/2/3 = ([\d./]+)", buf.getvalue()).group(1).split("/")
        self.assertEqual(len(set(acc)), 1)
        self.assertIn(f"在留出数据上 horizon 1/2/3 的贪心准确率都是 {acc[0]}", read("08m_reasoning_mtp.md"))


if __name__ == "__main__":
    unittest.main()
