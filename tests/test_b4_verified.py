"""Verified 待核实 items (bot-hub#123 B4).

Each test pins the corrected (or verified-unchanged) sentence positively and,
where the fact lives in this repository, derives it from the code or data.
"""
import ast
import json
import os
import pathlib
import re
import subprocess
import sys
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


class RopeQueryTable(unittest.TestCase):
    @unittest.skipUnless(has_torch(), "torch required")
    def test_t64_row_matches_rerun(self):
        # Fresh interpreter: the result must not depend on torch state left by earlier tests.
        script = (
            "import json, runpy, sys, torch\n"
            "torch.set_num_threads(1)\n"
            "ns = runpy.run_path(sys.argv[1])\n"
            "print(json.dumps(ns['length_experiment'](seed=2026)['rows']))\n"
        )
        env = dict(os.environ, OMP_NUM_THREADS="1")
        out = subprocess.run([sys.executable, "-c", script, str(CODE / "13_rope.py")],
                             capture_output=True, text=True, env=env, check=True).stdout
        rows = json.loads(out.strip().splitlines()[-1])
        text = read("08b_rope_yarn.md")
        table = text.split("### 表 P12-6", 1)[1].split("\n\n", 2)[1]
        stated = {}
        for line in table.splitlines()[2:]:
            cells = [c.strip() for c in line.strip("|").split("|")]
            stated[int(cells[0])] = (int(cells[1]), float(cells[2]), float(cells[3]))
        self.assertEqual(sorted(stated), [row["length"] for row in rows])
        for row in rows:
            count, loss_sum, mean_ce = stated[row["length"]]
            with self.subTest(length=row["length"]):
                self.assertEqual(count, row["count"])
                # Table shows 3 / 6 decimals; allow last-digit rounding noise only.
                self.assertLessEqual(abs(row["loss_sum"] - loss_sum), 1.5e-3)
                self.assertLessEqual(abs(row["mean_ce"] - mean_ce), 1.5e-6)
        self.assertIn("| 64 | 512 | 594.219 | 1.160584 |", table)


class ThroughputWindows(unittest.TestCase):
    def test_windows_stated(self):
        text = read("08e_attention_systems.md")
        static_end = max(int(x) for x in re.findall(r"静态 batch 则是 (\d+)、(\d+) ms", text)[0])
        cont_end = max(int(x) for x in re.findall(r"两个输出时间为 (\d+)、(\d+) ms", text)[0])
        self.assertEqual((static_end, cont_end), (60, 40))
        self.assertIn(f"`8/0.{static_end // 10:02d}≈133.333`", text)
        self.assertIn(f"注意两个窗口不一样长：各自从 t=0 算到自己最后一个输出 token，静态 {static_end} ms，连续 {cont_end} ms。", text)


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
    def test_v41_kv_scale_sentence(self):
        # DeepSeek-V4.1-Flash inference/model.py@314113f _compress_kv:
        # fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)
        text = read("09_modern_llms.md")
        self.assertIn("源码按 16 通道一组使用 E4M3 scale", text)
        self.assertIn("DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/model.py", text)


if __name__ == "__main__":
    unittest.main()
