"""B6（第三组）：正文测试账本移到 docs/testing.md 的回归检查。

被移动段落按原文保存在 MOVED 中：它们必须出现在 docs/testing.md，且不再出现在对应章节。
这些检查只锁定确切文本与少量标记词；改写后的同义账本句不会被自动发现。
"""
import ast
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAPTERS = ROOT / "chapters"
TESTING = ROOT / "docs" / "testing.md"
RUN = "运行本章测试文件，末尾应显示 `OK`；测试分类、计数与错误版本清单见 `docs/testing.md`。"
RUN_ALL = "运行上面的测试命令，末尾应显示 `OK`；本章相关的测试分类见 `docs/testing.md`。"
MOVED = {
 "01_word_embedding.md": [
  "测试同时覆盖 `[3,4]` 与 `[6,8]` 的尺度例子、零向量和维度不一致。"
 ],
 "01a_tokenizer.md": [
  "后面测试会特意保留这种情况。",
  "测试不只检查几个好看的输出，还检查空字符串、换行空格、带修饰符的 emoji、组合字符、非法 ID、确定性并列规则、重叠合并，以及文档边界。"
 ],
 "03_qkv_self_attention.md": [
  "测试在小分数上与朴素公式对照，在大分数上验证不溢出，并检查整体平移不改结果。",
  "测试还对空行反向传播，检查梯度有限。"
 ],
 "05_positional_encoding.md": [
  "测试会核对全部 64 个显示值，不只是第一行。",
  "未安装或版本不匹配会明确跳过并给出原因。跳过不是实验通过。与版本判断有关的标准库测试仍可执行。",
  "这些数值答案、两张表、公式的每个元素和相对位移关系，都由 `tests/test_positions.py` 核对。测试也包含“余弦先降后升”的反例，避免把漂亮的波形误读成单调距离。"
 ],
 "06_residual_layernorm_ffn.md": [
  "测试只在安装了兼容的 2.14.x 正式版时执行张量实验，缺失或版本不符会明确跳过，不能把跳过算成数值验收通过。",
  "测试也会逐字核对脚本生成的表格是否出现在正文中，防止只改代码或只改数字。"
 ],
 "08a_kv_cache.md": [
  "`tests/test_kv_cache.py` 对应形状/偏移、矩形 mask、全 logits 对齐、内存公式、独立请求和上限契约；还实际构造错误 offset 和错误 mask 的负面对照。"
 ],
 "08l_preference_optimization.md": [
  "测试刻意使用不同长度，以免相同长度掩盖错误。",
  "语言模型变长 token 的实现契约在前面和测试中单独验证。",
  "测试另外检查示例调用前后的 CPU 随机数状态与线程数，并注入一次中途异常验证恢复路径；将 reference 和原 SFT 对象同时改动的反例，必须被独立快照检查识别。",
  "测试还将错误真正注入可执行源码：翻转 DPO 符号、反转 β、删参考、混组、错标准差、漏对齐、错误 shift / mask / 长度平均、反转或丢掉 KL 权重、让 old 每步跟随 current、错误 clip、全 token 归约等。测试必须抓住这些行为改变，不能只检查源码里有没有某个词。",
  "在上面的基线环境实测，21 个测试通过；不加载第三方包时，7 个标准库测试通过、14 个 PyTorch 测试明确跳过。"
 ],
 "08n_quantized_serving.md": [
  "测试把中文和 emoji 的每一个字节切开，也枚举所有单次切分位置。",
  "`token_latencies` 用这个手算表测试。",
  "默认测试验证独立有理数格点 oracle、手算表、组尾与行边界、整数累加、校准漂移、独立 attention、UTF-8/SSE 和文件生成。它还真正改写并执行七种错误实现：漏减零点、把舍入改成截断、整数点积漏减零点、丢组尾、上下文比较差一位、漏检查 `[DONE]`、每个网络块单独解码 UTF-8。另外八项实际变异覆盖对称/仿射全零 scale、全负校准不含零、最大误差符号、未完成 SSE 事件、丢弃多行 data 换行，以及 tool_calls/error 的拒绝逻辑；加上附录批次的非对齐 GGUF 格式 fixture 回归，合计十六项。异常分支采用完整事件与精确错误原因，避免被后续“不完整”检查偶然拦截。正确基线先通过，错误变体再被同一行为断言抓住。",
  "默认 36 个测试中，35 个运行、1 个真实引擎测试跳过；本次提供 b5046 后，36 个全部实际通过。跳过不等于已运行。"
 ],
 "08o_qwen3_case.md": [
  "这些正是后面的独立数值测试检查的地方。",
  "精确换行由模板 golden 测试比对。",
  "测试会确认：在 user 文本里加 `/no_think`，模板仍保留这段文本，不自动追加 False 的空块。",
  "# 分开报告标准库、模板与数值测试；跳过不是通过。",
  "测试还封锁网络与可选依赖，确认标准库路径仍可完成。",
  "测试分成四层：",
  "1. **来源与算术。** 固定仓库、完整 revision、关键哈希；用独立手写乘法表达式检查真实参数总数，用明确字节数检查两份真实 KV 预算。修改快照或 revision 必须失败。另把 `hidden_size` 的 1024 改为 1025：字节数不变、JSON 仍合法，也必须得到精确的文件完整性错误，才能证明不是只有长度检查在起作用。\n2. **实际结构。** 用 `named_parameters()` 比对微型模型所有名字、形状、绑定去重后的总数；不是只比较两个计数公式。\n3. **独立数值。** Python 标量 oracle 从每个权重元素出发，自己做线性层、RMSNorm、split-half RoPE、因果注意力、SwiGLU 和专家混合，不调用生产 helper 或 tensor matmul。它同时对照 dense/MoE 的最终 logits 与每层真实 K/V。测试把所有 Q/K、input/post-attention 和最终 norm gain 用独立种子 251 均匀抽到 `[0.5,1.5)`，每个模块都不同；否则全为 1 的初始化会掩盖用错同宽度 norm 的错误。这只改变测试样本，不改变生产代码的初始化。\n4. **行为与反例。** 比较任意分块与完整前缀，改未来 token 后确认过去输出保持因果独立。dense 使用零误差检查；MoE 仅允许绝对误差 `1e-12`、相对误差 0，因为未来路由可以改变某专家的 minibatch 形状，进而使 CPU 矩阵乘法末位出现舍入差异，并非放宽注意力掩码规则。还检查 sparse dispatch 真正只调用选中的专家，以及坏缓存、预算边界和 non-thinking 下控制 ID 也计入答案段的约定。",
  "还实际执行 **16 个偏离本练习契约的变体**，并要求独立断言能抓住它们：重复计算绑定 head、用 D/Hq 代替显式 d、漏掉 QK norm 参数、漏计闭合 token、把未闭合思考误作答案、错排 GQA 头、丢失缓存 offset、错用相邻 RoPE 配对、跳过 Q norm、取消选中权重归一化、计算全部专家、再次旋转旧 K、K 错用 Q 的 norm gain、交换 input/post-attention norm、最终输出错用层内 norm、在多个闭合标签下改用最后一个作为分界（偏离本练习的首个闭合约定，并非官方做法错误）。三种 norm 错接是形状仍合法的真实源码替换，独立标量参考和可选官方参考都在 dense／MoE 两条路径上拒绝它们。不是在注释里列出错误后就算“做过负例”。",
  "此外，利用已有的独立环境，实际运行了可选官方参照测试 `test_random_tiny_official_dense_and_moe_shapes_logits_and_cache`：Transformers 4.51.3、PyTorch 2.14.1+cpu、CPU float32、种子 89、eager attention，分别构造官方随机 tiny dense／MoE 模型，严格加载同一套微型随机参数。同样先用种子 251 将所有 norm gain 独立抽到 `[0.5,1.5)`，再把这整套权重加载到官方模型。两者参数数目匹配，logits 与逐层 K/V 在 `atol=rtol=3e-6` 下通过；不以全一 gain 或某次误差恰好为零作为等价证据。测试不调用 `from_pretrained`，没有下载模型或 tokenizer；缺少这组独立环境时明确跳过，不能算通过。本次共发现 28 项测试：仓库基线环境 26 项通过、2 项官方参考测试明确跳过；无 PyTorch/Jinja2 的标准库环境 14 项通过、14 项明确跳过；独立官方环境 28 项全部通过。"
 ],
 "09_modern_llms.md": [
  "测试逐项对齐它们。",
  "所以测试还锁住本章声明的 revision 和关键配置值；更新版本时应重新查看原始证据，而不是只改 expected 让测试变绿。",
  "| 专项测试 | 17 项通过，无依赖跳过 | 只覆盖本章代码与快照契约 |",
  "专项测试包含六个**实际执行的数值错误变体**：漏转置、漏位置分支、bias 污染权重、按每层重复存 bank、提前暴露未完成组、调换视觉通道顺序。它们都能返回数值，由数值/坐标断言抓住；没有把语法错误或程序崩溃伪装成“成功检测”。另执行 SWA 来源与 Markdown 竖线转义两项契约变异；它们分别由精确拒绝原因和输出断言检出。此外还测试坏哈希、浮动 revision、路径越界、缺字段、非法 source 调度、补零、空序列账本和配置别名一致性。"
 ]
}

KEPT_MARKER_SENTENCES = {
    ("08l_preference_optimization.md", "测试覆盖"):
        "更复杂的数学、代码、事实问题还涉及等价表达、执行隔离、超时、测试覆盖和标注争议，不能用两题匹配器替代。",
    ("09_modern_llms.md", "错误变体"):
        "**固定对象 → 找到证据 → 说明边界 → 做小实验 → 用独立 oracle 与错误变体检查**",
}


def count_tests(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(
        1
        for cls in tree.body if isinstance(cls, ast.ClassDef)
        for f in cls.body
        if isinstance(f, ast.FunctionDef) and f.name.startswith("test_")
    )


def count_mutants(path, fn="mutant"):
    """统计 fn(...) 执行的不同语义变异：字面量参数的调用计 1；for 循环遍历的列表（可为同函数或模块级变量）按元素计，同一列表元素只计一次。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    module_lists = {
        n.targets[0].id: n.value for n in tree.body
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and isinstance(n.value, (ast.List, ast.Tuple))
    }
    seen = set()
    for func in ast.walk(tree):
        if not isinstance(func, ast.FunctionDef):
            continue
        lists = dict(module_lists)
        lists.update({
            n.targets[0].id: n.value for n in ast.walk(func)
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and isinstance(n.value, (ast.List, ast.Tuple))
        })
        for node in ast.walk(func):
            if isinstance(node, ast.For):
                it = lists.get(node.iter.id) if isinstance(node.iter, ast.Name) else node.iter
                if isinstance(it, (ast.List, ast.Tuple)) and any(
                        isinstance(c, ast.Call) and getattr(c.func, "id", None) == fn for c in ast.walk(node)):
                    seen.update(ast.dump(e) for e in it.elts)
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == fn and node.args \
                    and all(isinstance(a, ast.Constant) for a in node.args):
                seen.add(ast.dump(node))
    return len(seen)


def no_torch_run(pattern, blocked=("torch",)):
    code = (
        f"import sys, os, unittest; sys.modules.update(dict.fromkeys({list(blocked)!r})); "
        f"s = unittest.defaultTestLoader.discover('tests', pattern={pattern!r}); "
        "r = unittest.TextTestRunner(stream=open(os.devnull, 'w')).run(s); "
        "print(r.testsRun, len(r.skipped), len(r.failures) + len(r.errors))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
        env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True,
    ).stdout.split()
    return tuple(map(int, out))


class B6LedgerMoveGroup3Test(unittest.TestCase):
    def test_moved_text_lives_only_in_docs_testing(self):
        doc = TESTING.read_text(encoding="utf-8")
        for name, paras in MOVED.items():
            body = (CHAPTERS / name).read_text(encoding="utf-8")
            self.assertIn("## " + name, doc)
            for p in paras:
                with self.subTest(chapter=name, para=p[:30]):
                    self.assertIn(p, doc)
                    self.assertNotIn(p, body)

    def test_each_chapter_keeps_run_command_and_expectation(self):
        for name in MOVED:
            body = (CHAPTERS / name).read_text(encoding="utf-8")
            with self.subTest(chapter=name):
                m = re.search(r"unittest discover -s tests -p '?(test_\w+\.py)'?", body)
                if m:
                    self.assertEqual((body.count(RUN), body.count(RUN_ALL)), (1, 0))
                    self.assertTrue((ROOT / "tests" / m.group(1)).is_file())
                else:
                    self.assertEqual((body.count(RUN), body.count(RUN_ALL)), (0, 1))
                    self.assertIn("unittest discover -s tests -v", body)

    def test_05_table_value_count_derived_from_test_file(self):
        src = (ROOT / "tests" / "test_positions.py").read_text(encoding="utf-8")
        rows = int(re.search(r"assertEqual\(len\(rows\), (\d+), 'Keep all eight rows of table 5-2'", src).group(1))
        cols = int(re.search(r"assertEqual\(len\(cells\), (\d+), 'Keep all eight columns of table 5-2'", src).group(1))
        self.assertIn(f"测试会核对全部 {rows * cols} 个显示值", TESTING.read_text(encoding="utf-8"))

    def test_08l_counts_derived_from_test_file(self):
        total = count_tests(ROOT / "tests" / "test_preference_optimization.py")
        ran, skipped, bad = no_torch_run("test_preference_optimization.py")
        self.assertEqual((ran, bad), (total, 0))
        self.assertIn(f"在上面的基线环境实测，{total} 个测试通过；不加载第三方包时，{total - skipped} 个标准库测试通过、"
                      f"{skipped} 个 PyTorch 测试明确跳过。", TESTING.read_text(encoding="utf-8"))

    def test_08n_counts_derived_from_test_file(self):
        doc = TESTING.read_text(encoding="utf-8")
        path = ROOT / "tests" / "test_quantized_serving.py"
        total = count_tests(path)
        ran, skipped, bad = no_torch_run("test_quantized_serving.py")
        self.assertEqual((ran, bad), (total, 0))
        self.assertIn(f"默认 {total} 个测试中，{total - skipped} 个运行、{skipped} 个真实引擎测试跳过", doc)
        # 七种 + 八项源码变异由 load_mutant 执行；另加一项 GGUF 对齐 fixture 回归。
        self.assertEqual(count_mutants(path, "load_mutant"), 7 + 8)
        self.assertIn("p24_alignment_mutant", path.read_text(encoding="utf-8"))
        self.assertIn("它还真正改写并执行七种错误实现", doc)
        self.assertIn("另外八项实际变异", doc)
        self.assertIn("合计十六项", doc)

    def test_08o_counts_derived_from_test_file(self):
        doc = TESTING.read_text(encoding="utf-8")
        path = ROOT / "tests" / "test_qwen3_case.py"
        total = count_tests(path)
        # 仓库基线环境没有 Transformers 4.51.3：被 HF_VERSION 门控的类中的测试会跳过。
        official = sum(
            1
            for cls in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(cls, ast.ClassDef) and any("HF_VERSION" in ast.unparse(d) for d in cls.decorator_list)
            for f in cls.body if isinstance(f, ast.FunctionDef) and f.name.startswith("test_")
        )
        ran, skipped, bad = no_torch_run("test_qwen3_case.py", ("torch", "jinja2"))
        self.assertEqual((ran, bad), (total, 0))
        self.assertIn(f"本次共发现 {total} 项测试：仓库基线环境 {total - official} 项通过、{official} 项官方参考测试明确跳过；"
                      f"无 PyTorch/Jinja2 的标准库环境 {total - skipped} 项通过、{skipped} 项明确跳过", doc)
        self.assertIn(f"还实际执行 **{count_mutants(path)} 个偏离本练习契约的变体**", doc)

    def test_09_counts_derived_from_test_file(self):
        doc = TESTING.read_text(encoding="utf-8")
        path = ROOT / "tests" / "test_deepseek_case.py"
        total = count_tests(path)
        ran, skipped, bad = no_torch_run("test_deepseek_case.py")
        self.assertEqual((ran, skipped, bad), (total, 0, 0))
        self.assertIn(f"| 专项测试 | {total} 项通过，无依赖跳过 |", doc)
        # 六个数值变体 + 两项契约变异。
        self.assertEqual(count_mutants(path), 6 + 2)
        self.assertIn("专项测试包含六个**实际执行的数值错误变体**", doc)
        self.assertIn("另执行 SWA 来源与 Markdown 竖线转义两项契约变异", doc)

    def test_08o_p25_11_answers_heading_without_ledger(self):
        body = (CHAPTERS / "08o_qwen3_case.md").read_text(encoding="utf-8")
        sec = body[body.index("## P25.11 为什么这次不是让两个错误实现互相作证"):]
        sec = sec[:sec.index("\n## ", 1)] if "\n## " in sec[1:] else sec
        self.assertIn("本章的数值结论不靠两份实现互相比对：结构与数值以独立的标量参考为准，另可与官方实现对照，"
                      "避免两份实现共享同一个错误而一起通过；分层做法与变体清单见 `docs/testing.md`。", sec)
        self.assertIn("本章的数值对照只验证小型结构和数值路径，不表示已经加载或运行上述真实 checkpoint。", sec)
        self.assertNotIn("这个交叉检查", body)

    def test_not_rerun_claims_are_dated_and_comment_is_fenced(self):
        doc = TESTING.read_text(encoding="utf-8")
        self.assertIn("（上句“本次提供 b5046 后，36 个全部实际通过”为 2026-10-20 记录于 0749bef，B6 未复跑。）", doc)
        self.assertIn("（上段“独立官方环境 28 项全部通过”为 2026-10-20 记录于 0749bef，B6 未复跑。）", doc)
        self.assertIn("```bash\n# 分开报告标准库、模板与数值测试；跳过不是通过。\n```", doc)
        self.assertNotIn("\n\n# 分开报告", doc)

    def test_ledger_markers_gone_from_edited_chapters(self):
        for name in MOVED:
            body = (CHAPTERS / name).read_text(encoding="utf-8")
            for marker in ("跳过不是通过", "被跳过不算通过", "skip 不是通过", "跳过不等于 HF 通过", "skipped 不算",
                           "个测试通过", "故意制造的错误", "错误变体", "种语义错误", "测试覆盖", "负例还故意",
                           "测试故意", "测试确实执行", "逐轮确认",
                           "跳过不是实验通过", "跳过不等于已运行", "测试分成四层", "测试同时覆盖", "测试不只检查"):
                with self.subTest(chapter=name, marker=marker):
                    kept = KEPT_MARKER_SENTENCES.get((name, marker))
                    if kept:
                        # 非测试账本的正文句子（评测方法或本章方法论小结），保留原文。
                        self.assertEqual(body.count(marker), 1)
                        self.assertIn(kept, body)
                    else:
                        self.assertNotIn(marker, body)


if __name__ == "__main__":
    unittest.main()
