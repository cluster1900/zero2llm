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
 "07a_training_loop.md": [
  "测试另用微小扰动的有限差分验证梯度，不只检查输出里是否有这串数字。",
  "测试用一个 `Dropout(p=1)` 小对照说明模式的真实作用，而不虚构 BabyGPT 的随机层。",
  "测试检查这个重组合等于总体 loss（允许 float32 归约误差）。"
 ],
 "08_baby_transformer.md": [
  "S0 测试会改变后缀字符，检查前缀**所有 logits** 不变，而不只看最后选出来的字碰巧一样。"
 ],
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
  "测试还与 PyTorch 官方 LayerNorm 对照。",
  "测试只在安装了兼容的 2.14.x 正式版时执行张量实验，缺失或版本不符会明确跳过，不能把跳过算成数值验收通过。",
  "测试也会逐字核对脚本生成的表格是否出现在正文中，防止只改代码或只改数字。"
 ],
 "08a_kv_cache.md": [
  "这张表由脚本手算函数与测试核对，不用生成句子的表面效果来掩盖错误。",
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
  "测试另用独立标量 softmax 与加权求和核对基线，而不是调用同一个 attention 函数两次宣布正确。",
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

# B6 review B2（Wu 2026-11-05 决定「全搬」）：其他组章节中同类段尾测试句，原文移入 docs/testing.md。
MOVED_B2 = {
 "08c_modern_block.md": [
  "为守住旧行为，测试还构建一个兼容设置：`norm_kind='layernorm'`、`ffn_kind='gelu'`、`norm_bias=True`、`ffn_bias=True`、ε=1e-5、H=4D。它在相同 seed 下与 S2 的每个 state tensor、每个输出 logit 精确相同。",
  "源码的计数函数包含这两种情况，测试再用真实 `numel()` 交叉检查。",
  "`gate_walkthrough()` 打印每一层，测试把最终结果独立写成 `-3/(1+e)` 和 `8/(1+exp(-2))` 核对。"
 ],
 "08d_grouped_kv_mla.md": [
  "测试还写了“每个 batch、每个 query 头、每个位置分别算”的慢参考，不与快速路径共享映射函数。否则两个实现一起写错，full/cache 对齐也可能照样通过。",
  "测试还检查底层存储大小，避免只在 shape 上做一个漂亮的切片，底下仍占着大张量。",
  "float64 改为每数 8 字节，缓存字节数翻倍；这也是测试的一部分。"
 ],
 "08h_moe.md": [
  "测试还用独立标量输出/梯度参考、78x 手算和故意注入的错误验证判据。"
 ],
 "08i_pretraining_pipeline.md": [
  "测试还做了更直观的核对：关闭 dropout 后，把 A、B 一起 packing 得到的真实位置 logits，应与分别单独处理的结果接近；只改 A 的字，B 的输出应完全不变。",
  "标准库的 `scalar_loss` 用稳定 log-sum-exp 独立计算，tensor 测试再与 `torch.nn.functional.cross_entropy(..., reduction=\"sum\")` 比较，并核对忽略位置梯度为零。"
 ],
 "08j_training_frameworks.md": [
  "测试还用 Python 标量循环、稳定的指数求和独立算一次，避免两个实现都复制同一个索引错误。",
  "若可选包缺失或版本不在本章验证组合中，HF 测试会明确显示 skipped。"
 ],
 "08k_sft_lora.md": [
  "测试还使用手写的完整标签序列对照，避免实现与测试都照抄同一个错误 span。",
  "代码测试会发现：预测被忽略 token 的那组 logits 没有直接 loss 梯度，但 user 字符的 embedding 仍可能通过注意力影响 assistant 预测，因此收到梯度。"
 ],
 "09a_minigpt_healthcheck.md": [
  "测试还把同样形状的模型改为 float64，逐个确认缓存张量每元素 8 字节：MHA 为 8192 字节，GQA 为 4096 字节。若把元素大小写死为 4，这个独立 dtype 对照会失败。",
  "测试再把 router 权重全部置零，强制所有 token 选择专家 0、1，负载应为 `[16,16,0,0]`。"
 ],
 "08e_attention_systems.md": [
  "测试另外使用标准库 Decimal 的 **65 位十进制有效数字精度独立参考**，逐 query 直接算点积、指数和归一化，不调用生产实现的输入辅助函数、Softmax 或 block 更新。手算的 6 又提供了更容易人工核验的锚点。"
 ],
 "08f_sparse_linear_attention.md": [
  "代码 `receptive_positions()` 逐层合并可达集合，测试再用上面的闭式表达核验。"
 ]
}

# Wu 2026-11-05 17:10/17:12：独立核对句与运行/跳过说明全部搬入 docs/testing.md。
MOVED_R4 = {
    '03_qkv_self_attention.md': [
        '测试同时验证“全序列带因果 mask”与“逐个前缀重算最后位置”一致，避免只看上三角权重为零就认为端到端一定正确。',
    ],
    '07_encoder_decoder.md': [
        '打印到两位的百分数可能有舍入误差，测试用未舍入值检查行和。',
    ],
    '08_baby_transformer.md': [
        '测试在临时目录保存后加载，对同一批输入比较所有 logits。',
    ],
    '08c_modern_block.md': [
        '`norm_walkthrough()` 产生表中的原始张量；测试独立使用两种平方根公式核对，不是拿同一个实现与自己比较。',
        '测试把同一串 token 分成 `[12]`、12 次单 token、`[3,5,4]` 和 `[11,1]`，拼接每块 logits，与同权重的完整前向逐项比较。B>1、float32 和 float64 都覆盖。',
    ],
    '08d_grouped_kv_mla.md': [
        '测试把 S3 的同一份 state_dict 装入这个边界模型，逐项检查参数、logits、K/V 完全一致。',
        '测试因此另写一个不调用演示函数内部步骤的标量循环参考：独立列出输入和权重，用 sin/cos 手算位置 0、1、2 的旋转，拼成 4 维 Q/K，直接点积除以 2，再对可见前缀做 Softmax 和 value 加权。两条实现各自对齐这个参考，并逐项核对旋转后位置键。',
    ],
    '08e_attention_systems.md': [
        '测试检查失败后 A/B 数据和所有权完全不变。',
        '测试特意覆盖：\n\n- 第一块全被遮住，后面才出现合法 key。\n- 后面的合法分数比前面大，需要重新缩放。\n- 整行全被遮住，最终返回零。',
    ],
    '08f_sparse_linear_attention.md': [
        '测试核对最终张量的实际存储，不把整个脚本说成“常数内存”。',
        '本章的测试逐 batch、逐 query、逐历史位置用 Python 标量和 `math.exp` 重新算权重，不调用主实现的 φ、mask、矩阵乘法或状态递推。两个路径分别对齐它，而不是只互相对齐。',
    ],
    '08i_pretraining_pipeline.md': [
        '测试先做两步真实更新，再喂全 PAD 行，验证整个状态不变。',
    ],
    '08j_training_frameworks.md': [
        '测试要求“陌生词”得到 [2,1,3]，也要求保存后加载的词表和固定句子 ID 不变。这比只检查“tokenizer.json 文件存在”更有意义。',
    ],
    '08k_sft_lora.md': [
        '这些数值由测试独立写死，用来检查 autograd，而不是仅比较两个使用同一段代码的模型。',
        '测试在缺少 PyTorch 时会明确跳过真实训练相关部分。',
        '测试用无 site-packages 的独立进程实际核对。',
        '测试分别检查手算矩阵和整个模型的 logits。',
    ],
    '08m_reasoning_mtp.md': [
        '测试既检查这个数值，也检查梯度，不只判断 loss 是否“看起来正常”。',
        '测试使用固定种子、小张量、float64 对照，同时比较 logits 与 K/V，而不是宣称任意硬件精度都能逐位相同。',
    ],
    '08l_preference_optimization.md': [
        '测试不仅比较 loss，还逐元素检查梯度：无效位置应为零，有效位置应等于“目标指示值减去该 token 的概率”。',
        '测试中 PyTorch 项会明确跳过，标准库项仍执行。',
    ],
    '09_modern_llms.md': [
        '测试除比较两路，还直接用 `exp(-2)、exp(-1)、1` 对手工展开后的值 `[4,6]、[-1,3]、[4,0]` 求加权平均，避免两条路径共享同一个错位后一起通过。',
        '测试枚举各组末位置作为独立 oracle。',
        '测试用“由输出 feature 编号反推输入坐标”的独立公式检查每个元素，而不是只检查 shape。',
    ],
    '07a_training_loop.md': [
        '测试仍运行标量部分，只跳过张量部分。',
    ],
    '08g_ced_architecture.md': [
        '测试对此做了“干预”，不只检查变量名：把第一层 decoder 的 local V 投影放大三倍，重新计算。',
    ],
    '08h_moe.md': [
        '测试把这些字面算式与 autograd 对齐，没有拿生产 dispatch 函数自己验证自己。',
    ],
    '05_positional_encoding.md': [
        '回归测试只有在**已安装且版本为 PyTorch 2.14.x** 时才运行张量实验。本批验证环境为 PyTorch 2.14.1+cpu，CPU；测试不需要 GPU。',
    ],
}

# 全书守卫：以“测试”为叙述主语的句子（测试会/还/另/再/把/用/检查/核对…）不得留在章节正文，
# 除非列入 NARRATION_ALLOWLIST。每项注明状态：decided-kept（非测试叙述）或 pending Wu decision（评审边界项，待 Wu 决定）。
NARRATION = re.compile(r"测试(?:会|还|另|再|把|用|检查|核对|核验|验证|同时|使用|对|逐|分别|先|要求|既|不仅|除|枚举|独立|因此|也|将|确认|写|只|中|里|做|让|构建|封锁|刻意|故意|特意|固定|仍|在|进一步|直接|放大|改变|容差|因缺)|由[^。；]{0,30}测试|与测试核对|的测试逐")
NARRATION_ALLOWLIST = [
    ("08a_kv_cache.md", "float32 的运算顺序可能因为矩阵大小不同而有微小舍入差；测试使用明确的 `atol/rtol`，不是要求逐位相同，也不是放宽到“都能生成一句诗”。",
     "decided-kept", "Wu 2026-11-05 17:12 keep: tolerance statement (restored after review r5)"),
    ("07a_training_loop.md", "测试对表值使用绝对容差 1e-6，容纳舍入和这一级 CPU 数值差异，而不容许学习率改变造成的偏移。",
     "decided-kept", "Wu 2026-11-05 17:12 keep: tolerance statement (independent-check class)"),
    ("08b_rope_yarn.md", "浮点 cos(π/2) 未必严格等于 0，因此测试使用小容差，不要求每一位相等。",
     "decided-kept", "Wu 2026-11-05 17:12 keep: tolerance statement (independent-check class)"),
    ("08b_rope_yarn.md", "- **直接外推**：仍用 pθ，不改变转速；测试会出现训练未覆盖的相对距离。",
     "decided-kept", "测试 means test/inference time, not the repository tests"),
    ("08b_rope_yarn.md", "不同硬件/数值库末位可能有差异，测试对记录值使用明确的 CPU 容差。",
     "decided-kept", "Wu 2026-11-05 17:12 keep: tolerance statement (independent-check class)"),
    ("08b_rope_yarn.md", "纯旋转的范数测试还能原样用吗？",
     "decided-kept", "exercise question"),
    ("08c_modern_block.md", "半精度测试只验证这个局部数值策略，整模型消融仍在 CPU float32 下运行。",
     "decided-kept", "Wu 2026-11-05 17:12 keep: scope statement"),
    ("08c_modern_block.md", "测试容差为 atol=3e-6、rtol=1e-5。",
     "decided-kept", "Wu 2026-11-05 17:12 keep: tolerance statement (independent-check class)"),
    ("08g_ced_architecture.md", "如果完整路径和缓存路径共用一个错误 mask，两条路径可能同时看未来，对齐测试仍然会通过。",
     "decided-kept", "conceptual argument (a shared wrong mask makes alignment pass), not a report of what the tests do"),
    ("08g_ced_architecture.md", "测试放大第一层 decoder 的 local V 后，global bank 也确实不变。",
     "decided-kept", "Wu 2026-11-05 17:12 keep: self-test answer that names a test (08h L342 of this class was moved)"),
    ("08h_moe.md", "浮点末位可能随环境变化，判定依据是测试容差，而非逐字复制日志。",
     "decided-kept", "Wu 2026-11-05 17:12 keep: tolerance statement (independent-check class)"),
    ("08h_moe.md", "该测试同时检查部分丢弃与 capacity=0。",
     "decided-kept", "Wu 2026-11-05 17:12 keep: self-test answer that names a test (08h L342 of this class was moved)"),
    ("08h_moe.md", "见手算路由测试中五 token 的 [5,5,0,0] 断言。",
     "decided-kept", "Wu 2026-11-05 17:12 keep: self-test answer that names a test (08h L342 of this class was moved)"),
    ("08k_sft_lora.md", "对应 `test_truncation_exact_fit_and_all_boundaries`，测试逐一遍历边界，不只抽查 14。",
     "decided-kept", "Wu 2026-11-05 17:12 keep: self-test answer that names a test (08h L342 of this class was moved)"),
    ("08k_sft_lora.md", "另一个测试验证 user 内容虽然不被当成标签，仍可通过上下文影响 loss。",
     "decided-kept", "Wu 2026-11-05 17:12 keep: self-test answer that names a test (08h L342 of this class was moved)"),
    ("09_modern_llms.md", "测试直接把缺字段字典送给 extractor 时才得到 unknown，保护解析语义。",
     "decided-kept", "Wu 2026-11-05 17:12 keep: self-test answer that names a test (08h L342 of this class was moved)"),
    ("09a_minigpt_healthcheck.md", "下面是 [`code/28_minigpt_healthcheck.py`](../code/28_minigpt_healthcheck.py) 中 `cache_equivalence` 的前 14 行（第 120–133 行），原样摘录，由 `scripts/sync_code_excerpts.py` 从源码生成并由测试锁定：",
     "decided-kept", "Wu 2026-11-05 17:12 keep: 09a healthcheck: tests are the teaching subject (review boundary item)"),
    ("09a_minigpt_healthcheck.md", "这里检查的是**干预**，而非注意力图的形状：如果程序画出一个正确的 mask，却没有真的把它用进计算，干预测试仍会失败。",
     "decided-kept", "Wu 2026-11-05 17:12 keep: 09a healthcheck: tests are the teaching subject (review boundary item)"),
    ("09a_minigpt_healthcheck.md", "测试只让最后一层的位置 5 偷看位置 6。",
     "decided-kept", "Wu 2026-11-05 17:10 keep: 09a healthcheck: tests are the teaching subject (review boundary item)"),
    ("09a_minigpt_healthcheck.md", "数值测试进一步遍历每一个切点，并用 float64 检查。",
     "decided-kept", "Wu 2026-11-05 17:10 keep: 09a healthcheck: tests are the teaching subject (review boundary item)"),
    ("09a_minigpt_healthcheck.md", "测试改变 `[15,15]`，本次 logits 仍完全相同，但 checkpoint 的状态已经不忠实，往返检查必须报警。",
     "decided-kept", "Wu 2026-11-05 17:10 keep: 09a healthcheck: tests are the teaching subject (review boundary item)"),
    ("09a_minigpt_healthcheck.md", "测试使用没有自定义还原方法、没有执行副作用的普通字典子类，确认限制型反序列化会拒绝未允许的对象，而不是用危险载荷演示风险。",
     "decided-kept", "Wu 2026-11-05 17:10 keep: 09a healthcheck: tests are the teaching subject (review boundary item)"),
    ("09a_minigpt_healthcheck.md", "S2 的 RoPE 也必须使用连续位置，且旧 K 不应被再次旋转，相关数学与缓存契约仍由 P12 的原测试覆盖。",
     "decided-kept", "Wu 2026-11-05 17:12 keep: 09a healthcheck: tests are the teaching subject (review boundary item)"),
    ("09a_minigpt_healthcheck.md", "7. 测试因缺 Torch 跳过、负例因导入失败退出、负例输出有限但偏离 oracle，哪一种才是本章想看到的负例证据？",
     "decided-kept", "exercise question"),
    ("10_appendix.md", "公式手算、自测与固定配置证据由仓库测试核对；本页不宣布任何未运行硬件的性能。",
     "decided-kept", "Wu 2026-11-05 17:10 keep: overview scope statement (review boundary item)"),
]


def narration_sentences(text):
    code = False
    for line in text.split("\n"):
        if line.startswith("```"):
            code = not code
            continue
        if code or "unittest discover" in line:
            continue
        for sent in re.findall(r"[^。！？\n]+[。！？]?", line):
            if NARRATION.search(sent):
                yield sent.strip()

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

    def test_b2_moved_from_other_chapters(self):
        doc = TESTING.read_text(encoding="utf-8")
        for name, paras in MOVED_B2.items():
            body = (CHAPTERS / name).read_text(encoding="utf-8")
            self.assertEqual(body.count(RUN), 1)
            for p in paras:
                with self.subTest(chapter=name, para=p[:30]):
                    self.assertEqual(doc.count(p), 1)
                    self.assertNotIn(p, body)

    def test_r5_restored_kept_text(self):
        """Review r5: tolerance/scope notes and teaching content stay in the chapters."""
        kept = {
            "03_qkv_self_attention.md": ["这里没有 dropout，模型参数固定，以浮点容差比较。"],
            "08a_kv_cache.md": ["测试使用明确的 `atol/rtol`，不是要求逐位相同"],
            "08k_sft_lora.md": ["float32 整模因乘法顺序不同，按 `atol=2e-6、rtol=2e-5` 检查。这里验证的是数值等价，不是实测推理加速。"],
            "08l_preference_optimization.md": ["修改被排除位置的 logits，不应改变结果。这样才能发现“维度正确但监督错位”的错误。"],
            "08g_ced_architecture.md": ["把第一层 decoder 的 local V 投影放大三倍", "如果 global bank 也被改变了，说明路径很可能接回了 decoder 自己。"],
            "08h_moe.md": ["在 78x 的手算例子里", "输入第一维梯度为 78+20log2，第二维为 78。"],
        }
        for name, items in kept.items():
            text = (ROOT / "chapters" / name).read_text(encoding="utf-8")
            for item in items:
                self.assertEqual(text.count(item), 1, (name, item))

    def test_r4_moved_per_wu_rulings(self):
        doc = TESTING.read_text(encoding="utf-8")
        for name, paras in MOVED_R4.items():
            body = (CHAPTERS / name).read_text(encoding="utf-8")
            with self.subTest(chapter=name):
                self.assertEqual(body.count(RUN) + body.count(RUN_ALL), 1)
            for p in paras:
                with self.subTest(chapter=name, para=p[:30]):
                    self.assertEqual(doc.count(p), 1)
                    self.assertNotIn(p, body)

    def test_book_has_no_unlisted_test_narration(self):
        allowed = {}
        for name, sent, status, reason in NARRATION_ALLOWLIST:
            self.assertEqual(status, "decided-kept")
            self.assertTrue(reason)
            allowed.setdefault(name, []).append(sent)
        for path in sorted(CHAPTERS.glob("*.md")):
            body = path.read_text(encoding="utf-8")
            found = list(narration_sentences(body))
            with self.subTest(chapter=path.name):
                self.assertEqual(sorted(found), sorted(allowed.get(path.name, [])))

    def test_b2_dangling_fixes(self):
        ch = lambda n: (CHAPTERS / n).read_text(encoding="utf-8")
        self.assertIn("9. 不能。见 `test_mutations_fail_independent_oracle_not_only_cache_agreement`。", ch("08h_moe.md"))
        self.assertIn("脚本用标准库生成这些数值。这个例子是直接优化", ch("07a_training_loop.md"))
        self.assertIn("它便于人工检查。", ch("08k_sft_lora.md"))
        self.assertIn("第二条会执行标准库测试。脚本默认", ch("08j_training_frameworks.md"))
        self.assertIn("实测每层 K/V。统计不包含", ch("08d_grouped_kv_mla.md"))
        self.assertIn("缓存字节数翻倍。", ch("08d_grouped_kv_mla.md"))
        self.assertIn("源码的计数函数包含这两种情况。参数数相同", ch("08c_modern_block.md"))
        self.assertIn("`gate_walkthrough()` 打印每一层。\n", ch("08c_modern_block.md"))
        self.assertIn("逐层合并可达集合。\n", ch("08f_sparse_linear_attention.md"))
        self.assertIn("用稳定 log-sum-exp 独立计算。\n", ch("08i_pretraining_pipeline.md"))
        self.assertIn("\nMoE 中未被选中的专家梯度可以是 `None`", ch("09a_minigpt_healthcheck.md"))
        self.assertIn("“不要求模型预测这些 token”。预测被忽略 token 的那组 logits 没有直接 loss 梯度，但 user 字符的 embedding "
                      "仍可能通过注意力影响 assistant 预测，因此收到梯度。", ch("08k_sft_lora.md"))
        self.assertIn("两条路径可能一起忘记 √d。所以还需要标准库 Decimal 的 **65 位十进制有效数字精度独立参考**：逐 query 直接算点积、指数和归一化，不经过生产实现的 Softmax 或 block 更新。\n", ch("08e_attention_systems.md"))
        doc = TESTING.read_text(encoding="utf-8")
        self.assertNotIn("\n这也是测试的一部分。", doc)

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
