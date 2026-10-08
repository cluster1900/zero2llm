"""
03_scaled_dot_product.py - 缩放点积注意力与因果掩码（数学原理剖析与 PyTorch 对比）
适合人群：掌握基础统计学（方差、标准差）、了解基本矩阵概念的初学者

包含两大核心实验：
1. 为什么必须除以 √d_k？（方差爆炸与 Softmax 饱和实验）
2. 什么是因果掩码（Causal Mask）？（为什么答第 1 题不能偷看第 2 题）
3. 纯 Python 与 PyTorch 官方实现结果对比
"""

import math
import random
import torch
import torch.nn.functional as F

def experiment_why_scale():
    print("=" * 65)
    print("【深度探秘 1】：为什么要除以 √d_k？（方差爆炸危机）")
    print("=" * 65)
    
    # 模拟真实大模型：假设向量维度 d_k = 64
    d_k = 64
    random.seed(42)

    # 随机生成两个符合标准正态分布（均值 0，方差 1）的 64 维向量
    # 当 X、Y 独立、均值为 0 且方差为 1 时，Var(X*Y) = Var(X)*Var(Y) = 1
    # 64 个这样的数相加，总方差就会累加成 64！标准差就会变成 √64 = 8！
    vec_q = [random.gauss(0, 1) for _ in range(d_k)]
    vec_k1 = [random.gauss(0, 1) for _ in range(d_k)]
    vec_k2 = [random.gauss(0, 1) for _ in range(d_k)]
    vec_k3 = [random.gauss(0, 1) for _ in range(d_k)]

    # 计算未缩放的点积分数
    raw_score_1 = sum(q * k for q, k in zip(vec_q, vec_k1))
    raw_score_2 = sum(q * k for q, k in zip(vec_q, vec_k2))
    raw_score_3 = sum(q * k for q, k in zip(vec_q, vec_k3))

    raw_scores = [raw_score_1, raw_score_2, raw_score_3]
    print(f"未除以 √{d_k} 之前的原始点积分数: {[round(s, 2) for s in raw_scores]}")

    # 计算 Softmax（未缩放）
    def naive_softmax(scores):
        m = max(scores)
        exps = [math.exp(s - m) for s in scores]
        s = sum(exps)
        return [e / s for e in exps]

    weights_unscaled = naive_softmax(raw_scores)
    print("未缩放的注意力权重 (Softmax 结果):")
    for idx, w in enumerate(weights_unscaled):
        print(f"  选项 {idx+1}: {w * 100:6.2f}%")

    top = max(weights_unscaled) * 100
    print(f"\n[现象警示]：最大的一项占到了 {top:.2f}%。")
    print("原始分数只差几点到十几点，可是 e^x 会把差距急剧放大，注意力几乎全部堆到一个词上。")
    print("训练时，这种“一家独大”的区域很平，参数稍微改一点，输出几乎不动，模型就学得很吃力。")

    # 现在除以 √d_k 进行缩放！
    scale = math.sqrt(d_k)
    scaled_scores = [s / scale for s in raw_scores]
    weights_scaled = naive_softmax(scaled_scores)

    print(f"\n除以 √{d_k} = {scale:.1f} 之后的缩放分数: {[round(s, 2) for s in scaled_scores]}")
    print("缩放后的注意力权重 (平滑健康的 Softmax):")
    for idx, w in enumerate(weights_scaled):
        print(f"  选项 {idx+1}: {w * 100:6.2f}%")
    print("[神奇化解]：除以 √d_k 把方差重新拉回 1，让注意力权重平滑健康，所有词都能得到合理的关注！")

def experiment_causal_mask():
    print("\n" + "=" * 65)
    print("【深度探秘 2】：什么是因果掩码（Causal Mask / 下三角掩码）？")
    print("=" * 65)
    print("生活比喻：闭卷做题。你正在写第 2 个字，绝对不能偷看第 3 个字！")
    print("操作秘籍：在 Softmax 之前，把未来所有不能看的位置填上 -∞ (负无穷大)！")
    print("根据指数函数的性质：e^(-∞) = 0，做完 Softmax 后权重精准变成 0%！\n")

    seq_len = 3
    # 模拟 3 个词之间的注意力得分矩阵（3x3）
    scores = torch.tensor([
        [2.0, 1.5, 0.8],  # 词 1 的 Query 和 词 1, 2, 3 的点积
        [1.2, 3.0, 1.1],  # 词 2 的 Query 和 词 1, 2, 3 的点积
        [0.5, 1.8, 2.5]   # 词 3 的 Query 和 词 1, 2, 3 的点积
    ])

    # 制作下三角掩码 (对角线及以下为 True/1，右上角未来位置为 False/0)
    # torch.tril 表示 lower triangular (下三角)
    mask = torch.tril(torch.ones(seq_len, seq_len))
    print("掩码矩阵 (1 表示允许看，0 表示未来禁止看):")
    print(mask.numpy())

    # 把 0 的位置替换为 -1e9 (-10亿，相当于负无穷)
    masked_scores = scores.masked_fill(mask == 0, -1e9)
    print("\n加上掩码后的打分矩阵 (未来的位置被极小负数封印):")
    for row in masked_scores.numpy():
        print(["%.1f" % v if v > -100 else "-inf" for v in row])

    # 经过 Softmax
    causal_weights = F.softmax(masked_scores, dim=-1)
    print("\n最终因果注意力权重 (未来位置的权重被精准扼杀为 0%):")
    words = ["词1(我)", "词2(爱)", "词3(学)"]
    for i in range(seq_len):
        row_str = " | ".join([f"{causal_weights[i, j].item()*100:6.2f}%" for j in range(seq_len)])
        print(f"{words[i]:<8} ==> [ {row_str} ]")

def compare_with_pytorch():
    print("\n" + "=" * 65)
    print("【对照实验 3】：PyTorch 官方公式一键运行验证")
    print("=" * 65)
    
    # 模拟 Q, K, V 张量：Batch=1, Seq_Len=3, Dim=4
    torch.manual_seed(42)
    Q = torch.randn(1, 3, 4)
    K = torch.randn(1, 3, 4)
    V = torch.randn(1, 3, 4)

    d_k = Q.size(-1)

    # 按照公式一步步算：Softmax(Q · K^T / √d_k) · V
    # 1. 矩阵乘法：Q 乘 K 的转置
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, V)

    print("手写 PyTorch 公式计算结果 (形状:", output.shape, "):")
    print(output[0])

    # 官方自带的 scaled_dot_product_attention (PyTorch 2.0+ 高性能内核)
    official_output = F.scaled_dot_product_attention(Q, K, V)
    print("\nPyTorch 官方原生内核计算结果:")
    print(official_output[0])

    diff = (output - official_output).abs().max().item()
    print(f"\n两者最大误差: {diff:.8e} (完全一致！)")

if __name__ == "__main__":
    experiment_why_scale()
    experiment_causal_mask()
    compare_with_pytorch()
