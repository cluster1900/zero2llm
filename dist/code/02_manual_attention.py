"""
02_manual_attention.py - 纯 Python 纯手工算一遍自注意力机制（Self-Attention）
适合人群：仅懂高中数学（向量点积、e 的指数、百分比求和为 1）、懂基本 Python 循环

我们将完全不用任何深度学习黑盒库，只用普通的 Python 列表和 math 库，
把一个 3 词句子（“我”、“爱”、“学”）在注意力机制内部经历的每一步数字变化打印出来！
"""

import math

def dot_product(vec_a, vec_b):
    """高中平面/空间向量点积：两个向量对应元素相乘再相加"""
    return sum(a * b for a, b in zip(vec_a, vec_b))

def matmul_vec(matrix, vec):
    """矩阵乘以向量：把向量投影到新空间（矩阵的每一行与向量做点积）"""
    return [dot_product(row, vec) for row in matrix]

def softmax(scores):
    """
    高中概率思维版 Softmax 归一化：
    1. 取 e^x：保证所有分数都变成正数（e 的任意实数次方恒大于 0）
    2. 计算总和：作为分母
    3. 每个数字除以总和：变成百分比（所有人加起来等于 100%）
    """
    # 为了防止数值溢出，减去最大值（这是高中的技巧：上下同除以 e^max 不改变分数值）
    max_val = max(scores)
    exp_scores = [math.exp(s - max_val) for s in scores]
    sum_exp = sum(exp_scores)
    return [e / sum_exp for e in exp_scores]

def vector_add(vec_a, vec_b):
    """向量加法"""
    return [a + b for a, b in zip(vec_a, vec_b)]

def vector_scale(vec, scalar):
    """数乘向量：向量乘以一个系数（加权）"""
    return [x * scalar for x in vec]

def main():
    print("=" * 65)
    print("【第一步：定义输入的 3 个汉字和它们的身高体重（词向量）】")
    print("=" * 65)
    # 假设输入句子是：“我 爱 学”
    # 每个汉字用一个 2 维向量表示（维度 d_model = 2）
    words = ["我", "爱", "学"]
    x = [
        [1.0, 0.5],  # “我” 的向量
        [0.8, 1.2],  # “爱” 的向量
        [0.2, 1.5]   # “学” 的向量
    ]
    for w, vec in zip(words, x):
        print(f"字 '{w}' 的原始词向量: {vec}")

    print("\n" + "=" * 65)
    print("【第二步：角色变身！用权重矩阵生成 Q (查询), K (键), V (值)】")
    print("=" * 65)
    print("每个词都要化身 3 种角色：")
    print("  Q (Query) : 我想探寻谁？")
    print("  K (Key)   : 别人来找我时，我的特征名片是什么？")
    print("  V (Value) : 如果别人认可我，我能贡献给他的核心思想是什么？\n")

    # 下面三个 2×2 矩阵是事先写好的例子，不是训练出来的。
    # 代码把词向量竖着放。新向量的第 i 个数 = 矩阵第 i 行 · 词向量。
    # 以“我”=[1.0, 0.5] 和 W_Q 为例，可以在草稿纸上核对：
    #   Q0 = 0.5*1.0 + 0.2*0.5 = 0.6
    #   Q1 = 0.1*1.0 + 0.8*0.5 = 0.5
    # 论文常写成横着的一行 Q = X W_Q。那是同一次乘法的另一种写法。
    W_Q = [
        [0.5, 0.2],
        [0.1, 0.8]
    ]
    W_K = [
        [0.4, 0.3],
        [0.2, 0.7]
    ]
    W_V = [
        [1.0, 0.1],
        [0.0, 0.9]
    ]

    # 计算每一个词的 Q, K, V
    Q = [matmul_vec(W_Q, vec) for vec in x]
    K = [matmul_vec(W_K, vec) for vec in x]
    V = [matmul_vec(W_V, vec) for vec in x]

    d_k = len(K[0])  # Key 的维度，这里是 2
    scale_factor = math.sqrt(d_k)

    for i, w in enumerate(words):
        print(f"词 '{w}': Q={['%.3f' % v for v in Q[i]]}, "
              f"K={['%.3f' % v for v in K[i]]}, "
              f"V={['%.3f' % v for v in V[i]]}")

    print("\n" + "=" * 65)
    print("【第三步：计算注意力打分矩阵（点积并除以 √d_k）】")
    print("=" * 65)
    print(f"Key 的维度 d_k = {d_k}，缩放因子 √d_k = √{d_k} ≈ {scale_factor:.4f}")
    print("公式：Score(i, j) = (Q_i · K_j) / √d_k\n")

    raw_scores = []
    for i in range(len(words)):
        row_scores = []
        for j in range(len(words)):
            # 高中点积：Q 的第 i 个向量与 K 的第 j 个向量做点积
            dot = dot_product(Q[i], K[j])
            scaled = dot / scale_factor
            row_scores.append(scaled)
        raw_scores.append(row_scores)

    # 打印打分表
    header = f"{'':<6}" + "".join([f"{'目标:' + w:>12}" for w in words])
    print("打分矩阵 (未归一化):")
    print(header)
    for i, row in enumerate(raw_scores):
        row_str = "".join([f"{val:12.4f}" for val in row])
        print(f"查询:{words[i]} {row_str}")

    print("\n" + "=" * 65)
    print("【第四步：Softmax 归一化（把分数变成注意力百分比权重）】")
    print("=" * 65)
    print("每一行的注意力权重之和必须严格等于 1.0 (100%)\n")

    attention_weights = []
    for i, row in enumerate(raw_scores):
        weights = softmax(row)
        attention_weights.append(weights)

    print("注意力权重矩阵 (Attention Weights %):")
    print(header)
    for i, row in enumerate(attention_weights):
        row_str = "".join([f"{val * 100:11.2f}%" for val in row])
        print(f"查询:{words[i]} {row_str}")

    print("\n" + "=" * 65)
    print("【第五步：加权平均！将各词的 Value 融合为全新的上下文向量】")
    print("=" * 65)
    print("公式：新词向量 Output_i = Σ (Weight_{i, j} * V_j)\n")

    outputs = []
    for i in range(len(words)):
        # 初始为 0 向量
        new_vec = [0.0] * len(V[0])
        print(f"--- 正在计算新词向量: [{words[i]}] ---")
        for j in range(len(words)):
            w = attention_weights[i][j]
            v = V[j]
            contribution = vector_scale(v, w)
            new_vec = vector_add(new_vec, contribution)
            print(f"  来自 '{words[j]}' 的贡献 (占比 {w*100:5.2f}%): {[round(c, 4) for c in contribution]}")
        outputs.append(new_vec)
        print(f"  => 融合上下文后的新 '{words[i]}' 向量: {[round(o, 4) for o in new_vec]}\n")

    print("=" * 65)
    print("【最终结论与思考】")
    print("=" * 65)
    print("看看输入和输出的对比：")
    for i, w in enumerate(words):
        print(f"词 '{w}': 原始向量={x[i]}  ==>  注意力融合后={['%.4f' % o for o in outputs[i]]}")
    print("\n每个新向量都按百分比混入了三个字的 Value，所以数字变了。")
    print("这一步只是把“混合手续”走通。矩阵是我们手写的，模型还没有经过训练，")
    print("不要把它理解成电脑已经读懂了“我爱学”这句话。")

if __name__ == "__main__":
    main()
