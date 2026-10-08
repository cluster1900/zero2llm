"""
01_vector_similarity.py - 词向量与空间相似度计算（纯 Python + 向量计算）
用到的知识：向量坐标、Python 列表与 for 循环；计算步骤见下方注释

核心概念：
1. 词向量：把文字变成特征空间里的一个坐标点（箭头）
2. 向量点积（内积）：a · b = x1*x2 + y1*y2
3. 向量模长（长度）：|a| = sqrt(x1^2 + y1^2)
4. 余弦相似度：cos(θ) = (a · b) / (|a| * |b|)
"""

import math

def dot_product(vec_a, vec_b):
    """
    计算两个向量的点积（内积）
    点积的坐标公式：a · b = x1*x2 + y1*y2 + ... + xn*yn
    """
    assert len(vec_a) == len(vec_b), "两个向量的维度必须完全相同！"
    total = 0.0
    for i in range(len(vec_a)):
        total += vec_a[i] * vec_b[i]
    return total

def vector_magnitude(vec):
    """
    计算向量的模长（也就是箭头的长度）
    模长公式：|a| = √(x1² + x2² + ... + xn²)
    """
    sum_of_squares = 0.0
    for val in vec:
        sum_of_squares += val ** 2
    return math.sqrt(sum_of_squares)

def cosine_similarity(vec_a, vec_b):
    """
    计算余弦相似度 cos(θ)
    夹角余弦公式：cos(θ) = (a · b) / (|a| * |b|)
    取值范围：[-1, 1]
    - 接近 1 ：方向极其一致，含义非常相似
    - 接近 0 ：两个向量互相垂直（90度），毫无关联
    - 接近 -1：方向完全相反，反义词
    """
    dot = dot_product(vec_a, vec_b)
    mag_a = vector_magnitude(vec_a)
    mag_b = vector_magnitude(vec_b)
    
    if mag_a == 0 or mag_b == 0:
        return 0.0
    
    return dot / (mag_a * mag_b)

def euclidean_distance(vec_a, vec_b):
    """
    计算欧氏距离（平面两点间距离公式的扩展）
    距离公式：d = √((x1-x2)² + (y1-y2)² + ...)
    """
    sum_sq = 0.0
    for i in range(len(vec_a)):
        sum_sq += (vec_a[i] - vec_b[i]) ** 2
    return math.sqrt(sum_sq)

def main():
    print("=" * 60)
    print("【实验 1】：水果与特征坐标系（词向量的直观感受）")
    print("=" * 60)
    # 假设我们用一个 3 维坐标系来描述水果：
    # 维度 0：甜度（0 ~ 10）
    # 维度 1：脆度（0 ~ 10）
    # 维度 2：价格（元/斤）
    fruit_vocab = {
        "苹果 (Apple)":   [8.0, 9.0, 5.0],
        "香蕉 (Banana)":  [8.5, 1.0, 3.5],
        "梨子 (Pear)":    [7.8, 8.5, 4.8],
        "黄瓜 (Cucumber)":[1.0, 8.0, 2.0],
        "西瓜 (Watermelon)":[9.0, 3.0, 1.5]
    }

    target = "苹果 (Apple)"
    target_vec = fruit_vocab[target]
    print(f"基准词：{target}，坐标向量为：{target_vec}\n")

    print(f"{'目标词语':<18} | {'余弦相似度 (越高越近)':<20} | {'欧氏距离 (越小越近)':<15}")
    print("-" * 60)

    rows = []
    for word, vec in fruit_vocab.items():
        if word == target:
            continue
        sim = cosine_similarity(target_vec, vec)
        dist = euclidean_distance(target_vec, vec)
        rows.append((word, sim, dist))
        print(f"{word:<18} | {sim:^20.4f} | {dist:^15.4f}")

    rows.sort(key=lambda item: item[1], reverse=True)
    print("\n结论分析（直接按上面算出来的余弦相似度排序，避免口算说反）：")
    print(f"1. 和苹果方向最接近的是 {rows[0][0]}，余弦相似度 {rows[0][1]:.4f}。甜度和脆度都像。")
    print(f"2. 和苹果方向差得最远的是 {rows[-1][0]}，余弦相似度 {rows[-1][1]:.4f}。")
    print("   香蕉甜但很软，箭头落到了右下角；黄瓜很脆但几乎不甜，也不和苹果重合。")

    print("\n" + "=" * 60)
    print("【实验 2】：著名的词向量加减法（向量几何平移）")
    print("=" * 60)
    print("大模型世界里的名场面：vec('国王') - vec('男人') + vec('女人') ≈ vec('女王')")
    
    # 假设 3 个维度的含义是：[权力地位, 男性特征, 年龄成熟度]
    king   = [9.0,  8.0, 7.0]
    man    = [1.0,  8.0, 5.0]
    woman  = [1.0, -8.0, 5.0]
    queen  = [9.0, -8.0, 7.0]

    # 计算：国王 - 男人 + 女人
    result_vec = []
    for i in range(len(king)):
        val = king[i] - man[i] + woman[i]
        result_vec.append(val)

    print(f"国王向量 (King)  : {king}")
    print(f"男人向量 (Man)   : {man}")
    print(f"女人向量 (Woman) : {woman}")
    print(f"运算结果向量     : {result_vec}")
    print(f"女王向量 (Queen) : {queen}")
    
    sim_to_queen = cosine_similarity(result_vec, queen)
    print(f"\n运算结果与女王向量的余弦相似度为: {sim_to_queen:.4f}")
    print("这组数字是为了演示配好的：女王的坐标正好写成了“国王 - 男人 + 女人”。")
    print("所以相似度精确等于 1，不是从真实词典里查出来的巧合。")
    print("2013 年的 Word2Vec 在大量文本里学到的向量，只能近似满足这个平移，不会每个小数都对上。")

if __name__ == "__main__":
    main()
