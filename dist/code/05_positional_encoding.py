"""
05_positional_encoding.py - 正弦余弦位置编码（Positional Encoding）
适合人群：掌握基础三角函数（sin, cos、周期、波长）、向量内积的初学者

生活比喻：
石英钟的三根指针：
- 秒针（第 0 维度）：转得最快，滴答滴答（高频正弦波）
- 分针（中间维度）：转得适中（中频正弦波）
- 时针（高维度）：转得极慢（低频正弦波）
每一个时间点（位置 pos），三根指针的指针角度组合都是世界上独一无二的！
模型一查这个独特的“角度指纹”，就能立刻知道这个词在句子的第几号座位！
"""

import math
import torch
import torch.nn as nn

def generate_positional_encoding(max_seq_len, d_model):
    """
    生成原版 Transformer 的正弦余弦位置编码。
    公式里的 i 是“第几对维度”，从 0 数起，不是已经跳着走的列号：
      PE(pos, 2i)   = sin( pos / (10000 ^ (2i / d_model)) )
      PE(pos, 2i+1) = cos( pos / (10000 ^ (2i / d_model)) )
    例如 d_model = 8 时：
      i = 0 → 第 0、1 列，分母是 10000^0 = 1，转得最快
      i = 1 → 第 2、3 列，分母是 10000^(2/8) = 10
      i = 2 → 第 4、5 列，分母是 10000^(4/8) = 100
      i = 3 → 第 6、7 列，分母是 10000^(6/8) = 1000，转得最慢
    """
    pe = torch.zeros(max_seq_len, d_model)

    for pos in range(max_seq_len):
        # k 就是公式中的 i。这里必须用“第几对”，不能再乘一次 2。
        for k in range(d_model // 2):
            div_term = math.pow(10000.0, (2 * k) / d_model)
            pe[pos, 2 * k] = math.sin(pos / div_term)
            if 2 * k + 1 < d_model:
                pe[pos, 2 * k + 1] = math.cos(pos / div_term)

    return pe

def ascii_visualize(pe_matrix):
    """
    用简单的字符在控制台画出位置编码的波形强弱
    数值在 [-1, 1] 之间，我们映射为不同的字符
    """
    chars = " .:-=+*#%@"
    print("\n【位置编码字符热力图 (纵轴: 单词位置 pos 0~7, 横轴: 维度 dim 0~7)】")
    print("注意看：左侧低维度变化剧烈（高频），右侧高维度变化极其缓慢（低频平缓）！")
    print("-" * 55)
    
    rows, cols = pe_matrix.shape
    # 打印表头
    print("Pos \\ Dim | " + " ".join([f"d{c}" for c in range(cols)]))
    print("-" * 55)
    for r in range(rows):
        line = f"Pos {r:2d}    | "
        for c in range(cols):
            val = pe_matrix[r, c].item()
            # 归一化到 0 ~ 9 的字符索引
            norm_idx = int((val + 1.0) / 2.0 * 9.99)
            norm_idx = max(0, min(9, norm_idx))
            line += f" {chars[norm_idx]} "
        print(line)
    print("-" * 55)

def main():
    print("=" * 65)
    print("【正弦余弦位置编码：给词语发座位号】")
    print("=" * 65)

    seq_len = 8   # 假设句子有 8 个字（位置 0 到 7）
    d_model = 8   # 每个位置编码也是 8 维向量

    pe = generate_positional_encoding(seq_len, d_model)

    print("具体数值矩阵 (保留两位小数):")
    for pos in range(seq_len):
        vals = [f"{pe[pos, d].item():6.2f}" for d in range(d_model)]
        print(f"位置 pos={pos}: [{', '.join(vals)}]")

    # 字符画可视化
    ascii_visualize(pe)

    print("\n" + "=" * 65)
    print("【位置指纹会周期性回头，不是一条笔直下降的斜线】")
    print("=" * 65)
    print("拿【位置 0】去跟其他位置算余弦相似度：\n")

    base_pos = pe[0]
    for target_pos in range(seq_len):
        dot = torch.dot(base_pos, pe[target_pos]).item()
        sim = dot / (torch.norm(base_pos) * torch.norm(pe[target_pos])).item()
        bar = "█" * max(0, int(sim * 25))
        print(f"位置 0 与 位置 {target_pos} 的相似度: {sim:6.3f} | {bar}")

    print("\n[怎么读这张柱状图]：")
    print(f"1. 最快的那对正弦/余弦，周期是 2π ≈ {2 * math.pi:.2f} 个座位。")
    print("   所以走到第 6 个座位附近，这根快针转完差不多一圈，相似度会回升。")
    print("2. 这不是编码失败。请看数值表的 d2 列：位置 0 约是 0.00，位置 6 约是 0.56。")
    print("   慢针的读数不同，两个座位的整组指纹仍然不一样。")
    print("3. 在只有 8 维、句子又很短时，好几根慢针几乎不动，余弦会被它们托得很高。")
    print("   真正用来认座位的，是“快针 + 慢针”合在一起的那一整列数字。")

if __name__ == "__main__":
    main()
