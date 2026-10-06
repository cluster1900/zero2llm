"""
04_multi_head_attention.py - 多头注意力机制（Multi-Head Attention）
适合人群：已理解单个注意力机制，想搞懂“多头拆分与拼接”张量维度变换的学习者

生活比喻：
“三个臭皮匠，顶个诸葛亮”。
如果只有一个注意力头，模型只能用一种视角看句子（比如只能看语法）；
多头注意力把一个 8 维的向量切成 2 个 4 维的小分队，
分队 1 专攻“主谓宾关系”，分队 2 专攻“代词指代与上下文情感”！
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        参数说明（初学者通俗版）：
        - d_model: 词向量的总维度（每个词身上有多少个特征指标，比如 8）
        - num_heads: 头数（我们要分成几个专家分队，比如 2）
        - head_dim: 每个分队分到多少个特征指标 = d_model / num_heads = 8 / 2 = 4
        """
        super().__init__()
        assert d_model % num_heads == 0, "总维度 d_model 必须能被头数整除！"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # 线性投影矩阵：将输入分别投影为 Q, K, V
        # 相当于高中函数：y = x * W
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # 最后的输出融合矩阵 W_O：把拼在一起的多头特征重新揉合
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None, verbose=False):
        """
        前向传播计算：
        x 的形状: [batch_size, seq_len, d_model]
        - batch_size: 几句话（例如 1）
        - seq_len: 每句话几个字（例如 4）
        - d_model: 每个字几维（例如 8）
        """
        batch_size, seq_len, _ = x.shape

        if verbose:
            print(f"1. 输入张量 x 形状: {x.shape} (批次={batch_size}, 字数={seq_len}, 词向量维度={self.d_model})")

        # 1. 生成 Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        if verbose:
            print(f"2. 投影生成 Q, K, V 形状均为: {Q.shape}")

        # 2. 空间切分魔术：把总维度切成多个头
        # 变形前: [batch_size, seq_len, d_model]
        # 变形后: [batch_size, seq_len, num_heads, head_dim]
        # 再换轴: [batch_size, num_heads, seq_len, head_dim]
        # 为什么要换轴？因为把 num_heads 放到第 2 维后，每个头就可以独立看作一个传统的注意力！
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if verbose:
            print(f"3. 切分多头并换轴后 Q 形状: {Q.shape} (批次, 头数={self.num_heads}, 字数={seq_len}, 单头维度={self.head_dim})")

        # 3. 计算注意力分数 Score = (Q · K^T) / √head_dim
        # 最后的转置只对最后两个维度 (seq_len 和 head_dim) 翻转
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            # 加上掩码（比如因果掩码）
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax 归一化
        attn_weights = F.softmax(scores, dim=-1)

        # 4. 加权求和：Weights · V
        head_outputs = torch.matmul(attn_weights, V)

        if verbose:
            print(f"4. 各头独立注意力计算完成，输出形状: {head_outputs.shape}")

        # 5. 拼回原状（Concat）：
        # 先把维度换回: [batch_size, seq_len, num_heads, head_dim]
        # 再把最后两个维度合并: [batch_size, seq_len, d_model]
        out_concat = head_outputs.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        if verbose:
            print(f"5. 将所有头的特征拼接还原，形状: {out_concat.shape}")

        # 6. 过一层融合线性层
        output = self.W_o(out_concat)

        if verbose:
            print(f"6. 经过最终融合矩阵 W_O，最终输出形状: {output.shape}")

        return output, attn_weights

def main():
    print("=" * 65)
    print("【多头注意力机制演示与维度追踪】")
    print("=" * 65)

    # 设定超参数（用小数字，肉眼清晰可见）
    batch_size = 1
    seq_len = 4     # 4 个汉字："机器" "学习" "真" "酷"
    d_model = 8     # 总维度为 8
    num_heads = 2   # 切分成 2 个头，每个头维度为 4

    torch.manual_seed(100)
    # 随机生成一个模拟句子的词向量
    dummy_input = torch.randn(batch_size, seq_len, d_model)

    # 实例化模型
    mha = SimpleMultiHeadAttention(d_model=d_model, num_heads=num_heads)

    # 运行一次带详细日志的前向传播
    output, attn_weights = mha(dummy_input, verbose=True)

    print("\n" + "=" * 65)
    print("【观察每个头看到的注意力分布】")
    print("=" * 65)
    words = ["机器", "学习", "真", "酷"]
    for h in range(num_heads):
        print(f"\n--- 专家头 {h+1} 的注意力权重矩阵 ---")
        header = f"{'':<8}" + "".join([f"{w:>8}" for w in words])
        print(header)
        for i in range(seq_len):
            row_str = "".join([f"{attn_weights[0, h, i, j].item()*100:7.1f}%" for j in range(seq_len)])
            print(f"{words[i]:<6} {row_str}")

    print("\n[怎么读这两张表]：")
    print("两个头的百分比不一样，说明切成两队之后，它们可以走出不同的关注模式。")
    print("但这次的权重是随机初始化的（随机种子 100），模型没有训练过。")
    print("所以不能说头 1 已经是语法专家、头 2 已经懂了情感。")
    print("多头提供的是“允许以后分工”的结构。分工要等真正训练之后才会出现。")

if __name__ == "__main__":
    main()
