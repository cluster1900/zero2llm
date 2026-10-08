"""
06_transformer_block.py - 组装核心积木：Transformer Block
适合人群：已理解多头注意力，想搞懂残差连接、层归一化和前馈网络组装过程的学习者

三大积木组件：
1. 多头自注意力（Multi-Head Attention）：负责词与词之间的“横向沟通”
2. 残差连接（Residual Connection）：抄近道，输出 = 输入 + 处理结果，防止深层信息遗忘
3. 层归一化（Layer Normalization）：全科成绩标准化，把均值拉到 0，方差拉到 1
4. 前馈神经网络（FFN / MLP）：负责每个词“在自己脑海里消化深思”
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ManualLayerNorm(nn.Module):
    """
    从均值与方差实现层归一化（LayerNorm）：
    标准化公式： Z = (X - μ) / σ
    """
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        # gamma (缩放权重) 初始为 1，beta (平移偏置) 初始为 0
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        # 1. 沿着最后一个特征维度计算均值 μ
        mean = x.mean(dim=-1, keepdim=True)
        # 2. 计算方差 σ² = 平均((X - μ)²)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        # 3. 标准化：(X - μ) / √(σ² + ε)
        x_norm = (x - mean) / torch.sqrt(variance + self.eps)
        # 4. 可学习的缩放和平移：y = γ * x_norm + β
        return self.gamma * x_norm + self.beta

class FeedForwardNetwork(nn.Module):
    """
    前馈神经网络（FFN）：
    结构：先放大维度 4 倍（展开联想），经过激活函数，再收缩回原始维度。
    可以写成复合函数 f(x) = W2 · ReLU(W1 · x + b1) + b2
    """
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model  # 经典 Transformer 放大 4 倍
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 升维 -> 激活思考 -> 降维总结
        return self.linear2(self.relu(self.linear1(x)))

class TransformerBlock(nn.Module):
    """
    一个完整的 Transformer 编码器积木块（现代 Pre-LN 架构，GPT / LLaMA 同款）
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.ln1 = ManualLayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.ln2 = ManualLayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model)

    def forward(self, x, mask=None):
        # --- 第一阶段：注意力 + 残差连接 ---
        # 1. 先进行层归一化 (Pre-LN)
        norm_x1 = self.ln1(x)
        # 2. 计算自注意力。
        # mask 与本书其他代码约定相同：1 表示允许看，0 表示禁止看。
        # PyTorch 官方的布尔掩码相反：True 表示“这个位置不准看”。
        attn_mask = None if mask is None else (mask == 0)
        attn_out, _ = self.attn(
            norm_x1, norm_x1, norm_x1, attn_mask=attn_mask, need_weights=False
        )
        # 3. 残差连接（抄近道）：输入 + 增量
        x = x + attn_out

        # --- 第二阶段：前馈思考 + 残差连接 ---
        # 1. 先进行层归一化
        norm_x2 = self.ln2(x)
        # 2. 前馈网络深度消化
        ffn_out = self.ffn(norm_x2)
        # 3. 第二次残差连接
        x = x + ffn_out

        return x

def main():
    print("=" * 65)
    print("【Transformer Block 组装与运行测试】")
    print("=" * 65)

    batch_size = 2   # 2 句话
    seq_len = 5      # 每句话 5 个词
    d_model = 16     # 词向量维度 16
    num_heads = 4    # 4 个注意力头

    torch.manual_seed(42)
    sample_input = torch.randn(batch_size, seq_len, d_model)
    print(f"输入特征形状: {sample_input.shape}")

    # 1. 单独测试 LayerNorm 的魔力
    ln = ManualLayerNorm(d_model)
    normed = ln(sample_input)
    print("\nLayerNorm 测试（检查第一个词的均值与方差）：")
    print(f"标准化前：均值={sample_input[0, 0].mean().item():.4f}, 方差={sample_input[0, 0].var(unbiased=False).item():.4f}")
    print(f"标准化后：均值={normed[0, 0].mean().item():.4f} (≈0), 方差={normed[0, 0].var(unbiased=False).item():.4f} (≈1)")

    # 2. 测试完整的 TransformerBlock
    block = TransformerBlock(d_model=d_model, num_heads=num_heads)
    output = block(sample_input)

    print(f"\n整块 Transformer Block 输出形状: {output.shape}")
    print("输入与输出的形状完全一模一样！这就是为什么我们可以像拼乐高积木一样，把它们堆叠几十层、几百层！")

if __name__ == "__main__":
    main()
