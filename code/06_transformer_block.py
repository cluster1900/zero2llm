"""P07: residual + LayerNorm + position-wise ReLU FFN, on CPU.

Default: Pre-LN, no dropout, no final stack-level norm, no language-model head.
This teaching block is not a replica of a particular GPT or LLaMA checkpoint.
Run this file to regenerate every numerical table in chapter 6 and a smoke check.
"""

import math

import torch
import torch.nn as nn


class ManualLayerNorm(nn.Module):
    """Normalize the last feature axis, then learn one gamma/beta per feature."""

    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        if not isinstance(d_model, int) or isinstance(d_model, bool) or d_model <= 0:
            raise ValueError("d_model must be a positive integer")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        if x.ndim < 1 or x.shape[-1] != self.gamma.numel():
            raise ValueError("the last input dimension must equal d_model")
        # 1. 每个位置沿最后的特征轴求均值；不混合 batch 或 token。
        mean = x.mean(dim=-1, keepdim=True)
        # 2. 除以 D，不是 D-1：这里描述当前向量，不估计总体方差。
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        # 3. 先减均值，再除以标准差；epsilon 放在平方根里面。
        x_norm = (x - mean) / torch.sqrt(variance + self.eps)
        # 学到的 gamma/beta 之后，输出不必继续保持零均值、近单位方差。
        return self.gamma * x_norm + self.beta


class FeedForwardNetwork(nn.Module):
    """Each position uses the SAME two affine layers and a ReLU in between."""

    def __init__(self, d_model, d_ff=None):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model  # 教学默认值，不是所有 FFN 必须放大四倍。
        for name, width in [("d_model", d_model), ("d_ff", d_ff)]:
            if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.linear1 = nn.Linear(d_model, d_ff)  # includes b1
        self.linear2 = nn.Linear(d_ff, d_model)  # includes b2
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


class TransformerBlock(nn.Module):
    """One self-attention block; norm_first=True preserves the original API.

    mask is a [T,T] allowed-connection mask: 1/True means visible.
    With mask=None, attention is bidirectional, NOT automatically causal.
    This small example does not implement the padding contract of attention_core.
    """

    def __init__(self, d_model, num_heads, norm_first=True, d_ff=None, eps=1e-5):
        super().__init__()
        if not isinstance(num_heads, int) or isinstance(num_heads, bool) or num_heads <= 0:
            raise ValueError("num_heads must be a positive integer")
        self.ln1 = ManualLayerNorm(d_model, eps)
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if not isinstance(norm_first, bool):
            raise TypeError("norm_first must be a boolean")
        self.norm_first = norm_first
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=0.0)
        self.ln2 = ManualLayerNorm(d_model, eps)
        self.ffn = FeedForwardNetwork(d_model, d_ff)

    def _attention(self, x, mask):
        # PyTorch MHA 的布尔 mask 与本书约定相反：True = 禁止看。
        attn_mask = None if mask is None else (mask == 0)
        out, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        return out

    def forward(self, x, mask=None):
        if self.norm_first:
            # Pre-LN: 残差主干不经过 LN，LN 只在计算增量的支路上。
            x = x + self._attention(self.ln1(x), mask)
            x = x + self.ffn(self.ln2(x))
        else:
            # Post-LN: 先求增量并加回输入，然后归一化整个和。
            x = self.ln1(x + self._attention(x, mask))
            x = self.ln2(x + self.ffn(x))
        return x


def layer_norm_walkthrough():
    """Keep all three Z-score steps visible, using float64 for hand checking."""
    x = torch.tensor([1., 2., 3., 4.], dtype=torch.float64)
    mean = x.mean()
    centered = x - mean
    squared = centered.square()
    variance = squared.mean()
    eps = 1e-5
    denominator = torch.sqrt(variance + eps)
    normalized = ManualLayerNorm(4, eps).double()(x)
    return x, mean, centered, squared, variance, denominator, normalized


def ordering_walkthrough():
    """A single toy sublayer F(z)=2z isolates order; it is not attention."""
    x = torch.tensor([1., 2., 3., 4.], dtype=torch.float64)
    ln = ManualLayerNorm(4).double()
    pre_input, post_input = ln(x), x
    pre_delta, post_delta = 2 * pre_input, 2 * post_input
    pre_sum, post_sum = x + pre_delta, x + post_delta
    return [
        ("进入 F 的向量", pre_input, post_input),
        ("F 输出的增量", pre_delta, post_delta),
        ("与原输入相加", pre_sum, post_sum),
        ("子层最终输出", pre_sum, ln(post_sum)),
    ]


def toy_ffn():
    """Small hand-set weights, in PyTorch [out,in] storage order."""
    model = FeedForwardNetwork(2, 4).double()
    with torch.no_grad():
        model.linear1.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [1., 1.], [-1., 1.]]))
        model.linear1.bias.copy_(torch.tensor([0., 1., 0., 0.]))
        model.linear2.weight.copy_(torch.tensor([[1., 2., 1., -1.], [0., 1., -1., 2.]]))
        model.linear2.bias.copy_(torch.tensor([0.5, -0.5]))
    return model


def _vector(x, digits=4):
    return "[" + ", ".join(f"{v:.{digits}f}" for v in x.detach().tolist()) + "]"


def _table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    return "\n".join(lines)


def chapter_tables():
    """Generate chapter tables, rather than maintaining hand-copied numbers."""
    tables = {}
    x, mean, centered, squared, variance, denominator, normalized = layer_norm_walkthrough()
    delta = torch.tensor([0.5, -0.5, 0., 1.], dtype=torch.float64)
    tables["residual"] = _table(["量", "四个特征"], [
        ("输入 x", _vector(x)), ("增量 F(x)", _vector(delta)),
        ("x + F(x)", _vector(x + delta)), ("F(x)=0 时的输出", _vector(x + torch.zeros_like(x))),
    ])
    tables["zscore"] = _table(["特征 i", "x_i", "x_i − μ", "(x_i − μ)²", "标准化后 x̂_i"], [
        (i + 1, f"{x[i]:.1f}", f"{centered[i]:.1f}", f"{squared[i]:.2f}", f"{normalized[i]:.6f}")
        for i in range(x.numel())
    ])
    epsilon_rows = []
    for label, values, eps in [
        ("[1,2,3,4]", x, 1e-5), ("[1,2,3,4]", x, 1.0),
        ("[1,1.002,1.004,1.006]", torch.tensor([1., 1.002, 1.004, 1.006], dtype=torch.float64), 1e-5),
        ("[3,3,3,3]", torch.full((4,), 3., dtype=torch.float64), 1e-5),
    ]:
        y = ManualLayerNorm(4, eps).double()(values)
        epsilon_rows.append((label, f"{values.var(unbiased=False):.6f}", f"{eps:g}",
                             f"{y.var(unbiased=False):.6f}", _vector(y)))
    tables["epsilon"] = _table(["输入", "输入方差", "epsilon", "输出方差", "标准化后向量"], epsilon_rows)
    tables["ordering"] = _table(["步骤", "Pre-LN", "Post-LN"], [
        (step, _vector(pre), _vector(post)) for step, pre, post in ordering_walkthrough()
    ])
    ffn = toy_ffn()
    toy_x = torch.tensor([1., -2.], dtype=torch.float64)
    hidden = ffn.linear1(toy_x)
    tables["ffn_steps"] = _table(["步骤", "结果"], [
        ("输入 x", _vector(toy_x, 1)), ("第一层 xW₁+b₁", _vector(hidden, 1)),
        ("ReLU 后", _vector(ffn.relu(hidden), 1)), ("第二层输出", _vector(ffn(toy_x), 1)),
        ("去掉 ReLU 的输出（反例）", _vector(ffn.linear2(hidden), 1)),
    ])
    block = TransformerBlock(16, 4)
    rows = [(name, str(list(param.shape)), param.numel()) for name, param in block.ffn.named_parameters()]
    rows.append(("FFN 合计", "两个权重和两个偏置", sum(p.numel() for p in block.ffn.parameters())))
    tables["ffn_budget"] = _table(["参数", "PyTorch 存储形状", "个数"], rows)
    tables["block_budget"] = _table(["组件（D=16，d_ff=64）", "计数公式", "个数"], [
        ("MHA（含 Q/K/V/O bias）", "4D² + 4D", sum(p.numel() for p in block.attn.parameters())),
        ("两个 LayerNorm", "4D", sum(p.numel() for norm in [block.ln1, block.ln2] for p in norm.parameters())),
        ("FFN（含 b₁、b₂）", "2D·d_ff + d_ff + D", sum(p.numel() for p in block.ffn.parameters())),
        ("本块合计", "以上相加", sum(p.numel() for p in block.parameters())),
    ])
    return tables


def main():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    print("P07 CPU walkthrough; torch=" + torch.__version__)
    _, mean, _, _, variance, denominator, normalized = layer_norm_walkthrough()
    print(f"Z-score: mean={mean:.6f}, variance={variance:.6f}, denominator={denominator:.6f}")
    print(f"normalized variance={normalized.var(unbiased=False):.6f}")
    for name, table in chapter_tables().items():
        print(f"\n[{name}]\n{table}")

    x = torch.randn(2, 5, 16, dtype=torch.float64, requires_grad=True)
    causal = torch.ones(5, 5, dtype=torch.bool).tril()
    for norm_first in (True, False):
        block = TransformerBlock(16, 4, norm_first=norm_first).double()
        output = block(x, causal)
        output.square().mean().backward()
        finite = torch.isfinite(output).all() and torch.isfinite(x.grad).all()
        finite = finite and all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())
        assert finite
        print(f"{'Pre-LN' if norm_first else 'Post-LN'}: shape={list(output.shape)}, forward/backward finite={bool(finite)}")
        x.grad = None

    zero_block = TransformerBlock(16, 4).double()
    with torch.no_grad():
        for sublayer in (zero_block.attn, zero_block.ffn):
            for param in sublayer.parameters():
                param.zero_()
    error = (zero_block(x.detach()) - x.detach()).abs().max().item()
    assert error == 0.0
    print(f"Pre-LN zero-sublayer identity max error={error:.6f}")


if __name__ == "__main__":
    main()
