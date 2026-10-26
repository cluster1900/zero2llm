"""P04 extra: Softmax Collapse and the precision cliff (CPU, fixed inputs).

Run: OMP_NUM_THREADS=1 python code/03b_softmax_collapse.py
Source: Barron, Gnanapragasam, Tschannen, "A Fix for Softmax and Cross-Entropy",
arXiv:2610.31642, Sec. 2, Eq. 2-7. The fixed functions are our own PyTorch
reimplementation of the paper's algorithm (Eq. 5-6), written with attribution;
code and comments are not copied from the paper or project page.
Nothing here trains a model; it only compares gradients of single logit rows.
"""
import math
import sys

try:
    import torch
except ModuleNotFoundError as exc:
    if __name__ == "__main__" and exc.name == "torch":
        print("P04 extra requires PyTorch. Install dependencies: python -m pip install -r requirements.txt", file=sys.stderr)
        raise SystemExit(2) from None
    raise

# Paper Sec. 1 / Fig. 1: c_16 ~ 5.5 (bf16), c_32 ~ 16.6 (fp32), c_64 ~ 36.7 (fp64).
PAPER_CLIFFS = {torch.bfloat16: 5.5, torch.float32: 16.6, torch.float64: 36.7}


def margin(z, t):
    """Eq. 2: m = z_t - log sum_{j != t} exp(z_j), computed in float64."""
    z = z.double()
    others = torch.cat([z[:t], z[t + 1:]])
    return float(z[t] - torch.logsumexp(others, 0))


def measured_cliff(dtype):
    """Smallest margin m (0.01 grid) where 1 + exp(-m) rounds to exactly 1 in `dtype`."""
    one = torch.ones((), dtype=dtype)
    m = 0.0
    while True:
        if one + torch.tensor(math.exp(-m), dtype=dtype) == one:
            return round(m, 2)
        m += 0.01


def _shifted_and_rest(z, idx):
    """Shift every logit by z[idx], and log-sum-exp the *other* shifted logits.

    The slot `idx` is forced to an exact 0, and it never enters the sum, so the
    returned `rest` equals -m (Eq. 2) without first forming 1 + e^{-m}.
    """
    hit = torch.nn.functional.one_hot(idx, z.shape[-1]).bool()
    shifted = (z - z.gather(-1, idx.unsqueeze(-1))).masked_fill(hit, 0.0)
    rest = torch.logsumexp(shifted.masked_fill(hit, -math.inf), -1, keepdim=True)
    return shifted, rest


def fixed_log_softmax(z):
    """Eq. 5 reimplemented: shift by the largest logit, then subtract softplus(-m)."""
    shifted, rest = _shifted_and_rest(z, z.argmax(-1))
    return shifted - torch.nn.functional.softplus(rest)


def fixed_cross_entropy(z, t):
    """Eq. 6 reimplemented: hard-target CE written directly as softplus(-m)."""
    t = torch.as_tensor(t).expand(z.shape[:-1])
    _, rest = _shifted_and_rest(z, t)
    return torch.nn.functional.softplus(rest).squeeze(-1)


def standard_cross_entropy(z, t):
    """The usual -log_softmax(z)[t] (PyTorch's built-in max-shift implementation)."""
    return -torch.log_softmax(z, -1)[..., t]


def grad_of(loss_fn, z, t):
    z = z.clone().requires_grad_(True)
    loss = loss_fn(z, t)
    loss.sum().backward()
    return loss.detach(), z.grad.detach()


def compare(values, dtype, t=0):
    z = torch.tensor(values, dtype=dtype)
    out = {}
    for name, fn in (("standard", standard_cross_entropy), ("fixed", fixed_cross_entropy)):
        loss, g = grad_of(fn, z, t)
        out[name] = {"loss": float(loss), "grad": [float(x) for x in g], "sum": float(g.double().sum())}
    return out


def main():
    torch.manual_seed(0)
    for dtype, name in ((torch.bfloat16, "bf16"), (torch.float32, "fp32"), (torch.float64, "fp64")):
        print(f"{name}: measured cliff {measured_cliff(dtype):.2f}, paper {PAPER_CLIFFS[dtype]}")
    cases = (([9.0, -9.0, -9.0], torch.float32), ([4.0, -4.0, -4.0], torch.bfloat16),
             ([4.0, -4.0, -4.0], torch.float32))
    for values, dtype in cases:
        print(f"z={values} {str(dtype).split('.')[-1]} margin m={margin(torch.tensor(values), 0):.2f}")
        for kind, r in compare(values, dtype).items():
            g = ", ".join(f"{x:.3e}" for x in r["grad"])
            print(f"  {kind:8s} loss={r['loss']:.3e} grad=[{g}] sum={r['sum']:.3e}")


if __name__ == "__main__":
    main()
