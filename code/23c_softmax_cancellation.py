"""P22 extra: Softmax Cancellation in soft-target distillation (CPU, fixed inputs).

Run: OMP_NUM_THREADS=1 python code/23c_softmax_cancellation.py
Source: Barron, Gnanapragasam, Tschannen, "A Fix for Softmax and Cross-Entropy",
arXiv:2610.31642, Sec. 3, Eq. 8-9, and the project page https://jonbarron.info/softmaxfix/.
`soft_target_cross_entropy` transcribes the project-page JAX listing into PyTorch.
The page states no code license; we copy only the algorithm, with attribution.
Naming follows the paper: p = teacher softmax(z_T), q = student softmax(z_S).
Nothing here distills a model; it only compares gradients of single logit rows.
"""
import importlib.util
import sys
from pathlib import Path

try:
    import torch
except ModuleNotFoundError as exc:
    if __name__ == "__main__" and exc.name == "torch":
        print("P22 extra requires PyTorch. Install dependencies: python -m pip install -r requirements.txt", file=sys.stderr)
        raise SystemExit(2) from None
    raise

_spec = importlib.util.spec_from_file_location("p04_collapse", Path(__file__).with_name("03b_softmax_collapse.py"))
_p04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_p04)
fixed_log_softmax = _p04.fixed_log_softmax

# Project page, second listing: the values printed by the authors (JAX, float32).
PAPER_EXAMPLE = {"standard": [-5.960e-8, 1.452e-8, -1.123e-9],
                 "fixed": [-4.596e-8, 4.144e-8, 4.513e-9],
                 "exact": [-4.596e-8, 4.144e-8, 4.513e-9]}


def standard_soft_ce(z_S, z_T):
    """-sum p * log_softmax(z_S) with the framework's built-in log_softmax."""
    return -(torch.softmax(z_T, -1) * torch.log_softmax(z_S, -1)).sum(-1)


def soft_target_cross_entropy(z_S, z_T):
    """Project-page listing: forward loss unchanged, gradient replaced by g (Eq. 8-9)."""
    log_q = fixed_log_softmax(z_S)
    q, p = log_q.exp(), fixed_log_softmax(z_T).exp()
    delta = z_S - z_T                                              # Logit difference.
    d = delta - (p * delta).sum(-1, keepdim=True)                  # Center under p.
    s = d - torch.log1p((p * torch.expm1(d)).sum(-1, keepdim=True))  # log(q / p).
    u = torch.where(s.abs() < 1, p * torch.expm1(s), q - p)        # Difference q - p.
    g = u - q * u.sum(-1, keepdim=True)                            # Ensure sum(g) = 0.
    loss = -(p * log_q.detach()).sum(-1)                           # Forward loss.
    return loss + (g.detach() * (z_S - z_S.detach())).sum(-1)      # Custom gradient g.


def student_grad(loss_fn, z_S, z_T):
    z_S = z_S.clone().requires_grad_(True)
    loss_fn(z_S, z_T.detach()).sum().backward()
    return [float(x) for x in z_S.grad]


def exact_grad(z_S, z_T):
    """float64 reference: d/dz_S of soft CE is q - p (Sec. 3)."""
    return [float(x) for x in torch.softmax(z_S.double(), -1) - torch.softmax(z_T.double(), -1)]


def paper_case():
    z_T = torch.tensor([2.0, 1.0, 0.0], dtype=torch.float32)
    z_S = z_T + torch.tensor([-1e-7, 1e-7, 0.0], dtype=torch.float32)
    return z_S, z_T


def main():
    torch.manual_seed(0)
    z_S, z_T = paper_case()
    fmt = lambda xs: "[" + ", ".join(f"{x:.3e}" for x in xs) + "]"
    print("paper fp32 example, z_S = z_T + [-1e-7, 1e-7, 0]")
    print("  standard", fmt(student_grad(standard_soft_ce, z_S, z_T)), "paper", fmt(PAPER_EXAMPLE["standard"]))
    print("  fixed   ", fmt(student_grad(soft_target_cross_entropy, z_S, z_T)), "paper", fmt(PAPER_EXAMPLE["fixed"]))
    print("  exact64 ", fmt(exact_grad(z_S, z_T)))
    z = torch.randn(4, 8, generator=torch.Generator().manual_seed(0)).to(torch.bfloat16) * 4
    std = torch.tensor([student_grad(standard_soft_ce, r, r) for r in z])
    fix = torch.tensor([student_grad(soft_target_cross_entropy, r, r) for r in z])
    print("student == teacher, bf16, seed 0, 4 rows x 8 classes; true gradient is 0")
    print(f"  standard max|g| = {std.abs().max():.3e}")
    print(f"  fixed    max|g| = {fix.abs().max():.3e}")


if __name__ == "__main__":
    main()
