"""P22 extra: Softmax Cancellation in soft-target distillation (CPU, fixed inputs).

Run: OMP_NUM_THREADS=1 python code/23c_softmax_cancellation.py
Source: Barron, Gnanapragasam, Tschannen, "A Fix for Softmax and Cross-Entropy",
arXiv:2610.31642, Sec. 3, Eq. 8-9, and the project page https://jonbarron.info/softmaxfix/.
`soft_target_cross_entropy` is our own PyTorch reimplementation of the paper's
algorithm (Eq. 8-9), written with attribution. The project page states no code
license, so code and comments here are self-written, not copied.
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

# Paper Sec. 3 example: gradients printed by the authors (JAX, float32).
PAPER_EXAMPLE = {"standard": [-5.960e-8, 1.452e-8, -1.123e-9],
                 "fixed": [-4.596e-8, 4.144e-8, 4.513e-9],
                 "exact": [-4.596e-8, 4.144e-8, 4.513e-9]}


def standard_soft_ce(z_S, z_T):
    """-sum p * log_softmax(z_S) with the framework's built-in log_softmax."""
    return -(torch.softmax(z_T, -1) * torch.log_softmax(z_S, -1)).sum(-1)


def student_minus_teacher(z_S, z_T):
    """q - p computed from the logit gap, following the steps of Eq. 8-9.

    Working in log space keeps the tiny difference instead of subtracting two
    nearly equal probabilities. When z_S == z_T every step is an exact 0.
    """
    p = fixed_log_softmax(z_T).exp()
    q = fixed_log_softmax(z_S).exp()
    gap = z_S - z_T
    centred = gap - (p * gap).sum(-1, keepdim=True)
    log_ratio = centred - torch.log1p((p * torch.expm1(centred)).sum(-1, keepdim=True))
    # Far from the teacher plain q - p is already accurate (project-page variant).
    diff = torch.where(log_ratio.abs() < 1, p * torch.expm1(log_ratio), q - p)
    # Remove leftover rounding so the components add up to exactly zero.
    return diff - q * diff.sum(-1, keepdim=True)


class _SoftTargetCE(torch.autograd.Function):
    """Forward: the usual soft-target CE. Backward: q - p from student_minus_teacher."""

    @staticmethod
    def forward(ctx, z_S, z_T):
        ctx.save_for_backward(z_S, z_T)
        p = fixed_log_softmax(z_T).exp()
        return -(p * fixed_log_softmax(z_S)).sum(-1)

    @staticmethod
    def backward(ctx, grad_loss):
        z_S, z_T = ctx.saved_tensors
        return grad_loss.unsqueeze(-1) * student_minus_teacher(z_S, z_T), None


def soft_target_cross_entropy(z_S, z_T):
    """Soft-target CE whose gradient w.r.t. z_S is the accurate q - p (teacher held fixed)."""
    return _SoftTargetCE.apply(z_S, z_T)


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
