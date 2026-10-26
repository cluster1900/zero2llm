"""P23 extra: reasoning-effort budget reward and the LK loss.

Run: python code/24c_budget_lk.py   (pure Python, no PyTorch needed)
Sources: Kimi K3 report, arXiv:2607.24653 (S29), section 4.1.2 (budget
control) and section 4.1.4 Eq.16 (LK loss); LK losses, arXiv:2602.23881.
All distributions and token counts are made up by this book.
"""
import math


def budget_reward(task_reward, tokens, b0, tau):
    """S29 4.1.2: reward becomes -1 when T(y) > tau * b0(x); otherwise unchanged."""
    return -1.0 if tokens > tau * b0 else task_reward


def acceptance(p, q):
    """Per-token acceptance rate of lossless speculative sampling: sum_x min(p, q)."""
    return sum(min(a, b) for a, b in zip(p, q))


def lk_loss(p, q):
    """Eq.16: -log sum_x min(p(x), q(x)); p = target, q = draft, temperature 1."""
    return -math.log(acceptance(p, q))


def kl(p, q):
    return sum(a * (math.log(a) - math.log(b)) for a, b in zip(p, q) if a > 0)


# Book-made example: draft A has the smaller KL(p||q), draft B the higher acceptance.
TARGET = [0.6, 0.3, 0.1]
DRAFT_A = [0.45, 0.40, 0.15]
DRAFT_B = [0.65, 0.349, 0.001]


def demo():
    runs = [(1.0, 900), (1.0, 1500), (0.0, 700), (1.0, 2600)]  # (task reward, tokens)
    b0 = 1000
    for tau in (3.0, 2.0, 1.0):
        print(f"tau={tau}: rewards", [budget_reward(r, t, b0, tau) for r, t in runs])
    for name, q in (("A", DRAFT_A), ("B", DRAFT_B)):
        print(f"draft {name}: KL(p||q)={kl(TARGET, q):.4f} accept={acceptance(TARGET, q):.3f} "
              f"LK={lk_loss(TARGET, q):.4f}")


if __name__ == "__main__":
    demo()
