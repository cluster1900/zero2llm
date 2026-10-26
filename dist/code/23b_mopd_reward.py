"""P22 extra: per-token MOPD reward with clipping, and teacher routing.

Run: python code/23b_mopd_reward.py   (pure Python, no PyTorch needed)
Sources: Kimi K3 report, arXiv:2607.24653 (S29), section 4.1.3 Eq.15;
Nemotron 3.5 Super docs (S30): one teacher per domain, plus a frozen
self-anchor for prompts that no specialist owns.
Tiny made-up vocab distributions. No training, no quality claim.
"""
import math


def opd_reward(p_teacher, p_student, r_max):
    """Eq.15 for one sampled token: clip(log(pi_teacher / pi_theta), -R, R).

    In training the log-ratio is under stop-gradient: it is a reward number,
    not a path for gradients. Plain floats make that automatic here.
    """
    if r_max <= 0:
        raise ValueError("R_max must be positive")
    raw = math.log(p_teacher) - math.log(p_student)
    return max(-r_max, min(r_max, raw))


def expected_reward(teacher, student, r_max=float("inf")):
    """E_{y ~ student}[reward]. Without clipping this equals -KL(student || teacher)."""
    return sum(s * opd_reward(t, s, r_max) for t, s in zip(teacher, student) if s > 0)


def kl(p, q):
    return sum(a * (math.log(a) - math.log(b)) for a, b in zip(p, q) if a > 0)


def pick_teacher(domain, teachers, self_anchor):
    """Nemotron-style routing: the owning teacher, else the frozen self-anchor."""
    return teachers.get(domain, self_anchor)


def demo():
    teacher = [0.70, 0.20, 0.05, 0.05]
    student = [0.25, 0.25, 0.25, 0.25]
    for y in range(4):
        print(f"token {y}: raw={math.log(teacher[y] / student[y]):+.4f} "
              f"clipped(R=1)={opd_reward(teacher[y], student[y], 1.0):+.4f}")
    print(f"E[reward] = {expected_reward(teacher, student):+.6f}  -KL(s||t) = {-kl(student, teacher):+.6f}")


if __name__ == "__main__":
    demo()
