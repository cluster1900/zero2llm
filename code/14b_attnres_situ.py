"""P07/P13 extra: Attention Residuals weights and the SiTU-GLU bounded gate.

Run: python code/14b_attnres_situ.py   (pure Python, no PyTorch needed)
Sources: Kimi K3 report, arXiv:2607.24653 (S29), section 2.2 Eq.8-9 and
section 2.3.2 Eq.12; original AttnRes report, arXiv:2603.15031.
Scalar and tiny-vector toys only. No training, no speed or quality claim.
"""
import math


def rmsnorm(x, eps=1e-6):
    r = math.sqrt(sum(a * a for a in x) / len(x) + eps)
    return [a / r for a in x]


def attnres_weights(w, sources):
    """Eq.9: softmax over phi(w, k_i) = exp(w^T RMSNorm(k_i))."""
    logits = [sum(a * b for a, b in zip(w, rmsnorm(k))) for k in sources]
    top = max(logits)
    e = [math.exp(z - top) for z in logits]
    s = sum(e)
    return [a / s for a in e]


def attnres_mix(w, sources):
    """h_l = sum_i alpha_i v_i, with v_i = k_i (Eq.8)."""
    a = attnres_weights(w, sources)
    return [sum(ai * v[j] for ai, v in zip(a, sources)) for j in range(len(sources[0]))], a


def plain_residual(sources):
    """Standard residual: every earlier output added with weight 1."""
    return [sum(v[j] for v in sources) for j in range(len(sources[0]))]


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def swiglu(g, u):
    """Scalar SwiGLU: Swish(g) * u, unbounded in both factors."""
    return g * sigmoid(g) * u


def situ_glu(g, u, beta1=4.0, beta2=25.0):
    """Scalar Eq.12: [b1 tanh(g/b1) sigmoid(g)] * [b2 tanh(u/b2)]."""
    return beta1 * math.tanh(g / beta1) * sigmoid(g) * beta2 * math.tanh(u / beta2)


def demo():
    src = [[1.0, 0.0, 2.0], [0.5, -1.0, 0.0], [3.0, 3.0, -3.0]]
    h, a = attnres_mix([0.0, 0.0, 0.0], src)
    print("w = 0 -> weights", [round(x, 4) for x in a], "sum", round(sum(a), 12))
    print("AttnRes mix * L =", [round(x * len(src), 6) for x in h], "plain residual =", plain_residual(src))
    for x in (0.1, 1.0, 10.0, 100.0):
        print(f"x={x:>6}: SwiGLU(x,x)={swiglu(x, x):10.4f}  SiTU-GLU(x,x)={situ_glu(x, x):8.4f}")


if __name__ == "__main__":
    demo()
