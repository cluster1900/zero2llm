"""P16 extra: KDA delta-rule recurrence, chunked state passing, bounded decay.

Run: python code/17c_kda_hybrid.py   (pure Python, no PyTorch needed)
Source: Kimi K3 technical report, arXiv:2607.24653 (S29), section 2.1.1,
Eq.1 (recurrence), Eq.3-4 (chunkwise form), Eq.5 (lower-bounded decay).
Teaching reconstruction only. The UT transform and the Tensor Core kernels are
NOT implemented. The chunk form here passes the state between chunks and builds
each chunk from per-step transition products, which is exact but slow.
"""
import math

BF16_MAX = 3.3895313892515355e38  # largest finite bfloat16 value


def zeros(r, c):
    return [[0.0] * c for _ in range(r)]


def kda_step(S, k, v, alpha, beta):
    """One step of Eq.1: S <- (I - beta k k^T) Diag(alpha) S + beta k v^T."""
    dk, dv = len(S), len(S[0])
    D = [[alpha[i] * S[i][j] for j in range(dv)] for i in range(dk)]
    kD = [sum(k[i] * D[i][j] for i in range(dk)) for j in range(dv)]  # k^T D
    return [[D[i][j] - beta * k[i] * kD[j] + beta * k[i] * v[j] for j in range(dv)]
            for i in range(dk)]


def readout(S, q):
    """o = S^T q."""
    return [sum(S[i][j] * q[i] for i in range(len(S))) for j in range(len(S[0]))]


def kda_recurrent(qs, ks, vs, alphas, betas, S0=None):
    S = S0 if S0 is not None else zeros(len(ks[0]), len(vs[0]))
    outs = []
    for q, k, v, a, b in zip(qs, ks, vs, alphas, betas):
        S = kda_step(S, k, v, a, b)
        outs.append(readout(S, q))
    return outs, S


def _affine_step(M, c, k, v, alpha, beta):
    """Compose one step onto an affine map S -> M S + c (M is dk x dk)."""
    dk = len(M)
    A = [[(1.0 if i == j else 0.0) * alpha[j] - beta * k[i] * k[j] * alpha[j]
          for j in range(dk)] for i in range(dk)]
    M2 = [[sum(A[i][r] * M[r][j] for r in range(dk)) for j in range(dk)] for i in range(dk)]
    c2 = [[sum(A[i][r] * c[r][j] for r in range(dk)) + beta * k[i] * v[j]
           for j in range(len(c[0]))] for i in range(dk)]
    return M2, c2


def kda_chunked(qs, ks, vs, alphas, betas, chunk):
    """Same outputs, computed chunk by chunk from the state entering each chunk."""
    dk, dv = len(ks[0]), len(vs[0])
    S = zeros(dk, dv)
    outs = []
    for start in range(0, len(ks), chunk):
        M = [[1.0 if i == j else 0.0 for j in range(dk)] for i in range(dk)]
        c = zeros(dk, dv)
        for t in range(start, min(start + chunk, len(ks))):
            M, c = _affine_step(M, c, ks[t], vs[t], alphas[t], betas[t])
            St = [[sum(M[i][r] * S[r][j] for r in range(dk)) + c[i][j] for j in range(dv)]
                  for i in range(dk)]
            outs.append(readout(St, qs[t]))
        S = St
    return outs, S


def g_softplus(z, A=0.0):
    """Kimi Linear mapping: g = -e^A Softplus(z), unbounded below."""
    return -math.exp(A) * math.log1p(math.exp(z))


def g_bounded(z, A=0.0, g_min=-5.0):
    """Kimi K3 Eq.5: g = g_min * Sigmoid(e^A z), inside (g_min, 0)."""
    return g_min / (1.0 + math.exp(-math.exp(A) * z))


def max_log_reciprocal(gs, tile=16):
    """Largest log(1/Gamma) inside any tile: minus the cumulative log-decay."""
    worst = 0.0
    for start in range(0, len(gs), tile):
        worst = max(worst, -sum(gs[start:start + tile]))
    return worst


def demo():
    import random
    rnd = random.Random(0)
    T, dk, dv = 12, 3, 2

    def unit(n):
        x = [rnd.gauss(0, 1) for _ in range(n)]
        s = math.sqrt(sum(a * a for a in x))
        return [a / s for a in x]
    qs = [unit(dk) for _ in range(T)]
    ks = [unit(dk) for _ in range(T)]
    vs = [[rnd.gauss(0, 1) for _ in range(dv)] for _ in range(T)]
    alphas = [[math.exp(g_bounded(rnd.gauss(0, 2))) for _ in range(dk)] for _ in range(T)]
    betas = [1 / (1 + math.exp(-rnd.gauss(0, 1))) for _ in range(T)]
    o1, _ = kda_recurrent(qs, ks, vs, alphas, betas)
    o2, _ = kda_chunked(qs, ks, vs, alphas, betas, chunk=4)
    err = max(abs(a - b) for x, y in zip(o1, o2) for a, b in zip(x, y))
    print(f"recurrent vs chunked max |diff| = {err:.2e}")
    zs = [8.0] * 16
    lim = math.log(BF16_MAX)
    print(f"ln(BF16 max) = {lim:.2f}")
    print(f"16-token tile, z=8: softplus log(1/Gamma) = {max_log_reciprocal([g_softplus(z) for z in zs]):.1f}")
    print(f"16-token tile, z=8: bounded  log(1/Gamma) = {max_log_reciprocal([g_bounded(z) for z in zs]):.1f}")


if __name__ == "__main__":
    demo()
