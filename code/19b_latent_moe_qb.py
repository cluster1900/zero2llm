"""P18 extra: Stable LatentMoE shape ledger and Quantile Balancing (QB).

Run: python code/19b_latent_moe_qb.py   (pure Python, no PyTorch needed)
Source: Kimi K3 report, arXiv:2607.24653 (S29), section 2.3 Eq.11,
section 2.3.3 Eq.13-14 and Figure 5 (m=8 tokens, n=4 experts, k=1).
The score matrix below is made up by this book. It only has the same size and
the same starting loads (4,3,1,0) as Figure 5. It is not the report's data.
"""


def topk(row, k):
    return sorted(range(len(row)), key=lambda j: -row[j])[:k]


def route(scores, bias, k):
    """Eq.13 selection: argtop_k(s_i + b). Returns chosen expert lists."""
    return [topk([s + b for s, b in zip(row, bias)], k) for row in scores]


def loads(choice, n):
    out = [0] * n
    for c in choice:
        for j in c:
            out[j] += 1
    return out


def quantile_balance(scores, bias, k):
    """Eq.14: one QB update from one batch (applied on the NEXT batch)."""
    m, n = len(scores), len(scores[0])
    q = m * k // n
    cut = []
    for row in scores:
        biased = sorted((s + b for s, b in zip(row, bias)), reverse=True)
        cut.append(biased[k])  # the (k+1)-th largest biased score: alpha_i
    new = []
    for j in range(n):
        margins = sorted((scores[i][j] - cut[i] for i in range(m)), reverse=True)
        new.append(-margins[q])  # minus the (q+1)-th largest margin
    mean = sum(new) / n
    return [b - mean for b in new]


def routed_weights(row, chosen):
    """Eq.13 weights: bias is NOT used, raw scores renormalised over the chosen."""
    s = sum(row[j] for j in chosen)
    return {j: row[j] / s for j in chosen}


# Book-made scores, 8 tokens x 4 experts. Found by a seeded random search for a
# matrix whose top-1 loads are (4,3,1,0), like Figure 5. One QB step balances it.
SCORES_BALANCES = [
    [0.94, 0.17, 0.25, 0.93],
    [0.93, 0.86, 0.93, 0.71],
    [0.06, 0.62, 0.30, 0.34],
    [0.84, 0.34, 0.75, 0.74],
    [0.04, 0.83, 0.93, 0.82],
    [0.10, 0.58, 0.11, 0.04],
    [0.05, 1.00, 0.61, 0.50],
    [0.90, 0.86, 0.74, 0.05],
]

# Hand-written scores with the same starting loads (4,3,1,0). Here one QB step
# gives (3,3,1,1), and repeating QB on this same batch stays at (3,3,1,1).
SCORES_STALLS = [
    [0.90, 0.60, 0.20, 0.10],
    [0.85, 0.30, 0.70, 0.20],
    [0.80, 0.75, 0.10, 0.40],
    [0.70, 0.20, 0.30, 0.65],
    [0.30, 0.90, 0.20, 0.50],
    [0.40, 0.80, 0.60, 0.30],
    [0.20, 0.70, 0.10, 0.60],
    [0.30, 0.20, 0.90, 0.50],
]


def latent_moe_ledger(d=7168, latent=3584, routed=896, active=16, shared=2):
    """Numbers from S29 section 2.3 and Table 1; only arithmetic is done here."""
    return {"sparsity": routed // active, "latent_ratio": latent / d,
            "active_routed": active, "shared": shared}


def demo():
    k, n = 1, 4
    b0 = [0.0] * n
    for name, s in (("balances", SCORES_BALANCES), ("stalls", SCORES_STALLS)):
        b = [0.0] * n
        seen = [loads(route(s, b, k), n)]
        for _ in range(3):
            b = quantile_balance(s, b, k)
            seen.append(loads(route(s, b, k), n))
        print(f"{name}: loads per QB step {seen}")
    print("ledger:", latent_moe_ledger())


if __name__ == "__main__":
    demo()
