"""P23 extra: EAGLE-4-style cost-model draft length + low-rank candidate head (CPU).

Run: OMP_NUM_THREADS=1 python code/24d_eagle4_cost.py
Teaching reconstructions, NOT a copy of eagle-harness (LicenseRef-Eagle-4-
Source-Available, not OSI open source). Sources checked 2026-10-28:
  Paper PDF in eagle-hq/EAGLE-4@3b96ef6e1c0b2ef0c698fab74c8e383d8bf8053d
    (EAGLE4_Paper.pdf): §3.1 ΔA/ΔC ≥ τ; §3.2 rank-r candidate head then full
    LM head on top-K; Algorithm 1.
  LightSeek / TokenSpeed blog 2026-10-20: 750 tok/s (vendor report), rank-128
    over 248K vocab then top-128 exact, SWA 4096, append-only draft KV.
  fal technical deepdive 2026-10-20: batch-size tradeoff, harness intuition.
All numbers labelled vendor/author report are not reproduced here.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError as exc:
    if __name__ == "__main__" and exc.name == "torch":
        print("P23 requires PyTorch. Install dependencies: python -m pip install -r requirements.txt",
              file=sys.stderr)
        raise SystemExit(2) from None
    raise

_NAME = "zero2llm_s6"
if _NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(_NAME, Path(__file__).with_name("24_reasoning_mtp.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[_NAME] = module
    spec.loader.exec_module(module)
s6 = sys.modules[_NAME]


def expected_accepted(p: float, k: int) -> float:
    """E[accepted draft tokens + 1 bonus/correction] with i.i.d. accept rate p.

    Same closed form as P23.13's expected_tau((p,)*k): 1 + p + p^2 + ... + p^k.
    Teaching chain; EAGLE-4 trees use path-product scores instead (paper §3.1).
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0, 1]")
    total, run = 1.0, 1.0
    for _ in range(k):
        run *= p
        total += run
    return total


def verify_cost(n: int, c0: float, c1: float) -> float:
    """Affine verify cost for n draft nodes: c0 + c1 * n (profiled table stand-in).

    At small batch, verify is nearly free of n (memory-bound), so c1 is small.
    At large batch, compute is saturated, so c1 grows — the batch-size cliff.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    return c0 + c1 * n


def draft_cost(k: int, c_d: float) -> float:
    """Cost of drafting k chain tokens: k * c_d."""
    if k < 0:
        raise ValueError("k must be >= 0")
    return k * c_d


def round_cost(k: int, c_d: float, c0: float, c1: float) -> float:
    return draft_cost(k, c_d) + verify_cost(k, c0, c1)


def expected_speedup(p: float, k: int, c_d: float, c0: float, c1: float) -> float:
    """Tokens per unit cost vs ordinary decode (k=0: one token, cost = c0)."""
    base = verify_cost(0, c0, c1)  # one target forward, no draft
    if base <= 0:
        raise ValueError("baseline verify cost must be positive")
    return (expected_accepted(p, k) / round_cost(k, c_d, c0, c1)) / (1.0 / base)


def choose_draft_length(p: float, c_d: float, c0: float, c1: float, k_max: int = 8,
                        tau: float = 1.0) -> int:
    """Pick chain length k by the paper's ΔA/ΔC ≥ τ rule (teaching chain form).

    Paper §3.1 / Algorithm 1: after each draft step, stop when
    ΔA / c_draft < τ_d. Here ΔA = p^(k) (the marginal expected accept from
    adding one more draft token before the bonus), ΔC = c_d, and we also
    require the whole-round expected speedup not to fall below 1.
    Returns the largest k in 0..k_max that passes; 0 means "do not draft".
    """
    best = 0
    for k in range(1, k_max + 1):
        delta_a = p ** k          # marginal gain of the k-th draft token
        delta_c = c_d
        if delta_c <= 0:
            raise ValueError("c_d must be positive")
        if delta_a / delta_c < tau:
            break
        if expected_speedup(p, k, c_d, c0, c1) < 1.0:
            break
        best = k
    return best


def choose_across_batches(p: float, c_d: float, batch_costs: list[tuple[str, float, float]],
                          k_max: int = 8, tau: float = 1.0) -> list[tuple[str, int, float]]:
    """For each (label, c0, c1), return (label, chosen k, expected speedup)."""
    out = []
    for label, c0, c1 in batch_costs:
        k = choose_draft_length(p, c_d, c0, c1, k_max=k_max, tau=tau)
        sp = expected_speedup(p, k, c_d, c0, c1)
        out.append((label, k, sp))
    return out


def low_rank_factors(W: torch.Tensor, rank: int):
    """Truncated SVD factors of W (paper §3.2's natural init for A, B).

    W ≈ U[:, :r] @ diag(S[:r]) @ Vh[:r, :], so A = U[:, :r] * S[:r], B = Vh[:r].
    """
    # torch.linalg.svd on float64; full_matrices=False keeps (V, min) shapes.
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    A = U[:, :rank] * S[:rank]
    B = Vh[:rank, :]
    return A, B


def low_rank_candidate_head(W: torch.Tensor, h: torch.Tensor, rank: int, top_m: int,
                            generator: torch.Generator | None = None,
                            init: str = "svd"):
    """Teaching two-stage head: rank-r scores pick top_m, then exact W rows.

    Paper §3.2: z̃ = A(Bh), C = TopK(z̃, K), then z_C = W_C h.
    Default init is truncated SVD of W (paper's natural init). Pass init='random'
    for an untrained baseline. Real EAGLE-4 further trains A, B for retrieval.
    Returns (exact_top1, two_stage_top1, hit, full_muls, two_stage_muls).
    """
    V, d = W.shape
    if not (1 <= rank <= min(V, d)):
        raise ValueError("rank out of range")
    if not (1 <= top_m <= V):
        raise ValueError("top_m out of range")
    if init == "svd":
        A, B = low_rank_factors(W, rank)
    elif init == "random":
        if generator is None:
            raise ValueError("random init needs a generator")
        B = torch.randn(rank, d, generator=generator, dtype=W.dtype) / (d ** 0.5)
        A = torch.randn(V, rank, generator=generator, dtype=W.dtype) / (rank ** 0.5)
    else:
        raise ValueError("init must be 'svd' or 'random'")
    z_approx = A @ (B @ h)                         # V·r + r·d
    cand = torch.topk(z_approx, top_m).indices
    z_exact_cand = W[cand] @ h                     # top_m · d
    two_stage_top1 = cand[z_exact_cand.argmax()].item()
    exact_logits = W @ h                           # V · d
    exact_top1 = int(exact_logits.argmax().item())
    hit = two_stage_top1 == exact_top1
    full_muls = V * d
    two_stage_muls = V * rank + rank * d + top_m * d
    return exact_top1, two_stage_top1, hit, full_muls, two_stage_muls


def low_rank_hit_rate(vocab: int = 5000, d: int = 64, rank: int = 32, top_m: int = 128,
                      trials: int = 200, seed: int = 46) -> tuple[float, int, int]:
    """Fraction of trials where two-stage top-1 matches full-head top-1."""
    g = torch.Generator().manual_seed(seed)
    hits = 0
    full_muls = two_muls = 0
    for i in range(trials):
        g_i = torch.Generator().manual_seed(seed + 1000 + i)
        W = torch.randn(vocab, d, generator=g_i, dtype=torch.float64)
        h = torch.randn(d, generator=g_i, dtype=torch.float64)
        _, _, hit, fm, tm = low_rank_candidate_head(W, h, rank, top_m, generator=g_i)
        hits += int(hit)
        full_muls, two_muls = fm, tm
    return hits / trials, full_muls, two_muls


def train_tiny_target(steps: int = 80, seed: int = 460):
    """Reuse P23.13's tiny pattern model so greedy_decode stays lossless."""
    torch.manual_seed(seed)
    # Import BlockDraft's pattern through 24b to avoid duplicating the rule.
    spec = importlib.util.spec_from_file_location(
        "zero2llm_spec_drafts", Path(__file__).with_name("24b_spec_drafts.py"))
    drafts = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = drafts
    spec.loader.exec_module(drafts)
    return drafts.train_target(steps=steps, seed=seed)


class FixedProbDraft:
    """Drafter that proposes the target's own greedy tokens (perfect draft).

    Used only to show that cost-chosen k still keeps greedy_decode identical
    to the no-draft baseline. Not an EAGLE-4 drafter.
    """

    def __init__(self, target, tokens: torch.Tensor):
        self.target = target
        self.tokens = tokens  # full greedy continuation including prompt
        self.calls = 0

    @torch.no_grad()
    def __call__(self, prefix, limit):
        self.calls += 1
        n = prefix.shape[1]
        # Propose the next `limit` tokens from the precomputed greedy sequence.
        end = min(n + limit, self.tokens.shape[1])
        if end <= n:
            return prefix[:, :0]
        return self.tokens[:, n:end]


def demo():
    torch.set_num_threads(1)
    p = 0.7
    c_d = 0.15
    # Simulated batch regimes: (label, c0, c1). Small batch → tiny marginal verify cost.
    batches = [
        ("batch≈1 (memory-bound)", 1.0, 0.05),
        ("batch≈8", 1.0, 0.25),
        ("batch≈64 (compute-bound)", 1.0, 1.20),
    ]
    print("Cost-model chain length (teaching; paper uses trees + profiled tables):")
    print(f"  p={p}, c_d={c_d}, tau=1.0")
    for label, k, sp in choose_across_batches(p, c_d, batches):
        print(f"  {label}: k={k}, expected speedup≈{sp:.2f}×")

    rate, full_muls, two_muls = low_rank_hit_rate()
    print(f"Low-rank candidate head (vocab=5000, d=64, rank=32, top_m=128, 200 trials):")
    print(f"  top-1 match rate={rate:.3f}; mul counts full={full_muls} vs two-stage={two_muls} "
          f"({full_muls / two_muls:.2f}× fewer)")
    print("  Teaching SVD init of W; paper reports SVD 91.5% / trained 98.2% top-1-in-candidates "
          "at rank 128 (author report, not reproduced).")

    target = train_tiny_target()
    prompt = torch.tensor([[2, 7]])
    base = s6.greedy_decode(target, prompt, 10)
    k = choose_draft_length(p, c_d, 1.0, 0.05)
    draft = FixedProbDraft(target, base.tokens)
    out = s6.greedy_decode(target, prompt, 10, draft=draft, draft_tokens=k)
    assert torch.equal(out.tokens, base.tokens)
    print(f"Greedy verify with cost-chosen k={k}: identical to baseline "
          f"(target calls {out.target_calls} vs {base.target_calls}).")
    print("Lossless on greedy only; no 27B speedup claimed; source-available license noted.")


if __name__ == "__main__":
    demo()
