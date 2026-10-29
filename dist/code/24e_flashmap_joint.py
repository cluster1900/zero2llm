"""P23 extra: FlashMap teaching toy — joint vs independent block drafts (CPU).

Run: OMP_NUM_THREADS=1 python code/24e_flashmap_joint.py
Teaching reconstruction, NOT the paper's DiT / flow-map weights. Sources
checked 2026-10-29:
  Paper: FlashMap: Flow Map Speculative Decoding, arXiv:2610.24365v1
    (2026-10-17): §3 continuous flow drafter + distillation; §6.2 shuffle test
    that separates per-slot marginals from joint structure; Figure 1(c) / Table 1
    / Table 2 numbers are author-reported (not reproduced here).
  Code: al-jshen/flashmap@cd5b45bdf19b1d81ee00bf4c2f212c7c0ed93ac6 (MIT);
    frozen target embedding + LM head; DiT cross-attends to target hidden
    states; num_steps=1 (1 NFE) or 2.
  Weights (Apache-2.0): j-shen/FlashMap-Qwen3-8B, j-shen/FlashMap-Qwen3-4B.
All speedup / acceptance figures labelled "论文自报" in the chapter; this
script only demonstrates why joint structure raises mean acceptance length,
and that greedy verify still matches ordinary greedy (lossless).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

try:
    import torch
except ModuleNotFoundError as exc:
    if __name__ == "__main__" and exc.name == "torch":
        print(
            "P23 requires PyTorch. Install dependencies: "
            "python -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        raise SystemExit(2) from None
    raise

_NAME = "zero2llm_s6"
if _NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        _NAME, Path(__file__).with_name("24_reasoning_mtp.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[_NAME] = module
    spec.loader.exec_module(module)
s6 = sys.modules[_NAME]

# Two-mode toy mirroring the paper's "new york / san diego" joint intuition
# (FlashMap Fig.1(b) / §6.2): modes are equally likely; independent drafting
# can cross modes and get rejected early.
MODE_A = (0, 1)  # "new", "york"
MODE_B = (2, 3)  # "san", "diego"
MODES = (MODE_A, MODE_B)
K = 2
VOCAB = 4


def sample_joint_block(n: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Sample n blocks from the joint: pick a mode, emit both tokens."""
    mode = torch.randint(0, len(MODES), (n,), generator=generator)
    out = torch.empty(n, K, dtype=torch.long)
    for i, m in enumerate(mode.tolist()):
        out[i, 0] = MODES[m][0]
        out[i, 1] = MODES[m][1]
    return out


def sample_independent_block(n: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Factorized draft: each slot from its marginal (DFlash-like independence).

    Slot 0 marginal = {0, 2}; slot 1 marginal = {1, 3}. Cross-mode pairs
    ("new","diego") and ("san","york") each have probability 1/4.
    """
    slot0 = torch.tensor([MODE_A[0], MODE_B[0]])
    slot1 = torch.tensor([MODE_A[1], MODE_B[1]])
    i0 = torch.randint(0, 2, (n,), generator=generator)
    i1 = torch.randint(0, 2, (n,), generator=generator)
    return torch.stack([slot0[i0], slot1[i1]], dim=1)


def sample_shuffled_block(n: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Shuffle test stand-in: independent mode draw per slot (same marginals).

    Paper §6.2: shared noise across slots vs independent noise per slot.
    Here each slot picks a mode independently, then takes that mode's token
    at its position — marginals unchanged, joint structure destroyed.
    """
    return sample_independent_block(n, generator=generator)


def accept_prefix_length(draft: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Left-to-right accept until first mismatch; length in [0, K].

    Speculative decoding also adds a bonus/correction token from the target;
    this toy only counts matching draft slots (paper's γ includes +1 bonus —
    see Leviathan et al.; we keep the draft-only length for the shuffle Δ).
    """
    if draft.shape != target.shape:
        raise ValueError("draft and target must share shape")
    match = draft == target
    # First False ends the prefix; all-True → K.
    # cummin on bool: once False, stays False.
    keep = match.cummin(dim=1).values
    return keep.sum(dim=1).to(torch.float64)


def mean_accept(
    draft_fn,
    n: int = 20000,
    seed: int = 0,
) -> float:
    """Mean accepted draft length vs a joint-sampled target continuation."""
    gen_t = torch.Generator().manual_seed(seed)
    gen_d = torch.Generator().manual_seed(seed + 1)
    target = sample_joint_block(n, generator=gen_t)
    draft = draft_fn(n, generator=gen_d)
    return float(accept_prefix_length(draft, target).mean().item())


def shuffle_delta(n: int = 20000, seed: int = 7) -> tuple[float, float, float]:
    """γ_shared, γ_shuffled, Δ — teaching stand-in for paper Table 2."""
    g_shared = mean_accept(sample_joint_block, n=n, seed=seed)
    g_shuf = mean_accept(sample_shuffled_block, n=n, seed=seed)
    return g_shared, g_shuf, g_shared - g_shuf


class JointBlockDraft:
    """One call → whole block from the joint (FlashMap-like interface only)."""

    def __init__(self, truth_tokens: torch.Tensor, noise: float = 0.0, seed: int = 0):
        # truth_tokens: (1, T) full greedy continuation used to build a
        # correlated draft that is mostly right; noise flips whole modes.
        self.truth = truth_tokens
        self.noise = noise
        self.gen = torch.Generator().manual_seed(seed)
        self.calls = 0

    @torch.no_grad()
    def __call__(self, prefix: torch.Tensor, limit: int) -> torch.Tensor:
        self.calls += 1
        t = prefix.shape[1]
        # Take the next `limit` truth tokens; with prob `noise` replace the
        # whole block by the other mode (keeps joint, may reject early).
        block = self.truth[:, t : t + limit].clone()
        if limit >= K and self.noise > 0 and torch.rand(1, generator=self.gen).item() < self.noise:
            # Flip to the other mode's tokens for the first K slots.
            cur = tuple(block[0, :K].tolist())
            other = MODE_B if cur == MODE_A else MODE_A
            for i in range(min(K, limit)):
                block[0, i] = other[i]
        return block


def train_tiny_target(steps: int = 80, seed: int = 106):
    """Reuse S6 pattern target so greedy_decode stays the verifier."""
    torch.manual_seed(seed)
    s5 = s6.s5
    config = s5.MoEConfig(
        vocab_size=11,
        d_model=16,
        num_heads=4,
        num_kv_heads=2,
        num_layers=1,
        max_seq_len=32,
        expert_width=7,
        shared_width=5,
    )
    model = s6.MTPGPT(config, depth=3)
    opt = torch.optim.Adam(model.parameters(), lr=3e-2)
    gen = torch.Generator().manual_seed(seed)

    def pattern_batch(n, length, vocab=11):
        start = torch.randint(0, vocab, (n, 1), generator=gen)
        rows = [start]
        for _ in range(length - 1):
            rows.append((3 * rows[-1] + 1) % vocab)
        return torch.cat(rows, 1)

    for _ in range(steps):
        loss = model.forward_mtp(pattern_batch(16, 12), weight=1.0).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


def main() -> None:
    torch.set_num_threads(1)
    print("FlashMap teaching toy (joint vs independent block drafts)")
    print("Paper numbers are NOT measured here — see P23.16 (论文自报).")
    print()

    g_ind = mean_accept(sample_independent_block, n=40000, seed=1)
    g_joint = mean_accept(sample_joint_block, n=40000, seed=1)
    g_s, g_f, delta = shuffle_delta(n=40000, seed=2)
    print(f"mean accept length  independent (factorized): {g_ind:.3f}")
    print(f"mean accept length  joint (shared mode):     {g_joint:.3f}")
    print(f"shuffle test        γ_shared={g_s:.3f}  γ_shuffled={g_f:.3f}  Δ={delta:.3f}")
    print("  (paper Table 2: DFlash Δ=0 by construction; FlashMap 2-step recovers most Δ)")
    assert g_joint > g_ind + 0.2, "joint should beat independent on this toy"
    assert delta > 0.2, "shared-noise joint should beat shuffled"

    # Lossless greedy verify with a joint-style block draft.
    target = train_tiny_target()
    prompt = torch.tensor([[2, 7]])
    base = s6.greedy_decode(target, prompt, 10)
    draft = JointBlockDraft(base.tokens, noise=0.25, seed=3)
    out = s6.greedy_decode(target, prompt, 10, draft=draft, draft_tokens=K)
    assert torch.equal(out.tokens, base.tokens), "verify must keep greedy output"
    print(
        f"greedy verify: identical to baseline "
        f"(target_calls {out.target_calls} vs {base.target_calls}; "
        f"draft_calls={draft.calls})"
    )
    print("OK")


if __name__ == "__main__":
    main()
