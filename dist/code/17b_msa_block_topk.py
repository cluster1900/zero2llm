"""P16 extra: MSA-style block top-k selection and index recall (CPU float64).

Run: OMP_NUM_THREADS=1 python code/17b_msa_block_topk.py
Source: MSA, arXiv 2606.13392 (submitted 2026-06-11). Teaching reconstruction of
the selection rule only (max-pooled block scores, per-GQA-group top-k, local block
always kept). The paper's learned indexer, KL distillation and kernels are NOT
implemented; here the "indexer" is the exact q.k score, so recall measures only
the block-granularity loss. No speed or quality claim.
"""
import sys

try:
    import torch
except ModuleNotFoundError as exc:
    if __name__ == "__main__" and exc.name == "torch":
        print("P16 requires PyTorch. Install dependencies: python -m pip install -r requirements.txt", file=sys.stderr)
        raise SystemExit(2) from None
    raise


def block_scores(q, k, block):
    """q [Hq,d] for ONE query position, k [T,d]. Max-pool token scores inside each block -> [Hq, nblocks]."""
    t = k.shape[0]
    if block <= 0:
        raise ValueError("block must be positive")
    s = q @ k.T
    pad = (-t) % block
    s = torch.nn.functional.pad(s, (0, pad), value=float("-inf"))
    return s.reshape(q.shape[0], -1, block).amax(-1)


def select_blocks(q, k, block, topk, groups):
    """Average block scores inside each GQA group, keep top-k blocks + the local (last) block."""
    hq = q.shape[0]
    if hq % groups:
        raise ValueError("query heads must divide into groups")
    bs = block_scores(q, k, block).reshape(groups, hq // groups, -1).mean(1)
    n = bs.shape[-1]
    chosen = []
    for g in range(groups):
        local = n - 1
        rest = bs[g].clone()
        rest[local] = float("-inf")
        pick = rest.topk(min(topk - 1, n - 1)).indices.tolist() if topk > 1 else []
        chosen.append(sorted(set(pick) | {local}))
    return chosen


def token_recall(q, k, block, topk, groups, top_tokens):
    """Fraction of each head's exact top-`top_tokens` keys that fall in its group's selected blocks."""
    chosen = select_blocks(q, k, block, topk, groups)
    per = q.shape[0] // groups
    hits = total = 0
    for h in range(q.shape[0]):
        blocks = set(chosen[h // per])
        best = (q[h] @ k.T).topk(top_tokens).indices
        hits += sum(int(i) // block in blocks for i in best)
        total += top_tokens
    return hits / total


def attention_flops_ratio(t, block, topk):
    """Dense attention reads t keys per query; selection reads at most topk*block."""
    return t / min(t, topk * block)


def main():
    torch.manual_seed(0)
    t, d, hq, groups, block = 4096, 32, 8, 2, 128
    k = torch.randn(t, d, dtype=torch.float64)
    planted = torch.randint(0, t - block, (6,))
    q = torch.randn(hq, d, dtype=torch.float64) * 0.2
    q += k[planted].mean(0)       # queries resemble a few far-away keys
    for topk in (1, 4, 16):
        r = token_recall(q, k, block, topk, groups, 16)
        print(f"T={t} B_k={block} k={topk}: top-16 token recall={r:.3f}; key reads ratio={attention_flops_ratio(t, block, topk):.1f}x")
    print(f"paper setting at 1M tokens, B_k=128, k=16: key reads ratio={attention_flops_ratio(1 << 20, 128, 16):.0f}x (selection only; not the paper's 28.4x FLOPs figure)")


if __name__ == "__main__":
    main()
