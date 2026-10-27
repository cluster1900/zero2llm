"""P11 extra: cache engineering on a tiny RoPE model, drop + repos vs re-prefill (CPU).

Run: OMP_NUM_THREADS=1 python code/12b_cache_engineering.py
Source: Fu et al., "Cache Engineering for Agents", arXiv:2610.28577v1,
Sec. 3.3 Eq. 4 (drop) and Eq. 5 (repos: K <- R(q) R(p)^-1 K, values unchanged),
Sec. 3.4 and Fig. 5 (history-dependent cache states). Code: github.com/thu-nics/AKV.
This file is our own single-sequence reimplementation of the two operations on
the P12 RopeGPT; nothing is copied from AKV, and AKV's serving system (radix tree,
eviction, restore) is not implemented. The model is randomly initialised, so the
numbers show mechanics, not quality.
"""
import importlib.util
import math
from pathlib import Path
import sys

try:
    import torch
except ModuleNotFoundError as exc:
    if __name__ == "__main__" and exc.name == "torch":
        print("P11 extra requires PyTorch. Install dependencies: python -m pip install -r requirements.txt", file=sys.stderr)
        raise SystemExit(2) from None
    raise

_NAME = "zero2llm_s2"
if _NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(_NAME, Path(__file__).with_name("13_rope.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[_NAME] = module
    spec.loader.exec_module(module)
s2 = sys.modules[_NAME]
apply_rope = s2.apply_rope

SEED = 2026
TOKENS = [5, 11, 3, 17, 8, 2, 14, 9, 6, 12]   # logical indices 0..9
DROPPED = [2, 3, 4, 5]                         # a middle span, e.g. an old tool result
QUERY = [7, 1]                                 # tokens appended after the cache operations


def make_model(num_layers, seed=SEED):
    torch.manual_seed(seed)
    config = s2.RopeConfig(vocab_size=20, d_model=16, num_heads=2, num_layers=num_layers, max_seq_len=32)
    return s2.RopeGPT(config).double().eval()


def rerotate(k, old_positions, new_positions, base=10000.0, rotary_dim=None):
    """repos for one key tensor [..., T, D]: rotate by (new - old), i.e. R(new) R(old)^-1.

    Two rotations of the same channel pair add their angles, so undoing angle
    old*theta and applying new*theta is one rotation by (new-old)*theta.
    """
    dim = k.shape[-1] if rotary_dim is None else rotary_dim
    frequencies = base ** (-torch.arange(0, dim, 2, dtype=k.dtype) / dim)
    angles = (new_positions - old_positions).to(k.dtype)[:, None] * frequencies[None, :]
    c, s = angles.cos(), angles.sin()
    even, odd = k[..., :dim:2], k[..., 1:dim:2]
    rotated = torch.stack((even * c - odd * s, even * s + odd * c), dim=-1).flatten(-2)
    return torch.cat((rotated, k[..., dim:]), dim=-1)


class ActiveCache:
    """One request's KV plus the two bookkeeping columns the paper adds.

    layers: per layer (K, V), each [1, H, T_logical, d]; K is already rotated.
    active: bool [T_logical], False after drop (masked out, storage kept).
    key_pos: position currently encoded in each K; next_pos: index for the next append.
    """
    def __init__(self, layers, active, key_pos, next_pos):
        self.layers, self.active, self.key_pos, self.next_pos = layers, active, key_pos, next_pos


@torch.no_grad()
def append(model, cache, tokens):
    """append (Eq. 3): run new tokens over the ACTIVE cache, store their K/V. Returns logits."""
    idx = torch.tensor([tokens])
    T = idx.shape[1]
    q_pos = torch.arange(cache.next_pos if cache else 0, (cache.next_pos if cache else 0) + T)
    x = model.tok_emb(idx)
    new_layers = []
    for i, block in enumerate(model.blocks):
        attn = block.attn
        h = block.ln1(x)
        q, k, v = (w(h).view(1, T, attn.num_heads, attn.head_dim).transpose(1, 2)
                   for w in (attn.W_q, attn.W_k, attn.W_v))
        q = apply_rope(q, q_pos, attn.rope_base, attn.rotary_dim)
        k = apply_rope(k, q_pos, attn.rope_base, attn.rotary_dim)
        if cache is None:
            old_k = old_v = k[:, :, :0]
            visible_old = torch.zeros(T, 0, dtype=torch.bool)
        else:
            old_k, old_v = cache.layers[i]
            visible_old = cache.active[None, :].expand(T, -1)          # drop = mask, Eq. 4
        visible_new = torch.ones(T, T, dtype=torch.bool).tril()         # causal inside the chunk
        allowed = torch.cat((visible_old, visible_new), dim=1)
        all_k, all_v = torch.cat((old_k, k), 2), torch.cat((old_v, v), 2)
        scores = q @ all_k.transpose(-2, -1) / math.sqrt(attn.head_dim)
        weights = scores.masked_fill(~allowed, float("-inf")).softmax(-1)
        x = x + attn.W_o((weights @ all_v).transpose(1, 2).reshape(1, T, -1))
        x = x + block.mlp(block.ln2(x))
        new_layers.append((all_k, all_v))
    old_active = cache.active if cache else torch.zeros(0, dtype=torch.bool)
    old_pos = cache.key_pos if cache else torch.zeros(0, dtype=torch.long)
    out = ActiveCache(new_layers, torch.cat((old_active, torch.ones(T, dtype=torch.bool))),
                      torch.cat((old_pos, q_pos)), int(q_pos[-1]) + 1)
    return model.head(model.ln_f(x)), out


def drop(cache, indices):
    """drop(P): exclude logical indices from later attention; positions stay as they are."""
    active = cache.active.clone()
    active[list(indices)] = False
    return ActiveCache(cache.layers, active, cache.key_pos, cache.next_pos)


def repos(model, cache):
    """repos: active keys get positions 0..r-1 in logical order; V copied; next index r."""
    kept = cache.active.nonzero().flatten()
    new_pos = cache.key_pos.clone()
    new_pos[kept] = torch.arange(len(kept))
    layers = []
    for (k, v), block in zip(cache.layers, model.blocks):
        k = k.clone()
        k[:, :, kept] = rerotate(k[:, :, kept], cache.key_pos[kept], new_pos[kept],
                                 block.attn.rope_base, block.attn.rotary_dim)
        layers.append((k, v))
    return ActiveCache(layers, cache.active, new_pos, len(kept))


def active_kv(cache):
    keep = cache.active
    return [(k[:, :, keep], v[:, :, keep]) for k, v in cache.layers]


def compare(num_layers, dropped=DROPPED, use_repos=True):
    """Return max |difference| per layer for active K, V and for the query logits."""
    model = make_model(num_layers)
    kept_tokens = [t for i, t in enumerate(TOKENS) if i not in set(dropped)]
    # Path A: full prefill, then drop (+ repos), then append the query.
    _, cache = append(model, None, TOKENS)
    cache = drop(cache, dropped)
    if use_repos:
        cache = repos(model, cache)
    edited = active_kv(cache)
    logits_a, _ = append(model, cache, QUERY)
    # Path B: context engineering, re-prefill only the kept text, then append the query.
    _, fresh = append(model, None, kept_tokens)
    logits_b, _ = append(model, fresh, QUERY)
    rows = []
    for (ka, va), (kb, vb) in zip(edited, active_kv(fresh)):
        rows.append(((ka - kb).abs().max().item(), (va - vb).abs().max().item()))
    return rows, (logits_a - logits_b).abs().max().item()


def per_token_gap(num_layers=2, layer=1, dropped=DROPPED):
    """max |dK| per kept token at one layer: which kept tokens carry dropped context?"""
    model = make_model(num_layers)
    kept = [i for i in range(len(TOKENS)) if i not in set(dropped)]
    _, cache = append(model, None, TOKENS)
    edited = active_kv(repos(model, drop(cache, dropped)))[layer][0]
    _, fresh = append(model, None, [TOKENS[i] for i in kept])
    fresh_k = fresh.layers[layer][0]
    return {i: (edited[:, :, j] - fresh_k[:, :, j]).abs().max().item() for j, i in enumerate(kept)}


def main():
    print("P11 extra: drop + repos vs re-prefill (float64, seed", SEED, ")")
    print("tokens", TOKENS, "drop logical", DROPPED, "query", QUERY)
    for num_layers in (1, 2, 4):
        rows, logit_gap = compare(num_layers)
        print(f"\n{num_layers} layer(s), drop + repos:")
        for i, (dk, dv) in enumerate(rows):
            print(f"  layer {i}: max|dK| = {dk:.2e}  max|dV| = {dv:.2e}")
        print(f"  query logits max|diff| = {logit_gap:.2e}")
    gaps = per_token_gap()
    print("\n2 layers, layer 1 max|dK| per kept logical index:",
          ", ".join(f"{i}:{g:.1e}" for i, g in gaps.items()))
    rows, logit_gap = compare(1, use_repos=False)
    print(f"\n1 layer, drop WITHOUT repos: layer0 max|dK| = {rows[0][0]:.2e}, logits max|diff| = {logit_gap:.2e}")
    rows, logit_gap = compare(2, dropped=[8, 9])
    print(f"2 layers, drop the LAST two tokens + repos: max|dK|,|dV| = "
          f"{max(max(r) for r in rows):.2e}, logits max|diff| = {logit_gap:.2e}")


if __name__ == "__main__":
    main()
