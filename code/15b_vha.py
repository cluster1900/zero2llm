"""P14 extra: VHA (virtual heads) training form vs absorbed GQA form, CPU float64.

Run: OMP_NUM_THREADS=1 python code/15b_vha.py
Source: arXiv 2610.17959 (submitted 2026-10-13). Teaching reconstruction of the
paper's absorption identities only; no training, no checkpoint, no kernel.
Shapes: Hq real query heads; each spawns one virtual head per KV head (G=Hkv),
so Hv = Hq*G virtual heads attend through a plain GQA cache with Hkv heads.
"""
import sys

try:
    import torch
except ModuleNotFoundError as exc:
    if __name__ == "__main__" and exc.name == "torch":
        print("P14 VHA requires PyTorch. Install dependencies: python -m pip install -r requirements.txt", file=sys.stderr)
        raise SystemExit(2) from None
    raise


def make_params(d_model=16, hq=4, hkv=2, d=4, alpha=0.1, seed=0, postmix_b_zero=False):
    """Premix init I + alpha*eps (eps ~ N(0,1)); Postmix A random, B = 0 at init."""
    if hq <= 0 or hkv <= 0 or d <= 0:
        raise ValueError("head counts and head dim must be positive")
    g = torch.Generator().manual_seed(seed)
    dt = torch.float64
    hv = hq * hkv
    r = max(1, hq // 2)  # paper: r = H_q/2, half the PHYSICAL query heads
    p = {
        "hq": hq, "hkv": hkv, "d": d, "r": r,
        "WQ": torch.randn(hq, d_model, d, generator=g, dtype=dt) / d_model ** 0.5,
        "WK": torch.randn(hkv, d_model, d, generator=g, dtype=dt) / d_model ** 0.5,
        "WV": torch.randn(hkv, d_model, d, generator=g, dtype=dt) / d_model ** 0.5,
        "premix": torch.eye(d, dtype=dt) + alpha * torch.randn(hkv, d, d, generator=g, dtype=dt),
        "A": torch.randn(hv, r, generator=g, dtype=dt) / r ** 0.5,
        "B": torch.zeros(hv, r, dtype=dt) if postmix_b_zero else torch.randn(hv, r, generator=g, dtype=dt) * 0.1,
        "WO": torch.randn(hv, d, d_model, generator=g, dtype=dt) / (hv * d) ** 0.5,
    }
    return p


def _attend(q, k, v):
    """q [Hv,T,d], k/v [Hkv,T,d]; virtual head h reads KV head h % Hkv."""
    hv, t, d = q.shape
    idx = torch.arange(hv) % k.shape[0]
    scores = q @ k[idx].transpose(-1, -2) / d ** 0.5
    mask = torch.ones(t, t, dtype=torch.bool).triu(1)
    return torch.softmax(scores.masked_fill(mask, float("-inf")), -1) @ v[idx]


def training_form(x, p):
    """Explicit premix on each real query head, attention, explicit Postmix Y'=(I+BA^T)Y, then W_O."""
    hq, hkv = p["hq"], p["hkv"]
    q_real = torch.einsum("tm,hmd->htd", x, p["WQ"])
    q = torch.stack([q_real[i] @ p["premix"][g] for i in range(hq) for g in range(hkv)])
    k = torch.einsum("tm,hmd->htd", x, p["WK"])
    v = torch.einsum("tm,hmd->htd", x, p["WV"])
    y = _attend(q, k, v)                                   # [Hv,T,d]
    mix = torch.eye(y.shape[0], dtype=x.dtype) + p["B"] @ p["A"].T
    y2 = torch.einsum("hk,ktd->htd", mix, y)
    return torch.einsum("htd,hdm->tm", y2, p["WO"])


def absorb(p):
    """W~_Q^(i,g) = W_Q^(i) W_g^premix ; W_O' = ((I + A B^T) kron I_d) W_O."""
    hq, hkv, d = p["hq"], p["hkv"], p["d"]
    wq = torch.stack([p["WQ"][i] @ p["premix"][g] for i in range(hq) for g in range(hkv)])
    hv = wq.shape[0]
    mix_t = torch.eye(hv, dtype=wq.dtype) + p["A"] @ p["B"].T
    wo = (torch.kron(mix_t, torch.eye(d, dtype=wq.dtype)) @ p["WO"].reshape(hv * d, -1)).reshape(hv, d, -1)
    return {"WQ": wq, "WK": p["WK"], "WV": p["WV"], "WO": wo}


def gqa_form(x, w):
    """Plain GQA: Hv query heads, Hkv KV heads, no mixing at inference."""
    q = torch.einsum("tm,hmd->htd", x, w["WQ"])
    k = torch.einsum("tm,hmd->htd", x, w["WK"])
    v = torch.einsum("tm,hmd->htd", x, w["WV"])
    return torch.einsum("htd,hdm->tm", _attend(q, k, v), w["WO"])


def kv_cache_bytes(heads, d, tokens, layers=1, bytes_per=2):
    return 2 * heads * d * tokens * layers * bytes_per


def main():
    torch.manual_seed(0)
    p = make_params()
    x = torch.randn(5, 16, dtype=torch.float64)
    err = (training_form(x, p) - gqa_form(x, absorb(p))).abs().max().item()
    assert err < 1e-5, err
    p0 = make_params(postmix_b_zero=True)
    q_mix = torch.eye(p0["hq"] * p0["hkv"], dtype=torch.float64) + p0["B"] @ p0["A"].T
    assert torch.equal(q_mix, torch.eye(q_mix.shape[0], dtype=torch.float64))
    print(f"virtual heads Hv={p['hq'] * p['hkv']} over Hkv={p['hkv']} KV heads, postmix rank r={p["r"]} (= Hq/2)")
    print(f"training form vs absorbed GQA-{p['hkv']}: max |diff| = {err:.3e} (< 1e-5)")
    print("B = 0  =>  I + B A^T == I  (Postmix is identity at init)")
    hv = p["hq"] * p["hkv"]
    for name, heads in (("MHA (Hv KV heads)", hv), (f"VHA / GQA-{p['hkv']}", p["hkv"])):
        print(f"KV cache per token per layer, bf16, d={p['d']}: {name}: {kv_cache_bytes(heads, p['d'], 1)} bytes")


if __name__ == "__main__":
    main()
