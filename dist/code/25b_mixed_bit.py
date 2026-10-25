"""P24 extra: 4/6-bit group quantization error and a mixed-bit allocation (pure Python + torch CPU).

Run: OMP_NUM_THREADS=1 python code/25b_mixed_bit.py
Teaching format: symmetric signed codes in [-(2^(b-1)-1), 2^(b-1)-1] with one fp16
scale (2 bytes) per contiguous group. This is NOT llama.cpp's Q4_K/Q6_K block layout
(those use super-blocks with quantized sub-scales); it only shows the trade-off of
bit width vs group size and of giving the most sensitive tensor more bits.
"""
import sys

try:
    import torch
except ModuleNotFoundError as exc:
    if __name__ == "__main__" and exc.name == "torch":
        print("P24 requires PyTorch. Install dependencies: python -m pip install -r requirements.txt", file=sys.stderr)
        raise SystemExit(2) from None
    raise


def group_quant(w, bits, group):
    """Round-trip a 2D tensor; groups run along each row; returns reconstruction."""
    if bits < 2 or group <= 0 or w.ndim != 2 or w.shape[1] % group:
        raise ValueError("need bits>=2, positive group dividing the row length, 2D tensor")
    qmax = 2 ** (bits - 1) - 1
    g = w.reshape(w.shape[0], -1, group)
    scale = g.abs().amax(-1, keepdim=True) / qmax
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    codes = torch.clamp(torch.round(g / scale), -qmax, qmax)
    return (codes * scale).reshape_as(w)


def nbytes(shape, bits, group):
    n = shape[0] * shape[1]
    return n * bits // 8 + 2 * (n // group)


def rel_err(w, r):
    return ((w - r).norm() / w.norm()).item()


def make_tensors(seed=0):
    """Three toy tensors; 'v' and 'down' get heavy-tailed columns to be outlier-sensitive."""
    g = torch.Generator().manual_seed(seed)
    body = torch.randn(64, 256, generator=g, dtype=torch.float64)
    v = torch.randn(64, 256, generator=g, dtype=torch.float64)
    v[:, ::17] *= 12
    torch.manual_seed(seed)
    down = torch.distributions.StudentT(2.0).sample((64, 256)).to(torch.float64)
    return {"body": body, "v": v, "down": down}


def plan_error(tensors, plan):
    """plan: name -> (bits, group). Sum of squared errors over all tensors, and total bytes."""
    sq = sum(((w - group_quant(w, *plan[k])) ** 2).sum().item() for k, w in tensors.items())
    total = sum(w.pow(2).sum().item() for w in tensors.values())
    return (sq / total) ** 0.5, sum(nbytes(w.shape, *plan[k]) for k, w in tensors.items())


def main():
    torch.manual_seed(0)
    t = make_tensors()
    w = t["down"]
    for bits in (4, 6):
        for group in (32, 128):
            print(f"down {bits}-bit group {group}: rel err={rel_err(w, group_quant(w, bits, group)):.4f}; bytes={nbytes(w.shape, bits, group)}")
    uniform = {k: (4, 32) for k in t}
    errs = {k: rel_err(x, group_quant(x, 4, 32)) for k, x in t.items()}
    worst = max(errs, key=errs.get)
    mixed = dict(uniform, **{worst: (6, 128)})
    six = {k: (6, 128) for k in t}
    print("per-tensor 4-bit/g32 rel err:", {k: round(v, 4) for k, v in errs.items()}, "-> upgrade", worst)
    for name, plan in (("all 4-bit g32", uniform), (f"{worst} 6-bit g128, rest 4-bit g32", mixed), ("all 6-bit g128", six)):
        e, b = plan_error(t, plan)
        print(f"{name}: total rel err={e:.4f}; bytes={b}")


if __name__ == "__main__":
    main()
