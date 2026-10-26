"""P24 extra: FP4 (E2M1) with per-tensor, MXFP4 and NVFP4-style block scales.

Run: python code/25c_fp4_block.py   (pure Python, no PyTorch needed)
Sources: MX formats paper, arXiv:2310.10537, Table 1 and Algorithm 1
(MXFP4 = 32-element blocks, E8M0 power-of-two scale, E2M1 elements);
NVIDIA Transformer Engine NVFP4 docs (16-element blocks, E4M3 block scale,
one FP32 per-tensor scale). Simulation in float64 on made-up data. It checks
the rounding arithmetic only and says nothing about any model's accuracy.
"""
import math
import random

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
FP4_MAX = 6.0
E4M3_MAX = 448.0
E2M1_EMAX = 2  # 6 = 1.5 * 2**2: exponent of the largest normal E2M1 value


def round_e2m1(x):
    """Nearest E2M1 value, saturating at +-6 (ties go to the smaller magnitude)."""
    a = min(abs(x), FP4_MAX)
    best = min(E2M1, key=lambda g: (abs(g - a), g))
    return math.copysign(best, x)


def round_e4m3(x):
    """Round a positive scale to FP8 E4M3 (3 mantissa bits, min normal 2^-6, max 448)."""
    if x <= 0:
        return 0.0
    x = min(x, E4M3_MAX)
    e = max(math.floor(math.log2(x)), -6)
    step = 2.0 ** (e - 3)
    return min(round(x / step) * step, E4M3_MAX)


def quant_per_tensor(xs):
    s = max(abs(v) for v in xs) / FP4_MAX
    return [round_e2m1(v / s) * s for v in xs] if s > 0 else list(xs)


def quant_mxfp4(xs, block=32):
    """Algorithm 1: X = 2^(floor(log2 amax) - emax_elem), P = quantize(V / X), clamped."""
    out = []
    for i in range(0, len(xs), block):
        blk = xs[i:i + block]
        amax = max(abs(v) for v in blk)
        if amax == 0:
            out += [0.0] * len(blk)
            continue
        X = 2.0 ** (math.floor(math.log2(amax)) - E2M1_EMAX)
        out += [round_e2m1(v / X) * X for v in blk]
    return out


def quant_nvfp4(xs, block=16):
    """x = e2m1 * s_block * s_global, s_block stored in E4M3 (TE docs formulas)."""
    gmax = max(abs(v) for v in xs)
    s_global = gmax / (E4M3_MAX * FP4_MAX)
    out = []
    for i in range(0, len(xs), block):
        blk = xs[i:i + block]
        amax = max(abs(v) for v in blk)
        s_block = round_e4m3((amax / FP4_MAX) / s_global) if s_global > 0 else 0.0
        s = s_block * s_global
        out += [round_e2m1(v / s) * s if s > 0 else 0.0 for v in blk]
    return out


def rel_mse(xs, ys):
    return sum((a - b) ** 2 for a, b in zip(xs, ys)) / sum(a * a for a in xs)


def make_data(n=1024, seed=0, outliers=4, size=40.0):
    """Gaussian values with a few large outliers, like a weight row with spikes."""
    rnd = random.Random(seed)
    xs = [rnd.gauss(0, 1) for _ in range(n)]
    for j in rnd.sample(range(n), outliers):
        xs[j] = math.copysign(size, xs[j])
    return xs


def demo():
    xs = make_data()
    for name, f in (("per-tensor", quant_per_tensor), ("MXFP4 b=32", quant_mxfp4),
                    ("NVFP4 b=16", quant_nvfp4)):
        print(f"{name:>11}: relative MSE = {rel_mse(xs, f(xs)):.4f}")
    print("bits/element: MXFP4 4+8/32 =", 4 + 8 / 32, " NVFP4 4+8/16 =", 4 + 8 / 16)


if __name__ == "__main__":
    demo()
