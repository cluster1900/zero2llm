"""P12 / S2: adjacent-pair RoPE, explicit KV cache, and a CPU failure case.

Run: OMP_NUM_THREADS=1 python code/13_rope.py
No downloads. This implements ordinary RoPE, NOT a complete YaRN extension.
The optional --skip-training flag prints only deterministic arithmetic/cache demos.
"""
import argparse
from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path
import sys

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ModuleNotFoundError as exc:
    if __name__ == "__main__" and exc.name == "torch":
        print("P12 requires PyTorch. Install dependencies: python -m pip install -r requirements.txt", file=sys.stderr)
        raise SystemExit(2) from None
    raise

_NAME = "zero2llm_s1"
if _NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(_NAME, Path(__file__).with_name("12_kv_cache.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[_NAME] = module
    spec.loader.exec_module(module)
kv = sys.modules[_NAME]
s0 = kv.s0
KVCache = kv.KVCache
cache_bytes = kv.cache_bytes
expected_cache_bytes = kv.expected_cache_bytes


@dataclass(frozen=True)
class RopeConfig:
    vocab_size: int
    d_model: int = 48
    num_heads: int = 4
    num_layers: int = 2
    max_seq_len: int = 128
    rope_base: float = 10000.0
    rotary_dim: int | None = None

    def __post_init__(self):
        for name in ("vocab_size", "d_model", "num_heads", "num_layers", "max_seq_len"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        head_dim = self.d_model // self.num_heads
        dim = head_dim if self.rotary_dim is None else self.rotary_dim
        if type(dim) is not int or dim <= 0 or dim % 2 or dim > head_dim:
            raise ValueError("rotary_dim must be positive, even, and <= head_dim")
        if isinstance(self.rope_base, bool) or not isinstance(self.rope_base, (int, float)) or not math.isfinite(self.rope_base) or self.rope_base <= 0:
            raise ValueError("rope_base must be finite and positive")


def apply_rope(x, positions, base=10000.0, rotary_dim=None):
    """Rotate adjacent pairs of [..., T, D], return same shape/dtype.

    theta_i = base ** (-2*i/rotary_dim), i=0,...,rotary_dim/2-1.
    Only the first rotary_dim channels rotate; the remaining tail is untouched.
    Float64 inputs retain float64 angle accuracy. Half inputs compute angles and
    rotation in float32, then cast back. Position precision is still finite.
    Fractional positions are allowed for the standalone interpolation demo.
    """
    if not isinstance(x, torch.Tensor) or x.ndim < 2 or not x.is_floating_point():
        raise ValueError("x must be floating with shape [..., T, D]")
    dim = x.shape[-1] if rotary_dim is None else rotary_dim
    if type(dim) is not int or dim <= 0 or dim % 2 or dim > x.shape[-1]:
        raise ValueError("rotary_dim must be positive, even, and <= head_dim")
    if isinstance(base, bool) or not isinstance(base, (int, float)) or not math.isfinite(base) or base <= 0:
        raise ValueError("base must be finite and positive")
    if not isinstance(positions, torch.Tensor) or positions.ndim != 1 or positions.shape[0] != x.shape[-2]:
        raise ValueError("positions must be one-dimensional with length T")
    if positions.device != x.device or positions.dtype == torch.bool or positions.is_complex():
        raise ValueError("positions must be real and on the same device as x")
    if not torch.isfinite(positions).all() or (positions < 0).any():
        raise ValueError("positions must be finite and nonnegative")
    work_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    frequencies = base ** (-torch.arange(0, dim, 2, device=x.device, dtype=work_dtype) / dim)
    angles = positions.to(work_dtype)[:, None] * frequencies[None, :]
    c, s = angles.cos(), angles.sin()
    even = x[..., :dim:2].to(work_dtype)
    odd = x[..., 1:dim:2].to(work_dtype)
    rotated = torch.stack((even * c - odd * s, even * s + odd * c), dim=-1)
    rotated = rotated.flatten(-2).to(x.dtype)
    return torch.cat((rotated, x[..., dim:]), dim=-1)


class RopeAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.rope_base = config.rope_base
        self.rotary_dim = config.rotary_dim
        self.W_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_o = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        def project(layer):
            return layer(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q, k, v = project(self.W_q), project(self.W_k), project(self.W_v)
        positions = torch.arange(T, device=x.device)
        q = apply_rope(q, positions, self.rope_base, self.rotary_dim)
        k = apply_rope(k, positions, self.rope_base, self.rotary_dim)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        allowed = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        weights = scores.masked_fill(~allowed, float("-inf")).softmax(-1)
        attended = (weights @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(attended)


class RopeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = RopeAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(nn.Linear(config.d_model, 4 * config.d_model), nn.GELU(),
                                 nn.Linear(4 * config.d_model, config.d_model))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class RopeGPT(nn.Module):
    """S2 changes position handling only: still LayerNorm, GELU, and MHA."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_seq_len = config.max_seq_len
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([RopeBlock(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def cache_extra_signature(self):
        """Runtime settings without Parameters also affect cached activations."""
        return tuple((name, type(module), getattr(module, "eps", None),
                      getattr(module, "approximate", None))
                     for name, module in self.named_modules())

    def forward(self, idx, targets=None):
        s0.validate_tokens(idx, self.config)
        if idx.device != self.tok_emb.weight.device:
            raise ValueError("tokens must be on the model device")
        if targets is not None:
            s0.validate_tokens(targets, self.config)
            if targets.shape != idx.shape or targets.device != idx.device:
                raise ValueError("targets must have the same shape/device as inputs")
        x = self.tok_emb(idx)  # No learned absolute position embedding.
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        loss = None if targets is None else F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


def cache_signature(model):
    """Detect parameter replacement/update and changed positional configuration.

    This is not a content hash. Deliberate .data mutation is unsupported, as in S1.
    S3 can reuse this helper; S4 owns its different KV-head shape validation.
    """
    settings = tuple((a.rope_base, a.rotary_dim, a.num_heads, a.head_dim)
                     for a in (block.attn for block in model.blocks))
    extra = model.cache_extra_signature() if hasattr(model, "cache_extra_signature") else ()
    return (kv.weight_versions(model), model.config, model.max_seq_len, settings, extra)


def _validate_cache(model, idx, cache):
    if model.training:
        raise ValueError("cached_forward requires model.eval()")
    s0.validate_tokens(idx, model.config)
    if idx.device != model.tok_emb.weight.device:
        raise ValueError("tokens must be on the model device")
    B, T = idx.shape
    if cache is None:
        return 0
    if not isinstance(cache, KVCache) or cache.owner is not model:
        raise ValueError("cache belongs to a different model/request")
    if cache.versions != cache_signature(model):
        raise ValueError("model weights or configuration changed; reset cache")
    if len(cache.layers) != model.config.num_layers:
        raise ValueError("cache layer count mismatch")
    for pair in cache.layers:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("each cache layer must contain one K/V pair")
        if any(not isinstance(t, torch.Tensor) or t.ndim != 4 for t in pair):
            raise ValueError("cache K/V must have [B, H, T, D] shape")
    offset = cache.length
    expected = (B, model.config.num_heads, offset, model.config.d_model // model.config.num_heads)
    for pair in cache.layers:
        if any(tuple(t.shape) != expected for t in pair):
            raise ValueError("cache batch/head/length shape mismatch")
        if any(t.device != model.tok_emb.weight.device or t.dtype != model.tok_emb.weight.dtype for t in pair):
            raise ValueError("cache dtype/device mismatch; reset cache")
    if offset + T > model.max_seq_len:
        raise ValueError("cache plus chunk exceeds max_seq_len; reset request")
    return offset


@torch.no_grad()
def cached_forward(model, idx, cache=None):
    """Same S2 weights as full forward; cache stores ROTATED K and ordinary V.

    Positions and RoPE settings are fixed for this request. No sliding eviction,
    autocast, ragged/padded batches, or dynamic frequency rescaling. Reset after
    weight or configuration changes. Treat all returned cache tensors as read-only.
    """
    offset = _validate_cache(model, idx, cache)
    B, T = idx.shape
    positions = torch.arange(offset, offset + T, device=idx.device)
    x = model.tok_emb(idx)
    layers = []
    for i, block in enumerate(model.blocks):
        attn = block.attn
        normalized = block.ln1(x)
        def project(layer):
            return layer(normalized).view(B, T, attn.num_heads, attn.head_dim).transpose(1, 2)
        q, k, v = project(attn.W_q), project(attn.W_k), project(attn.W_v)
        q = apply_rope(q, positions, attn.rope_base, attn.rotary_dim)
        k = apply_rope(k, positions, attn.rope_base, attn.rotary_dim)
        if cache is not None:
            k = torch.cat((cache.layers[i][0], k), dim=2)
            v = torch.cat((cache.layers[i][1], v), dim=2)
        allowed = torch.arange(offset + T, device=x.device)[None, :] <= positions[:, None]
        scores = q @ k.transpose(-2, -1) / math.sqrt(attn.head_dim)
        weights = scores.masked_fill(~allowed, float("-inf")).softmax(-1)
        attended = (weights @ v).transpose(1, 2).contiguous().view(B, T, -1)
        x = x + attn.W_o(attended)
        x = x + block.mlp(block.ln2(x))
        layers.append((k, v))
    return model.head(model.ln_f(x)), KVCache(tuple(layers), model, cache_signature(model))


def rotation_walkthrough():
    """Independent tiny inputs behind chapter tables, in float64."""
    positions = torch.tensor([0., math.pi / 2, math.pi], dtype=torch.float64)
    x = torch.tensor([[3., 4.]], dtype=torch.float64).repeat(3, 1)
    rotated = apply_rope(x, positions)
    q = torch.tensor([[1., 2.]], dtype=torch.float64)
    k = torch.tensor([[3., 4.]], dtype=torch.float64)
    rows = []
    for m, n in ((0., 1.), (2., 3.), (0., 2.)):
        rq = apply_rope(q, torch.tensor([m], dtype=torch.float64))
        rk = apply_rope(k, torch.tensor([n], dtype=torch.float64))
        rows.append((m, n, (rq * rk).sum().item()))
    return rotated, rotated.norm(dim=-1), rows



def frequency_walkthrough():
    """D_rot=8, base=10000: frequency, angle at p=10, wavelength."""
    return [(i, 10000. ** (-2*i/8), 10 * 10000. ** (-2*i/8),
             2 * math.pi / (10000. ** (-2*i/8))) for i in range(4)]


def interpolation_walkthrough():
    """L=8 -> L'=32, s=4; PI changes phases, not token count or mask."""
    return [(p, float(p), p / 4.) for p in (0, 7, 8, 31)]

def retrieval_batch(generator, batch_size, length):
    """[BOS=9, value(0..3), distractors(0..7), QUERY=8] -> value.

    Only QUERY-position classification is scored. This is NOT natural-language
    next-token perplexity; longer tests increase distance AND distractor count.
    """
    if type(length) is not int or length < 4:
        raise ValueError("retrieval length must be an integer >= 4")
    tokens = torch.randint(0, 8, (batch_size, length), generator=generator)
    targets = torch.randint(0, 4, (batch_size,), generator=generator)
    tokens[:, 0], tokens[:, 1], tokens[:, -1] = 9, targets, 8
    return tokens, targets


def length_experiment(seed=2026, steps=400, train_length=8, eval_lengths=(8, 16, 32, 64), eval_batches=8):
    """Fixed-length short training, independently generated test batches, CPU only.

    Returns final-query CE/accuracy, not every-token language-model perplexity.
    max_seq_len is already 64 during training: permitting positions is not
    evidence they were trained. Training sees only positions 0..7.
    """
    if type(steps) is not int or steps < 1 or type(eval_batches) is not int or eval_batches < 1:
        raise ValueError("steps and eval_batches must be positive integers")
    if type(train_length) is not int or train_length < 4 or not eval_lengths or any(type(n) is not int or n < 4 for n in eval_lengths):
        raise ValueError("train/evaluation lengths must be integers >= 4")
    # Do not alter the caller's CPU RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = RopeGPT(RopeConfig(10, 32, 4, 2, max(train_length, *eval_lengths)))
        train_gen = torch.Generator().manual_seed(seed + 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)
        model.train()
        initial = None
        for step in range(steps):
            inputs, targets = retrieval_batch(train_gen, 32, train_length)
            logits, _ = model(inputs)
            loss = F.cross_entropy(logits[:, -1], targets)
            if initial is None:
                initial = loss.item()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        rows = []
        with torch.no_grad():
            for length in eval_lengths:
                # Separate, repeatable test stream, never used by the optimizer.
                test_gen = torch.Generator().manual_seed(seed + 10000 + length)
                loss_sum, correct, count = 0., 0, 0
                for _ in range(eval_batches):
                    inputs, targets = retrieval_batch(test_gen, 64, length)
                    logits, _ = model(inputs)
                    logits = logits[:, -1]
                    loss_sum += F.cross_entropy(logits, targets, reduction="sum").item()
                    correct += (logits.argmax(-1) == targets).sum().item()
                    count += targets.numel()
                rows.append({"length": length, "count": count, "loss_sum": loss_sum,
                             "mean_ce": loss_sum / count, "correct": correct,
                             "accuracy": correct / count})
        return {"seed": seed, "steps": steps, "train_length": train_length,
                "first_train_ce": initial, "last_train_ce": loss.item(), "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2025, 2026, 2027])
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(f"torch={torch.__version__}; device=cpu; dtype=float32; threads=1")
    rotated, norms, phases = rotation_walkthrough()
    print("rotate [3,4] by 0, pi/2, pi (radians):")
    for row, norm in zip(rotated.tolist(), norms.tolist()):
        print(f"  [{row[0]:.6f}, {row[1]:.6f}] norm={norm:.6f}")
    print("q=[1,2], k=[3,4], theta=1:")
    for m, n, score in phases:
        print(f"  m={m:g} n={n:g} dot={score:.9f}")
    print("D_rot=8, base=10000: pair, theta, angle at position10, wavelength:")
    for i, theta, angle, wavelength in frequency_walkthrough():
        print(f"  {i} {theta:.6f} {angle:.6f} {wavelength:.6f}")
    print("L=8 -> L'=32: position, extrapolation phase/theta, PI phase/theta:")
    for p, unchanged, interpolated in interpolation_walkthrough():
        print(f"  {p} {unchanged:.6f} {interpolated:.6f}")
    torch.manual_seed(42)
    model = RopeGPT(RopeConfig(17, 16, 4, 2, 64)).eval()
    tokens = torch.randint(17, (2, 12))
    with torch.no_grad():
        reference = model(tokens)[0]
        cache, outputs, start = None, [], 0
        for size in (3, 4, 2, 3):
            logits, cache = cached_forward(model, tokens[:, start:start + size], cache)
            outputs.append(logits)
            start += size
        error = (reference - torch.cat(outputs, 1)).abs().max().item()
    print(f"same-weights full/cache max_abs_error={error:.9g}; KV bytes={cache_bytes(cache)}")
    print("training: first-token retrieval with random distractors; NOT full YaRN")
    if not args.skip_training:
        for seed in args.seeds:
            result = length_experiment(seed=seed)
            print(f"seed={result['seed']}; steps={result['steps']}; train_T={result['train_length']}; batch=32; lr=.003")
            print(f"first_train_ce={result['first_train_ce']:.6f}; last_train_ce={result['last_train_ce']:.6f}")
            for row in result["rows"]:
                print(f"T={row['length']:2d} N={row['count']} correct={row['correct']:3d} CE_sum={row['loss_sum']:.6f} CE_mean={row['mean_ce']:.6f} accuracy={row['accuracy']:.6f}")


if __name__ == "__main__":
    main()
