"""P11 / S1: same S0 weights, explicit per-request MHA KV cache, CPU experiment.

Run: OMP_NUM_THREADS=1 python code/12_kv_cache.py
No training/download. Timings are this small CPU workload, not guaranteed speedup.
"""
import importlib.util
import math
from pathlib import Path
import statistics
import sys
import time
from dataclasses import dataclass

try:
    import torch
except ModuleNotFoundError as exc:
    if __name__ == "__main__" and exc.name == "torch":
        print("P11 requires PyTorch. Install dependencies: python -m pip install -r requirements.txt", file=sys.stderr)
        raise SystemExit(2) from None
    raise

# Numeric filenames are loaded once under a stable module name.
_NAME = "zero2llm_s0"
if _NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(_NAME, Path(__file__).with_name("07_baby_gpt.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[_NAME] = module
    spec.loader.exec_module(module)
s0 = sys.modules[_NAME]


@dataclass(frozen=True)
class KVCache:
    """Caller-owned state, not stored on model. Tensors must be treated as read-only.

    Each pair is [B, H, cached_T, d]. Do not reuse after changing weights;
    pass None to reset for every independent request. No ragged/padded batches.
    """
    layers: tuple
    owner: object
    versions: tuple

    @property
    def length(self):
        return self.layers[0][0].shape[2]


def weight_versions(model):
    """Identity catches assign=True/replaced Parameters; version catches in-place updates.

    dtype/device also catches module conversions that may preserve Parameter
    identity. Deliberate .data mutation bypasses PyTorch version tracking and is
    unsupported; never edit parameters through .data while reusing a cache.
    """
    return tuple((id(p), p._version, p.dtype, p.device) for p in model.parameters())


@torch.no_grad()
def cached_forward(model, idx, cache=None):
    """Return new-token logits [B, chunk_T, V] and a NEW cache.

    eval only, absolute positions start at cached_T. Reject overflow: discarding
    old K/V cannot reproduce S0's sliding-window position reset. CPU teaching
    path without autocast; model and cache must share device and floating dtype.
    """
    if model.training:
        raise ValueError("cached_forward requires model.eval()")
    s0.validate_tokens(idx, model.config)
    B, T = idx.shape
    offset = 0
    if cache is not None:
        if not isinstance(cache, KVCache) or cache.owner is not model:
            raise ValueError("cache belongs to a different model/request")
        if cache.versions != weight_versions(model):
            raise ValueError("model weights changed; reset cache")
        if len(cache.layers) != model.config.num_layers:
            raise ValueError("cache layer count mismatch")
        for pair in cache.layers:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError("each cache layer must contain one K/V pair")
            if any(not isinstance(t, torch.Tensor) or t.ndim != 4 for t in pair):
                raise ValueError("cache K/V must have [B, H, T, D] shape")
        offset = cache.length
        for k, v in cache.layers:
            expected = (B, model.config.num_heads, offset,
                        model.config.d_model // model.config.num_heads)
            if tuple(k.shape) != expected or tuple(v.shape) != expected:
                raise ValueError("cache batch/head/length shape mismatch")
            if any(t.device != model.tok_emb.weight.device or
                   t.dtype != model.tok_emb.weight.dtype for t in (k, v)):
                raise ValueError("cache dtype/device mismatch; reset cache")
    if offset + T > model.max_seq_len:
        raise ValueError("cache plus chunk exceeds max_seq_len; reset request")
    pos = torch.arange(offset, offset + T, device=idx.device)
    x = model.tok_emb(idx) + model.pos_emb(pos)
    new_layers = []
    for i, block in enumerate(model.blocks):
        normalized = block.ln1(x)
        attn = block.attn
        def project(layer):
            return layer(normalized).view(B, T, attn.num_heads, attn.head_dim).transpose(1, 2)
        q, k, v = project(attn.W_q), project(attn.W_k), project(attn.W_v)
        if cache is not None:
            k = torch.cat((cache.layers[i][0], k), dim=2)
            v = torch.cat((cache.layers[i][1], v), dim=2)
        # Rectangular [chunk_T, cached_T + chunk_T], not tril(chunk_T, total_T).
        query_positions = torch.arange(offset, offset + T, device=x.device)[:, None]
        key_positions = torch.arange(offset + T, device=x.device)[None, :]
        allowed = key_positions <= query_positions
        scores = q @ k.transpose(-2, -1) / math.sqrt(attn.head_dim)
        weights = scores.masked_fill(~allowed, float("-inf")).softmax(dim=-1)
        attended = (weights @ v).transpose(1, 2).contiguous().view(B, T, -1)
        x = x + attn.W_o(attended)
        x = x + block.mlp(block.ln2(x))
        new_layers.append((k, v))
    logits = model.head(model.ln_f(x))
    return logits, KVCache(tuple(new_layers), model, weight_versions(model))


def cache_bytes(cache):
    """Actual logical tensor payload; excludes allocator/Python overhead."""
    return sum(t.numel() * t.element_size() for pair in cache.layers for t in pair)


def expected_cache_bytes(config, batch_size, length, dtype=torch.float32):
    """MHA only: 2 * L * B * H * T * d * element_size; no GQA/MQA here."""
    for value in (batch_size, length):
        if type(value) is not int or value < 0:
            raise ValueError("batch_size and length must be nonnegative integers")
    if batch_size == 0 or length > config.max_seq_len:
        raise ValueError("invalid batch_size or cache length")
    size = torch.empty((), dtype=dtype).element_size()
    return (2 * config.num_layers * batch_size * config.num_heads * length *
            (config.d_model // config.num_heads) * size)


@torch.no_grad()
def generate_cached(model, idx, max_new_tokens=8, temperature=0.0):
    """Token API; fresh cache each call. <=0 greedy, >0 multinomial.

    Conservative capacity rule: prompt + requested output <= max_seq_len,
    including the last output token (which need not itself be forwarded).
    """
    s0.validate_generation(max_new_tokens, temperature)
    s0.validate_tokens(idx, model.config)
    if model.training:
        raise ValueError("generate_cached requires model.eval()")
    if idx.shape[1] + max_new_tokens > model.max_seq_len:
        raise ValueError("prompt plus output budget exceeds max_seq_len")
    result = idx.clone()
    if not max_new_tokens:
        return result
    logits, cache = cached_forward(model, idx)
    for step in range(max_new_tokens):
        last = logits[:, -1]
        token = last.argmax(-1, keepdim=True) if temperature <= 0 else torch.multinomial(
            ((last - last.amax(dim=-1, keepdim=True)) /
             max(temperature, torch.finfo(last.dtype).tiny)).softmax(-1), 1)
        result = torch.cat((result, token), dim=1)
        if step + 1 < max_new_tokens:
            logits, cache = cached_forward(model, token, cache)
    return result


@torch.no_grad()
def benchmark(model, tokens, prompt_length=16, repeats=7, warmup=2):
    """Fixed token replay, same weights/tokens. CPU wall time, no GPU sync needed.

    First-token metric is prompt forward only (no tokenizer/sampling/network).
    Later metric is mean forward time per supplied decode token within a run.
    Min/median/max are across runs, not a production TTFT/TPOT claim.
    """
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("benchmark is CPU only")
    s0.validate_tokens(tokens, model.config)
    if not 0 < prompt_length < tokens.shape[1] or repeats < 1 or warmup < 0:
        raise ValueError("need prompt and later tokens, positive repeats, nonnegative warmup")
    records = {name: [] for name in ("full_first_ms", "cache_first_ms", "full_later_ms", "cache_later_ms")}
    def run(cached):
        start = time.perf_counter()
        if cached:
            _, cache = cached_forward(model, tokens[:, :prompt_length])
        else:
            model(tokens[:, :prompt_length])
        first = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        for t in range(prompt_length, tokens.shape[1]):
            if cached:
                _, cache = cached_forward(model, tokens[:, t:t+1], cache)
            else:
                model(tokens[:, :t+1])
        later = (time.perf_counter() - start) * 1000 / (tokens.shape[1] - prompt_length)
        return first, later
    for n in range(warmup + repeats):
        # Alternate order so one implementation is not always warmed by the other.
        for cached in ([False, True] if n % 2 == 0 else [True, False]):
            first, later = run(cached)
            if n >= warmup:
                prefix = "cache" if cached else "full"
                records[prefix + "_first_ms"].append(first)
                records[prefix + "_later_ms"].append(later)
    return {key: {"min": min(values), "median": statistics.median(values), "max": max(values)}
            for key, values in records.items()}


def rectangular_mask_walkthrough():
    """P=3 cached keys + C=2 new queries, independent hand-computable example."""
    scores = torch.tensor([[1., 2., 3., 4., 5.]], dtype=torch.float64).log().repeat(2, 1)
    values = torch.tensor([10., 20., 30., 40., 50.], dtype=torch.float64)
    allowed = torch.arange(5)[None, :] <= torch.arange(3, 5)[:, None]
    weights = scores.masked_fill(~allowed, float("-inf")).softmax(-1)
    wrong_mask = torch.ones(2, 5, dtype=torch.bool).tril()
    wrong_weights = scores.masked_fill(~wrong_mask, float("-inf")).softmax(-1)
    return allowed, weights, weights @ values, wrong_weights @ values


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = s0.BabyGPT(s0.vocab_size).eval()
    allowed, weights, outputs, wrong = rectangular_mask_walkthrough()
    print("P=3 C=2 allowed:", allowed.int().tolist())
    print("weights:", weights.tolist(), "outputs:", outputs.tolist())
    print("wrong naive tril outputs:", wrong.tolist())
    print("default L=2 B=1 H=4 T=5 d=12 float32 bytes:",
          expected_cache_bytes(model.config, 1, 5))
    tokens = torch.randint(s0.vocab_size, (1, 48))
    with torch.no_grad():
        full, _ = model(tokens)
        cache = None
        pieces = []
        for chunk in tokens.split(7, dim=1):
            logits, cache = cached_forward(model, chunk, cache)
            pieces.append(logits)
        error = (full - torch.cat(pieces, dim=1)).abs().max().item()
    print(f"CPU torch={torch.__version__}, threads={torch.get_num_threads()}, seed=42, dtype={full.dtype}")
    print("S0 defaults: d_model=48, heads=4, layers=2, capacity=128; random weights, not quality evaluation")
    print(f"B=1 T=48 chunks=7 max_logit_error={error:.3g}")
    print(f"KV payload={cache_bytes(cache)} bytes, formula={expected_cache_bytes(model.config, 1, 48)} bytes")
    print("prompt=16 later=32, warmup=2 repeats=7, CPU perf_counter, no asynchronous GPU work")
    print(benchmark(model, tokens))
    print("Not guaranteed faster: Python checks/cat and small CPU tensors can dominate. Payload excludes overhead.")


if __name__ == "__main__":
    main()
