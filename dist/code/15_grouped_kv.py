"""P14 / S4: real grouped K/V projections and compact request cache.

Run: OMP_NUM_THREADS=1 python code/15_grouped_kv.py
S4 retains S3's RoPE, RMSNorm and SwiGLU. The independent float64 MLA
matrix experiment is NOT a full MLA model, checkpoint converter or GPU kernel.
"""
import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path
import sys

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:
    if __name__ == "__main__" and exc.name == "torch":
        print("P14 requires PyTorch. Install dependencies: python -m pip install -r requirements.txt", file=sys.stderr)
        raise SystemExit(2) from None
    raise

_NAME = "zero2llm_s3"
if _NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(_NAME, Path(__file__).with_name("14_modern_block.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[_NAME] = module
    spec.loader.exec_module(module)
s3 = sys.modules[_NAME]
s2 = s3.s2


def head_mapping(num_heads, num_kv_heads):
    """Contiguous groups: Hq=4,Hkv=2 gives (0,0,1,1), NOT (0,1,0,1)."""
    if any(type(n) is not int or n <= 0 for n in (num_heads, num_kv_heads)):
        raise ValueError("num_heads and num_kv_heads must be positive integers")
    if num_heads % num_kv_heads:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    return tuple(h // (num_heads // num_kv_heads) for h in range(num_heads))


@dataclass(frozen=True)
class GroupedConfig(s3.ModernConfig):
    num_kv_heads: int = 2

    def __post_init__(self):
        super().__post_init__()
        head_mapping(self.num_heads, self.num_kv_heads)


def grouped_attention(q, k, v, offset=0):
    """q:[B,Hq,C,d], k/v:[B,Hkv,P+C,d] -> [B,Hq,C,d].

    Only the temporary attention operands are expanded. Returned request
    caches MUST keep the original compact k/v. This simple implementation
    still materializes attention scores, so it is not a memory-efficient kernel.
    """
    if any(not isinstance(t, torch.Tensor) or t.ndim != 4 or not t.is_floating_point() for t in (q, k, v)):
        raise ValueError("q/k/v must be floating rank-four tensors")
    B, Hq, C, d = q.shape
    if min(B, Hq, C, d) <= 0 or tuple(k.shape) != tuple(v.shape):
        raise ValueError("nonempty q and matching k/v shapes required")
    if k.shape[0] != B or k.shape[-1] != d:
        raise ValueError("q/k/v batch and head width must match")
    Hkv, total = k.shape[1:3]
    head_mapping(Hq, Hkv)
    if type(offset) is not int or offset < 0 or total != offset + C:
        raise ValueError("key length must equal offset plus query length")
    if any(t.dtype != q.dtype or t.device != q.device for t in (k, v)):
        raise ValueError("q/k/v dtype/device mismatch")
    repeats = Hq // Hkv
    keys = k.repeat_interleave(repeats, dim=1)
    values = v.repeat_interleave(repeats, dim=1)
    query_positions = torch.arange(offset, offset + C, device=q.device)[:, None]
    key_positions = torch.arange(total, device=q.device)[None, :]
    allowed = key_positions <= query_positions
    scores = q @ keys.transpose(-2, -1) / math.sqrt(d)
    weights = scores.masked_fill(~allowed, float("-inf")).softmax(dim=-1)
    return weights @ values


class GroupedAttention(nn.Module):
    """S2 RoPE attention, with Wk/Wv output width Hkv*d rather than Hq*d."""
    def __init__(self, config):
        super().__init__()
        self.rope_base = config.rope_base
        self.rotary_dim = config.rotary_dim
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.d_model // config.num_heads
        head_mapping(self.num_heads, self.num_kv_heads)
        D, kv_width = config.d_model, self.num_kv_heads * self.head_dim
        self.W_q = nn.Linear(D, D, bias=False)
        self.W_k = nn.Linear(D, kv_width, bias=False)
        self.W_v = nn.Linear(D, kv_width, bias=False)
        self.W_o = nn.Linear(D, D, bias=False)

    def project(self, x, offset=0):
        B, T, _ = x.shape
        q = self.W_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # Never rotate V or rotate an old cached K a second time.
        positions = torch.arange(offset, offset + T, device=x.device)
        q = s2.apply_rope(q, positions, self.rope_base, self.rotary_dim)
        k = s2.apply_rope(k, positions, self.rope_base, self.rotary_dim)
        return q, k, v

    def attend(self, q, k, v, offset=0):
        B, _, T, _ = q.shape
        y = grouped_attention(q, k, v, offset)
        return self.W_o(y.transpose(1, 2).contiguous().view(B, T, -1))

    def forward(self, x):
        return self.attend(*self.project(x))


class GroupedGPT(s3.ModernGPT):
    """S4 adds GQA only. Hkv=Hq is structurally identical to S3 attention."""
    def __init__(self, config):
        if not isinstance(config, GroupedConfig):
            raise ValueError("GroupedGPT requires GroupedConfig")
        super().__init__(config)
        for block in self.blocks:
            old = block.attn
            block.attn = GroupedAttention(config)
            with torch.no_grad():
                block.attn.W_q.weight.copy_(old.W_q.weight)
                block.attn.W_o.weight.copy_(old.W_o.weight)
                if config.num_kv_heads == config.num_heads:
                    block.attn.W_k.weight.copy_(old.W_k.weight)
                    block.attn.W_v.weight.copy_(old.W_v.weight)


@dataclass(frozen=True)
class GroupedCache:
    """Caller-owned [B,Hkv,T,d] tensors. Treat payloads as read-only.

    Frozen metadata does not make tensors immutable. Start a new request with
    None; never reuse after editing weights, configuration, or token history.
    """
    layers: tuple
    owner: object
    versions: tuple
    @property
    def length(self):
        return self.layers[0][0].shape[2]


def cache_signature(model):
    """Include S2 position settings, grouped head counts and normalization epsilon.

    Parameters use identity, in-place version and dtype/device. Deliberate .data
    edits bypass PyTorch version tracking and are unsupported, as in S1/S2.
    """
    norms = [model.ln_f] + [n for b in model.blocks for n in (b.ln1, b.ln2)]
    return (s2.cache_signature(model), tuple(b.attn.num_kv_heads for b in model.blocks),
            tuple(n.eps for n in norms))


@torch.no_grad()
def cached_forward(model, idx, cache=None):
    """Compact GQA cache; eval only, no padding/ragged batches/autocast."""
    if not isinstance(model, GroupedGPT):
        raise ValueError("use S4 cached_forward with GroupedGPT")
    if model.training:
        raise ValueError("cached_forward requires model.eval()")
    s2.s0.validate_tokens(idx, model.config)
    if idx.device != model.tok_emb.weight.device:
        raise ValueError("tokens must be on the model device")
    B, T = idx.shape
    offset = 0
    if cache is not None:
        if not isinstance(cache, GroupedCache) or cache.owner is not model:
            raise ValueError("cache belongs to a different model/request")
        if cache.versions != cache_signature(model):
            raise ValueError("model weights/config changed; reset cache")
        if not isinstance(cache.layers, tuple) or len(cache.layers) != model.config.num_layers:
            raise ValueError("cache layer count mismatch")
        for pair in cache.layers:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError("each cache layer must contain one K/V pair")
            if any(not isinstance(t, torch.Tensor) or t.ndim != 4 for t in pair):
                raise ValueError("cache K/V must have [B,Hkv,T,d] shape")
        offset = cache.length
        expected = (B, model.config.num_kv_heads, offset,
                    model.config.d_model // model.config.num_heads)
        for pair in cache.layers:
            if any(tuple(t.shape) != expected for t in pair):
                raise ValueError("cache batch/head/length shape mismatch")
            if any(t.dtype != model.tok_emb.weight.dtype or
                   t.device != model.tok_emb.weight.device for t in pair):
                raise ValueError("cache dtype/device mismatch; reset cache")
    if offset + T > model.max_seq_len:
        raise ValueError("cache plus chunk exceeds max_seq_len; reset request")
    x = model.tok_emb(idx)
    layers = []
    for i, block in enumerate(model.blocks):
        q, k, v = block.attn.project(block.ln1(x), offset=offset)
        if cache is not None:
            k = torch.cat((cache.layers[i][0], k), dim=2)
            v = torch.cat((cache.layers[i][1], v), dim=2)
        x = x + block.attn.attend(q, k, v, offset=offset)
        x = x + block.mlp(block.ln2(x))
        layers.append((k, v))  # Hkv heads, NOT repeat_interleave's Hq copies.
    return model.head(model.ln_f(x)), GroupedCache(
        tuple(layers), model, cache_signature(model))


def cache_bytes(cache):
    """Logical tensor payload only; excludes temporary repeats and allocator overhead."""
    return sum(t.numel() * t.element_size() for pair in cache.layers for t in pair)


def expected_cache_bytes(config, batch_size, length, dtype=torch.float32):
    for value in (batch_size, length):
        if type(value) is not int or value < 0:
            raise ValueError("batch_size and length must be nonnegative integers")
    if batch_size == 0 or length > config.max_seq_len:
        raise ValueError("invalid batch size or cache length")
    head_mapping(config.num_heads, config.num_kv_heads)
    return (2 * config.num_layers * batch_size * length * config.num_kv_heads *
            (config.d_model // config.num_heads) * torch.empty((), dtype=dtype).element_size())


def mean_pool_kv_weight(weight, num_heads, num_kv_heads):
    """Educational checkpoint INITIALIZATION, not a function-preserving conversion.

    PyTorch Linear weight is [out_features,in_features]; group contiguous heads.
    Actual uptraining and quality evaluation are deliberately not implemented.
    """
    head_mapping(num_heads, num_kv_heads)
    if (not isinstance(weight, torch.Tensor) or weight.ndim != 2 or
            not weight.is_floating_point() or min(weight.shape) <= 0 or weight.shape[0] % num_heads):
        raise ValueError("weight must have [Hq*d,input_dim] shape")
    d, input_dim = weight.shape[0] // num_heads, weight.shape[1]
    return weight.reshape(num_kv_heads, num_heads // num_kv_heads, d, input_dim).mean(1).reshape(-1, input_dim)


def mapping_walkthrough():
    """At a single key position all weights equal 1; V alone exposes head mapping."""
    q = torch.zeros(1, 4, 1, 2, dtype=torch.float64)
    k = torch.zeros(1, 2, 1, 2, dtype=torch.float64)
    v = torch.tensor([[[[10., 11.]], [[20., 21.]]]], dtype=torch.float64)
    return grouped_attention(q, k, v).squeeze(0).squeeze(1)


def mla_matrix_demo():
    """Independent float64 algebra, not a pretrained model or production MLA.

    Three positions, two heads, latent rank 2, content width 2, value width 2,
    and a separate shared positional key of width 2. Row-vector convention.
    Retains explicit K/V only for the numerical reference, not for a serving cache.
    """
    dtype = torch.float64
    x = torch.tensor([[1., 0., 1., 0.], [0., 1., 0., 1.], [1., 1., 0., 0.]], dtype=dtype)
    down = torch.tensor([[1., 0.], [0., 1.], [.5, 0.], [0., .5]], dtype=dtype)
    c = x @ down
    up_k = torch.tensor([[[1., 0.], [0., 1.]], [[1., .5], [-.5, 1.]]], dtype=dtype)
    up_v = torch.tensor([[[1., 2.], [0., 1.]], [[-1., 0.], [1., 2.]]], dtype=dtype)
    q_content = torch.tensor([[[1., .2], [.3, 1.], [.5, -.5]],
                              [[.1, 1.], [1., -.2], [.4, .8]]], dtype=dtype)
    # One frequency theta=1 is enough to expose a position-dependent term.
    positions = torch.arange(3, dtype=dtype)
    def rotate2(z):
        a, b = z[..., 0], z[..., 1]
        cosine, sine = positions.cos(), positions.sin()
        return torch.stack((a * cosine - b * sine, a * sine + b * cosine), dim=-1)
    k_rope = rotate2(x[:, :2])  # [T,dr], shared across heads, stored separately.
    q_rope = rotate2(q_content * .5)  # [H,T,dr], current queries are not cached.
    k_content = torch.einsum("tr,hrd->htd", c, up_k)
    values = torch.einsum("tr,hrd->htd", c, up_v)
    positional_scores = torch.einsum("hqd,td->hqt", q_rope, k_rope)
    scale = math.sqrt(q_content.shape[-1] + q_rope.shape[-1])
    full_scores = (torch.einsum("hqd,htd->hqt", q_content, k_content) + positional_scores) / scale
    q_absorbed = torch.einsum("hqd,hrd->hqr", q_content, up_k)
    latent_scores = (torch.einsum("hqr,tr->hqt", q_absorbed, c) + positional_scores) / scale
    allowed = torch.arange(3)[None, :] <= torch.arange(3)[:, None]
    full_weights = full_scores.masked_fill(~allowed, -float("inf")).softmax(-1)
    latent_weights = latent_scores.masked_fill(~allowed, -float("inf")).softmax(-1)
    full_output = torch.einsum("hqt,htd->hqd", full_weights, values)
    latent_sum = torch.einsum("hqt,tr->hqr", latent_weights, c)
    latent_output = torch.einsum("hqr,hrd->hqd", latent_sum, up_v)
    # Rank restriction is a different question from the identities above.
    target = torch.diag(torch.tensor([3., 2., 1.], dtype=dtype))
    rank_down = torch.tensor([[3., 0.], [0., 2.], [0., 0.]], dtype=dtype)
    rank_up = torch.tensor([[1., 0., 0.], [0., 1., 0.]], dtype=dtype)
    low_rank = rank_down @ rank_up
    return {
        "latent": c, "k_rope": k_rope, "full_scores": full_scores,
        "latent_scores": latent_scores, "full_output": full_output,
        "latent_output": latent_output, "positional_scores": positional_scores,
        "score_error": (full_scores - latent_scores).abs().max().item(),
        "output_error": (full_output - latent_output).abs().max().item(),
        "compact_bytes": (c.numel() + k_rope.numel()) * c.element_size(),
        "expanded_bytes": (k_content.numel() + values.numel() + k_rope.numel()) * c.element_size(),
        "rank_target": int(torch.linalg.matrix_rank(target)),
        "rank_compressed": int(torch.linalg.matrix_rank(low_rank)),
        "rank_loss": torch.linalg.vector_norm(target - low_rank).item(),
    }


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    print(f"CPU torch={torch.__version__}, threads=1, seed=42; no training or speed claim")
    print("Hq=4 Hkv=2 mapping:", head_mapping(4, 2))
    print("one-key head outputs:", mapping_walkthrough().tolist())
    print("S4 D=48 Hq=4 d=12 L=2 B=2 T=12 float32")
    tokens = torch.arange(24).reshape(2, 12) % 17
    for Hkv in (4, 2, 1):
        model = GroupedGPT(GroupedConfig(vocab_size=17, num_kv_heads=Hkv)).eval()
        with torch.no_grad():
            full, _ = model(tokens)
            cache, pieces, start = None, [], 0
            for size in (3, 4, 1, 4):
                out, cache = cached_forward(model, tokens[:, start:start + size], cache)
                pieces.append(out)
                start += size
            error = (full - torch.cat(pieces, 1)).abs().max().item()
        kv_parameters = sum(b.attn.W_k.weight.numel() + b.attn.W_v.weight.numel() for b in model.blocks)
        print(f"Hkv={Hkv}: KV parameters={kv_parameters}, cache={cache_bytes(cache)} bytes, "
              f"formula={expected_cache_bytes(model.config, 2, 12)}, shape={tuple(cache.layers[0][0].shape)}, max_error={error:.3g}")
        torch.testing.assert_close(full, torch.cat(pieces, 1), atol=3e-6, rtol=1e-5)
        assert cache_bytes(cache) == expected_cache_bytes(model.config, 2, 12)
    pooled = mean_pool_kv_weight(torch.arange(8, dtype=torch.float64).reshape(4, 2), 4, 2)
    print("mean-pooled example:", pooled.tolist(), "(initialization, not equivalence)")
    demo = mla_matrix_demo()
    print(f"MLA matrix scores error={demo['score_error']:.3g}, output error={demo['output_error']:.3g}")
    print(f"MLA toy payload compact={demo['compact_bytes']} bytes vs expanded={demo['expanded_bytes']} bytes; both include RoPE keys")
    print(f"Rank example: {demo['rank_target']} -> {demo['rank_compressed']}, Frobenius error={demo['rank_loss']:.1f}")
    assert demo["score_error"] < 1e-12 and demo["output_error"] < 1e-12
    assert demo["rank_loss"] == 1.0


if __name__ == "__main__":
    main()
