"""P17: CPU dataflow toy, NOT DeepSeek weights, CSA2 or a serving engine.

Two-dimensional, fixed-weight causal encoder -> decoder global K/V. Decoder
local K/V come from each decoder layer's input. Separate branch softmaxes and
plain residual addition are teaching choices, not the production attention.
Run: OMP_NUM_THREADS=1 python code/18_ced_architecture.py
"""
from dataclasses import dataclass
import math
import sys

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:
    if __name__ == "__main__" and exc.name == "torch":
        print("P17 requires PyTorch. Install dependencies: python -m pip install -r requirements.txt", file=sys.stderr)
        raise SystemExit(2) from None
    raise


@dataclass(frozen=True)
class Config:
    encoder_layers: int = 2
    decoder_layers: int = 2
    window: int = 3  # Includes the current token: at most window-1 older keys.
    position_scale: float = 0.01
    uniform: bool = False  # Zero Q/K and identity V: hand-computable averages.

    def __post_init__(self):
        for name in ("encoder_layers", "decoder_layers", "window"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.position_scale):
            raise ValueError("position_scale must be finite")
        if type(self.uniform) is not bool:
            raise ValueError("uniform must be bool")


@dataclass(frozen=True)
class KV:
    k: torch.Tensor  # [B,T,2], one head, no compression
    v: torch.Tensor
    start: int  # Absolute position of the first row; not a ring-buffer index.


@dataclass(frozen=True)
class Cache:
    encoder_local: tuple
    decoder_global: tuple
    decoder_local: tuple
    next_position: int
    approximate: bool
    owner: object
    signature: tuple


@dataclass
class Work:
    encoder_token_layers: int = 0
    decoder_token_layers: int = 0
    global_kv_token_layers: int = 0  # One K+V projection pair per token/layer.

    @property
    def block_visits(self):
        return self.encoder_token_layers + self.decoder_token_layers


@dataclass(frozen=True)
class Result:
    output: torch.Tensor  # Full: [B,C,2]; bounded prefill: [B,R,2]. NOT logits.
    encoder: torch.Tensor  # Encoder output for all C supplied input vectors.
    output_start: int
    cache: Cache
    work: Work


class Projections(nn.Module):
    """Fixed initialized buffers, so this experiment needs no training/checkpoint."""
    def __init__(self, index, uniform=False, decoder=False):
        super().__init__()
        eye = torch.eye(2, dtype=torch.float64)
        q = torch.zeros_like(eye) if uniform else torch.tensor(
            [[0.3 + 0.03 * index, -0.1], [0.2, 0.25]], dtype=torch.float64)
        k = torch.zeros_like(eye) if uniform else torch.tensor(
            [[0.2, 0.1], [-0.15, 0.3 + 0.02 * index]], dtype=torch.float64)
        v = eye if uniform else torch.tensor(
            [[0.25, 0.05], [-0.05, 0.2 + 0.02 * index]], dtype=torch.float64)
        for name, tensor in (("q", q), ("k", k), ("v", v)):
            self.register_buffer(name, tensor.clone())
        if decoder:
            self.register_buffer("global_k", k.clone() * 0.7)
            self.register_buffer("global_v", v.clone() * (1.0 if uniform else 0.8))


def attention(q, kv, query_start, window=None):
    """Causal [B,C,2] x [B,K,2], with explicit absolute positions.

    This allocates dense scores even for local attention. It is not a sparse
    kernel, and counting useful paths is NOT counting executed FLOPs.
    """
    query_positions = torch.arange(query_start, query_start + q.shape[1])[:, None]
    key_positions = torch.arange(kv.start, kv.start + kv.k.shape[1])[None, :]
    allowed = key_positions <= query_positions
    if window is not None:
        allowed = allowed & (key_positions > query_positions - window)
    if not bool(allowed.any(dim=-1).all()):
        raise ValueError("each query must have a visible key")
    scores = (q @ kv.k.transpose(-2, -1)) / math.sqrt(2)
    return scores.masked_fill(~allowed, -torch.inf).softmax(-1) @ kv.v


def join_kv(k, v, start, old=None):
    if old is None:
        return KV(k, v, start)
    if old.start + old.k.shape[1] != start:
        raise ValueError("cache positions must be contiguous")
    return KV(torch.cat((old.k, k), dim=1), torch.cat((old.v, v), dim=1), old.start)


def tail_kv(kv, keep):
    size = kv.k.shape[1]
    n = min(size, keep)
    # size-n works for n=0, unlike -0 (which would keep the whole tensor).
    return KV(kv.k[:, size-n:].clone(), kv.v[:, size-n:].clone(), kv.start + size - n)


class ToyCED(nn.Module):
    """A dataflow experiment. Not a language model and not a production replica."""
    def __init__(self, config=Config()):
        super().__init__()
        self.config = config
        self.enc = nn.ModuleList(Projections(i, config.uniform) for i in range(config.encoder_layers))
        self.dec = nn.ModuleList(Projections(i + 3, config.uniform, True) for i in range(config.decoder_layers))
        self._cache_owner = object()

    def signature(self):
        return (self.config, tuple((name, id(t), t._version, tuple(t.shape), t.dtype, t.device)
                                   for name, t in self.named_buffers()))

    def validate(self, x, cache):
        if not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[-1] != 2:
            raise ValueError("input must have shape [B,T,2]")
        if min(x.shape) < 1 or x.dtype != torch.float64 or x.device.type != "cpu":
            raise ValueError("use nonempty CPU float64 input")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("input must be finite")
        if any(t.device.type != "cpu" or t.dtype != torch.float64 for t in self.buffers()):
            raise ValueError("toy weights must remain CPU float64")
        if cache is None:
            return
        if not isinstance(cache, Cache) or cache.owner is not self._cache_owner:
            raise ValueError("cache belongs to another model")
        if cache.signature != self.signature():
            raise ValueError("weights/config changed; start a fresh request")
        n = cache.next_position
        if type(n) is not int or n < 1 or type(cache.approximate) is not bool:
            raise ValueError("invalid cache metadata")
        groups = ((cache.encoder_local, len(self.enc), False),
                  (cache.decoder_global, len(self.dec), True),
                  (cache.decoder_local, len(self.dec), False))
        for group, expected, global_branch in groups:
            if len(group) != expected:
                raise ValueError("cache layer count mismatch")
            for kv in group:
                if not isinstance(kv, KV) or type(kv.start) is not int or kv.start < 0:
                    raise ValueError("invalid cache positions")
                if any(not isinstance(t, torch.Tensor) or t.ndim != 3 or
                       t.shape[0] != x.shape[0] or t.shape[2] != 2 or
                       t.dtype != x.dtype or t.device != x.device or
                       not bool(torch.isfinite(t).all()) for t in (kv.k, kv.v)):
                    raise ValueError("cache tensor shape/dtype/device/values mismatch")
                if kv.k.shape != kv.v.shape or kv.start + kv.k.shape[1] != n:
                    raise ValueError("cache endpoint mismatch")
                if global_branch:
                    if kv.start != 0:
                        raise ValueError("global cache must cover the complete prefix")
                elif kv.k.shape[1] != min(n, self.config.window - 1):
                    raise ValueError("local cache must keep exactly window-1 available rows")

    def encode(self, x, offset, old, work):
        positions = torch.arange(offset, offset + x.shape[1], dtype=x.dtype)
        positional = positions[:, None] * x.new_tensor([1.0, -1.0]) * self.config.position_scale
        h = x + positional
        local = []
        for i, layer in enumerate(self.enc):
            kv = join_kv(h @ layer.k, h @ layer.v, offset, None if old is None else old[i])
            h = h + attention(h @ layer.q, kv, offset, self.config.window)
            local.append(tail_kv(kv, self.config.window - 1))
            work.encoder_token_layers += x.shape[0] * x.shape[1]
        return h, tuple(local)

    def global_from_encoder(self, encoder, offset, old, work):
        global_kv = []
        for i, layer in enumerate(self.dec):
            # Architectural edge under test: input is encoder, NOT decoder h.
            kv = join_kv(encoder @ layer.global_k, encoder @ layer.global_v,
                         offset, None if old is None else old[i])
            global_kv.append(kv)
            work.global_kv_token_layers += encoder.shape[0] * encoder.shape[1]
        return tuple(global_kv)

    def decode(self, h, offset, global_kv, old, work):
        local = []
        for i, layer in enumerate(self.dec):
            # Local K/V depend on this decoder layer, unlike global K/V.
            kv = join_kv(h @ layer.k, h @ layer.v, offset, None if old is None else old[i])
            q = h @ layer.q
            global_out = attention(q, global_kv[i], offset)
            local_out = attention(q, kv, offset, self.config.window)
            h = h + global_out + local_out
            local.append(tail_kv(kv, self.config.window - 1))
            work.decoder_token_layers += h.shape[0] * h.shape[1]
        return h, tuple(local)

    @torch.no_grad()
    def forward(self, x, cache=None):
        """Exact full/chunked path of THIS toy; cache=None starts a new request."""
        self.validate(x, cache)
        offset = 0 if cache is None else cache.next_position
        work = Work()
        encoder, enc_local = self.encode(x, offset, None if cache is None else cache.encoder_local, work)
        global_kv = self.global_from_encoder(encoder, offset, None if cache is None else cache.decoder_global, work)
        output, dec_local = self.decode(encoder, offset, global_kv, None if cache is None else cache.decoder_local, work)
        new = Cache(enc_local, global_kv, dec_local, offset + x.shape[1],
                    False if cache is None else cache.approximate, self._cache_owner, self.signature())
        return Result(output, encoder, offset, new, work)

    @torch.no_grad()
    def bounded_prefill(self, x):
        """Encode ALL prompt vectors, decode only last W; intentionally APPROXIMATE.

        Preserves exact encoder/global states, discards older decoder local
        context. Does not implement production encoder replay or prefix offload.
        """
        self.validate(x, None)
        work = Work()
        encoder, enc_local = self.encode(x, 0, None, work)
        global_kv = self.global_from_encoder(encoder, 0, None, work)
        replay = min(x.shape[1], self.config.window)
        start = x.shape[1] - replay
        output, dec_local = self.decode(encoder[:, start:], start, global_kv, None, work)
        new = Cache(enc_local, global_kv, dec_local, x.shape[1], start > 0,
                    self._cache_owner, self.signature())
        return Result(output, encoder, start, new, work)


def path_budget(tokens, encoder_layers, decoder_layers, window, batch=1):
    """Transparent block-visit and K+V-projection-pair counts, NOT FLOPs/time."""
    for number in (tokens, encoder_layers, decoder_layers, window, batch):
        if type(number) is not int or number < 1:
            raise ValueError("budget arguments must be positive integers")
    replay = min(tokens, window)
    return {"full_blocks": batch * tokens * (encoder_layers + decoder_layers),
            "bounded_blocks": batch * (tokens * encoder_layers + replay * decoder_layers),
            "global_projection_pairs_each_path": batch * tokens * decoder_layers,
            "replay_tokens": replay}


def hand_walkthrough():
    """W=2, E=D=1, uniform weights, x=[1,2,3] in the first coordinate."""
    model = ToyCED(Config(1, 1, 2, 0.0, True))
    x = torch.tensor([[[1., 0.], [2., 0.], [3., 0.]]], dtype=torch.float64)
    full, bounded = model(x), model.bounded_prefill(x)
    return full, bounded


def self_test():
    full, bounded = hand_walkthrough()
    torch.testing.assert_close(full.encoder[0, :, 0], torch.tensor([2., 3.5, 5.5], dtype=torch.float64))
    torch.testing.assert_close(full.output[0, :, 0], torch.tensor([6., 9., 41/3], dtype=torch.float64))
    torch.testing.assert_close(bounded.output[0, :, 0], torch.tensor([9.75, 41/3], dtype=torch.float64))
    two_layer = ToyCED(Config(1, 2, 2, 0.0, True))
    hand_x = torch.tensor([[[1., 0.], [2., 0.], [3., 0.]]], dtype=torch.float64)
    exact_two, replay_two = two_layer(hand_x), two_layer.bounded_prefill(hand_x)
    torch.testing.assert_close(exact_two.output[0, -1, 0], torch.tensor(86/3, dtype=torch.float64))
    torch.testing.assert_close(replay_two.output[0, -1, 0], torch.tensor(697/24, dtype=torch.float64))
    model = ToyCED()
    x = torch.tensor([[[1., .5], [-1., 2.], [3., -.5], [.5, 1.], [2., -2.], [1., 3.]]], dtype=torch.float64)
    full = model(x)
    first, second = model(x[:, :2]), None
    second = model(x[:, 2:], first.cache)
    torch.testing.assert_close(torch.cat((first.output, second.output), 1), full.output, rtol=1e-12, atol=1e-12)
    for cut in range(1, x.shape[1]):
        changed = x.clone(); changed[:, cut:] += 17
        torch.testing.assert_close(model(changed).output[:, :cut], full.output[:, :cut], rtol=0, atol=0)
    bounded = model.bounded_prefill(x)
    if (bounded.output[:, -1] - full.output[:, -1]).abs().max().item() <= 1e-8:
        raise AssertionError("replay example must expose an approximation difference")
    budget = path_budget(6, 2, 2, 3)
    assert full.work.block_visits == budget["full_blocks"] == 24
    assert bounded.work.block_visits == budget["bounded_blocks"] == 18
    assert full.work.global_kv_token_layers == bounded.work.global_kv_token_layers == 12
    return model, x, full, bounded


def main():
    model, x, full, bounded = self_test()
    hand, hand_bounded = hand_walkthrough()
    print("P17 CED 数据流教学实验 | CPU float64 | 不加载真实权重，不测速度")
    print("手算 encoder:", hand.encoder[0, :, 0].tolist())
    print("手算 full decoder:", hand.output[0, :, 0].tolist())
    print("手算 bounded 后缀:", hand_bounded.output[0, :, 0].tolist())
    print("形状 input / encoder / output:", tuple(x.shape), tuple(full.encoder.shape), tuple(full.output.shape))
    print("每层 global K/V:", tuple(full.cache.decoder_global[0].k.shape), "来源=encoder 最终输出")
    print("每层 local K/V:", tuple(full.cache.decoder_local[0].k.shape), "绝对起点=", full.cache.decoder_local[0].start)
    print("完整 / 有界 replay 的 block visits:", full.work.block_visits, bounded.work.block_visits)
    print("两条路径各自 global K+V 投影对计数:", full.work.global_kv_token_layers)
    print("有界 replay 的输出范围:", bounded.output_start, "到", bounded.cache.next_position - 1)
    print("最后位置 full / replay 最大差:", f"{(full.output[:, -1] - bounded.output[:, -1]).abs().max().item():.9f}")
    print("self-test: PASS（仅本教学模型；replay 近似，不宣称等价或加速比）")


if __name__ == "__main__":
    main()
