"""P18 / S5: replace S4's SwiGLU FFN with a CPU teaching MoE.

Run: OMP_NUM_THREADS=1 python code/19_moe.py
Softmax routing, explicit selected-weight normalization, shared SwiGLU experts,
load diagnostics, and separate auxiliary-loss / selection-bias demonstrations.
This is NOT DeepSeekMoE/V3 reproduction, distributed training, or a benchmark.
"""
from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path
import sys

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:
    if __name__ == '__main__' and exc.name == 'torch':
        print('P18 requires PyTorch. Install dependencies: python -m pip install -r requirements.txt',
              file=sys.stderr)
        raise SystemExit(2) from None
    raise

_NAME = 'zero2llm_s4'
if _NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(_NAME, Path(__file__).with_name('15_grouped_kv.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[_NAME] = module
    spec.loader.exec_module(module)
s4 = sys.modules[_NAME]


def nonnegative_int(name, value):
    if type(value) is not int or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')


def finite_nonnegative(name, value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')


def validate_routing(num_experts, top_k, capacity, balance_mode):
    s4.s3.positive_int('num_experts', num_experts)
    s4.s3.positive_int('top_k', top_k)
    if top_k > num_experts:
        raise ValueError('top_k must not exceed num_experts')
    if capacity is not None:
        nonnegative_int('capacity', capacity)
    if balance_mode not in ('none', 'aux', 'bias'):
        raise ValueError('balance_mode must be none, aux, or bias')


@dataclass(frozen=True)
class RouteInfo:
    indices: torch.Tensor       # [N,k], selected expert IDs
    weights: torch.Tensor       # [N,k], BEFORE dropping, each row sums to one
    kept: torch.Tensor          # [N,k], accepted assignment mask
    requested: torch.Tensor     # [E], before capacity
    accepted: torch.Tensor      # [E], after capacity
    aux_loss: torch.Tensor      # scalar, unscaled, batch-level teaching loss


def route(logits, top_k, balance_bias=None, capacity=None):
    """Stable top-k: higher affinity first, equal scores use lower expert ID.

    Bias affects SELECTION ONLY. Combine weights normalize original logits over
    selected experts. This is softmax teaching routing, not V3's sigmoid affinity.
    Capacity is an integer per expert per invocation, accepted in flattened token
    order. Dropped branches are zero, WITHOUT re-normalizing surviving weights.
    """
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 2 or
            not logits.is_floating_point() or logits.shape[1] <= 0):
        raise ValueError('logits must be floating [N,E] with E > 0')
    N, E = logits.shape
    validate_routing(E, top_k, capacity, 'none')
    if not torch.isfinite(logits).all():
        raise ValueError('router logits must be finite')
    work = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    probs = work.softmax(-1)
    # Softmax is monotone, but may underflow distinct logits to identical zeros.
    # Without a nonzero bias, sort the logits themselves to retain their order.
    scores = work
    if balance_bias is not None:
        if (not isinstance(balance_bias, torch.Tensor) or balance_bias.shape != (E,) or
                not balance_bias.is_floating_point() or balance_bias.device != logits.device or
                not torch.isfinite(balance_bias).all()):
            raise ValueError('balance_bias must be finite floating [E] on logits device')
        if torch.count_nonzero(balance_bias):
            scores = probs + balance_bias.to(work.dtype)
        if not torch.isfinite(scores).all():
            raise ValueError('biased routing scores must be finite')
    indices = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :top_k]
    # Equivalent to selected probs / selected probs.sum, without 0/0 when
    # an explicit selection bias chooses logits whose full softmax underflows.
    weights = work.gather(1, indices).softmax(-1)
    requested = torch.bincount(indices.reshape(-1), minlength=E)
    kept = torch.ones_like(indices, dtype=torch.bool)
    if capacity is not None:
        for expert_id in range(E):
            positions = (indices == expert_id).nonzero(as_tuple=False)
            rejected = positions[capacity:]
            kept[rejected[:, 0], rejected[:, 1]] = False
    accepted = torch.bincount(indices[kept], minlength=E)
    aux = (E * (requested.to(work.dtype) / (N * top_k) * probs.mean(0)).sum()
           if N else work.sum() * 0)
    return RouteInfo(indices, weights, kept, requested, accepted, aux)


class TeachingMoE(nn.Module):
    """Bias-free SwiGLU experts; actual sparse dispatch, no all-expert evaluation.

    Input [...,D]. Shared experts run for all tokens and are SUMMED, not averaged.
    The enclosing Transformer block owns the residual; this module does not add x.
    last_info describes only the latest call and is not persistent request state.
    """
    def __init__(self, d_model, num_experts=4, top_k=2, expert_width=16,
                 shared_experts=1, shared_width=16, balance_mode='aux', capacity=None):
        super().__init__()
        for name, value in (('d_model', d_model), ('expert_width', expert_width),
                            ('shared_width', shared_width)):
            s4.s3.positive_int(name, value)
        nonnegative_int('shared_experts', shared_experts)
        validate_routing(num_experts, top_k, capacity, balance_mode)
        self.d_model = d_model
        self.top_k = top_k
        self.capacity = capacity
        self.balance_mode = balance_mode
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([s4.s3.SwiGLU(d_model, expert_width) for _ in range(num_experts)])
        self.shared = nn.ModuleList([s4.s3.SwiGLU(d_model, shared_width) for _ in range(shared_experts)])
        self.register_buffer('balance_bias', torch.zeros(num_experts))
        self.last_info = None

    def forward(self, x):
        validate_routing(len(self.experts), self.top_k, self.capacity, self.balance_mode)
        if (not isinstance(x, torch.Tensor) or x.ndim < 2 or
                not x.is_floating_point() or x.shape[-1] != self.d_model or
                x.dtype != self.router.weight.dtype or x.device != self.router.weight.device):
            raise ValueError('MoE expects floating [...,D] matching model dtype/device')
        if not torch.isfinite(x).all():
            raise ValueError('MoE input must be finite')
        flat = x.reshape(-1, self.d_model)
        info = route(self.router(flat), self.top_k,
                     self.balance_bias if self.balance_mode == 'bias' else None, self.capacity)
        output = flat * 0
        for expert_id, expert in enumerate(self.experts):
            token_ids, slots = ((info.indices == expert_id) & info.kept).nonzero(as_tuple=True)
            if token_ids.numel() == 0:
                continue
            values = expert(flat.index_select(0, token_ids))
            values = values * info.weights[token_ids, slots, None].to(values.dtype)
            output = output.index_add(0, token_ids, values)
        for expert in self.shared:
            output = output + expert(flat)
        if not torch.isfinite(output).all():
            raise ValueError('MoE expert output must be finite')
        self.last_info = info
        return output.reshape_as(x)

    @torch.no_grad()
    def update_balance_bias(self, requested, step=0.01):
        """Explicit end-of-step demonstration; NEVER called by forward/cache.

        Use pre-drop requested loads. Overload lowers bias; underload raises it.
        Counts need to be aggregated externally for multi-batch/multi-device use.
        """
        if self.balance_mode != 'bias':
            raise ValueError('bias update requires balance_mode=bias')
        finite_nonnegative('step', step)
        E = len(self.experts)
        if (not isinstance(requested, torch.Tensor) or requested.shape != (E,) or
                requested.dtype not in (torch.int32, torch.int64) or
                requested.device != self.balance_bias.device or (requested < 0).any()):
            raise ValueError('requested must be nonnegative integer [E] on bias device')
        loads = requested.to(self.balance_bias.dtype)
        updated = self.balance_bias + step * torch.sign(loads.mean() - loads)
        if not torch.isfinite(updated).all():
            raise ValueError('updated balance bias must be finite')
        self.balance_bias.copy_(updated)


def parameter_budget(d_model, num_experts, top_k, expert_width, shared_experts=0, shared_width=1):
    """MoE-module weights only: total and per-token active parameter budgets.

    Includes the full router and all shared experts, excludes normalization,
    attention, embeddings, output head, buffers, FLOPs, KV and communication.
    Active budget assumes no drops, and is NOT a measured runtime/memory number.
    """
    for name, value in (('d_model', d_model), ('expert_width', expert_width), ('shared_width', shared_width)):
        s4.s3.positive_int(name, value)
    nonnegative_int('shared_experts', shared_experts)
    validate_routing(num_experts, top_k, None, 'none')
    expert = 3 * d_model * expert_width
    shared = shared_experts * 3 * d_model * shared_width
    router = d_model * num_experts
    return {'expert': expert, 'router': router, 'shared': shared,
            'total': num_experts * expert + shared + router,
            'active': top_k * expert + shared + router}


@dataclass(frozen=True)
class MoEConfig(s4.GroupedConfig):
    num_experts: int = 4
    top_k: int = 2
    expert_width: int = 16
    shared_experts: int = 1
    shared_width: int = 16
    balance_mode: str = 'aux'
    aux_weight: float = 0.01

    def __post_init__(self):
        super().__post_init__()
        validate_routing(self.num_experts, self.top_k, None, self.balance_mode)
        for name in ('expert_width', 'shared_width'):
            s4.s3.positive_int(name, getattr(self, name))
        nonnegative_int('shared_experts', self.shared_experts)
        finite_nonnegative('aux_weight', self.aux_weight)
        if self.ffn_kind != 'swiglu' or self.ffn_bias or self.intermediate_size is not None:
            raise ValueError('S5 requires bias-free swiglu; use expert_width/shared_width, not intermediate_size')


class MoEGPT(s4.GroupedGPT):
    """S5 retains S4 attention, norms, residuals, logits, and compact GQA cache.

    Dropless LM only: per-call capacity would depend on batching/chunk boundaries.
    Auxiliary loss is added only when targets exist AND balance_mode is aux.
    Router-bias updates are explicit and invalidate existing request caches.
    """
    def __init__(self, config):
        if not isinstance(config, MoEConfig):
            raise ValueError('MoEGPT requires MoEConfig')
        super().__init__(config)
        for block in self.blocks:
            block.mlp = TeachingMoE(config.d_model, config.num_experts, config.top_k,
                config.expert_width, config.shared_experts, config.shared_width, config.balance_mode)

    def validate_lm_routing(self):
        for block in self.blocks:
            if not isinstance(block.mlp, TeachingMoE):
                raise ValueError('S5 requires TeachingMoE FFN modules')
            if block.mlp.capacity is not None:
                raise ValueError('capacity dropping is standalone only; S5 LM must be dropless')

    def cache_extra_signature(self):
        # S2's existing hook also covers module types, norm eps and GELU settings.
        runtime = tuple((b.mlp.top_k, b.mlp.capacity, b.mlp.balance_mode)
                        for b in self.blocks)
        buffers = tuple((name, id(value), value._version, value.dtype, value.device)
                        for name, value in self.named_buffers())
        return super().cache_extra_signature(), runtime, buffers

    def forward(self, idx, targets=None):
        self.validate_lm_routing()
        logits, loss = super().forward(idx, targets)
        if loss is not None:
            terms = [b.mlp.last_info.aux_loss for b in self.blocks if b.mlp.balance_mode == 'aux']
            if terms:
                loss = loss + self.config.aux_weight * torch.stack(terms).mean()
        return logits, loss


def cached_forward(model, idx, cache=None):
    if not isinstance(model, MoEGPT):
        raise ValueError('S5 cached_forward requires MoEGPT')
    model.validate_lm_routing()
    return s4.cached_forward(model, idx, cache)


def routing_walkthrough(capacity=None):
    """Rows sum to 1 before log: exact hand-routing fixture in the chapter."""
    probs = torch.tensor([[.4, .3, .2, .1], [.1, .2, .3, .4], [.4, .4, .1, .1]], dtype=torch.float64)
    return route(probs.log(), 2, capacity=capacity)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(180)
    print(f'P18 / S5 CPU teaching MoE; torch={torch.__version__}; seed=180')
    for capacity in (None, 1):
        info = routing_walkthrough(capacity)
        print(f'capacity={capacity}; selected={info.indices.tolist()}')
        print('weights=', [[round(v, 6) for v in row] for row in info.weights.tolist()])
        print(f'requested={info.requested.tolist()}; accepted={info.accepted.tolist()}; aux={info.aux_loss.item():.6f}')
    print('module budget:', parameter_budget(4, 4, 2, 2, 1, 2))
    config = MoEConfig(vocab_size=17, d_model=16, num_heads=4, num_kv_heads=2,
                       num_layers=2, max_seq_len=16, expert_width=7, shared_width=5)
    model = MoEGPT(config)
    tokens = torch.tensor([[1, 2, 3, 2, 6, 7], [4, 0, 2, 1, 4, 3]])
    _, loss = model(tokens, tokens.roll(-1, 1))  # Synthetic gradient smoke, not a corpus/evaluation.
    loss.backward()
    print('finite training loss/gradients:', bool(torch.isfinite(loss)) and
          all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()))
    for i, block in enumerate(model.blocks):
        print(f'layer {i} requested load:', block.mlp.last_info.requested.tolist())
    model.eval()
    full = model(tokens)[0]
    first, cache = cached_forward(model, tokens[:, :2])
    last, cache = cached_forward(model, tokens[:, 2:], cache)
    print(f'full/cache maximum absolute error: {(full - torch.cat((first, last), 1)).abs().max().item():.3g}')
    print(f'compact KV payload bytes: {s4.cache_bytes(cache)}; no GPU timing measured')


if __name__ == '__main__':
    main()
