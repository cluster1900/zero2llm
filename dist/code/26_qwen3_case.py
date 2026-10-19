"""P25: pinned Qwen3 metadata and a separate, tiny CPU mechanism probe.

python code/26_qwen3_case.py             # standard library, offline metadata
python code/26_qwen3_case.py --torch      # optional PyTorch + Jinja2 exercises
No pretrained weights or BPE vocabulary are loaded. The tiny model is NOT a
checkpoint implementation, a converter, a tokenizer, or an inference benchmark.
"""
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys

DATA = Path(__file__).with_name('data') / 'qwen3'
PINS = {
    'dense': ('Qwen/Qwen3-0.6B', 'c1899de289a04d12100db370d81485cdf75e47ca'),
    'moe': ('Qwen/Qwen3-30B-A3B', 'ad44e777bcd18fa416d9da3bd8f70d33ebb85d39'),
}


def integer(name, value, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def load_checkpoint(kind, root=DATA):
    """Read ONLY local pinned bytes; hash validation precedes JSON consumption."""
    if kind not in PINS:
        raise ValueError('kind must be dense or moe')
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    record = manifest['models'][kind]
    if (record['repo_id'], record['revision']) != PINS[kind]:
        raise ValueError('checkpoint revision does not match the explicit pin')
    result = {'provenance': record, 'implementation': manifest['implementation']}
    for name in ('config.json', 'tokenizer_config.json', 'generation_config.json', 'LICENSE'):
        raw = (root / kind / name).read_bytes()
        entry = record['files'][name]
        expected_url = f'https://huggingface.co/{record["repo_id"]}/resolve/{record["revision"]}/{name}'
        if (entry['url'] != expected_url or len(raw) != entry['bytes'] or
                hashlib.sha256(raw).hexdigest() != entry['sha256']):
            raise ValueError(f'fixture integrity failure: {kind}/{name}')
        if name.endswith('.json'):
            result[name[:-5]] = json.loads(raw)
    validate_config(result['config'])
    return result


def validate_config(c):
    """Reject unsupported assumptions rather than silently count another family."""
    if c.get('model_type') not in ('qwen3', 'qwen3_moe'):
        raise ValueError('only selected qwen3 / qwen3_moe architectures supported')
    for name in ('hidden_size', 'num_hidden_layers', 'num_attention_heads',
                 'num_key_value_heads', 'head_dim', 'intermediate_size',
                 'vocab_size', 'max_position_embeddings'):
        integer(name, c.get(name))
    if c['head_dim'] % 2 or c['num_attention_heads'] % c['num_key_value_heads']:
        raise ValueError('even head_dim and divisible GQA head counts required')
    if (c.get('attention_bias') is not False or c.get('hidden_act') != 'silu' or
            c.get('use_sliding_window') is not False or c.get('rope_scaling') is not None):
        raise ValueError('this probe supports bias-free, SiLU, full attention, unscaled RoPE only')
    if type(c.get('tie_word_embeddings')) is not bool:
        raise ValueError('explicit tie_word_embeddings required')
    for name in ('rms_norm_eps', 'rope_theta'):
        if type(c.get(name)) not in (int, float) or not math.isfinite(c[name]) or c[name] <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if c['model_type'] == 'qwen3_moe':
        for name in ('num_experts', 'num_experts_per_tok', 'moe_intermediate_size',
                     'decoder_sparse_step'):
            integer(name, c.get(name))
        if c['num_experts_per_tok'] > c['num_experts'] or c.get('norm_topk_prob') is not True:
            raise ValueError('valid top-k and normalized routing required')
        if (not isinstance(c.get('mlp_only_layers'), list) or
                any(type(i) is not int or not 0 <= i < c['num_hidden_layers']
                    for i in c['mlp_only_layers'])):
            raise ValueError('mlp_only_layers must contain valid zero-based layer indices')


def is_moe_layer(c, layer):
    return (c['model_type'] == 'qwen3_moe' and layer not in c['mlp_only_layers']
            and (layer + 1) % c['decoder_sparse_step'] == 0)


@dataclass(frozen=True)
class Parameter:
    name: str
    shape: tuple
    component: str
    @property
    def count(self):
        return math.prod(self.shape)


def parameter_inventory(c):
    """Unique ARCHITECTURAL parameters, not bytes in a safetensors checkpoint.

    Shared embedding/output weight is counted once. Q/K norm vectors have d
    entries shared across heads, not H*d entries. No weights are allocated.
    """
    validate_config(c)
    D, V, d = c['hidden_size'], c['vocab_size'], c['head_dim']
    H, K = c['num_attention_heads'], c['num_key_value_heads']
    rows = [Parameter('embed_tokens.weight', (V, D), 'embedding')]
    if not c['tie_word_embeddings']:
        rows.append(Parameter('lm_head.weight', (V, D), 'output_head'))
    for i in range(c['num_hidden_layers']):
        prefix = f'layers.{i}.'
        for name, shape in (('q_proj', (H*d, D)), ('k_proj', (K*d, D)),
                            ('v_proj', (K*d, D)), ('o_proj', (D, H*d))):
            rows.append(Parameter(prefix + 'self_attn.' + name + '.weight', shape, 'attention'))
        for name in ('q_norm', 'k_norm'):
            rows.append(Parameter(prefix + 'self_attn.' + name + '.weight', (d,), 'qk_norm'))
        for name in ('input_layernorm', 'post_attention_layernorm'):
            rows.append(Parameter(prefix + name + '.weight', (D,), 'block_norm'))
        if is_moe_layer(c, i):
            rows.append(Parameter(prefix + 'mlp.gate.weight', (c['num_experts'], D), 'router'))
            experts = [(f'mlp.experts.{e}.', c['moe_intermediate_size'])
                       for e in range(c['num_experts'])]
        else:
            experts = [('mlp.', c['intermediate_size'])]
        for expert, width in experts:
            for name, shape in (('gate_proj', (width, D)), ('up_proj', (width, D)),
                                ('down_proj', (D, width))):
                rows.append(Parameter(prefix + expert + name + '.weight', shape,
                                      'experts' if is_moe_layer(c, i) else 'dense_ffn'))
    rows.append(Parameter('norm.weight', (D,), 'final_norm'))
    return tuple(rows)


def parameter_budget(c):
    """Report total + an EXPLICIT per-token active-matrix convention.

    Active counts whole embedding/head matrices (including unvisited rows), all
    attention/norm/router weights, and k experts per sparse layer. It is not
    arithmetic operations, actual fetched bytes, or batch-wide union of experts.
    """
    counts = defaultdict(int)
    for row in parameter_inventory(c):
        counts[row.component] += row.count
    total = sum(counts.values())
    active = total
    if c['model_type'] == 'qwen3_moe':
        active -= counts['experts']
        active += counts['experts'] // c['num_experts'] * c['num_experts_per_tok']
    return {'components': dict(counts), 'unique_total': total,
            'non_embedding': total - counts['embedding'] - counts['output_head'],
            'active_matrix_convention': active}


def cache_bytes(c, batch, length, element_bytes=2):
    """Logical unquantized K AND V payload, all layers, compact KV heads only."""
    validate_config(c)
    integer('batch', batch)
    integer('length', length, 0)
    integer('element_bytes', element_bytes)
    if length > c['max_position_embeddings']:
        raise ValueError('length exceeds configured position limit')
    return (2 * c['num_hidden_layers'] * batch * length *
            c['num_key_value_heads'] * c['head_dim'] * element_bytes)


def component_table(record):
    c, p, impl = record['config'], record['provenance'], record['implementation']
    config_url = p['files']['config.json']['url']
    source = impl['files'][c['model_type']]['url']
    return [
        ('身份', p['repo_id'] + '@' + p['revision'], config_url),
        ('形状', f"D={c['hidden_size']}; L={c['num_hidden_layers']}; Hq/Hkv/d="
         f"{c['num_attention_heads']}/{c['num_key_value_heads']}/{c['head_dim']}", config_url),
        ('RoPE', f"theta={c['rope_theta']}; scaling={c['rope_scaling']}", config_url),
        ('QK-norm / RMSNorm', f"每头宽度 d 的 Q/K RMSNorm；eps={c['rms_norm_eps']}", source),
        ('FFN', (f"SwiGLU experts={c['num_experts']}; top-k={c['num_experts_per_tok']}; "
                  f"width={c['moe_intermediate_size']}; shared=0"
                  if c['model_type'] == 'qwen3_moe' else
                  f"dense SwiGLU width={c['intermediate_size']}"), source),
        ('模板', 'tokenizer_config.chat_template；不是裸字符串拼接/BPE实现',
         p['files']['tokenizer_config.json']['url']),
    ]


def render_prompt(record, user_text, enable_thinking=True):
    """Execute the PINNED official template for one user turn, without BPE.

    Requires existing requirements.txt's Jinja2. No tools/multimodal/tokenization
    API is claimed. `<think>` in text is not by itself a reasoning guarantee.
    """
    if not isinstance(user_text, str) or not user_text or type(enable_thinking) is not bool:
        raise ValueError('nonempty user text and boolean enable_thinking required')
    sandbox = importlib.import_module('jinja2.sandbox')
    env = sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    template = env.from_string(record['tokenizer_config']['chat_template'])
    return template.render(messages=[{'role': 'user', 'content': user_text}],
                           tools=None, add_generation_prompt=True,
                           enable_thinking=enable_thinking)


def completion_budget(ids, max_new_tokens, enable_thinking=True, think_end=151668):
    """Count actual generated IDs including delimiters; never manufacture an answer.

    This is an accounting exercise, NOT a thinking-budget controller. For thinking
    mode, no closing token means every observed ID is an unfinished thought and
    zero answer IDs. Non-thinking's empty block is IN the prompt, not generated.
    """
    integer('max_new_tokens', max_new_tokens, 0)
    integer('think_end', think_end, 0)
    if type(enable_thinking) is not bool or not isinstance(ids, (list, tuple)):
        raise ValueError('boolean mode and token-ID list/tuple required')
    for token in ids:
        integer('token ID', token, 0)
    if len(ids) > max_new_tokens:
        raise ValueError('generated IDs exceed max_new_tokens')
    split = ids.index(think_end) if enable_thinking and think_end in ids else None
    if not enable_thinking:
        reasoning, control, answer, complete = 0, 0, len(ids), True
    elif split is None:
        reasoning, control, answer, complete = len(ids), 0, 0, False
    else:
        reasoning, control, answer, complete = split, 1, len(ids)-split-1, True
    return {'reasoning_ids': reasoning, 'closing_ids': control, 'answer_ids': answer,
            'generated_ids': len(ids), 'remaining': max_new_tokens-len(ids),
            'thinking_closed': complete}


def context_budget(prompt_ids, max_new_tokens, context_limit):
    for name, value, minimum in (('prompt_ids', prompt_ids, 1),
                                 ('max_new_tokens', max_new_tokens, 0),
                                 ('context_limit', context_limit, 1)):
        integer(name, value, minimum)
    if prompt_ids + max_new_tokens > context_limit:
        raise ValueError('prompt plus output budget exceeds context limit')
    return context_limit - prompt_ids - max_new_tokens


def tiny_config(kind='dense'):
    c = deepcopy(load_checkpoint(kind)['config'])
    c.update(hidden_size=8, vocab_size=17, num_hidden_layers=2,
             num_attention_heads=4, num_key_value_heads=2, head_dim=4,
             intermediate_size=12, max_position_embeddings=64)
    if kind == 'moe':
        c.update(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=6)
    return c


def rms_norm(x, gain, eps):
    # Float64 stays float64 for independent numerical checks. Production BF16
    # kernels and their accumulation/casting behavior are NOT reproduced here.
    return x * (x.square().mean(-1, keepdim=True) + eps).rsqrt() * gain


def split_half_rope(x, offset, theta):
    torch = importlib.import_module('torch')
    d = x.shape[-1]
    positions = torch.arange(offset, offset+x.shape[-2], dtype=x.dtype, device=x.device)
    inv = theta ** (-torch.arange(0, d, 2, dtype=x.dtype, device=x.device) / d)
    angles = positions[:, None] * inv[None, :]
    angles = torch.cat((angles, angles), -1)
    rotated = torch.cat((-x[..., d//2:], x[..., :d//2]), -1)
    return x * angles.cos() + rotated * angles.sin()


def grouped_attention(q, k, v, offset):
    torch = importlib.import_module('torch')
    repeats = q.shape[1] // k.shape[1]
    keys = k.repeat_interleave(repeats, dim=1)
    values = v.repeat_interleave(repeats, dim=1)
    allowed = torch.arange(k.shape[2], device=q.device)[None, :] <= (
        offset + torch.arange(q.shape[2], device=q.device)[:, None])
    scores = q @ keys.transpose(-1, -2) / math.sqrt(q.shape[-1])
    return scores.masked_fill(~allowed, float('-inf')).softmax(-1) @ values


@dataclass(frozen=True)
class TinyCache:
    layers: tuple
    owner: object
    signature: tuple
    @property
    def length(self):
        return self.layers[0][0].shape[2]


def build_tiny_model(config):
    """Small mechanism model with explicit head_dim, no pretrained state loading.

    Eval-only equal-length unpadded batches. No quantization, training loss,
    auxiliary router objective, dynamic RoPE, fused kernel, or tensor parallelism.
    """
    validate_config(config)
    c = deepcopy(config)
    if parameter_budget(c)['unique_total'] > 100000 or c['max_position_embeddings'] > 128:
        raise ValueError('tiny probe refuses real checkpoint-sized allocations')
    torch = importlib.import_module('torch')
    nn = torch.nn
    D, d, H, K = (c[k] for k in ('hidden_size', 'head_dim', 'num_attention_heads', 'num_key_value_heads'))

    class Norm(nn.Module):
        def __init__(self, width):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(width))
        def forward(self, x):
            return rms_norm(x, self.weight, c['rms_norm_eps'])

    class FFN(nn.Module):
        def __init__(self, width):
            super().__init__()
            self.gate_proj = nn.Linear(D, width, bias=False)
            self.up_proj = nn.Linear(D, width, bias=False)
            self.down_proj = nn.Linear(width, D, bias=False)
        def forward(self, x):
            return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

    class MoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(D, c['num_experts'], bias=False)
            self.experts = nn.ModuleList([FFN(c['moe_intermediate_size']) for _ in range(c['num_experts'])])
        def forward(self, x):
            flat = x.reshape(-1, D)
            logits = self.gate(flat)
            # Stable ties are a teaching convention, not the official topk's tie contract.
            selected = torch.argsort(logits, dim=-1, descending=True, stable=True)[:, :c['num_experts_per_tok']]
            weights = logits.gather(1, selected).softmax(-1)
            out = torch.zeros_like(flat)
            for expert_id, expert in enumerate(self.experts):
                row, slot = torch.where(selected == expert_id)
                if row.numel():
                    out.index_add_(0, row, expert(flat[row]) * weights[row, slot, None])
            return out.reshape_as(x)

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(D, H*d, bias=False)
            self.k_proj = nn.Linear(D, K*d, bias=False)
            self.v_proj = nn.Linear(D, K*d, bias=False)
            self.o_proj = nn.Linear(H*d, D, bias=False)
            self.q_norm, self.k_norm = Norm(d), Norm(d)
        def project(self, x, offset):
            B, T, _ = x.shape
            q = self.q_norm(self.q_proj(x).view(B, T, H, d)).transpose(1, 2)
            k = self.k_norm(self.k_proj(x).view(B, T, K, d)).transpose(1, 2)
            v = self.v_proj(x).view(B, T, K, d).transpose(1, 2)
            return split_half_rope(q, offset, c['rope_theta']), split_half_rope(k, offset, c['rope_theta']), v
        def forward(self, x, old, offset):
            q, k, v = self.project(x, offset)
            if old is not None:
                k = torch.cat((old[0], k), 2)
                v = torch.cat((old[1], v), 2)
            y = grouped_attention(q, k, v, offset)
            return self.o_proj(y.transpose(1, 2).reshape(x.shape[0], x.shape[1], H*d)), (k, v)

    class Block(nn.Module):
        def __init__(self, index):
            super().__init__()
            self.self_attn = Attention()
            self.input_layernorm, self.post_attention_layernorm = Norm(D), Norm(D)
            self.mlp = MoE() if is_moe_layer(c, index) else FFN(c['intermediate_size'])
        def forward(self, x, old, offset):
            y, cache = self.self_attn(self.input_layernorm(x), old, offset)
            x = x + y
            return x + self.mlp(self.post_attention_layernorm(x)), cache

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = deepcopy(c)
            self.embed_tokens = nn.Embedding(c['vocab_size'], D)
            self.layers = nn.ModuleList([Block(i) for i in range(c['num_hidden_layers'])])
            self.norm = Norm(D)
            self.lm_head = nn.Linear(D, c['vocab_size'], bias=False)
            if c['tie_word_embeddings']:
                self.lm_head.weight = self.embed_tokens.weight
        def signature(self):
            return tuple((id(p), p._version, p.dtype, p.device) for p in self.parameters())
        @torch.no_grad()
        def forward(self, ids, cache=None):
            if self.training:
                raise ValueError('tiny probe requires eval mode')
            if self.config != c:
                raise ValueError('configuration is immutable; construct a new tiny model')
            if (not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.dtype != torch.long
                    or min(ids.shape) <= 0 or (ids < 0).any() or (ids >= c['vocab_size']).any()
                    or ids.device != self.embed_tokens.weight.device):
                raise ValueError('nonempty valid long token IDs [B,T] on model device required')
            offset = 0
            if cache is not None:
                if not isinstance(cache, TinyCache) or cache.owner is not self:
                    raise ValueError('cache belongs to another model/request')
                if cache.signature != self.signature():
                    raise ValueError('weights or dtype changed; reset cache')
                if not isinstance(cache.layers, tuple) or len(cache.layers) != len(self.layers):
                    raise ValueError('wrong cache layer count')
                if any(not isinstance(pair, tuple) or len(pair) != 2 or
                       any(not isinstance(t, torch.Tensor) or t.ndim != 4 for t in pair)
                       for pair in cache.layers):
                    raise ValueError('cache requires rank-four K/V pairs')
                offset = cache.length
                expected = (ids.shape[0], K, offset, d)
                if offset <= 0 or any(tuple(t.shape) != expected or
                        t.dtype != self.embed_tokens.weight.dtype or t.device != ids.device
                        or not torch.isfinite(t).all() for pair in cache.layers for t in pair):
                    raise ValueError('invalid compact cache shape/dtype/device/value')
            if offset + ids.shape[1] > c['max_position_embeddings']:
                raise ValueError('context limit exceeded')
            x, pairs = self.embed_tokens(ids), []
            for i, layer in enumerate(self.layers):
                x, pair = layer(x, None if cache is None else cache.layers[i], offset)
                pairs.append(pair)
            return self.lm_head(self.norm(x)), TinyCache(tuple(pairs), self, self.signature())
    return TinyModel()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--torch', '--tiny', dest='tiny', action='store_true', help='run random CPU probe, not pretrained inference')
    args = parser.parse_args(argv)
    for kind in PINS:
        record = load_checkpoint(kind)
        budget = parameter_budget(record['config'])
        print(f'[{kind}]')
        for component, value, source in component_table(record):
            print(f'{component}: {value}\n  source: {source}')
        print(f"unique architectural parameters: {budget['unique_total']:,}")
        print(f"non-embedding/head parameters: {budget['non_embedding']:,}")
        print(f"active-matrix convention: {budget['active_matrix_convention']:,}")
        print(f"BF16 logical KV bytes, B=1 T=4096: {cache_bytes(record['config'], 1, 4096):,}")
    print('pretrained inference: NOT RUN; no weights or BPE vocabulary loaded')
    if args.tiny:
        try:
            torch = importlib.import_module('torch')
            render_prompt(load_checkpoint('dense'), '1+1?', False)
        except ModuleNotFoundError as exc:
            if exc.name not in ('torch', 'jinja2', 'jinja2.sandbox'):
                raise
            print('P25 --torch requires PyTorch/Jinja2: python -m pip install -r requirements.txt', file=sys.stderr)
            return 2
        torch.set_num_threads(1)
        for kind in PINS:
            torch.manual_seed(250)
            model = build_tiny_model(tiny_config(kind)).double().eval()
            ids = torch.tensor([[1, 2, 3, 4, 5]])
            full, full_cache = model(ids)
            left, cache = model(ids[:, :2])
            right, cache = model(ids[:, 2:], cache)
            error = float((full-torch.cat((left, right), 1)).abs().max())
            actual = sum(p.numel() for p in model.parameters())
            payload = sum(t.numel()*t.element_size() for pair in cache.layers for t in pair)
            print(f'tiny {kind}: params={actual}; shape={tuple(full.shape)}; cache_bytes={payload}; max_cache_error={error:.3e}')
            assert error < 1e-10
            assert actual == parameter_budget(model.config)['unique_total']
            assert payload == cache_bytes(model.config, 1, 5, 8)
        print('budget example:', completion_budget([151667, 7, 8, 151668, 9], 6))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
