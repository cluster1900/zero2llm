"""P27: one CPU integration health check over the existing S0--S6 lessons.

Run: OMP_NUM_THREADS=1 python code/28_minigpt_healthcheck.py [--mtp] [--json]
Random tiny models, synthetic integer tokens, no downloads, GPU or quality claim.
This orchestrator imports the lessons; it does not implement another Transformer.
"""
import argparse
from dataclasses import asdict, dataclass
import importlib.util
import inspect
import io
import json
from pathlib import Path
import platform
import sys
import textwrap

try:
    import torch
except ModuleNotFoundError as exc:
    if __name__ == '__main__' and exc.name == 'torch':
        print('P27 requires PyTorch. Install dependencies: python -m pip install -r requirements.txt; '
              'NOT RUN, not passed.', file=sys.stderr)
        raise SystemExit(2) from None
    raise

_NAME = 'zero2llm_s6'
if _NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(_NAME, Path(__file__).with_name('24_reasoning_mtp.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[_NAME] = module
    spec.loader.exec_module(module)
s6 = sys.modules[_NAME]
s5, s4 = s6.s5, s6.s5.s4
s3, s2 = s4.s3, s4.s2
s1, s0 = s2.kv, s2.s0

ATOL, RTOL = 3e-6, 1e-5
SEED = 270
COMMON = dict(vocab_size=17, d_model=16, num_heads=4, num_layers=2, max_seq_len=16)
VOCAB = tuple(f'synthetic_id_{i}' for i in range(17))
FAMILIES = dict(S0='baby', S1='baby', S2='rope', S3='modern', S4='grouped', S5='moe', S6='mtp')
CLASSES = {'S0': (s0.GPTConfig, s0.BabyGPT), 'S1': (s0.GPTConfig, s0.BabyGPT),
           'S2': (s2.RopeConfig, s2.RopeGPT), 'S3': (s3.ModernConfig, s3.ModernGPT),
           'S4': (s4.GroupedConfig, s4.GroupedGPT), 'S5': (s5.MoEConfig, s5.MoEGPT),
           'S6': (s5.MoEConfig, s6.MTPGPT)}


@dataclass(frozen=True)
class Stage:
    name: str
    model: object
    cached_forward: object = None


def construct(stage, config, depth=0):
    """Explicit allowlist; never import arbitrary checkpoint-specified classes."""
    if stage not in CLASSES:
        raise ValueError('unknown stage')
    config_type, model_type = CLASSES[stage]
    if type(config) is not config_type:
        raise ValueError('stage/config type mismatch')
    if stage != 'S6' and depth != 0:
        raise ValueError('MTP depth requires S6')
    if stage in ('S0', 'S1'):
        return model_type.from_config(config)
    return model_type(config, depth=depth) if stage == 'S6' else model_type(config)


def make_stages(include_mtp=False, seed=SEED):
    """S1 is the SAME S0 model, not a separately initialized lookalike."""
    result = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        base = construct('S0', s0.GPTConfig(**COMMON)).eval()
        result += [Stage('S0', base), Stage('S1', base, s1.cached_forward)]
        specs = [('S2', s2.RopeConfig(**COMMON), s2.cached_forward),
                 ('S3', s3.ModernConfig(**COMMON), s3.cached_forward),
                 ('S4', s4.GroupedConfig(**COMMON, num_kv_heads=2), s4.cached_forward),
                 ('S5', s5.MoEConfig(**COMMON, num_kv_heads=2, expert_width=7,
                                    shared_width=5), s5.cached_forward)]
        for name, config, forward in specs:
            torch.manual_seed(seed)
            result.append(Stage(name, construct(name, config).eval(), forward))
        if include_mtp:
            # A distinct initialization makes the subsequent copy observable:
            # omitting it must NOT accidentally pass because RNG draws match S5.
            torch.manual_seed(seed + 1)
            model = construct('S6', specs[-1][1], depth=2).eval()
            # Hold all S5 tensors fixed: added heads alone define the S6 treatment.
            model.load_state_dict({**model.state_dict(), **result[-1].model.state_dict()}, strict=True)
            result.append(Stage('S6', model, s5.cached_forward))
    return result


def fixture():
    """Two distinct synthetic documents; labels shifted WITHOUT wraparound."""
    streams = torch.tensor([[1, 2, 3, 2, 6, 7, 4, 8, 9],
                            [4, 0, 2, 1, 4, 3, 8, 6, 5]], dtype=torch.long)
    return streams[:, :-1], streams[:, 1:]


def check_close(actual, expected, atol=ATOL, rtol=RTOL):
    if actual.shape != expected.shape or not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError('shape/nonfinite output mismatch')
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    return float((actual - expected).detach().abs().max())


@torch.no_grad()
def causal_error(model, tokens, cut=3):
    """Intervene only on FUTURE inputs, then inspect all earlier logits."""
    if not 0 < cut < tokens.shape[1]:
        raise ValueError('cut must split prefix and nonempty future')
    changed = tokens.clone()
    changed[:, cut:] = (changed[:, cut:] + 5) % model.config.vocab_size
    return check_close(model(changed)[0][:, :cut], model(tokens)[0][:, :cut])


@torch.no_grad()
def cache_equivalence(stage, tokens, sizes=(3, 1, 4)):
    if stage.cached_forward is None:
        raise ValueError('stage has no cache path')
    if any(type(size) is not int or size <= 0 for size in sizes) or sum(sizes) != tokens.shape[1]:
        raise ValueError('positive chunks must cover tokens exactly')
    expected = stage.model(tokens)[0]
    cache, pieces, start = None, [], 0
    for size in sizes:
        output, cache = stage.cached_forward(stage.model, tokens[:, start:start + size], cache)
        pieces.append(output)
        start += size
        if cache.length != start:
            raise AssertionError('cache length differs from consumed prefix')
    error = check_close(torch.cat(pieces, 1), expected)
    # Independent integer ledger, not the module's expected_cache_bytes helper.
    config = stage.model.config
    heads = getattr(config, 'num_kv_heads', config.num_heads)
    expected_bytes = 2 * config.num_layers * tokens.shape[0] * heads * tokens.shape[1] * (
        config.d_model // config.num_heads) * next(stage.model.parameters()).element_size()
    actual_bytes = sum(t.numel() * t.element_size() for pair in cache.layers for t in pair)
    if actual_bytes != expected_bytes:
        raise AssertionError('KV tensor payload differs from independent ledger')
    return error, actual_bytes


def _metadata(stage, model, vocabulary):
    if stage not in CLASSES or type(model) is not CLASSES[stage][1]:
        raise ValueError('stage/model type mismatch')
    if type(model.config) is not CLASSES[stage][0]:
        raise ValueError('stage/config type mismatch')
    if (not isinstance(vocabulary, (tuple, list)) or len(vocabulary) != model.config.vocab_size or
            any(type(token) is not str for token in vocabulary) or len(set(vocabulary)) != len(vocabulary)):
        raise ValueError('vocabulary must list each unique token in ID order')
    dtype = next(model.parameters()).dtype
    if dtype not in (torch.float32, torch.float64):
        raise ValueError('healthcheck checkpoints support float32/float64 only')
    return {'format': 'zero2llm-healthcheck-v1', 'family': FAMILIES[stage],
            'config': asdict(model.config), 'vocabulary': list(vocabulary),
            'mtp_depth': len(model.mtp_heads) if stage == 'S6' else 0,
            'dtype': str(dtype)}


def save_checkpoint(destination, stage, model, vocabulary):
    """Trusted local lesson format, not HF/GGUF or a full training-resume file."""
    metadata = _metadata(stage, model, vocabulary)
    state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    if any(t.is_floating_point() and (t.dtype != next(model.parameters()).dtype or
            not torch.isfinite(t).all()) for t in state.values()):
        raise ValueError('checkpoint tensors must be finite and have one floating dtype')
    torch.save({'metadata': metadata, 'state_dict': state}, destination)


def load_checkpoint(source, stage, expected_config, vocabulary, depth=0):
    """Require expected semantic configuration BEFORE strict tensor loading.

    Read only your own trusted files. weights_only=True reduces pickle exposure;
    it does not turn arbitrary downloaded files into trustworthy artifacts.
    S0/S1 share a family. Every other stage boundary is deliberately rejected.
    """
    with torch.random.fork_rng(devices=[]):
        model = construct(stage, expected_config, depth)
    checkpoint = torch.load(source, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or set(checkpoint) != {'metadata', 'state_dict'}:
        raise ValueError('checkpoint envelope mismatch')
    meta = checkpoint['metadata']
    if not isinstance(meta, dict) or meta.get('dtype') not in ('torch.float32', 'torch.float64'):
        raise ValueError('unsupported checkpoint dtype/metadata')
    model.to(dtype={'torch.float32': torch.float32, 'torch.float64': torch.float64}[meta['dtype']])
    if meta != _metadata(stage, model, vocabulary):
        raise ValueError('checkpoint architecture/config/vocabulary/depth mismatch')
    state = checkpoint['state_dict']
    if not isinstance(state, dict) or any(not isinstance(t, torch.Tensor) for t in state.values()):
        raise ValueError('checkpoint state must contain tensors')
    expected_state = model.state_dict()
    # load_state_dict can silently cast integer/bool/complex tensors. Validate
    # each named target's exact dtype, including any non-floating buffers, first.
    if any(key in expected_state and tensor.dtype != expected_state[key].dtype
           for key, tensor in state.items()):
        raise ValueError('checkpoint tensor dtype mismatch')
    if any(not torch.isfinite(tensor).all() for tensor in state.values()):
        raise ValueError('checkpoint tensor nonfinite mismatch')
    model.load_state_dict(state, strict=True)
    return model.eval()


def checkpoint_error(stage, tokens):
    buffer = io.BytesIO()
    save_checkpoint(buffer, stage.name, stage.model, VOCAB)
    buffer.seek(0)
    depth = len(stage.model.mtp_heads) if stage.name == 'S6' else 0
    loaded = load_checkpoint(buffer, stage.name, stage.model.config, VOCAB, depth)
    # torch serialization should preserve every state tensor exactly.
    for key, tensor in stage.model.state_dict().items():
        if not torch.equal(tensor, loaded.state_dict()[key]):
            raise AssertionError('checkpoint changed state tensor')
    with torch.no_grad():
        return check_close(loaded(tokens)[0], stage.model(tokens)[0], atol=0, rtol=0)



def compatibility_checks(stages):
    """Same architecture may be loadable; same tensor SHAPES are not sufficient."""
    by_name = {stage.name: stage for stage in stages}
    def serialized(stage):
        buffer = io.BytesIO()
        save_checkpoint(buffer, stage.name, stage.model, VOCAB)
        return buffer.getvalue()
    baseline = serialized(by_name['S0'])
    loaded = load_checkpoint(io.BytesIO(baseline), 'S1', by_name['S0'].model.config, VOCAB)
    for key, value in loaded.state_dict().items():
        if not torch.equal(value, by_name['S0'].model.state_dict()[key]):
            raise AssertionError('S0/S1 checkpoint must preserve weights')
    from dataclasses import replace
    cases = [
        ('S0_to_S2', baseline, 'S2', by_name['S2'].model.config, VOCAB, 0),
        ('vocabulary_order', baseline, 'S0', by_name['S0'].model.config, tuple(reversed(VOCAB)), 0),
        ('rope_base_same_shapes', serialized(by_name['S2']), 'S2',
         replace(by_name['S2'].model.config, rope_base=20000.), VOCAB, 0),
        ('GQA_head_count', serialized(by_name['S4']), 'S4',
         replace(by_name['S4'].model.config, num_kv_heads=1), VOCAB, 0),
        ('S4_to_S5', serialized(by_name['S4']), 'S5', by_name['S5'].model.config, VOCAB, 0),
    ]
    if 'S6' in by_name:
        cases.append(('MTP_depth', serialized(by_name['S6']), 'S6', by_name['S6'].model.config, VOCAB, 0))
    rejected = []
    for label, data, name, config, vocabulary, depth in cases:
        try:
            load_checkpoint(io.BytesIO(data), name, config, vocabulary, depth)
        except ValueError as exc:
            if 'architecture/config/vocabulary/depth mismatch' not in str(exc):
                raise
            rejected.append(label)
        else:
            raise AssertionError(f'incompatible checkpoint accepted: {label}')
    return {'S0_to_S1': 'same family and identical tensors', 'rejected': rejected}


def component_oracles():
    """Small hand-computable checks independent of model/cache self-consistency."""
    import math
    # One adjacent pair, quarter turn: the expected vector is literal.
    rotated = s2.apply_rope(torch.tensor([[[3., 4.]]], dtype=torch.float64),
                           torch.tensor([math.pi / 2], dtype=torch.float64))
    check_close(rotated, torch.tensor([[[-4., 3.]]], dtype=torch.float64), atol=1e-12, rtol=0)
    rms = s3.RMSNorm(2, eps=.5).double()
    x = torch.tensor([[1., 3.]], dtype=torch.float64)
    check_close(rms(x), torch.tensor([[1 / math.sqrt(5.5), 3 / math.sqrt(5.5)]],
                                     dtype=torch.float64), atol=1e-12, rtol=0)
    gate = s3.gate_walkthrough()['output']
    check_close(gate, torch.tensor([-3 / (1 + math.e), 8 / (1 + math.exp(-2))],
                                  dtype=torch.float64), atol=1e-12, rtol=0)
    check_close(s4.mapping_walkthrough(), torch.tensor([[10., 11.], [10., 11.],
                [20., 21.], [20., 21.]], dtype=torch.float64), atol=0, rtol=0)
    info = s5.routing_walkthrough()
    if info.indices.tolist() != [[0, 1], [3, 2], [0, 1]] or info.requested.tolist() != [2, 2, 1, 1]:
        raise AssertionError('MoE hand routing/count mismatch')
    check_close(info.weights, torch.tensor([[4 / 7, 3 / 7], [4 / 7, 3 / 7], [.5, .5]],
                                          dtype=torch.float64), atol=1e-12, rtol=0)
    return {'RoPE': [-4, 3], 'GQA_mapping': [0, 0, 1, 1], 'MoE_load': [2, 2, 1, 1],
            'RMSNorm_and_SwiGLU': 'scalar hand oracles passed'}


def finite_backward(model, tokens, targets):
    model.zero_grad(set_to_none=True)
    _, loss = model(tokens, targets)
    if not torch.isfinite(loss):
        raise AssertionError('nonfinite loss')
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads) or not any(g.abs().sum() > 0 for g in grads):
        raise AssertionError('missing/nonfinite/zero gradients')
    # Unselected experts and unused S6 heads can legitimately have grad=None.
    if model.tok_emb.weight.grad is None or model.head.weight.grad is None:
        raise AssertionError('backbone or main head detached')
    value = float(loss.detach())
    model.zero_grad(set_to_none=True)
    return value


def mutate_function(function, old, new):
    """Compile an actual in-memory source mutant; leave lesson files untouched.

    A changed upstream source anchor is a HARD failure, never a passing negative.
    Numerical negative_controls must produce finite wrong logits, never treat
    an exception as 'caught'. Guard tests separately require a specific refusal
    and inspect the bad state accepted when that actual source guard is removed.
    """
    raw = inspect.unwrap(function)
    source = textwrap.dedent(inspect.getsource(raw))
    if source.count(old) != 1:
        raise AssertionError('mutation anchor must occur exactly once')
    namespace = dict(raw.__globals__)
    exec(compile(source.replace(old, new, 1), '<p27-behavior-mutant>', 'exec'), namespace)
    return namespace[raw.__name__]


@torch.no_grad()
def negative_controls(base, tokens):
    """Three real finite wrong computations, not hand-written failure flags."""
    good = s1.cached_forward
    variants = {
        'wrong_mask': mutate_function(good, 'allowed = key_positions <= query_positions',
                                     'allowed = torch.ones(T, offset + T, dtype=torch.bool, device=x.device).tril()'),
        'wrong_position': mutate_function(good, 'pos = torch.arange(offset, offset + T, device=idx.device)',
                                         'pos = torch.arange(T, device=idx.device)'),
    }
    expected = base(tokens)[0]
    first_ok, cache_ok = good(base, tokens[:, :3], None)
    rest_ok, _ = good(base, tokens[:, 3:], cache_ok)
    before_split = float((torch.cat((first_ok, rest_ok), 1) - expected).abs().max())
    rows = []
    for name, forward in variants.items():
        first, cache = forward(base, tokens[:, :3], None)
        rest, _ = forward(base, tokens[:, 3:], cache)
        actual = torch.cat((first, rest), 1)
        if not torch.isfinite(actual).all():
            raise AssertionError('negative must return finite logits, not crash/NaN')
        delta = float((actual - expected).abs().max())
        if torch.allclose(actual, expected, atol=ATOL, rtol=RTOL):
            raise AssertionError(f'{name} survived the equivalence oracle')
        rows.append({'name': name, 'caught': True, 'max_logit_error': delta,
                     'before_error': before_split})
    # Same model/shape/weight signature, WRONG request history: owner checks cannot
    # know which prompt the caller intended. No forgery or signature bypass here.
    _, stale = good(base, tokens[:, :3], None)
    other = (tokens[:, :4] + 7) % base.config.vocab_size
    fresh, _ = good(base, other, None)
    before_fresh = check_close(fresh, base(other)[0])
    polluted, _ = good(base, other, stale)
    if not torch.isfinite(polluted).all():
        raise AssertionError('cache pollution must produce finite logits')
    if torch.allclose(polluted, fresh, atol=ATOL, rtol=RTOL):
        raise AssertionError('cache pollution survived fresh-request oracle')
    rows.append({'name': 'cache_pollution', 'caught': True,
                 'max_logit_error': float((polluted - fresh).abs().max()),
                 'before_error': before_fresh})
    return rows


def controlled_ablation(tokens, targets):
    """No ranking of independently drawn S0--S5 weights: actual paired controls."""
    models = s3.make_ablation_models(s3.ModernConfig(**COMMON), seed=SEED)
    names = list(models)
    rows = []
    for index, (name, model) in enumerate(models.items()):
        model.eval()
        checked = []
        if index:
            old, new = models[names[index - 1]].state_dict(), model.state_dict()
            excluded = s3._is_norm if index == 1 else s3._is_ffn
            checked = sorted(key for key in old if not excluded(key))
            if set(checked) != {key for key in new if not excluded(key)}:
                raise AssertionError('ablation changed unrelated state keys')
            if any(not torch.equal(old[key], new[key]) for key in checked):
                raise AssertionError('ablation changed unrelated weights')
        loss = finite_backward(model, tokens, targets)
        error, _ = cache_equivalence(Stage('S3', model, s3.cached_forward), tokens)
        rows.append({'name': name, 'treatment': ('bias-free control', 'normalization only',
                    'FFN family and matched width')[index], 'unchanged_tensors': len(checked),
                     'width': model.config.ffn_width, 'parameters': s3.parameter_count(model),
                     'ffn_parameters': s3.parameter_count(model.blocks[0].mlp),
                     'loss': loss, 'cache_error': error})
    return rows


@torch.no_grad()
def full_greedy(model, prompt, count):
    """Uncached reference loop, independent of S6's decode/cache algorithm."""
    result = prompt.clone()
    for _ in range(count):
        token = model(result)[0][:, -1].argmax(-1, keepdim=True)
        result = torch.cat((result, token), 1)
    return result


def mtp_check(model, tokens):
    result = model.forward_mtp(tokens, weight=0.2)
    expected = []
    # Scalar-indexed oracle is independent of future_labels slicing/masks.
    for horizon in range(1, len(result.labels) + 1):
        labels = [[row[t + horizon] if t + horizon < len(row) else -100
                   for t in range(len(row))] for row in tokens.tolist()]
        expected.append(labels)
        if result.labels[horizon - 1].tolist() != labels:
            raise AssertionError('MTP future labels misaligned')
    reference = []
    for logits, labels in zip(result.logits, expected):
        terms = [torch.logsumexp(logits[b, t], 0) - logits[b, t, label]
                 for b, row in enumerate(labels) for t, label in enumerate(row) if label != -100]
        reference.append(sum(terms) / len(terms))
    for actual, wanted in zip(result.losses, reference):
        check_close(actual, wanted)
    check_close(result.loss, reference[0] + .2 * torch.stack(reference[1:]).mean() + result.moe_aux)
    model.zero_grad(set_to_none=True)
    result.loss.backward()
    if any(head.weight.grad is None or not torch.isfinite(head.weight.grad).all() or
           not head.weight.grad.abs().sum() > 0 for head in model.mtp_heads):
        raise AssertionError('MTP heads must receive finite nonzero gradients')
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise AssertionError('MTP nonfinite backbone gradients')
    model.zero_grad(set_to_none=True)
    prompt = tokens[:1, :2]
    expected_tokens = full_greedy(model, prompt, 7)
    proposer = s6.MTPDraft(model)
    decoded = s6.greedy_decode(model, prompt, 7, draft=proposer, draft_tokens=3)
    if not torch.equal(decoded.tokens, expected_tokens):
        raise AssertionError('draft/verify differs from independent full-prefix greedy')
    return {'label_counts': [sum(y != -100 for row in labels for y in row) for labels in expected],
            'greedy_equal': True, 'proposed': decoded.proposed, 'accepted': decoded.accepted,
            'target_calls': decoded.target_calls, 'target_tokens': decoded.target_tokens,
            'draft_calls': proposer.calls, 'draft_tokens_processed': proposer.tokens_processed}


def run_healthcheck(include_mtp=False):
    stages = make_stages(include_mtp)
    tokens, targets = fixture()
    report = {'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
              'machine': platform.machine(), 'platform': platform.platform(),
              'device': 'cpu', 'dtype': 'float32', 'threads': torch.get_num_threads(), 'seed': SEED,
              'batch': 2, 'length': 8, 'atol': ATOL, 'rtol': RTOL,
              's6_initialization_seed': SEED + 1 if include_mtp else None}, 'stages': []}
    for stage in stages:
        model = stage.model
        loss = finite_backward(model, tokens, targets)
        causal_checks = [{'cut': cut, 'max_logit_error': causal_error(model, tokens, cut)}
                         for cut in range(1, tokens.shape[1])]
        row = {'stage': stage.name, 'parameters': s3.parameter_count(model),
               'loss': loss,
               'causal_checks': causal_checks,
               'causal_error': max(check['max_logit_error'] for check in causal_checks),
               'checkpoint_error': checkpoint_error(stage, tokens)}
        if stage.cached_forward:
            # Chunked prefill, T=1 decode, whole request; different seam positions.
            row['chunk_checks'] = []
            for sizes in ((3, 1, 4), (1,) * 8, (8,), (7, 1)):
                error, payload = cache_equivalence(stage, tokens, sizes)
                row['chunk_checks'].append({'sizes': list(sizes), 'max_logit_error': error,
                                            'kv_bytes': payload})
            row['cache_error'] = max(check['max_logit_error'] for check in row['chunk_checks'])
            row['kv_bytes'] = row['chunk_checks'][-1]['kv_bytes']
            other = tokens.flip(1)
            fresh, _ = stage.cached_forward(model, other, None)
            check_close(fresh, model(other)[0])
            # Exact capacity and overflow; never silently truncate old positions.
            long = tokens.repeat(1, 2)
            _, at_limit = stage.cached_forward(model, long, None)
            try:
                stage.cached_forward(model, tokens[:, :1], at_limit)
            except ValueError as exc:
                if 'exceeds max_seq_len' not in str(exc):
                    raise
            else:
                raise AssertionError('cache capacity overflow accepted')
            row['capacity_rejected'] = True
        if stage.name in ('S5', 'S6'):
            model(tokens)
            loads = [block.mlp.last_info.requested.tolist() for block in model.blocks]
            if any(sum(load) != tokens.numel() * model.config.top_k for load in loads):
                raise AssertionError('MoE assignment ledger mismatch')
            row['expert_loads'] = loads
        report['stages'].append(row)
    report['components'] = component_oracles()
    report['compatibility'] = compatibility_checks(stages)
    report['ablation'] = controlled_ablation(tokens, targets)
    report['negatives'] = negative_controls(stages[0].model, tokens)
    report['mtp'] = mtp_check(stages[-1].model, tokens) if include_mtp else {'status': 'not requested; use --mtp'}
    report['timing'] = {'protocol': {'prompt': 3, 'later_tokens': 5, 'warmup': 2, 'repeats': 5,
                        'clock': 'perf_counter', 'synchronization': 'CPU synchronous; no GPU'},
                        'milliseconds': s1.benchmark(stages[0].model, tokens, prompt_length=3, repeats=5, warmup=2)}
    report['scope'] = 'CPU toy correctness/timing only; random weights; synthetic IDs; no GPU, general speedup or language-quality claim'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mtp', action='store_true', help='also run S6 label/gradient/greedy verification')
    parser.add_argument('--json', action='store_true', help='emit machine-readable report to stdout')
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = run_healthcheck(args.mtp)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return
    print('P27 miniGPT healthcheck:', report['environment'])
    for row in report['stages']:
        print(f"{row['stage']}: params={row['parameters']} loss={row['loss']:.6f} "
              f"causal={row['causal_error']:.3g} checkpoint={row['checkpoint_error']:.3g} "
              f"cache={row.get('cache_error', 'n/a')} KV_bytes={row.get('kv_bytes', 0)}")
        print('  causal cuts checked:', [check['cut'] for check in row['causal_checks']])
        if 'expert_loads' in row:
            print('  expert requested assignments:', row['expert_loads'])
    for row in report['ablation']:
        print('ablation:', row)
    for row in report['negatives']:
        print(f"negative {row['name']}: caught={row['caught']} finite_error={row['max_logit_error']:.6g}")
    print('components:', report['components'])
    print('checkpoint compatibility:', report['compatibility'])
    print('CPU timing:', report['timing'])
    print('S6:', report['mtp'])
    print('PASS:', report['scope'])


if __name__ == '__main__':
    main()
