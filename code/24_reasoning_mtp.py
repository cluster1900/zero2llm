"""P23 / S6: future-label losses and GREEDY speculative decoding on a CPU.

Run: OMP_NUM_THREADS=1 python code/24_reasoning_mtp.py
Independent linear future heads are a teaching side branch, NOT V3's sequential
MTP modules or V4.1's DSpark. No R1 training or stochastic speculative sampler is
implemented. A target forward-call count is NOT a speed measurement.
"""
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ModuleNotFoundError as exc:
    if __name__ == '__main__' and exc.name == 'torch':
        print('P23 requires PyTorch. Install dependencies: python -m pip install -r requirements.txt',
              file=sys.stderr)
        raise SystemExit(2) from None
    raise

_NAME = 'zero2llm_s5'
if _NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(_NAME, Path(__file__).with_name('19_moe.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[_NAME] = module
    spec.loader.exec_module(module)
s5 = sys.modules[_NAME]


def future_labels(tokens, horizon, valid=None, segments=None, supervise=None):
    """At hidden position t, label token[t+horizon], NOT token[t+horizon-1].

    -100 means no loss. All positions from t through t+horizon must be valid and
    in one segment; supervision is required only at the destination. This guards
    LABELS, not attention: it does not implement document-isolated attention.
    Segment IDs must be integer, and padding validity must be supplied explicitly.
    """
    s5.s4.s3.positive_int('horizon', horizon)
    if (not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or
            tokens.dtype != torch.long or min(tokens.shape) <= 0 or (tokens < 0).any()):
        raise ValueError('tokens must be nonnegative long [B,T], B,T > 0')
    B, T = tokens.shape
    for name, value in (('valid', valid), ('supervise', supervise)):
        if value is not None and (not isinstance(value, torch.Tensor) or
                value.shape != tokens.shape or value.dtype != torch.bool or
                value.device != tokens.device):
            raise ValueError(f'{name} must be boolean [B,T] on token device')
    if segments is not None and (not isinstance(segments, torch.Tensor) or
            segments.shape != tokens.shape or segments.dtype != torch.long or
            segments.device != tokens.device or (segments < 0).any()):
        raise ValueError('segments must be nonnegative long [B,T] on token device')
    valid = torch.ones_like(tokens, dtype=torch.bool) if valid is None else valid
    supervise = torch.ones_like(valid) if supervise is None else supervise
    labels = torch.full_like(tokens, -100)
    if horizon >= T:
        return labels
    count = T - horizon
    keep = supervise[:, horizon:].clone()
    for offset in range(horizon + 1):
        keep &= valid[:, offset:offset + count]
        if segments is not None:
            keep &= segments[:, offset:offset + count] == segments[:, :count]
    labels[:, :count] = torch.where(keep, tokens[:, horizon:], -100)
    return labels


def masked_cross_entropy(logits, labels):
    """Mean over valid labels; an empty horizon has differentiable zero loss."""
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 3 or
            not logits.is_floating_point() or min(logits.shape) <= 0 or
            not torch.isfinite(logits).all()):
        raise ValueError('logits must be finite floating [B,T,V]')
    if (not isinstance(labels, torch.Tensor) or labels.shape != logits.shape[:2] or
            labels.dtype != torch.long or labels.device != logits.device or
            ((labels != -100) & ((labels < 0) | (labels >= logits.shape[-1]))).any()):
        raise ValueError('labels must be long [B,T], valid token IDs or -100')
    keep = labels != -100
    if not keep.any():
        # Summing finite but huge values can overflow before multiplication by 0.
        # An empty differentiable reduction is exactly zero, with zero gradients.
        return logits[..., :0].sum()
    return F.cross_entropy(logits[keep], labels[keep])


@dataclass(frozen=True)
class MTPResult:
    logits: tuple                 # main horizon 1, then horizons 2, 3, ...
    labels: tuple                 # each [B,T], -100 at invalid destinations
    losses: tuple                 # mean CE per horizon, empty horizon -> zero
    loss: torch.Tensor            # CE1 + weight * mean(extra CE) + MoE auxiliary
    moe_aux: torch.Tensor          # already scaled by S5 config.aux_weight


class MTPGPT(s5.MoEGPT):
    """S6: optional independent linear heads on S5's normalized hidden states.

    Ordinary forward and s5.cached_forward are inherited without modification.
    depth=0 adds NO parameters, draws NO extra random values, and preserves S5's
    state_dict keys. Cache payloads match; owner-specific caches are not portable
    between model instances. Updating weights still invalidates old caches.
    """
    def __init__(self, config, depth=0):
        s5.nonnegative_int('depth', depth)
        super().__init__(config)
        self.mtp_heads = nn.ModuleList([
            nn.Linear(config.d_model, config.vocab_size, bias=False) for _ in range(depth)])

    def mtp_logits(self, idx):
        self.validate_lm_routing()
        s5.s4.s2.s0.validate_tokens(idx, self.config)
        if idx.device != self.tok_emb.weight.device:
            raise ValueError('tokens must be on the model device')
        hidden = self.tok_emb(idx)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.ln_f(hidden)
        return (self.head(hidden),) + tuple(head(hidden) for head in self.mtp_heads)

    def forward_mtp(self, idx, valid=None, segments=None, supervise=None, weight=0.2):
        """Training-only explicit path; it does not consume pre-shifted targets.

        Use one document per row for isolated contexts. Label boundary masks alone
        do not stop attention to earlier documents. Auxiliary heads send gradients
        into the backbone: this is an objective ablation, NOT frozen distillation.
        """
        s5.finite_nonnegative('weight', weight)
        logits = self.mtp_logits(idx)
        labels = tuple(future_labels(idx, h, valid, segments, supervise)
                       for h in range(1, len(logits) + 1))
        losses = tuple(masked_cross_entropy(x, y) for x, y in zip(logits, labels))
        auxiliary = losses[0] * 0
        terms = [b.mlp.last_info.aux_loss for b in self.blocks if b.mlp.balance_mode == 'aux']
        if terms:
            auxiliary = self.config.aux_weight * torch.stack(terms).mean()
        total = losses[0] + auxiliary
        if len(losses) > 1:
            total = total + weight * torch.stack(losses[1:]).mean()
        return MTPResult(logits, labels, losses, total, auxiliary)


def trim_cache(cache, length):
    """Discard rejected speculative K/V positions; preserve ownership/signature.

    Returns cloned prefixes, never edits the caller's original cache payloads.
    Only the S4/S5 compact cache format is supported.
    """
    if not isinstance(cache, s5.s4.GroupedCache):
        raise ValueError('expected a compact S4/S5 cache')
    if type(length) is not int or not 0 < length <= cache.length:
        raise ValueError('trim length must be in [1, cache.length]')
    return s5.s4.GroupedCache(tuple(tuple(t[:, :, :length, :].clone() for t in pair)
                                  for pair in cache.layers), cache.owner, cache.versions)


@dataclass(frozen=True)
class DecodeResult:
    tokens: torch.Tensor           # prompt followed by the generated tokens
    cache: object                  # contains tokens[:, :-1] if any were generated
    pending: torch.Tensor | None   # final generated token, not yet put into cache
    target_calls: int              # actual invocations of forward, including prefill
    target_tokens: int             # sum of actual chunk lengths (includes rejects)
    draft_requests: int            # proposer invocations, NOT its internal forwards
    proposed: int                  # all proposals, even ones beyond an earlier reject
    accepted: int                  # retained proposal tokens only
    rejected_rounds: int
    bonus_tokens: int              # one after a completely accepted nonempty draft


def _decode_inputs(model, prompt, max_new_tokens, eos_id):
    if model.training:
        raise ValueError('decoding requires model.eval()')
    s5.s4.s2.s0.validate_tokens(prompt, model.config)
    if prompt.shape[0] != 1:
        raise ValueError('teaching decoder supports one unpadded request only')
    if prompt.device != model.tok_emb.weight.device:
        raise ValueError('prompt must be on model device')
    s5.nonnegative_int('max_new_tokens', max_new_tokens)
    if prompt.shape[1] + max_new_tokens > model.max_seq_len:
        raise ValueError('prompt plus generation budget exceeds max_seq_len')
    if eos_id is not None and (type(eos_id) is not int or not 0 <= eos_id < model.config.vocab_size):
        raise ValueError('eos_id must be in the vocabulary')


@torch.no_grad()
def greedy_decode(model, prompt, max_new_tokens, eos_id=None, draft=None,
                  draft_tokens=3, forward=None):
    """Exact GREEDY draft/verify, with smallest-ID argmax tie breaking.

    draft(prefix, limit) returns 0..limit long token IDs [1,K] on prompt device.
    It sees a clone and cannot change the committed prefix. It must not mutate
    model weights/configuration. forward defaults to the real s5.cached_forward;
    injection is useful for independent testing/instrumentation, not speed claims.

    Invariant after each emission: cache represents tokens[:, :-1], and pending
    is tokens[:, -1:]. Verify [pending, *draft] in ONE causal target call. At the
    first mismatch keep only accepted draft tokens and emit target argmax. Reject
    the whole remaining suffix; its logits were conditioned on a wrong prefix.
    All accepted -> emit one bonus token. No random sampling is implemented.
    """
    _decode_inputs(model, prompt, max_new_tokens, eos_id)
    s5.s4.s3.positive_int('draft_tokens', draft_tokens)
    if draft is not None and not callable(draft):
        raise ValueError('draft must be callable or None')
    if forward is None:
        forward = s5.cached_forward
    calls = processed = requests = proposed = accepted = rejected = bonus = 0
    tokens = prompt.clone()
    cache = pending = None

    def target(chunk, old_cache):
        nonlocal calls, processed
        logits, new_cache = forward(model, chunk, old_cache)
        calls += 1
        processed += chunk.shape[1]
        if (logits.shape != (1, chunk.shape[1], model.config.vocab_size) or
                not torch.isfinite(logits).all()):
            raise ValueError('target must return finite [1,chunk,V] logits')
        return logits, new_cache

    def result():
        return DecodeResult(tokens, cache, pending, calls, processed, requests,
                            proposed, accepted, rejected, bonus)

    if max_new_tokens == 0 or (eos_id is not None and prompt[0, -1].item() == eos_id):
        return result()  # No work is needed; no cache is built.
    logits, cache = target(prompt, None)
    pending = logits[:, -1].argmax(-1, keepdim=True)
    tokens = torch.cat((tokens, pending), dim=1)
    while tokens.shape[1] - prompt.shape[1] < max_new_tokens:
        if eos_id is not None and pending.item() == eos_id:
            break
        remaining = max_new_tokens - (tokens.shape[1] - prompt.shape[1])
        # Reserve one output slot for the correction or the all-accepted bonus.
        limit = min(draft_tokens, remaining - 1)
        proposal = prompt[:, :0].clone()
        if draft is not None and limit > 0:
            proposal = draft(tokens.clone(), limit)
            requests += 1
            if (not isinstance(proposal, torch.Tensor) or proposal.dtype != torch.long or
                    proposal.ndim != 2 or proposal.shape[0] != 1 or
                    proposal.shape[1] > limit or proposal.device != prompt.device or
                    ((proposal < 0) | (proposal >= model.config.vocab_size)).any()):
                raise ValueError('draft must return long [1,K], 0 <= K <= limit, with valid IDs')
        K = proposal.shape[1]
        proposed += K
        old_length = cache.length
        verify, candidate_cache = target(torch.cat((pending, proposal), dim=1), cache)
        emitted = []
        all_accepted = True
        for j in range(K):
            correct = int(verify[0, j].argmax())
            candidate = int(proposal[0, j])
            if candidate != correct:
                emitted.append(correct)
                rejected += 1
                all_accepted = False
                break
            emitted.append(candidate)
            accepted += 1
            if eos_id is not None and candidate == eos_id:
                all_accepted = False  # Stop here, without producing any bonus.
                break
        if all_accepted:
            emitted.append(int(verify[0, K].argmax()))
            bonus += int(K > 0)
        new_tokens = torch.tensor([emitted], dtype=torch.long, device=prompt.device)
        tokens = torch.cat((tokens, new_tokens), dim=1)
        pending = new_tokens[:, -1:]
        # +1 feeds the old pending; -1 leaves the new pending outside the cache.
        cache = trim_cache(candidate_cache, old_length + len(emitted))
    return result()


class MTPDraft:
    """Untrained side-head proposer for demonstrations, NOT a fast draft engine.

    Recomputes the full prefix once per request; counts that cost separately. The
    main head proposes token 1 and independent side heads propose tokens 2..D+1.
    These are marginal predictions from ONE prefix, not an autoregressive chain.
    Greedy verification remains correct even when the draft is inaccurate.
    """
    def __init__(self, model):
        if not isinstance(model, MTPGPT) or model.training:
            raise ValueError('MTPDraft requires MTPGPT in eval mode')
        self.model = model
        self.calls = 0
        self.tokens_processed = 0

    @torch.no_grad()
    def __call__(self, prefix, limit):
        s5.s4.s3.positive_int('limit', limit)
        logits = self.model.mtp_logits(prefix)
        self.calls += 1
        self.tokens_processed += prefix.shape[1]
        return torch.stack([x[:, -1].argmax(-1) for x in logits[:limit]], dim=1)


@dataclass(frozen=True)
class AdditionResult:
    answer: int
    steps: int
    complete: bool
    trace: tuple


def column_add(a, b, budget):
    """Deterministic anytime addition. Budget = digit-column operations, NOT words.

    Unknown high columns stay zero; append final carry only after all input
    columns are processed. No access to the answer/checker inside this solver.
    """
    for name, value in (('a', a), ('b', b), ('budget', budget)):
        s5.nonnegative_int(name, value)
    left, right = str(a)[::-1], str(b)[::-1]
    width = max(len(left), len(right))
    carry = answer = 0
    trace = []
    for i in range(min(budget, width)):
        x = int(left[i]) if i < len(left) else 0
        y = int(right[i]) if i < len(right) else 0
        subtotal = x + y + carry
        trace.append((x, y, carry, subtotal % 10, subtotal // 10))
        answer += (subtotal % 10) * 10 ** i
        carry = subtotal // 10
    complete = len(trace) == width
    if complete:
        answer += carry * 10 ** width
    return AdditionResult(answer, len(trace), complete, tuple(trace))


ARITHMETIC_TASKS = ((2, 3), (12, 23), (48, 37), (105, 207), (999, 1), (1234, 8765))


def arithmetic_curve(budgets=(0, 1, 2, 3, 4), tasks=ARITHMETIC_TASKS):
    """Exact same six tasks at each budget; holdout-style checker uses Python +.

    This is a controlled ALGORITHM experiment, not an LLM/R1 quality benchmark.
    Counts actual used columns as well as the allowed maximum. No training data,
    learned verifier, repeated samples, or selection by a hidden answer is used.
    """
    if not tasks:
        raise ValueError('at least one task is required')
    rows = []
    for budget in budgets:
        answers = [column_add(a, b, budget) for a, b in tasks]
        correct = sum(out.answer == a + b for out, (a, b) in zip(answers, tasks))
        rows.append({'budget': budget, 'correct': correct, 'total': len(tasks),
                     'accuracy': correct / len(tasks),
                     'mean_steps': sum(out.steps for out in answers) / len(tasks)})
    return rows


def main():
    torch.set_num_threads(1)
    torch.manual_seed(230)
    print(f'P23 / S6 CPU teaching branch; torch={torch.__version__}; seed=230')
    for row in arithmetic_curve():
        print(f"column budget={row['budget']}: correct={row['correct']}/{row['total']}; "
              f"actual mean columns={row['mean_steps']:.3f}")
    config = s5.MoEConfig(vocab_size=11, d_model=16, num_heads=4, num_kv_heads=2,
        num_layers=1, max_seq_len=24, expert_width=7, shared_width=5)
    model = MTPGPT(config, depth=2)
    data = torch.tensor([[1, 2, 3, 4, 5]])
    training = model.forward_mtp(data)
    training.loss.backward()
    print('future labels:', [label.tolist() for label in training.labels])
    print('CE per horizon:', [round(float(loss.detach()), 6) for loss in training.losses])
    print('finite loss/gradients:', bool(torch.isfinite(training.loss)) and
          all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()))
    model.eval()
    prompt = data[:, :2]
    ordinary = greedy_decode(model, prompt, 10)
    drafter = MTPDraft(model)
    speculative = greedy_decode(model, prompt, 10, draft=drafter, draft_tokens=3)
    print('greedy tokens equal:', torch.equal(ordinary.tokens, speculative.tokens))
    print('generated IDs:', ordinary.tokens[:, prompt.shape[1]:].tolist())
    print(f'target calls: ordinary={ordinary.target_calls}, draft/verify={speculative.target_calls}')
    print(f'target input tokens: ordinary={ordinary.target_tokens}, draft/verify={speculative.target_tokens}')
    print(f'drafter full-prefix calls={drafter.calls}; input tokens={drafter.tokens_processed}')
    print(f'accepted/proposed={speculative.accepted}/{speculative.proposed}; '
          f'rejected rounds={speculative.rejected_rounds}; bonuses={speculative.bonus_tokens}')
    print('No speedup claimed. Random speculative sampling is NOT implemented.')


if __name__ == '__main__':
    main()
