"""P23 extra: block drafts, confidence cutoff, ECHO-style recursive draft head (CPU).

Run: OMP_NUM_THREADS=1 python code/24b_spec_drafts.py [--svg PATH]
Teaching reconstructions, NOT the papers' code or models:
  DFlash  arXiv 2602.06036 (2026-02-05): block of drafts proposed in one call.
  DSpark  arXiv 2607.05147 (2026-07-06): confidence target c* = 1 - 0.5*||p_d - p_t||_1, cutoff.
  ECHO    arXiv 2610.24766 (2026-10-17): h_t = T^L(F(h_{t-1}, e(x_t))) along the HORIZON,
          T^L = the backbone's own final decoder layer; dynamic depth sampling (DDS).
          Book simplification: the target is frozen and only the fusion layer F is
          trained; the paper trains F with the shared backbone during pretraining.
All drafts are verified by the unchanged greedy_decode from 24_reasoning_mtp.py,
so output must equal ordinary greedy decoding. No speedup is claimed.
"""
import importlib.util
from pathlib import Path
import sys

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:
    if __name__ == "__main__" and exc.name == "torch":
        print("P23 requires PyTorch. Install dependencies: python -m pip install -r requirements.txt", file=sys.stderr)
        raise SystemExit(2) from None
    raise

_NAME = "zero2llm_s6"
if _NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(_NAME, Path(__file__).with_name("24_reasoning_mtp.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[_NAME] = module
    spec.loader.exec_module(module)
s6 = sys.modules[_NAME]
s5 = s6.s5

DDS_PI = (0.6, 0.3, 0.1)     # ECHO: P(D = 1, 2, 3), E[D] = 1.5
DDS_F = (1.0, 0.4, 0.1)      # ECHO eq. 4: f_k = P(D >= k), how often horizon k is trained
DDS_W = tuple(sum(p / d for d, p in enumerate(DDS_PI, 1) if d >= k) for k in (1, 2, 3))  # E[1{D>=k}/D]: actual loss weight under the mean over 1..D


def pattern_batch(n, length, vocab=11, gen=None):
    """Deterministic rule x_{t+1} = (3 x_t + 1) mod vocab from random starts."""
    start = torch.randint(0, vocab, (n, 1), generator=gen)
    rows = [start]
    for _ in range(length - 1):
        rows.append((3 * rows[-1] + 1) % vocab)
    return torch.cat(rows, 1)


def trunk(model, idx):
    """Final-layer hidden h^L (before ln_f); logits are model.head(model.ln_f(h))."""
    h = model.tok_emb(idx)
    for block in model.blocks:
        h = block(h)
    return h


def hidden(model, idx):
    return model.ln_f(trunk(model, idx))


def train_target(steps=150, seed=230, depth=3):
    torch.manual_seed(seed)
    config = s5.MoEConfig(vocab_size=11, d_model=16, num_heads=4, num_kv_heads=2,
                          num_layers=1, max_seq_len=32, expert_width=7, shared_width=5)
    model = s6.MTPGPT(config, depth=depth)
    opt = torch.optim.Adam(model.parameters(), lr=3e-2)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        loss = model.forward_mtp(pattern_batch(16, 12, gen=gen), weight=1.0).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


class BlockDraft:
    """DFlash-like interface: ONE drafter call returns a whole block (here: MTP heads)."""
    def __init__(self, model, tau=None):
        self.model, self.tau, self.calls, self.cut = model, tau, 0, 0

    @torch.no_grad()
    def __call__(self, prefix, limit):
        logits = self.model.mtp_logits(prefix)[:limit]
        self.calls += 1
        probs = [F.softmax(x[:, -1], -1) for x in logits]
        tokens = [p.argmax(-1) for p in probs]
        if self.tau is not None:  # DSpark-style cutoff: stop at first low-confidence slot
            keep = 0
            for p in probs:
                if p.max().item() < self.tau:
                    break
                keep += 1
            self.cut += len(tokens) - keep
            tokens = tokens[:keep]
        if not tokens:
            return prefix[:, :0]
        return torch.stack(tokens, 1)


def acceptance_target(p_draft, p_target):
    """DSpark confidence label c* = 1 - 0.5 * ||p_d - p_t||_1 = sum_v min(p_d, p_t)."""
    return 1 - 0.5 * (p_draft - p_target).abs().sum(-1)


def speculative_accept_rate(p_draft, p_target, n, gen):
    """Monte Carlo of the standard sampling rule: accept x~p_d w.p. min(1, p_t/p_d)."""
    x = torch.multinomial(p_draft, n, replacement=True, generator=gen)
    ratio = (p_target[x] / p_draft[x]).clamp(max=1)
    return (torch.rand(n, generator=gen, dtype=ratio.dtype) < ratio).float().mean().item()


def expected_tau(a):
    """E[tau] = 1 + sum_d prod_{i<=d} a_i (bonus/correction token included)."""
    total, run = 1.0, 1.0
    for x in a:
        run *= x
        total += run
    return total


class ECHOHead(nn.Module):
    """Fusion F = Linear([RMSNorm(h); RMSNorm(e(x))]); T^L = target.blocks[-1] (shared, frozen here).

    Horizon k at position t:  h^(k)_t = T^L(F(h^(k-1)_t, e(x_{t+k})))  predicts x_{t+k+1}.
    h^(0) is the target's own final-layer state (it predicts x_{t+1}). Each horizon
    consumes ONE more teacher-forced token; it is NOT a loop refining the same position.
    """
    def __init__(self, target):
        super().__init__()
        d = target.config.d_model
        self.target = target
        self.norm_h = nn.RMSNorm(d)
        self.norm_e = nn.RMSNorm(d)
        self.fuse = nn.Linear(2 * d, d, bias=False)

    def horizon(self, h_prev, tokens):
        """h_prev [B,M,d] (horizon k-1 at t=0..M-1), tokens [B,M] = x_{t+k}  ->  h^(k) [B,M,d]."""
        z = self.fuse(torch.cat((self.norm_h(h_prev), self.norm_e(self.target.tok_emb(tokens))), -1))
        return self.target.blocks[-1](z)

    def horizons(self, idx, depth):
        """Teacher-forced states h^(1..depth); h^(k) has length N-k (positions t=0..N-k-1)."""
        h = trunk(self.target, idx)
        out = []
        for k in range(1, depth + 1):
            h = self.horizon(h[:, :idx.shape[1] - k], idx[:, k:])
            out.append(h)
        return out

    def logits(self, h):
        return self.target.head(self.target.ln_f(h))


def horizon_labels(idx, k):
    """h^(k)_t predicts x_{t+k+1}: labels idx[:, k+1:] for t = 0..N-k-2."""
    return idx[:, k + 1:]


def echo_loss(head, idx, depth):
    """Mean CE over horizons 1..depth; horizon k is supervised k positions further ahead."""
    losses = []
    for k, h in enumerate(head.horizons(idx, depth), start=1):
        logits = head.logits(h[:, :-1])
        losses.append(F.cross_entropy(logits.flatten(0, 1), horizon_labels(idx, k).flatten()))
    return torch.stack(losses).mean()


def train_echo(target, steps=200, seed=231, lr=3e-2):
    """DDS: draw D ~ pi once per optimizer step and train horizons 1..D (so f_k = P(D >= k))."""
    torch.manual_seed(seed)
    head = ECHOHead(target)
    for p in target.parameters():
        p.requires_grad_(False)   # book simplification: frozen target, only F trains
    opt = torch.optim.Adam([p for n, p in head.named_parameters() if not n.startswith('target.')], lr=lr)
    gen = torch.Generator().manual_seed(seed)
    pi = torch.tensor(DDS_PI)
    curve = []
    for _ in range(steps):
        depth = int(torch.multinomial(pi, 1, generator=gen)) + 1
        loss = echo_loss(head, pattern_batch(16, 12, gen=gen), depth)
        opt.zero_grad()
        loss.backward()
        opt.step()
        curve.append(float(loss.detach()))
    return head.eval(), curve


@torch.no_grad()
def horizon_accuracy(head, idx, k):
    h = head.horizons(idx, k)[-1][:, :-1]
    return (head.logits(h).argmax(-1) == horizon_labels(idx, k)).float().mean().item()


class ECHODraft:
    """Draft j uses h^(j)_{n-2}: fuse with the latest token (pending, then each new draft)."""
    def __init__(self, head):
        self.head, self.calls = head, 0

    @torch.no_grad()
    def __call__(self, prefix, limit):
        self.calls += 1
        n = prefix.shape[1]
        if n < 2:
            return prefix[:, :0]
        seq = prefix
        h = trunk(self.head.target, prefix)[:, :n - 1]
        out = []
        for j in range(1, limit + 1):
            h = self.head.horizon(h, seq[:, j:j + n - 1])
            x = self.head.logits(h[:, -1]).argmax(-1, keepdim=True)
            out.append(x)
            seq = torch.cat((seq, x), 1)
        return torch.cat(out, 1)


def loss_svg(curve, path):
    w, hgt, pad = 480, 240, 36
    hi = max(curve)
    pts = " ".join(f"{pad + i * (w - 2 * pad) / (len(curve) - 1):.1f},{hgt - pad - v / hi * (hgt - 2 * pad):.1f}"
                   for i, v in enumerate(curve))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{hgt}" viewBox="0 0 {w} {hgt}">'
        f'<rect width="{w}" height="{hgt}" fill="white"/>'
        f'<line x1="{pad}" y1="{hgt - pad}" x2="{w - pad}" y2="{hgt - pad}" stroke="black"/>'
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{hgt - pad}" stroke="black"/>'
        f'<polyline fill="none" stroke="#1f77b4" stroke-width="2" points="{pts}"/>'
        f'<text x="{w / 2}" y="{hgt - 8}" font-size="12" text-anchor="middle" font-family="sans-serif">training step (seed 231, CPU; mean CE over horizons 1..D)</text>'
        f'<text x="{pad}" y="{pad - 10}" font-size="12" font-family="sans-serif">ECHO fusion-layer loss, max {hi:.2f}</text>'
        '</svg>\n')


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    torch.set_num_threads(1)
    target = train_target()
    prompt = torch.tensor([[2, 7]])
    base = s6.greedy_decode(target, prompt, 12)
    print('target greedy IDs:', base.tokens[:, 2:].tolist())
    for name, draft in (('block', BlockDraft(target)), ('block+cutoff 0.9', BlockDraft(target, 0.9))):
        r = s6.greedy_decode(target, prompt, 12, draft=draft, draft_tokens=3)
        assert torch.equal(r.tokens, base.tokens)
        print(f'{name}: identical={torch.equal(r.tokens, base.tokens)}; target calls {r.target_calls} vs {base.target_calls}; '
              f'accepted/proposed {r.accepted}/{r.proposed}; cut={draft.cut}')
    head, curve = train_echo(target)
    r = s6.greedy_decode(target, prompt, 12, draft=ECHODraft(head), draft_tokens=3)
    assert torch.equal(r.tokens, base.tokens)
    held = pattern_batch(64, 12, gen=torch.Generator().manual_seed(99))
    acc = [horizon_accuracy(head, held, k) for k in (1, 2, 3)]
    print(f'ECHO-style draft: identical=True; accepted/proposed {r.accepted}/{r.proposed}; '
          f'loss {curve[0]:.3f} -> {curve[-1]:.3f}; held-out accuracy at horizons 1/2/3 = '
          + '/'.join(f'{a:.3f}' for a in acc))
    gen = torch.Generator().manual_seed(7)
    p_d = torch.softmax(torch.randn(11, generator=gen, dtype=torch.float64), 0)
    p_t = torch.softmax(torch.randn(11, generator=gen, dtype=torch.float64), 0)
    c = acceptance_target(p_d, p_t).item()
    mc = speculative_accept_rate(p_d, p_t, 200000, gen)
    assert abs(c - mc) < 0.01
    print(f'c* = sum min(p_d,p_t) = {c:.4f}; Monte Carlo acceptance = {mc:.4f}')
    print(f'E[tau] for per-position acceptance (0.8,0.6,0.4): {expected_tau((0.8, 0.6, 0.4)):.3f}')
    if '--svg' in argv:
        loss_svg(curve, argv[argv.index('--svg') + 1])
    print('Greedy only; no speedup claimed; teaching reconstructions, not paper models.')


if __name__ == '__main__':
    main()
