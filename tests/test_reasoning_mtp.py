"""P23 independent label/gradient/cache/greedy contracts plus executable mutants.

No test claims a speedup or a stochastic decoding distribution. Greedy correctness
uses an independent finite-state target AND unbatched full-context S5 inference.
"""
from fractions import Fraction
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import types
import unittest

from torch_test_support import torch_status, torch_reason

READY, VERSION, PRESENT = torch_status()
SCRIPT = Path(__file__).resolve().parents[1] / 'code/24_reasoning_mtp.py'
if READY:
    import torch
    spec = importlib.util.spec_from_file_location('mtp_test', SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)


def scalar_labels(tokens, horizon, valid=None, segments=None, supervise=None):
    """Python-indexed oracle, without production slicing/masking helpers."""
    rows = []
    for b, row in enumerate(tokens.tolist()):
        out = []
        for t in range(len(row)):
            dest = t + horizon
            keep = dest < len(row)
            if keep and valid is not None:
                keep = all(bool(valid[b, j]) for j in range(t, dest + 1))
            if keep and segments is not None:
                keep = len({int(segments[b, j]) for j in range(t, dest + 1)}) == 1
            if keep and supervise is not None:
                keep = bool(supervise[b, dest])
            out.append(row[dest] if keep else -100)
        rows.append(out)
    return torch.tensor(rows, dtype=torch.long)


def scalar_ce(logits, labels):
    """Explicit log-sum-exp and Python selection, retains autograd."""
    terms = []
    for b in range(labels.shape[0]):
        for t in range(labels.shape[1]):
            y = int(labels[b, t])
            if y != -100:
                row = logits[b, t]
                peak = row.max().detach()
                terms.append(peak + torch.log(sum(torch.exp(v - peak) for v in row)) - row[y])
    return sum(terms) / len(terms) if terms else logits.sum() * 0


if READY:
    class OracleModel(torch.nn.Module):
        """Token-history-in-cache target, deliberately not a Transformer."""
        def __init__(self, vocab=7, length=48, next_fn=None):
            super().__init__()
            self.config = types.SimpleNamespace(vocab_size=vocab, max_seq_len=length)
            self.max_seq_len = length
            self.tok_emb = torch.nn.Embedding(vocab, 1)
            self.next_fn = next_fn or (lambda ids: (sum((i+1)*(v+1) for i, v in enumerate(ids))
                                                   + len(ids)) % vocab)
            self.calls = []
            self.eval()

        def scores(self, ids):
            row = torch.arange(self.config.vocab_size, dtype=torch.float64) / 100
            row[self.next_fn(ids)] = 3.0
            return row


def oracle_forward(model, chunk, cache=None):
    """Each chunk row is checked against a fresh full-prefix scalar oracle."""
    old = [] if cache is None else [int(v) for v in cache.layers[0][0].flatten()]
    assert cache is None or cache.owner is model
    model.calls.append((tuple(old), tuple(chunk[0].tolist())))
    history = old[:]
    rows = []
    for token in chunk[0].tolist():
        history.append(token)
        rows.append(model.scores(history))
    key = torch.tensor(history, dtype=torch.float64).reshape(1, 1, -1, 1)
    cache = m.s5.s4.GroupedCache(((key, key.clone()),), model, ('independent-oracle',))
    return torch.stack(rows).unsqueeze(0), cache


def oracle_sequence(model, prefix, count, eos=None):
    history = list(prefix)
    if eos is not None and history[-1] == eos:
        return history
    for _ in range(count):
        history.append(int(model.scores(history).argmax()))
        if eos is not None and history[-1] == eos:
            break
    return history


def oracle_draft(model, corrupt_at=None):
    def draft(prefix, limit):
        history = prefix[0].tolist()
        proposed = []
        for i in range(limit):
            token = int(model.scores(history).argmax())
            if i == corrupt_at:
                token = (token + 1) % model.config.vocab_size
            history.append(token)
            proposed.append(token)
        return torch.tensor([proposed], dtype=torch.long)
    return draft


class MTPCLITests(unittest.TestCase):
    def test_missing_torch_cli_and_import(self):
        blocker = """
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'torch':
        raise ModuleNotFoundError("No module named 'torch'", name='torch')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
"""
        result = subprocess.run([sys.executable, '-c', blocker +
            "runpy.run_path(sys.argv[1], run_name='__main__')", str(SCRIPT)],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('python -m pip install -r requirements.txt', result.stderr)
        self.assertNotIn('Traceback', result.stderr)
        result = subprocess.run([sys.executable, '-c', blocker + """
try:
    runpy.run_path(sys.argv[1], run_name='without_torch')
except ModuleNotFoundError as exc:
    assert exc.name == 'torch'
else:
    raise AssertionError('missing dependency must not silently pass')
""", str(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(READY, torch_reason('P23 CLI', VERSION, PRESENT))
    def test_cli_pins_documented_outputs_and_actual_cost_counters(self):
        result = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True,
                                text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        expected = {
            'greedy tokens equal': 'True',
            'generated IDs': '[[5, 6, 3, 1, 9, 2, 5, 6, 3, 1]]',
            'target calls': 'ordinary=10, draft/verify=6',
            'target input tokens': 'ordinary=11, draft/verify=18',
        }
        output = result.stdout.splitlines()
        for name, value in expected.items():
            matches = [line for line in output if line.startswith(name + ':')]
            self.assertEqual(matches, [f'{name}: {value}'])
        self.assertEqual(output.count('drafter full-prefix calls=4; input tokens=24'), 1)
        self.assertEqual(output.count('accepted/proposed=4/11; rejected rounds=4; bonuses=0'), 1)


@unittest.skipUnless(READY, torch_reason('P23', VERSION, PRESENT))
class MTPTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(230)
        self.config = m.s5.MoEConfig(vocab_size=11, d_model=8, num_heads=2,
            num_kv_heads=1, num_layers=1, max_seq_len=24, expert_width=4, shared_width=3)
        self.tokens = torch.tensor([[1, 2, 3, 4, 5], [3, 2, 4, 1, 0]])

    def test_hand_labels_and_no_circular_roll(self):
        x = torch.tensor([[1, 2, 3, 4, 5]])
        expected = [[2, 3, 4, 5, -100], [3, 4, 5, -100, -100],
                    [4, 5, -100, -100, -100], [5, -100, -100, -100, -100]]
        for h, row in enumerate(expected, 1):
            self.assertEqual(m.future_labels(x, h).tolist(), [row])
        self.assertEqual(m.future_labels(x, 5).tolist(), [[-100] * 5])
        self.assertEqual(m.future_labels(x, 19).tolist(), [[-100] * 5])

    def test_label_masks_match_scalar_oracle_including_intermediate_boundaries(self):
        valid = torch.tensor([[1, 1, 1, 1, 0], [1, 0, 1, 1, 1]], dtype=torch.bool)
        # Reused ID 0 must not allow a horizon to jump across segment 1.
        segments = torch.tensor([[0, 0, 1, 0, 0], [4, 4, 4, 5, 5]])
        supervise = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 0, 1]], dtype=torch.bool)
        for h in range(1, 8):
            for v, s, target in ((valid, segments, supervise), (valid, None, None),
                                  (None, segments, None), (None, None, supervise)):
                torch.testing.assert_close(m.future_labels(self.tokens, h, v, s, target),
                                           scalar_labels(self.tokens, h, v, s, target))
        self.assertEqual(m.future_labels(self.tokens[:1], 2, supervise=supervise[:1]).tolist(),
                         [[3, 4, 5, -100, -100]])  # source prompt can be unsupervised.

    def test_masked_ce_scalar_and_analytic_gradients(self):
        logits = torch.tensor([[[0., math.log(2), math.log(3)], [0., 0., 0.],
                                [math.log(3), math.log(2), 0.]]], requires_grad=True,
                              dtype=torch.float64)
        labels = torch.tensor([[2, -100, 1]])
        loss = m.masked_cross_entropy(logits, labels)
        self.assertAlmostEqual(float(loss.detach()), (math.log(2) + math.log(3))/2, places=13)
        loss.backward()
        # (softmax - onehot) / 2; invalid middle position receives zero.
        expected = torch.tensor([[[1/12, 1/6, -1/4], [0, 0, 0],
                                  [1/4, -1/3, 1/12]]], dtype=torch.float64)
        torch.testing.assert_close(logits.grad, expected, rtol=1e-13, atol=1e-13)
        reference = scalar_ce(logits.detach().clone().requires_grad_(), labels)
        torch.testing.assert_close(loss, reference, rtol=1e-13, atol=1e-13)
        empty = logits.detach().clone().requires_grad_()
        zero = m.masked_cross_entropy(empty, torch.full_like(labels, -100))
        zero.backward()
        self.assertEqual(float(zero.detach()), 0)
        self.assertTrue(torch.equal(empty.grad, torch.zeros_like(empty)))

    def test_disabled_is_exact_s5_parameters_rng_logits_losses_and_cache(self):
        torch.manual_seed(88)
        base = m.s5.MoEGPT(self.config).double().eval()
        after_base = torch.rand(3)
        torch.manual_seed(88)
        off = m.MTPGPT(self.config, depth=0).double().eval()
        after_off = torch.rand(3)
        self.assertTrue(torch.equal(after_base, after_off))
        self.assertEqual(list(base.state_dict()), list(off.state_dict()))
        self.assertEqual(sum(p.numel() for p in base.parameters()), sum(p.numel() for p in off.parameters()))
        for name in base.state_dict():
            self.assertTrue(torch.equal(base.state_dict()[name], off.state_dict()[name]), name)
        a = base(self.tokens, self.tokens)
        b = off(self.tokens, self.tokens)
        self.assertTrue(torch.equal(a[0], b[0]))
        self.assertTrue(torch.equal(a[1], b[1]))
        ca = cb = None
        for lo, hi in ((0, 2), (2, 3), (3, 5)):
            la, ca = m.s5.cached_forward(base, self.tokens[:, lo:hi], ca)
            lb, cb = m.s5.cached_forward(off, self.tokens[:, lo:hi], cb)
            self.assertTrue(torch.equal(la, lb))
            for pa, pb in zip(ca.layers, cb.layers):
                for va, vb in zip(pa, pb):
                    self.assertTrue(torch.equal(va, vb))
        with self.assertRaisesRegex(ValueError, 'different model'):
            m.s5.cached_forward(off, self.tokens[:, :1], ca)

    def test_all_ignored_extreme_finite_logits_have_exact_zero_gradients(self):
        for dtype, value in ((torch.float32, 3e38), (torch.float16, 40000)):
            logits = torch.full((1, 2, 3), value, dtype=dtype, requires_grad=True)
            loss = m.masked_cross_entropy(logits, torch.full((1, 2), -100, dtype=torch.long))
            self.assertEqual(float(loss.detach()), 0)
            loss.backward()
            self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_enabled_sideheads_do_not_change_ordinary_s5_path(self):
        base = m.s5.MoEGPT(self.config).double().eval()
        model = m.MTPGPT(self.config, depth=2).double().eval()
        missing = model.load_state_dict(base.state_dict(), strict=False)
        self.assertEqual(missing.missing_keys, ['mtp_heads.0.weight', 'mtp_heads.1.weight'])
        self.assertEqual(missing.unexpected_keys, [])
        calls = []
        handles = [head.register_forward_hook(lambda *args: calls.append(1)) for head in model.mtp_heads]
        torch.testing.assert_close(model(self.tokens)[0], base(self.tokens)[0], rtol=0, atol=0)
        self.assertEqual(calls, [])
        extra = sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in base.parameters())
        self.assertEqual(extra, 2 * self.config.d_model * self.config.vocab_size)
        outputs = model.mtp_logits(self.tokens)
        torch.testing.assert_close(outputs[0], base(self.tokens)[0], rtol=0, atol=0)
        self.assertEqual(len(calls), 2)
        params_before = sum(p.numel() for p in model.parameters())
        disabled_loss = model.forward_mtp(self.tokens, weight=0)
        self.assertEqual(len(calls), 4)  # Zero weight is not zero head computation.
        self.assertEqual(sum(p.numel() for p in model.parameters()), params_before)
        torch.testing.assert_close(disabled_loss.loss,
                                   disabled_loss.losses[0] + disabled_loss.moe_aux, rtol=0, atol=0)
        for handle in handles:
            handle.remove()

    def test_future_suffix_intervention_cannot_change_any_earlier_head(self):
        model = m.MTPGPT(self.config, depth=3).double().eval()
        original = model.mtp_logits(self.tokens)
        for cut in range(1, self.tokens.shape[1]):
            changed = self.tokens.clone()
            changed[:, cut:] = (changed[:, cut:] + 3) % self.config.vocab_size
            modified = model.mtp_logits(changed)
            for old, new in zip(original, modified):
                torch.testing.assert_close(old[:, :cut], new[:, :cut], rtol=1e-12, atol=1e-12)

    def check_draft_head_positions(self, module):
        """Independent scalar projections, not the drafter's mtp_logits tuple.

        This reproduces the CLI's fixed model seed. All three expected IDs differ,
        so skipping, swapping, or reusing the main/side heads is observable even
        though a correct target verifier can repair those proposals downstream.
        """
        torch.manual_seed(230)
        config = module.s5.MoEConfig(vocab_size=11, d_model=16, num_heads=4,
            num_kv_heads=2, num_layers=1, max_seq_len=24, expert_width=7, shared_width=5)
        model = module.MTPGPT(config, depth=2).eval()
        prefix = torch.tensor([[1, 2, 5]])
        with torch.no_grad():
            hidden = model.tok_emb(prefix)
            for block in model.blocks:
                hidden = block(hidden)
            last = model.ln_f(hidden)[0, -1].tolist()
        expected = []
        for head in (model.head, model.mtp_heads[0], model.mtp_heads[1]):
            weights = head.weight.detach().tolist()
            scores = [sum(value * weight for value, weight in zip(last, row)) for row in weights]
            # Python max visits vocabulary IDs in ascending order: ties use lowest.
            expected.append(max(range(config.vocab_size), key=lambda token: scores[token]))
        self.assertEqual(expected, [6, 0, 9])
        drafter = module.MTPDraft(model)
        for limit in (1, 2, 3):
            self.assertEqual(drafter(prefix, limit).tolist(), [expected[:limit]])
        self.assertEqual((drafter.calls, drafter.tokens_processed), (3, 9))

    def test_fixed_seed_draft_head_to_position_scalar_oracle(self):
        self.check_draft_head_positions(m)

    def test_objective_scalar_oracle_gradients_and_short_sequences(self):
        model = m.MTPGPT(self.config, depth=3).double()
        supervision = torch.tensor([[0, 1, 0, 1, 1], [0, 0, 1, 1, 1]], dtype=torch.bool)
        result = model.forward_mtp(self.tokens, supervise=supervision, weight=.6)
        reference_losses = [scalar_ce(x, scalar_labels(self.tokens, h, supervise=supervision))
                            for h, x in enumerate(result.logits, 1)]
        reference = reference_losses[0] + .6 * sum(reference_losses[1:])/3 + result.moe_aux
        parameters = tuple(model.parameters())
        actual_grads = torch.autograd.grad(result.loss, parameters, retain_graph=True, allow_unused=True)
        expected_grads = torch.autograd.grad(reference, parameters, allow_unused=True)
        torch.testing.assert_close(result.loss, reference, rtol=1e-12, atol=1e-12)
        for actual, expected in zip(actual_grads, expected_grads):
            if actual is None or expected is None:
                self.assertIs(actual, expected)
            else:
                torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)
        # Side loss must reach the backbone; it is not detached frozen distillation.
        auxiliary = model.forward_mtp(self.tokens, weight=1)
        gradient = torch.autograd.grad(sum(auxiliary.losses[1:]), model.tok_emb.weight)[0]
        self.assertGreater(float(gradient.abs().sum()), 0)
        short = model.forward_mtp(self.tokens[:, :1], weight=1)
        self.assertTrue(all(float(v.detach()) == 0 for v in short.losses))
        self.assertTrue(torch.isfinite(short.loss))  # MoE auxiliary may remain nonzero.

    def test_sidehead_weight_finite_difference(self):
        model = m.MTPGPT(self.config, depth=1).double()
        output = model.forward_mtp(self.tokens, weight=.4)
        output.loss.backward()
        parameter = model.mtp_heads[0].weight
        analytical = float(parameter.grad[3, 2])
        old = float(parameter[3, 2].detach())
        epsilon = 1e-5
        values = []
        with torch.no_grad():
            for delta in (epsilon, -epsilon):
                parameter[3, 2] = old + delta
                values.append(float(model.forward_mtp(self.tokens, weight=.4).loss))
            parameter[3, 2] = old
        self.assertAlmostEqual(analytical, (values[0] - values[1])/(2*epsilon), places=9)

    def test_validation_rejects_invalid_masks_labels_logits_and_options(self):
        for horizon in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                m.future_labels(self.tokens, horizon)
        for kwargs in ({'valid': torch.ones_like(self.tokens)},
                       {'segments': torch.full_like(self.tokens, -1)},
                       {'supervise': torch.ones(1, 2, dtype=torch.bool)}):
            with self.assertRaises(ValueError):
                m.future_labels(self.tokens, 1, **kwargs)
        for labels in (torch.tensor([[-1]]), torch.tensor([[3]]), torch.tensor([[0.]])):
            with self.assertRaises(ValueError):
                m.masked_cross_entropy(torch.zeros(1, 1, 3), labels)
        with self.assertRaises(ValueError):
            m.masked_cross_entropy(torch.full((1, 1, 3), float('nan')), torch.tensor([[0]]))
        for depth in (-1, True, .5):
            with self.assertRaises(ValueError):
                m.MTPGPT(self.config, depth)
        for weight in (-1, float('nan'), True):
            with self.assertRaises(ValueError):
                m.MTPGPT(self.config).forward_mtp(self.tokens, weight=weight)


@unittest.skipUnless(READY, torch_reason('P23', VERSION, PRESENT))
class GreedyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(230)

    def check_oracle(self, implementation=m if READY else None, corrupt_at=None,
                     count=12, eos=None, model=None, draft=None):
        model = model or OracleModel()
        prompt = torch.tensor([[1, 2]])
        expected = oracle_sequence(model, prompt[0].tolist(), count, eos)
        draft = oracle_draft(model, corrupt_at) if draft is None else draft
        result = implementation.greedy_decode(model, prompt, count, eos, draft=draft,
                                               draft_tokens=3, forward=oracle_forward)
        self.assertEqual(result.tokens.tolist(), [expected])
        self.assertEqual(result.target_calls, len(model.calls))
        self.assertEqual(result.target_tokens, sum(len(chunk) for _, chunk in model.calls))
        if len(expected) > prompt.shape[1]:
            self.assertEqual(result.cache.length, len(expected) - 1)
            self.assertEqual(result.cache.layers[0][0].flatten().tolist(), expected[:-1])
            self.assertEqual(result.pending.tolist(), [[expected[-1]]])
        return result, model

    def test_perfect_draft_actual_calls_tokens_and_bonus(self):
        result, _ = self.check_oracle(count=10)
        self.assertEqual((result.target_calls, result.target_tokens), (4, 11))
        self.assertEqual((result.proposed, result.accepted, result.bonus_tokens), (6, 6, 2))
        self.assertEqual(result.rejected_rounds, 0)
        model = OracleModel()
        baseline = m.greedy_decode(model, torch.tensor([[1, 2]]), 10, forward=oracle_forward)
        self.assertEqual((baseline.target_calls, baseline.target_tokens), (10, 11))
        self.assertTrue(torch.equal(baseline.tokens, result.tokens))

    def test_every_rejection_position_and_subsequent_cache_positions(self):
        for position in (0, 1, 2):
            result, model = self.check_oracle(corrupt_at=position, count=17)
            self.assertGreater(result.rejected_rounds, 0)
            self.assertGreater(result.target_tokens, len(result.tokens[0]) - 1)
            # The next call's prefix must be an actual committed sequence prefix,
            # not a suffix left from a rejected candidate branch.
            for old, _ in model.calls:
                self.assertEqual(list(old), result.tokens[0, :len(old)].tolist())

    def test_short_empty_drafts_and_budget_boundaries(self):
        for count in range(0, 16):
            for length in (0, 1, 2):
                model = OracleModel()
                def short(prefix, limit):
                    return oracle_draft(model)(prefix, limit)[:, :min(length, limit)]
                self.check_oracle(count=count, model=model, draft=short)

    def test_eos_prompt_first_accepted_corrected_rejected_and_bonus(self):
        # Increasing scalar target: prompt [1,2] -> [3,4,5,6,7,8,...].
        for eos in (2, 3, 4, 5, 7):
            model = OracleModel(vocab=11, next_fn=lambda ids: (ids[-1]+1) % 11)
            result, _ = self.check_oracle(model=model, eos=eos, count=10)
            if eos == 2:
                self.assertEqual(result.target_calls, 0)
                self.assertIsNone(result.cache)
            if eos in (4, 5):
                self.assertEqual(result.bonus_tokens, 0)
            if eos == 7:
                self.assertEqual(result.bonus_tokens, 1)
        model = OracleModel(vocab=11, next_fn=lambda ids: (ids[-1]+1) % 11)
        self.check_oracle(model=model, eos=5, corrupt_at=1)  # corrected EOS
        model = OracleModel(vocab=11, next_fn=lambda ids: (ids[-1]+1) % 11)
        def false_eos(prefix, limit):
            return torch.full((1, limit), 9, dtype=torch.long)
        result, _ = self.check_oracle(model=model, eos=9, count=10, draft=false_eos)
        self.assertEqual(result.tokens[0, -1].item(), 9)
        self.assertGreater(result.rejected_rounds, 0)

    def test_ties_follow_smallest_id_in_both_paths(self):
        model = OracleModel()
        model.scores = lambda ids: torch.tensor([0., 2., 2., 1., 0., 0., 0.], dtype=torch.float64)
        result, _ = self.check_oracle(model=model, count=9, corrupt_at=0)
        self.assertEqual(result.tokens[0, 2:].tolist(), [1] * 9)

    def test_real_s5_full_context_oracle_and_cache_logits(self):
        config = m.s5.MoEConfig(vocab_size=11, d_model=8, num_heads=2,
            num_kv_heads=1, num_layers=2, max_seq_len=24, expert_width=4, shared_width=3)
        model = m.MTPGPT(config, depth=2).double().eval()
        prompt = torch.tensor([[1, 2, 3]])
        reference = prompt.clone()
        with torch.no_grad():
            for _ in range(13):
                row = model(reference)[0][:, -1]
                reference = torch.cat((reference, row.argmax(-1, keepdim=True)), dim=1)
        drafter = m.MTPDraft(model)
        calls = []
        def spy(model, chunk, cache):
            calls.append(chunk.shape[1])
            return m.s5.cached_forward(model, chunk, cache)
        result = m.greedy_decode(model, prompt, 13, draft=drafter, forward=spy)
        self.assertTrue(torch.equal(reference, result.tokens))
        self.assertEqual(result.target_calls, len(calls))
        self.assertEqual(result.target_tokens, sum(calls))
        self.assertEqual(result.draft_requests, drafter.calls)
        self.assertGreater(drafter.tokens_processed, 0)
        # Full prefix reconstructs every layer's K/V independently of decoder trims.
        _, fresh = m.s5.cached_forward(model, reference[:, :-1])
        for a, b in zip(result.cache.layers, fresh.layers):
            for actual, expected in zip(a, b):
                torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)
        continuation, final_cache = m.s5.cached_forward(model, result.pending, result.cache)
        full = model(reference)[0][:, -1:]
        torch.testing.assert_close(continuation, full, rtol=1e-10, atol=1e-11)
        self.assertEqual(final_cache.length, reference.shape[1])
        # Changed weights cannot consume an earlier request cache.
        with torch.no_grad():
            model.mtp_heads[0].weight.add_(.01)
        with self.assertRaisesRegex(ValueError, 'changed'):
            m.s5.cached_forward(model, result.pending, result.cache)

    def test_trim_is_nonmutating_and_validates_bounds(self):
        model = OracleModel()
        _, cache = oracle_forward(model, torch.tensor([[1, 2, 3, 4]]))
        trimmed = m.trim_cache(cache, 2)
        self.assertEqual(cache.length, 4)
        self.assertEqual(trimmed.length, 2)
        self.assertIs(trimmed.owner, cache.owner)
        self.assertEqual(trimmed.versions, cache.versions)
        trimmed.layers[0][0].fill_(99)
        self.assertEqual(cache.layers[0][0].flatten().tolist(), [1, 2, 3, 4])
        for length in (0, -1, 5, True):
            with self.assertRaises(ValueError):
                m.trim_cache(cache, length)

    def test_invalid_requests_and_drafter_outputs(self):
        model = OracleModel()
        prompt = torch.tensor([[1, 2]])
        for count in (-1, True, 47):
            with self.assertRaises(ValueError):
                m.greedy_decode(model, prompt, count, forward=oracle_forward)
        for eos in (-1, 7, True):
            with self.assertRaises(ValueError):
                m.greedy_decode(model, prompt, 5, eos_id=eos, forward=oracle_forward)
        for proposal in (torch.tensor([[-1]]), torch.tensor([[7]]), torch.tensor([[1.]]),
                         torch.tensor([1]), torch.ones(1, 4, dtype=torch.long), [[1]]):
            with self.assertRaises(ValueError):
                m.greedy_decode(model, prompt, 5, draft=lambda p, n: proposal, forward=oracle_forward)
        with self.assertRaises(ValueError):
            m.greedy_decode(model, prompt.repeat(2, 1), 5, forward=oracle_forward)
        model.train()
        with self.assertRaises(ValueError):
            m.greedy_decode(model, prompt, 5, forward=oracle_forward)


@unittest.skipUnless(READY, torch_reason('P23', VERSION, PRESENT))
class BudgetAndMutationTests(unittest.TestCase):
    def test_arithmetic_curve_and_carry_by_independent_integer_oracle(self):
        rows = m.arithmetic_curve()
        self.assertEqual([row['correct'] for row in rows], [0, 1, 3, 5, 6])
        self.assertEqual([row['mean_steps'] for row in rows], [0, 1, 11/6, 14/6, 15/6])
        self.assertEqual(m.column_add(48, 37, 1).answer, 5)
        complete = m.column_add(48, 37, 2)
        self.assertEqual(complete.answer, 85)
        self.assertEqual(complete.trace, ((8, 7, 0, 5, 1), (4, 3, 1, 8, 0)))
        self.assertEqual(m.column_add(999, 1, 3).answer, 1000)
        self.assertEqual(m.column_add(999, 1, 2).answer, 0)
        for a in range(0, 150, 7):
            for b in range(0, 150, 11):
                value = m.column_add(a, b, 9)
                self.assertEqual(value.answer, a+b)
                self.assertTrue(value.complete)
                self.assertEqual(value.steps, max(len(str(a)), len(str(b))))
        for args in ((1, -1, 2), (True, 1, 2), (1, 1, -1)):
            with self.assertRaises(ValueError):
                m.column_add(*args)
        with self.assertRaises(ValueError):
            m.arithmetic_curve(tasks=())

    def test_hand_sampling_correction_is_math_only_not_a_sampler_claim(self):
        p, q = (Fraction(3, 5), Fraction(2, 5)), (Fraction(1, 5), Fraction(4, 5))
        accepted_mass = [min(a, b) for a, b in zip(p, q)]
        rejected_mass = 1 - sum(accepted_mass)
        residual = [max(a-b, 0) for a, b in zip(p, q)]
        normalizer = sum(residual)
        restored = [a + rejected_mass*r/normalizer for a, r in zip(accepted_mass, residual)]
        self.assertEqual(restored, list(p))
        self.assertEqual(accepted_mass, [Fraction(1, 5), Fraction(2, 5)])
        self.assertEqual(rejected_mass, Fraction(2, 5))

    def test_deleted_correction_mutant_fails_fast_without_timeout(self):
        """Execute F07 itself. A bounded spy prevents a hang if the guard regresses.

        The spy's AssertionError is NOT an accepted kill: only the production
        no-progress RuntimeError after exactly prefill + first verification passes.
        """
        source = SCRIPT.read_text(encoding='utf-8')
        old = '                emitted.append(correct)\n'
        self.assertEqual(source.count(old), 1)
        mutant = types.ModuleType('mtp_mutant_missing_correction')
        mutant.__file__ = str(SCRIPT)
        sys.modules[mutant.__name__] = mutant
        try:
            exec(compile(source.replace(old, ''), str(SCRIPT), 'exec'), mutant.__dict__)
            model = OracleModel()
            def bounded_forward(target, chunk, cache):
                if len(target.calls) >= 2:
                    raise AssertionError('no-progress guard failed before another target call')
                return oracle_forward(target, chunk, cache)
            with self.assertRaisesRegex(RuntimeError, '^draft/verify made no progress$'):
                mutant.greedy_decode(model, torch.tensor([[1, 2]]), 6,
                    draft=oracle_draft(model, corrupt_at=0), forward=bounded_forward)
            self.assertEqual(len(model.calls), 2)
        finally:
            del sys.modules[mutant.__name__]
        # No proposal is NOT no progress: target-only fallback must still finish.
        result, _ = GreedyTests().check_oracle(count=6,
            draft=lambda prefix, limit: prefix[:, :0])
        self.assertEqual(result.target_calls, 6)
        self.assertEqual(result.proposed, 0)

    def test_executable_mutants_are_killed_by_independent_contracts(self):
        source = SCRIPT.read_text(encoding='utf-8')
        contracts = GreedyTests()
        def label_check(module):
            x = torch.tensor([[1, 2, 3, 4, 5]])
            valid = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.bool)
            segments = torch.tensor([[0, 0, 1, 0, 0]])
            supervise = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.bool)
            for h in (1, 2, 3):
                for v, s, sup in ((None, None, None), (valid, None, None),
                                  (None, segments, None), (None, None, supervise)):
                    self.assertTrue(torch.equal(module.future_labels(x, h, v, s, sup),
                                                scalar_labels(x, h, v, s, sup)))
        def loss_check(module):
            logits = torch.tensor([[[0., 0.], [0., 0.]]])
            self.assertAlmostEqual(float(module.masked_cross_entropy(logits, torch.tensor([[1, -100]]))),
                                   math.log(2), places=6)
        def decode_check(module):
            for corrupt in (None, 0, 1, 2):
                contracts.check_oracle(implementation=module, corrupt_at=corrupt, count=13)
            contracts.check_oracle(implementation=module, count=10)
            model = OracleModel(vocab=11, next_fn=lambda ids: (ids[-1]+1) % 11)
            contracts.check_oracle(implementation=module, count=10, eos=5, model=model)
        def draft_check(module):
            MTPTests().check_draft_head_positions(module)
        cases = [
            ('future off by one', 'torch.where(keep, tokens[:, horizon:], -100)',
             'torch.where(keep, tokens[:, horizon-1:T-1], -100)', label_check),
            ('ignore segment boundary',
             'keep &= segments[:, offset:offset + count] == segments[:, :count]',
             'keep &= torch.ones_like(keep)', label_check),
            ('source not destination mask', 'keep = supervise[:, horizon:].clone()',
             'keep = supervise[:, :count].clone()', label_check),
            ('wrong CE denominator', 'F.cross_entropy(logits[keep], labels[keep])',
             'F.cross_entropy(logits[keep], labels[keep]) * keep.float().mean()', loss_check),
            ('ignore rejection', 'if candidate != correct:', 'if False:', decode_check),
            ('verify offset', 'correct = int(verify[0, j].argmax())',
             'correct = int(verify[0, j+1].argmax())', decode_check),
            ('wrong bonus', 'emitted.append(int(verify[0, K].argmax()))',
             'emitted.append(int(verify[0, 0].argmax()))', decode_check),
            ('cache off by one', 'old_length + len(emitted))',
             'old_length + len(emitted) + 1)', decode_check),
            ('ignore accepted EOS', 'if eos_id is not None and candidate == eos_id:',
             'if False:', decode_check),
            ('no bonus reservation', 'limit = min(draft_tokens, remaining - 1)',
             'limit = min(draft_tokens, remaining)', decode_check),
            ('skip main draft head', 'for x in logits[:limit]',
             'for x in logits[1:limit+1]', draft_check),
        ]
        label_check(m)
        loss_check(m)
        decode_check(m)
        draft_check(m)
        for index, (name, old, new, contract) in enumerate(cases):
            with self.subTest(mutant=name):
                self.assertEqual(source.count(old), 1)
                module = types.ModuleType(f'mtp_mutant_{index}')
                module.__file__ = str(SCRIPT)
                sys.modules[module.__name__] = module
                try:
                    exec(compile(source.replace(old, new), str(SCRIPT), 'exec'), module.__dict__)
                    with self.assertRaises((AssertionError, ValueError, IndexError)):
                        contract(module)
                finally:
                    del sys.modules[module.__name__]


if __name__ == '__main__':
    unittest.main()
