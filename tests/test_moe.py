"""P18 independent behavior contract, defined before the S5 implementation.

Mutation sentinels must reject: ignoring router weights; using all E experts;
putting balancing bias into combine weights; averaging shared experts; reversing
capacity order; re-normalizing after dropping; silently resetting capacity per
cached chunk; and ignoring runtime/buffer changes in cache identity. Full/cache
agreement alone is NOT an oracle because both paths share the same MoE module.
"""
from dataclasses import replace
import copy
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import unittest

from torch_test_support import torch_status, torch_reason

READY, VERSION, PRESENT = torch_status()
SCRIPT = Path(__file__).resolve().parents[1] / 'code/19_moe.py'
if READY:
    import torch
    spec = importlib.util.spec_from_file_location('moe_test', SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)


def scalar_reference(layer, x):
    """Independent scalar routing + explicit SwiGLU algebra, differentiable.

    No production route/dispatch/forward/SwiGLU helper is called. Python sorting
    defines the discrete choice; selected weights retain their autograd graph.
    """
    flat = x.reshape(-1, x.shape[-1])
    outputs, counts = [], [0] * len(layer.experts)
    def expert(row, module):
        gate = torch.stack([sum(row[j] * module.gate.weight[i, j]
                                for j in range(row.numel()))
                            for i in range(module.gate.out_features)])
        up = torch.stack([sum(row[j] * module.up.weight[i, j]
                              for j in range(row.numel()))
                          for i in range(module.up.out_features)])
        hidden = gate * torch.sigmoid(gate) * up
        return torch.stack([sum(hidden[j] * module.down.weight[i, j]
                                for j in range(hidden.numel()))
                            for i in range(row.numel())])
    for row in flat:
        logits = torch.stack([sum(row[j] * layer.router.weight[i, j]
                                  for j in range(row.numel()))
                              for i in range(len(layer.experts))])
        exps = torch.exp(logits - logits.max())
        probs = exps / exps.sum()
        scores = probs.detach().tolist()
        if layer.balance_mode == 'bias':
            scores = [score + float(bias) for score, bias in
                      zip(scores, layer.balance_bias.detach())]
        chosen = sorted(range(len(scores)), key=lambda e: (-scores[e], e))[:layer.top_k]
        shifted = logits[chosen] - max(logits[chosen].detach().tolist())
        weights = shifted.exp() / shifted.exp().sum()
        out = row * 0
        for slot, e in enumerate(chosen):
            accepted = layer.capacity is None or counts[e] < layer.capacity
            counts[e] += 1
            if accepted:
                out = out + weights[slot] * expert(row, layer.experts[e])
        for shared in layer.shared:
            out = out + expert(row, shared)
        outputs.append(out)
    return torch.stack(outputs).reshape_as(x)


class MoECLITests(unittest.TestCase):
    def test_missing_torch_cli_and_import_contract(self):
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
        imported = subprocess.run([sys.executable, '-c', blocker + """
try:
    runpy.run_path(sys.argv[1], run_name='moe_without_torch')
except ModuleNotFoundError as exc:
    assert exc.name == 'torch'
else:
    raise AssertionError('import should report missing dependency')
""", str(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(imported.returncode, 0, imported.stderr)


@unittest.skipUnless(READY, torch_reason('P18', VERSION, PRESENT))
class MoETests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(180)
        self.config = m.MoEConfig(vocab_size=17, d_model=16, num_heads=4,
            num_kv_heads=2, num_layers=2, max_seq_len=16, expert_width=7,
            shared_width=5)
        self.tokens = torch.tensor([[1, 2, 3, 2, 6, 7], [4, 0, 2, 1, 4, 3]])

    def layer(self, **kwargs):
        return m.TeachingMoE(4, 4, 2, 3, shared_experts=2,
                            shared_width=2, **kwargs).double()

    def chunks(self, model, sizes):
        out, cache, start = [], None, 0
        for size in sizes:
            logits, cache = m.cached_forward(model, self.tokens[:, start:start + size], cache)
            out.append(logits)
            start += size
        self.assertEqual(start, self.tokens.shape[1])
        return torch.cat(out, 1), cache

    def test_hand_route_weights_counts_aux_and_ties(self):
        info = m.routing_walkthrough()
        self.assertEqual(info.indices.tolist(), [[0, 1], [3, 2], [0, 1]])
        torch.testing.assert_close(info.weights, torch.tensor(
            [[4/7, 3/7], [4/7, 3/7], [.5, .5]], dtype=torch.float64), rtol=1e-14, atol=1e-14)
        self.assertEqual(info.requested.tolist(), [2, 2, 1, 1])
        self.assertEqual(info.accepted.tolist(), [2, 2, 1, 1])
        self.assertAlmostEqual(info.aux_loss.item(), 16/15)
        equal = m.route(torch.zeros(5, 4, dtype=torch.float64), 2)
        self.assertEqual(equal.indices.tolist(), [[0, 1]] * 5)
        torch.testing.assert_close(equal.weights, torch.full((5, 2), .5, dtype=torch.float64))
        self.assertAlmostEqual(equal.aux_loss.item(), 1.)
        self.assertEqual(equal.requested.tolist(), [5, 5, 0, 0])  # aux=1 is not proof of balanced hard routes.

    def test_hand_scalar_output_and_router_input_gradients(self):
        class Scale(torch.nn.Module):
            def __init__(self, factor):
                super().__init__()
                self.factor = factor
            def forward(self, x):
                return x * self.factor
        layer = m.TeachingMoE(2, 3, 2, 1, shared_experts=0).double()
        layer.experts = torch.nn.ModuleList([Scale(1), Scale(10), Scale(100)])
        layer.shared = torch.nn.ModuleList([Scale(3), Scale(5)])
        with torch.no_grad():
            layer.router.weight.copy_(torch.tensor([[0., 0.], [math.log(2), 0.],
                                                    [math.log(4), 0.]], dtype=torch.float64))
        x = torch.tensor([[1., 0.]], dtype=torch.float64, requires_grad=True)
        output = layer(x)
        torch.testing.assert_close(output, torch.tensor([[78., 0.]], dtype=torch.float64))
        output.sum().backward()
        torch.testing.assert_close(layer.router.weight.grad, torch.tensor(
            [[0., 0.], [-20., 0.], [20., 0.]], dtype=torch.float64), rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(x.grad, torch.tensor([[78 + 20 * math.log(2), 78.]], dtype=torch.float64))

    def test_independent_scalar_output_and_all_parameter_gradients(self):
        for k, shared, capacity, mode in ((1, 0, None, 'none'), (2, 2, None, 'aux'),
                                          (4, 1, None, 'none'), (2, 2, 1, 'bias'), (2, 0, 0, 'none')):
            with self.subTest(k=k, shared=shared, capacity=capacity, mode=mode):
                layer = m.TeachingMoE(4, 4, k, 3, shared_experts=shared,
                    shared_width=2, capacity=capacity, balance_mode=mode).double()
                if mode == 'bias':
                    layer.balance_bias.copy_(torch.tensor([.1, -.2, .3, 0.]))
                reference = copy.deepcopy(layer)
                x = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
                xr = x.detach().clone().requires_grad_()
                output, expected = layer(x), scalar_reference(reference, xr)
                torch.testing.assert_close(output, expected, rtol=1e-11, atol=1e-12)
                probe = torch.randn_like(output)
                (output * probe).sum().backward()
                (expected * probe).sum().backward()
                torch.testing.assert_close(x.grad, xr.grad, rtol=1e-10, atol=1e-12)
                for (name, p), (rname, q) in zip(layer.named_parameters(), reference.named_parameters()):
                    self.assertEqual(name, rname)
                    pg = torch.zeros_like(p) if p.grad is None else p.grad
                    qg = torch.zeros_like(q) if q.grad is None else q.grad
                    torch.testing.assert_close(pg, qg, rtol=1e-10, atol=1e-12, msg=name)

    def test_sparse_execution_empty_experts_and_top1_router_gradient(self):
        layer = m.TeachingMoE(4, 4, 1, 3, shared_experts=0, balance_mode='none').double()
        with torch.no_grad():
            layer.router.weight.zero_()
            layer.router.weight[0].fill_(1.)
        calls = [0] * 4
        handles = []
        for e, expert in enumerate(layer.experts):
            def hook(module, args, result, expert_id=e):
                calls[expert_id] += 1
            handles.append(expert.register_forward_hook(hook))
        x = torch.ones(5, 4, dtype=torch.float64, requires_grad=True)
        output = layer(x)
        self.assertEqual(layer.last_info.requested.tolist(), [5, 0, 0, 0])
        self.assertEqual(calls, [1, 0, 0, 0])
        output.square().sum().backward()
        torch.testing.assert_close(layer.router.weight.grad, torch.zeros_like(layer.router.weight))
        self.assertTrue(all(p.grad is None for e in layer.experts[1:] for p in e.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in layer.experts[0].parameters()))
        for handle in handles:
            handle.remove()
        layer.zero_grad(set_to_none=True)
        layer(x)
        layer.last_info.aux_loss.backward()
        self.assertGreater(layer.router.weight.grad.abs().sum().item(), 0.)

    def test_empty_token_input_finite_backward_and_zero_load(self):
        layer = self.layer()
        x = torch.empty(2, 0, 4, dtype=torch.float64, requires_grad=True)
        output = layer(x)
        self.assertEqual(output.shape, x.shape)
        self.assertEqual(layer.last_info.requested.tolist(), [0, 0, 0, 0])
        self.assertEqual(layer.last_info.aux_loss.item(), 0.)
        (output.sum() + layer.last_info.aux_loss).backward()
        self.assertEqual(x.grad.numel(), 0)

    def test_capacity_fixed_order_no_renormalization_shared_survives(self):
        info = m.routing_walkthrough(1)
        self.assertEqual(info.kept.tolist(), [[True, True], [True, True], [False, False]])
        self.assertEqual(info.accepted.tolist(), [1, 1, 1, 1])
        # A partially dropped token must retain 3/7, not be renormalized to 1.
        probs = torch.tensor([[.5, .1, .4], [.4, .3, .3]], dtype=torch.float64)
        partial = m.route(probs.log(), 2, capacity=1)
        self.assertEqual(partial.indices.tolist(), [[0, 2], [0, 1]])
        self.assertEqual(partial.kept.tolist(), [[True, True], [False, True]])
        self.assertAlmostEqual((partial.weights * partial.kept)[1].sum().item(), 3/7)
        layer = self.layer(capacity=0)
        x = torch.randn(3, 4, dtype=torch.float64)
        torch.testing.assert_close(layer(x), sum(expert(x) for expert in layer.shared))
        self.assertEqual(layer.last_info.accepted.tolist(), [0, 0, 0, 0])
        self.assertEqual(sum(layer.last_info.requested.tolist()), 6)

    def test_selection_bias_does_not_enter_combine_weights(self):
        logits = torch.tensor([[0., math.log(2), math.log(4)]], dtype=torch.float64)
        base = m.route(logits, 2)
        reordered = m.route(logits, 2, torch.tensor([0., .5, 0.], dtype=torch.float64))
        self.assertEqual(base.indices.tolist(), [[2, 1]])
        self.assertEqual(reordered.indices.tolist(), [[1, 2]])
        torch.testing.assert_close(reordered.weights, torch.tensor([[1/3, 2/3]], dtype=torch.float64))
        selected = m.route(logits, 2, torch.tensor([1., 0., 0.], dtype=torch.float64))
        self.assertEqual(selected.indices.tolist(), [[0, 2]])
        torch.testing.assert_close(selected.weights, torch.tensor([[1/5, 4/5]], dtype=torch.float64))
        # Selected logits may both underflow in the full softmax: still finite.
        extreme = m.route(torch.tensor([[0., -1000., -1001.]]), 2,
                          torch.tensor([0., 2., 2.]))
        self.assertTrue(torch.isfinite(extreme.weights).all())
        self.assertAlmostEqual(extreme.weights.sum().item(), 1.)

    def test_bias_update_direction_explicit_and_state_dict(self):
        layer = self.layer(balance_mode='bias')
        before = layer.balance_bias.clone()
        layer(torch.ones(2, 4, dtype=torch.float64))
        torch.testing.assert_close(layer.balance_bias, before)
        layer.update_balance_bias(torch.tensor([4, 0, 2, 2]), .1)
        torch.testing.assert_close(layer.balance_bias, torch.tensor([-.1, .1, 0., 0.], dtype=torch.float64))
        self.assertFalse(layer.balance_bias.requires_grad)
        self.assertNotIn('balance_bias', dict(layer.named_parameters()))
        fresh = self.layer(balance_mode='bias')
        fresh.load_state_dict(layer.state_dict())
        torch.testing.assert_close(fresh.balance_bias, layer.balance_bias)
        for loads in (torch.tensor([-1, 0, 0, 0]), torch.zeros(4), torch.zeros(3, dtype=torch.int64)):
            with self.assertRaises(ValueError):
                layer.update_balance_bias(loads)
        for step in (True, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                layer.update_balance_bias(torch.ones(4, dtype=torch.int64), step)
        with self.assertRaisesRegex(ValueError, 'balance_mode'):
            self.layer().update_balance_bias(torch.ones(4, dtype=torch.int64))

    def test_auxiliary_formula_and_router_gradients_independent(self):
        logits = torch.tensor([[2., 1., 0.], [3., .5, -1.]], dtype=torch.float64, requires_grad=True)
        info = m.route(logits, 1)
        # Both choose expert 0: f=[1,0,0], so unscaled L=3*mean(p0).
        reference_logits = logits.detach().clone().requires_grad_()
        exponentials = (reference_logits - reference_logits.max(-1, keepdim=True).values).exp()
        expected = 3 * (exponentials[:, 0] / exponentials.sum(-1)).mean()
        torch.testing.assert_close(info.aux_loss, expected)
        actual_gradient = torch.autograd.grad(info.aux_loss, logits)[0]
        expected_gradient = torch.autograd.grad(expected, reference_logits)[0]
        torch.testing.assert_close(actual_gradient, expected_gradient)
        self.assertGreater(actual_gradient.abs().sum().item(), 0.)

    def test_parameter_active_budget_and_fine_grained_example(self):
        budget = m.parameter_budget(4, 4, 2, 2, 1, 2)
        self.assertEqual(budget, dict(expert=24, router=16, shared=24, total=136, active=88))
        layer = m.TeachingMoE(4, 4, 2, 2, shared_experts=1, shared_width=2)
        self.assertEqual(sum(p.numel() for p in layer.parameters()), 136)
        coarse = m.parameter_budget(4, 2, 1, 8)
        fine = m.parameter_budget(4, 4, 2, 4)
        self.assertEqual(coarse['total'] - coarse['router'], 192)
        self.assertEqual(fine['total'] - fine['router'], 192)
        self.assertEqual(coarse['active'] - coarse['router'], 96)
        self.assertEqual(fine['active'] - fine['router'], 96)
        self.assertEqual((coarse['total'], fine['total']), (200, 208))

    def test_s5_really_replaces_s4_ffn_and_keeps_all_other_weights(self):
        from dataclasses import fields
        base_config = m.s4.GroupedConfig(**{field.name: getattr(self.config, field.name)
                                         for field in fields(m.s4.GroupedConfig)})
        torch.manual_seed(5)
        base = m.s4.GroupedGPT(base_config)
        torch.manual_seed(5)
        model = m.MoEGPT(self.config)
        for block in model.blocks:
            self.assertIsInstance(block.mlp, m.TeachingMoE)
            self.assertIsInstance(block.attn, m.s4.GroupedAttention)
        for name, value in base.state_dict().items():
            if '.mlp.' not in name:
                torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
        self.assertFalse(torch.allclose(base(self.tokens)[0], model(self.tokens)[0]))

    def test_s5_aux_loss_training_backward_optimizer_and_checkpoint(self):
        model = m.MoEGPT(self.config)
        targets = self.tokens.roll(-1, 1)
        logits, total = model(self.tokens, targets)
        ce = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
        aux = sum(b.mlp.last_info.aux_loss for b in model.blocks) / len(model.blocks)
        torch.testing.assert_close(total, ce + .01 * aux)
        total.backward()
        for name, p in model.named_parameters():
            self.assertTrue(p.grad is None or torch.isfinite(p.grad).all(), name)
        before = model.blocks[0].mlp.router.weight.detach().clone()
        torch.optim.SGD(model.parameters(), lr=.1).step()
        self.assertFalse(torch.equal(before, model.blocks[0].mlp.router.weight))
        model.eval()
        fresh = m.MoEGPT(self.config).eval()
        fresh.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(model(self.tokens)[0], fresh(self.tokens)[0], rtol=0, atol=0)
        for mode in ('none', 'bias'):
            no_aux = m.MoEGPT(replace(self.config, balance_mode=mode))
            logits, loss = no_aux(self.tokens, targets)
            torch.testing.assert_close(loss, torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten()))

    def test_full_cache_alignment_each_mode_and_extreme_k(self):
        for mode in ('none', 'aux', 'bias'):
            for k in (1, 2, 4):
                with self.subTest(mode=mode, k=k):
                    model = m.MoEGPT(replace(self.config, balance_mode=mode, top_k=k)).eval()
                    if mode == 'bias':
                        for block in model.blocks:
                            block.mlp.balance_bias.copy_(torch.tensor([.1, -.2, .3, 0.]))
                    full = model(self.tokens)[0]
                    for sizes in ([6], [1] * 6, [2, 3, 1]):
                        actual, cache = self.chunks(model, sizes)
                        torch.testing.assert_close(actual, full, rtol=1e-5, atol=2e-6)
                        self.assertEqual(cache.layers[0][0].shape, (2, 2, 6, 4))
                        self.assertEqual(m.s4.cache_bytes(cache), 1536)

    def test_future_mutation_batch_independence_and_cache_reset(self):
        model = m.MoEGPT(self.config).eval()
        baseline = model(self.tokens)[0]
        changed = self.tokens.clone()
        changed[:, 3:] = (changed[:, 3:] + 1) % 17
        torch.testing.assert_close(model(changed)[0][:, :3], baseline[:, :3])
        torch.testing.assert_close(model(self.tokens[:1])[0], baseline[:1])
        _, cache = m.cached_forward(model, self.tokens[:, :2])
        old = [tuple(t.clone() for t in pair) for pair in cache.layers]
        final, extended = m.cached_forward(model, self.tokens[:, 2:], cache)
        torch.testing.assert_close(final, baseline[:, 2:], rtol=1e-5, atol=2e-6)
        self.assertEqual(cache.length, 2)
        self.assertEqual(extended.length, 6)
        for before, after in zip(old, cache.layers):
            for a, b in zip(before, after):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        reset, _ = m.cached_forward(model, changed, None)
        torch.testing.assert_close(reset, model(changed)[0])

    def test_runtime_and_buffer_mutations_invalidate_cache(self):
        mutations = [lambda model: setattr(model.blocks[0].mlp, 'top_k', 1),
                     lambda model: setattr(model.blocks[0].mlp, 'balance_mode', 'none'),
                     lambda model: model.blocks[0].mlp.balance_bias.add_(.1),
                     lambda model: setattr(model.blocks[0].mlp, 'balance_bias', torch.zeros(4)),
                     lambda model: model.blocks[0].mlp.router.weight.add_(.1),
                     lambda model: setattr(model.blocks[0].ln2, 'eps', .1),
                     lambda model: setattr(model.blocks[0].attn, 'rope_base', 100.),
                     lambda model: setattr(model, 'config', replace(model.config, aux_weight=.2)),
                     lambda model: model.double()]
        for mutate in mutations:
            model = m.MoEGPT(self.config).eval()
            _, cache = m.cached_forward(model, self.tokens[:, :2])
            with torch.no_grad():
                mutate(model)
            with self.assertRaisesRegex(ValueError, 'changed'):
                m.cached_forward(model, self.tokens[:, 2:3], cache)
        model = m.MoEGPT(replace(self.config, balance_mode='bias')).eval()
        _, cache = m.cached_forward(model, self.tokens[:, :2])
        model.blocks[0].mlp.update_balance_bias(torch.tensor([4, 0, 0, 0]))
        with self.assertRaisesRegex(ValueError, 'changed'):
            m.cached_forward(model, self.tokens[:, 2:3], cache)

    def test_capacity_runtime_rejected_by_both_lm_paths(self):
        model = m.MoEGPT(self.config).eval()
        _, cache = m.cached_forward(model, self.tokens[:, :2])
        model.blocks[0].mlp.capacity = 2
        for action in (lambda: model(self.tokens),
                       lambda: m.cached_forward(model, self.tokens, None),
                       lambda: m.cached_forward(model, self.tokens[:, 2:3], cache)):
            with self.assertRaisesRegex(ValueError, 'dropless'):
                action()

    def test_cache_foreign_training_length_and_bad_tokens(self):
        model = m.MoEGPT(self.config).eval()
        _, cache = m.cached_forward(model, self.tokens[:, :2])
        with self.assertRaisesRegex(ValueError, 'different'):
            m.cached_forward(m.MoEGPT(self.config).eval(), self.tokens, cache)
        with self.assertRaisesRegex(ValueError, 'eval'):
            m.cached_forward(model.train(), self.tokens)
        model.eval()
        with self.assertRaisesRegex(ValueError, 'max_seq_len'):
            m.cached_forward(model, torch.ones(2, 15, dtype=torch.long), cache)
        for tokens in (self.tokens.float(), self.tokens[:, :0], self.tokens * 100):
            with self.assertRaises(ValueError):
                m.cached_forward(model, tokens)
        with self.assertRaises(ValueError):
            m.cached_forward(m.s4.GroupedGPT(m.s4.GroupedConfig(vocab_size=17)), self.tokens)

    def test_invalid_configs_and_nonfinite_inputs(self):
        for kwargs in (dict(num_experts=0), dict(num_experts=True), dict(top_k=0),
                       dict(top_k=5), dict(top_k=1.5), dict(expert_width=0),
                       dict(shared_experts=-1), dict(shared_experts=True), dict(shared_width=0),
                       dict(balance_mode='aux+bias'), dict(aux_weight=-1), dict(aux_weight=True),
                       dict(aux_weight=float('nan')), dict(aux_weight=float('inf')),
                       dict(ffn_kind='gelu'), dict(ffn_bias=True), dict(intermediate_size=8)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replace(self.config, **kwargs)
        for kwargs in (dict(capacity=-1), dict(capacity=1.1), dict(capacity=True)):
            with self.assertRaises(ValueError):
                self.layer(**kwargs)
        for x in (torch.randn(4), torch.randn(2, 3), torch.ones(2, 4, dtype=torch.long),
                  torch.full((2, 4), float('nan'), dtype=torch.float64),
                  torch.full((2, 4), float('inf'), dtype=torch.float64)):
            with self.assertRaises(ValueError):
                self.layer()(x)
        for logits in (torch.ones(3), torch.empty(3, 0), torch.ones(3, 4, dtype=torch.long),
                       torch.full((3, 4), float('nan')), torch.full((3, 4), float('inf'))):
            with self.assertRaises(ValueError):
                m.route(logits, 2)
        for bias in (torch.zeros(3), torch.full((4,), float('nan')), torch.zeros(4, dtype=torch.long)):
            with self.assertRaises(ValueError):
                m.route(torch.zeros(2, 4), 2, bias)
        with self.assertRaises(ValueError):
            m.MoEGPT(m.s4.GroupedConfig(vocab_size=17))

    def test_mutations_fail_independent_oracle_not_only_cache_agreement(self):
        from unittest.mock import patch
        layer = self.layer(balance_mode='bias')
        layer.balance_bias.copy_(torch.tensor([.4, 0., -.1, .2]))
        x = torch.tensor([[1., .2, -.3, .7], [-.3, .5, 1., -.4]], dtype=torch.float64)
        expected = scalar_reference(layer, x)
        original = m.route
        def uniform(*args, **kwargs):
            result = original(*args, **kwargs)
            return replace(result, weights=torch.full_like(result.weights, 1 / layer.top_k))
        def all_experts(logits, top_k, balance_bias=None, capacity=None):
            return original(logits, logits.shape[-1], balance_bias, capacity)
        def biased_weights(logits, top_k, balance_bias=None, capacity=None):
            result = original(logits, top_k, balance_bias, capacity)
            wrong = (logits.softmax(-1) + balance_bias).gather(1, result.indices)
            return replace(result, weights=wrong / wrong.sum(-1, keepdim=True))
        for mutant in (uniform, all_experts, biased_weights):
            with self.subTest(mutant=mutant.__name__), patch.object(m, 'route', mutant):
                with self.assertRaises(AssertionError):
                    torch.testing.assert_close(layer(x), expected, rtol=1e-9, atol=1e-10)
        handles = [expert.register_forward_hook(lambda module, args, y: y / 2) for expert in layer.shared]
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(layer(x), expected, rtol=1e-9, atol=1e-10)
        for handle in handles:
            handle.remove()


    def test_executable_capacity_mutants_fail_literal_output_oracle(self):
        from unittest.mock import patch
        layer = m.TeachingMoE(3, 3, 2, 1, shared_experts=0, capacity=1).double()
        layer.experts = torch.nn.ModuleList([torch.nn.Identity() for _ in range(3)])
        probs = torch.tensor([[.5, .1, .4], [.4, .3, .3]], dtype=torch.float64)
        with torch.no_grad():
            layer.router.weight.zero_()
            layer.router.weight[:, :2].copy_(probs.log().T)
        x = torch.eye(3, dtype=torch.float64)[:2]
        expected = torch.tensor([[1., 0., 0.], [0., 3/7, 0.]], dtype=torch.float64)
        torch.testing.assert_close(layer(x), expected)
        original = m.route
        def renormalize(logits, top_k, balance_bias=None, capacity=None):
            info = original(logits, top_k, balance_bias, capacity)
            wrong = info.weights * info.kept
            return replace(info, weights=wrong / wrong.sum(-1, keepdim=True).clamp_min(1e-30))
        def accept_last(logits, top_k, balance_bias=None, capacity=None):
            info = original(logits, top_k, balance_bias, capacity)
            wrong = torch.zeros_like(info.kept)
            for e in range(logits.shape[-1]):
                positions = (info.indices == e).nonzero()[-capacity:]
                wrong[positions[:, 0], positions[:, 1]] = True
            return replace(info, kept=wrong)
        for mutant in (renormalize, accept_last):
            with self.subTest(mutant=mutant.__name__), patch.object(m, 'route', mutant):
                with self.assertRaises(AssertionError):
                    torch.testing.assert_close(layer(x), expected, rtol=1e-12, atol=1e-12)

    def test_extreme_logits_do_not_create_artificial_unbiased_ties(self):
        logits = torch.tensor([[0., -1001., -1000.]], dtype=torch.float64)
        for bias in (None, torch.zeros(3, dtype=torch.float64)):
            info = m.route(logits, 2, bias)
            self.assertEqual(info.indices.tolist(), [[0, 2]])
            self.assertTrue(torch.isfinite(info.weights).all())


if __name__ == '__main__':
    unittest.main()
