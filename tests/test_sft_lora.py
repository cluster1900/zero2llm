"""P21 contracts: independent token, scalar, matrix and derivative oracles.

Semantic mutants execute changed source in memory; repository files are untouched.
PyTorch-only tests explicitly skip when PyTorch is missing, not fake-pass.
"""
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import types
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "code/22_sft_lora.py"
spec = importlib.util.spec_from_file_location("sft_lora_lesson", SCRIPT)
lesson = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = lesson
spec.loader.exec_module(lesson)
torch = lesson.torch


def mutant(old, new):
    source = SCRIPT.read_text(encoding="utf-8")
    if source.count(old) != 1:
        raise AssertionError(f"mutation must match exactly once: {old!r}")
    module = types.ModuleType("sft_mutant")
    sys.modules[module.__name__] = module
    exec(compile(source.replace(old, new), str(SCRIPT), "exec"), module.__dict__)
    return module


def pair(user="问红", answer="红"):
    return [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]


def token_oracle(module):
    row = module.encode_chat(pair())
    # Literal IDs are intentionally NOT derived from encode_chat or spans.
    assert row.input_ids == (1, 3, 12, 13, 5, 4, 13, 6)
    assert row.labels == (-100, -100, -100, -100, -100, -100, 13, 6)
    assert row.assistant_spans == ((6, 8),)


class TemplateTests(unittest.TestCase):
    def test_hand_example_literal_ids(self):
        token_oracle(lesson)
        self.assertEqual(len(lesson.TOKENS), 17)
        self.assertEqual(len(set(lesson.TOKENS)), 17)

    def test_multi_turn_system_and_both_assistant_eos(self):
        messages = [{"role": "system", "content": "只答颜色"}] + pair() + pair("问蓝", "蓝")
        row = lesson.encode_chat(messages)
        self.assertEqual(row.input_ids, (1, 2, 8, 9, 10, 11, 5, 3, 12, 13, 5, 4, 13, 6, 3, 12, 14, 5, 4, 14, 6))
        self.assertEqual(row.labels, (-100,) * 12 + (13, 6) + (-100,) * 5 + (14, 6))
        self.assertEqual(row.assistant_spans, ((12, 14), (19, 21)))
        self.assertEqual((row.kept_turns, row.dropped_turns), (2, 0))

    def test_training_prefix_equals_inference_for_every_turn(self):
        messages = [{"role": "system", "content": "只答颜色"}] + pair() + pair("问蓝", "蓝")
        row = lesson.encode_chat(messages)
        for turn, (start, _) in enumerate(row.assistant_spans):
            prefix = lesson.generation_prompt(messages[:2 + 2 * turn])
            self.assertEqual(prefix, row.input_ids[:start])
            self.assertEqual(prefix[-1], lesson.ASSISTANT)

    def test_padding_and_attention_mask_have_separate_jobs(self):
        rows = [lesson.encode_chat(pair()), lesson.encode_chat(pair(answer="红蓝"))]
        batch = lesson.collate(rows, pad_to=10)
        self.assertEqual(batch["input_ids"][0][-2:], [0, 0])
        self.assertEqual(batch["labels"][0][-2:], [-100, -100])
        self.assertEqual(batch["attention_mask"][0], [1] * 8 + [0, 0])
        self.assertEqual(batch["labels"][0][7], 6)  # Real assistant EOS is NOT padding.
        self.assertEqual(batch["attention_mask"][0][2], 1)  # User context remains visible.
        self.assertEqual(batch["labels"][0][2], -100)  # But is not a target.

    def test_truncation_exact_fit_and_all_boundaries(self):
        messages = pair() + pair("问蓝", "蓝")
        full = lesson.encode_chat(messages)
        self.assertEqual(len(full.input_ids), 15)
        for boundary in range(1, 8):
            with self.subTest(boundary=boundary), self.assertRaisesRegex(ValueError, "no complete"):
                lesson.encode_chat(messages, max_length=boundary)
        for boundary in range(8, 15):
            with self.subTest(boundary=boundary):
                truncated = lesson.encode_chat(messages, max_length=boundary)
                self.assertEqual(truncated.input_ids, (1, 3, 12, 13, 5, 4, 13, 6))
                self.assertEqual(truncated.labels, (-100,) * 6 + (13, 6))
                self.assertEqual((truncated.kept_turns, truncated.dropped_turns), (1, 1))
        self.assertEqual(lesson.encode_chat(messages, max_length=15), full)

    def test_long_system_or_first_answer_is_rejected_not_fabricated(self):
        for messages in ([{"role": "system", "content": "只" * 30}] + pair(), pair(answer="红" * 30)):
            with self.assertRaisesRegex(ValueError, "no complete"):
                lesson.encode_chat(messages, max_length=10)

    def test_empty_answer_still_supervises_stop(self):
        row = lesson.encode_chat(pair(answer=""))
        self.assertEqual(row.input_ids, (1, 3, 12, 13, 5, 4, 6))
        self.assertEqual(row.labels, (-100,) * 6 + (6,))

    def test_content_cannot_inject_role_ids(self):
        row = lesson.encode_chat(pair(user="<assistant>"))
        self.assertEqual(row.input_ids.count(4), 1)
        self.assertTrue(all(value == 7 for value in row.input_ids[2:13]))

    def test_validation_and_no_implicit_role_guessing(self):
        bad = [[], [{"role": "tool", "content": "红"}], pair()[::-1], pair() + pair()[1:],
               [{"role": "user", "content": 1}, pair()[1]],
               [{"role": "user", "content": "红", "hidden": 1}, pair()[1]], pair() + [{"role": "system", "content": "只"}], pair()[:1]]
        for messages in bad:
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                lesson.encode_chat(messages)
        for maximum in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                lesson.encode_chat(pair(), max_length=maximum)
        with self.assertRaises(ValueError):
            lesson.generation_prompt(pair())
        with self.assertRaises(ValueError):
            lesson.collate([])
        with self.assertRaises(ValueError):
            lesson.collate([lesson.encode_chat(pair())], pad_to=7)

    def test_scalar_loss_hand_answer_and_large_offsets(self):
        # Only score positions 0 and 2. Their target probabilities are 1/2 and 1/4.
        logits = [[[math.log(.5), math.log(.5)], [1000., -1000.], [math.log(.75), math.log(.25)], [0., 0.]]]
        labels = [[-100, 0, -100, 1]]
        value, count = lesson.scalar_causal_loss(logits, labels)
        self.assertEqual(count, 2)
        self.assertAlmostEqual(value, 1.0397207708399179)
        shifted = [[[x + 1000 for x in row] for row in logits[0]]]
        self.assertAlmostEqual(lesson.scalar_causal_loss(shifted, labels)[0], value, places=12)
        with self.assertRaises(ValueError):
            lesson.scalar_causal_loss(logits, [[-100] * 4])

    def test_parameter_arithmetic_reference(self):
        self.assertEqual(lesson.lora_parameter_count(3, 2, 1), 5)
        self.assertEqual(lesson.lora_parameter_count(16, 16, 2), 64)
        self.assertEqual(lesson.lora_parameter_count(4096, 4096, 8), 65536)
        self.assertEqual(65536 / (4096 * 4096), 1 / 256)
        for rank in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                lesson.lora_parameter_count(3, 2, rank)

    def test_template_semantic_mutations_rejected(self):
        changes = [
            ("labels.extend([IGNORE] * len(user_ids))", "labels.extend(user_ids)"),
            ("labels.extend([IGNORE] + assistant_ids[1:])", "labels.extend([IGNORE] + assistant_ids[1:-1] + [IGNORE])"),
            ("labels.extend([IGNORE] + assistant_ids[1:])", "labels.extend(assistant_ids)"),
        ]
        for old, new in changes:
            with self.subTest(mutation=new), self.assertRaises(AssertionError):
                token_oracle(mutant(old, new))
        broken = mutant(
            "len(ids) + len(user_ids) + len(assistant_ids) > max_length:",
            "len(ids) + len(user_ids) + len(assistant_ids) >= max_length:")
        with self.assertRaises(ValueError):
            broken.encode_chat(pair(), max_length=8)

    def test_no_torch_cli_is_honest_and_stdlib_works(self):
        # -S disables all installed site-packages even if this test uses a torch venv.
        plain = subprocess.run([sys.executable, "-S", str(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(plain.returncode, 0, plain.stderr)
        self.assertIn("NOT RUN", json.loads(plain.stdout)["training"])
        trained = subprocess.run([sys.executable, "-S", str(SCRIPT), "--train"], capture_output=True, text=True)
        self.assertEqual(trained.returncode, 2)
        self.assertIn("training did NOT run", trained.stderr)


@unittest.skipIf(torch is None, "PyTorch unavailable: real SFT/LoRA checks NOT RUN")
class TorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_loss_matches_independent_log_probability_sum(self):
        logits = torch.tensor([[[.3, -.4, .8], [.1, .2, .3], [-1., 2., .5], [5., 6., 7.]]], dtype=torch.float64)
        labels = torch.tensor([[-100, 2, -100, 1]])
        # Direct exponentials of tiny numbers: no production logsumexp or loss helper.
        expected = (-math.log(math.exp(.8) / sum(math.exp(x) for x in (.3, -.4, .8)))
                    -math.log(math.exp(2.) / sum(math.exp(x) for x in (-1., 2., .5)))) / 2
        self.assertAlmostEqual(float(lesson.masked_causal_loss(logits, labels)), expected, places=14)
        self.assertAlmostEqual(lesson.scalar_causal_loss(logits.tolist(), labels.tolist())[0], expected, places=14)

    def test_shifted_target_gradient_mask_not_input_context_mask(self):
        row = lesson.encode_chat([{"role": "system", "content": "只答颜色"}] + pair())
        batch = lesson._tensor_batch([row])
        logits = torch.zeros(1, len(row.input_ids), len(lesson.TOKENS), dtype=torch.float64, requires_grad=True)
        lesson.masked_causal_loss(logits, batch["labels"]).backward()
        active_positions = torch.nonzero(logits.grad.abs().sum(-1)[0], as_tuple=False).flatten().tolist()
        self.assertEqual(active_positions, [11, 12])  # BEFORE answer and BEFORE EOS.
        self.assertAlmostEqual(float(logits.grad[0, 11, 13]), (1 / 17 - 1) / 2)
        self.assertAlmostEqual(float(logits.grad[0, 12, 6]), (1 / 17 - 1) / 2)
        torch.manual_seed(7)
        model = lesson.TinyChatLM()
        lesson._forward_loss(model, batch).backward()
        self.assertGreater(float(model.embedding.weight.grad[lesson.VOCAB["问"]].abs().sum()), 0)

    def test_all_ignored_batch_rejects_before_any_optimizer_update(self):
        torch.manual_seed(1)
        model = lesson.TinyChatLM()
        batch = lesson._tensor_batch([lesson.encode_chat(pair())])
        batch["labels"].fill_(-100)
        initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        with self.assertRaisesRegex(ValueError, "no supervised"):
            lesson._updates(model, batch, steps=1, lr=0.2)
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(parameter, initial[name]), name)
            self.assertIsNone(parameter.grad)

    def test_padding_does_not_change_real_logits_or_loss(self):
        torch.manual_seed(3)
        model = lesson.TinyChatLM().eval()
        row = lesson.encode_chat(pair())
        plain = lesson._tensor_batch([row])
        padded = {key: torch.tensor(value) for key, value in lesson.collate([row], pad_to=12).items()}
        with torch.no_grad():
            real = model(plain["input_ids"], plain["attention_mask"])
            extra = model(padded["input_ids"], padded["attention_mask"])
        torch.testing.assert_close(real, extra[:, :8], atol=1e-6, rtol=1e-6)
        self.assertAlmostEqual(float(lesson.masked_causal_loss(real, plain["labels"])), float(lesson.masked_causal_loss(extra, padded["labels"])), places=6)
        padded["input_ids"][0, 8:] = lesson.VOCAB["黄"]
        torch.testing.assert_close(extra[:, :8], model(padded["input_ids"], padded["attention_mask"])[:, :8])

    def test_future_tokens_cannot_change_past_predictions(self):
        torch.manual_seed(5)
        model = lesson.TinyChatLM().eval()
        inputs = torch.tensor([[1, 3, 12, 13, 5, 4, 13, 6]])
        changed = inputs.clone()
        changed[0, 6] = 14
        first, second = model(inputs, torch.ones_like(inputs)), model(changed, torch.ones_like(inputs))
        torch.testing.assert_close(first[:, :6], second[:, :6], atol=0, rtol=0)
        self.assertGreater(float((first[:, 6:] - second[:, 6:]).detach().abs().max()), 0)

    def make_matrix_case(self, module=lesson):
        base = torch.nn.Linear(3, 2, bias=False, dtype=torch.float64)
        with torch.no_grad():
            base.weight.copy_(torch.tensor([[1., 2., 3.], [4., 5., 6.]]))
        layer = module.LoRALinear(base, rank=1, alpha=2)
        with torch.no_grad():
            layer.A.copy_(torch.tensor([[1., 0., 2.]]))
            layer.B.copy_(torch.tensor([[3.], [-1.]]))
        return layer

    def matrix_oracle(self, module):
        layer = self.make_matrix_case(module)
        actual = layer(torch.tensor([[2., -1., .5]], dtype=torch.float64))
        torch.testing.assert_close(actual, torch.tensor([[19.5, 0.]], dtype=torch.float64), rtol=0, atol=0)

    def test_lora_matrix_hand_example_and_parameter_count(self):
        self.matrix_oracle(lesson)
        layer = self.make_matrix_case()
        self.assertEqual(sum(p.numel() for p in layer.parameters() if p.requires_grad), 5)
        self.assertEqual(sum(p.numel() for p in layer.parameters() if not p.requires_grad), 6)

    def rank_two_oracle(self, module):
        # alpha=6, rank=2: scale MUST be 3, not alpha alone.
        base = torch.nn.Linear(3, 2, bias=False, dtype=torch.float64)
        with torch.no_grad():
            base.weight.copy_(torch.tensor([[1., 2., 3.], [4., 5., 6.]]))
        layer = module.LoRALinear(base, rank=2, alpha=6)
        with torch.no_grad():
            layer.A.copy_(torch.tensor([[1., 0., 2.], [0., 1., -1.]]))
            layer.B.copy_(torch.tensor([[3., 1.], [-1., 2.]]))
        x = torch.tensor([[2., -1., .5]], dtype=torch.float64, requires_grad=True)
        y = layer(x)
        torch.testing.assert_close(y, torch.tensor([[24., -12.]], dtype=torch.float64), rtol=0, atol=0)
        y.sum().backward()
        torch.testing.assert_close(layer.A.grad, torch.tensor([[12., -6., 3.], [18., -9., 4.5]], dtype=torch.float64), rtol=0, atol=0)
        torch.testing.assert_close(layer.B.grad, torch.tensor([[9., -4.5], [9., -4.5]], dtype=torch.float64), rtol=0, atol=0)
        torch.testing.assert_close(x.grad, torch.tensor([[11., 16., 12.]], dtype=torch.float64), rtol=0, atol=0)
        torch.testing.assert_close(layer.merged().weight, torch.tensor([[10., 5., 18.], [1., 11., -6.]], dtype=torch.float64), rtol=0, atol=0)
        self.assertIsNone(base.weight.grad)

    def test_rank_two_scaling_gradients_and_merge_independent(self):
        self.rank_two_oracle(lesson)

    def test_lora_gradients_match_manual_chain_rule(self):
        layer = self.make_matrix_case()
        x = torch.tensor([[2., -1., .5]], dtype=torch.float64, requires_grad=True)
        layer(x).sum().backward()
        torch.testing.assert_close(layer.A.grad, torch.tensor([[8., -4., 2.]], dtype=torch.float64), rtol=0, atol=0)
        torch.testing.assert_close(layer.B.grad, torch.tensor([[6.], [6.]], dtype=torch.float64), rtol=0, atol=0)
        torch.testing.assert_close(x.grad, torch.tensor([[9., 7., 17.]], dtype=torch.float64), rtol=0, atol=0)
        self.assertIsNone(layer.base.weight.grad)

    def test_zero_b_initialization_first_a_gradient_then_both(self):
        torch.manual_seed(4)
        base = torch.nn.Linear(3, 2, bias=True)
        layer = lesson.LoRALinear(base, rank=1, alpha=2)
        x = torch.tensor([[2., -1., .5]])
        torch.testing.assert_close(layer(x), base(x), rtol=0, atol=0)
        layer(x).sum().backward()
        self.assertTrue(torch.equal(layer.A.grad, torch.zeros_like(layer.A)))
        self.assertGreater(float(layer.B.grad.abs().sum()), 0)
        self.assertIsNone(base.weight.grad)
        self.assertIsNone(base.bias.grad)
        torch.optim.SGD([layer.A, layer.B], lr=.1).step()
        layer.zero_grad(set_to_none=True)
        layer(x).sum().backward()
        self.assertGreater(float(layer.A.grad.abs().sum()), 0)
        self.assertGreater(float(layer.B.grad.abs().sum()), 0)

    def test_merge_is_independent_nonmutating_and_equivalent(self):
        layer = self.make_matrix_case()
        expected_weight = torch.tensor([[7., 2., 15.], [2., 5., 2.]], dtype=torch.float64)
        before = layer.base.weight.detach().clone()
        merged = layer.merged()
        torch.testing.assert_close(merged.weight, expected_weight, rtol=0, atol=0)
        self.assertTrue(torch.equal(layer.base.weight, before))
        torch.testing.assert_close(layer.merged().weight, expected_weight, rtol=0, atol=0)
        inputs = torch.tensor([[2., -1., .5], [-3., 7., 2.]], dtype=torch.float64)
        torch.testing.assert_close(merged(inputs), layer(inputs), rtol=0, atol=0)
        self.assertNotEqual(merged.weight.data_ptr(), layer.base.weight.data_ptr())

    def test_lora_invalid_config(self):
        for rank in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                lesson.LoRALinear(torch.nn.Linear(3, 2), rank=rank)
        for alpha in (0, -1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                lesson.LoRALinear(torch.nn.Linear(3, 2), alpha=alpha)

    def test_actual_sft_and_lora_continuation(self):
        report = lesson.train_demo()
        self.assertEqual(report["batch_shape"], [4, 21])
        self.assertEqual(report["supervised_targets"], 10)
        self.assertEqual(report["base_parameters"], 3760)
        self.assertEqual(report["lora_trainable_parameters"], 128)
        self.assertLess(report["full_loss_after"], report["full_loss_before"])
        self.assertLess(report["lora_loss_after"], report["lora_loss_before"])
        self.assertTrue(report["frozen_base_unchanged"])
        self.assertLess(report["merged_logits_max_error"], 2e-6)
        self.assertFalse(report["heldout_quality_evaluated"])
        self.assertFalse(report["hf_peft_qlora_run"])

    def test_loss_semantic_mutations_rejected_by_scalar_oracle(self):
        logits = torch.tensor([[[.3, -.4, .8], [.1, .2, .3], [-1., 2., .5], [5., 6., 7.]]], dtype=torch.float64)
        labels = torch.tensor([[-100, 2, -100, 1]])
        expected = (-math.log(math.exp(.8) / sum(math.exp(x) for x in (.3, -.4, .8)))
                    -math.log(math.exp(2.) / sum(math.exp(x) for x in (-1., 2., .5)))) / 2
        for old, new in [("logits[:, :-1].reshape", "logits[:, 1:].reshape"),
                         ("return total / count\n", "return total / labels.numel()\n")]:
            with self.subTest(mutation=new), self.assertRaises(AssertionError):
                self.assertAlmostEqual(float(mutant(old, new).masked_causal_loss(logits, labels)), expected, places=12)

    def test_lora_semantic_mutations_rejected_by_matrix_oracle(self):
        # rank=1, alpha=2 makes omitting alpha/r observably wrong.
        broken = mutant("self.scale = alpha / rank", "self.scale = alpha")
        with self.assertRaises(AssertionError):
            self.rank_two_oracle(broken)
        broken = mutant("self.scale = alpha / rank", "self.scale = 1.0")
        with self.assertRaises(AssertionError):
            self.matrix_oracle(broken)
        broken = mutant("result.weight.add_(self.scale * (self.B @ self.A))", "result.weight.add_(self.B @ self.A)")
        layer = self.make_matrix_case(broken)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(layer.merged().weight, torch.tensor([[7., 2., 15.], [2., 5., 2.]], dtype=torch.float64))
        broken = mutant("self.base.requires_grad_(False)", "self.base.requires_grad_(True)")
        layer = self.make_matrix_case(broken)
        with self.assertRaises(AssertionError):
            self.assertEqual(sum(p.numel() for p in layer.parameters() if p.requires_grad), 5)
        broken = mutant("self.B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))", "self.B = nn.Parameter(base.weight.new_ones(base.out_features, rank))")
        layer = broken.LoRALinear(torch.nn.Linear(3, 2, bias=False), rank=1, alpha=2)
        x = torch.tensor([[2., -1., .5]])
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(layer(x), layer.base(x), rtol=0, atol=0)

    def test_causal_mask_mutation_rejected(self):
        broken = mutant("causal = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()", "causal = torch.ones(length, length, dtype=torch.bool, device=x.device)")
        torch.manual_seed(5)
        model = broken.TinyChatLM().eval()
        inputs = torch.tensor([[1, 3, 12, 13, 5, 4, 13, 6]])
        changed = inputs.clone()
        changed[0, 6] = 14
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(model(inputs, torch.ones_like(inputs))[:, :6], model(changed, torch.ones_like(inputs))[:, :6], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
