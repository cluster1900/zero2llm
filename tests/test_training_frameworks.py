"""P20: standard-library ledger + optional pinned CPU Transformers contracts.

SKIP means optional framework coverage did not run, never a passing HF test.
Mutation checks deliberately break the lesson and verify an independent oracle
or smoke assertion rejects it. No network or pretrained weights are required.
"""
import importlib.util
from importlib import metadata
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "code/21_training_frameworks.py"
spec = importlib.util.spec_from_file_location("framework_lesson", SCRIPT)
lesson = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = lesson
spec.loader.exec_module(lesson)
READY, REASON, VERSIONS = lesson.framework_status()


def mutated(old, new):
    source = SCRIPT.read_text(encoding="utf-8")
    if source.count(old) != 1:
        raise AssertionError("mutation must match exactly one site")
    module = types.ModuleType("framework_mutant")
    module.__file__ = str(SCRIPT)
    sys.modules[module.__name__] = module
    exec(compile(source.replace(old, new), str(SCRIPT), "exec"), module.__dict__)
    return module


class LedgerTests(unittest.TestCase):
    def assert_hand_ledger(self, module):
        # Byte counts hand-calculated independently of lesson helpers.
        expected = [
            [(20, 20, 40, 40, 40)] * 4,
            [(20, 20, 12, 12, 12)] * 2 + [(20, 20, 8, 8, 8)] * 2,
            [(20, 6, 12, 12, 12)] * 2 + [(20, 4, 8, 8, 8)] * 2,
            [(6, 6, 12, 12, 12)] * 2 + [(4, 4, 8, 8, 8)] * 2,
        ]
        for stage, rows in enumerate(expected):
            result = module.zero_ledger(module.MemorySpec(), stage)
            self.assertEqual([(r.parameters, r.gradients, r.first_moment,
                               r.second_moment, r.master_weights) for r in result], rows)
            self.assertEqual([r.total for r in result], [sum(row) for row in rows])

    def test_hand_table_every_state_every_rank(self):
        self.assert_hand_ledger(lesson)
        lesson.ledger_self_test()

    def test_partition_conservation_uneven_and_empty_ranks(self):
        for parameters in (0, 1, 2, 7, 10, 31, 100):
            for world_size in (1, 2, 3, 4, 11):
                sizes = lesson.shard_sizes(parameters, world_size)
                # Independent ownership assignment by round-robin parameter IDs.
                expected = tuple(sum(i % world_size == rank for i in range(parameters))
                                 for rank in range(world_size))
                self.assertEqual(sizes, expected)
                self.assertEqual(sum(sizes), parameters)
                self.assertEqual(max(sizes), (parameters + world_size - 1) // world_size)
                self.assertLessEqual(max(sizes) - min(sizes), 1)
        self.assertEqual(lesson.shard_sizes(2, 4), (1, 1, 0, 0))

    def test_aggregate_conservation_and_one_rank_equivalence(self):
        for count in (1, 10, 99):
            for ranks in (1, 3, 8):
                spec = lesson.MemorySpec(count, ranks)
                all_rows = [lesson.zero_ledger(spec, stage) for stage in range(4)]
                self.assertEqual([sum(r.total for r in rows) for rows in all_rows],
                                 [16*count*ranks, 4*count*ranks + 12*count,
                                  2*count*ranks + 14*count, 16*count])
                peaks = [max(r.total for r in rows) for rows in all_rows]
                self.assertEqual(peaks, sorted(peaks, reverse=True))
                if ranks == 1:
                    self.assertEqual(peaks, [16*count] * 4)

    def test_gradient_dtype_is_not_optimizer_dtype(self):
        spec = lesson.MemorySpec(10, 4, parameter_bytes=2, gradient_bytes=4,
                                 moment_bytes=4, master_bytes=0)
        self.assertEqual([r.total for r in lesson.zero_ledger(spec, 2)], [56, 56, 44, 44])
        self.assertEqual(lesson.zero_ledger(spec, 1)[0].optimizer, 24)
        self.assertEqual(lesson.zero_ledger(spec, 1)[0].gradients, 40)

    def test_units_use_bytes_decimal_gb_binary_gib(self):
        self.assertEqual(lesson.byte_units(1_000_000_000)["GB"], 1)
        self.assertEqual(lesson.byte_units(1_073_741_824)["GiB"], 1)
        self.assertAlmostEqual(lesson.byte_units(1_000_000_000)["GiB"], .9313225746154785)
        self.assertEqual(lesson.byte_units(0), {"bytes": 0, "GB": 0, "GiB": 0})

    def test_invalid_inputs(self):
        for n, d in ((-1, 4), (10, 0), (True, 2), (10, 1.5)):
            with self.assertRaises(ValueError):
                lesson.shard_sizes(n, d)
        for kwargs in ({"parameters": 0}, {"world_size": False}, {"gradient_bytes": 0},
                       {"moment_bytes": 1.5}, {"master_bytes": -1}):
            with self.assertRaises(ValueError):
                lesson.MemorySpec(**kwargs)
        for stage in (-1, 4, True, 1.0):
            with self.assertRaises(ValueError):
                lesson.zero_ledger(lesson.MemorySpec(), stage)
        with self.assertRaises(ValueError):
            lesson.byte_units(-1)

    def test_semantic_ledger_mutations_are_detected(self):
        changes = [
            ("quotient + (rank < remainder)", "quotient"),
            ("g_count = owned if stage >= 2", "g_count = owned if stage >= 1"),
            ("o_count * spec.master_bytes,", "0,"),
            ("p_count = owned if stage >= 3", "p_count = owned if stage >= 2"),
        ]
        for old, new in changes:
            with self.subTest(mutation=old):
                mutant = mutated(old, new)
                with self.assertRaises(AssertionError):
                    self.assert_hand_ledger(mutant)

    def test_metadata_gate_absent_and_wrong_version(self):
        real_version = metadata.version
        def missing(name):
            if name == "transformers":
                raise metadata.PackageNotFoundError(name)
            return real_version(name)
        with mock.patch.object(lesson.metadata, "version", side_effect=missing):
            ready, reason, _ = lesson.framework_status()
            self.assertFalse(ready)
            self.assertIn("transformers=not installed", reason)
        with mock.patch.object(lesson.metadata, "version", return_value="99.0.0"):
            ready, reason, _ = lesson.framework_status()
            self.assertFalse(ready)
            self.assertIn("skipped is not passed", reason)
            with self.assertRaisesRegex(RuntimeError, "SKIP"):
                lesson.hf_smoke()

    def test_cli_ledger_without_any_framework_import(self):
        program = '''
import builtins, runpy, sys
old_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'transformers', 'tokenizers', 'accelerate'):
        raise AssertionError('standard-library mode imported a framework')
    return old_import(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = [sys.argv[1]]
runpy.run_path(sys.argv[0], run_name='__main__')
'''
        result = subprocess.run([sys.executable, "-c", program, str(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ledger self-test: PASS", result.stdout)
        self.assertIn("HF smoke: NOT RUN", result.stdout)

    def test_missing_optional_cli_exit_two_not_false_pass(self):
        program = '''
import importlib.metadata, runpy, sys
old_version = importlib.metadata.version
def missing(name):
    if name == 'transformers':
        raise importlib.metadata.PackageNotFoundError(name)
    return old_version(name)
importlib.metadata.version = missing
sys.argv = [sys.argv[1], '--hf']
runpy.run_path(sys.argv[0], run_name='__main__')
'''
        result = subprocess.run([sys.executable, "-c", program, str(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("HF smoke: SKIP", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertNotIn('"status": "PASS"', result.stdout)


@unittest.skipUnless(READY, "P20 optional HF tests: " + REASON)
class FrameworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.tokenizer, self.model, self.batch = lesson.make_fixture()
        self.model.eval()

    def scalar_ce(self, logits, labels):
        # Python scalar loop: no lesson shift helper, torch CE or log_softmax.
        losses = []
        for row in range(labels.shape[0]):
            for target_position in range(1, labels.shape[1]):
                target = labels[row, target_position].item()
                if target == -100:
                    continue
                values = logits[row, target_position - 1].detach().double().tolist()
                peak = max(values)
                log_z = peak + math.log(sum(math.exp(value - peak) for value in values))
                losses.append(log_z - values[target])
        return sum(losses) / len(losses)

    def test_local_fixture_ids_masks_and_config_shapes(self):
        self.assertEqual(self.batch["input_ids"].tolist(),
                         [[2, 4, 5, 6, 11, 3], [2, 7, 8, 10, 3, 0]])
        self.assertEqual(self.batch["labels"].tolist(),
                         [[-100, 4, 5, 6, 11, 3], [-100, 7, 8, 10, 3, -100]])
        self.assertEqual(self.batch["attention_mask"].tolist(), [[1]*6, [1]*5+[0]])
        self.assertEqual(self.tokenizer("陌生词")["input_ids"], [2, 1, 3])
        self.assertEqual(self.model.config.vocab_size, 12)
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 2704)
        with self.torch.no_grad():
            output = self.model(**self.batch)
        self.assertEqual(tuple(output.logits.shape), (2, 6, 12))
        self.assertAlmostEqual(output.loss.item(), self.scalar_ce(output.logits, self.batch["labels"]), places=6)

    def test_independent_shifted_ce_and_gradient(self):
        torch = self.torch
        logits = torch.tensor([[[0., 1., 2.], [8., -2., 0.], [1., 0., -1.],
                               [-1., 1., 0.]]], dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([[-100, 2, -100, 1]])
        loss = lesson.shifted_loss_reference(logits, labels)
        self.assertAlmostEqual(loss.item(), self.scalar_ce(logits, labels), places=13)
        loss.backward()
        # Only predictions at t=0 and t=2 have next-token supervision.
        self.assertEqual(logits.grad[0, 1].abs().sum().item(), 0)
        self.assertEqual(logits.grad[0, 3].abs().sum().item(), 0)
        self.assertGreater(logits.grad[0, 0].abs().sum().item(), 0)
        self.assertGreater(logits.grad[0, 2].abs().sum().item(), 0)
        mutant = mutated("targets = labels[:, 1:]", "targets = labels[:, :-1]")
        self.assertGreater(abs(mutant.shifted_loss_reference(logits, labels).item() - loss.item()), .01)

    def test_reference_rejects_all_ignored_bad_shapes_and_bad_targets(self):
        torch = self.torch
        logits = torch.zeros(1, 3, 4)
        for labels in (torch.full((1, 3), -100), torch.zeros(1, 3),
                       torch.tensor([[-100, 9, 1]]), torch.zeros(1, 2, dtype=torch.long)):
            with self.assertRaises(ValueError):
                lesson.shifted_loss_reference(logits, labels)

    def test_causality_and_deterministic_initialization(self):
        torch = self.torch
        _, repeated_model, _ = lesson.make_fixture()
        for name, value in self.model.named_parameters():
            torch.testing.assert_close(value, dict(repeated_model.named_parameters())[name], atol=0, rtol=0)
        changed = self.batch["input_ids"].clone()
        changed[:, 3:] = 9
        with torch.no_grad():
            original = self.model(input_ids=self.batch["input_ids"], attention_mask=self.batch["attention_mask"]).logits
            other = self.model(input_ids=changed, attention_mask=self.batch["attention_mask"]).logits
        torch.testing.assert_close(original[:, :3], other[:, :3], atol=0, rtol=0)

    def test_real_trainer_updates_reference_and_local_checkpoint(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            result = lesson.hf_smoke(directory, steps=3)
            self.assertEqual(result["steps"], 3)
            self.assertEqual(result["shape"], [2, 6, 12])
            self.assertLess(result["loss_after"], result["loss_before"])
            self.assertGreater(result["parameter_max_delta"], 1e-7)
            self.assertLess(result["reference_parameter_max_error"], 2e-6)
            self.assertEqual(result["reload_logits_max_error"], 0)
            self.assertFalse(result["resume_tested"])
            export = Path(directory) / "export"
            self.assertEqual(json.loads((export / "experiment.json").read_text())["versions"], VERSIONS)
            self.assertEqual(json.loads((export / "config.json").read_text())["n_embd"], 16)
            self.assertTrue((export / "tokenizer.json").is_file())
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                lesson.hf_smoke(directory, steps=1)

    def test_unshifted_count_mutation_is_detected(self):
        mutant = mutated('    batch["labels"][:, 0] = -100', '    pass  # bad: counts first BOS as a target')
        with self.assertRaises(AssertionError):
            mutant.hf_smoke(steps=1)

    def test_missing_optimizer_step_is_detected(self):
        with mock.patch.object(self.torch.optim.SGD, "step", return_value=None):
            with self.assertRaisesRegex(AssertionError, "did not update"):
                lesson.hf_smoke(steps=1)

    def test_corrupted_reload_is_detected(self):
        model_class = type(self.model)
        load = model_class.from_pretrained
        def corrupted(*args, **kwargs):
            model = load(*args, **kwargs)
            with self.torch.no_grad():
                model.transformer.wte.weight[4, 0] += .2
            return model
        with mock.patch.object(model_class, "from_pretrained", side_effect=corrupted):
            with self.assertRaises(AssertionError):
                lesson.hf_smoke(steps=1)


if __name__ == "__main__":
    unittest.main()
