"""P19: independent packing/loss/attention oracles and behavioral resume mutants."""
import copy
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from torch_test_support import torch_reason, torch_status

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/20_pretraining_pipeline.py"
spec = importlib.util.spec_from_file_location("p19_pipeline", SCRIPT)
p = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = p
spec.loader.exec_module(p)
READY, VERSION, PRESENT = torch_status()
if READY:
    import torch


def scalar_attention_oracle(q, k, v, segments):
    """Python dot products and explicit document-prefix loops, no production mask."""
    result = []
    for i, query in enumerate(q):
        visible = [j for j in range(i + 1) if segments[i] >= 0 and segments[j] == segments[i]]
        if not visible:
            result.append([0.] * len(v[0]))
            continue
        scores = [sum(a * b for a, b in zip(query, k[j])) / math.sqrt(len(query)) for j in visible]
        weights = [math.exp(score - max(scores)) for score in scores]
        result.append([sum(weight * v[j][d] for weight, j in zip(weights, visible)) / sum(weights)
                       for d in range(len(v[0]))])
    return result


class PipelineStandardLibraryTests(unittest.TestCase):
    def test_clean_dedup_before_split_and_no_contamination(self):
        data = p.prepare_data()
        self.assertEqual(data["cleaning"], {"raw": 12, "empty": ["blank"],
                         "duplicates": [["duplicate", "d00"]], "kept": 10})
        self.assertEqual([len(data["splits"][s]) for s in ("train", "validation", "test")], [6, 2, 2])
        p.assert_no_contamination(data["splits"])
        self.assertEqual(p.prepare_data()["data_version"], data["data_version"])
        bad = copy.deepcopy(data["splits"])
        bad["test"].append(p.Document("new_id", " " + bad["train"][0].text + "\n"))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            p.assert_no_contamination(bad)
        with self.assertRaisesRegex(ValueError, "deduplicate"):
            p.split_documents([p.Document(str(i), "春") for i in range(5)])

    def test_exact_scope_unknown_token_and_metadata(self):
        docs, report = p.clean_deduplicate([p.Document("a", "春风"), p.Document("b", "春风月")])
        self.assertEqual(len(docs), 2)  # near duplicates are deliberately NOT detected
        self.assertEqual(report["duplicates"], [])
        self.assertEqual(p.encode("春Z"), [4, p.UNK])
        with self.assertRaisesRegex(ValueError, "generated"):
            p.clean_deduplicate([p.Document("x", "春", provenance="unverified web")])
        with self.assertRaisesRegex(ValueError, "unique"):
            p.clean_deduplicate([p.Document("a", "春"), p.Document("a", "月")])
        changed = p.synthetic_documents()
        changed[0] = p.Document("d00", "春风", source="generated/changed-provenance")
        self.assertNotEqual(p.prepare_data(changed)["data_version"], p.prepare_data()["data_version"])

    def test_hand_packing_has_independent_shift_boundary_positions_and_mask(self):
        row = p.pack_documents([p.Document("a", "春风"), p.Document("b", "月")], 8)[0]
        # Literal oracle does not call encoder, shift, or mask helpers.
        self.assertEqual(row["inputs"], [1, 4, 5, 2, 1, 6, 2, 0])
        self.assertEqual(row["labels"], [4, 5, 2, -100, 6, 2, -100, -100])
        self.assertEqual(row["positions"], [0, 1, 2, 3, 0, 1, 2, 0])
        self.assertEqual(row["segments"], [0, 0, 0, 0, 1, 1, 1, -1])
        expected_visible = [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3], [4], [4, 5], [4, 5, 6], []]
        self.assertEqual([[j for j, yes in enumerate(mask) if yes] for mask in row["allowed"]], expected_visible)
        # Valid baseline first; real global-shift mutant predicts document B at A's EOS.
        mutant = row["inputs"][1:] + [p.IGNORE]
        self.assertEqual(mutant[3], p.BOS)
        self.assertNotEqual(mutant, row["labels"])
        table = p.hand_table()
        self.assertNotEqual(p.scalar_loss(table["logits"], mutant), table["loss"])

    def test_scalar_full_vocabulary_loss_and_unequal_document_weights(self):
        table = p.hand_table()
        self.assertAlmostEqual(table["loss"]["loss_sum"], 9 * math.log(2), places=13)
        self.assertEqual(table["loss"]["count"], 5)
        self.assertAlmostEqual(table["loss"]["mean"], 9 * math.log(2) / 5, places=13)
        first, second = 4 * math.log(2) / 3, 5 * math.log(2) / 2
        self.assertAlmostEqual((3 * first + 2 * second) / 5, table["loss"]["mean"], places=13)
        self.assertNotAlmostEqual((first + second) / 2, table["loss"]["mean"], places=6)
        # Wrong unshifted labels turn a copy classifier into apparent good training.
        x, correct = [4, 5, 6], [5, 6, 2]
        scores = [[12. if c == token else 0. for c in range(p.VOCAB_SIZE)] for token in x]
        self.assertGreater(p.scalar_loss(scores, correct)["mean"], 12)
        self.assertLess(p.scalar_loss(scores, x)["mean"], .001)
        self.assertEqual(p.scalar_loss([[1., 2.]], [-100]), dict(loss_sum=0., count=0, mean=None))

    def test_pack_count_conservation_and_explicit_long_document_rejection(self):
        data = p.prepare_data()
        summary = p.data_summary(data)
        for name, docs in data["splits"].items():
            count = sum(y != -100 for row in data["rows"][name] for y in row["labels"])
            self.assertEqual(count, sum(len(doc.text) + 1 for doc in docs))
            self.assertEqual(count, summary["splits"][name]["valid_targets"])
            self.assertAlmostEqual(sum(summary["splits"][name]["source_token_fractions"].values()), 1.)
        with self.assertRaisesRegex(ValueError, "chunking policy"):
            p.pack_documents([p.Document("long", "春风月明")], 5)
        for invalid in [2, True, 3.5]:
            with self.assertRaises(ValueError):
                p.pack_documents([], invalid)
        pad = p.pack_documents([], 5)[0]
        self.assertEqual(pad["inputs"], [0] * 5)
        self.assertTrue(all(y == -100 for y in pad["labels"]))
        self.assertFalse(any(any(row) for row in pad["allowed"]))

    def test_exercise_answers_and_mixture_accounting(self):
        row = p.pack_documents([p.Document("a", "春风"), p.Document("b", "月明")], 8)[0]
        self.assertNotIn(p.PAD, row["inputs"])
        self.assertEqual(sum(sid >= 0 for sid in row["segments"]), 8)
        self.assertEqual(sum(y != p.IGNORE for y in row["labels"]), 6)
        # Independent chosen probabilities make actual CE equal 1 and 3 nats.
        scores = [[math.log(math.exp(-loss) / (1 - math.exp(-loss))), 0.]
                  for loss in [1.] * 2 + [3.] * 8]
        self.assertAlmostEqual(p.scalar_loss(scores, [0] * 10)["mean"], 2.6)
        before = (.5 * 2) / (.5 * 2 + .5 * 8)
        after = ((2 / 3) * 2) / ((2 / 3) * 2 + (1 / 3) * 8)
        self.assertAlmostEqual(before, 1 / 5)
        self.assertAlmostEqual(after, 1 / 3)
        self.assertNotAlmostEqual(after, before * 2)

    def test_missing_torch_is_graceful_and_data_only_never_imports_it(self):
        blocker = '''
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "torch":
        raise ModuleNotFoundError("No module named 'torch'", name="torch")
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
path = sys.argv[1]
sys.argv = [path] + sys.argv[2:]
runpy.run_path(path, run_name="__main__")
'''
        for arguments, status in [(["--data-only"], 0), ([], 2)]:
            result = subprocess.run([sys.executable, "-c", blocker, str(SCRIPT), *arguments], capture_output=True, text=True)
            self.assertEqual(result.returncode, status, result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertIn('"kept": 10', result.stdout)
            if status:
                self.assertIn("--data-only", result.stderr)


@unittest.skipUnless(READY, torch_reason("P19 tensor/resume checks", VERSION, PRESENT))
class PipelineTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        cls.old_determinism = torch.are_deterministic_algorithms_enabled()
        cls.old_warn = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)
        torch.use_deterministic_algorithms(cls.old_determinism, warn_only=cls.old_warn)

    def test_attention_matches_independent_scalar_oracle_and_kills_boundary_mutant(self):
        q = [[1., 0.], [0., 1.], [2., -1.], [.5, 2.], [1., 1.]]
        k = [[.2, 1.], [2., -1.], [1., .5], [-1., 2.], [.1, .3]]
        v = [[1., 3.], [2., -1.], [8., -4.], [4., 2.], [9., 9.]]
        segments = [0, 0, 1, 1, -1]
        tensors = [torch.tensor([item], dtype=torch.float64) for item in (q, k, v)]
        mask = torch.tensor([p.attention_mask(segments)])
        result = p.tensor_attention(*tensors, mask)
        expected = torch.tensor([scalar_attention_oracle(q, k, v, segments)], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-13, atol=1e-13)
        self.assertTrue(torch.equal(result[0, 4], torch.zeros(2, dtype=torch.float64)))
        # Actual bad computation: causal-only attention permits cross-document reads.
        broken = torch.tril(torch.ones(1, 5, 5, dtype=torch.bool))
        bad = p.tensor_attention(*tensors, broken)
        self.assertGreater(float((bad[:, 2:4] - expected[:, 2:4]).abs().max()), 1.)
        # Future K/V changes must not affect earlier positions.
        tensors[1][:, 3:] += 100
        tensors[2][:, 3:] -= 50
        actual = p.tensor_attention(*tensors, mask)
        torch.testing.assert_close(actual[:, :3], expected[:, :3], rtol=0, atol=0)

    def test_packed_model_equals_documents_alone_and_isolates_both_directions(self):
        torch.manual_seed(41)
        model = p.make_model(10).eval()
        a, b = p.Document("a", "春风"), p.Document("b", "月")
        packed = p.pack_documents([a, b], 10)[0]
        with torch.no_grad():
            together = model(**p.tensor_row(packed)[0])
            alone_a = model(**p.tensor_row(p.pack_documents([a], 10)[0])[0])
            alone_b = model(**p.tensor_row(p.pack_documents([b], 10)[0])[0])
            torch.testing.assert_close(together[:, :4], alone_a[:, :4], rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(together[:, 4:7], alone_b[:, :3], rtol=1e-6, atol=1e-6)
            changed = copy.deepcopy(packed)
            changed["inputs"][1:3] = [10, 11]
            output = model(**p.tensor_row(changed)[0])
            torch.testing.assert_close(output[:, 4:7], together[:, 4:7], rtol=0, atol=0)
            changed = copy.deepcopy(packed)
            changed["inputs"][5] = 12
            output = model(**p.tensor_row(changed)[0])
            torch.testing.assert_close(output[:, :4], together[:, :4], rtol=0, atol=0)
            # Same changed A, but remove document boundary in a real forward pass.
            broken = copy.deepcopy(packed)
            broken["allowed"] = [[j <= i and j < 7 for j in range(10)] for i in range(10)]
            bad_before = model(**p.tensor_row(broken)[0])
            broken["inputs"][1:3] = [10, 11]
            bad_after = model(**p.tensor_row(broken)[0])
            self.assertGreater(float((bad_before[:, 4:7] - bad_after[:, 4:7]).abs().max()), .001)

    def test_tensor_loss_matches_scalar_oracle_and_masks_gradients(self):
        table = p.hand_table()
        logits = torch.tensor([table["logits"]], dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([table["row"]["labels"]])
        total, count = p.tensor_loss(logits, labels)
        self.assertEqual(count, 5)
        self.assertAlmostEqual(float(total.detach()), 9 * math.log(2), places=13)
        (total / count).backward()
        self.assertTrue(torch.equal(logits.grad[0, [3, 6, 7]], torch.zeros(3, p.VOCAB_SIZE, dtype=torch.float64)))
        probabilities = [.5, .25, .5, .125, .25]
        for i, probability in zip([0, 1, 2, 4, 5], probabilities):
            self.assertAlmostEqual(float(logits.grad[0, i, labels[0, i]]), (probability - 1) / 5, places=13)
        # Ignored rows' arbitrary finite scores cannot affect loss.
        altered = logits.detach().clone()
        altered[:, [3, 6, 7]] = 1000
        self.assertEqual(float(p.tensor_loss(altered, labels)[0]), float(total.detach()))

    def test_all_ignored_safe_attention_loss_and_optimizer_skip(self):
        data = p.prepare_data()
        trainer = p.Trainer(data)
        trainer.run_steps(2)  # Populate AdamW momentum before testing a skipped row.
        before = trainer.state_dict()
        row = p.pack_documents([], 12)[0]
        inputs, labels = p.tensor_row(row)
        logits = trainer.model(**inputs)
        self.assertTrue(bool(torch.isfinite(logits).all()))
        self.assertTrue(torch.equal(logits, torch.zeros_like(logits)))
        total, count = p.tensor_loss(logits, labels)
        self.assertEqual((float(total.detach()), count), (0., 0))
        total.backward()
        self.assertTrue(all(param.grad is None or bool(torch.isfinite(param.grad).all()) for param in trainer.model.parameters()))
        self.assertIsNone(trainer.train_row(row))
        self.assertTrue(p.exact_equal(before, trainer.state_dict()))
        # A zero-gradient AdamW step still changes learned parameters via moments/decay.
        broken = p.Trainer(data)
        broken.load_state_dict(before)
        broken.optimizer.zero_grad(set_to_none=True)
        loss, _ = p.tensor_loss(broken.model(**inputs), labels)
        loss.backward()
        broken.optimizer.step()
        self.assertFalse(p.exact_equal(before["model"], broken.state_dict()["model"]))

    def test_evaluate_weights_unequal_target_counts_not_row_means(self):
        data = p.prepare_data()
        # Two differently sized documents as separate blocks: 2 versus 5 targets.
        data["rows"]["validation"] = [p.pack_documents([doc], 12)[0] for doc in
                                       [p.Document("a", "光"), p.Document("b", "春风月明")]]
        trainer = p.Trainer(data)
        trainer.model.eval()
        numerators, counts = [], []
        with torch.no_grad():
            for row in data["rows"]["validation"]:
                inputs, _ = p.tensor_row(row)
                logits = trainer.model(**inputs)[0].tolist()
                oracle = p.scalar_loss(logits, row["labels"])
                numerators.append(oracle["loss_sum"])
                counts.append(oracle["count"])
        self.assertEqual(counts, [2, 5])
        expected = math.fsum(numerators) / sum(counts)
        trainer.model.train()
        old = trainer.state_dict()
        observations = []
        handle = trainer.model.register_forward_pre_hook(lambda module, args: observations.append(
                    (module.training, torch.is_grad_enabled())))
        try:
            actual = trainer.evaluate()["mean"]
        finally:
            handle.remove()
        self.assertAlmostEqual(actual, expected, places=6)
        self.assertEqual(observations, [(False, False), (False, False)])
        self.assertTrue(trainer.model.training)
        self.assertTrue(p.exact_equal(old, trainer.state_dict()))
        mutant_row_mean = sum(a / b for a, b in zip(numerators, counts)) / 2
        self.assertGreater(abs(actual - mutant_row_mean), .01)

    def test_exact_resume_file_roundtrip_all_state_and_snapshot_no_alias(self):
        data = p.prepare_data()
        full = p.Trainer(data)
        full.run_steps(12)
        partial = p.Trainer(data)
        partial.run_steps(3)
        snapshot = partial.state_dict()
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "resume.pt"
            partial.save(filename)
            partial.run_steps(1)
            self.assertEqual(snapshot["step"], 3)
            self.assertFalse(p.exact_equal(snapshot["model"], partial.state_dict()["model"]))
            resumed = p.Trainer(data)
            resumed.load(filename)
            self.assertTrue(p.exact_equal(snapshot, resumed.state_dict()))
            resumed.run_steps(9)
        self.assertTrue(p.exact_equal(full.state_dict(), resumed.state_dict()))
        self.assertEqual(full.history, resumed.history)
        self.assertEqual(full.tokens, 84)
        self.assertEqual(full.evaluate(), resumed.evaluate())

    def test_resume_omission_mutants_change_actual_losses_and_final_weights(self):
        data = p.prepare_data()
        full = p.Trainer(data)
        full.run_steps(12)
        partial = p.Trainer(data)
        partial.run_steps(5)  # After a shuffle, mid-epoch: both RNG/cursor omissions matter.
        saved = partial.state_dict()
        baseline = p.Trainer(data)
        baseline.load_state_dict(saved)
        baseline.run_steps(7)
        self.assertTrue(p.exact_equal(full.state_dict(), baseline.state_dict()))
        for omission in ("optimizer", "torch_rng", "shuffle_rng", "cursor"):
            with self.subTest(omission=omission):
                mutant = p.Trainer(data)
                fresh = mutant.state_dict()
                mutant.load_state_dict(saved)
                # Simulate omitting ONE load operation, not invalid metadata rejection.
                if omission == "optimizer":
                    mutant.optimizer.load_state_dict(fresh["optimizer"])
                elif omission == "torch_rng":
                    mutant.torch_rng = fresh["torch_rng"]
                elif omission == "shuffle_rng":
                    mutant.shuffle_rng.setstate(fresh["shuffle_rng"])
                else:
                    mutant.cursor = fresh["cursor"]
                mutant.run_steps(7)
                self.assertFalse(p.exact_equal(full.state_dict()["model"], mutant.state_dict()["model"]))
                self.assertNotEqual([r["loss"] for r in full.history[5:]], [r["loss"] for r in mutant.history[5:]])

    def test_checkpoint_rejects_changed_data_schedule_version_or_incomplete_state(self):
        data = p.prepare_data()
        trained = p.Trainer(data)
        trained.run_steps(2)
        saved = trained.state_dict()
        for key, replacement in [("schema", 999), ("torch_version", "different"), ("data_version", "different")]:
            bad = copy.deepcopy(saved)
            bad[key] = replacement
            with self.assertRaisesRegex(ValueError, key):
                p.Trainer(data).load_state_dict(bad)
        with self.assertRaisesRegex(ValueError, "config"):
            p.Trainer(data, horizon=13).load_state_dict(saved)
        bad = copy.deepcopy(saved)
        del bad["optimizer"]
        with self.assertRaisesRegex(ValueError, "schema fields"):
            p.Trainer(data).load_state_dict(bad)
        bad = copy.deepcopy(saved)
        bad["cursor"] = 99
        with self.assertRaisesRegex(ValueError, "cursor"):
            p.Trainer(data).load_state_dict(bad)

    def test_learning_rate_and_reproducibility_keep_callers_rng(self):
        torch.manual_seed(99)
        before_rng = torch.get_rng_state().clone()
        trainer = p.Trainer(p.prepare_data())
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        trainer.run_steps(12)
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        self.assertEqual([r["lr"] for r in trainer.history[:3]], [.0015, .003, .003])
        self.assertAlmostEqual(trainer.history[-1]["lr"], .0003)
        with self.assertRaises(ValueError):
            trainer.run_steps(1)


if __name__ == "__main__":
    unittest.main()
