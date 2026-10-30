"""P27 chart renderer: geometry binding + gold JSON regeneration contract."""
from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "plot_p27_healthcheck.py"
GOLD_MTP = ROOT / "docs" / "gold" / "p27_healthcheck_mtp.json"
GOLD = ROOT / "docs" / "gold" / "p27_healthcheck.json"
OUT_DIR = ROOT / "assets" / "diagrams" / "p27"
CHAPTER = ROOT / "chapters" / "09a_minigpt_healthcheck.md"


def load_renderer():
    spec = importlib.util.spec_from_file_location("p27_renderer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class P27ChartRendererTests(unittest.TestCase):
    def _fixture(self):
        return {
            "stages": [
                {"stage": "S0", "causal_error": 0.0},
                {"stage": "S1", "causal_error": 0.1, "cache_error": 0.2},
                {"stage": "S2", "causal_error": 0.05, "cache_error": 0.03},
                {"stage": "S3", "causal_error": 0.04, "cache_error": 0.02},
                {"stage": "S4", "causal_error": 0.03, "cache_error": 0.01},
                {"stage": "S5", "causal_error": 0.02, "cache_error": 0.005},
                {"stage": "S6", "causal_error": 0.01, "cache_error": 0.004},
            ],
            "negatives": [
                {"name": "wrong_mask", "caught": True, "max_logit_error": 0.3, "before_error": 1e-7},
                {"name": "wrong_position", "caught": True, "max_logit_error": 0.2, "before_error": 0.0},
                {"name": "cache_pollution", "caught": True, "max_logit_error": 0.4, "before_error": 2e-7},
            ],
            "timing": {
                "milliseconds": {
                    "full_first_ms": {"min": 9.0, "median": 10.0, "max": 12.0},
                    "cache_first_ms": {"min": 4.0, "median": 5.0, "max": 6.0},
                    "full_later_ms": {"min": 2.0, "median": 3.0, "max": 4.0},
                    "cache_later_ms": {"min": 1.0, "median": 1.5, "max": 2.0},
                }
            },
        }

    def test_renderer_binds_geometry_to_fixture_values(self):
        renderer = load_renderer()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "plots"
            paths = renderer.render(self._fixture(), out)
            self.assertEqual(
                [p.name for p in paths],
                ["cache_causal.svg", "negative_controls.svg", "cpu_timing.svg"],
            )
            cache = (out / "cache_causal.svg").read_text(encoding="utf-8")
            self.assertIn('data-stage="S6"', cache)
            self.assertIn('data-value="0.2"', cache)
            self.assertGreaterEqual(cache.count("<rect "), 14)
            self.assertIn("<line ", cache)
            negative = (out / "negative_controls.svg").read_text(encoding="utf-8")
            self.assertEqual(negative.count('data-phase="before"'), 3)
            self.assertEqual(negative.count('data-phase="after"'), 3)
            for row in self._fixture()["negatives"]:
                name = row["name"]
                self.assertIn(
                    f'data-control="{name}" data-phase="before" data-value="{row["before_error"]:.12g}"',
                    negative,
                )
                self.assertIn(
                    f'data-control="{name}" data-phase="after" data-value="{row["max_logit_error"]:.12g}"',
                    negative,
                )
            timing = (out / "cpu_timing.svg").read_text(encoding="utf-8")
            self.assertEqual(timing.count("data-median="), 4)
            self.assertIn('data-min="9"', timing)

    def test_missing_section_fails_before_writing_files(self):
        renderer = load_renderer()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "plots"
            broken = self._fixture()
            del broken["timing"]
            with self.assertRaises(ValueError):
                renderer.render(broken, out)
            self.assertFalse(out.exists())

    def test_missing_before_error_fails_before_writing_files(self):
        renderer = load_renderer()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "plots"
            broken = self._fixture()
            del broken["negatives"][0]["before_error"]
            with self.assertRaises(ValueError):
                renderer.render(broken, out)
            self.assertFalse(out.exists())
            self.assertFalse(any(Path(td).rglob("*.svg")))

    def test_uncaught_negative_fails(self):
        renderer = load_renderer()
        broken = self._fixture()
        broken["negatives"][0]["caught"] = False
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                renderer.render(broken, Path(td) / "plots")

    def test_renderer_source_mutation_breaks_value_binding(self):
        """Mutate the renderer so bars no longer bind real values; catch it."""
        source = SCRIPT.read_text(encoding="utf-8")
        mutated = source.replace('data-value="{value:.12g}"', 'data-value="0"', 1)
        self.assertNotEqual(source, mutated)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mutated_script = root / "renderer.py"
            mutated_script.write_text(mutated, encoding="utf-8")
            report = root / "report.json"
            report.write_text(json.dumps(self._fixture()), encoding="utf-8")
            out = root / "plots"
            result = subprocess.run(
                [sys.executable, str(mutated_script), str(report), "--out-dir", str(out)],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            rendered = (out / "cache_causal.svg").read_text(encoding="utf-8")
            self.assertNotIn('data-value="0.2"', rendered)


class P27GoldArtifactsTests(unittest.TestCase):
    def test_gold_mtp_report_has_s0_s6_and_caught_negatives(self):
        self.assertTrue(GOLD_MTP.is_file(), "missing gold MTP JSON")
        report = json.loads(GOLD_MTP.read_text(encoding="utf-8"))
        stages = report["stages"]
        self.assertEqual([row["stage"] for row in stages], [f"S{i}" for i in range(7)])
        for row in stages:
            self.assertTrue(math.isfinite(float(row["causal_error"])))
            if "cache_error" in row:
                self.assertTrue(math.isfinite(float(row["cache_error"])))
                self.assertLessEqual(float(row["cache_error"]), 3e-6)
            self.assertLessEqual(float(row["causal_error"]), 3e-6)
        negatives = report["negatives"]
        self.assertEqual(len(negatives), 3)
        for row in negatives:
            self.assertIs(row["caught"], True)
            self.assertGreater(float(row["max_logit_error"]), 0.0)
            before = float(row["before_error"])
            self.assertTrue(math.isfinite(before))
            self.assertLess(before, float(row["max_logit_error"]))
            self.assertLessEqual(before, 3e-6)
        mtp = report["mtp"]
        # Stable teaching counts on this fixture: draft proposes 7, verifier accepts 3.
        self.assertEqual(mtp["proposed"], 7)
        self.assertEqual(mtp["accepted"], 3)
        self.assertEqual(mtp["label_counts"], [14, 12, 10])
        self.assertIs(mtp["greedy_equal"], True)

    def test_committed_svgs_match_regeneration_from_gold(self):
        renderer = load_renderer()
        report = json.loads(GOLD_MTP.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            renderer.render(report, out)
            for name in ("cache_causal.svg", "negative_controls.svg", "cpu_timing.svg"):
                committed = (OUT_DIR / name).read_text(encoding="utf-8")
                regenerated = (out / name).read_text(encoding="utf-8")
                self.assertEqual(
                    committed,
                    regenerated,
                    f"{name} drifted from docs/gold/p27_healthcheck_mtp.json",
                )
                self.assertIn("28_minigpt_healthcheck.py", committed)
                self.assertIn("<rect ", committed)

    def test_chapter_references_committed_svgs(self):
        body = CHAPTER.read_text(encoding="utf-8")
        for rel in (
            "assets/diagrams/p27/cache_causal.svg",
            "assets/diagrams/p27/negative_controls.svg",
            "assets/diagrams/p27/cpu_timing.svg",
            "docs/gold/p27_healthcheck_mtp.json",
        ):
            self.assertIn(rel, body)

    def test_non_mtp_gold_has_s0_s5_only(self):
        report = json.loads(GOLD.read_text(encoding="utf-8"))
        self.assertEqual([row["stage"] for row in report["stages"]], [f"S{i}" for i in range(6)])


if __name__ == "__main__":
    unittest.main()
