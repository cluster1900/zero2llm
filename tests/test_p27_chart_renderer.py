import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "plot_p27_healthcheck.py"


def load_renderer():
    spec = importlib.util.spec_from_file_location("p27_renderer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class P27ChartRendererTests(unittest.TestCase):
    def _report(self):
        return {
            "stages": [
                {"stage": "S0", "causal_error": 0.0, "cache_error": 0.0},
                {"stage": "S1", "causal_error": 0.1, "cache_error": 0.2},
                {"stage": "S2", "causal_error": 0.05, "cache_error": 0.03},
                {"stage": "S3", "causal_error": 0.04, "cache_error": 0.02},
                {"stage": "S4", "causal_error": 0.03, "cache_error": 0.01},
                {"stage": "S5", "causal_error": 0.02, "cache_error": 0.005},
            ],
            "negatives": [
                {"name": "wrong_mask", "caught": True, "max_logit_error": 0.3},
                {"name": "wrong_position", "caught": True, "max_logit_error": 0.2},
                {"name": "cache_pollution", "caught": True, "max_logit_error": 0.4},
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

    def test_renderer_calls_real_chart_geometry_and_binds_values(self):
        renderer = load_renderer()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "plots"
            paths = renderer.render(self._report(), out)
            self.assertEqual(
                [p.name for p in paths],
                ["cache_causal.svg", "negative_controls.svg", "cpu_timing.svg"],
            )
            cache = (out / "cache_causal.svg").read_text(encoding="utf-8")
            self.assertGreaterEqual(cache.count("<rect "), 12)
            self.assertIn('data-stage="S1"', cache)
            self.assertIn('data-value="0.2"', cache)
            self.assertIn("<line ", cache)
            negative = (out / "negative_controls.svg").read_text(encoding="utf-8")
            self.assertEqual(negative.count('data-phase="before"'), 3)
            self.assertEqual(negative.count('data-phase="after"'), 3)
            self.assertIn('data-value="0.3"', negative)
            timing = (out / "cpu_timing.svg").read_text(encoding="utf-8")
            self.assertEqual(timing.count('data-median='), 4)
            self.assertIn('data-min="9"', timing)
            self.assertIn('data-max="12"', timing)

    def test_cli_uses_healthcheck_json_and_reports_source(self):
        renderer = load_renderer()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "healthcheck.json"
            source.write_text(json.dumps(self._report()), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(source), "--out-dir", str(root / "plots")],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("DATA_SOURCE=code/28_minigpt_healthcheck.py --json", result.stdout)

    def test_missing_section_fails_before_writing_files(self):
        renderer = load_renderer()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "plots"
            broken = self._report()
            del broken["timing"]
            with self.assertRaises(ValueError):
                renderer.render(broken, out)
            self.assertFalse(out.exists())

    def test_source_mutation_would_break_value_binding(self):
        source = SCRIPT.read_text(encoding="utf-8")
        mutated = source.replace('data-value="{value:.12g}"', 'data-value="0"', 1)
        self.assertNotEqual(source, mutated)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mutated_script = root / "renderer.py"
            mutated_script.write_text(mutated, encoding="utf-8")
            report = root / "report.json"
            report.write_text(json.dumps(self._report()), encoding="utf-8")
            out = root / "plots"
            result = subprocess.run(
                [sys.executable, str(mutated_script), str(report), "--out-dir", str(out)],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            rendered = (out / "cache_causal.svg").read_text(encoding="utf-8")
            self.assertNotIn('data-value="0.2"', rendered)


if __name__ == "__main__":
    unittest.main()
