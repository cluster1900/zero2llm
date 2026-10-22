import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "plot_p27_healthcheck.py"


class P27ChartRendererTests(unittest.TestCase):
    def _report(self):
        return {
            "stages": [
                {"stage": "S0", "cache_error": 0.0, "causal_error": 0.1},
                {"stage": "S1", "cache_error": 0.2, "causal_error": 0.05},
            ],
            "negatives": [
                {"name": "future", "caught": True, "max_logit_error": 0.3},
                {"name": "cache", "caught": True, "max_logit_error": 0.2},
            ],
            "timing": {
                "protocol": {"clock": "perf_counter"},
                "milliseconds": {
                    "full_first_median": 10.0,
                    "cache_later_median": 4.0,
                },
            },
        }

    def test_renderer_invokes_real_geometry_and_values(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "healthcheck.json"
            out = root / "plots"
            src.write_text(json.dumps(self._report()), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(src), "--out-dir", str(out)],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            files = sorted(out.glob("*.svg"))
            self.assertEqual(
                [p.name for p in files],
                ["cache_causal.svg", "cpu_timing.svg", "negative_controls.svg"],
            )
            for path in files:
                body = path.read_text(encoding="utf-8")
                self.assertIn("<polyline", body)
                self.assertIn("<circle", body)
            self.assertIn("0.2", (out / "cache_causal.svg").read_text(encoding="utf-8"))
            self.assertIn("0.3", (out / "negative_controls.svg").read_text(encoding="utf-8"))
            self.assertIn("10", (out / "cpu_timing.svg").read_text(encoding="utf-8"))

    def test_renderer_rejects_missing_sections_and_does_not_emit_fake_plots(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "bad.json"
            out = root / "plots"
            src.write_text(json.dumps({"stages": []}), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(src), "--out-dir", str(out)],
                text=True, capture_output=True, check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(any(out.glob("*.svg")) if out.exists() else False)


if __name__ == "__main__":
    unittest.main()
