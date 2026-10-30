"""P27 chart renderer: geometry binding + gold JSON regeneration contract."""
from __future__ import annotations

import importlib.util
import json
import math
import re
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


def _y_ticks(svg: str) -> list[float]:
    return [float(value) for value in re.findall(r'data-tick-value="([^"]+)"', svg)]


def _median_labels(svg: str) -> list[tuple[float, float, str]]:
    return [
        (float(x), float(y), text)
        for x, y, text in re.findall(
            r'<text x="([^"]+)" y="([^"]+)"[^>]*data-role="median-label"[^>]*>([^<]*)</text>',
            svg,
        )
    ]


def _whiskers(svg: str) -> list[tuple[str, float, float, float, float]]:
    return [
        (role, float(x1), float(y1), float(x2), float(y2))
        for x1, y1, x2, y2, role in re.findall(
            r'<line x1="([^"]+)" y1="([^"]+)" x2="([^"]+)" y2="([^"]+)"[^>]*'
            r'data-role="(whisker(?:-cap)?)"',
            svg,
        )
    ]


def _label_bbox(x: float, y: float, text: str, font_size: float = 10.0) -> tuple[float, float, float, float]:
    half = len(text) * font_size * 0.3
    return (x - half, x + half, y - font_size, y)


def _ranges_overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return not (a1 < min(b0, b1) or a0 > max(b0, b1))


def _bbox_hits_whisker(bbox: tuple[float, float, float, float],
                       whiskers: list[tuple[str, float, float, float, float]]) -> bool:
    x0, x1, y0, y1 = bbox
    for role, xa, ya, xb, yb in whiskers:
        if role == "whisker":
            if x0 <= xa <= x1 and _ranges_overlap(y0, y1, ya, yb):
                return True
        elif y0 <= ya <= y1 and _ranges_overlap(x0, x1, xa, xb):
            return True
    return False


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

    def test_linear_charts_have_numeric_y_ticks_including_0_and_max(self):
        renderer = load_renderer()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "plots"
            renderer.render(self._fixture(), out)
            for name in ("negative_controls.svg", "cpu_timing.svg"):
                svg = (out / name).read_text(encoding="utf-8")
                ticks = _y_ticks(svg)
                self.assertGreaterEqual(len(ticks), 5, name)
                self.assertLessEqual(len(ticks), 6, name)
                self.assertEqual(ticks[0], 0.0, name)
                self.assertEqual(ticks[-1], max(ticks), name)
                labels = re.findall(r'data-role="y-tick"[^>]*>([^<]*)</text>', svg)
                self.assertIn("0", labels, name)
                self.assertIn(renderer._format_tick(ticks[-1]), labels, name)
            cache = (out / "cache_causal.svg").read_text(encoding="utf-8")
            self.assertNotIn('data-role="y-tick"', cache)

    def _assert_timing_labels_clear(self, svg: str) -> None:
        labels = _median_labels(svg)
        whiskers = _whiskers(svg)
        self.assertEqual(len(labels), 4)
        self.assertGreaterEqual(len(whiskers), 12)
        boxes = []
        for x, y, text in labels:
            bbox = _label_bbox(x, y, text)
            boxes.append(bbox)
            self.assertFalse(
                _bbox_hits_whisker(bbox, whiskers),
                f"median label {text!r} at {bbox} intersects a whisker",
            )
        for index, first in enumerate(boxes):
            for other in boxes[index + 1:]:
                self.assertFalse(
                    _ranges_overlap(first[0], first[1], other[0], other[1])
                    and _ranges_overlap(first[2], first[3], other[2], other[3]),
                    f"adjacent median labels overlap: {first} vs {other}",
                )

    def test_timing_median_labels_clear_whiskers(self):
        renderer = load_renderer()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "plots"
            renderer.render(self._fixture(), out)
            self._assert_timing_labels_clear((out / "cpu_timing.svg").read_text(encoding="utf-8"))
            renderer.render(json.loads(GOLD_MTP.read_text(encoding="utf-8")), out)
            gold_svg = (out / "cpu_timing.svg").read_text(encoding="utf-8")
            self._assert_timing_labels_clear(gold_svg)
            ticks = _y_ticks(gold_svg)
            self.assertEqual(ticks[0], 0.0)
            self.assertEqual(ticks[-1], max(ticks))
            negative = (out / "negative_controls.svg").read_text(encoding="utf-8")
            neg_ticks = _y_ticks(negative)
            self.assertEqual(neg_ticks[0], 0.0)
            self.assertEqual(neg_ticks[-1], max(neg_ticks))

    def test_old_timing_label_placement_detected_as_overlap(self):
        source = SCRIPT.read_text(encoding="utf-8")
        old_call = "_timing_label_position(whisker_x, y, series_index, label_text)"
        mutated = source.replace(old_call, "(whisker_x, y - 6)", 1)
        self.assertNotEqual(source, mutated)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "renderer.py").write_text(mutated, encoding="utf-8")
            report = root / "report.json"
            report.write_text(json.dumps(self._fixture()), encoding="utf-8")
            out = root / "plots"
            result = subprocess.run(
                [sys.executable, str(root / "renderer.py"), str(report), "--out-dir", str(out)],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            svg = (out / "cpu_timing.svg").read_text(encoding="utf-8")
            hits = [
                text
                for x, y, text in _median_labels(svg)
                if _bbox_hits_whisker(_label_bbox(x, y, text), _whiskers(svg))
            ]
            self.assertTrue(
                hits,
                "old placement at whisker_x, bar top-6 must be flagged as overlapping a whisker",
            )

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

    def _p27_sections(self, body: str) -> dict[str, str]:
        sections = {}
        for part in re.split(r"(?=^## P27\.)", body, flags=re.M):
            match = re.match(r"^## P27\.(\d+)", part)
            if match:
                sections[f"P27.{match.group(1)}"] = part
        return sections

    def test_chapter_references_committed_svgs(self):
        body = CHAPTER.read_text(encoding="utf-8")
        rels = (
            "assets/diagrams/p27/cache_causal.svg",
            "assets/diagrams/p27/negative_controls.svg",
            "assets/diagrams/p27/cpu_timing.svg",
        )
        for rel in rels:
            self.assertEqual(body.count(rel), 1, rel)
        self.assertIn("docs/gold/p27_healthcheck_mtp.json", body)
        sections = self._p27_sections(body)
        self.assertIn(rels[0], sections["P27.5"])
        self.assertIn(rels[1], sections["P27.8"])
        self.assertIn(rels[2], sections["P27.10"])

    def test_chapter_p27_10_timing_medians_match_gold(self):
        body = CHAPTER.read_text(encoding="utf-8")
        p27_10 = self._p27_sections(body)["P27.10"]
        paragraph = next(part for part in re.split(r"\n\s*\n", p27_10) if "中位数" in part)
        quoted = [float(n) for n in re.findall(r"\d+\.\d{3}", paragraph)]
        ms = json.loads(GOLD_MTP.read_text(encoding="utf-8"))["timing"]["milliseconds"]
        expected = [
            round(ms["full_first_ms"]["median"], 3),
            round(ms["cache_first_ms"]["median"], 3),
            round(ms["full_later_ms"]["median"], 3),
            round(ms["cache_later_ms"]["median"], 3),
        ]
        self.assertEqual(quoted, expected)

    def test_non_mtp_gold_has_s0_s5_only(self):
        report = json.loads(GOLD.read_text(encoding="utf-8"))
        self.assertEqual([row["stage"] for row in report["stages"]], [f"S{i}" for i in range(6)])


if __name__ == "__main__":
    unittest.main()
