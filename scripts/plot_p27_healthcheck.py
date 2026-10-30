#!/usr/bin/env python3
"""Render P27 charts from the real JSON emitted by code/28_minigpt_healthcheck.py."""
from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Iterable

# Keep this in the source so the provenance is visible to readers and check_book.
DATA_SOURCE = "code/28_minigpt_healthcheck.py --json"


def _finite(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _report(path: Path) -> dict:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON report: {path}") from exc
    if not isinstance(report, dict):
        raise ValueError("healthcheck report must be an object")
    stages = report.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("report.stages must be a non-empty list")
    names = [row.get("stage") for row in stages if isinstance(row, dict)]
    if names != [f"S{i}" for i in range(len(names))]:
        raise ValueError("report.stages must be ordered S0, S1, ...")
    for row in stages:
        if not isinstance(row, dict):
            raise ValueError("each stage must be an object")
        _finite(row.get("causal_error"), f"{row.get('stage')}.causal_error")
        if "cache_error" in row:
            _finite(row["cache_error"], f"{row.get('stage')}.cache_error")
    negatives = report.get("negatives")
    if not isinstance(negatives, list) or len(negatives) != 3:
        raise ValueError("report.negatives must contain exactly three controls")
    for row in negatives:
        if not isinstance(row, dict) or not row.get("name"):
            raise ValueError("each negative control needs a name")
        if not isinstance(row.get("caught"), bool) or not row["caught"]:
            raise ValueError(f"negative control {row.get('name')} was not caught")
        _finite(row.get("before_error"), f"negative {row.get('name')}.before_error")
        _finite(row.get("max_logit_error", row.get("after_error")),
                f"negative {row.get('name')}.max_logit_error")
    timing = report.get("timing", {}).get("milliseconds")
    if not isinstance(timing, dict):
        raise ValueError("report.timing.milliseconds is required")
    for key in ("full_first_ms", "cache_first_ms", "full_later_ms", "cache_later_ms"):
        if key not in timing:
            raise ValueError(f"timing.{key} is required")
        stats = timing[key]
        if isinstance(stats, dict):
            for name in ("min", "median", "max"):
                _finite(stats.get(name), f"timing.{key}.{name}")
        else:
            _finite(stats, f"timing.{key}")
    return report


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _stats(value: object, label: str) -> tuple[float, float, float]:
    if isinstance(value, dict):
        return (_finite(value.get("min"), f"{label}.min"),
                _finite(value.get("median"), f"{label}.median"),
                _finite(value.get("max"), f"{label}.max"))
    number = _finite(value, label)
    return number, number, number


def _svg_start(title: str, width: int = 960, height: int = 460) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f"<title id=\"title\">{_esc(title)}</title>",
        f'<desc id="desc">Generated from {_esc(DATA_SOURCE)}; values are not hand-entered.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
    ]


def _axis(svg: list[str], left: int, top: int, right: int, bottom: int,
          title: str, y_label: str) -> None:
    svg.extend([
        f'<text x="{left}" y="28" font-size="18" font-family="sans-serif">{_esc(title)}</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#222"/>',
        f'<text x="18" y="{top + (bottom-top)//2}" font-size="12" '
        f'font-family="sans-serif" transform="rotate(-90 18 {top + (bottom-top)//2})">{_esc(y_label)}</text>',
    ])


def _log_y(value: float, minimum: float, maximum: float, top: int, bottom: int) -> float:
    if value <= 0:
        return float(bottom)
    lo = math.log10(minimum)
    hi = math.log10(maximum)
    return bottom - (math.log10(value) - lo) / max(hi - lo, 1e-12) * (bottom - top)


def _render_error_chart(report: dict, output: Path) -> None:
    stages = report["stages"]
    labels = [row["stage"] for row in stages]
    cache = [row.get("cache_error") for row in stages]
    causal = [row["causal_error"] for row in stages]
    positive = [v for v in cache + causal if v is not None and float(v) > 0]
    minimum = min(positive + [1e-12])
    maximum = max(positive + [1.0])
    left, top, right, bottom = 90, 70, 900, 355
    svg = _svg_start(f"P27 cache and causal errors ({labels[-1]})")
    _axis(svg, left, top, right, bottom, f"P27 cache and causal errors ({labels[0]}–{labels[-1]})",
          "max absolute error (log10)")
    for exponent in range(math.floor(math.log10(minimum)), math.ceil(math.log10(maximum)) + 1):
        value = 10.0 ** exponent
        y = _log_y(value, minimum, maximum, top, bottom)
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#ddd"/>')
        svg.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-size="11">{value:.0e}</text>')
    group_width = (right - left) / max(1, len(labels))
    bar_width = min(28.0, group_width / 3.2)
    colors = ("#1565c0", "#ef6c00")
    for index, label in enumerate(labels):
        center = left + group_width * (index + 0.5)
        svg.append(f'<text x="{center:.1f}" y="{bottom+24}" text-anchor="middle" font-size="12">{_esc(label)}</text>')
        for series, values in (("cache", cache), ("causal", causal)):
            value = values[index]
            x = center + (-bar_width if series == "cache" else 0)
            if value is None:
                svg.append(f'<text x="{x+bar_width/2:.1f}" y="{bottom-6}" text-anchor="middle" '
                           f'font-size="10" fill="#666">n/a</text>')
                continue
            value = _finite(value, f"{label}.{series}")
            y = _log_y(value, minimum, maximum, top, bottom)
            height = max(2.0, bottom - y) if value > 0 else 2.0
            svg.append(f'<rect x="{x:.1f}" y="{bottom-height:.1f}" width="{bar_width:.1f}" '
                       f'height="{height:.1f}" fill="{colors[0 if series == "cache" else 1]}" '
                       f'data-stage="{_esc(label)}" data-series="{series}" data-value="{value:.12g}"/>')
            if value == 0:
                svg.append(f'<text x="{x+bar_width/2:.1f}" y="{bottom-8}" text-anchor="middle" '
                           f'font-size="10">0</text>')
    svg.extend([
        f'<rect x="{left}" y="390" width="14" height="14" fill="{colors[0]}"/>'
        f'<text x="{left+20}" y="402" font-size="12">cache equivalence</text>',
        f'<rect x="{left+180}" y="390" width="14" height="14" fill="{colors[1]}"/>'
        f'<text x="{left+200}" y="402" font-size="12">causal intervention</text>',
        "</svg>",
    ])
    output.write_text("\n".join(svg) + "\n", encoding="utf-8")


def _render_negative_chart(report: dict, output: Path) -> None:
    rows = report["negatives"]
    left, top, right, bottom = 90, 70, 900, 355
    maximum = max(_finite(row.get("max_logit_error", row.get("after_error")),
                          f"negative {row['name']}") for row in rows)
    maximum = max(maximum, 1.0)
    svg = _svg_start("P27 negative controls: before and after")
    _axis(svg, left, top, right, bottom, "P27 negative controls: alarm error", "max absolute logit error")
    group_width = (right - left) / len(rows)
    bar_width = min(34.0, group_width / 3.2)
    for index, row in enumerate(rows):
        name = str(row["name"])
        before = _finite(row.get("before_error"), f"negative {name}.before_error")
        after = _finite(row.get("after_error", row.get("max_logit_error")),
                        f"negative {name}.after_error")
        center = left + group_width * (index + 0.5)
        svg.append(f'<text x="{center:.1f}" y="{bottom+24}" text-anchor="middle" font-size="11">{_esc(name)}</text>')
        for offset, label, value, color in ((-bar_width, "before", before, "#90a4ae"),
                                             (0, "after", after, "#c62828")):
            height = max(2.0, value / maximum * (bottom-top)) if value > 0 else 2.0
            x = center + offset
            svg.append(f'<rect x="{x:.1f}" y="{bottom-height:.1f}" width="{bar_width:.1f}" '
                       f'height="{height:.1f}" fill="{color}" data-control="{_esc(name)}" '
                       f'data-phase="{label}" data-value="{value:.12g}"/>')
            if value == 0:
                svg.append(f'<text x="{x+bar_width/2:.1f}" y="{bottom-8}" text-anchor="middle" '
                           f'font-size="10">0</text>')
            else:
                svg.append(f'<text x="{x+bar_width/2:.1f}" y="{bottom-height-6:.1f}" text-anchor="middle" '
                           f'font-size="10">{value:.4g}</text>')
    svg.extend([
        f'<rect x="{left}" y="390" width="14" height="14" fill="#90a4ae"/>'
        f'<text x="{left+20}" y="402" font-size="12">before</text>',
        f'<rect x="{left+120}" y="390" width="14" height="14" fill="#c62828"/>'
        f'<text x="{left+140}" y="402" font-size="12">after / caught</text>',
        "</svg>",
    ])
    output.write_text("\n".join(svg) + "\n", encoding="utf-8")


def _render_timing_chart(report: dict, output: Path) -> None:
    timing = report["timing"]["milliseconds"]
    keys = ("full_first_ms", "cache_first_ms", "full_later_ms", "cache_later_ms")
    stats = {key: _stats(timing[key], f"timing.{key}") for key in keys}
    maximum = max(value[2] for value in stats.values())
    maximum = max(maximum, 1.0)
    left, top, right, bottom = 90, 70, 900, 355
    svg = _svg_start("P27 CPU timing: full/cache first/later")
    _axis(svg, left, top, right, bottom, "P27 CPU timing (milliseconds)", "median ms; whisker=min..max")
    categories = ("first token", "later token")
    group_width = (right-left) / 2
    bar_width = min(42.0, group_width / 4.8)
    for category_index, category in enumerate(categories):
        center = left + group_width * (category_index + 0.5)
        svg.append(f'<text x="{center:.1f}" y="{bottom+24}" text-anchor="middle" font-size="12">{category}</text>')
        selected = (("full", stats["full_first_ms"], "#1565c0"),
                    ("cache", stats["cache_first_ms"], "#ef6c00")) if category_index == 0 else (
                    ("full", stats["full_later_ms"], "#1565c0"),
                    ("cache", stats["cache_later_ms"], "#ef6c00"))
        for series_index, (series, (minimum, median, maximum_value), color) in enumerate(selected):
            x = center + (series_index - 1) * bar_width
            height = max(2.0, median / maximum * (bottom-top)) if median > 0 else 2.0
            y = bottom - height
            svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{height:.1f}" '
                       f'fill="{color}" data-series="{series}" data-category="{category}" '
                       f'data-min="{minimum:.12g}" data-median="{median:.12g}" data-max="{maximum_value:.12g}"/>')
            whisker_x = x + bar_width / 2
            y_min = bottom - minimum / maximum * (bottom-top)
            y_max = bottom - maximum_value / maximum * (bottom-top)
            svg.extend([
                f'<line x1="{whisker_x:.1f}" y1="{y_min:.1f}" x2="{whisker_x:.1f}" y2="{y_max:.1f}" stroke="#222"/>',
                f'<line x1="{whisker_x-6:.1f}" y1="{y_min:.1f}" x2="{whisker_x+6:.1f}" y2="{y_min:.1f}" stroke="#222"/>',
                f'<line x1="{whisker_x-6:.1f}" y1="{y_max:.1f}" x2="{whisker_x+6:.1f}" y2="{y_max:.1f}" stroke="#222"/>',
                f'<text x="{whisker_x:.1f}" y="{y-6:.1f}" text-anchor="middle" font-size="10">{median:.4g}</text>',
            ])
    svg.extend([
        f'<rect x="{left}" y="390" width="14" height="14" fill="#1565c0"/>'
        f'<text x="{left+20}" y="402" font-size="12">full forward</text>',
        f'<rect x="{left+150}" y="390" width="14" height="14" fill="#ef6c00"/>'
        f'<text x="{left+170}" y="402" font-size="12">cache forward</text>',
        "</svg>",
    ])
    output.write_text("\n".join(svg) + "\n", encoding="utf-8")


def render(report: dict, out_dir: Path) -> list[Path]:
    """Validate and render all charts; no output is written for invalid reports."""
    _report_path = report  # keep a descriptive local for tracebacks in callers
    if not isinstance(_report_path, dict):
        raise ValueError("report must be an object")
    # Reuse the same validation logic as the CLI without a temporary file.
    stages = _report_path.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("report.stages must be a non-empty list")
    # Serialize through a temporary in-memory round trip to exercise all checks.
    validated = json.loads(json.dumps(_report_path))
    # This branch intentionally mirrors _report's checks for programmatic callers.
    names = [row.get("stage") for row in stages if isinstance(row, dict)]
    if names != [f"S{i}" for i in range(len(names))]:
        raise ValueError("report.stages must be ordered S0, S1, ...")
    # Keep the file-oriented validation as the canonical path used by main.
    for row in stages:
        if not isinstance(row, dict):
            raise ValueError("each stage must be an object")
    negatives = validated.get("negatives")
    if not isinstance(negatives, list) or len(negatives) != 3:
        raise ValueError("report.negatives must contain exactly three controls")
    timing = validated.get("timing", {}).get("milliseconds")
    if not isinstance(timing, dict):
        raise ValueError("report.timing.milliseconds is required")
    for row in stages:
        _finite(row.get("causal_error"), f"{row.get('stage')}.causal_error")
        if "cache_error" in row:
            _finite(row["cache_error"], f"{row.get('stage')}.cache_error")
    for row in negatives:
        if not isinstance(row, dict) or not row.get("name") or row.get("caught") is not True:
            raise ValueError("negative controls must be caught")
        _finite(row.get("before_error"), f"negative {row.get('name')}.before_error")
        _finite(row.get("max_logit_error", row.get("after_error")), "negative error")
    for key in ("full_first_ms", "cache_first_ms", "full_later_ms", "cache_later_ms"):
        if key not in timing:
            raise ValueError(f"timing.{key} is required")
        _stats(timing[key], f"timing.{key}")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / "cache_causal.svg", out_dir / "negative_controls.svg", out_dir / "cpu_timing.svg"]
    _render_error_chart(validated, paths[0])
    _render_negative_chart(validated, paths[1])
    _render_timing_chart(validated, paths[2])
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="JSON produced by code/28_minigpt_healthcheck.py --json")
    parser.add_argument("--out-dir", type=Path, default=Path("assets/diagrams/p27"))
    args = parser.parse_args()
    report = _report(args.report)
    paths = render(report, args.out_dir)
    print(f"DATA_SOURCE={DATA_SOURCE}")
    print("wrote " + ", ".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
