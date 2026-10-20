#!/usr/bin/env python3
"""Render P27 charts from the real --json output of code/28_minigpt_healthcheck.py."""
import argparse
import json
from pathlib import Path

def esc(v):
    return str(v).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def svg(title, lines, out):
    text = "\n".join(f'  <text x="24" y="{42+i*22}" font-family="sans-serif" font-size="13">{esc(line)}</text>' for i,line in enumerate(lines))
    out.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{90+len(lines)*22}"><title>{esc(title)}</title><rect width="100%" height="100%" fill="white"/><text x="24" y="22" font-family="sans-serif" font-size="16">{esc(title)}</text>{text}</svg>\n')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("report", type=Path, help="JSON produced by code/28_minigpt_healthcheck.py --json")
    ap.add_argument("--out-dir", type=Path, default=Path("assets/diagrams/p27"))
    args=ap.parse_args()
    report=json.loads(args.report.read_text())
    stages=report["stages"]
    args.out_dir.mkdir(parents=True,exist_ok=True)
    svg("P27 S0–S6 cache equivalence and causal error",
        [f"{r['stage']}: cache_error={r.get('cache_error','n/a')}, causal_error={r['causal_error']}" for r in stages],
        args.out_dir/"cache_causal.svg")
    svg("P27 negative controls: observed failures",
        [f"{r['name']}: caught={r['caught']}, max_logit_error={r.get('max_logit_error','n/a')}" for r in report["negatives"]],
        args.out_dir/"negative_controls.svg")
    timing=report["timing"]
    svg("P27 CPU timing protocol",
        [f"protocol={timing['protocol']}", f"milliseconds={timing['milliseconds']}"],
        args.out_dir/"cpu_timing.svg")
    print(f"wrote 3 charts to {args.out_dir}")

if __name__=="__main__":
    main()
