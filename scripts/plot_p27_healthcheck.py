#!/usr/bin/env python3
"""Render P27 charts from real --json output of code/28_minigpt_healthcheck.py."""
import argparse, json
from pathlib import Path

def esc(v):
    return str(v).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def chart(title, labels, values, out, suffix=""):
    vals=[float(v) for v in values]
    lo=min(vals+[0.0]); hi=max(vals+[1.0])
    span=max(hi-lo,1e-12)
    pts=[]
    for i,v in enumerate(vals):
        x=70+(620*i/max(1,len(vals)-1))
        y=250-150*(v-lo)/span
        pts.append(f"{x:.1f},{y:.1f}")
    circles="".join(f'<circle cx="{p.split(",")[0]}" cy="{p.split(",")[1]}" r="4" fill="#1565c0"/>' for p in pts)
    labels_svg="".join(f'<text x="{70+(620*i/max(1,len(labels)-1)):.1f}" y="274" font-size="11" text-anchor="middle">{esc(l)}</text>' for i,l in enumerate(labels))
    body=f'''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="320" viewBox="0 0 760 320">
<title>{esc(title)}</title><desc>Generated from the current healthcheck JSON; no hand-entered benchmark values.</desc>
<rect width="760" height="320" fill="white"/><text x="24" y="24" font-size="16">{esc(title)}</text>
<line x1="70" y1="250" x2="690" y2="250" stroke="#333"/><line x1="70" y1="100" x2="70" y2="250" stroke="#333"/>
<polyline points="{" ".join(pts)}" fill="none" stroke="#1565c0" stroke-width="3"/>{circles}{labels_svg}
<text x="70" y="302" font-size="11">min={lo:.6g} max={hi:.6g}{esc(suffix)}</text></svg>'''
    out.write_text(body+"\n")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("report", type=Path, help="JSON produced by code/28_minigpt_healthcheck.py --json")
    ap.add_argument("--out-dir", type=Path, default=Path("assets/diagrams/p27"))
    args=ap.parse_args()
    report=json.loads(args.report.read_text()); stages=report["stages"]
    args.out_dir.mkdir(parents=True,exist_ok=True)
    labels=[r["stage"] for r in stages]
    chart("P27 cache equivalence error",labels,[r.get("cache_error",0.0) for r in stages],args.out_dir/"cache_causal.svg")
    neg=report["negatives"]
    chart("P27 negative controls: max logit error",[r["name"] for r in neg],[r.get("max_logit_error",0.0) for r in neg],args.out_dir/"negative_controls.svg")
    timing=report["timing"]["milliseconds"]
    keys=list(timing); chart("P27 CPU timing experiment",keys,[timing[k] for k in keys],args.out_dir/"cpu_timing.svg"," (milliseconds)")
    print(f"wrote 3 plotted SVGs to {args.out_dir}")

if __name__=="__main__":
    main()
