import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).parents[1]

def test_renderer_invokes_real_geometry(tmp_path):
    report={"stages":[{"stage":"S0","cache_error":0.0,"causal_error":0.1},
                      {"stage":"S1","cache_error":0.2,"causal_error":0.05}],
            "negatives":[{"name":"future","caught":True,"max_logit_error":0.3},
                         {"name":"cache","caught":True,"max_logit_error":0.2}],
            "timing":{"protocol":{"clock":"perf_counter"},
                      "milliseconds":{"full_first_median":10.0,"cache_later_median":4.0}}}
    src=tmp_path/"healthcheck.json"; src.write_text(json.dumps(report))
    out=tmp_path/"plots"
    subprocess.run([sys.executable,str(ROOT/"scripts/plot_p27_healthcheck.py"),str(src),"--out-dir",str(out)],check=True)
    files=sorted(out.glob("*.svg"))
    assert [p.name for p in files]==["cache_causal.svg","cpu_timing.svg","negative_controls.svg"]
    for p in files:
        body=p.read_text()
        assert "<polyline" in body and "<circle" in body
        assert "10" in body or "0.3" in body or "0.2" in body

def test_renderer_rejects_missing_sections(tmp_path):
    src=tmp_path/"bad.json"; src.write_text(json.dumps({"stages":[]}))
    out=tmp_path/"plots"
    result=subprocess.run([sys.executable,str(ROOT/"scripts/plot_p27_healthcheck.py"),str(src),"--out-dir",str(out)])
    assert result.returncode != 0
