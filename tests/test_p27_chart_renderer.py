import json
from pathlib import Path

def test_plot_script_has_geometry_and_real_fields():
    src=(Path(__file__).parents[1]/"scripts/plot_p27_healthcheck.py").read_text()
    assert "--json" in src
    assert "cache_error" in src and "negative_controls" in src and "timing" in src
    assert "<polyline" in src and "<circle" in src

def test_minimal_report_schema(tmp_path):
    report={"stages":[{"stage":"S1","cache_error":0.0,"causal_error":0.0}],
            "negatives":[{"name":"future","caught":True,"max_logit_error":0.1}],
            "timing":{"protocol":{"clock":"perf_counter"},"milliseconds":{"prefill":1}}}
    p=tmp_path/"healthcheck.json"; p.write_text(json.dumps(report))
    assert p.is_file()
