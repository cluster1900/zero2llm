from pathlib import Path
import hashlib

ROOT=Path(__file__).parents[1]

def test_appendix_g_has_formula_and_hand_example():
    body=(ROOT/"chapters/10_appendix.md").read_text()
    assert "SHA256(bytes)" in body
    assert "726 bytes = 0.708984 KiB" in body
    assert "长度相同不能推出内容相同" in body

def test_appendix_g_keeps_versioned_sources():
    body=(ROOT/"chapters/10_appendix.md").read_text()
    assert "qwen-config" in body
    assert "2026-10-20" in body
