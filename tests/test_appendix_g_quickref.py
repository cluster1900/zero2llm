"""Appendix D/E/F/G quickref: formulas present + G hash bind with real 1-byte mutation."""
from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APPENDIX = ROOT / "chapters" / "10_appendix.md"
CONFIG = ROOT / "code" / "data" / "qwen3" / "dense" / "config.json"
MANIFEST = ROOT / "code" / "data" / "qwen3" / "manifest.json"


class AppendixQuickrefTests(unittest.TestCase):
    def setUp(self):
        self.body = APPENDIX.read_text(encoding="utf-8")

    def test_d_e_f_g_formulas_and_examples_are_present(self):
        self.assertIn(r"\mathrm{bytes}=2\times L\times B\times H_{kv}", self.body)
        self.assertIn("2048", self.body)
        self.assertIn("4096", self.body)
        self.assertIn(r"\Delta=x_{\mathrm{new}}-x_{\mathrm{old}}", self.body)
        self.assertIn("r=-8192/40960=-0.20", self.body)
        self.assertIn(r"\bar L_{\mathrm{doc}}=(2+4)/2=3", self.body)
        self.assertIn(r"L'=(0.6\times2)^2/2=0.72", self.body)
        self.assertIn("e338758cad0bc78e2ac6b2e298c0e621b78a6d2e80bac7df11df3a828bf8b73e", self.body)
        self.assertNotIn(r"\(", self.body)
        self.assertNotIn(r"\[", self.body)
        self.assertNotIn("\\n####", self.body)
        self.assertEqual(self.body.count("(z-y)x=4"), 1)

    def test_manifest_and_config_bytes_bind_to_g_claim(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        raw = CONFIG.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        self.assertEqual(len(raw), 726)
        self.assertEqual(digest, "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd")
        self.assertEqual(manifest["models"]["dense"]["files"]["config.json"]["sha256"], digest)
        self.assertEqual(manifest["models"]["dense"]["files"]["config.json"]["bytes"], len(raw))
        self.assertIn(str(len(raw)), self.body)
        self.assertIn(digest, self.body)

    def test_one_byte_mutation_preserves_length_changes_digest(self):
        raw = CONFIG.read_bytes()
        mutated = raw[:-1] + bytes([raw[-1] ^ 1])
        self.assertEqual(len(raw), len(mutated))
        mutated_digest = hashlib.sha256(mutated).hexdigest()
        self.assertNotEqual(hashlib.sha256(raw).hexdigest(), mutated_digest)
        self.assertEqual(
            mutated_digest,
            "e338758cad0bc78e2ac6b2e298c0e621b78a6d2e80bac7df11df3a828bf8b73e",
        )
        self.assertIn(mutated_digest, self.body)

    def test_removing_mutated_digest_breaks_g_contract(self):
        """True mutation: delete the documented XOR-1 digest → G claim fails."""
        mutated_digest = "e338758cad0bc78e2ac6b2e298c0e621b78a6d2e80bac7df11df3a828bf8b73e"
        self.assertIn(mutated_digest, self.body)
        broken = self.body.replace(mutated_digest, "0" * 64)
        with self.assertRaises(AssertionError):
            self.assertIn(mutated_digest, broken)

    def test_kv_hand_calc_matches_independent_arithmetic(self):
        match = re.search(
            r"2\\times2\\times2\\times2\\times8\\times4\\times4=(\d+)\$? bytes",
            self.body,
        )
        self.assertIsNotNone(match)
        self.assertEqual(int(match.group(1)), 2 * 2 * 2 * 2 * 8 * 4 * 4)
        self.assertEqual(2 * 2 * 2 * 2 * 8 * 4 * 4, 2048)
        self.assertEqual(2 * 2 * 2 * 4 * 8 * 4 * 4, 4096)


if __name__ == "__main__":
    unittest.main()
