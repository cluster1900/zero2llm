import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).parents[1]
APPENDIX = ROOT / "chapters" / "10_appendix.md"
CONFIG = ROOT / "code/data/qwen3/dense/config.json"
MANIFEST = ROOT / "code/data/qwen3/manifest.json"


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
        self.assertIn("L'=(0.6\\times2)^2/2=0.72", self.body)
        self.assertIn("e338758cad0bc78e2ac6b2e298c0e621b78a6d2e80bac7df11df3a828bf8b73e", self.body)

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
        self.assertNotEqual(hashlib.sha256(raw).hexdigest(), hashlib.sha256(mutated).hexdigest())
        self.assertIn(hashlib.sha256(mutated).hexdigest(), self.body)

    def test_hash_mutation_breaks_the_document_contract(self):
        original = "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd"
        mutated = original[:-1] + ("0" if original[-1] != "0" else "1")
        broken = self.body.replace(original, mutated)
        self.assertNotIn(original, broken)
        self.assertNotEqual(broken, self.body)


if __name__ == "__main__":
    unittest.main()
