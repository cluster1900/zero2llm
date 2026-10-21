import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).parents[1]

class AppendixGQuickrefTests(unittest.TestCase):
    def test_formula_and_hand_example(self):
        body=(ROOT/"chapters/10_appendix.md").read_text()
        self.assertIn("SHA256(bytes)",body)
        self.assertIn("726 bytes = 0.708984 KiB",body)
        self.assertIn("长度相同不能推出内容相同",body)

    def test_manifest_hash_and_length_bind_to_local_config(self):
        body=(ROOT/"chapters/10_appendix.md").read_text()
        manifest=json.loads((ROOT/"code/data/qwen3/manifest.json").read_text())
        raw=(ROOT/"code/data/qwen3/dense/config.json").read_bytes()
        digest=hashlib.sha256(raw).hexdigest()
        self.assertEqual(len(raw),726)
        self.assertEqual(digest,"660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd")
        self.assertEqual(manifest["models"]["dense"]["files"]["config.json"]["sha256"],digest)
        self.assertIn("726 字节",body)
        self.assertIn(digest,body)

    def test_one_byte_mutation_changes_digest(self):
        raw=(ROOT/"code/data/qwen3/dense/config.json").read_bytes()
        mutated=raw[:-1]+bytes([raw[-1]^1])
        self.assertEqual(len(raw),len(mutated))
        self.assertNotEqual(hashlib.sha256(raw).hexdigest(),hashlib.sha256(mutated).hexdigest())

    def test_versioned_source_is_present(self):
        body=(ROOT/"chapters/10_appendix.md").read_text()
        self.assertIn("qwen-config",body)
        self.assertIn("2026-10-20",body)

if __name__=="__main__":
    unittest.main()
