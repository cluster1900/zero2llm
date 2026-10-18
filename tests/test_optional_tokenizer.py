"""The optional comparison must skip absent or mismatched distributions."""
import importlib.metadata
from pathlib import Path
import runpy
import unittest
from unittest.mock import patch

FOUNDATIONS = Path(__file__).with_name('test_foundations.py')


class OptionalTokenizerTests(unittest.TestCase):
    def test_missing_or_other_version_skips(self):
        for installed in [None, '0.23.3']:
            with self.subTest(installed=installed):
                options = ({'side_effect': importlib.metadata.PackageNotFoundError}
                           if installed is None else {'return_value': installed})
                with patch('importlib.metadata.version', **options):
                    namespace = runpy.run_path(str(FOUNDATIONS))
                result = unittest.TestResult()
                namespace['HFTests']('test_round_trip').run(result)
                self.assertEqual(len(result.skipped), 1)
                self.assertEqual(result.errors, [])
                self.assertEqual(result.failures, [])
                self.assertIn('0.21.4', result.skipped[0][1])
