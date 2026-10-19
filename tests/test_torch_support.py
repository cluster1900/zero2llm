"""Dependency checks themselves run even without the optional tensor library."""
import importlib.metadata
import unittest
from unittest.mock import patch
from torch_test_support import compatible_torch, torch_status, torch_reason


class TorchSupportTests(unittest.TestCase):
    def test_stable_series_only(self):
        for v in ('2.14.0', '2.14.1+cpu', '2.14.12+cu130'):
            self.assertTrue(compatible_torch(v))
        for v in (None, '', '2.14', '2.14.1rc1', '2.14.1.dev0', '2.15.0', '12.14.1'):
            self.assertFalse(compatible_torch(v))

    def test_metadata_and_module_are_both_required(self):
        with patch('importlib.util.find_spec', return_value=None), patch('importlib.metadata.version', return_value='2.14.1'):
            self.assertEqual(torch_status(), (False, '2.14.1', False))
        with patch('importlib.util.find_spec', return_value=object()), patch('importlib.metadata.version', return_value='2.13.9'):
            self.assertEqual(torch_status(), (False, '2.13.9', True))
        with patch('importlib.metadata.version', side_effect=importlib.metadata.PackageNotFoundError):
            ready, version, present = torch_status()
            self.assertFalse(ready)
            self.assertEqual(version, 'not installed')
        self.assertIn('2.13.9', torch_reason('experiment', '2.13.9', True))
