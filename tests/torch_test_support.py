"""Shared CPU test gate; inspect metadata without importing optional PyTorch."""
import importlib.util
import importlib.metadata
import re


def compatible_torch(version):
    """Accept verified stable 2.14 patch releases and local wheel tags."""
    return bool(version and re.fullmatch(r'2\.14\.\d+(?:\+[A-Za-z0-9.]+)?', version))


def torch_status():
    present = importlib.util.find_spec('torch') is not None
    try:
        version = importlib.metadata.version('torch')
    except importlib.metadata.PackageNotFoundError:
        version = 'not installed'
    return present and compatible_torch(version), version, present


def torch_reason(label, version, present):
    return (f'{label} requires installed stable torch 2.14.x; found {version} '
            f'(module present={present}). Install requirements.txt; skipped is not passed.')
