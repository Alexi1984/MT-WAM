import os
import sys
import pytest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)


def pytest_collection_modifyitems(config, items):
    gpu_items = [it for it in items if "gpu" in it.keywords]
    if not gpu_items:
        return
    import torch

    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="needs CUDA; run with `pytest -m gpu`")
    for it in gpu_items:
        it.add_marker(skip_gpu)
