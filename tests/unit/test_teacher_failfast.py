import numpy as np
import pytest
from mtwam.datasets.lerobot.teacher_cache import DynamicBranchCacheMissing
from mtwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset


class _Stub(RobotVideoDataset):
    def __init__(self):
        pass

    def _get(self, idx):
        raise DynamicBranchCacheMissing(f"missing teacher cache for idx={idx}")


def test_missing_teacher_cache_raises_not_resampled(monkeypatch):
    ds = _Stub()
    calls = {"randint": 0}
    monkeypatch.setattr(
        np.random,
        "randint",
        lambda *a, **k: calls.__setitem__("randint", calls["randint"] + 1) or 0,
    )
    with pytest.raises(DynamicBranchCacheMissing):
        ds.__getitem__(0)
    assert calls["randint"] == 0
