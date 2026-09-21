import numpy as np
import pytest
import torch
from mtwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
from mtwam.datasets.lerobot.latent_cache import assert_window_identity


class _FakeMulti:
    def __init__(self, n=1000, fail_first=True):
        self.num_frames, self.calls, self.fail_first = (n, 0, fail_first)
        self.ds_names = ["/data/suiteA_lerobot"]

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise RuntimeError("simulated decode failure")
        return {
            "task": "t",
            "dataset_index": torch.tensor(0),
            "episode_index": torch.tensor(int(idx)),
            "frame_index": torch.tensor(0),
            "x_is_pad": torch.zeros(1, dtype=torch.bool),
        }


def _rigged_base(multi):
    ds = object.__new__(BaseLerobotDataset)
    ds.multi_dataset = multi
    ds._split_lerobot_sample = lambda s: s
    ds.state_meta = [{"key": "s", "lerobot_key": "x"}]
    ds.action_meta = [{"key": "a", "lerobot_key": "x"}]
    ds.image_meta = [{"key": "i", "lerobot_key": "x"}]
    ds._get_state = lambda meta, s: torch.zeros(1)
    ds._get_action = lambda meta, s: torch.zeros(1)
    ds._get_image = lambda meta, s: torch.zeros(1)
    ds._get_additional_data = lambda sample, raw: sample
    ds.processor = None
    return ds


def test_retry_resample_identity_is_exposed():
    multi = _FakeMulti(fail_first=True)
    ds = _rigged_base(multi)
    np.random.seed(42)
    substituted = int(np.random.randint(multi.num_frames))
    assert substituted != 3
    np.random.seed(42)
    sample = BaseLerobotDataset.__getitem__(ds, 3)
    assert sample["idx"] == substituted
    wid = sample["window_ids"]
    assert int(wid["episode_index"]) == substituted
    ep_plan = {
        "global_start": 3,
        "n_windows": 1,
        "episode_index": 3,
        "repo_key": "suiteA_lerobot",
    }
    with pytest.raises(RuntimeError, match="identity drift"):
        assert_window_identity(wid, ep_plan, 3)


def test_clean_fetch_identity_passes():
    ds = _rigged_base(_FakeMulti(fail_first=False))
    sample = BaseLerobotDataset.__getitem__(ds, 7)
    wid = sample["window_ids"]
    assert sample["idx"] == 7 and int(wid["episode_index"]) == 7
    ep_plan = {
        "global_start": 7,
        "n_windows": 1,
        "episode_index": 7,
        "repo_key": "suiteA_lerobot",
    }
    assert_window_identity(wid, ep_plan, 7)
