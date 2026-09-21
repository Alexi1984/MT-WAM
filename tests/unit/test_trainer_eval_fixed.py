import pytest
import torch
from mtwam.trainer import Wan22Trainer
from mtwam.utils.eval_windows import build_val_window_set, shard_round_robin

_BASE = {"video": torch.zeros(3, 4, 8, 8), "prompt": "t"}


def test_eval_sample_passes_teacher_and_cache_through():
    sample = dict(_BASE)
    sample["teacher_frames"] = torch.zeros(2, 3, 3, 16, 16)
    sample["cached_latent"] = torch.zeros(48, 3, 4, 4, dtype=torch.bfloat16)
    out = Wan22Trainer._to_batched_eval_sample(sample)
    assert out["teacher_frames"].shape == (1, 2, 3, 3, 16, 16)
    assert out["cached_latent"].shape == (1, 48, 3, 4, 4)
    assert out["cached_latent"].dtype == torch.bfloat16


def test_eval_sample_without_supply_adds_no_keys():
    out = Wan22Trainer._to_batched_eval_sample(dict(_BASE))
    assert "teacher_frames" not in out and "cached_latent" not in out


def test_shard_round_robin_partition():
    plan = [
        {
            "repo_key": "suiteA_lerobot",
            "episode_index": 0,
            "global_start": 0,
            "n_windows": 40,
        },
        {
            "repo_key": "suiteB_lerobot",
            "episode_index": 0,
            "global_start": 40,
            "n_windows": 40,
        },
    ]
    windows = build_val_window_set(plan, n_per_repo=8, seed=20260705)
    world = 3
    shards = [shard_round_robin(windows, r, world) for r in range(world)]
    flat = [w["global_idx"] for s in shards for w in s]
    assert sorted(flat) == sorted((w["global_idx"] for w in windows))
    assert len(set(flat)) == len(flat)
    for s in shards:
        assert {w["repo_key"] for w in s} == {"suiteA_lerobot", "suiteB_lerobot"}
    with pytest.raises(ValueError, match="shard spec"):
        shard_round_robin(windows, 3, 3)
