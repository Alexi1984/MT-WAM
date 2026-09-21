import json
import os
import pytest
import torch
from mtwam.datasets.lerobot.latent_cache import (
    META_FILENAME,
    build_cache_meta,
    build_episode_plan,
    episode_cache_path,
    episode_is_complete,
    repo_key_for,
    save_episode_latents,
    shard_episodes,
    write_cache_meta,
)
from mtwam.datasets.lerobot.robot_video_dataset import (
    DynamicBranchCacheMissing,
    read_latent_window,
)


class _FakeSubDS:
    def __init__(self, ep_lens, episodes=None):
        starts = torch.tensor([sum(ep_lens[:i]) for i in range(len(ep_lens))])
        self.episode_data_index = {"from": starts, "to": starts + torch.tensor(ep_lens)}
        self.episodes = episodes
        self.num_frames = int(sum(ep_lens))


class _FakeMulti:
    def __init__(self, subs, names):
        self._datasets = subs
        self.ds_names = names


def _plan2x():
    return build_episode_plan(
        _FakeMulti(
            [_FakeSubDS([3, 2]), _FakeSubDS([4])],
            ["/data/suiteA_lerobot", "/data/suiteB_lerobot"],
        )
    )


def test_episode_plan_globals_and_repo_keys():
    plan = _plan2x()
    assert [
        (p["repo_key"], p["episode_index"], p["global_start"], p["n_windows"])
        for p in plan
    ] == [
        ("suiteA_lerobot", 0, 0, 3),
        ("suiteA_lerobot", 1, 3, 2),
        ("suiteB_lerobot", 0, 5, 4),
    ]


def test_episode_plan_uses_loaded_episode_ids():
    plan = build_episode_plan(
        _FakeMulti([_FakeSubDS([2, 2], episodes=[7, 9])], ["/data/x"])
    )
    assert [p["episode_index"] for p in plan] == [7, 9]


def test_shard_episodes_partition_is_exact():
    plan = _plan2x()
    shards = [shard_episodes(plan, s, 2) for s in range(2)]
    ids = sorted((p["plan_idx"] for s in shards for p in s))
    assert ids == [p["plan_idx"] for p in plan]
    assert all((p["plan_idx"] % 2 == s for s in range(2) for p in shards[s]))
    with pytest.raises(ValueError, match="shard"):
        shard_episodes(plan, 2, 2)


def test_cache_meta_recipe_and_layout():
    meta = build_cache_meta(
        {
            "num_frames": 49,
            "num_extra_ref_frames": 16,
            "action_video_freq_ratio": 4,
            "video_size": [224, 224],
            "concat_multi_camera": "horizontal",
        },
        ["/data/suiteA_lerobot"],
        latent_shape=(48, 4, 14, 28),
    )
    assert meta["layout"] == "per_repo"
    assert (
        meta["num_frames"],
        meta["num_extra_ref_frames"],
        meta["action_video_freq_ratio"],
    ) == (49, 16, 4)
    assert meta["latent_shape"] == [48, 4, 14, 28]


def test_write_meta_atomic_and_idempotent(tmp_path):
    meta = build_cache_meta(
        {"num_frames": 33, "action_video_freq_ratio": 4, "video_size": [224, 224]},
        ["/d"],
    )
    p1 = write_cache_meta(str(tmp_path), meta)
    p2 = write_cache_meta(str(tmp_path), meta)
    assert p1 == p2 == str(tmp_path / META_FILENAME)
    assert json.load(open(p1))["num_extra_ref_frames"] == 0
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp") or ".tmp." in f]


def test_save_then_consumer_read_roundtrip(tmp_path):
    pytest.importorskip("safetensors")
    torch.manual_seed(0)
    lat = torch.randn(5, 48, 4, 14, 28, dtype=torch.float32)
    path = episode_cache_path(str(tmp_path), "suiteA_lerobot", 3)
    save_episode_latents(path, lat)
    assert episode_is_complete(path, 5) and (not episode_is_complete(path, 6))
    got = read_latent_window(os.path.dirname(path), 3, 2)
    assert got.dtype == torch.bfloat16 and got.shape == (48, 4, 14, 28)
    assert torch.equal(got, lat[2].to(torch.bfloat16))
    with pytest.raises(DynamicBranchCacheMissing):
        read_latent_window(os.path.dirname(path), 4, 0)


def test_repo_key_strips_and_basenames(tmp_path):
    d = tmp_path / "libero_object_no_noops_lerobot"
    d.mkdir()
    assert repo_key_for(str(d) + "/") == "libero_object_no_noops_lerobot"


_RECIPE_K0 = {"num_frames": 33, "action_video_freq_ratio": 4, "video_size": [224, 224]}
_RECIPE_K16 = {
    "num_frames": 49,
    "num_extra_ref_frames": 16,
    "action_video_freq_ratio": 4,
    "video_size": [224, 224],
}


def test_write_meta_refuses_recipe_mismatch(tmp_path):
    write_cache_meta(str(tmp_path), build_cache_meta(_RECIPE_K0, ["/d"]))
    with pytest.raises(RuntimeError, match="cache configuration mismatch"):
        write_cache_meta(str(tmp_path), build_cache_meta(_RECIPE_K16, ["/d"]))
    assert json.load(open(tmp_path / META_FILENAME))["num_frames"] == 33


def test_write_meta_legacy_missing_keys_warns_not_fails(tmp_path):
    write_cache_meta(str(tmp_path), build_cache_meta(_RECIPE_K0, ["/d"]))
    with_images = build_cache_meta(
        {**_RECIPE_K0, "shape_meta": {"images": [{"key": "image"}]}}, ["/d"]
    )
    assert with_images["image_keys"] == ["image"]
    with pytest.warns(UserWarning, match="is missing keys"):
        write_cache_meta(str(tmp_path), with_images)


def test_image_keys_read_from_shape_meta_nesting():
    nested = build_cache_meta(
        {
            **_RECIPE_K0,
            "shape_meta": {"images": [{"key": "image"}, {"key": "wrist_image"}]},
        },
        ["/d"],
    )
    assert nested["image_keys"] == ["image", "wrist_image"]
    top_level_only = build_cache_meta(
        {**_RECIPE_K0, "images": [{"key": "image"}]}, ["/d"]
    )
    assert "image_keys" not in top_level_only


def test_episode_is_complete_reconciles_file_metadata(tmp_path):
    pytest.importorskip("safetensors")
    path = episode_cache_path(str(tmp_path), "suiteA_lerobot", 0)
    sm_k0 = {
        "episode_index": 0,
        "repo_key": "suiteA_lerobot",
        "num_frames": 33,
        "num_extra_ref_frames": 0,
    }
    save_episode_latents(path, torch.randn(4, 8, 3, 2, 2), str_meta=sm_k0)
    assert episode_is_complete(path, 4, expected_meta=sm_k0)
    assert not episode_is_complete(path, 5, expected_meta=sm_k0)
    with pytest.raises(RuntimeError, match="__metadata__ mismatches"):
        episode_is_complete(
            path,
            4,
            expected_meta={**sm_k0, "num_frames": 49, "num_extra_ref_frames": 16},
        )
    assert episode_is_complete(
        path, 4, expected_meta={**sm_k0, "image_keys": '["image"]'}
    )


def test_assert_window_identity_guard():
    from mtwam.datasets.lerobot.latent_cache import assert_window_identity

    ep = {
        "global_start": 100,
        "n_windows": 5,
        "episode_index": 7,
        "repo_key": "suiteA_lerobot",
    }
    ok = {"repo_id": "/data/suiteA_lerobot", "episode_index": 7, "frame_index": 2}
    assert_window_identity(ok, ep, 102)
    assert_window_identity(None, ep, 102)
    with pytest.raises(RuntimeError, match="identity drift"):
        assert_window_identity({**ok, "frame_index": 3}, ep, 102)
    with pytest.raises(RuntimeError, match="identity drift"):
        assert_window_identity({**ok, "episode_index": 8}, ep, 102)
    with pytest.raises(RuntimeError, match="identity drift"):
        assert_window_identity({**ok, "repo_id": "/data/suiteB_lerobot"}, ep, 102)


def test_read_cached_latent_prefers_passthrough_ids(tmp_path):
    pytest.importorskip("safetensors")
    from types import SimpleNamespace
    from mtwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset

    lat = torch.randn(3, 4, 2, 2, 2)
    save_episode_latents(os.path.join(str(tmp_path), "ep_000002.safetensors"), lat)
    stub = SimpleNamespace(latent_cache_dir=str(tmp_path), _latent_cache_layout="flat")
    got = RobotVideoDataset._read_cached_latent(
        stub, 0, window_ids={"repo_id": "x", "episode_index": 2, "frame_index": 1}
    )
    assert torch.equal(got, lat[1].to(torch.bfloat16))
