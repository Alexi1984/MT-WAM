import pytest
import torch
from mtwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset
from mtwam.datasets.lerobot.teacher_cache import (
    TEACHER_GRID_DINO,
    TEACHER_GRID_TRAJ,
    TEACHER_HORIZON_P,
    DynamicBranchCacheMissing,
    atomic_save_teacher,
    teacher_cache_path,
    teacher_key_for_camera,
)


class _FakeMultiDataset:
    def __init__(self, ds_names, rows):
        self.ds_names = ds_names
        self.dataset_dirs = ds_names
        self._rows = rows

    def __getitem__(self, j):
        ep, fr, di = self._rows[j]
        return {
            "episode_index": torch.tensor([ep], dtype=torch.int64),
            "frame_index": torch.tensor([fr], dtype=torch.int64),
            "dataset_index": torch.tensor(di, dtype=torch.int64),
        }


class _FakeBaseDataset:
    def __init__(self, multi_dataset):
        self.multi_dataset = multi_dataset


_DS_NAMES = ["/data/libero_object_lerobot", "/data/libero_goal_lerobot"]
_ROWS = {7: (3, 12, 0), 9: (5, 4, 1)}
_VSI = (0, 4, 8, 12, 16, 20, 24, 28, 32)


class _Stub(RobotVideoDataset):
    def __init__(self, cache_dir, enable, online=False, concat="horizontal"):
        self.enable_dynamic_branch = enable
        self.teacher_feature_cache_dir = str(cache_dir)
        self.online_teacher = online
        self.concat_multi_camera = concat
        self.lerobot_dataset = _FakeBaseDataset(_FakeMultiDataset(_DS_NAMES, _ROWS))
        self.video_sample_indices = list(_VSI)


def _fake_pixels(num_cameras):
    return (
        torch.zeros(3, 3, 8, 8)
        if num_cameras == 1
        else torch.zeros(num_cameras, 3, 3, 8, 8)
    )


def _fake_obs_pixels(num_cameras, t_obs=33, h=40, w=40):
    return (
        torch.zeros(t_obs, 3, h, w)
        if num_cameras == 1
        else torch.zeros(num_cameras, t_obs, 3, h, w)
    )


def _write_cache(cache_dir, ds, loaded_idx, num_cameras, fill):
    P = TEACHER_HORIZON_P
    for camera in range(num_cameras):
        key = teacher_key_for_camera(ds, loaded_idx, list(_VSI), camera)
        atomic_save_teacher(
            {
                "teacher_tracks": torch.full(
                    (P, TEACHER_GRID_TRAJ, 2), float(fill + camera)
                ),
                "teacher_track_vis": torch.ones(P, TEACHER_GRID_TRAJ, dtype=torch.bool),
                "teacher_dino": torch.zeros(P, TEACHER_GRID_DINO, 768),
            },
            teacher_cache_path(str(cache_dir), key),
        )


def test_disabled_leaves_data_dict_unchanged(tmp_path):
    ds = _Stub(tmp_path, enable=False)
    data = {"video": "sentinel"}
    out = ds._maybe_attach_teacher(data, {"idx": 7, "pixel_values": _fake_pixels(2)})
    assert out is data
    assert not any((k.startswith("teacher_") for k in data))


def test_enabled_loads_and_stacks_per_camera(tmp_path):
    ds = _Stub(tmp_path, enable=True)
    num_cameras = 2
    _write_cache(tmp_path, ds.lerobot_dataset, 7, num_cameras, fill=1.0)
    data = {}
    ds._maybe_attach_teacher(
        data, {"idx": 7, "pixel_values": _fake_pixels(num_cameras)}
    )
    P = TEACHER_HORIZON_P
    assert data["teacher_tracks"].shape == (num_cameras, P, TEACHER_GRID_TRAJ, 2)
    assert data["teacher_track_vis"].shape == (num_cameras, P, TEACHER_GRID_TRAJ)
    assert data["teacher_track_vis"].dtype == torch.bool
    assert data["teacher_dino"].shape == (num_cameras, P, TEACHER_GRID_DINO, 768)
    assert torch.allclose(
        data["teacher_tracks"][0], torch.full((P, TEACHER_GRID_TRAJ, 2), 1.0)
    )
    assert torch.allclose(
        data["teacher_tracks"][1], torch.full((P, TEACHER_GRID_TRAJ, 2), 2.0)
    )


def test_single_camera_ndim4(tmp_path):
    ds = _Stub(tmp_path, enable=True)
    _write_cache(tmp_path, ds.lerobot_dataset, 9, 1, fill=5.0)
    data = {}
    ds._maybe_attach_teacher(data, {"idx": 9, "pixel_values": _fake_pixels(1)})
    assert data["teacher_tracks"].shape == (1, TEACHER_HORIZON_P, TEACHER_GRID_TRAJ, 2)


def test_keys_on_loaded_idx_not_a_stale_index(tmp_path):
    ds = _Stub(tmp_path, enable=True)
    _write_cache(tmp_path, ds.lerobot_dataset, 7, 2, fill=1.0)
    with pytest.raises(DynamicBranchCacheMissing):
        ds._maybe_attach_teacher({}, {"idx": 9, "pixel_values": _fake_pixels(2)})


def test_online_teacher_skips_offline_cache(tmp_path):
    ds = _Stub(tmp_path, enable=True, online=True, concat="horizontal")
    data = {"video": "sentinel"}
    out = ds._maybe_attach_teacher(
        data, {"idx": 7, "pixel_values": _fake_obs_pixels(2)}
    )
    assert out is data
    assert (
        "teacher_tracks" not in data
        and "teacher_track_vis" not in data
        and ("teacher_dino" not in data)
    )
    assert "teacher_frames" in data


def test_online_teacher_frames_shape_horizontal(tmp_path):
    ds = _Stub(tmp_path, enable=True, online=True, concat="horizontal")
    data = {}
    ds._maybe_attach_teacher(data, {"idx": 7, "pixel_values": _fake_obs_pixels(2)})
    P = TEACHER_HORIZON_P
    assert data["teacher_frames"].shape == (2, P + 1, 3, 224, 224)
    assert data["teacher_frames"].dtype == torch.float32


def test_online_teacher_frames_shape_robotwin(tmp_path):
    ds = _Stub(tmp_path, enable=True, online=True, concat="robotwin")
    data = {}
    ds._maybe_attach_teacher(data, {"idx": 7, "pixel_values": _fake_obs_pixels(3)})
    P = TEACHER_HORIZON_P
    assert data["teacher_frames"].shape == (3, P + 1, 3, 224, 280)


def test_online_teacher_frames_full_clip_even_when_horizon_1(tmp_path):
    ds = _Stub(tmp_path, enable=True, online=True, concat="horizontal")
    ds.dynamic_branch_horizon = 1
    data = {}
    ds._maybe_attach_teacher(data, {"idx": 7, "pixel_values": _fake_obs_pixels(2)})
    P = TEACHER_HORIZON_P
    assert data["teacher_frames"].shape == (2, P + 1, 3, 224, 224)


def test_prepare_teacher_frames_helper_geometry():
    from mtwam.datasets.lerobot.teacher_cache import prepare_teacher_frames_for_camera

    cam = torch.zeros(33, 3, 50, 60)
    vsi = list(range(0, 33, 4))
    P = TEACHER_HORIZON_P
    sq = prepare_teacher_frames_for_camera(cam, vsi, (224, 224))
    assert sq.shape == (P + 1, 3, 224, 224) and sq.dtype == torch.float32
    rect = prepare_teacher_frames_for_camera(cam, vsi, (224, 280))
    assert rect.shape == (P + 1, 3, 224, 280)
    with pytest.raises(ValueError):
        prepare_teacher_frames_for_camera(
            torch.ones(33, 3, 8, 8) * 2.0, vsi, (224, 224)
        )
