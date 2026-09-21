import torch
from mtwam.datasets.lerobot.teacher_cache import (
    build_teacher_cache_key,
    TEACHER_VIDEO_SIZE,
    derive_teacher_window_id,
    teacher_key_for_camera,
)


class _FakeMultiDataset:
    def __init__(self, ds_names, rows):
        self.ds_names = ds_names
        self.dataset_dirs = ds_names
        self._rows = rows

    def __getitem__(self, j):
        episode, frame, di = self._rows[j]
        return {
            "episode_index": torch.tensor([episode], dtype=torch.int64),
            "frame_index": torch.tensor([frame], dtype=torch.int64),
            "dataset_index": torch.tensor(di, dtype=torch.int64),
        }


class _FakeBaseDataset:
    def __init__(self, multi_dataset):
        self.multi_dataset = multi_dataset


_DS_NAMES = ["/data/libero_object_lerobot", "/data/libero_goal_lerobot"]
_ROWS = {0: (3, 12, 0), 1: (3, 13, 0), 2: (5, 12, 0), 3: (3, 12, 1)}
_VSI = (0, 4, 8, 12, 16, 20, 24, 28, 32)


def _ds():
    return _FakeBaseDataset(_FakeMultiDataset(_DS_NAMES, _ROWS))


def test_teacher_video_size_is_canonical_224():
    assert tuple(TEACHER_VIDEO_SIZE) == (224, 224)


def test_derive_window_id_reads_raw_row_not_idx():
    ds = _ds()
    got = derive_teacher_window_id(ds, 0)
    assert int(got["episode_index"]) == 3
    assert int(got["frame_index"]) == 12
    assert int(got["dataset_index"]) == 0
    assert got["repo_id"] == "/data/libero_object_lerobot"
    assert derive_teacher_window_id(ds, 3)["repo_id"] == "/data/libero_goal_lerobot"


def test_key_equals_hand_built_key_with_canonical_fields():
    ds = _ds()
    for j, (ep, fr, di) in _ROWS.items():
        for camera in (0, 1):
            expected = build_teacher_cache_key(
                _DS_NAMES[di], ep, fr, _VSI, TEACHER_VIDEO_SIZE, camera
            )
            assert teacher_key_for_camera(ds, j, _VSI, camera) == expected


def test_keys_are_distinct_across_window_camera_and_subdataset():
    ds = _ds()
    k = lambda j, cam: teacher_key_for_camera(ds, j, _VSI, cam)
    base = k(0, 0)
    assert k(1, 0) != base
    assert k(2, 0) != base
    assert k(0, 1) != base
    assert k(3, 0) != base
