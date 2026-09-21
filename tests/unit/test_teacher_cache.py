import os
import pytest
import torch
from mtwam.datasets.lerobot.teacher_cache import (
    build_teacher_cache_key,
    teacher_cache_path,
    atomic_save_teacher,
    load_teacher_payload,
    TEACHER_GRID_TRAJ,
    TEACHER_GRID_DINO,
    TEACHER_HORIZON_P,
)

BASE = dict(
    repo_id="libero/spatial",
    episode_index=3,
    frame_index=12,
    video_sample_indices=(0, 4, 8, 12, 16, 20, 24, 28, 32),
    video_size=(224, 224),
    camera=0,
)


def test_key_is_deterministic_hex():
    k1 = build_teacher_cache_key(**BASE)
    k2 = build_teacher_cache_key(**BASE)
    assert k1 == k2
    assert isinstance(k1, str)
    assert len(k1) == 64


def test_key_changes_with_window_and_camera():
    base = build_teacher_cache_key(**BASE)
    assert build_teacher_cache_key(**{**BASE, "camera": 1}) != base
    assert build_teacher_cache_key(**{**BASE, "frame_index": 13}) != base
    assert (
        build_teacher_cache_key(**{**BASE, "video_sample_indices": (0, 4, 8)}) != base
    )


def _fake_teacher_payload(num_cameras=2):
    P = TEACHER_HORIZON_P
    return {
        "teacher_tracks": torch.zeros(num_cameras, P, TEACHER_GRID_TRAJ, 2),
        "teacher_track_vis": torch.ones(
            num_cameras, P, TEACHER_GRID_TRAJ, dtype=torch.bool
        ),
        "teacher_dino": torch.zeros(num_cameras, P, TEACHER_GRID_DINO, 768),
    }


def test_atomic_save_roundtrip_and_no_tmp(tmp_path):
    path = tmp_path / "sample.teacher_P2.pt"
    atomic_save_teacher(_fake_teacher_payload(), path)
    got = load_teacher_payload(str(path))
    assert got["teacher_tracks"].shape == (2, TEACHER_HORIZON_P, TEACHER_GRID_TRAJ, 2)
    assert got["teacher_track_vis"].shape == (2, TEACHER_HORIZON_P, TEACHER_GRID_TRAJ)
    assert got["teacher_track_vis"].dtype == torch.bool
    assert got["teacher_dino"].shape == (2, TEACHER_HORIZON_P, TEACHER_GRID_DINO, 768)
    assert not list(tmp_path.glob(".*tmp*"))


def test_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_teacher_payload(str(tmp_path / "does_not_exist.pt"))


def test_load_validates_shape(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save(
        {
            "teacher_tracks": torch.zeros(2, TEACHER_HORIZON_P, 5, 2),
            "teacher_track_vis": torch.ones(2, TEACHER_HORIZON_P, 5, dtype=torch.bool),
            "teacher_dino": torch.zeros(2, TEACHER_HORIZON_P, TEACHER_GRID_DINO, 768),
        },
        path,
    )
    with pytest.raises(ValueError):
        load_teacher_payload(str(path))


import importlib.util
import pathlib

_PRODUCER_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "precompute_teacher_feats.py"
)


def _load_producer():
    spec = importlib.util.spec_from_file_location(
        "precompute_teacher_feats", _PRODUCER_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_WINDOW = dict(
    repo_id="libero/object",
    episode_index=1,
    frame_index=5,
    video_sample_indices=(0, 4, 8, 12),
    video_size=(224, 224),
)


def _fake_extractor_factory(counter):
    P = TEACHER_HORIZON_P

    def _extract(cam_frames):
        counter["n"] += 1
        return (
            torch.randn(P, TEACHER_GRID_TRAJ, 2),
            torch.ones(P, TEACHER_GRID_TRAJ, dtype=torch.bool),
            torch.randn(P, TEACHER_GRID_DINO, 768),
        )

    return _extract


def test_build_camera_payload_packages_extractor_output():
    producer = _load_producer()
    P = TEACHER_HORIZON_P
    out = (
        torch.zeros(P, TEACHER_GRID_TRAJ, 2),
        torch.ones(P, TEACHER_GRID_TRAJ, dtype=torch.bool),
        torch.zeros(P, TEACHER_GRID_DINO, 768),
    )
    payload = producer.build_camera_payload(None, lambda _frames: out)
    assert set(payload) == {"teacher_tracks", "teacher_track_vis", "teacher_dino"}
    assert payload["teacher_tracks"].shape == (P, TEACHER_GRID_TRAJ, 2)
    assert payload["teacher_track_vis"].shape == (P, TEACHER_GRID_TRAJ)
    assert payload["teacher_dino"].shape == (P, TEACHER_GRID_DINO, 768)


def test_process_window_assembles_and_saves_per_camera(tmp_path):
    producer = _load_producer()
    counter = {"n": 0}
    window = {**_WINDOW, "per_camera_frames": [None, None]}
    statuses = producer.process_window(
        window, str(tmp_path), _fake_extractor_factory(counter), overwrite=False
    )
    assert statuses == ["new", "new"]
    assert counter["n"] == 2
    for camera in (0, 1):
        key = build_teacher_cache_key(
            _WINDOW["repo_id"],
            _WINDOW["episode_index"],
            _WINDOW["frame_index"],
            _WINDOW["video_sample_indices"],
            _WINDOW["video_size"],
            camera,
        )
        payload = load_teacher_payload(teacher_cache_path(str(tmp_path), key))
        assert payload["teacher_tracks"].shape == (
            TEACHER_HORIZON_P,
            TEACHER_GRID_TRAJ,
            2,
        )
        assert payload["teacher_track_vis"].shape == (
            TEACHER_HORIZON_P,
            TEACHER_GRID_TRAJ,
        )
        assert payload["teacher_dino"].shape == (
            TEACHER_HORIZON_P,
            TEACHER_GRID_DINO,
            768,
        )
    k0 = build_teacher_cache_key(
        *[
            _WINDOW[f]
            for f in (
                "repo_id",
                "episode_index",
                "frame_index",
                "video_sample_indices",
                "video_size",
            )
        ],
        0,
    )
    k1 = build_teacher_cache_key(
        *[
            _WINDOW[f]
            for f in (
                "repo_id",
                "episode_index",
                "frame_index",
                "video_sample_indices",
                "video_size",
            )
        ],
        1,
    )
    assert k0 != k1


def test_process_window_writes_p_suffix_from_window_horizon(tmp_path):
    producer = _load_producer()
    window = {**_WINDOW, "horizon_p": 1, "per_camera_frames": [None]}
    producer.process_window(
        window, str(tmp_path), _fake_extractor_factory({"n": 0}), overwrite=False
    )
    key = build_teacher_cache_key(
        _WINDOW["repo_id"],
        _WINDOW["episode_index"],
        _WINDOW["frame_index"],
        _WINDOW["video_sample_indices"],
        _WINDOW["video_size"],
        0,
    )
    assert os.path.exists(teacher_cache_path(str(tmp_path), key, horizon_p=1))
    assert not os.path.exists(teacher_cache_path(str(tmp_path), key, horizon_p=2))
    w2 = {**_WINDOW, "per_camera_frames": [None]}
    producer.process_window(
        w2, str(tmp_path), _fake_extractor_factory({"n": 0}), overwrite=False
    )
    assert os.path.exists(
        teacher_cache_path(str(tmp_path), key, horizon_p=TEACHER_HORIZON_P)
    )


def test_process_window_skips_existing_without_recompute(tmp_path):
    producer = _load_producer()
    counter = {"n": 0}
    extractor = _fake_extractor_factory(counter)
    window = {**_WINDOW, "per_camera_frames": [None, None]}
    producer.process_window(window, str(tmp_path), extractor, overwrite=False)
    assert counter["n"] == 2
    statuses = producer.process_window(
        window, str(tmp_path), extractor, overwrite=False
    )
    assert statuses == ["skip", "skip"]
    assert counter["n"] == 2
    statuses = producer.process_window(window, str(tmp_path), extractor, overwrite=True)
    assert statuses == ["overwrite", "overwrite"]
    assert counter["n"] == 4
