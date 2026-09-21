import os
import sys
import torch
import pytest
from mtwam.datasets.lerobot.teacher_cache import (
    teacher_geometry,
    build_teacher_cache_key,
    load_teacher_payload,
    atomic_save_teacher,
    teacher_cache_path,
    TEACHER_GRID_TRAJ,
    TEACHER_GRID_DINO,
)

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
)
from precompute_teacher_feats import _grid_query_points


def test_geometry_values():
    assert teacher_geometry("horizontal") == ((224, 224), (14, 14), (16, 16))
    assert teacher_geometry("robotwin") == ((224, 280), (16, 20), (16, 20))


def test_geometry_unknown_layout_raises():
    with pytest.raises(ValueError, match="unknown camera_layout"):
        teacher_geometry("diagonal")


def test_geometry_none_layout_raises():
    with pytest.raises(ValueError, match="unknown camera_layout"):
        teacher_geometry(None)


def test_geometry_dino_grid_matches_patch14_divisibility():
    for layout in ("horizontal", "robotwin"):
        (H, W), _traj, (dgh, dgw) = teacher_geometry(layout)
        assert H % 14 == 0 and W % 14 == 0
        assert (dgh, dgw) == (H // 14, W // 14)


def test_cache_key_video_size_prevents_libero_robotwin_collision():
    k_lib = build_teacher_cache_key("repo", 3, 7, (0, 4, 8), (224, 224), 0)
    k_rt = build_teacher_cache_key("repo", 3, 7, (0, 4, 8), (224, 280), 0)
    assert k_lib != k_rt


def test_load_payload_validates_against_layout_grids(tmp_path):
    P, g = (2, 320)
    payload = {
        "teacher_tracks": torch.zeros(P, g, 2),
        "teacher_track_vis": torch.ones(P, g, dtype=torch.bool),
        "teacher_dino": torch.zeros(P, g, 768),
    }
    path = teacher_cache_path(str(tmp_path), "k")
    atomic_save_teacher(payload, path)
    ok = load_teacher_payload(path, traj_grid=320, dino_grid=320)
    assert tuple(ok["teacher_tracks"].shape) == (P, 320, 2)
    with pytest.raises(ValueError, match="teacher_tracks"):
        load_teacher_payload(
            path, traj_grid=TEACHER_GRID_TRAJ, dino_grid=TEACHER_GRID_DINO
        )


def test_load_payload_default_is_libero_backward_compat(tmp_path):
    P = 2
    payload = {
        "teacher_tracks": torch.zeros(P, TEACHER_GRID_TRAJ, 2),
        "teacher_track_vis": torch.ones(P, TEACHER_GRID_TRAJ, dtype=torch.bool),
        "teacher_dino": torch.zeros(P, TEACHER_GRID_DINO, 768),
    }
    path = teacher_cache_path(str(tmp_path), "k2")
    atomic_save_teacher(payload, path)
    out = load_teacher_payload(path)
    assert tuple(out["teacher_dino"].shape) == (P, TEACHER_GRID_DINO, 768)


def test_grid_query_points_rectangular_and_square_backcompat():
    sq = _grid_query_points(14, 224, "cpu")
    assert sq.shape == (1, 196, 3)
    assert sq[0, 0, 0] == 0.0
    assert tuple(sq[0, 0, 1:].tolist()) == (8.0, 8.0)
    rt = _grid_query_points((16, 20), (224, 280), "cpu")
    assert rt.shape == (1, 320, 3)
    assert tuple(rt[0, 0, 1:].tolist()) == (7.0, 7.0)
