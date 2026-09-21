import pytest
import torch
from mtwam.datasets.lerobot.teacher_cache import (
    TEACHER_GRID_DINO,
    TEACHER_GRID_TRAJ,
    TEACHER_HORIZON_P,
)
from mtwam.models.wan22.mtwam import MTWAM


class _Stub(MTWAM):
    def __init__(self):
        pass


def _teacher_sample(num_cameras=2):
    P = TEACHER_HORIZON_P
    return {
        "teacher_tracks": torch.zeros(
            num_cameras, P, TEACHER_GRID_TRAJ, 2, dtype=torch.float32
        ),
        "teacher_track_vis": torch.ones(
            num_cameras, P, TEACHER_GRID_TRAJ, dtype=torch.bool
        ),
        "teacher_dino": torch.zeros(
            num_cameras, P, TEACHER_GRID_DINO, 768, dtype=torch.float32
        ),
    }


def test_disabled_leaves_inputs_unchanged():
    m = _Stub()
    inputs = {"context": "sentinel"}
    out = m._attach_teacher_to_inputs(
        _teacher_sample(), inputs, device="cpu", dtype=torch.float32, enable=False
    )
    assert out is inputs
    assert not any((k.startswith("teacher_") for k in inputs))


def test_enabled_casts_floats_to_dtype_and_vis_to_bool():
    m = _Stub()
    inputs = {}
    m._attach_teacher_to_inputs(
        _teacher_sample(num_cameras=2),
        inputs,
        device="cpu",
        dtype=torch.bfloat16,
        enable=True,
    )
    P = TEACHER_HORIZON_P
    assert inputs["teacher_tracks"].dtype == torch.bfloat16
    assert inputs["teacher_dino"].dtype == torch.bfloat16
    assert inputs["teacher_track_vis"].dtype == torch.bool
    assert inputs["teacher_tracks"].shape == (2, P, TEACHER_GRID_TRAJ, 2)
    assert inputs["teacher_track_vis"].shape == (2, P, TEACHER_GRID_TRAJ)
    assert inputs["teacher_dino"].shape == (2, P, TEACHER_GRID_DINO, 768)


class _FakeCoTracker:
    def __call__(self, video, queries):
        bsz, T, n = (int(video.shape[0]), int(video.shape[1]), int(queries.shape[1]))
        return (torch.zeros(bsz, T, n, 2), torch.ones(bsz, T, n))


class _FakeDino:
    def __init__(self, n_dino):
        self.n_dino = n_dino

    def __call__(self, x, is_training):
        return {"x_norm_patchtokens": torch.zeros(int(x.shape[0]), self.n_dino, 768)}


def test_online_branch_computes_teacher_from_frames():
    m = _Stub()
    m.teacher_models = {
        "cotracker": _FakeCoTracker(),
        "dino": _FakeDino(TEACHER_GRID_DINO),
    }
    m.dynamic_branch_camera_layout = "horizontal"
    B, C, P = (2, 2, TEACHER_HORIZON_P)
    sample = {"teacher_frames": torch.rand(B, C, P + 1, 3, 224, 224)}
    inputs = {}
    m._attach_teacher_to_inputs(
        sample, inputs, device="cpu", dtype=torch.float32, enable=True
    )
    assert inputs["teacher_tracks"].shape == (B, C, P, TEACHER_GRID_TRAJ, 2)
    assert inputs["teacher_track_vis"].shape == (B, C, P, TEACHER_GRID_TRAJ)
    assert inputs["teacher_track_vis"].dtype == torch.bool
    assert inputs["teacher_dino"].shape == (B, C, P, TEACHER_GRID_DINO, 768)


def test_online_branch_without_teacher_models_raises():
    m = _Stub()
    m.teacher_models = None
    m.dynamic_branch_camera_layout = "horizontal"
    P = TEACHER_HORIZON_P
    sample = {"teacher_frames": torch.rand(1, 2, P + 1, 3, 224, 224)}
    with pytest.raises(RuntimeError):
        m._attach_teacher_to_inputs(
            sample, {}, device="cpu", dtype=torch.float32, enable=True
        )


def test_extract_teacher_batch_robotwin_geometry():
    from mtwam.datasets.lerobot.teacher_extract import extract_teacher_batch

    B, C, P = (2, 3, TEACHER_HORIZON_P)
    traj_grid_hw, dino_grid_hw, video_size = ((16, 20), (16, 20), (224, 280))
    n_traj, n_dino = (16 * 20, 16 * 20)
    frames = torch.rand(B, C, P + 1, 3, 224, 280)
    models = {"cotracker": _FakeCoTracker(), "dino": _FakeDino(n_dino)}
    tr, vi, di = extract_teacher_batch(
        frames, models, traj_grid_hw, video_size, dino_grid_hw, device="cpu"
    )
    assert tr.shape == (B, C, P, n_traj, 2)
    assert vi.shape == (B, C, P, n_traj) and vi.dtype == torch.bool
    assert di.shape == (B, C, P, n_dino, 768)


class _FakeCoTrackerRamp:
    def __call__(self, video, queries):
        bsz, T, n = (int(video.shape[0]), int(video.shape[1]), int(queries.shape[1]))
        t = torch.arange(T, dtype=torch.float32).view(1, T, 1, 1)
        vis = (torch.arange(T) % 2 == 0).float().view(1, T, 1).expand(bsz, T, n).clone()
        return (t.expand(bsz, T, n, 2).clone(), vis)


class _FakeDinoContent:
    def __init__(self, n_dino):
        self.n_dino = n_dino

    def __call__(self, x, is_training):
        m = x.mean(dim=(1, 2, 3))
        return {
            "x_norm_patchtokens": m.view(-1, 1, 1)
            .expand(int(x.shape[0]), self.n_dino, 768)
            .clone()
        }


def _stub_with_horizon(h):
    from types import SimpleNamespace

    m = _Stub()
    m.teacher_models = {
        "cotracker": _FakeCoTrackerRamp(),
        "dino": _FakeDinoContent(TEACHER_GRID_DINO),
    }
    m.dynamic_branch_camera_layout = "horizontal"
    if h is not None:
        m.mot = SimpleNamespace(
            dynamic_branch=SimpleNamespace(traj_head=SimpleNamespace(horizon=h))
        )
    return m


def test_online_full_clip_slices_last_h_same_source():
    B, C, P = (2, 2, TEACHER_HORIZON_P)
    sample = {"teacher_frames": torch.rand(B, C, P + 1, 3, 224, 224)}
    outs = {}
    for h in (2, 1):
        inputs = {}
        _stub_with_horizon(h)._attach_teacher_to_inputs(
            sample, inputs, device="cpu", dtype=torch.float32, enable=True
        )
        outs[h] = inputs
    assert outs[1]["teacher_tracks"].shape == (B, C, 1, TEACHER_GRID_TRAJ, 2)
    assert outs[1]["teacher_track_vis"].shape == (B, C, 1, TEACHER_GRID_TRAJ)
    assert outs[1]["teacher_dino"].shape == (B, C, 1, TEACHER_GRID_DINO, 768)
    assert torch.equal(outs[1]["teacher_tracks"], outs[2]["teacher_tracks"][:, :, -1:])
    assert torch.equal(
        outs[1]["teacher_track_vis"], outs[2]["teacher_track_vis"][:, :, -1:]
    )
    assert torch.equal(outs[1]["teacher_dino"], outs[2]["teacher_dino"][:, :, -1:])
    assert torch.allclose(
        outs[1]["teacher_tracks"],
        torch.full_like(outs[1]["teacher_tracks"], 2.0 / 16.0),
    )
    for k in ("teacher_tracks", "teacher_track_vis", "teacher_dino"):
        assert not torch.equal(outs[1][k], outs[2][k][:, :, :1]), (
            f"{k}: P axis not discriminated"
        )


def test_online_full_clip_noop_when_h_equals_pmax():
    B, C, P = (1, 2, TEACHER_HORIZON_P)
    sample = {"teacher_frames": torch.rand(B, C, P + 1, 3, 224, 224)}
    ref, with_branch = ({}, {})
    _stub_with_horizon(None)._attach_teacher_to_inputs(
        sample, ref, device="cpu", dtype=torch.float32, enable=True
    )
    _stub_with_horizon(P)._attach_teacher_to_inputs(
        sample, with_branch, device="cpu", dtype=torch.float32, enable=True
    )
    for k in ("teacher_tracks", "teacher_track_vis", "teacher_dino"):
        assert with_branch[k].shape[2] == P
        assert torch.equal(ref[k], with_branch[k])
