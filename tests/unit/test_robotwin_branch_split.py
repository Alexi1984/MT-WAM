import torch
import pytest
from types import SimpleNamespace
from mtwam.models.wan22.branch_losses import (
    split_branch_cameras_robotwin,
    _run_head_per_camera,
    assemble_branch_loss,
)
from mtwam.models.wan22.dynamic_branch import TrajMAEHead, TexMAEHead


def _encode_grid(h, w, H):
    hidden = torch.zeros(1, h * w, H)
    for r in range(h):
        for c in range(w):
            hidden[0, r * w + c, 0] = float(r)
            hidden[0, r * w + c, 1] = float(c)
    return hidden


def _pairs(t):
    return {(int(t[0, j, 0]), int(t[0, j, 1])) for j in range(t.shape[1])}


def test_split_robotwin_carves_per_p0_1_truth_table():
    h, w, H = (12, 10, 3)
    high, left, right = split_branch_cameras_robotwin(_encode_grid(h, w, H), (h, w))
    assert (
        high.shape == (1, 80, H)
        and left.shape == (1, 20, H)
        and (right.shape == (1, 20, H))
    )
    assert _pairs(high) == {(r, c) for r in range(0, 8) for c in range(0, 10)}
    assert _pairs(left) == {(r, c) for r in range(8, 12) for c in range(0, 5)}
    assert _pairs(right) == {(r, c) for r in range(8, 12) for c in range(5, 10)}


def test_split_robotwin_rejects_bad_grid():
    with pytest.raises(ValueError, match="divisible by 3"):
        split_branch_cameras_robotwin(torch.randn(1, 7 * 10, 3), (7, 10))
    with pytest.raises(ValueError, match="by 2"):
        split_branch_cameras_robotwin(torch.randn(1, 12 * 9, 3), (12, 9))
    with pytest.raises(ValueError, match="St_f"):
        split_branch_cameras_robotwin(torch.randn(1, 5, 3), (12, 10))


def _tiny_branch(H, grid, P):
    return SimpleNamespace(
        traj_head=TrajMAEHead(in_dim=H, decoder_dim=8, grid=grid, horizon=P),
        tex_head=TexMAEHead(in_dim=H, decoder_dim=8, grid=grid, horizon=P),
    )


def test_run_head_per_camera_robotwin_stacks_three_cameras():
    B, h, w, H, P, grid = (1, 12, 10, 8, 2, 9)
    head = TrajMAEHead(in_dim=H, decoder_dim=8, grid=grid, horizon=P)
    out = _run_head_per_camera(
        head, torch.randn(B, h * w, H), 3, (h, w), camera_layout="robotwin"
    )
    assert out.shape == (B, 3, P, grid, 2)


def test_assemble_branch_loss_robotwin_finite():
    B, h, w, H, C, P, grid = (1, 12, 10, 8, 3, 2, 9)
    branch = _tiny_branch(H, grid, P)
    bh = {"traj": torch.randn(B, h * w, H), "tex": torch.randn(B, h * w, H)}
    teacher = {
        "tracks": torch.randn(B, C, P, grid, 2),
        "track_vis": torch.ones(B, C, P, grid, dtype=torch.bool),
        "dino": torch.nn.functional.normalize(torch.randn(B, C, P, grid, 768), dim=-1),
        "frame_pad": torch.zeros(B, P, dtype=torch.bool),
    }
    total, d = assemble_branch_loss(
        branch,
        bh,
        teacher,
        lambda_traj=0.1,
        lambda_tex=0.01,
        validity_mask=True,
        grid_hw=(h, w),
        camera_layout="robotwin",
    )
    assert set(d) >= {"loss_traj", "loss_tex"}
    assert float(total) > 0.0 and torch.isfinite(torch.tensor(float(total)))


def test_unknown_camera_layout_raises():
    head = TrajMAEHead(in_dim=8, decoder_dim=8, grid=9, horizon=2)
    with pytest.raises(ValueError, match="unknown camera_layout"):
        _run_head_per_camera(
            head, torch.randn(1, 120, 8), 3, (12, 10), camera_layout="diagonal"
        )


def test_default_horizontal_path_rejects_robotwin_grid():
    head = TrajMAEHead(in_dim=8, decoder_dim=8, grid=9, horizon=2)
    with pytest.raises(ValueError, match="divisible"):
        _run_head_per_camera(head, torch.randn(1, 120, 8), 3, (12, 10))
