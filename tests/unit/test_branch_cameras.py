import torch
import pytest
from types import SimpleNamespace
from mtwam.models.wan22.branch_losses import split_branch_cameras, assemble_branch_loss
from mtwam.models.wan22.dynamic_branch import TrajMAEHead, TexMAEHead


def test_split_branch_cameras_horizontal_contiguous_halves():
    B, h, w, H, C = (1, 2, 4, 3, 2)
    hidden = torch.zeros(B, h * w, H)
    for r in range(h):
        for c in range(w):
            hidden[0, r * w + c, 0] = float(r)
            hidden[0, r * w + c, 1] = float(c)
    out = split_branch_cameras(hidden, (h, w), C)
    assert out.shape == (B * C, h * (w // C), H)
    cam0, cam1 = (out[0], out[1])
    assert set(cam0[:, 1].tolist()) == {0.0, 1.0}
    assert set(cam1[:, 1].tolist()) == {2.0, 3.0}
    assert set(cam0[:, 0].tolist()) == {0.0, 1.0}
    assert set(cam1[:, 0].tolist()) == {0.0, 1.0}


def test_split_rejects_indivisible_width():
    with pytest.raises(ValueError, match="divisible"):
        split_branch_cameras(torch.randn(1, 6, 3), (2, 3), 2)


def test_split_rejects_grid_mismatch():
    with pytest.raises(ValueError, match="St_f"):
        split_branch_cameras(torch.randn(1, 5, 3), (2, 4), 2)


def _tiny_branch(H, grid, P):
    return SimpleNamespace(
        traj_head=TrajMAEHead(in_dim=H, decoder_dim=8, grid=grid, horizon=P),
        tex_head=TexMAEHead(in_dim=H, decoder_dim=8, grid=grid, horizon=P),
    )


def test_assemble_branch_loss_c2_per_camera_shapes_and_finite():
    B, h, w, H, C, P, grid = (1, 2, 4, 8, 2, 2, 4)
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
    )
    assert set(d) >= {"loss_traj", "loss_tex"}
    assert float(total) > 0.0
    assert torch.isfinite(torch.tensor(float(total)))


def test_c2_requires_grid_hw():
    B, H, C, P, grid = (1, 8, 2, 2, 4)
    branch = _tiny_branch(H, grid, P)
    bh = {"traj": torch.randn(B, 8, H), "tex": torch.randn(B, 8, H)}
    teacher = {
        "tracks": torch.randn(B, C, P, grid, 2),
        "track_vis": torch.ones(B, C, P, grid, dtype=torch.bool),
        "dino": torch.nn.functional.normalize(torch.randn(B, C, P, grid, 768), dim=-1),
        "frame_pad": torch.zeros(B, P, dtype=torch.bool),
    }
    with pytest.raises(ValueError, match="grid_hw"):
        assemble_branch_loss(
            branch,
            bh,
            teacher,
            lambda_traj=0.1,
            lambda_tex=0.0,
            validity_mask=True,
            grid_hw=None,
        )


def test_c1_backward_compat_no_grid_hw():
    B, st_f, H, P, grid = (1, 5, 8, 2, 4)
    branch = _tiny_branch(H, grid, P)
    bh = {"traj": torch.randn(B, st_f, H), "tex": torch.randn(B, st_f, H)}
    teacher = {
        "tracks": torch.randn(B, 1, P, grid, 2),
        "track_vis": torch.ones(B, 1, P, grid, dtype=torch.bool),
        "dino": torch.nn.functional.normalize(torch.randn(B, 1, P, grid, 768), dim=-1),
        "frame_pad": torch.zeros(B, P, dtype=torch.bool),
    }
    total, d = assemble_branch_loss(
        branch, bh, teacher, lambda_traj=0.1, lambda_tex=0.01, validity_mask=True
    )
    assert set(d) >= {"loss_traj", "loss_tex"}
    assert float(total) > 0.0
