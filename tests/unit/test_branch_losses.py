import torch
from mtwam.models.wan22.branch_losses import compute_traj_loss, compute_tex_loss


def test_traj_validity_on_masks_invisible_point():
    pred = torch.zeros(1, 1, 2, 4, 2)
    tgt = torch.zeros(1, 1, 2, 4, 2)
    vis = torch.ones(1, 1, 2, 4, dtype=torch.bool)
    vis[..., 0] = False
    frame_pad = torch.zeros(1, 2, dtype=torch.bool)
    pred[..., 0, :] = 99.0
    on = compute_traj_loss(pred, tgt, vis, frame_pad, validity_mask=True)
    assert torch.isclose(on, torch.tensor(0.0))
    off = compute_traj_loss(pred, tgt, vis, frame_pad, validity_mask=False)
    assert off > 0


def test_frame_pad_always_masks_both_modes():
    pred = torch.zeros(1, 1, 2, 4, 2)
    tgt = torch.zeros(1, 1, 2, 4, 2)
    vis = torch.ones(1, 1, 2, 4, dtype=torch.bool)
    frame_pad = torch.tensor([[False, True]])
    pred[:, :, 1] = 99.0
    for fl in (True, False):
        assert torch.isclose(
            compute_traj_loss(pred, tgt, vis, frame_pad, validity_mask=fl),
            torch.tensor(0.0),
        )


def test_tex_cosine_extremes_and_per_camera_average():
    a = torch.nn.functional.normalize(torch.randn(1, 2, 2, 256, 768), dim=-1)
    frame_pad = torch.zeros(1, 2, dtype=torch.bool)
    assert torch.isclose(
        compute_tex_loss(a, a.clone(), frame_pad), torch.tensor(0.0), atol=1e-05
    )
    assert torch.isclose(
        compute_tex_loss(a, -a, frame_pad), torch.tensor(2.0), atol=0.0001
    )
