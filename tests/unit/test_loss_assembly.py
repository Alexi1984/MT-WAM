import torch
from factories import make_tiny_mtwam


def test_tiny_mtwam_constructs_with_branch_heads_in_params():
    m = make_tiny_mtwam(enable_dynamic_branch=True)
    assert m.loss_lambda_traj == 0.0 and m.loss_lambda_tex == 0.0
    br = m.mot.dynamic_branch
    assert br.traj_head is not None and br.tex_head is not None
    ids = {id(p) for p in m.parameters()}
    assert all((id(p) in ids for p in br.traj_head.parameters()))
    assert all((id(p) in ids for p in br.tex_head.parameters()))


def test_assemble_branch_loss_keys_and_lambda_zero():
    from mtwam.models.wan22.branch_losses import assemble_branch_loss

    m = make_tiny_mtwam(enable_dynamic_branch=True)
    br = m.mot.dynamic_branch
    B, st_f, H = (1, 5, 16)
    bh = {"traj": torch.randn(B, st_f, H), "tex": torch.randn(B, st_f, H)}
    teacher = {
        "tracks": torch.randn(B, 1, 2, 196, 2),
        "track_vis": torch.ones(B, 1, 2, 196, dtype=torch.bool),
        "dino": torch.nn.functional.normalize(torch.randn(B, 1, 2, 256, 768), dim=-1),
        "frame_pad": torch.zeros(B, 2, dtype=torch.bool),
    }
    total0, d0 = assemble_branch_loss(
        br, bh, teacher, lambda_traj=0.0, lambda_tex=0.0, validity_mask=True
    )
    assert set(d0) >= {"loss_traj", "loss_tex"}
    assert float(total0) == 0.0
    totalp, _ = assemble_branch_loss(
        br, bh, teacher, lambda_traj=0.1, lambda_tex=0.01, validity_mask=True
    )
    assert float(totalp) > 0.0
