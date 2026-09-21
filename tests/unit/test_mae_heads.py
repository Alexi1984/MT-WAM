import torch
from mtwam.models.wan22.dynamic_branch import TrajMAEHead, TexMAEHead


def test_traj_head_shape():
    head = TrajMAEHead(in_dim=16, decoder_dim=8, grid=196, horizon=2)
    out = head(torch.randn(2, 5, 16))
    assert out.shape == (2, 2, 196, 2)


def test_tex_head_shape():
    head = TexMAEHead(in_dim=16, decoder_dim=8, grid=256, horizon=2, feat_dim=768)
    out = head(torch.randn(2, 5, 16))
    assert out.shape == (2, 2, 256, 768)
