import torch
import pytest
from mtwam.models.wan22.dynamic_branch import (
    TrajMAEHead,
    TexMAEHead,
    _get_2d_sincos_pos_embed,
)


def test_sincos_square_backward_compat():
    for gs in (14, 16):
        p1 = _get_2d_sincos_pos_embed(64, gs)
        p2 = _get_2d_sincos_pos_embed(64, gs, gs)
        assert p1.shape == (gs * gs, 64)
        assert (p1 == p2).all()


def test_sincos_rectangular_shape_and_finite():
    p = _get_2d_sincos_pos_embed(64, 16, 20)
    assert p.shape == (16 * 20, 64)
    assert bool((p == p).all())


def test_head_square_int_grid_unchanged():
    h = TrajMAEHead(in_dim=8, decoder_dim=16, grid=196, horizon=2)
    assert h.grid == 196 and h.mask_pos.shape == (1, 196, 16)
    out = h(torch.randn(2, 49, 8))
    assert out.shape == (2, 2, 196, 2)


def test_head_int_vs_square_tuple_byte_equal():
    a = TrajMAEHead(in_dim=8, decoder_dim=16, grid=196, horizon=2)
    b = TrajMAEHead(in_dim=8, decoder_dim=16, grid=(14, 14), horizon=2)
    assert a.mask_pos.shape == b.mask_pos.shape == (1, 196, 16)
    assert torch.equal(a.mask_pos, b.mask_pos)


def test_head_rectangular_grid_and_st_f_decoupled():
    h = TexMAEHead(in_dim=8, decoder_dim=16, grid=(16, 20), horizon=2)
    assert h.grid == 320 and h.mask_pos.shape == (1, 320, 16)
    for st_f in (80, 20):
        out = h(torch.randn(1, st_f, 8))
        assert out.shape == (1, 2, 320, 768)


def test_head_bad_square_int_grid_rejected():
    with pytest.raises(ValueError, match="perfect square"):
        TrajMAEHead(in_dim=8, decoder_dim=16, grid=200, horizon=2)
