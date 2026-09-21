import pytest
import torch
from mtwam.models.wan22.branch_losses import derive_branch_frame_pad


def test_frame_pad_valid_libero_like():
    B, tf, horizon = (2, 4, 2)
    image_is_pad = torch.zeros(B, 1 + horizon * tf, dtype=torch.bool)
    image_is_pad[1, -tf:] = True
    fp = derive_branch_frame_pad(image_is_pad, B, horizon, tf, torch.device("cpu"))
    assert fp.shape == (B, horizon)
    assert fp.dtype == torch.bool
    assert fp[0].tolist() == [False, False]
    assert fp[1].tolist() == [False, True]


def test_frame_pad_none_returns_zeros():
    fp = derive_branch_frame_pad(None, 3, 2, 4, torch.device("cpu"))
    assert fp.shape == (3, 2)
    assert fp.dtype == torch.bool
    assert not fp.any()


def test_frame_pad_raises_when_fewer_than_horizon_future_frames():
    B, tf, horizon = (2, 4, 2)
    image_is_pad = torch.zeros(B, 1 + 1 * tf, dtype=torch.bool)
    with pytest.raises(ValueError, match="horizon"):
        derive_branch_frame_pad(image_is_pad, B, horizon, tf, torch.device("cpu"))
