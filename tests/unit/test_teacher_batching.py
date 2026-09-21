import os
import pytest
import torch
from mtwam.datasets.lerobot.teacher_extract import (
    _extract_teacher_batch_loop,
    extract_teacher_batch,
)

_GRID = (2, 2)
_IMG = (32, 32)


class _FakeCoTracker:
    def __call__(self, video, queries=None):
        n, t = (video.shape[0], video.shape[1])
        npt = queries.shape[1]
        base = queries[:, :, 1:]
        drift = video.mean(dim=(2, 3, 4))
        offs = torch.arange(npt, dtype=torch.float32).view(1, 1, npt, 1)
        tracks = base.unsqueeze(1) + drift.view(n, t, 1, 1) + offs
        vis = drift.view(n, t, 1).expand(n, t, npt) % 2 < 1
        return (tracks, vis)


class _FakeDino:
    def __call__(self, x, is_training=True):
        m = x.shape[0]
        sig = x.mean(dim=(1, 2, 3))
        toks = sig.view(m, 1, 1) + torch.arange(4, dtype=torch.float32).view(1, 4, 1)
        return {"x_norm_patchtokens": toks.expand(m, 4, 768).contiguous()}


def _frames(B=2, C=2, P=2):
    torch.manual_seed(0)
    f = torch.rand(B, C, P + 1, 3, *_IMG)
    for b in range(B):
        for c in range(C):
            f[b, c] += 10.0 * (b * C + c)
    return f.clamp_max(50.0) / 50.0


@pytest.fixture()
def models():
    return {"cotracker": _FakeCoTracker(), "dino": _FakeDino()}


def test_batched_equals_loop_rearrangement(models):
    frames = _frames(B=3, C=2)
    kw = dict(traj_grid_hw=_GRID, img_size=_IMG, dino_grid_hw=_GRID, device="cpu")
    t_new, v_new, d_new = extract_teacher_batch(frames, models, **kw)
    t_old, v_old, d_old = _extract_teacher_batch_loop(frames, models, **kw)
    assert torch.allclose(t_new, t_old, atol=1e-06)
    assert torch.equal(v_new.cpu(), v_old)
    assert torch.allclose(d_new, d_old, atol=1e-06)
    assert t_new.shape == (3, 2, 2, 4, 2) and d_new.shape == (3, 2, 2, 4, 768)
    assert v_new.dtype == torch.bool


def test_batched_degenerate_b1_c1(models):
    frames = _frames(B=1, C=1)
    kw = dict(traj_grid_hw=_GRID, img_size=_IMG, dino_grid_hw=_GRID, device="cpu")
    t_new, v_new, d_new = extract_teacher_batch(frames, models, **kw)
    t_old, v_old, d_old = _extract_teacher_batch_loop(frames, models, **kw)
    assert torch.allclose(t_new, t_old, atol=1e-06)
    assert torch.equal(v_new.cpu(), v_old)
    assert torch.allclose(d_new, d_old, atol=1e-06)


def test_batched_chunked_matches_unchunked(models, monkeypatch):
    frames = _frames(B=2, C=2)
    kw = dict(traj_grid_hw=_GRID, img_size=_IMG, dino_grid_hw=_GRID, device="cpu")
    t_full, v_full, d_full = extract_teacher_batch(frames, models, **kw)
    monkeypatch.setenv("MTWAM_TEACHER_BATCH_CHUNK", "3")
    t_chunk, v_chunk, d_chunk = extract_teacher_batch(frames, models, **kw)
    assert torch.allclose(t_full, t_chunk, atol=1e-06)
    assert torch.equal(v_full, v_chunk)
    assert torch.allclose(d_full, d_chunk, atol=1e-06)


def test_batched_longer_strip_p4(models):
    frames = _frames(B=2, C=1, P=4)
    kw = dict(traj_grid_hw=_GRID, img_size=_IMG, dino_grid_hw=_GRID, device="cpu")
    t_new, v_new, d_new = extract_teacher_batch(frames, models, **kw)
    t_old, v_old, d_old = _extract_teacher_batch_loop(frames, models, **kw)
    assert t_new.shape == (2, 1, 4, 4, 2)
    assert torch.allclose(t_new, t_old, atol=1e-06)
    assert torch.allclose(d_new, d_old, atol=1e-06)
