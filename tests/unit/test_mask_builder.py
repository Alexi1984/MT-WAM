import torch
from types import SimpleNamespace
from mtwam.models.wan22.mtwam import MTWAM


def _mask(Sv, Sa, St, tpf):
    stub = SimpleNamespace(
        video_expert=SimpleNamespace(
            build_video_to_video_mask=lambda video_seq_len,
            video_tokens_per_frame,
            device,
            num_ref_latents=1: torch.ones(Sv, Sv, dtype=torch.bool)
        )
    )
    return MTWAM._build_mot_attention_mask(
        stub,
        video_seq_len=Sv,
        action_seq_len=Sa,
        video_tokens_per_frame=tpf,
        device=torch.device("cpu"),
        traj_seq_len=St,
    )


def test_square_and_legacy_when_st_zero():
    m = _mask(4, 3, 0, 2)
    assert m.shape == (7, 7) and m.dtype == torch.bool


def test_blocks_with_branch():
    Sv, Sa, tpf = (4, 3, 2)
    St_f, St = (2, 4)
    m = _mask(Sv, Sa, St, tpf)
    assert m.shape == (Sv + Sa + St,) * 2
    ff = min(tpf, Sv)
    a0, a1 = (Sv, Sv + Sa)
    t0, t1 = (Sv + Sa, Sv + Sa + St_f)
    s0, s1 = (t1, Sv + Sa + St)
    assert m[a0:a1, :ff].all() and (not m[a0:a1, ff:Sv].any())
    assert m[a0:a1, a0:a1].all()
    assert m[a0:a1, t0:t1].all()
    assert not m[a0:a1, s0:s1].any()
    assert m[t0:t1, :ff].all() and (not m[t0:t1, ff:Sv].any())
    assert m[t0:t1, t0:t1].all()
    assert not m[t0:t1, s0:s1].any() and (not m[s0:s1, t0:t1].any())
    assert not m[t0:t1, a0:a1].any()
    assert not m[:Sv, t0:s1].any()
    assert not m[:Sv, a0:a1].any()
