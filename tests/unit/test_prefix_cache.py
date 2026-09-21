import torch
import pytest
from types import SimpleNamespace
from factories import make_tiny_mot, build_tiny_payload, assemble_mot_kwargs
from mtwam.models.wan22.mtwam import MTWAM


def _joint_mask(mot, meta):
    stub = SimpleNamespace(video_expert=mot.mixtures["video"])
    return MTWAM._build_mot_attention_mask(
        stub,
        video_seq_len=meta["Sv"],
        action_seq_len=meta["Sa"],
        video_tokens_per_frame=meta["tpf"],
        device=torch.device("cpu"),
        traj_seq_len=0,
    )


def _prefill(mot, vpre, mask, Sv):
    return mot.prefill_video_cache(
        prefix_tokens=vpre["tokens"],
        prefix_freqs=vpre["freqs"],
        prefix_t_mod=vpre["t_mod"],
        prefix_context_payload={
            "context": vpre["context"],
            "mask": vpre["context_mask"],
        },
        prefix_attention_mask=mask[:Sv, :Sv],
    )


def _fwd_action(mot, apre, cache, mask, Sv):
    return mot.forward_action_with_video_cache(
        action_tokens=apre["tokens"],
        action_freqs=apre["freqs"],
        action_t_mod=apre["t_mod"],
        action_context_payload={
            "context": apre["context"],
            "mask": apre["context_mask"],
        },
        prefix_kv_cache=cache,
        attention_mask=mask,
        prefix_seq_len=Sv,
    )


def test_prefix_cache_shapes():
    mot = make_tiny_mot()
    mot.eval()
    vpre, apre, meta = build_tiny_payload(mot)
    Sv = meta["Sv"]
    mask = _joint_mask(mot, meta)
    cache = _prefill(mot, vpre, mask, Sv)
    assert len(cache) == mot.num_layers
    assert cache[0]["k"].shape[1] == Sv and cache[0]["v"].shape[1] == Sv
    assert _fwd_action(mot, apre, cache, mask, Sv).shape == apre["tokens"].shape


def test_cache_path_equals_joint_forward():
    mot = make_tiny_mot()
    mot.eval()
    vpre, apre, meta = build_tiny_payload(mot)
    Sv = meta["Sv"]
    mask = _joint_mask(mot, meta)
    ref = mot.forward(**assemble_mot_kwargs(mot, vpre, apre, mask))["action"]
    cache = _prefill(mot, vpre, mask, Sv)
    out = _fwd_action(mot, apre, cache, mask, Sv)
    assert torch.equal(out, ref)


def _joint_mask_traj(mot, meta, traj_seq_len, num_ref_latents=1, dual_f0=False):
    stub = SimpleNamespace(video_expert=mot.mixtures["video"])
    return MTWAM._build_mot_attention_mask(
        stub,
        video_seq_len=meta["Sv"],
        action_seq_len=meta["Sa"],
        video_tokens_per_frame=meta["tpf"],
        device=torch.device("cpu"),
        traj_seq_len=traj_seq_len,
        num_ref_latents=num_ref_latents,
        dual_f0=dual_f0,
    )


@pytest.mark.parametrize("mot_layers, m", [(4, 1), (4, 3), (12, 10)])
def test_cache_path_with_branch_equals_joint_forward(mot_layers, m):
    mot = make_tiny_mot(
        enable_dynamic_branch=True, dynamic_branch_num_layers=m, num_layers=mot_layers
    )
    mot.eval()
    vpre, apre, meta = build_tiny_payload(mot)
    Sv, Sa, tpf = (meta["Sv"], meta["Sa"], meta["tpf"])
    St = 2 * tpf
    ff = tpf
    mask = _joint_mask_traj(mot, meta, St)
    payload = {"enable": True, "first_frame_tokens": ff}
    ref = mot.forward(
        **assemble_mot_kwargs(mot, vpre, apre, mask), dynamic_branch_payload=payload
    )["action"]
    non_action = list(range(Sv)) + list(range(Sv + Sa, Sv + Sa + St))
    idx = torch.tensor(non_action)
    prefix_mask = mask[idx][:, idx]
    cache = mot.prefill_video_cache(
        prefix_tokens=vpre["tokens"],
        prefix_freqs=vpre["freqs"],
        prefix_t_mod=vpre["t_mod"],
        prefix_context_payload={
            "context": vpre["context"],
            "mask": vpre["context_mask"],
        },
        prefix_attention_mask=prefix_mask,
        dynamic_branch_payload=payload,
    )
    tail = [c for c in cache if c["k"].shape[1] == Sv + St]
    trunk = [c for c in cache if c["k"].shape[1] == Sv]
    assert len(tail) == m and len(trunk) == mot_layers - m
    assert cache[-1]["k"].shape[1] == Sv + St
    out = mot.forward_action_with_video_cache(
        action_tokens=apre["tokens"],
        action_freqs=apre["freqs"],
        action_t_mod=apre["t_mod"],
        action_context_payload={
            "context": apre["context"],
            "mask": apre["context_mask"],
        },
        prefix_kv_cache=cache,
        attention_mask=mask,
        prefix_seq_len=Sv,
    )
    assert torch.allclose(out, ref, atol=1e-06, rtol=1e-05)


@pytest.mark.parametrize("mot_layers, m", [(4, 1), (4, 3)])
def test_cache_path_with_dual_f0_equals_joint_forward(mot_layers, m):
    mot = make_tiny_mot(
        enable_dynamic_branch=True, dynamic_branch_num_layers=m, num_layers=mot_layers
    )
    mot.mixtures["video"].video_attention_mask_mode = "first_frame_causal"
    mot.eval()
    vpre, apre, meta = build_tiny_payload(mot, T=3)
    Sv, Sa, ff = (meta["Sv"], meta["Sa"], meta["tpf"])
    St = 2 * ff
    payload = {
        "enable": True,
        "first_frame_tokens": ff,
        "num_ref_latents": 2,
        "dual_f0": True,
    }
    mask = _joint_mask_traj(mot, meta, St, num_ref_latents=2, dual_f0=True)
    ref = mot.forward(
        **assemble_mot_kwargs(mot, vpre, apre, mask), dynamic_branch_payload=payload
    )["action"]
    non_action = list(range(Sv)) + list(range(Sv + Sa, Sv + Sa + St))
    idx = torch.tensor(non_action)
    prefix_mask = mask[idx][:, idx]
    cache = mot.prefill_video_cache(
        prefix_tokens=vpre["tokens"],
        prefix_freqs=vpre["freqs"],
        prefix_t_mod=vpre["t_mod"],
        prefix_context_payload={
            "context": vpre["context"],
            "mask": vpre["context_mask"],
        },
        prefix_attention_mask=prefix_mask,
        dynamic_branch_payload=payload,
    )
    tail = [c for c in cache if c["k"].shape[1] == Sv + St]
    assert len(tail) == m and cache[-1]["k"].shape[1] == Sv + St
    assert all((c["k"].shape[1] != Sv + 2 * 2 * ff for c in cache))
    out = mot.forward_action_with_video_cache(
        action_tokens=apre["tokens"],
        action_freqs=apre["freqs"],
        action_t_mod=apre["t_mod"],
        action_context_payload={
            "context": apre["context"],
            "mask": apre["context_mask"],
        },
        prefix_kv_cache=cache,
        attention_mask=mask,
        prefix_seq_len=Sv,
    )
    assert torch.allclose(out, ref, atol=1e-06, rtol=1e-05)


def test_dual_f0_absent_key_and_false_are_byte_identical():
    mot = make_tiny_mot(enable_dynamic_branch=True, dynamic_branch_num_layers=1)
    mot.mixtures["video"].video_attention_mask_mode = "first_frame_causal"
    mot.eval()
    vpre, apre, meta = build_tiny_payload(mot, T=3)
    ff = meta["tpf"]
    St = 2 * 2 * ff
    mask = _joint_mask_traj(mot, meta, St, num_ref_latents=2, dual_f0=False)
    kwargs = assemble_mot_kwargs(mot, vpre, apre, mask)
    out_absent = mot.forward(
        **kwargs,
        dynamic_branch_payload={
            "enable": True,
            "first_frame_tokens": ff,
            "num_ref_latents": 2,
        },
    )
    out_false = mot.forward(
        **kwargs,
        dynamic_branch_payload={
            "enable": True,
            "first_frame_tokens": ff,
            "num_ref_latents": 2,
            "dual_f0": False,
        },
    )
    assert torch.equal(out_absent["action"], out_false["action"])
    assert torch.equal(out_absent["video"], out_false["video"])
