import torch
import torch.nn as nn
import pytest
from types import SimpleNamespace
from factories import make_tiny_mot, build_tiny_payload, assemble_mot_kwargs
from mtwam.models.wan22.mtwam import MTWAM
from mtwam.models.wan22.dynamic_branch import RoleMoEFFN


def _donor_ffn(hidden=16, ffn_dim=32):
    return nn.Sequential(
        nn.Linear(hidden, ffn_dim),
        nn.GELU(approximate="tanh"),
        nn.Linear(ffn_dim, hidden),
    )


def _mask(mot, meta, traj_seq_len):
    stub = SimpleNamespace(video_expert=mot.mixtures["video"])
    return MTWAM._build_mot_attention_mask(
        stub,
        video_seq_len=meta["Sv"],
        action_seq_len=meta["Sa"],
        video_tokens_per_frame=meta["tpf"],
        device=torch.device("cpu"),
        traj_seq_len=traj_seq_len,
    )


def test_rolemoe_odd_length_raises():
    moe = RoleMoEFFN(_donor_ffn())
    with pytest.raises(ValueError):
        moe(torch.randn(1, 3, 16))


def test_rolemoe_shape_preserved():
    moe = RoleMoEFFN(_donor_ffn())
    assert moe(torch.randn(2, 4, 16)).shape == (2, 4, 16)


def test_rolemoe_init_equals_shared_ffn():
    torch.manual_seed(0)
    donor = _donor_ffn()
    moe = RoleMoEFFN(donor)
    x = torch.randn(2, 4, 16)
    for key, value in donor.state_dict().items():
        assert torch.equal(moe.ffn_t.state_dict()[key], value)
        assert torch.equal(moe.ffn_s.state_dict()[key], value)
    torch.testing.assert_close(moe(x), donor(x), atol=1e-06, rtol=1e-05)


def test_rolemoe_gradient_isolation():
    moe = RoleMoEFFN(_donor_ffn())
    moe(torch.randn(1, 4, 16))[:, :2].sum().backward()
    assert all((p.grad is not None for p in moe.ffn_t.parameters()))
    assert any((torch.count_nonzero(p.grad) > 0 for p in moe.ffn_t.parameters()))
    assert all(
        (
            p.grad is None or torch.count_nonzero(p.grad) == 0
            for p in moe.ffn_s.parameters()
        )
    )


def test_rolemoe_perturb_one_expert_only_changes_its_half():
    moe = RoleMoEFFN(_donor_ffn())
    x = torch.randn(1, 4, 16)
    out0 = moe(x)
    with torch.no_grad():
        moe.ffn_s[0].weight.add_(1.0)
    out1 = moe(x)
    assert torch.equal(out0[:, :2], out1[:, :2])
    assert not torch.equal(out0[:, 2:], out1[:, 2:])


def test_branch_ffn_moe_off_keeps_sequential():
    mot = make_tiny_mot(
        enable_dynamic_branch=True, dynamic_branch_num_layers=2, ffn_moe=False
    )
    for blk in mot.dynamic_branch.blocks:
        assert isinstance(blk.ffn, nn.Sequential)
        assert not isinstance(blk.ffn, RoleMoEFFN)


def test_branch_ffn_moe_on_replaces_with_rolemoe():
    mot = make_tiny_mot(
        enable_dynamic_branch=True, dynamic_branch_num_layers=2, ffn_moe=True
    )
    for blk in mot.dynamic_branch.blocks:
        assert isinstance(blk.ffn, RoleMoEFFN)


def test_branch_ffn_moe_params_under_mot():
    mot = make_tiny_mot(
        enable_dynamic_branch=True, dynamic_branch_num_layers=2, ffn_moe=True
    )
    mot_ids = {id(p) for p in mot.parameters()}
    blk0 = mot.dynamic_branch.blocks[0].ffn
    assert all((id(p) in mot_ids for p in blk0.ffn_t.parameters()))
    assert all((id(p) in mot_ids for p in blk0.ffn_s.parameters()))


def _run_forward(ffn_moe):
    torch.manual_seed(0)
    mot = make_tiny_mot(
        enable_dynamic_branch=True,
        dynamic_branch_num_layers=3,
        num_layers=4,
        ffn_moe=ffn_moe,
    )
    mot.eval()
    vpre, apre, meta = build_tiny_payload(mot)
    St = 2 * meta["tpf"]
    out = mot.forward(
        **assemble_mot_kwargs(mot, vpre, apre, _mask(mot, meta, St)),
        dynamic_branch_payload={"enable": True, "first_frame_tokens": meta["tpf"]},
    )
    return (out, mot._last_branch_hidden)


def test_branch_ffn_moe_forward_byte_identical_at_init():
    out_off, h_off = _run_forward(False)
    out_on, h_on = _run_forward(True)
    assert torch.equal(out_off["video"], out_on["video"])
    assert torch.equal(out_off["action"], out_on["action"])
    assert torch.equal(h_off["traj"], h_on["traj"])
    assert torch.equal(h_off["tex"], h_on["tex"])


def _run_prefill(ffn_moe):
    torch.manual_seed(0)
    mot = make_tiny_mot(
        enable_dynamic_branch=True,
        dynamic_branch_num_layers=3,
        num_layers=4,
        ffn_moe=ffn_moe,
    )
    mot.eval()
    vpre, apre, meta = build_tiny_payload(mot)
    Sv, Sa, tpf = (meta["Sv"], meta["Sa"], meta["tpf"])
    St = 2 * tpf
    mask = _mask(mot, meta, St)
    non_action = list(range(Sv)) + list(range(Sv + Sa, Sv + Sa + St))
    idx = torch.tensor(non_action)
    prefix_mask = mask[idx][:, idx]
    return mot.prefill_video_cache(
        prefix_tokens=vpre["tokens"],
        prefix_freqs=vpre["freqs"],
        prefix_t_mod=vpre["t_mod"],
        prefix_context_payload={
            "context": vpre["context"],
            "mask": vpre["context_mask"],
        },
        prefix_attention_mask=prefix_mask,
        dynamic_branch_payload={"enable": True, "first_frame_tokens": tpf},
    )


def test_branch_ffn_moe_prefill_byte_identical_at_init():
    cache_off, cache_on = (_run_prefill(False), _run_prefill(True))
    assert len(cache_off) == len(cache_on)
    for c_off, c_on in zip(cache_off, cache_on):
        assert torch.equal(c_off["k"], c_on["k"])
        assert torch.equal(c_off["v"], c_on["v"])


def test_rolemoe_invalid_routing_raises():
    with pytest.raises(ValueError):
        RoleMoEFFN(_donor_ffn(), routing="bogus")


def test_rolemoe_default_routing_is_role():
    donor = _donor_ffn()
    x = torch.randn(2, 4, 16)
    assert torch.equal(RoleMoEFFN(donor)(x), RoleMoEFFN(donor, routing="role")(x))


def test_branch_ffn_moe_routing_defaults_role():
    mot = make_tiny_mot(
        enable_dynamic_branch=True, dynamic_branch_num_layers=2, ffn_moe=True
    )
    for blk in mot.dynamic_branch.blocks:
        assert blk.ffn.routing == "role"


def test_mot_role_moe_text_mask():
    torch.manual_seed(0)
    mot = make_tiny_mot(
        enable_dynamic_branch=True,
        dynamic_branch_num_layers=2,
        ffn_moe=True,
        ffn_moe_routing="role",
    )
    mot.dynamic_branch.read_text = True
    vpre, apre, meta = build_tiny_payload(mot, T=2)
    tpf = meta["tpf"]
    st = 2 * tpf
    L = vpre["context"].shape[1]
    ctx_payload = {
        "context": vpre["context"],
        "mask": torch.ones(1, meta["Sv"], L, dtype=torch.bool),
    }
    payload = {"enable": True, "first_frame_tokens": tpf}
    total = meta["Sv"] + meta["Sa"] + st
    kwargs = assemble_mot_kwargs(
        mot, vpre, apre, torch.ones(total, total, dtype=torch.bool)
    )
    kwargs["context_all"]["video"] = ctx_payload
    mot(**kwargs, dynamic_branch_payload=payload)
    bh = mot._last_branch_hidden
    assert bh["traj"].shape[1] == tpf and bh["tex"].shape[1] == tpf
    cache = mot.prefill_video_cache(
        prefix_tokens=vpre["tokens"],
        prefix_freqs=vpre["freqs"],
        prefix_t_mod=vpre["t_mod"],
        prefix_context_payload=ctx_payload,
        prefix_attention_mask=torch.ones(
            meta["Sv"] + st, meta["Sv"] + st, dtype=torch.bool
        ),
        dynamic_branch_payload=payload,
    )
    assert cache[-1]["k"].shape[1] == meta["Sv"] + st
