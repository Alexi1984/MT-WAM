import os
from types import SimpleNamespace
import pytest
import torch
from omegaconf import OmegaConf
from factories import (
    assemble_mot_kwargs,
    build_tiny_payload,
    make_tiny_mot,
    make_tiny_video_expert,
)
from mtwam.models.wan22._load_guards import BEHAVIORAL_BRANCH_KEYS
from mtwam.models.wan22.dynamic_branch import DynamicBranch, RoleMoEFFN
from mtwam.models.wan22.mtwam import MTWAM
from mtwam.models.wan22.mot import MoT

_YAML = os.path.join(
    os.path.dirname(__file__), "..", "..", "configs", "model", "mtwam.yaml"
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


def test_depth_default_is_ten_and_yaml_matches_behavioral_key():
    import inspect
    import mtwam.runtime as runtime

    cfg = OmegaConf.load(_YAML)
    assert cfg.dynamic_branch_num_layers == 10
    assert BEHAVIORAL_BRANCH_KEYS["dynamic_branch_num_layers"] == 10
    assert (
        inspect.signature(runtime.create_mtwam)
        .parameters["dynamic_branch_num_layers"]
        .default
        == 10
    )


def test_depth_blocks_count_and_tail_hot_start():
    video = make_tiny_video_expert()
    N = len(video.blocks)
    for m in range(1, N + 1):
        branch = DynamicBranch(video_expert=video, num_layers=m)
        assert len(branch.blocks) == m
        assert branch.num_layers == m
        for i, blk in enumerate(branch.blocks):
            donor_sd = video.blocks[N - m + i].state_dict()
            blk_sd = blk.state_dict()
            assert set(donor_sd) == set(blk_sd)
            for k in donor_sd:
                assert torch.equal(donor_sd[k], blk_sd[k])


@pytest.mark.parametrize("bad_m", [0, -1, 5])
def test_depth_out_of_range_fails_loud(bad_m):
    video = make_tiny_video_expert()
    with pytest.raises(ValueError, match="dynamic_branch_num_layers"):
        DynamicBranch(video_expert=video, num_layers=bad_m)


def test_depth_role_moe_wraps_every_block():
    mot = make_tiny_mot(
        enable_dynamic_branch=True, dynamic_branch_num_layers=3, ffn_moe=True
    )
    branch = mot.dynamic_branch
    assert len(branch.blocks) == 3
    for blk in branch.blocks:
        assert isinstance(blk.ffn, RoleMoEFFN)


@pytest.mark.parametrize("m", [1, 2, 4])
def test_depth_forward_forks_exactly_the_last_m_layers(monkeypatch, m):
    mot = make_tiny_mot(enable_dynamic_branch=True, dynamic_branch_num_layers=m)
    mot.eval()
    vpre, apre, meta = build_tiny_payload(mot)
    mask = _mask(mot, meta, 2 * meta["tpf"])
    calls = []
    original = MoT._build_expert_attention_io

    def _spy(self, *, expert, block, x, freqs, t_mod):
        calls.append((expert is getattr(self, "dynamic_branch", None), block))
        return original(self, expert=expert, block=block, x=x, freqs=freqs, t_mod=t_mod)

    monkeypatch.setattr(MoT, "_build_expert_attention_io", _spy)
    mot.forward(
        **assemble_mot_kwargs(mot, vpre, apre, mask),
        dynamic_branch_payload={"enable": True, "first_frame_tokens": meta["tpf"]},
    )
    n_experts = len(mot.expert_order)
    branch_layers, branch_blocks, layer, i = ([], [], 0, 0)
    while i < len(calls):
        for _ in range(n_experts):
            assert calls[i][0] is False
            i += 1
        if i < len(calls) and calls[i][0]:
            branch_layers.append(layer)
            branch_blocks.append(calls[i][1])
            i += 1
        layer += 1
    N = mot.num_layers
    assert layer == N
    assert branch_layers == list(range(N - m, N))
    assert branch_blocks == list(mot.dynamic_branch.blocks)


def test_depth_mask_is_depth_independent():
    masks = []
    for m in (1, 3):
        mot = make_tiny_mot(enable_dynamic_branch=True, dynamic_branch_num_layers=m)
        vpre, apre, meta = build_tiny_payload(mot)
        masks.append(_mask(mot, meta, 2 * meta["tpf"]))
    assert torch.equal(masks[0], masks[1])
