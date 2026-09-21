import pytest
import torch
from factories import make_tiny_video_expert, make_tiny_mismatched_expert
from mtwam.models.wan22.dynamic_branch import DynamicBranch


def test_rejects_geometry_mismatch():
    video = make_tiny_video_expert()
    donor = make_tiny_mismatched_expert()
    with pytest.raises(ValueError):
        DynamicBranch(video_expert=video, num_layers=2, _donor_blocks=donor.blocks)


def test_hotstart_matches_trunk_tail():
    video = make_tiny_video_expert()
    M, N = (2, len(video.blocks))
    branch = DynamicBranch(video_expert=video, num_layers=M)
    for i in range(M):
        trunk_sd = video.blocks[N - M + i].state_dict()
        branch_sd = branch.blocks[i].state_dict()
        assert set(branch_sd) == set(trunk_sd)
        for k, v in trunk_sd.items():
            assert torch.equal(branch_sd[k], v)


def test_role_embeddings_unequal_params():
    branch = DynamicBranch(video_expert=make_tiny_video_expert(), num_layers=2)
    assert isinstance(branch.role_t, torch.nn.Parameter)
    assert isinstance(branch.role_s, torch.nn.Parameter)
    assert branch.role_t.shape == branch.role_s.shape
    assert not torch.allclose(branch.role_t, branch.role_s)
