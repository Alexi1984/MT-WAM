import torch
from factories import (
    make_tiny_mot,
    make_tiny_mtwam,
    make_tiny_video_expert,
    make_tiny_action_expert,
    DummyVAE,
)
from mtwam.trainer import Wan22Trainer


def test_branch_params_under_mot():
    mot = make_tiny_mot(enable_dynamic_branch=True, dynamic_branch_num_layers=1)
    mot_ids = {id(p) for p in mot.parameters()}
    assert all((id(p) in mot_ids for p in mot.dynamic_branch.parameters()))
    assert (
        mot.dynamic_branch.role_t.requires_grad
        and mot.dynamic_branch.role_s.requires_grad
    )


def test_branch_and_heads_under_dit():
    model = make_tiny_mtwam(enable_dynamic_branch=True, dynamic_branch_num_layers=1)
    assert model.dit is model.mot
    branch = model.mot.dynamic_branch
    assert branch.traj_head is not None and branch.tex_head is not None
    dit_ids = {id(p) for p in model.dit.parameters()}
    assert all((id(p) in dit_ids for p in branch.parameters()))


def test_freeze_then_dit_only_unfreezes_branch():
    model = make_tiny_mtwam(enable_dynamic_branch=True, dynamic_branch_num_layers=1)
    model.requires_grad_(False)
    assert not any((p.requires_grad for p in model.mot.dynamic_branch.parameters()))
    Wan22Trainer._apply_dit_only_train_mode(model)
    assert all((p.requires_grad for p in model.mot.dynamic_branch.parameters()))


def test_expert_order_unchanged_with_branch():
    mot = make_tiny_mot(enable_dynamic_branch=True, dynamic_branch_num_layers=1)
    assert mot.expert_order == ["video", "action"]


def test_checkpoint_saves_branch_keys(tmp_path):
    model = make_tiny_mtwam(enable_dynamic_branch=True, dynamic_branch_num_layers=1)
    ckpt = tmp_path / "tiny.pt"
    model.save_checkpoint(str(ckpt))
    payload = torch.load(str(ckpt), map_location="cpu")
    assert "mot" in payload
    keys = list(payload["mot"].keys())
    assert "dynamic_branch.role_t" in keys and "dynamic_branch.role_s" in keys
    assert any((k.startswith("dynamic_branch.blocks.") for k in keys))
    assert any((k.startswith("dynamic_branch.traj_head.") for k in keys))
    assert any((k.startswith("dynamic_branch.tex_head.") for k in keys))


def test_checkpoint_load_restores_branch(tmp_path):
    model = make_tiny_mtwam(enable_dynamic_branch=True, dynamic_branch_num_layers=1)
    ckpt = tmp_path / "tiny.pt"
    model.save_checkpoint(str(ckpt))
    orig = model.mot.dynamic_branch.role_t.detach().clone()
    with torch.no_grad():
        model.mot.dynamic_branch.role_t.add_(1.0)
    assert not torch.equal(model.mot.dynamic_branch.role_t, orig)
    model.load_checkpoint(str(ckpt))
    assert torch.equal(model.mot.dynamic_branch.role_t, orig)


def test_branch_construction_is_rng_neutral():
    torch.manual_seed(0)
    make_tiny_mtwam(enable_dynamic_branch=False)
    after_off = torch.rand(8)
    torch.manual_seed(0)
    make_tiny_mtwam(enable_dynamic_branch=True)
    after_on = torch.rand(8)
    assert torch.equal(after_off, after_on), (
        "branch construction shifted the global RNG -> downstream random-init modules would differ with vs without the branch (fork_rng regression)"
    )


def test_branch_params_match_model_dtype():
    from mtwam.models.wan22.mtwam import MTWAM
    from mtwam.models.wan22.dynamic_branch import DynamicBranch
    from mtwam.models.wan22.mot import MoT

    dt = torch.bfloat16
    video = make_tiny_video_expert().to(dt)
    action = make_tiny_action_expert().to(dt)
    mot = MoT(
        mixtures={"video": video, "action": action}, mot_checkpoint_mixed_attn=False
    )
    with torch.random.fork_rng(devices=[]):
        mot.dynamic_branch = DynamicBranch(
            video_expert=video,
            num_layers=1,
            build_heads=True,
            grid_traj=196,
            grid_tex=256,
            horizon=2,
        )
    model = MTWAM(
        video_expert=video,
        action_expert=action,
        mot=mot,
        vae=DummyVAE(),
        text_dim=32,
        device="cpu",
        torch_dtype=dt,
    )
    br = model.mot.dynamic_branch
    assert br.role_t.dtype == dt and br.role_s.dtype == dt
    assert all((p.dtype == dt for p in br.blocks.parameters()))
    assert all((p.dtype == dt for p in br.traj_head.parameters()))
    assert all((p.dtype == dt for p in br.tex_head.parameters()))
