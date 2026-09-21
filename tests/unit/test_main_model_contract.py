import pytest
from mtwam.models.wan22 import mtwam as model_module
from mtwam.runtime import create_mtwam

_CONFIG = {"text_dim": 4096, "video_attention_mask_mode": "first_frame_causal"}
_SCHEDULER = {"train_shift": 5.0, "infer_shift": 5.0, "num_train_timesteps": 1000}


@pytest.mark.parametrize(
    "override",
    [
        {"enable_dynamic_branch": False},
        {"dynamic_branch_dual_f0": False},
        {"dynamic_branch_ffn_moe": False},
        {"dynamic_branch_read_text": True},
        {"dynamic_branch_ffn_moe_routing": "gated"},
        {"dynamic_branch_num_layers": 8},
        {"dynamic_branch_horizon": 1},
        {"dynamic_branch_teacher_p_max": 1},
        {"dynamic_branch_camera_layout": "vertical"},
        {"dynamic_branch_teacher_geometry": "robotwin_dense"},
        {"num_extra_ref_frames": 16},
    ],
)
def test_unsupported_topology_is_rejected_before_asset_loading(monkeypatch, override):
    calls = []
    monkeypatch.setattr(
        model_module,
        "load_wan22_ti2v_5b_components",
        lambda **kwargs: calls.append(kwargs),
    )
    with pytest.raises(ValueError, match="MT-WAM"):
        create_mtwam(
            model_id="unused",
            tokenizer_model_id="unused",
            video_dit_config=_CONFIG,
            action_scheduler=_SCHEDULER,
            device="cpu",
            **override,
        )
    assert calls == []


def test_factory_defaults_select_main_model(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        model_module.MTWAM,
        "from_wan22_pretrained",
        lambda **kwargs: captured.update(kwargs),
    )
    create_mtwam(
        model_id="unused",
        tokenizer_model_id="unused",
        video_dit_config=_CONFIG,
        action_scheduler=_SCHEDULER,
        device="cpu",
    )
    assert captured["enable_dynamic_branch"] is True
    assert captured["dynamic_branch_dual_f0"] is True
    assert captured["dynamic_branch_ffn_moe"] is True
    assert captured["dynamic_branch_ffn_moe_routing"] == "role"
    assert captured["dynamic_branch_num_layers"] == 10
    assert captured["dynamic_branch_horizon"] == 2
    assert captured["loss_lambda_traj"] == 0.5
    assert captured["loss_lambda_tex"] == 0.25


def test_history_mutation_is_rejected_in_train_and_infer():
    from factories import make_tiny_mtwam

    model = make_tiny_mtwam()
    model.num_extra_ref_frames = 16
    with pytest.raises(ValueError, match="num_extra_ref_frames=0 only"):
        model.build_inputs({})
    with pytest.raises(ValueError, match="num_extra_ref_frames=0 only"):
        model.infer_action(prompt=None, action_horizon=32)
