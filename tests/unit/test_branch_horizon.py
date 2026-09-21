import pytest
import torch
from mtwam.datasets.lerobot.teacher_cache import (
    TEACHER_HORIZON_P,
    prepare_teacher_frames_for_camera,
    teacher_cache_path,
)
from mtwam.models.wan22.branch_losses import derive_branch_frame_pad

VSI = list(range(0, 33, 4))


def _tagged_window(t_obs=33):
    w = torch.zeros(t_obs, 3, 32, 32)
    for i in range(t_obs):
        w[i] = i / 100.0
    return w


def test_horizon2_parity_selects_raw_0_16_32():
    reps = prepare_teacher_frames_for_camera(
        _tagged_window(), VSI, (32, 32), horizon_p=2
    )
    assert reps.shape[0] == 3
    for k, raw_idx in enumerate((0, 16, 32)):
        assert torch.allclose(
            reps[k], torch.full_like(reps[k], raw_idx / 100.0), atol=1e-06
        )


def test_horizon2_default_equals_explicit():
    a = prepare_teacher_frames_for_camera(_tagged_window(), VSI, (32, 32))
    b = prepare_teacher_frames_for_camera(_tagged_window(), VSI, (32, 32), horizon_p=2)
    assert torch.allclose(a, b, atol=1e-06)


def test_horizon1_lastN_keeps_terminal_f2_not_near_f1():
    reps = prepare_teacher_frames_for_camera(
        _tagged_window(), VSI, (32, 32), horizon_p=1
    )
    assert reps.shape[0] == 2
    assert torch.allclose(reps[0], torch.full_like(reps[0], 0 / 100.0), atol=1e-06)
    assert torch.allclose(reps[1], torch.full_like(reps[1], 32 / 100.0), atol=1e-06)
    assert not torch.allclose(reps[1], torch.full_like(reps[1], 16 / 100.0), atol=1e-06)


def test_horizon_p_guard_bounds_1_to_pmax():
    with pytest.raises(ValueError, match="horizon_p must be in"):
        prepare_teacher_frames_for_camera(_tagged_window(), VSI, (32, 32), horizon_p=0)
    with pytest.raises(ValueError, match="horizon_p must be in"):
        prepare_teacher_frames_for_camera(
            _tagged_window(), VSI, (32, 32), horizon_p=TEACHER_HORIZON_P + 1
        )


def test_cache_path_p_suffix_separates_p1_p2():
    assert teacher_cache_path("/x", "key", horizon_p=2).endswith(".teacher_P2.pt")
    assert teacher_cache_path("/x", "key", horizon_p=1).endswith(".teacher_P1.pt")
    assert teacher_cache_path("/x", "key").endswith(".teacher_P2.pt")


def test_frame_pad_lastN_picks_terminal_latent():
    tf = 4
    img = torch.zeros(1, 1 + 2 * tf, dtype=torch.bool)
    img[:, 1 : 1 + tf] = False
    img[:, 1 + tf : 1 + 2 * tf] = True
    fp1 = derive_branch_frame_pad(img, 1, 1, tf, torch.device("cpu"))
    assert fp1.shape == (1, 1)
    assert bool(fp1[0, 0]) is True
    fp2 = derive_branch_frame_pad(img, 1, 2, tf, torch.device("cpu"))
    assert fp2.shape == (1, 2)
    assert fp2.tolist() == [[False, True]]


def test_frame_pad_none_is_all_false():
    fp = derive_branch_frame_pad(None, 3, 1, 4, torch.device("cpu"))
    assert fp.shape == (3, 1)
    assert not fp.any()


import inspect as _inspect


def test_from_wan22_signature_has_horizon_param():
    from mtwam.models.wan22.mtwam import MTWAM

    assert (
        "dynamic_branch_horizon"
        in _inspect.signature(MTWAM.from_wan22_pretrained).parameters
    )


def test_tiny_mtwam_horizon1_head_shape():
    from factories import make_tiny_mtwam

    br = make_tiny_mtwam(
        enable_dynamic_branch=True, dynamic_branch_horizon=1
    ).mot.dynamic_branch
    assert br.traj_head.horizon == 1 and br.tex_head.horizon == 1
    assert tuple(br.traj_head.frame_emb.shape) == (1, 1, 8)
    assert tuple(br.tex_head.frame_emb.shape) == (1, 1, 8)


def test_tiny_mtwam_horizon2_default_head_shape_parity():
    from factories import make_tiny_mtwam

    br = make_tiny_mtwam(enable_dynamic_branch=True).mot.dynamic_branch
    assert br.traj_head.horizon == 2 and br.tex_head.horizon == 2
    assert tuple(br.traj_head.frame_emb.shape) == (2, 1, 8)
    assert tuple(br.tex_head.frame_emb.shape) == (2, 1, 8)


def test_dynamic_branch_horizon_below_1_raises():
    from factories import make_tiny_mtwam

    with pytest.raises(ValueError, match="horizon must be >= 1"):
        make_tiny_mtwam(enable_dynamic_branch=True, dynamic_branch_horizon=0)


import os as _os
from omegaconf import OmegaConf as _OmegaConf

_YAML = _os.path.join(
    _os.path.dirname(__file__), "..", "..", "configs", "model", "mtwam.yaml"
)


def test_yaml_horizon_default_two():
    cfg = _OmegaConf.load(_YAML)
    assert cfg.get("dynamic_branch_horizon", "MISSING") == 2


def test_create_mtwam_signature_has_horizon():
    from mtwam.runtime import create_mtwam

    assert "dynamic_branch_horizon" in set(_inspect.signature(create_mtwam).parameters)


def test_create_mtwam_threads_horizon_to_from_wan22(monkeypatch):
    import mtwam.runtime as rt
    from mtwam.models.wan22.mtwam import MTWAM

    captured = {}
    monkeypatch.setattr(
        MTWAM,
        "from_wan22_pretrained",
        staticmethod(lambda **kw: captured.update(kw) or "SENTINEL"),
    )
    rt.create_mtwam(
        model_id="x",
        tokenizer_model_id="y",
        video_dit_config={"text_dim": 4096},
        action_scheduler={
            "train_shift": 5.0,
            "infer_shift": 5.0,
            "num_train_timesteps": 1000,
        },
        enable_dynamic_branch=True,
        dynamic_branch_horizon=1,
        device="cpu",
    )
    assert captured["dynamic_branch_horizon"] == 1


def test_validate_branch_horizon_states():
    from mtwam.runtime import validate_branch_horizon

    validate_branch_horizon(False, 2, 0)
    validate_branch_horizon(False, 2, 16)
    validate_branch_horizon(True, 2, 16)
    with pytest.raises(ValueError, match="enable_dynamic_branch"):
        validate_branch_horizon(False, 1, 0)
    with pytest.raises(ValueError, match="K=0 only"):
        validate_branch_horizon(True, 1, 16)
    validate_branch_horizon(True, 1, 0)


def test_behavioral_horizon_mismatch_caught():
    from mtwam.models.wan22._load_guards import check_branch_arch_consistency

    train = {"enable_dynamic_branch": True, "dynamic_branch_horizon": 1}
    errs = check_branch_arch_consistency(train, {"enable_dynamic_branch": True})
    assert (
        len(errs) == 1
        and "dynamic_branch_horizon" in errs[0]
        and ("configuration mismatch" in errs[0])
    )
    assert check_branch_arch_consistency(train, dict(train)) == []
    assert check_branch_arch_consistency({}, {}) == []


def test_horizon_head_form_mismatch_load_raises():
    from factories import make_tiny_mtwam

    sd_p1 = make_tiny_mtwam(
        enable_dynamic_branch=True, dynamic_branch_horizon=1
    ).mot.dynamic_branch.state_dict()
    p2_branch = make_tiny_mtwam(
        enable_dynamic_branch=True, dynamic_branch_horizon=2
    ).mot.dynamic_branch
    with pytest.raises(RuntimeError, match="size mismatch|shape"):
        p2_branch.load_state_dict(sd_p1)


def _mk_horizon_cfg(model_h=None, data_h=None, num_frames=None):
    m = {} if model_h is None else {"dynamic_branch_horizon": model_h}
    d = {} if data_h is None else {"dynamic_branch_horizon": data_h}
    if num_frames is not None:
        d["num_frames"] = num_frames
    cfg = _OmegaConf.create({"model": m, "data": {"train": d}})
    _OmegaConf.set_struct(cfg, True)
    return cfg


def test_reconcile_horizon_both_default_is_2():
    from mtwam.runtime import reconcile_branch_horizon

    cfg = _mk_horizon_cfg()
    assert reconcile_branch_horizon(cfg) == 2
    assert (
        int(cfg.model.dynamic_branch_horizon) == 2
        and int(cfg.data.train.dynamic_branch_horizon) == 2
    )


def test_reconcile_horizon_model_only_propagates_to_data():
    from mtwam.runtime import reconcile_branch_horizon

    cfg = _mk_horizon_cfg(model_h=1)
    assert reconcile_branch_horizon(cfg) == 1
    assert int(cfg.data.train.dynamic_branch_horizon) == 1


def test_reconcile_horizon_data_only_propagates_to_model():
    from mtwam.runtime import reconcile_branch_horizon

    cfg = _mk_horizon_cfg(data_h=1)
    assert reconcile_branch_horizon(cfg) == 1
    assert int(cfg.model.dynamic_branch_horizon) == 1


def test_reconcile_horizon_both_equal_ok():
    from mtwam.runtime import reconcile_branch_horizon

    assert reconcile_branch_horizon(_mk_horizon_cfg(model_h=1, data_h=1)) == 1


def test_reconcile_horizon_both_differ_raises():
    from mtwam.runtime import reconcile_branch_horizon

    with pytest.raises(ValueError, match="drift"):
        reconcile_branch_horizon(_mk_horizon_cfg(model_h=1, data_h=3))


def test_kreject_bypass_closed_via_effective_horizon():
    from mtwam.runtime import reconcile_branch_horizon, validate_branch_horizon

    cfg = _mk_horizon_cfg(data_h=1)
    h_eff = reconcile_branch_horizon(cfg)
    with pytest.raises(ValueError, match="K=0 only"):
        validate_branch_horizon(True, h_eff, 16)


def test_reconcile_horizon_writes_val_node_when_present():
    from mtwam.runtime import reconcile_branch_horizon

    cfg = _OmegaConf.create(
        {"model": {"dynamic_branch_horizon": 1}, "data": {"train": {}, "val": {}}}
    )
    _OmegaConf.set_struct(cfg, True)
    assert reconcile_branch_horizon(cfg) == 1
    assert int(cfg.data.val.dynamic_branch_horizon) == 1


def test_reconcile_horizon_no_val_node_still_ok():
    from mtwam.runtime import reconcile_branch_horizon

    cfg = _mk_horizon_cfg(model_h=1)
    assert reconcile_branch_horizon(cfg) == 1


def test_window_p_max_from_cfg_truth_table():
    from mtwam.runtime import window_p_max_from_cfg
    from omegaconf import OmegaConf as _OC

    assert window_p_max_from_cfg(_OC.create({})) == 2
    assert window_p_max_from_cfg(_OC.create({"num_frames": 33})) == 2
    assert window_p_max_from_cfg(_OC.create({"num_frames": 17})) == 1
    assert (
        window_p_max_from_cfg(
            _OC.create({"num_frames": 49, "num_extra_ref_frames": 16})
        )
        == 2
    )


def test_reconcile_q_window_default_derives_1():
    from mtwam.runtime import reconcile_branch_horizon

    cfg = _mk_horizon_cfg(num_frames=17)
    assert reconcile_branch_horizon(cfg) == 1
    assert (
        int(cfg.model.dynamic_branch_horizon) == 1
        and int(cfg.data.train.dynamic_branch_horizon) == 1
    )


def test_reconcile_33_window_default_still_2():
    from mtwam.runtime import reconcile_branch_horizon

    assert reconcile_branch_horizon(_mk_horizon_cfg(num_frames=33)) == 2


def test_reconcile_q_window_explicit_h2_out_of_range_raises():
    from mtwam.runtime import reconcile_branch_horizon

    with pytest.raises(ValueError, match="out of range \\[1, 1\\]"):
        reconcile_branch_horizon(_mk_horizon_cfg(model_h=2, num_frames=17))


def test_validate_branch_horizon_q_window_states():
    from mtwam.runtime import validate_branch_horizon

    validate_branch_horizon(True, 1, 0, p_max=1)
    validate_branch_horizon(False, 1, 0, p_max=1)
    with pytest.raises(ValueError, match="enable_dynamic_branch"):
        validate_branch_horizon(False, 1, 0, p_max=2)


def test_create_mtwam_threads_teacher_p_max_to_from_wan22(monkeypatch):
    import mtwam.runtime as rt
    from mtwam.models.wan22.mtwam import MTWAM

    captured = {}
    monkeypatch.setattr(
        MTWAM,
        "from_wan22_pretrained",
        staticmethod(lambda **kw: captured.update(kw) or "SENTINEL"),
    )
    rt.create_mtwam(
        model_id="x",
        tokenizer_model_id="y",
        video_dit_config={"text_dim": 4096},
        action_scheduler={
            "train_shift": 5.0,
            "infer_shift": 5.0,
            "num_train_timesteps": 1000,
        },
        enable_dynamic_branch=True,
        dynamic_branch_horizon=1,
        dynamic_branch_teacher_p_max=1,
        device="cpu",
    )
    assert captured["dynamic_branch_teacher_p_max"] == 1


def test_factory_injects_teacher_p_max_from_window():
    from mtwam.runtime import model_factory_extra_kwargs

    extra = model_factory_extra_kwargs(
        "mtwam.runtime.create_mtwam", "horizontal", 0, teacher_p_max=1
    )
    assert extra["dynamic_branch_teacher_p_max"] == 1
    extra2 = model_factory_extra_kwargs("mtwam.runtime.create_mtwam", "horizontal", 0)
    assert "dynamic_branch_teacher_p_max" not in extra2


def test_attach_teacher_short_slice_raises_when_head_exceeds_window():
    import torch
    from types import SimpleNamespace
    from mtwam.models.wan22.mtwam import MTWAM

    class _Stub(MTWAM):
        def __init__(self):
            self.teacher_models = {"cotracker": _Ramp(), "dino": _Zeros()}
            self.dynamic_branch_camera_layout = "horizontal"
            self.mot = SimpleNamespace(
                dynamic_branch=SimpleNamespace(traj_head=SimpleNamespace(horizon=2))
            )

    class _Ramp:
        def __call__(self, video, queries):
            b, t, n = (int(video.shape[0]), int(video.shape[1]), int(queries.shape[1]))
            return (torch.zeros(b, t, n, 2), torch.ones(b, t, n))

    class _Zeros:
        def __call__(self, x, is_training):
            return {"x_norm_patchtokens": torch.zeros(int(x.shape[0]), 256, 768)}

    m = _Stub()
    sample = {"teacher_frames": torch.rand(1, 2, 2, 3, 224, 224)}
    with pytest.raises(ValueError, match="too short for this head"):
        m._attach_teacher_to_inputs(
            sample, {}, device="cpu", dtype=torch.float32, enable=True
        )


def test_infer_action_requires_action_horizon():
    from factories import make_tiny_mtwam

    m = make_tiny_mtwam()
    with pytest.raises(ValueError, match="requires action_horizon"):
        m.infer_action(prompt=None, action_horizon=None)


def test_infer_action_checks_action_horizon_matches_training_chunk():
    from factories import make_tiny_mtwam

    m = make_tiny_mtwam()
    m.num_frames = 17
    with pytest.raises(ValueError, match="training chunk"):
        m.infer_action(prompt=None, action_horizon=32)
    m.num_frames = 49
    m.num_extra_ref_frames = 16
    with pytest.raises(ValueError, match="num_extra_ref_frames=0 only"):
        m.infer_action(prompt=None, action_horizon=16)
    m2 = make_tiny_mtwam()
    assert m2.num_frames is None
    with pytest.raises(Exception) as ei:
        m2.infer_action(prompt=None, action_horizon=16)
    assert "training chunk" not in str(ei.value)


def test_traj_loss_rejects_teacher_P_vs_head_h_mismatch():
    import torch
    from mtwam.models.wan22.branch_losses import compute_traj_loss

    pred = torch.zeros(1, 1, 1, 4, 2)
    target = torch.zeros(1, 1, 2, 4, 2)
    vis = torch.ones(1, 1, 2, 4, dtype=torch.bool)
    fp = torch.zeros(1, 1, dtype=torch.bool)
    with pytest.raises(ValueError, match="P/h mismatch"):
        compute_traj_loss(pred, target, vis, fp, validity_mask=True)


def test_tex_loss_rejects_teacher_P_vs_head_h_mismatch():
    import torch
    from mtwam.models.wan22.branch_losses import compute_tex_loss

    pred = torch.zeros(1, 1, 1, 4, 8)
    target = torch.zeros(1, 1, 2, 4, 8)
    fp = torch.zeros(1, 1, dtype=torch.bool)
    with pytest.raises(ValueError, match="P/h mismatch"):
        compute_tex_loss(pred, target, fp)


def test_losses_pass_when_P_equals_h():
    import torch
    from mtwam.models.wan22.branch_losses import compute_traj_loss, compute_tex_loss

    pred = torch.randn(1, 2, 1, 4, 2)
    target = torch.randn(1, 2, 1, 4, 2)
    vis = torch.ones(1, 2, 1, 4, dtype=torch.bool)
    fp = torch.zeros(1, 1, dtype=torch.bool)
    assert torch.isfinite(compute_traj_loss(pred, target, vis, fp, validity_mask=True))
    p2 = torch.randn(1, 2, 1, 4, 8)
    t2 = torch.randn(1, 2, 1, 4, 8)
    assert torch.isfinite(compute_tex_loss(p2, t2, fp))


def test_resolve_horizon_explicit_value_passthrough():
    from mtwam.runtime import resolve_branch_horizon

    assert resolve_branch_horizon(1, True, 2) == 1
    assert resolve_branch_horizon("2", True, None) == 2
    assert resolve_branch_horizon(2, False, None) == 2


def test_resolve_horizon_none_is_fine_when_branch_off():
    from mtwam.runtime import resolve_branch_horizon

    assert resolve_branch_horizon(None, False, None) is None
    assert resolve_branch_horizon(None, False, 2) is None


def test_resolve_horizon_derives_from_window_p_max():
    from mtwam.runtime import resolve_branch_horizon

    assert resolve_branch_horizon(None, True, 2) == 2
    assert resolve_branch_horizon(None, True, 1) == 1


def test_resolve_horizon_refuses_to_guess_when_window_unknown():
    from mtwam.runtime import resolve_branch_horizon

    with pytest.raises(ValueError) as ei:
        resolve_branch_horizon(None, True, None)
    msg = str(ei.value)
    assert "reconcile_branch_horizon" in msg
    assert "dynamic_branch_horizon" in msg


def test_resolve_horizon_never_silently_returns_legacy_2():
    from mtwam.runtime import resolve_branch_horizon

    try:
        got = resolve_branch_horizon(None, True, None)
    except ValueError:
        got = "raised"
    assert got == "raised", f"must refuse, not guess -- got {got!r}"


def test_eval_resolves_horizon_from_training_config(tmp_path):
    pytest.importorskip(
        "libero", reason="LIBERO sim package not installed on this host"
    )
    from experiments.libero.eval_libero_single import _resolve_eval_branch_horizon

    run = tmp_path / "run"
    (run / "checkpoints" / "weights").mkdir(parents=True)
    ckpt = run / "checkpoints" / "weights" / "step_000010.pt"
    ckpt.write_bytes(b"")
    _OmegaConf.save(
        _OmegaConf.create({"model": {"dynamic_branch_horizon": 1}}), run / "config.yaml"
    )
    cfg = _OmegaConf.create(
        {
            "model": {"dynamic_branch_horizon": None},
            "data": {"train": {"num_frames": 33}},
        }
    )
    _resolve_eval_branch_horizon(cfg, str(ckpt))
    assert int(cfg.model.dynamic_branch_horizon) == 1
