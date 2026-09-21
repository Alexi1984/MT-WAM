import torch
import pytest
from types import SimpleNamespace
from factories import make_tiny_mot, build_tiny_payload
from mtwam.models.wan22.mtwam import MTWAM


def _mask(mot, Sv, Sa, tpf, St, nrl, dual_f0):
    stub = SimpleNamespace(video_expert=mot.mixtures["video"])
    return MTWAM._build_mot_attention_mask(
        stub,
        video_seq_len=Sv,
        action_seq_len=Sa,
        video_tokens_per_frame=tpf,
        device=torch.device("cpu"),
        traj_seq_len=St,
        num_ref_latents=nrl,
        dual_f0=dual_f0,
    )


def _tiny_geometry():
    mot = make_tiny_mot(enable_dynamic_branch=True, dynamic_branch_num_layers=1)
    mot.mixtures["video"].video_attention_mask_mode = "first_frame_causal"
    mot.eval()
    vpre, apre, meta = build_tiny_payload(mot, T=3)
    assert meta["tpf"] == 2 and meta["Sv"] == 6
    return (mot, vpre, apre, meta)


def test_dual_f0_mask_truth_table():
    mot, _, _, meta = _tiny_geometry()
    Sv, Sa, ff = (meta["Sv"], meta["Sa"], meta["tpf"])
    St = 2 * ff
    m = _mask(mot, Sv, Sa, ff, St, nrl=2, dual_f0=True)
    assert m.shape == (Sv + Sa + St, Sv + Sa + St)
    s0, s1, nz = (slice(0, 2), slice(2, 4), slice(4, 6))
    ac, t, s = (slice(6, 9), slice(9, 11), slice(11, 13))
    assert m[s0, s0].all() and m[s0, s1].all() and m[s1, s0].all() and m[s1, s1].all()
    assert not m[s0, nz].any() and (not m[s1, nz].any())
    assert m[nz, s0].all() and m[nz, s1].all() and m[nz, nz].all()
    assert not m[:Sv, ac].any() and (not m[:Sv, t].any()) and (not m[:Sv, s].any())
    assert m[ac, s1].all() and m[ac, ac].all() and m[ac, t].all()
    assert not m[ac, s0].any() and (not m[ac, nz].any()) and (not m[ac, s].any())
    assert m[t, s0].all() and m[t, t].all()
    assert (
        not m[t, s1].any()
        and (not m[t, nz].any())
        and (not m[t, ac].any())
        and (not m[t, s].any())
    )
    assert m[s, s0].all() and m[s, s].all()
    assert (
        not m[s, s1].any()
        and (not m[s, nz].any())
        and (not m[s, ac].any())
        and (not m[s, t].any())
    )


def test_dual_f0_off_is_byte_identical_to_absent():
    mot, _, _, meta = _tiny_geometry()
    Sv, Sa, ff = (meta["Sv"], meta["Sa"], meta["tpf"])
    stub = SimpleNamespace(video_expert=mot.mixtures["video"])
    kwargs = dict(
        video_seq_len=Sv,
        action_seq_len=Sa,
        video_tokens_per_frame=ff,
        device=torch.device("cpu"),
        traj_seq_len=2 * 2 * ff,
        num_ref_latents=2,
    )
    m_absent = MTWAM._build_mot_attention_mask(stub, **kwargs)
    m_false = MTWAM._build_mot_attention_mask(stub, **kwargs, dual_f0=False)
    assert torch.equal(m_absent, m_false)


def test_dual_f0_traj_len_assertion_basis():
    mot, _, _, meta = _tiny_geometry()
    Sv, Sa, ff = (meta["Sv"], meta["Sa"], meta["tpf"])
    with pytest.raises(ValueError, match="branch length mismatch"):
        _mask(mot, Sv, Sa, ff, St=2 * 2 * ff, nrl=2, dual_f0=True)
    _mask(mot, Sv, Sa, ff, St=2 * ff, nrl=2, dual_f0=True)
    with pytest.raises(ValueError, match="branch length mismatch"):
        _mask(mot, Sv, Sa, ff, St=2 * ff, nrl=2, dual_f0=False)


def test_dual_f0_requires_nrl2():
    mot, _, _, meta = _tiny_geometry()
    Sv, Sa, ff = (meta["Sv"], meta["Sa"], meta["tpf"])
    for bad_nrl in (1, 3):
        with pytest.raises(ValueError, match="num_ref_latents == 2"):
            _mask(mot, Sv, Sa, ff, St=2 * ff, nrl=bad_nrl, dual_f0=True)


def test_video_mask_dual_f0_slot_carveout_expert_level():
    mot, _, _, meta = _tiny_geometry()
    expert = mot.mixtures["video"]
    Sv, ff = (meta["Sv"], meta["tpf"])
    base = expert.build_video_to_video_mask(
        video_seq_len=Sv,
        video_tokens_per_frame=ff,
        device=torch.device("cpu"),
        num_ref_latents=2,
    )
    carved = expert.build_video_to_video_mask(
        video_seq_len=Sv,
        video_tokens_per_frame=ff,
        device=torch.device("cpu"),
        num_ref_latents=2,
        dual_f0_branch_slot=True,
    )
    assert torch.equal(base, carved)
    with pytest.raises(ValueError, match="num_ref_latents == 2"):
        expert.build_video_to_video_mask(
            video_seq_len=Sv,
            video_tokens_per_frame=ff,
            device=torch.device("cpu"),
            num_ref_latents=1,
            dual_f0_branch_slot=True,
        )


def test_dual_f0_fork_takes_slot0_only():
    mot, vpre, apre, meta = _tiny_geometry()
    Sv, Sa, ff = (meta["Sv"], meta["Sa"], meta["tpf"])
    from factories import assemble_mot_kwargs

    m = _mask(mot, Sv, Sa, ff, St=2 * ff, nrl=2, dual_f0=True)
    payload = {
        "enable": True,
        "first_frame_tokens": ff,
        "num_ref_latents": 2,
        "dual_f0": True,
    }
    mot.forward(
        **assemble_mot_kwargs(mot, vpre, apre, m), dynamic_branch_payload=payload
    )
    assert mot._last_branch_hidden["traj"].shape[1] == ff
    assert mot._last_branch_hidden["tex"].shape[1] == ff
    m_off = _mask(mot, Sv, Sa, ff, St=2 * 2 * ff, nrl=2, dual_f0=False)
    payload_off = {"enable": True, "first_frame_tokens": ff, "num_ref_latents": 2}
    mot.forward(
        **assemble_mot_kwargs(mot, vpre, apre, m_off),
        dynamic_branch_payload=payload_off,
    )
    assert mot._last_branch_hidden["traj"].shape[1] == 2 * ff


def _fake_inputs(T=3, H=2, W=4, Sa=4, L=2, C=1, P=2, F_raw=9, pad_tail=False):
    image_is_pad = torch.zeros(1, F_raw, dtype=torch.bool)
    if pad_tail:
        image_is_pad[:, -4:] = True
    return {
        "input_latents": torch.randn(1, 48, T, H, W),
        "context": torch.randn(1, L, 32),
        "context_mask": None,
        "action": torch.randn(1, Sa, 7),
        "action_is_pad": torch.zeros(1, Sa, dtype=torch.bool),
        "image_is_pad": image_is_pad,
        "num_ref_latents": 1,
        "reference_latents": torch.randn(1, 48, 1, H, W),
        "fuse_vae_embedding_in_latents": True,
        "teacher_tracks": torch.randn(1, C, P, 196, 2),
        "teacher_track_vis": torch.ones(1, C, P, 196, dtype=torch.bool),
        "teacher_dino": torch.randn(1, C, P, 256, 768),
    }


def _spy_mask_calls(monkeypatch):
    from mtwam.models.wan22.mtwam import MTWAM as _FW

    calls = []
    orig = _FW._build_mot_attention_mask

    def spy(self, **kw):
        calls.append(kw)
        return orig(self, **kw)

    monkeypatch.setattr(_FW, "_build_mot_attention_mask", spy)
    return calls


def test_training_loss_dual_f0_wiring(monkeypatch):
    from factories import make_tiny_mtwam

    m = make_tiny_mtwam(enable_dynamic_branch=True)
    m.mot.mixtures["video"].video_attention_mask_mode = "first_frame_causal"
    m.dynamic_branch_dual_f0 = True
    m.loss_lambda_traj, m.loss_lambda_tex = (0.1, 0.01)
    inputs = _fake_inputs()
    monkeypatch.setattr(m, "build_inputs", lambda sample, tiled=False: inputs)
    calls = _spy_mask_calls(monkeypatch)
    torch.manual_seed(0)
    loss, loss_dict = m.training_loss({"unused": True})
    assert torch.isfinite(loss)
    assert loss_dict["loss_traj"] > 0 and loss_dict["loss_tex"] > 0
    assert m.mot._last_branch_hidden["traj"].shape[1] == 2
    (kw,) = calls
    assert kw["num_ref_latents"] == 2 and kw["dual_f0"] is True
    assert kw["traj_seq_len"] == 2 * 2 and kw["video_seq_len"] == 4 * 2


def test_training_loss_dual_f0_pad_alignment(monkeypatch):
    from factories import make_tiny_mtwam

    m = make_tiny_mtwam(enable_dynamic_branch=True)
    m.mot.mixtures["video"].video_attention_mask_mode = "first_frame_causal"
    m.dynamic_branch_dual_f0 = True
    m.loss_lambda_traj, m.loss_lambda_tex = (0.1, 0.01)
    inputs = _fake_inputs(pad_tail=True)
    monkeypatch.setattr(m, "build_inputs", lambda sample, tiled=False: inputs)
    torch.manual_seed(0)
    loss, _ = m.training_loss({"unused": True})
    assert torch.isfinite(loss)


def test_training_loss_dual_f0_k_guard(monkeypatch):
    from factories import make_tiny_mtwam

    m = make_tiny_mtwam(enable_dynamic_branch=True)
    m.dynamic_branch_dual_f0 = True
    m.loss_lambda_traj = 0.1
    inputs = _fake_inputs(T=4, F_raw=13)
    inputs["num_ref_latents"] = 2
    inputs["reference_latents"] = torch.randn(1, 48, 2, 2, 4)
    monkeypatch.setattr(m, "build_inputs", lambda sample, tiled=False: inputs)
    with pytest.raises(ValueError, match="K=0 only"):
        m.training_loss({"unused": True})


def test_training_loss_dual_f0_off_and_absent_attr_parity(monkeypatch):
    from factories import make_tiny_mtwam

    torch.manual_seed(7)
    m = make_tiny_mtwam(enable_dynamic_branch=True)
    m.mot.mixtures["video"].video_attention_mask_mode = "first_frame_causal"
    m.loss_lambda_traj, m.loss_lambda_tex = (0.1, 0.01)
    inputs = _fake_inputs()
    monkeypatch.setattr(m, "build_inputs", lambda sample, tiled=False: inputs)
    assert m.dynamic_branch_dual_f0 is False
    torch.manual_seed(1)
    loss_false, _ = m.training_loss({"unused": True})
    del m.dynamic_branch_dual_f0
    torch.manual_seed(1)
    loss_absent, _ = m.training_loss({"unused": True})
    assert torch.equal(loss_false, loss_absent)


def test_infer_action_dual_f0_wiring(monkeypatch):
    from factories import make_tiny_mtwam

    m = make_tiny_mtwam(enable_dynamic_branch=True)
    m.mot.mixtures["video"].video_attention_mask_mode = "first_frame_causal"
    m.dynamic_branch_dual_f0 = True
    fixed_latent = torch.randn(1, 48, 1, 2, 4)
    monkeypatch.setattr(
        m,
        "_encode_input_image_latents_tensor",
        lambda input_image, tiled=False: fixed_latent,
    )
    seen = {}
    orig_prefill = m.mot.prefill_video_cache

    def spy_prefill(**kw):
        seen["prefix_len"] = int(kw["prefix_tokens"].shape[1])
        seen["payload"] = kw["dynamic_branch_payload"]
        return orig_prefill(**kw)

    monkeypatch.setattr(m.mot, "prefill_video_cache", spy_prefill)
    calls = _spy_mask_calls(monkeypatch)
    out = m.infer_action(
        prompt=None,
        input_image=torch.randn(3, 16, 16),
        action_horizon=4,
        context=torch.randn(1, 2, 32),
        context_mask=torch.ones(1, 2, dtype=torch.bool),
        num_inference_steps=2,
        seed=0,
    )
    assert out["action"].shape == (4, 7)
    assert seen["prefix_len"] == 4
    assert (
        seen["payload"]["dual_f0"] is True and seen["payload"]["num_ref_latents"] == 2
    )
    (kw,) = calls
    assert (
        kw["num_ref_latents"] == 2
        and kw["dual_f0"] is True
        and (kw["traj_seq_len"] == 2 * 2)
    )


_MTWAM = "mtwam.runtime.create_mtwam"


def test_injection_dual_f0_false_is_zero_footprint():
    from mtwam.runtime import model_factory_extra_kwargs

    assert model_factory_extra_kwargs(
        _MTWAM, "horizontal", 0, dual_f0=False
    ) == model_factory_extra_kwargs(_MTWAM, "horizontal", 0)


def test_injection_dual_f0_true_injected_for_mtwam():
    from mtwam.runtime import model_factory_extra_kwargs

    extra = model_factory_extra_kwargs(_MTWAM, "horizontal", 0, dual_f0=True)
    assert extra["dynamic_branch_dual_f0"] is True


def test_validate_dual_f0_three_states():
    from mtwam.runtime import validate_branch_dual_f0

    validate_branch_dual_f0(False, False, 0)
    validate_branch_dual_f0(False, False, 16)
    with pytest.raises(ValueError, match="enable_dynamic_branch"):
        validate_branch_dual_f0(False, True, 0)
    with pytest.raises(ValueError, match="K=0 only"):
        validate_branch_dual_f0(True, True, 16)
    validate_branch_dual_f0(True, True, 0)


def test_behavioral_dual_f0_mismatch_caught():
    from mtwam.models.wan22._load_guards import check_branch_arch_consistency

    train = {"enable_dynamic_branch": True, "dynamic_branch_dual_f0": True}
    errs = check_branch_arch_consistency(train, {"enable_dynamic_branch": True})
    assert (
        len(errs) == 1
        and "dual_f0" in errs[0]
        and ("configuration mismatch" in errs[0])
    )
    assert check_branch_arch_consistency(train, dict(train)) == []
    assert check_branch_arch_consistency({}, {}) == []


def test_training_loss_dual_f0_horizon1_wiring(monkeypatch):
    from factories import make_tiny_mtwam

    m = make_tiny_mtwam(enable_dynamic_branch=True, dynamic_branch_horizon=1)
    m.mot.mixtures["video"].video_attention_mask_mode = "first_frame_causal"
    m.dynamic_branch_dual_f0 = True
    m.loss_lambda_traj, m.loss_lambda_tex = (0.1, 0.01)
    inputs = _fake_inputs(T=2, P=1, F_raw=5)
    monkeypatch.setattr(m, "build_inputs", lambda sample, tiled=False: inputs)
    calls = _spy_mask_calls(monkeypatch)
    torch.manual_seed(0)
    loss, loss_dict = m.training_loss({"unused": True})
    assert torch.isfinite(loss)
    assert loss_dict["loss_traj"] > 0 and loss_dict["loss_tex"] > 0
    assert m.mot._last_branch_hidden["traj"].shape[1] == 2
    (kw,) = calls
    assert kw["num_ref_latents"] == 2 and kw["dual_f0"] is True
    assert kw["traj_seq_len"] == 2 * 2


def test_infer_action_dual_f0_num_frames_positive(monkeypatch):
    from factories import make_tiny_mtwam

    m = make_tiny_mtwam(enable_dynamic_branch=True)
    m.mot.mixtures["video"].video_attention_mask_mode = "first_frame_causal"
    m.dynamic_branch_dual_f0 = True
    m.num_frames = 17
    m.num_extra_ref_frames = 0
    fixed_latent = torch.randn(1, 48, 1, 2, 4)
    monkeypatch.setattr(
        m,
        "_encode_input_image_latents_tensor",
        lambda input_image, tiled=False: fixed_latent,
    )
    prefill_called = {"n": 0}
    orig_prefill = m.mot.prefill_video_cache

    def spy_prefill(**kw):
        prefill_called["n"] += 1
        return orig_prefill(**kw)

    monkeypatch.setattr(m.mot, "prefill_video_cache", spy_prefill)
    out = m.infer_action(
        prompt=None,
        input_image=torch.randn(3, 16, 16),
        action_horizon=16,
        context=torch.randn(1, 2, 32),
        context_mask=torch.ones(1, 2, dtype=torch.bool),
        num_inference_steps=2,
        seed=0,
    )
    assert out["action"].shape == (16, 7)
    assert prefill_called["n"] == 1


def test_infer_action_dual_f0_num_frames_mismatch_rejects(monkeypatch):
    from factories import make_tiny_mtwam

    m = make_tiny_mtwam(enable_dynamic_branch=True)
    m.mot.mixtures["video"].video_attention_mask_mode = "first_frame_causal"
    m.dynamic_branch_dual_f0 = True
    m.num_frames = 17
    m.num_extra_ref_frames = 0
    encode_called = {"n": 0}

    def spy_encode(input_image, tiled=False):
        encode_called["n"] += 1
        return torch.randn(1, 48, 1, 2, 4)

    monkeypatch.setattr(m, "_encode_input_image_latents_tensor", spy_encode)
    prefill_called = {"n": 0}
    orig_prefill = m.mot.prefill_video_cache

    def spy_prefill(**kw):
        prefill_called["n"] += 1
        return orig_prefill(**kw)

    monkeypatch.setattr(m.mot, "prefill_video_cache", spy_prefill)
    with pytest.raises(ValueError, match="training chunk"):
        m.infer_action(
            prompt=None,
            input_image=torch.randn(3, 16, 16),
            action_horizon=32,
            context=torch.randn(1, 2, 32),
            context_mask=torch.ones(1, 2, dtype=torch.bool),
            num_inference_steps=2,
            seed=0,
        )
    assert encode_called["n"] == 0
    assert prefill_called["n"] == 0


def test_infer_action_vs_training_loss_dual_f0_mask_language_parity(monkeypatch):
    from factories import make_tiny_mtwam

    fixed_latent = torch.randn(1, 48, 1, 2, 4)
    m_t = make_tiny_mtwam(enable_dynamic_branch=True)
    m_t.mot.mixtures["video"].video_attention_mask_mode = "first_frame_causal"
    m_t.dynamic_branch_dual_f0 = True
    m_t.loss_lambda_traj, m_t.loss_lambda_tex = (0.1, 0.01)
    inputs = _fake_inputs()
    inputs["reference_latents"] = fixed_latent.clone()
    monkeypatch.setattr(m_t, "build_inputs", lambda sample, tiled=False: inputs)
    calls = _spy_mask_calls(monkeypatch)
    torch.manual_seed(0)
    m_t.training_loss({"unused": True})
    n_train = len(calls)
    m_i = make_tiny_mtwam(enable_dynamic_branch=True)
    m_i.mot.mixtures["video"].video_attention_mask_mode = "first_frame_causal"
    m_i.dynamic_branch_dual_f0 = True
    monkeypatch.setattr(
        m_i,
        "_encode_input_image_latents_tensor",
        lambda input_image, tiled=False: fixed_latent.clone(),
    )
    m_i.infer_action(
        prompt=None,
        input_image=torch.randn(3, 16, 16),
        action_horizon=4,
        context=torch.randn(1, 2, 32),
        context_mask=torch.ones(1, 2, dtype=torch.bool),
        num_inference_steps=2,
        seed=0,
    )
    calls_train, calls_infer = (calls[:n_train], calls[n_train:])
    (kw_t,) = calls_train
    (kw_i,) = calls_infer
    for key in ["num_ref_latents", "dual_f0", "traj_seq_len"]:
        assert kw_t[key] == kw_i[key], (
            f"Training and inference dual_f0 mask arguments differ for {key}: train={kw_t[key]} infer={kw_i[key]}"
        )
    assert kw_t["num_ref_latents"] == 2
    assert kw_t["dual_f0"] is True
    assert kw_t["traj_seq_len"] == 2 * 2


def test_from_wan22_pretrained_check_dual_f0_bounds_three_states():
    from mtwam.models.wan22.mtwam import MTWAM

    MTWAM._check_dual_f0_bounds(False, False, 0)
    MTWAM._check_dual_f0_bounds(False, False, 16)
    MTWAM._check_dual_f0_bounds(True, False, 0)
    MTWAM._check_dual_f0_bounds(True, False, 16)
    with pytest.raises(ValueError, match="enable_dynamic_branch"):
        MTWAM._check_dual_f0_bounds(False, True, 0)
    with pytest.raises(ValueError, match="K=0 only"):
        MTWAM._check_dual_f0_bounds(True, True, 16)
    MTWAM._check_dual_f0_bounds(True, True, 0)


def test_dual_f0_signature_threaded_through_all_layers():
    import inspect
    from mtwam.runtime import create_mtwam
    from mtwam.models.wan22.mtwam import MTWAM

    assert "dynamic_branch_dual_f0" in inspect.signature(create_mtwam).parameters
    assert (
        "dynamic_branch_dual_f0"
        in inspect.signature(MTWAM.from_wan22_pretrained).parameters
    )
