import torch
from factories import make_tiny_mtwam


def _fake_inputs(T=3, H=2, W=4, Sa=4, L=2, C=1, P=2, F_raw=9):
    return {
        "input_latents": torch.randn(1, 48, T, H, W),
        "context": torch.randn(1, L, 32),
        "context_mask": None,
        "action": torch.randn(1, Sa, 7),
        "action_is_pad": torch.zeros(1, Sa, dtype=torch.bool),
        "image_is_pad": torch.zeros(1, F_raw, dtype=torch.bool),
        "num_ref_latents": 1,
        "reference_latents": torch.randn(1, 48, 1, H, W),
        "fuse_vae_embedding_in_latents": True,
        "teacher_tracks": torch.randn(1, C, P, 196, 2),
        "teacher_track_vis": torch.ones(1, C, P, 196, dtype=torch.bool),
        "teacher_dino": torch.randn(1, C, P, 256, 768),
    }


def _run_with_noise_frame_override(m, monkeypatch, base_inputs, seed, noise_frame_fill):
    inp = dict(base_inputs)
    lat = base_inputs["input_latents"].clone()
    lat[:, :, 1:] = noise_frame_fill
    inp["input_latents"] = lat
    monkeypatch.setattr(m, "build_inputs", lambda sample, tiled=False: inp)
    torch.manual_seed(seed)
    return m.training_loss({"unused": True})


def test_lambda_video_zero_noise_frames_dont_affect_action_branch(monkeypatch):
    m = make_tiny_mtwam(enable_dynamic_branch=True)
    m.mot.mixtures["video"].video_attention_mask_mode = "first_frame_causal"
    m.loss_lambda_video = 0.0
    m.loss_lambda_traj, m.loss_lambda_tex = (0.1, 0.01)
    base = _fake_inputs()
    _, d_a = _run_with_noise_frame_override(
        m, monkeypatch, base, seed=0, noise_frame_fill=0.0
    )
    _, d_b = _run_with_noise_frame_override(
        m, monkeypatch, base, seed=0, noise_frame_fill=7.3
    )
    assert d_a["loss_action"] == d_b["loss_action"]
    assert d_a["loss_traj"] == d_b["loss_traj"]
    assert d_a["loss_tex"] == d_b["loss_tex"]


def test_lambda_video_zero_video_loss_DOES_change(monkeypatch):
    m = make_tiny_mtwam(enable_dynamic_branch=False)
    m.mot.mixtures["video"].video_attention_mask_mode = "first_frame_causal"
    m.loss_lambda_video = 1.0
    m.loss_lambda_traj = m.loss_lambda_tex = 0.0
    base = _fake_inputs()
    _, d_a = _run_with_noise_frame_override(
        m, monkeypatch, base, seed=0, noise_frame_fill=0.0
    )
    _, d_b = _run_with_noise_frame_override(
        m, monkeypatch, base, seed=0, noise_frame_fill=7.3
    )
    assert d_a["loss_video"] != d_b["loss_video"]


def test_lambda_video_zero_action_branch_still_learn(monkeypatch):
    m = make_tiny_mtwam(enable_dynamic_branch=True)
    m.mot.mixtures["video"].video_attention_mask_mode = "first_frame_causal"
    m.loss_lambda_video = 0.0
    m.loss_lambda_traj, m.loss_lambda_tex = (0.1, 0.01)
    inputs = _fake_inputs()
    monkeypatch.setattr(m, "build_inputs", lambda sample, tiled=False: inputs)
    torch.manual_seed(0)
    loss, d = m.training_loss({"unused": True})
    assert torch.isfinite(loss) and float(loss) > 0.0
    assert d["loss_action"] > 0.0 and d["loss_traj"] > 0.0 and (d["loss_tex"] > 0.0)
    loss.backward()
    br = m.mot.dynamic_branch
    br_g = [p.grad.norm().item() for p in br.parameters() if p.grad is not None]
    assert len(br_g) > 0 and max(br_g) > 0
