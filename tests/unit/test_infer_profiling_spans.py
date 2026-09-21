import pytest
import torch
from torch.profiler import ProfilerActivity, profile
from factories import make_tiny_mtwam

EXPECTED_SPANS = [
    "mtwam/infer_vae_encode",
    "mtwam/infer_prefill",
    "mtwam/infer_denoise",
    "mtwam/infer_denoise_step",
]


def _run_infer(m, monkeypatch, num_inference_steps):
    fixed_latent = torch.randn(1, 48, 1, 2, 4)
    monkeypatch.setattr(
        m,
        "_encode_input_image_latents_tensor",
        lambda input_image, tiled=False: fixed_latent,
    )
    return m.infer_action(
        prompt=None,
        input_image=torch.randn(3, 16, 16),
        action_horizon=4,
        context=torch.randn(1, 2, 32),
        context_mask=torch.ones(1, 2, dtype=torch.bool),
        num_inference_steps=num_inference_steps,
        seed=0,
    )


@pytest.mark.parametrize("steps", [1, 3])
def test_infer_action_emits_named_spans_with_the_right_counts(monkeypatch, steps):
    m = make_tiny_mtwam(enable_dynamic_branch=True)
    m.mot.mixtures["video"].video_attention_mask_mode = "first_frame_causal"
    m.eval()
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        out = _run_infer(m, monkeypatch, steps)
    assert out["action"].shape == (4, 7)
    counts = {}
    for ev in prof.key_averages():
        if ev.key in EXPECTED_SPANS:
            counts[ev.key] = counts.get(ev.key, 0) + int(ev.count)
    missing = [s for s in EXPECTED_SPANS if s not in counts]
    assert not missing, (
        f"Missing inference profiling spans: {missing}; found {sorted(counts)}"
    )
    assert counts["mtwam/infer_vae_encode"] == 1
    assert counts["mtwam/infer_prefill"] == 1
    assert counts["mtwam/infer_denoise"] == 1
    assert counts["mtwam/infer_denoise_step"] == steps, (
        f"Expected {steps} denoising step spans, found {counts['mtwam/infer_denoise_step']}"
    )


def test_spans_do_not_change_the_numerical_output(monkeypatch):
    def _once(use_profiler):
        torch.manual_seed(7)
        m = make_tiny_mtwam(enable_dynamic_branch=True)
        m.mot.mixtures["video"].video_attention_mask_mode = "first_frame_causal"
        m.eval()
        if use_profiler:
            with profile(activities=[ProfilerActivity.CPU]):
                return _run_infer(m, monkeypatch, 2)["action"]
        return _run_infer(m, monkeypatch, 2)["action"]

    assert torch.equal(_once(False), _once(True))
