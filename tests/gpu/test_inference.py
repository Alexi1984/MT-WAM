import gc
import os
from pathlib import Path
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

ROOT = Path(__file__).resolve().parents[2]
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("MTWAM_RUN_GPU_TESTS") != "1",
        reason="Set MTWAM_RUN_GPU_TESTS=1 and provide local model assets.",
    ),
]


@pytest.mark.parametrize("benchmark", ["libero", "robotwin"])
def test_cached_action_inference(benchmark, monkeypatch):
    checkpoint = os.environ.get(f"MTWAM_{benchmark.upper()}_CHECKPOINT")
    assert checkpoint and Path(checkpoint).is_file(), (
        "Provide a checkpoint for this benchmark."
    )
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(config_name=f"train_{benchmark}")
    cfg.paths.wan = os.environ.get(
        "MTWAM_WAN_PATH", str(ROOT / "pretrained_models/Wan2.2-TI2V-5B")
    )
    cfg.paths.tokenizer = os.environ.get(
        "MTWAM_TOKENIZER_PATH", str(ROOT / "pretrained_models/Wan2.1-T2V-1.3B")
    )
    model = instantiate(
        cfg.model,
        device="cuda",
        model_dtype=torch.bfloat16,
        skip_dit_load_from_pretrain=True,
        load_text_encoder=False,
        action_dit_pretrained_path=None,
    )
    try:
        model.load_checkpoint(checkpoint)
        captured = []
        original = model.mot.prefill_video_cache

        def prefill(**kwargs):
            captured.append(
                (kwargs["prefix_tokens"].shape[1], kwargs["dynamic_branch_payload"])
            )
            return original(**kwargs)

        monkeypatch.setattr(model.mot, "prefill_video_cache", prefill)
        height, width = cfg.data.train.video_size
        horizon = cfg.data.train.num_frames - 1
        with torch.inference_mode():
            result = model.infer_action(
                prompt=None,
                input_image=torch.zeros(1, 3, height, width),
                action_horizon=horizon,
                proprio=torch.zeros(1, cfg.model.proprio_dim),
                context=torch.zeros(
                    1, cfg.model.tokenizer_max_len, cfg.model.video_dit_config.text_dim
                ),
                context_mask=torch.ones(
                    1, cfg.model.tokenizer_max_len, dtype=torch.bool
                ),
                num_inference_steps=10,
                seed=42,
            )
        assert result["action"].shape == (
            horizon,
            cfg.model.action_dit_config.action_dim,
        )
        assert torch.isfinite(result["action"]).all()
        assert len(captured) == 1
        prefix_length, payload = captured[0]
        assert payload["dual_f0"] is True
        assert payload["num_ref_latents"] == 2
        assert prefix_length == 2 * payload["first_frame_tokens"]
    finally:
        monkeypatch.undo()
        del model
        gc.collect()
        torch.cuda.empty_cache()
