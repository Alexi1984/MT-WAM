import gc
import os
from pathlib import Path
import pytest
import torch
from mtwam.datasets.lerobot.teacher_extract import (
    extract_teacher_for_camera,
    load_teacher_models,
)

ROOT = Path(__file__).resolve().parents[2]
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("MTWAM_RUN_GPU_TESTS") != "1",
        reason="Set MTWAM_RUN_GPU_TESTS=1 and provide local teacher checkpoints.",
    ),
]


def test_teacher_output_shapes():
    cotracker = os.environ.get(
        "MTWAM_COTRACKER_CHECKPOINT", str(ROOT / "pretrained_models/scaled_offline.pth")
    )
    dino = os.environ.get(
        "MTWAM_DINO_CHECKPOINT",
        str(ROOT / "pretrained_models/dinov2_vitb14_pretrain.pth"),
    )
    models = load_teacher_models(cotracker, dino, device="cuda")
    try:
        generator = torch.Generator().manual_seed(42)
        frames = torch.rand(3, 3, 224, 224, generator=generator)
        tracks, visibility, features = extract_teacher_for_camera(
            frames, models, device="cuda"
        )
        assert tracks.shape == (2, 196, 2)
        assert visibility.shape == (2, 196)
        assert visibility.dtype == torch.bool
        assert features.shape == (2, 256, 768)
        assert tracks.device.type == features.device.type == "cpu"
        assert torch.isfinite(tracks).all()
        assert torch.isfinite(features).all()
    finally:
        del models
        gc.collect()
        torch.cuda.empty_cache()
