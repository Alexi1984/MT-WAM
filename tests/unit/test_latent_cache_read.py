import pytest
import torch
from safetensors.torch import save_file
from mtwam.datasets.lerobot.robot_video_dataset import read_latent_window
from mtwam.datasets.lerobot.teacher_cache import DynamicBranchCacheMissing


def _write_fake_cache(cache_dir, episode_index, n_windows=5, shape=(2, 3, 4, 4)):
    latents = torch.stack(
        [torch.full(shape, float(w), dtype=torch.bfloat16) for w in range(n_windows)],
        dim=0,
    )
    path = cache_dir / f"ep_{int(episode_index):06d}.safetensors"
    save_file({"latents": latents}, str(path))
    return latents


def test_read_latent_window_reads_correct_slot(tmp_path):
    latents = _write_fake_cache(tmp_path, 22922, n_windows=5)
    for fr in range(5):
        got = read_latent_window(str(tmp_path), episode_index=22922, frame_index=fr)
        assert got.shape == (2, 3, 4, 4), got.shape
        assert torch.equal(got.float(), latents[fr].float()), (
            f"slot mismatch at frame {fr}"
        )


def test_read_latent_window_bf16_roundtrip(tmp_path):
    _write_fake_cache(tmp_path, 7, n_windows=2)
    got = read_latent_window(str(tmp_path), episode_index=7, frame_index=1)
    assert got.dtype == torch.bfloat16
    assert torch.equal(got.float(), torch.full((2, 3, 4, 4), 1.0))


def test_read_latent_window_missing_file_raises(tmp_path):
    with pytest.raises(DynamicBranchCacheMissing):
        read_latent_window(str(tmp_path), episode_index=99999, frame_index=0)


def test_read_latent_window_episode_id_zero_padded(tmp_path):
    _write_fake_cache(tmp_path, 3, n_windows=1)
    got = read_latent_window(str(tmp_path), episode_index=3, frame_index=0)
    assert got.shape == (2, 3, 4, 4)


def _write_meta(tmp_path, **extra):
    import json

    meta = {
        "num_frames": 33,
        "num_extra_ref_frames": 0,
        "action_video_freq_ratio": 4,
        "layout": "per_repo",
        **extra,
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    return meta


def test_gate_image_keys_mismatch_and_reorder_refused(tmp_path):
    from mtwam.datasets.lerobot.robot_video_dataset import validate_latent_cache_meta

    _write_meta(tmp_path, image_keys=["image", "wrist_image"])
    common = dict(num_frames=33, num_extra_ref_frames=0, action_video_freq_ratio=4)
    with pytest.raises(ValueError, match="image_keys"):
        validate_latent_cache_meta(
            str(tmp_path), **common, image_keys=["image", "birdview_image"]
        )
    with pytest.raises(ValueError, match="image_keys"):
        validate_latent_cache_meta(
            str(tmp_path), **common, image_keys=["wrist_image", "image"]
        )


def test_gate_image_keys_match_passes(tmp_path):
    from mtwam.datasets.lerobot.robot_video_dataset import validate_latent_cache_meta

    _write_meta(tmp_path, image_keys=["image", "wrist_image"])
    meta = validate_latent_cache_meta(
        str(tmp_path),
        num_frames=33,
        num_extra_ref_frames=0,
        action_video_freq_ratio=4,
        image_keys=["image", "wrist_image"],
    )
    assert meta["image_keys"] == ["image", "wrist_image"]


def test_gate_q_window_vs_33_cache_refused(tmp_path):
    from mtwam.datasets.lerobot.robot_video_dataset import validate_latent_cache_meta

    _write_meta(tmp_path)
    with pytest.raises(ValueError, match="window recipe"):
        validate_latent_cache_meta(
            str(tmp_path),
            num_frames=17,
            num_extra_ref_frames=0,
            action_video_freq_ratio=4,
        )


def test_gate_q_window_matched_cache_passes(tmp_path):
    from mtwam.datasets.lerobot.robot_video_dataset import validate_latent_cache_meta

    _write_meta(tmp_path, num_frames=17)
    meta = validate_latent_cache_meta(
        str(tmp_path), num_frames=17, num_extra_ref_frames=0, action_video_freq_ratio=4
    )
    assert int(meta["num_frames"]) == 17


def test_gate_legacy_meta_without_image_keys_passes(tmp_path):
    from mtwam.datasets.lerobot.robot_video_dataset import validate_latent_cache_meta

    _write_meta(tmp_path)
    meta = validate_latent_cache_meta(
        str(tmp_path),
        num_frames=33,
        num_extra_ref_frames=0,
        action_video_freq_ratio=4,
        image_keys=["image", "wrist_image"],
    )
    assert "image_keys" not in meta
