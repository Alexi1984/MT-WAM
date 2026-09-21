from pathlib import Path
import pytest
from mtwam.models.wan22.helpers.io import ModelConfig
from mtwam.utils.eval_config import find_dataset_stats


def test_local_model_directory_resolves_shards_without_download(monkeypatch, tmp_path):
    def download(self):
        raise AssertionError("Local paths must never trigger a remote download.")

    monkeypatch.setattr(ModelConfig, "download", download)
    paths = [tmp_path / f"diffusion_pytorch_model-{i}.safetensors" for i in (1, 2)]
    for path in paths:
        path.write_bytes(b"fixture")
    cfg = ModelConfig(
        model_id=str(tmp_path),
        origin_file_pattern="diffusion_pytorch_model*.safetensors",
    )
    cfg.download_if_necessary()
    assert cfg.path == [str(path) for path in paths]
    missing = ModelConfig(model_id=str(tmp_path), origin_file_pattern="missing.pth")
    with pytest.raises(FileNotFoundError, match="Model files not found"):
        missing.download_if_necessary()
    absent = ModelConfig(model_id=str(tmp_path / "absent"), origin_file_pattern="*.pth")
    with pytest.raises(FileNotFoundError, match="Model directory not found"):
        absent.download_if_necessary()


def test_explicit_missing_statistics_never_fall_back_to_another_file(tmp_path):
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"fixture")
    (tmp_path / "dataset_stats.json").write_text("{}")
    assert find_dataset_stats(checkpoint) == tmp_path / "dataset_stats.json"
    with pytest.raises(FileNotFoundError, match="statistics not found"):
        find_dataset_stats(checkpoint, tmp_path / "wrong.json")
