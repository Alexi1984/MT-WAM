import pytest
import torch
from safetensors.torch import save_file
from factories import make_tiny_video_expert
from mtwam.models.wan22.helpers.loader import (
    _load_video_dit_from_path,
    _normalize_video_dit_state_dict,
    load_wan22_ti2v_5b_components,
)

TINY_CFG = dict(
    hidden_dim=16,
    in_dim=48,
    ffn_dim=32,
    out_dim=48,
    text_dim=32,
    freq_dim=256,
    eps=1e-06,
    patch_size=(1, 2, 2),
    num_heads=2,
    attn_head_dim=8,
    num_layers=4,
    has_image_input=False,
    seperated_timestep=True,
)


def _tiny_state_dict():
    torch.manual_seed(20260712)
    return {k: v.clone() for k, v in make_tiny_video_expert().state_dict().items()}


def _save(tmp_path, name, sd):
    p = tmp_path / name
    save_file({k: v.contiguous() for k, v in sd.items()}, str(p))
    return str(p)


def test_native_layout_loads_exactly(tmp_path):
    src = _tiny_state_dict()
    path = _save(tmp_path, "native.safetensors", src)
    model, resolved = _load_video_dit_from_path(
        path, TINY_CFG, torch_dtype=torch.float32, device="cpu"
    )
    assert resolved == path
    got = model.state_dict()
    for k in src:
        assert torch.allclose(got[k].float(), src[k].float()), f"tensor mismatch at {k}"


def test_dit_prefix_layout_loads_exactly(tmp_path):
    src = _tiny_state_dict()
    path = _save(
        tmp_path, "prefixed.safetensors", {f"dit.{k}": v for k, v in src.items()}
    )
    model, _ = _load_video_dit_from_path(
        path, TINY_CFG, torch_dtype=torch.float32, device="cpu"
    )
    got = model.state_dict()
    assert torch.allclose(
        got["blocks.0.self_attn.q.weight"].float(),
        src["blocks.0.self_attn.q.weight"].float(),
    )
    assert torch.allclose(
        got["head.modulation"].float(), src["head.modulation"].float()
    )


def test_model_prefix_layout_loads_exactly(tmp_path):
    src = _tiny_state_dict()
    path = _save(
        tmp_path, "model_pfx.safetensors", {f"model.{k}": v for k, v in src.items()}
    )
    model, _ = _load_video_dit_from_path(
        path, TINY_CFG, torch_dtype=torch.float32, device="cpu"
    )
    assert torch.allclose(
        model.state_dict()["patch_embedding.weight"].float(),
        src["patch_embedding.weight"].float(),
    )


def test_unknown_layout_refused(tmp_path):
    src = _tiny_state_dict()
    weird = {f"transformer.{k}": v for k, v in src.items()}
    path = _save(tmp_path, "weird.safetensors", weird)
    with pytest.raises(ValueError, match="Unrecognized video DiT state-dict layout"):
        _load_video_dit_from_path(
            path, TINY_CFG, torch_dtype=torch.float32, device="cpu"
        )


def test_mixed_prefix_refused(tmp_path):
    src = _tiny_state_dict()
    src["dit.blocks.0.self_attn.q.weight"] = src["blocks.0.self_attn.q.weight"].clone()
    path = _save(tmp_path, "mixed.safetensors", src)
    with pytest.raises(ValueError, match="not exact"):
        _load_video_dit_from_path(
            path, TINY_CFG, torch_dtype=torch.float32, device="cpu"
        )


def test_missing_key_refused(tmp_path):
    src = _tiny_state_dict()
    src.pop("blocks.3.ffn.2.weight")
    path = _save(tmp_path, "missing.safetensors", src)
    with pytest.raises(ValueError, match="not exact"):
        _load_video_dit_from_path(
            path, TINY_CFG, torch_dtype=torch.float32, device="cpu"
        )


def test_unexpected_key_refused(tmp_path):
    src = _tiny_state_dict()
    src["blocks.0.some_new_adapter.weight"] = torch.zeros(4, 4)
    path = _save(tmp_path, "extra.safetensors", src)
    with pytest.raises(ValueError, match="not exact"):
        _load_video_dit_from_path(
            path, TINY_CFG, torch_dtype=torch.float32, device="cpu"
        )


def test_empty_state_dict_refused():
    with pytest.raises(ValueError, match="empty state dict"):
        _normalize_video_dit_state_dict({})


def test_missing_file_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        _load_video_dit_from_path(
            str(tmp_path / "nope*.safetensors"),
            TINY_CFG,
            torch_dtype=torch.float32,
            device="cpu",
        )


def test_mutual_exclusion_with_skip_guard_fires_first(tmp_path):
    with pytest.raises(ValueError, match="mutually exclusive"):
        load_wan22_ti2v_5b_components(
            device="cpu",
            torch_dtype=torch.float32,
            dit_config=TINY_CFG,
            skip_dit_load_from_pretrain=True,
            video_dit_pretrained_path=str(tmp_path / "whatever.safetensors"),
        )
