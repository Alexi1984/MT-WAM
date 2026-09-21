import pytest
import torch
from factories import make_tiny_mtwam


def _save(model, tmp_path, name, drop_proprio=False, add_proprio=None):
    path = tmp_path / name
    model.save_checkpoint(str(path), step=0)
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if drop_proprio:
        payload.pop("proprio_encoder", None)
    if add_proprio is not None:
        payload["proprio_encoder"] = add_proprio
    torch.save(payload, str(path))
    return path


def test_missing_proprio_weights_is_fatal(tmp_path):
    m = make_tiny_mtwam(proprio_dim=8)
    assert m.proprio_encoder is not None
    ckpt = _save(m, tmp_path, "no_proprio.pt", drop_proprio=True)
    with pytest.raises(ValueError, match="proprio_encoder"):
        m.load_checkpoint(str(ckpt))


def test_missing_proprio_weights_cannot_be_bypassed(tmp_path):
    m = make_tiny_mtwam(proprio_dim=8)
    ckpt = _save(m, tmp_path, "no_proprio2.pt", drop_proprio=True)
    with pytest.raises(TypeError, match="allow_uninitialized_proprio"):
        m.load_checkpoint(str(ckpt), allow_uninitialized_proprio=True)


def test_checkpoint_proprio_but_model_without_is_fatal(tmp_path):
    trained = make_tiny_mtwam(proprio_dim=8)
    ckpt = _save(trained, tmp_path, "with_proprio.pt")
    evaluated = make_tiny_mtwam(proprio_dim=None)
    assert evaluated.proprio_encoder is None
    with pytest.raises(ValueError, match="proprio_dim=None"):
        evaluated.load_checkpoint(str(ckpt))


def test_matching_proprio_round_trip_is_unaffected(tmp_path):
    src = make_tiny_mtwam(proprio_dim=8)
    with torch.no_grad():
        src.proprio_encoder.weight.fill_(0.5)
        src.proprio_encoder.bias.fill_(-0.25)
    ckpt = _save(src, tmp_path, "match.pt")
    dst = make_tiny_mtwam(proprio_dim=8)
    assert not torch.allclose(dst.proprio_encoder.weight, src.proprio_encoder.weight)
    dst.load_checkpoint(str(ckpt))
    assert torch.allclose(
        dst.proprio_encoder.weight, torch.full_like(dst.proprio_encoder.weight, 0.5)
    )
    assert torch.allclose(
        dst.proprio_encoder.bias, torch.full_like(dst.proprio_encoder.bias, -0.25)
    )


def test_no_proprio_on_either_side_is_untouched(tmp_path):
    m = make_tiny_mtwam(proprio_dim=None)
    ckpt = _save(m, tmp_path, "neither.pt")
    m.load_checkpoint(str(ckpt))
