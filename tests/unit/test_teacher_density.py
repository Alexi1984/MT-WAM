import pytest
import torch
from mtwam.datasets.lerobot.teacher_cache import (
    TEACHER_GEOMETRY,
    teacher_dino_size,
    teacher_geometry,
)
from mtwam.datasets.lerobot.teacher_extract import extract_teacher_batch
from mtwam.models.wan22._load_guards import (
    BEHAVIORAL_BRANCH_KEYS,
    check_branch_arch_consistency,
)
from mtwam.models.wan22.dynamic_branch import TexMAEHead, TrajMAEHead


def test_robotwin_dense_geometry():
    assert teacher_geometry("robotwin_dense") == ((224, 280), (28, 28), (20, 20))
    assert teacher_dino_size("robotwin_dense") == (280, 280)


def test_legacy_geometry_byte_compat_pin():
    assert teacher_geometry("horizontal") == ((224, 224), (14, 14), (16, 16))
    assert teacher_geometry("robotwin") == ((224, 280), (16, 20), (16, 20))
    assert teacher_dino_size("horizontal") == (224, 224)
    assert teacher_dino_size("robotwin") == (224, 280)


def test_horizontal_dense_geometry():
    assert teacher_geometry("horizontal_dense") == ((224, 224), (28, 28), (20, 20))
    assert teacher_dino_size("horizontal_dense") == (280, 280)


def test_horizontal_dense_divisibility_invariants():
    g = TEACHER_GEOMETRY["horizontal_dense"]
    vh, vw = g["video_size"]
    assert vh % g["traj_grid_hw"][0] == 0 and vw % g["traj_grid_hw"][1] == 0
    dh, dw = g["dino_size"]
    assert (dh // 14, dw // 14) == tuple(g["dino_grid_hw"])


def test_extract_batch_horizontal_dense_shapes():
    B, C, P = (1, 2, 2)
    frames = torch.rand(B, C, P + 1, 3, 224, 224)
    models = _dense_models()
    tr, vi, di = extract_teacher_batch(
        frames,
        models,
        (28, 28),
        (224, 224),
        (20, 20),
        device="cpu",
        dino_size=(280, 280),
    )
    assert tr.shape == (B, C, P, 784, 2)
    assert vi.shape == (B, C, P, 784)
    assert di.shape == (B, C, P, 400, 768)
    assert models["cotracker"].seen_hw == (224, 224)
    assert models["dino"].seen_hw == (280, 280)


def test_dense_divisibility_invariants():
    g = TEACHER_GEOMETRY["robotwin_dense"]
    vh, vw = g["video_size"]
    assert vh % g["traj_grid_hw"][0] == 0 and vw % g["traj_grid_hw"][1] == 0
    dh, dw = g["dino_size"]
    assert (dh // 14, dw // 14) == tuple(g["dino_grid_hw"])


def test_teacher_dino_size_unknown_layout_raises():
    with pytest.raises(ValueError, match="unknown camera_layout"):
        teacher_dino_size("nope")


def test_traj_head_dense_grid_shape():
    head = TrajMAEHead(in_dim=16, decoder_dim=8, grid=(28, 28), horizon=2)
    out = head(torch.randn(2, 5, 16))
    assert out.shape == (2, 2, 784, 2)


def test_tex_head_dense_grid_shape():
    head = TexMAEHead(in_dim=16, decoder_dim=8, grid=(20, 20), horizon=2)
    out = head(torch.randn(2, 5, 16))
    assert out.shape == (2, 2, 400, 768)


def test_dense_heads_have_no_grid_shaped_params():
    h196 = TrajMAEHead(in_dim=16, decoder_dim=8, grid=196, horizon=2)
    h784 = TrajMAEHead(in_dim=16, decoder_dim=8, grid=(28, 28), horizon=2)
    s196 = {k: tuple(v.shape) for k, v in h196.state_dict().items()}
    s784 = {k: tuple(v.shape) for k, v in h784.state_dict().items()}
    assert s196 == s784


class _ShapeAwareCoTracker:
    def __init__(self):
        self.seen_hw = None

    def __call__(self, video, queries):
        self.seen_hw = tuple(video.shape[-2:])
        bsz, T = (int(video.shape[0]), int(video.shape[1]))
        n = int(queries.shape[1])
        return (torch.zeros(bsz, T, n, 2), torch.ones(bsz, T, n))


class _ShapeAwareDino:
    def __init__(self):
        self.seen_hw = None

    def __call__(self, x, is_training):
        self.seen_hw = tuple(x.shape[-2:])
        gh, gw = (x.shape[-2] // 14, x.shape[-1] // 14)
        return {"x_norm_patchtokens": torch.zeros(int(x.shape[0]), gh * gw, 768)}


def _dense_models():
    return {"cotracker": _ShapeAwareCoTracker(), "dino": _ShapeAwareDino()}


def test_extract_batch_dense_geometry_shapes():
    B, C, P = (1, 2, 2)
    frames = torch.rand(B, C, P + 1, 3, 224, 280)
    models = _dense_models()
    tr, vi, di = extract_teacher_batch(
        frames,
        models,
        (28, 28),
        (224, 280),
        (20, 20),
        device="cpu",
        dino_size=(280, 280),
    )
    assert tr.shape == (B, C, P, 784, 2)
    assert vi.shape == (B, C, P, 784)
    assert di.shape == (B, C, P, 400, 768)
    assert models["cotracker"].seen_hw == (224, 280)
    assert models["dino"].seen_hw == (280, 280)


def test_extract_batch_dino_size_none_is_legacy():
    B, C, P = (1, 1, 2)
    frames = torch.rand(B, C, P + 1, 3, 224, 280)
    models = _dense_models()
    tr, vi, di = extract_teacher_batch(
        frames, models, (16, 20), (224, 280), (16, 20), device="cpu"
    )
    assert tr.shape == (B, C, P, 320, 2)
    assert di.shape == (B, C, P, 320, 768)
    assert models["dino"].seen_hw == (224, 280)


def test_extract_batch_dense_grid_without_dino_size_fails_loud():
    B, C, P = (1, 1, 2)
    frames = torch.rand(B, C, P + 1, 3, 224, 280)
    with pytest.raises((AssertionError, RuntimeError)):
        extract_teacher_batch(
            frames, _dense_models(), (28, 28), (224, 280), (20, 20), device="cpu"
        )


def test_behavioral_keys_include_teacher_geometry():
    assert "dynamic_branch_teacher_geometry" in BEHAVIORAL_BRANCH_KEYS
    assert BEHAVIORAL_BRANCH_KEYS["dynamic_branch_teacher_geometry"] is None


def test_arch_consistency_catches_density_mismatch():
    train_cfg = {
        "enable_dynamic_branch": True,
        "dynamic_branch_teacher_geometry": "robotwin_dense",
    }
    eval_cfg = {"enable_dynamic_branch": True}
    errors = check_branch_arch_consistency(train_cfg, eval_cfg)
    assert any(("dynamic_branch_teacher_geometry" in e for e in errors))
    assert not check_branch_arch_consistency(train_cfg, dict(train_cfg))


def _mini_cfg(model=None, train=None, val=None):
    from omegaconf import OmegaConf

    d = {"model": model or {}, "data": {"train": train or {}}}
    if val is not None:
        d["data"]["val"] = val
    return OmegaConf.create(d)


def test_reconcile_tg_data_side_writes_back_model_node():
    from mtwam.runtime import reconcile_teacher_geometry

    cfg = _mini_cfg(train={"teacher_geometry_override": "horizontal_dense"})
    assert reconcile_teacher_geometry(cfg) == "horizontal_dense"
    assert cfg.model.dynamic_branch_teacher_geometry == "horizontal_dense"


def test_reconcile_tg_null_touches_nothing():
    from omegaconf import OmegaConf
    from mtwam.runtime import reconcile_teacher_geometry

    cfg = _mini_cfg(model={"enable_dynamic_branch": True}, train={"num_frames": 33})
    before = OmegaConf.to_container(cfg, resolve=True)
    assert reconcile_teacher_geometry(cfg) is None
    assert OmegaConf.to_container(cfg, resolve=True) == before


def test_reconcile_tg_drift_refuses():
    import pytest as _pt
    from mtwam.runtime import reconcile_teacher_geometry

    cfg = _mini_cfg(
        model={"dynamic_branch_teacher_geometry": "robotwin_dense"},
        train={"teacher_geometry_override": "horizontal_dense"},
    )
    with _pt.raises(ValueError, match="drift"):
        reconcile_teacher_geometry(cfg)


def test_reconcile_tg_model_side_inherits_to_data():
    from mtwam.runtime import reconcile_teacher_geometry

    cfg = _mini_cfg(model={"dynamic_branch_teacher_geometry": "robotwin_dense"})
    assert reconcile_teacher_geometry(cfg) == "robotwin_dense"
    assert cfg.data.train.teacher_geometry_override == "robotwin_dense"


def test_reconcile_tg_propagates_to_explicit_val_node():
    from mtwam.runtime import reconcile_teacher_geometry

    cfg = _mini_cfg(
        train={"teacher_geometry_override": "horizontal_dense"}, val={"num_frames": 33}
    )
    reconcile_teacher_geometry(cfg)
    assert cfg.data.val.teacher_geometry_override == "horizontal_dense"


def test_factory_injection_and_fail_loud():
    from mtwam.runtime import model_factory_extra_kwargs

    extra = model_factory_extra_kwargs(
        "mtwam.runtime.create_mtwam",
        "robotwin",
        0,
        teacher_geometry_override="robotwin_dense",
    )
    assert extra["dynamic_branch_teacher_geometry"] == "robotwin_dense"
    assert extra["dynamic_branch_camera_layout"] == "robotwin"
