import os
from omegaconf import OmegaConf

_YAML = os.path.join(
    os.path.dirname(__file__), "..", "..", "configs", "model", "mtwam.yaml"
)


def test_yaml_selects_main_model():
    cfg = OmegaConf.load(_YAML)
    assert float(cfg.loss.get("lambda_traj", -1)) == 0.5
    assert float(cfg.loss.get("lambda_tex", -1)) == 0.25
    assert bool(cfg.loss.get("traj_validity_mask", False)) is True
    assert bool(cfg.get("enable_dynamic_branch", False)) is True
    assert int(cfg.get("dynamic_branch_num_layers", -1)) == 10
    assert bool(cfg.get("dynamic_branch_read_text", True)) is False


def test_create_mtwam_threads_branch_config(monkeypatch):
    import mtwam.runtime as rt
    from mtwam.models.wan22.mtwam import MTWAM

    captured = {}

    def _spy(**kwargs):
        captured.update(kwargs)
        return "SENTINEL"

    monkeypatch.setattr(MTWAM, "from_wan22_pretrained", staticmethod(_spy))
    out = rt.create_mtwam(
        model_id="x",
        tokenizer_model_id="y",
        video_dit_config={"text_dim": 4096},
        action_scheduler={
            "train_shift": 5.0,
            "infer_shift": 5.0,
            "num_train_timesteps": 1000,
        },
        loss={"lambda_traj": 0.1, "lambda_tex": 0.01, "traj_validity_mask": False},
        enable_dynamic_branch=True,
        dynamic_branch_num_layers=4,
        dynamic_branch_read_text=True,
        device="cpu",
    )
    assert out == "SENTINEL"
    assert captured["loss_lambda_traj"] == 0.1
    assert captured["loss_lambda_tex"] == 0.01
    assert captured["loss_traj_validity_mask"] is False
    assert captured["enable_dynamic_branch"] is True
    assert captured["dynamic_branch_num_layers"] == 4
    assert captured["dynamic_branch_read_text"] is True


def test_branch_config_is_public_factory_in_factories():
    import inspect
    from mtwam.runtime import create_mtwam

    parameters = set(inspect.signature(create_mtwam).parameters)
    assert (
        "enable_dynamic_branch" in parameters
        and "dynamic_branch_num_layers" in parameters
    )
    assert "dynamic_branch_read_text" in parameters


def test_yaml_role_moe_enabled():
    cfg = OmegaConf.load(_YAML)
    assert bool(cfg.get("dynamic_branch_ffn_moe", False)) is True


def test_create_mtwam_threads_ffn_moe(monkeypatch):
    import mtwam.runtime as rt
    from mtwam.models.wan22.mtwam import MTWAM

    captured = {}

    def _spy(**kwargs):
        captured.update(kwargs)
        return "SENTINEL"

    monkeypatch.setattr(MTWAM, "from_wan22_pretrained", staticmethod(_spy))
    out = rt.create_mtwam(
        model_id="x",
        tokenizer_model_id="y",
        video_dit_config={"text_dim": 4096},
        action_scheduler={
            "train_shift": 5.0,
            "infer_shift": 5.0,
            "num_train_timesteps": 1000,
        },
        enable_dynamic_branch=True,
        dynamic_branch_ffn_moe=True,
        device="cpu",
    )
    assert out == "SENTINEL"
    assert captured["dynamic_branch_ffn_moe"] is True


def test_ffn_moe_is_public_factory():
    import inspect
    from mtwam.runtime import create_mtwam

    assert "dynamic_branch_ffn_moe" in set(inspect.signature(create_mtwam).parameters)


def test_yaml_dynamic_branch_ffn_moe_routing_default_role():
    cfg = OmegaConf.load(_YAML)
    assert str(cfg.get("dynamic_branch_ffn_moe_routing", "MISSING")) == "role"


def test_ffn_moe_routing_is_public_factory():
    import inspect
    from mtwam.runtime import create_mtwam

    assert "dynamic_branch_ffn_moe_routing" in set(
        inspect.signature(create_mtwam).parameters
    )


def test_yaml_has_only_main_loss_components():
    cfg = OmegaConf.load(_YAML)
    assert set(cfg.loss) == {
        "lambda_video",
        "lambda_action",
        "lambda_traj",
        "lambda_tex",
        "traj_validity_mask",
    }
