import inspect
from pathlib import Path
import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import get_method
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "name",
    [
        "train_libero",
        "train_robotwin",
        "eval_libero",
        "eval_libero_plus",
        "eval_robotwin",
    ],
)
def test_complete_entry_config_matches_live_constructor_signatures(name):
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = OmegaConf.to_container(compose(config_name=name), resolve=True)

    def check(value):
        if isinstance(value, dict):
            if "_target_" in value:
                signature = inspect.signature(get_method(value["_target_"]))
                if not any(
                    (p.kind == p.VAR_KEYWORD for p in signature.parameters.values())
                ):
                    unknown = (
                        set(value)
                        - set(signature.parameters)
                        - {"_target_", "_recursive_", "_convert_", "_partial_"}
                    )
                    assert not unknown, (value["_target_"], unknown)
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)

    check(cfg)
    assert cfg["model"]["dynamic_branch_dual_f0"] is True
    assert cfg["model"]["dynamic_branch_ffn_moe_routing"] == "role"
    assert cfg["data"]["train"]["num_frames"] == 33
    assert cfg["data"]["train"]["num_extra_ref_frames"] == 0
