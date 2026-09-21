import ast
import hashlib
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import types
import numpy as np
import pytest
import torch
import yaml
from mtwam.datasets.lerobot.teacher_extract import load_teacher_models

ROOT = Path(__file__).resolve().parents[2]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checked_patch_is_idempotent_and_rejects_unknown_source(tmp_path):
    setup = script("setup_dependencies")
    path = tmp_path / "dependency.py"
    original = b"coords_init = coords.view(B * T, N, 2)\n"
    path.write_bytes(original)
    digest = hashlib.sha256(original).hexdigest()
    args = (path, digest, "coords.view", "coords.reshape")
    assert setup.checked_replace(*args) == "patched"
    updated = path.read_bytes()
    assert setup.checked_replace(*args) == "already patched"
    assert path.read_bytes() == updated
    path.write_bytes(updated + b"unexpected = True\n")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Unrecognized source"):
        setup.checked_replace(*args)
    assert path.read_bytes() == before


def test_cotracker_reshape_preserves_noncontiguous_batch_coordinates():
    coordinates = torch.arange(24).reshape(2, 1, 6, 2).expand(2, 3, 6, 2)
    with pytest.raises(RuntimeError):
        coordinates.view(6, 6, 2)
    actual = coordinates.reshape(6, 6, 2)
    expected = torch.stack([coordinates[b, t] for b in range(2) for t in range(3)])
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("missing", ["cotracker", "dino"])
def test_teacher_checkpoints_are_required_before_optional_imports(
    tmp_path, monkeypatch, missing
):
    present = tmp_path / "weights.pth"
    present.write_bytes(b"fixture")

    def reject(*args, **kwargs):
        raise AssertionError("A remote fallback must never be called")

    monkeypatch.setattr(torch.hub, "load", reject)
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        load_teacher_models(
            None if missing == "cotracker" else present,
            None if missing == "dino" else present,
            device="cpu",
        )


@pytest.mark.parametrize(
    "name,dimension", [("train_libero", 7), ("train_robotwin", 14)]
)
def test_action_backbone_preparation_resolves_root_config(name, dimension):
    prepare = script("preprocess_action_dit_backbone")
    video, action, model = prepare._load_model_config(name, ["paths.wan=./fixture-wan"])
    assert action["action_dim"] == dimension
    assert action["num_layers"] == video["num_layers"] == 30
    assert model.model_id == "./fixture-wan"
    assert model.redirect_common_files is False


def test_robotwin_runtime_configs_have_exact_single_axis_changes():
    root = ROOT / "third_party/RoboTwin/task_config"
    clean = yaml.safe_load((root / "demo_clean.yml").read_text())
    changed = {
        "bg": {"random_background": True, "clean_background_rate": 0.02},
        "light": {"random_light": True, "crazy_random_light_rate": 0.02},
        "cluttered": {"cluttered_table": True, "clean_background_rate": 0.02},
        "tableheight": {"random_table_height": 0.03},
    }
    for axis, differences in changed.items():
        data = yaml.safe_load((root / f"eval_c2r_{axis}.yml").read_text())
        randomization = data.pop("domain_randomization")
        assert data == {k: v for k, v in clean.items() if k != "domain_randomization"}
        assert randomization == clean["domain_randomization"] | differences
        assert data["data_type"]["pointcloud"] is False
    tasks = yaml.safe_load((root / "_eval_step_limit.yml").read_text())
    assert len(tasks) == 50
    assert all(((root.parent / "envs" / f"{name}.py").is_file() for name in tasks))


@pytest.fixture
def robotwin_base_methods():
    path = ROOT / "third_party/RoboTwin/envs/_base_task.py"
    tree = ast.parse(path.read_text())
    names = {"get_cluttered_table", "create_table_and_wall"}
    methods = [
        node
        for cls in tree.body
        if isinstance(cls, ast.ClassDef)
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in methods} == names
    namespace = {"math": math}
    exec(
        compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace


@pytest.mark.parametrize(
    "random_value,expected_clutter", [(0.01, False), (0.02, True), (0.5, True)]
)
def test_cluttered_config_reaches_real_object_placement(
    robotwin_base_methods, random_value, expected_clutter
):
    config = yaml.safe_load(
        (ROOT / "third_party/RoboTwin/task_config/eval_c2r_cluttered.yml").read_text()
    )["domain_randomization"]
    attempts = []
    actor = types.SimpleNamespace(
        set_name=lambda name: None,
        get_pose=lambda: types.SimpleNamespace(p=np.array([0.1, 0.1, 0.75])),
    )

    def place_actor(scene, **kwargs):
        attempts.append(kwargs)
        return True, actor

    robotwin_base_methods.update(
        np=types.SimpleNamespace(
            random=types.SimpleNamespace(
                rand=lambda: random_value, randint=lambda n: 0
            ),
            array=np.array,
        ),
        get_available_cluttered_objects=lambda task_objects: (
            ["fixture"],
            {
                "fixture": {
                    "ids": [0],
                    "params": {0: {"radius": 0.02, "z_offset": 0.0, "z_max": 0.05}},
                    "type": "fixture",
                }
            },
        ),
        rand_create_cluttered_actor=place_actor,
    )
    task = types.SimpleNamespace(
        table_xy_bias=[0.0, 0.0],
        table_z_bias=0.0,
        clean_background_rate=config["clean_background_rate"],
        scene=types.SimpleNamespace(get_all_actors=lambda: []),
        size_dict=[],
        cluttered_objs=[],
        prohibited_area=[],
    )
    assert config["cluttered_table"] is True
    robotwin_base_methods["get_cluttered_table"](task)
    assert bool(attempts) is expected_clutter
    assert len(task.record_cluttered_objects) == len(attempts)


def test_cluttered_config_keeps_background_textures_disabled(robotwin_base_methods):
    config = yaml.safe_load(
        (ROOT / "third_party/RoboTwin/task_config/eval_c2r_cluttered.yml").read_text()
    )["domain_randomization"]
    textures = []

    def create_actor(scene, pose, **kwargs):
        textures.append(kwargs["texture_id"])

    def reject_texture_lookup(path):
        raise AssertionError(
            "Clutter-only evaluation must not load background textures"
        )

    robotwin_base_methods.update(
        os=types.SimpleNamespace(listdir=reject_texture_lookup),
        sapien=types.SimpleNamespace(Pose=lambda **kwargs: kwargs),
        create_box=create_actor,
        create_table=create_actor,
    )
    task = types.SimpleNamespace(
        random_background=config["random_background"],
        clean_background_rate=config["clean_background_rate"],
        eval_mode=True,
        table_z_bias=0.0,
        scene=object(),
    )
    robotwin_base_methods["create_table_and_wall"](task)
    assert textures == [None, None]


def test_native_robotwin_help_does_not_require_simulator():
    result = subprocess.run(
        [sys.executable, "script/eval_policy.py", "--help"],
        cwd=ROOT / "third_party/RoboTwin",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout


def test_libero_resource_configuration_preserves_existing_settings(tmp_path):
    setup = script("setup_dependencies")
    source, config = (tmp_path / "source", tmp_path / "config")
    setup.configure_libero(source, config)
    setup.configure_libero(source, config)
    path = config / "config.yaml"
    value = yaml.safe_load(path.read_text())
    assert value["init_states"] == str(source / "libero/libero/init_files")
    path.write_text("assets: user-owned\n")
    with pytest.raises(
        ValueError, match="Existing LIBERO resource configuration differs"
    ):
        setup.configure_libero(source, config)
    assert path.read_text() == "assets: user-owned\n"


def test_teacher_producer_respects_both_clean_episode_lists(tmp_path, monkeypatch):
    from hydra import compose, initialize_config_dir
    from mtwam.datasets.lerobot import base_lerobot_dataset

    producer = script("precompute_teacher_feats")
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(
            config_name="train_robotwin",
            overrides=[f"teacher_cache.out_dir={tmp_path}"],
        )
    captured = []

    class Base:
        def __init__(self, **kwargs):
            captured.append(kwargs)

        def _set_return_images(self, enabled):
            assert enabled

    monkeypatch.setattr(base_lerobot_dataset, "BaseLerobotDataset", Base)
    monkeypatch.setattr(producer, "load_teacher_models", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        producer, "build_dataset_windows", lambda *args, **kwargs: iter(())
    )
    producer.main(cfg)
    assert [len(node["include_episodes"]) for node in captured] == [2350, 150]
    assert set(captured[0]["include_episodes"]).isdisjoint(
        captured[1]["include_episodes"]
    )
    assert all(
        (node["obs_size"] == 33 and node["action_size"] == 32 for node in captured)
    )
