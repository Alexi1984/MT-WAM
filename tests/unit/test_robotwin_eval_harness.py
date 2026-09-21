from __future__ import annotations
import ast
import csv
import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import yaml
from hydra import compose, initialize_config_dir
from mtwam.utils.eval_config import save_resolved_config
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, relative_path: str):
    path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MANAGER = _load_module(
    "robotwin_manager_under_test", "experiments/robotwin/run_robotwin_manager.py"
)
SINGLE = _load_module(
    "robotwin_single_under_test", "experiments/robotwin/eval_robotwin_single.py"
)


def _make_release_root(root: Path, tasks: list[str]) -> Path:
    task_config_dir = root / "task_config"
    task_config_dir.mkdir(parents=True)
    task_map = {task: 1000 for task in tasks}
    (task_config_dir / "_eval_step_limit.yml").write_text(
        yaml.safe_dump(task_map, sort_keys=False), encoding="utf-8"
    )
    for task_config in MANAGER.C2R_TASK_CONFIG_BY_NAME.values():
        (task_config_dir / f"{task_config}.yml").write_text(
            "domain_randomization: {}\n", encoding="utf-8"
        )
    return root


def _make_manager_cfg(tmp_path: Path, *, task_name: str | None = "beat_block_hammer"):
    release_root = _make_release_root(tmp_path / "release", ["beat_block_hammer"])
    ckpt = tmp_path / "step_000001.pt"
    ckpt.write_bytes(b"checkpoint")
    stats = tmp_path / "dataset_stats.json"
    stats.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "results" / "stable-run"
    cfg = OmegaConf.create(
        {
            "ckpt": str(ckpt),
            "seed": 0,
            "EVALUATION": {
                "robotwin_root": str(release_root),
                "task_name": task_name,
                "task_config": "demo_clean",
                "c2r_configs": None,
                "instruction_type": "seen",
                "eval_num_episodes": 1,
                "output_dir": str(output_dir),
                "dataset_stats_path": str(stats),
            },
            "MULTIRUN": {"num_gpus": 1, "max_tasks_per_gpu": 1},
        }
    )
    with initialize_config_dir(
        version_base="1.3", config_dir=str(PROJECT_ROOT / "configs")
    ):
        base = compose(config_name="eval_robotwin")
    cfg = OmegaConf.merge(base, cfg)
    return (cfg, release_root, output_dir)


class _CompletedWorker:
    def __init__(self, return_code: int = 0):
        self.return_code = return_code

    def poll(self):
        return self.return_code

    def terminate(self):
        self.return_code = -15

    def wait(self, timeout=None):
        return self.return_code

    def kill(self):
        self.return_code = -9


def _install_successful_workers(monkeypatch, launches: list[str], rate: float = 0.5):
    task_config_to_name = {
        task_config: name
        for name, task_config in MANAGER.C2R_TASK_CONFIG_BY_NAME.items()
    }

    def fake_popen(cmd, **kwargs):
        overrides = {
            token.split("=", 1)[0]: token.split("=", 1)[1]
            for token in cmd
            if "=" in token
        }
        task_name = overrides["EVALUATION.task_name"]
        task_config = overrides["EVALUATION.task_config"]
        output_dir = Path(ast.literal_eval(overrides["EVALUATION.output_dir"]))
        assert overrides["EVALUATION.robotwin_root"]
        assert overrides["EVALUATION.instruction_type"] == "seen"
        assert overrides["EVALUATION.eval_num_episodes"] == "1"
        assert overrides["seed"] == "0"
        snapshot = Path(cmd[cmd.index("--config-path") + 1]) / (
            cmd[cmd.index("--config-name") + 1] + ".yaml"
        )
        worker_cfg = OmegaConf.load(snapshot)
        assert worker_cfg.model.dynamic_branch_dual_f0 is True
        assert worker_cfg.model.dynamic_branch_ffn_moe is True
        assert worker_cfg.data.train.processor.action_output_dim == 14
        config_name = task_config_to_name[task_config]
        launches.append(task_config)
        result_file = (
            output_dir / task_name / MANAGER._config_result_filename(config_name)
        )
        result_file.parent.mkdir(parents=True, exist_ok=True)
        result_file.write_text(f"Timestamp: test\n\n{rate}\n", encoding="utf-8")
        return _CompletedWorker()

    monkeypatch.setattr(MANAGER.subprocess, "Popen", fake_popen)


def test_resolved_config_survives_real_worker_and_policy_round_trip(tmp_path):
    with initialize_config_dir(
        version_base="1.3", config_dir=str(PROJECT_ROOT / "configs")
    ):
        cfg = compose(config_name="eval_robotwin")
    cfg.paths.wan = str(tmp_path / "weights, (main model)")
    cfg.EVALUATION.negative_prompt = "don't move [left], keep (x)"
    snapshot = save_resolved_config(
        cfg, tmp_path / "config" / "evaluation.yaml", PROJECT_ROOT
    )
    process = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "experiments/robotwin/eval_robotwin_single.py"),
            "--config-path",
            str(snapshot.parent),
            "--config-name",
            snapshot.stem,
            "--cfg",
            "job",
            "--resolve",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    assert process.returncode == 0, process.stderr
    worker = yaml.safe_load(process.stdout)
    policy_module = _load_module(
        "robotwin_policy_round_trip",
        "experiments/robotwin/mtwam_policy/deploy_policy.py",
    )
    policy = OmegaConf.to_container(policy_module._load_sim_cfg(snapshot), resolve=True)
    for key in ("model", "data", "EVALUATION", "mixed_precision", "seed"):
        assert policy[key] == worker[key]
    assert worker["model"]["model_id"] == str(tmp_path / "weights, (main model)")
    assert worker["EVALUATION"]["negative_prompt"] == "don't move [left], keep (x)"


def test_external_release_task_list_and_six_config_contract(tmp_path):
    tasks = [f"task_{idx:02d}" for idx in range(50)]
    release_root = _make_release_root(tmp_path / "external-release", tasks)
    assert MANAGER._load_all_tasks(release_root) == tasks
    MANAGER._validate_c2r_task_configs(release_root)
    assert MANAGER.C2R_CONFIG_NAMES == (
        "easy",
        "bg",
        "light",
        "cluttered",
        "tableheight",
        "hard",
    )
    assert [
        MANAGER._config_result_filename(name) for name in MANAGER.C2R_CONFIG_NAMES
    ] == [
        "_result_clean.txt",
        "_result_bg.txt",
        "_result_light.txt",
        "_result_cluttered.txt",
        "_result_tableheight.txt",
        "_result_random.txt",
    ]


def test_result_parser_rejects_corrupt_and_out_of_range_values(tmp_path):
    result_file = tmp_path / "result.txt"
    result_file.write_text("Timestamp: test\n\n0.75\n", encoding="utf-8")
    assert MANAGER._parse_success_rate(result_file) == pytest.approx(0.75)
    for invalid in ("no-rate\n", "nan\n", "1.01\n", "-0.1\n"):
        result_file.write_text(invalid, encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid success rate"):
            MANAGER._parse_success_rate(result_file)


def test_manifest_creation_exact_resume_and_identity_mismatch(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    payload = {"schema_version": 1, "checkpoint": {"path": "/a", "size_bytes": 1}}
    assert MANAGER._write_or_validate_manifest(run_dir, payload) == "created"
    assert MANAGER._write_or_validate_manifest(run_dir, payload) == "validated"
    with pytest.raises(ValueError, match="Resume identity mismatch"):
        MANAGER._write_or_validate_manifest(
            run_dir,
            {"schema_version": 1, "checkpoint": {"path": "/b", "size_bytes": 1}},
        )
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    (legacy_dir / "old-result.txt").write_text("0.5", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Non-empty output directory"):
        MANAGER._write_or_validate_manifest(legacy_dir, payload)


def test_manager_runs_six_configs_writes_summary_and_resumes(monkeypatch, tmp_path):
    cfg, _, output_dir = _make_manager_cfg(tmp_path)
    launches: list[str] = []
    _install_successful_workers(monkeypatch, launches, rate=0.5)
    MANAGER.main.__wrapped__(cfg)
    assert launches == [
        MANAGER.C2R_TASK_CONFIG_BY_NAME[name] for name in MANAGER.C2R_CONFIG_NAMES
    ]
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert [
        item["name"] for item in summary["manifest"]["evaluation"]["configs"]
    ] == list(MANAGER.C2R_CONFIG_NAMES)
    assert summary["completeness"] == {
        "expected_results": 6,
        "completed_results": 6,
        "missing": [],
    }
    assert summary["overall"]["mean_success_rate"] == {
        name: 0.5 for name in MANAGER.C2R_CONFIG_NAMES
    }
    assert summary["overall"]["retention"] == {
        name: 1.0 for name in MANAGER.C2R_CONFIG_NAMES if name != "easy"
    }
    with (output_dir / "summary.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0][:7] == [
        "task_name",
        "easy_success_rate",
        "bg_success_rate",
        "light_success_rate",
        "cluttered_success_rate",
        "tableheight_success_rate",
        "hard_success_rate",
    ]
    launches.clear()

    def unexpected_popen(*args, **kwargs):
        raise AssertionError("valid resume must not launch a worker")

    monkeypatch.setattr(MANAGER.subprocess, "Popen", unexpected_popen)
    MANAGER.main.__wrapped__(cfg)
    assert launches == []


def test_manager_runs_easy_hard_subset_writes_summary_and_checks_resume_identity(
    monkeypatch, tmp_path
):
    cfg, _, output_dir = _make_manager_cfg(tmp_path)
    cfg.EVALUATION.c2r_configs = ["easy", "hard"]
    launches: list[str] = []
    _install_successful_workers(monkeypatch, launches, rate=0.5)
    MANAGER.main.__wrapped__(cfg)
    assert launches == ["demo_clean", "demo_randomized"]
    manifest = json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert all(
        (
            item["task_config_sha256"]
            == MANAGER._sha256_file(
                Path(cfg.EVALUATION.robotwin_root)
                / "task_config"
                / (item["task_config"] + ".yml")
            )
            for item in manifest["evaluation"]["configs"]
        )
    )
    assert [
        {k: v for k, v in item.items() if k != "task_config_sha256"}
        for item in manifest["evaluation"]["configs"]
    ] == [
        {"name": "easy", "task_config": "demo_clean", "result_suffix": "clean"},
        {"name": "hard", "task_config": "demo_randomized", "result_suffix": "random"},
    ]
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["completeness"] == {
        "expected_results": 2,
        "completed_results": 2,
        "missing": [],
    }
    assert summary["overall"] == {
        "mean_success_rate": {"easy": 0.5, "hard": 0.5},
        "retention": {"hard": 1.0},
    }
    with (output_dir / "summary.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == [
        "task_name",
        "easy_success_rate",
        "hard_success_rate",
        "hard_retention",
    ]

    def unexpected_popen(*args, **kwargs):
        raise AssertionError("valid subset resume must not launch a worker")

    monkeypatch.setattr(MANAGER.subprocess, "Popen", unexpected_popen)
    MANAGER.main.__wrapped__(cfg)
    cfg.EVALUATION.c2r_configs = ["easy", "bg"]
    with pytest.raises(ValueError, match="Resume identity mismatch"):
        MANAGER.main.__wrapped__(cfg)


@pytest.mark.parametrize(
    ("invalid_configs", "message"),
    [
        ([], "must not be empty"),
        (["easy", "easy"], "must not contain duplicates"),
        (["easy", "unknown"], "Unsupported C2R configs"),
    ],
)
def test_manager_rejects_invalid_config_subset_before_creating_output(
    monkeypatch, tmp_path, invalid_configs, message
):
    cfg, _, output_dir = _make_manager_cfg(tmp_path)
    cfg.EVALUATION.c2r_configs = invalid_configs
    with pytest.raises(ValueError, match=message):
        MANAGER.main.__wrapped__(cfg)
    assert not output_dir.exists()


def test_manager_reruns_only_corrupt_resume_result(monkeypatch, tmp_path):
    cfg, _, output_dir = _make_manager_cfg(tmp_path)
    launches: list[str] = []
    _install_successful_workers(monkeypatch, launches)
    MANAGER.main.__wrapped__(cfg)
    corrupt_file = (
        output_dir / "beat_block_hammer" / MANAGER._config_result_filename("cluttered")
    )
    corrupt_file.write_text("truncated", encoding="utf-8")
    launches.clear()
    _install_successful_workers(monkeypatch, launches, rate=0.25)
    MANAGER.main.__wrapped__(cfg)
    assert launches == ["eval_c2r_cluttered"]
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["completeness"]["missing"] == []
    assert summary["overall"]["mean_success_rate"]["cluttered"] == pytest.approx(0.25)


def test_manager_rejects_resume_with_changed_identity(monkeypatch, tmp_path):
    cfg, _, _ = _make_manager_cfg(tmp_path)
    launches: list[str] = []
    _install_successful_workers(monkeypatch, launches)
    MANAGER.main.__wrapped__(cfg)
    cfg.EVALUATION.instruction_type = "unseen"
    with pytest.raises(ValueError, match="Resume identity mismatch"):
        MANAGER.main.__wrapped__(cfg)


def test_manager_rejects_changed_runtime_config_on_resume(monkeypatch, tmp_path):
    cfg, release_root, output_dir = _make_manager_cfg(tmp_path)
    _install_successful_workers(monkeypatch, [])
    MANAGER.main.__wrapped__(cfg)
    manifest_before = (output_dir / "run_manifest.json").read_bytes()
    task_config = release_root / "task_config/demo_clean.yml"
    task_config.write_text("domain_randomization: {random_background: true}\n")
    with pytest.raises(ValueError, match="Resume identity mismatch"):
        MANAGER.main.__wrapped__(cfg)
    assert (output_dir / "run_manifest.json").read_bytes() == manifest_before


def test_full_mode_requires_exactly_fifty_official_tasks(monkeypatch, tmp_path):
    cfg, _, _ = _make_manager_cfg(tmp_path, task_name=None)
    with pytest.raises(ValueError, match="exactly 50 tasks"):
        MANAGER.main.__wrapped__(cfg)


def test_policy_symlink_creation_tolerates_same_target_race_and_rejects_conflict(
    monkeypatch, tmp_path
):
    release_root = tmp_path / "release"
    (release_root / "policy").mkdir(parents=True)
    policy_source = tmp_path / "mtwam_policy"
    policy_source.mkdir()
    policy_target = release_root / "policy" / "mtwam_policy"
    original_symlink_to = Path.symlink_to

    def simulate_same_target_race(self, target, *, target_is_directory=False):
        original_symlink_to(self, target, target_is_directory=target_is_directory)
        raise FileExistsError("another worker created the policy symlink")

    monkeypatch.setattr(Path, "symlink_to", simulate_same_target_race)
    result = SINGLE._ensure_policy_symlink(release_root, policy_source)
    assert result == policy_target
    assert result.resolve() == policy_source.resolve()
    conflicting_source = tmp_path / "conflicting_policy"
    conflicting_source.mkdir()
    policy_target.unlink()
    original_symlink_to(policy_target, conflicting_source, target_is_directory=True)
    with pytest.raises(RuntimeError, match="Policy symlink conflict"):
        SINGLE._ensure_policy_symlink(release_root, policy_source)


def test_single_entry_uses_mtwam_adapter_with_external_release_cwd(
    monkeypatch, tmp_path
):
    release_root = tmp_path / "external-release"
    (release_root / "policy").mkdir(parents=True)
    ckpt = tmp_path / "step.pt"
    ckpt.write_bytes(b"checkpoint")
    stats = tmp_path / "dataset_stats.json"
    stats.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "stable-output"
    cfg = OmegaConf.create(
        {
            "ckpt": str(ckpt),
            "gpu_id": 0,
            "seed": 0,
            "mixed_precision": "bf16",
            "EVALUATION": {
                "robotwin_root": str(release_root),
                "policy_name": "mtwam_policy",
                "task_name": "beat_block_hammer",
                "task_config": "eval_c2r_bg",
                "instruction_type": "seen",
                "eval_num_episodes": 1,
                "output_dir": str(output_dir),
                "dataset_stats_path": str(stats),
                "device": "cuda",
                "action_horizon": None,
                "replan_steps": 8,
                "num_inference_steps": 10,
                "sigma_shift": None,
                "text_cfg_scale": 1.0,
                "negative_prompt": "",
                "rand_device": "cpu",
                "tiled": False,
                "timing_enabled": False,
                "skip_get_obs_within_replan": True,
            },
        }
    )
    with initialize_config_dir(
        version_base="1.3", config_dir=str(PROJECT_ROOT / "configs")
    ):
        cfg = OmegaConf.merge(compose(config_name="eval_robotwin"), cfg)
    captured = {}

    class FakeProcess:
        stdout = io.StringIO("worker-output\n")

        def wait(self):
            return 0

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(SINGLE.subprocess, "Popen", fake_popen)
    SINGLE.main.__wrapped__(cfg)
    assert captured["cmd"][2] == str(SINGLE.EVAL_POLICY_ENTRY)
    assert Path(captured["cwd"]) == release_root
    output_index = captured["cmd"].index("--eval_output_dir")
    assert ast.literal_eval(captured["cmd"][output_index + 1]) == str(
        output_dir / "beat_block_hammer"
    )
    cfg_index = captured["cmd"].index("--sim_cfg_path")
    loaded = OmegaConf.load(ast.literal_eval(captured["cmd"][cfg_index + 1]))
    assert loaded.model.dynamic_branch_dual_f0 is True
    assert loaded.data.train.processor.action_output_dim == 14
    assert loaded.EVALUATION.replan_steps == 8
    assert "--model_overrides" not in captured["cmd"]
    assert (release_root / "policy" / "mtwam_policy").is_symlink()
    assert list(output_dir.glob("eval_beat_block_hammer_bg_*.log"))
    assert (output_dir / "eval_config_beat_block_hammer_bg.yaml").is_file()


def test_vendored_eval_policy_accepts_all_six_task_configs():
    source_path = PROJECT_ROOT / "third_party/RoboTwin/script/eval_policy.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    function_node = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_result_suffix_from_task_config"
        )
    )
    namespace = {}
    module = ast.fix_missing_locations(
        ast.Module(body=[function_node], type_ignores=[])
    )
    exec(compile(module, str(source_path), "exec"), namespace)
    suffix = namespace["_result_suffix_from_task_config"]
    assert {
        task_config: suffix(task_config)
        for task_config in MANAGER.C2R_TASK_CONFIG_BY_NAME.values()
    } == {
        "demo_clean": "clean",
        "eval_c2r_bg": "bg",
        "eval_c2r_light": "light",
        "eval_c2r_cluttered": "cluttered",
        "eval_c2r_tableheight": "tableheight",
        "demo_randomized": "random",
    }
