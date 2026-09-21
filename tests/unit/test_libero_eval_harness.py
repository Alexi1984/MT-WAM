import importlib.util
import json
import subprocess
import sys
from pathlib import Path
import pytest
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from mtwam.utils.eval_config import evaluation_identity, save_resolved_config
from mtwam.utils.libero_results import read_tasks, result_complete, variant_result_done

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "libero_manager_tests", ROOT / "experiments/libero/run_libero_manager.py"
)
MANAGER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MANAGER)


def make_config(tmp_path):
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="eval_libero_plus")
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"checkpoint")
    stats = tmp_path / "dataset_stats.json"
    stats.write_text("{}")
    tasks = tmp_path / "tasks.txt"
    tasks.write_text("libero_10,0\nlibero_10,1\nlibero_goal,0\nlibero_goal,1\n")
    cfg.ckpt = str(checkpoint)
    cfg.EVALUATION.dataset_stats_path = str(stats)
    cfg.EVALUATION.output_dir = str(tmp_path / "results")
    cfg.MULTIRUN.task_file = str(tasks)
    cfg.MULTIRUN.num_gpus = 1
    cfg.MULTIRUN.max_tasks_per_gpu = 2
    return cfg


def complete_result(suite, task_id, cfg, gpu_id=0):
    return dict(
        task_suite=suite,
        task_id=task_id,
        task_description="task",
        successes=1,
        total_episodes=1,
        gpu_id=gpu_id,
        success_episodes=[0],
        failure_episodes=[],
        start_time="test",
        duration=1.0,
        seed=int(cfg.seed),
        configuration_sha256=evaluation_identity(cfg),
    )


def test_result_validation_requires_matching_complete_episodes(tmp_path):
    cfg = make_config(tmp_path)
    good = complete_result("libero_10", 0, cfg)
    digest = evaluation_identity(cfg)
    assert result_complete(good, "libero_10", 0, 1, int(cfg.seed), digest)
    for update in (
        {"seed": 0},
        {"task_suite": "libero_goal"},
        {"success_episodes": []},
        {"failure_episodes": [0]},
        {"total_episodes": 2},
        {"successes": 0},
        {"configuration_sha256": "wrong"},
        {"duration": float("nan")},
    ):
        assert not result_complete(
            {**good, **update}, "libero_10", 0, 1, int(cfg.seed), digest
        )
    assert not result_complete(
        {"task_id": 0, "duration": 1}, "libero_10", 0, 1, int(cfg.seed), digest
    )


def test_manager_preserves_config_and_reruns_only_incomplete_tasks(
    monkeypatch, tmp_path
):
    cfg = make_config(tmp_path)
    cfg.EVALUATION.negative_prompt = "don't move [left], keep (x)"
    launched = []
    summaries = []

    class Process:
        def poll(self):
            return 0

    def fake_popen(command, **kwargs):
        config_dir = Path(command[command.index("--config-path") + 1])
        config_name = command[command.index("--config-name") + 1]
        worker = OmegaConf.load(config_dir / (config_name + ".yaml"))
        assert worker.model.dynamic_branch_dual_f0 is True
        assert worker.EVALUATION.negative_prompt == cfg.EVALUATION.negative_prompt
        task_file = json.loads(
            next(
                (
                    v.split("=", 1)[1]
                    for v in command
                    if v.startswith("EVALUATION.worker_task_file=")
                )
            )
        )
        tasks = read_tasks(task_file)
        launched.extend(tasks)
        for suite, task_id in tasks:
            output = (
                Path(worker.EVALUATION.output_dir)
                / suite
                / f"gpu0_task{task_id}_results.json"
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(complete_result(suite, task_id, worker)))
        return Process()

    monkeypatch.setattr(MANAGER.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        MANAGER.subprocess, "run", lambda command, **kwargs: summaries.append(command)
    )
    MANAGER.main.__wrapped__(cfg)
    assert set(launched) == set(read_tasks(cfg.MULTIRUN.task_file))
    assert "--classification-sidecar" in summaries[-1]
    launched.clear()
    MANAGER.main.__wrapped__(cfg)
    assert launched == []
    broken = Path(cfg.EVALUATION.output_dir) / "libero_goal/gpu0_task1_results.json"
    broken.write_text("{}")
    MANAGER.main.__wrapped__(cfg)
    assert launched == [("libero_goal", 1)]
    snapshot = Path(cfg.EVALUATION.output_dir) / "workers/evaluation.yaml"
    before = snapshot.read_bytes()
    cfg.seed += 1
    with pytest.raises(ValueError, match="Resume identity mismatch"):
        MANAGER.main.__wrapped__(cfg)
    assert snapshot.read_bytes() == before


def test_libero_worker_reads_snapshot_through_real_hydra(tmp_path):
    cfg = make_config(tmp_path)
    cfg.paths.wan = str(tmp_path / "model files, (main)")
    snapshot = save_resolved_config(cfg, tmp_path / "snapshots/evaluation.yaml", ROOT)
    tasks = Path(cfg.MULTIRUN.task_file)
    command = MANAGER.build_worker_command(snapshot, tasks, 0) + [
        "--cfg",
        "job",
        "--resolve",
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    worker = yaml.safe_load(result.stdout)
    stored = OmegaConf.to_container(OmegaConf.load(snapshot), resolve=True)
    for key in ("model", "data", "mixed_precision", "seed"):
        assert worker[key] == stored[key]
    assert worker["EVALUATION"]["worker_task_file"] == str(tasks)


def test_full_manifest_and_axis_sidecar_have_identical_task_sets():
    import csv

    manifest = ROOT / "experiments/libero/ood_manifests/ood_variants_full10030.txt"
    sidecar = manifest.with_name("ood_variants_full10030_sidecar.tsv")
    tasks = set(read_tasks(manifest))
    with sidecar.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(tasks) == len(rows) == 10030
    assert tasks == {(row["suite"], int(row["task_id"])) for row in rows}
    assert len({row["axis"] for row in rows}) == 7


def test_observation_preprocessing_preserves_camera_orientation_and_state():
    import numpy as np
    import torch
    from types import SimpleNamespace
    from experiments.libero import eval_libero_single as worker

    height, width = 224, 224
    image = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)
    wrist = image[::-1].copy()
    angle = np.pi / 3
    obs = {
        "agentview_image": image,
        "robot0_eye_in_hand_image": wrist,
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
        "robot0_eef_quat": np.array([0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)]),
        "robot0_gripper_qpos": np.array([0.04, -0.04]),
    }
    captured = []

    def transform(batch):
        captured.append(batch["state"]["default"].clone())
        return batch

    def normalize(batch):
        batch["state"]["default"] = batch["state"]["default"] * 2 - 0.5
        return batch

    processor = SimpleNamespace(
        num_output_cameras=2,
        shape_meta={
            "images": [{"shape": [3, height, width]}] * 2,
            "state": [{"key": "default"}],
        },
        action_state_transform=transform,
        normalizer=SimpleNamespace(forward=normalize),
    )
    cfg = OmegaConf.create({"data": {"train": {"concat_multi_camera": "horizontal"}}})
    model_image, proprio, images = worker._obs_to_model_input(
        obs,
        cfg,
        processor,
        width * 2,
        height,
        "cpu",
        torch.float32,
    )
    expected_rgb = np.concatenate([image[::-1, ::-1], wrist[::-1, ::-1]], axis=1)
    expected_image = (
        torch.tensor(expected_rgb).permute(2, 0, 1).unsqueeze(0).float() * (2.0 / 255.0)
        - 1
    )
    assert torch.equal(model_image, expected_image)
    expected_state = torch.tensor(
        [[0.1, 0.2, 0.3, 0.0, 0.0, angle, 0.04, -0.04]], dtype=torch.float32
    )
    torch.testing.assert_close(captured[0], expected_state, atol=1e-6, rtol=1e-5)
    assert torch.equal(proprio, captured[0] * 2 - 0.5)
    assert np.array_equal(images["image"], image[::-1, ::-1])
