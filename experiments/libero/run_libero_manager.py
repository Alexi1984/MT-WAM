import json
import os
import subprocess
import sys
import time
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from mtwam.utils.eval_config import (
    evaluation_identity,
    file_sha256,
    find_dataset_stats,
    resolve_path,
    resolved_config,
    save_resolved_config,
)
from mtwam.utils.libero_results import read_tasks, variant_result_done

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKER = PROJECT_ROOT / "experiments/libero/eval_libero_single.py"


def create_task_file(output_file, task_suite_names):
    from libero.libero import benchmark

    registry = benchmark.get_benchmark_dict()
    tasks = [
        (name, task_id)
        for name in task_suite_names
        for task_id in range(int(registry[name]().n_tasks))
    ]
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        "".join((f"{suite},{task_id}\n" for suite, task_id in tasks))
    )
    return output_file


def build_worker_command(snapshot, task_file, gpu_id):
    return [
        sys.executable,
        str(WORKER),
        "--config-path",
        str(snapshot.parent),
        "--config-name",
        snapshot.stem,
        f"gpu_id={gpu_id}",
        "EVALUATION.worker_task_file=" + json.dumps(str(task_file)),
    ]


def run_evaluation(cfg, snapshot, tasks, output_dir):
    digest = evaluation_identity(OmegaConf.load(snapshot))
    tasks = [
        (suite, task_id)
        for suite, task_id in tasks
        if not variant_result_done(
            output_dir,
            suite,
            task_id,
            int(cfg.EVALUATION.num_trials),
            int(cfg.seed),
            digest,
        )
    ]
    if not tasks:
        (output_dir / "failed_tasks.txt").unlink(missing_ok=True)
        print("All requested tasks already have complete matching results.")
        return
    num_gpus = int(cfg.MULTIRUN.num_gpus)
    per_gpu = int(cfg.MULTIRUN.max_tasks_per_gpu)
    if num_gpus < 1 or per_gpu < 1:
        raise ValueError("MULTIRUN.num_gpus and max_tasks_per_gpu must be positive.")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_ids = visible.split(",") if visible else [str(i) for i in range(num_gpus)]
    if len(gpu_ids) < num_gpus:
        raise ValueError("MULTIRUN.num_gpus exceeds CUDA_VISIBLE_DEVICES.")
    count = min(len(tasks), num_gpus * per_gpu)
    running = []
    failures = []
    try:
        for index in range(count):
            chunk = output_dir / "workers" / f"tasks_{index}.txt"
            chunk.write_text(
                "".join(
                    (f"{suite},{task_id}\n" for suite, task_id in tasks[index::count])
                )
            )
            gpu_id = index % num_gpus
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=gpu_ids[gpu_id], PYTHONUNBUFFERED="1")
            log = (chunk.parent / f"worker_{index}.log").open("a")
            try:
                process = subprocess.Popen(
                    build_worker_command(snapshot, chunk, gpu_id),
                    cwd=PROJECT_ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            except BaseException:
                log.close()
                raise
            running.append((process, log, index))
        while running:
            for state in running[:]:
                process, log, index = state
                code = process.poll()
                if code is None:
                    continue
                log.close()
                running.remove(state)
                if code:
                    failures.append(index)
            if running:
                time.sleep(0.2)
    finally:
        for process, log, index in running:
            if process.poll() is None:
                process.terminate()
        for process, log, index in running:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()
    incomplete = [
        (suite, task_id)
        for suite, task_id in tasks
        if not variant_result_done(
            output_dir,
            suite,
            task_id,
            int(cfg.EVALUATION.num_trials),
            int(cfg.seed),
            digest,
        )
    ]
    if failures or incomplete:
        (output_dir / "failed_tasks.txt").write_text(
            "".join((f"{suite},{task_id}\n" for suite, task_id in incomplete))
        )
        raise RuntimeError(
            f"LIBERO evaluation incomplete: failed workers={failures}, incomplete tasks={len(incomplete)}."
        )
    failed_path = output_dir / "failed_tasks.txt"
    if failed_path.exists():
        failed_path.unlink()


@hydra.main(version_base="1.3", config_path="../../configs", config_name="eval_libero")
def main(cfg: DictConfig):
    output_dir = resolve_path(cfg.EVALUATION.output_dir, PROJECT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    configured = cfg.MULTIRUN.get("task_file")
    if configured:
        task_file = resolve_path(configured, PROJECT_ROOT)
        if not task_file.is_file():
            raise FileNotFoundError(f"Task manifest not found: {task_file}")
    else:
        task_file = create_task_file(
            output_dir / "tasks.txt", cfg.MULTIRUN.task_suite_names
        )
    tasks = read_tasks(task_file)
    if cfg.MULTIRUN.get("create_only", False):
        print(f"Task manifest: {task_file} ({len(tasks)} tasks)")
        return
    if cfg.ckpt is None:
        raise ValueError("Set ckpt to an MT-WAM checkpoint.")
    checkpoint = resolve_path(cfg.ckpt, PROJECT_ROOT)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    explicit_stats = cfg.EVALUATION.get("dataset_stats_path")
    stats = find_dataset_stats(
        checkpoint,
        resolve_path(explicit_stats, PROJECT_ROOT) if explicit_stats else None,
    )
    cfg.ckpt = str(checkpoint)
    cfg.EVALUATION.dataset_stats_path = str(stats)
    manifest = dict(
        schema_version=1,
        checkpoint_sha256=file_sha256(checkpoint),
        statistics_sha256=file_sha256(stats),
        configuration_sha256=evaluation_identity(resolved_config(cfg, PROJECT_ROOT)),
        tasks=[list(task) for task in tasks],
    )
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError(
                "Resume identity mismatch; choose a new EVALUATION.output_dir."
            )
    else:
        if any(output_dir.glob("*/gpu*_task*_results.json")):
            raise ValueError(
                "Existing results have no run manifest; choose a new EVALUATION.output_dir."
            )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    snapshot = save_resolved_config(
        cfg, output_dir / "workers" / "evaluation.yaml", PROJECT_ROOT
    )
    print(f"Running {len(tasks)} LIBERO tasks from {snapshot}")
    run_evaluation(cfg, snapshot, tasks, output_dir)
    summary_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "experiments/libero/summarize_results.py"),
        "--output_dir",
        str(output_dir),
        "--manifest",
        str(task_file),
        "--strict",
    ]
    if cfg.benchmark == "libero_plus":
        summary_cmd.extend(
            [
                "--classification-sidecar",
                str(
                    PROJECT_ROOT
                    / "experiments/libero/ood_manifests/ood_variants_full10030_sidecar.tsv"
                ),
            ]
        )
    subprocess.run(summary_cmd, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
