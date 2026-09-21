from mtwam.utils.eval_config import find_dataset_stats
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import hydra
import yaml
from mtwam.utils.eval_config import save_resolved_config, evaluation_identity
from omegaconf import DictConfig, ListConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SINGLE_ENTRY = PROJECT_ROOT / "experiments" / "robotwin" / "eval_robotwin_single.py"
TERMINATE_TIMEOUT_SEC = 10
POLL_INTERVAL_SEC = 2
EXPECTED_FULL_TASK_COUNT = 50
RUN_MANIFEST_SCHEMA_VERSION = 1
C2R_CONFIGS: tuple[tuple[str, str, str], ...] = (
    ("easy", "demo_clean", "clean"),
    ("bg", "eval_c2r_bg", "bg"),
    ("light", "eval_c2r_light", "light"),
    ("cluttered", "eval_c2r_cluttered", "cluttered"),
    ("tableheight", "eval_c2r_tableheight", "tableheight"),
    ("hard", "demo_randomized", "random"),
)
C2R_CONFIG_NAMES = tuple((item[0] for item in C2R_CONFIGS))
C2R_TASK_CONFIG_BY_NAME = {item[0]: item[1] for item in C2R_CONFIGS}
C2R_RESULT_SUFFIX_BY_NAME = {item[0]: item[2] for item in C2R_CONFIGS}


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path.resolve()


def _resolve_optional_path(path_value: Any, *, base: Path) -> Path | None:
    if path_value is None:
        return None
    text = str(path_value).strip()
    if text == "" or text.lower() in {"none", "null"}:
        return None
    return _resolve_path(text, base=base)


def _resolve_dataset_stats_path(cfg: DictConfig, ckpt_path: Path) -> Path:
    value = cfg.EVALUATION.get("dataset_stats_path")
    explicit = _resolve_path(str(value), base=PROJECT_ROOT) if value else None
    return find_dataset_stats(ckpt_path, explicit)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hydra_string_override(key: str, value: Any) -> str:
    return f"{key}={json.dumps(str(value), ensure_ascii=False)}"


def _resolve_c2r_config_names(raw_value: Any) -> tuple[str, ...]:
    if raw_value is None:
        return C2R_CONFIG_NAMES
    if not isinstance(raw_value, (list, tuple, ListConfig)):
        raise ValueError("`EVALUATION.c2r_configs` must be a non-empty list or null.")
    config_names = tuple((str(value) for value in raw_value))
    if not config_names:
        raise ValueError("`EVALUATION.c2r_configs` must not be empty.")
    if len(set(config_names)) != len(config_names):
        raise ValueError("`EVALUATION.c2r_configs` must not contain duplicates.")
    unsupported = [name for name in config_names if name not in C2R_CONFIG_NAMES]
    if unsupported:
        raise ValueError(
            f"Unsupported C2R configs: {unsupported}; supported={list(C2R_CONFIG_NAMES)}"
        )
    return config_names


def _task_list_file(robotwin_root: Path) -> Path:
    return robotwin_root / "task_config" / "_eval_step_limit.yml"


def _load_all_tasks(robotwin_root: Path) -> list[str]:
    task_list_file = _task_list_file(robotwin_root)
    if not task_list_file.is_file():
        raise FileNotFoundError(f"Task list file not found: {task_list_file}")
    with task_list_file.open("r", encoding="utf-8") as f:
        task_map = yaml.safe_load(f)
    if not isinstance(task_map, dict) or len(task_map) == 0:
        raise ValueError(f"Invalid task map in: {task_list_file}")
    tasks: list[str] = []
    seen: set[str] = set()
    for raw_task in task_map:
        task = str(raw_task)
        if task in seen:
            raise ValueError(f"Duplicate task in {task_list_file}: {task}")
        seen.add(task)
        tasks.append(task)
    return tasks


def _validate_c2r_task_configs(
    robotwin_root: Path, config_names: tuple[str, ...] = C2R_CONFIG_NAMES
) -> None:
    missing = [
        C2R_TASK_CONFIG_BY_NAME[config_name]
        for config_name in config_names
        if not (
            robotwin_root
            / "task_config"
            / f"{C2R_TASK_CONFIG_BY_NAME[config_name]}.yml"
        ).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing C2R task configs under {robotwin_root / 'task_config'}: {missing}"
        )


def _parse_success_rate(result_file: Path) -> float:
    if not result_file.is_file():
        raise FileNotFoundError(f"Result file not found: {result_file}")
    text = result_file.read_text(encoding="utf-8")
    last_value: float | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "":
            continue
        try:
            last_value = float(stripped)
        except ValueError:
            continue
    if (
        last_value is None
        or not math.isfinite(last_value)
        or (not 0.0 <= last_value <= 1.0)
    ):
        raise ValueError(f"Invalid success rate in: {result_file}")
    return float(last_value)


def _config_result_filename(config_name: str) -> str:
    try:
        suffix = C2R_RESULT_SUFFIX_BY_NAME[config_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported C2R config: {config_name}") from exc
    return f"_result_{suffix}.txt"


def _mean_or_none(values: list[float | None]) -> float | None:
    valid = [v for v in values if v is not None]
    if len(valid) == 0:
        return None
    return float(sum(valid) / len(valid))


def _retention_or_none(value: float | None, easy_value: float | None) -> float | None:
    if value is None or easy_value is None or easy_value <= 0.0:
        return None
    return float(value / easy_value)


def _to_jsonable(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _manifest_payload(
    *,
    ckpt_path: Path,
    dataset_stats_path: Path,
    robotwin_root: Path,
    task_list_file: Path,
    tasks: list[str],
    config_names: tuple[str, ...],
    cfg: DictConfig,
) -> dict[str, Any]:
    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "checkpoint": {
            "path": str(ckpt_path),
            "size_bytes": ckpt_path.stat().st_size,
            "sha256": _sha256_file(ckpt_path),
        },
        "dataset_stats": {
            "path": str(dataset_stats_path),
            "size_bytes": dataset_stats_path.stat().st_size,
            "sha256": _sha256_file(dataset_stats_path),
        },
        "robotwin": {
            "root": str(robotwin_root),
            "task_list_path": str(task_list_file),
            "task_list_sha256": _sha256_file(task_list_file),
        },
        "evaluation": {
            "instruction_type": str(cfg.EVALUATION.instruction_type),
            "eval_num_episodes": int(cfg.EVALUATION.eval_num_episodes),
            "seed": int(cfg.seed),
            "configuration_sha256": evaluation_identity(cfg),
            "configs": [
                {
                    "name": name,
                    "task_config": C2R_TASK_CONFIG_BY_NAME[name],
                    "task_config_sha256": _sha256_file(
                        robotwin_root
                        / "task_config"
                        / (C2R_TASK_CONFIG_BY_NAME[name] + ".yml")
                    ),
                    "result_suffix": C2R_RESULT_SUFFIX_BY_NAME[name],
                }
                for name in config_names
            ],
            "tasks": list(tasks),
        },
    }


def _write_or_validate_manifest(run_output_dir: Path, payload: dict[str, Any]) -> str:
    manifest_path = run_output_dir / "run_manifest.json"
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Invalid existing run manifest: {manifest_path}") from exc
        if existing != payload:
            raise ValueError(
                f"Resume identity mismatch for {run_output_dir}; existing run_manifest.json does not match the requested checkpoint/config/task set."
            )
        return "validated"
    existing_entries = [
        p for p in run_output_dir.iterdir() if p.name != "run_manifest.json"
    ]
    if existing_entries:
        raise RuntimeError(
            f"Non-empty output directory is missing run_manifest.json: {run_output_dir}"
        )
    temp_path = run_output_dir / ".run_manifest.json.tmp"
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temp_path, manifest_path)
    return "created"


def _next_missing_config(task_rates: dict[str, float | None]) -> str | None:
    for config_name, success_rate in task_rates.items():
        if success_rate is None:
            return config_name
    return None


@dataclass
class RunningState:
    task_name: str
    gpu_id: int
    config_name: str
    process: subprocess.Popen[str]


@hydra.main(
    version_base="1.3", config_path="../../configs", config_name="eval_robotwin"
)
def main(cfg: DictConfig):
    if cfg.ckpt is None:
        raise ValueError("`ckpt` must not be None.")
    if not SINGLE_ENTRY.is_file():
        raise FileNotFoundError(f"Single evaluation entry not found: {SINGLE_ENTRY}")
    config_names = _resolve_c2r_config_names(cfg.EVALUATION.get("c2r_configs", None))
    ckpt_path = _resolve_path(str(cfg.ckpt), base=PROJECT_ROOT)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    dataset_stats_path = _resolve_dataset_stats_path(cfg, ckpt_path)
    robotwin_root = _resolve_path(str(cfg.EVALUATION.robotwin_root), base=PROJECT_ROOT)
    if not robotwin_root.is_dir():
        raise FileNotFoundError(f"RoboTwin root not found: {robotwin_root}")
    _validate_c2r_task_configs(robotwin_root, config_names)
    all_tasks = _load_all_tasks(robotwin_root)
    task_name_cfg = cfg.EVALUATION.task_name
    if task_name_cfg is None or str(task_name_cfg).strip() == "":
        if len(all_tasks) != EXPECTED_FULL_TASK_COUNT:
            raise ValueError(
                f"Full C2R evaluation requires exactly {EXPECTED_FULL_TASK_COUNT} tasks, got {len(all_tasks)} from {_task_list_file(robotwin_root)}"
            )
        tasks = all_tasks
    else:
        task_name = str(task_name_cfg)
        if task_name not in all_tasks:
            raise ValueError(
                f"Requested task is not in official task list: {task_name}"
            )
        tasks = [task_name]
    num_gpus = int(cfg.MULTIRUN.num_gpus)
    if num_gpus <= 0:
        raise ValueError("`MULTIRUN.num_gpus` must be > 0.")
    max_tasks_per_gpu = int(cfg.MULTIRUN.max_tasks_per_gpu)
    if max_tasks_per_gpu <= 0:
        raise ValueError("`MULTIRUN.max_tasks_per_gpu` must be > 0.")
    gpu_ids = list(range(num_gpus))
    run_output_dir = _resolve_path(str(cfg.EVALUATION.output_dir), base=PROJECT_ROOT)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    manager_log = run_output_dir / "manager.log"
    failed_tasks_file = run_output_dir / "failed_tasks.txt"
    summary_csv = run_output_dir / "summary.csv"
    summary_json = run_output_dir / "summary.json"
    manifest = _manifest_payload(
        ckpt_path=ckpt_path,
        dataset_stats_path=dataset_stats_path,
        robotwin_root=robotwin_root,
        task_list_file=_task_list_file(robotwin_root),
        tasks=tasks,
        config_names=config_names,
        cfg=cfg,
    )
    manifest_status = _write_or_validate_manifest(run_output_dir, manifest)
    worker_snapshot = save_resolved_config(
        cfg, run_output_dir / "workers" / "evaluation.yaml", PROJECT_ROOT
    )
    task_rates: dict[str, dict[str, float | None]] = {
        task: {config_name: None for config_name in config_names} for task in tasks
    }
    failed_records: list[dict[str, Any]] = []
    running_states: list[RunningState] = []

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with manager_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

    log(
        f"manager start tasks={len(tasks)} configs={list(config_names)} gpu_ids={gpu_ids} max_tasks_per_gpu={max_tasks_per_gpu} output_dir={run_output_dir} manifest={manifest_status}"
    )
    for task_name in tasks:
        for config_name in config_names:
            result_file = (
                run_output_dir / task_name / _config_result_filename(config_name)
            )
            if not result_file.exists():
                continue
            try:
                success_rate = _parse_success_rate(result_file)
            except Exception as exc:
                log(
                    f"resume invalid result -> rerun task={task_name} config={config_name} file={result_file} error={repr(exc)}"
                )
                continue
            task_rates[task_name][config_name] = success_rate
            log(
                f"resume skip task={task_name} config={config_name} success_rate={success_rate:.4f}"
            )
    pending_tasks = deque(
        (
            task_name
            for task_name in tasks
            if _next_missing_config(task_rates[task_name]) is not None
        )
    )

    def build_cmd(*, task_name: str, gpu_id: int, config_name: str) -> list[str]:
        task_config = C2R_TASK_CONFIG_BY_NAME[config_name]
        cmd = [
            sys.executable,
            str(SINGLE_ENTRY),
            "--config-path",
            str(worker_snapshot.parent),
            "--config-name",
            worker_snapshot.stem,
            _hydra_string_override("ckpt", str(ckpt_path)),
            f"gpu_id={gpu_id}",
            f"seed={int(cfg.seed)}",
            f"EVALUATION.task_name={task_name}",
            f"EVALUATION.task_config={task_config}",
            _hydra_string_override("EVALUATION.output_dir", str(run_output_dir)),
            _hydra_string_override(
                "EVALUATION.dataset_stats_path", str(dataset_stats_path)
            ),
            _hydra_string_override("EVALUATION.robotwin_root", str(robotwin_root)),
            f"EVALUATION.instruction_type={str(cfg.EVALUATION.instruction_type)}",
            f"EVALUATION.eval_num_episodes={int(cfg.EVALUATION.eval_num_episodes)}",
        ]
        return cmd

    def launch_config(task_name: str, gpu_id: int, config_name: str) -> RunningState:
        cmd = build_cmd(task_name=task_name, gpu_id=gpu_id, config_name=config_name)
        log(
            f"launch task={task_name} config={config_name} gpu={gpu_id} cmd={' '.join(cmd)}"
        )
        process = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), text=True)
        return RunningState(
            task_name=task_name, gpu_id=gpu_id, config_name=config_name, process=process
        )

    def terminate_all_running() -> None:
        for state in list(running_states):
            if state.process.poll() is not None:
                continue
            log(
                f"terminating task={state.task_name} config={state.config_name} gpu={state.gpu_id}"
            )
            state.process.terminate()
        deadline = time.time() + TERMINATE_TIMEOUT_SEC
        for state in list(running_states):
            if state.process.poll() is not None:
                continue
            remaining = max(0.0, deadline - time.time())
            try:
                state.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                log(
                    f"killing task={state.task_name} config={state.config_name} gpu={state.gpu_id}"
                )
                state.process.kill()
                state.process.wait()

    def gpu_running_count(gpu_id: int) -> int:
        return sum(
            (
                1
                for state in running_states
                if state.gpu_id == gpu_id and state.process.poll() is None
            )
        )

    def try_launch_pending(gpu_id: int) -> None:
        while pending_tasks and gpu_running_count(gpu_id) < max_tasks_per_gpu:
            task_name = pending_tasks.popleft()
            config_name = _next_missing_config(task_rates[task_name])
            if config_name is None:
                continue
            running_states.append(launch_config(task_name, gpu_id, config_name))

    def write_outputs() -> list[dict[str, str]]:
        overall_rates = {
            config_name: _mean_or_none([task_rates[t][config_name] for t in tasks])
            for config_name in config_names
        }
        missing = [
            {"task_name": task, "config": config_name}
            for task in tasks
            for config_name in config_names
            if task_rates[task][config_name] is None
        ]
        rate_headers = [f"{name}_success_rate" for name in config_names]
        retention_headers = [
            f"{name}_retention" for name in config_names if name != "easy"
        ]
        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task_name", *rate_headers, *retention_headers])
            for task in tasks:
                easy_rate = task_rates[task].get("easy")
                writer.writerow(
                    [task]
                    + [task_rates[task][name] for name in config_names]
                    + [
                        _retention_or_none(task_rates[task][name], easy_rate)
                        for name in config_names
                        if name != "easy"
                    ]
                )
            overall_easy = overall_rates.get("easy")
            writer.writerow(
                ["__overall__"]
                + [overall_rates[name] for name in config_names]
                + [
                    _retention_or_none(overall_rates[name], overall_easy)
                    for name in config_names
                    if name != "easy"
                ]
            )
        per_task = []
        for task in tasks:
            easy_rate = task_rates[task].get("easy")
            per_task.append(
                {
                    "task_name": task,
                    "success_rate": {
                        name: _to_jsonable(task_rates[task][name])
                        for name in config_names
                    },
                    "retention": {
                        name: _to_jsonable(
                            _retention_or_none(task_rates[task][name], easy_rate)
                        )
                        for name in config_names
                        if name != "easy"
                    },
                }
            )
        expected_count = len(tasks) * len(config_names)
        payload = {
            "manifest": manifest,
            "completeness": {
                "expected_results": expected_count,
                "completed_results": expected_count - len(missing),
                "missing": missing,
            },
            "per_task": per_task,
            "overall": {
                "mean_success_rate": {
                    name: _to_jsonable(overall_rates[name]) for name in config_names
                },
                "retention": {
                    name: _to_jsonable(
                        _retention_or_none(
                            overall_rates[name], overall_rates.get("easy")
                        )
                    )
                    for name in config_names
                    if name != "easy"
                },
            },
        }
        summary_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with failed_tasks_file.open("w", encoding="utf-8") as f:
            for rec in failed_records:
                f.write(
                    f"{rec['task_name']},{rec['config']},gpu={rec['gpu_id']},return_code={rec['return_code']},reason={rec['reason']}\n"
                )
        return missing

    for gpu_id in gpu_ids:
        try_launch_pending(gpu_id)
    has_failure = False
    failure_message = ""
    while running_states:
        progressed = False
        for state in list(running_states):
            gpu_id = state.gpu_id
            return_code = state.process.poll()
            if return_code is None:
                continue
            progressed = True
            running_states.remove(state)
            if return_code != 0:
                has_failure = True
                failure_message = f"worker failed: task={state.task_name}, config={state.config_name}, gpu={gpu_id}, return_code={return_code}"
                failed_records.append(
                    {
                        "task_name": state.task_name,
                        "config": state.config_name,
                        "gpu_id": gpu_id,
                        "return_code": return_code,
                        "reason": "process_failed",
                    }
                )
                log(failure_message)
                terminate_all_running()
                running_states.clear()
                break
            result_file = (
                run_output_dir
                / state.task_name
                / _config_result_filename(state.config_name)
            )
            try:
                success_rate = _parse_success_rate(result_file)
            except Exception as exc:
                has_failure = True
                failure_message = f"result parse failed: task={state.task_name}, config={state.config_name}, gpu={gpu_id}, error={repr(exc)}"
                failed_records.append(
                    {
                        "task_name": state.task_name,
                        "config": state.config_name,
                        "gpu_id": gpu_id,
                        "return_code": return_code,
                        "reason": "result_parse_failed",
                    }
                )
                log(failure_message)
                terminate_all_running()
                running_states.clear()
                break
            task_rates[state.task_name][state.config_name] = success_rate
            log(
                f"done task={state.task_name} config={state.config_name} gpu={gpu_id} success_rate={success_rate:.4f}"
            )
            next_config = _next_missing_config(task_rates[state.task_name])
            if next_config is not None:
                running_states.append(
                    launch_config(state.task_name, gpu_id, next_config)
                )
            else:
                try_launch_pending(gpu_id)
        if has_failure:
            break
        if not progressed:
            time.sleep(POLL_INTERVAL_SEC)
    if has_failure:
        for task_name in pending_tasks:
            failed_records.append(
                {
                    "task_name": task_name,
                    "config": "not_started",
                    "gpu_id": -1,
                    "return_code": -1,
                    "reason": "aborted_not_started",
                }
            )
    missing = write_outputs()
    log(f"summary saved: {summary_csv} and {summary_json}")
    if has_failure:
        raise RuntimeError(failure_message)
    if missing:
        raise RuntimeError(
            f"Completeness gate failed with {len(missing)} missing results"
        )
    log("manager finished successfully")


if __name__ == "__main__":
    main()
