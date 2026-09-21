import hashlib
import json
import os
from pathlib import Path
from omegaconf import OmegaConf


def resolve_path(value, project_root):
    path = Path(os.path.expandvars(str(value))).expanduser()
    return (path if path.is_absolute() else Path(project_root) / path).resolve()


def resolved_config(cfg, project_root):
    resolved = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    for key, value in resolved.get("paths", {}).items():
        if value is not None:
            resolved.paths[key] = str(resolve_path(value, project_root))
    for key in (
        "ckpt",
        "output_dir",
        "EVALUATION.output_dir",
        "EVALUATION.dataset_stats_path",
        "EVALUATION.robotwin_root",
    ):
        value = OmegaConf.select(resolved, key)
        if value is not None:
            OmegaConf.update(resolved, key, str(resolve_path(value, project_root)))
    for split in ("train", "val"):
        for key in (
            "include_episodes_file",
            "exclude_episodes_file",
            "latent_cache_dir",
            "teacher_feature_cache_dir",
        ):
            full_key = f"data.{split}.{key}"
            value = OmegaConf.select(resolved, full_key)
            if value is not None:
                OmegaConf.update(
                    resolved, full_key, str(resolve_path(value, project_root))
                )
    payload = OmegaConf.to_container(resolved, resolve=True)
    payload["hydra"] = {
        "job": {"chdir": False},
        "run": {"dir": "."},
        "output_subdir": None,
    }
    return OmegaConf.create(payload)


def save_resolved_config(cfg, path, project_root):
    payload = resolved_config(cfg, project_root)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    OmegaConf.save(payload, temporary)
    temporary.replace(path)
    return path.resolve()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluation_identity(cfg):
    payload = OmegaConf.to_container(cfg, resolve=True)
    evaluation = dict(payload.get("EVALUATION", {}))
    for key in (
        "output_dir",
        "task_suite_name",
        "task_id",
        "task_name",
        "task_config",
        "worker_task_file",
    ):
        evaluation.pop(key, None)
    identity = {
        key: payload.get(key)
        for key in ("model", "data", "mixed_precision", "seed", "benchmark")
    }
    identity["evaluation"] = evaluation
    for key, value in (
        ("checkpoint", payload.get("ckpt")),
        ("statistics", evaluation.get("dataset_stats_path")),
    ):
        path = Path(str(value)).expanduser().resolve() if value is not None else None
        if path is not None and path.is_file():
            stat = path.stat()
            identity[key] = {
                "path": str(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        else:
            identity[key] = value
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def find_dataset_stats(checkpoint, explicit=None):
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Dataset statistics not found: {path}")
        return path
    for parent in list(Path(checkpoint).resolve().parents)[:4]:
        path = parent / "dataset_stats.json"
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Set EVALUATION.dataset_stats_path to the checkpoint training statistics."
    )
