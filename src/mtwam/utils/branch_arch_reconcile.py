import logging
import os
from typing import Any, Optional, Tuple
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def train_cfg_path_for(ckpt_path: str) -> Tuple[Optional[str], list]:
    ckpt_abs = os.path.abspath(ckpt_path)
    run_dir = os.path.dirname(os.path.dirname(os.path.dirname(ckpt_abs)))
    candidates = [
        os.path.join(run_dir, "config.yaml"),
        os.path.join(os.path.dirname(ckpt_abs), "config.yaml"),
    ]
    return (next((p for p in candidates if os.path.exists(p)), None), candidates)


def _set_branch_horizon(model_cfg: Any, value: int) -> None:
    if isinstance(model_cfg, DictConfig):
        OmegaConf.update(
            model_cfg, "dynamic_branch_horizon", int(value), force_add=True
        )
    else:
        model_cfg["dynamic_branch_horizon"] = int(value)


def resolve_eval_branch_horizon(model_cfg: Any, ckpt_path: str, window_cfg: Any) -> int:
    cur = model_cfg.get("dynamic_branch_horizon", None)
    if cur is not None:
        return int(cur)
    cfg_path, _ = train_cfg_path_for(ckpt_path)
    if cfg_path is not None:
        train_h = (OmegaConf.load(cfg_path).get("model", None) or {}).get(
            "dynamic_branch_horizon", None
        )
        if train_h is not None:
            _set_branch_horizon(model_cfg, int(train_h))
            logger.info(
                "Eval branch horizon = %d (from training config %s)",
                int(train_h),
                cfg_path,
            )
            return int(train_h)
    from mtwam.runtime import window_p_max_from_cfg

    derived = int(window_p_max_from_cfg(window_cfg))
    _set_branch_horizon(model_cfg, derived)
    logger.info(
        "Eval branch horizon = %d (window-derived; no horizon in the training config)",
        derived,
    )
    return derived


ALLOW_MISSING_TRAIN_CONFIG_ENV = "MTWAM_ALLOW_MISSING_TRAIN_CONFIG"


def _nondefault_behavioral_keys(eval_model_cfg: Any) -> list:
    from mtwam.models.wan22._load_guards import BEHAVIORAL_BRANCH_KEYS, parse_bool

    out = []
    for key, default in BEHAVIORAL_BRANCH_KEYS.items():
        val = eval_model_cfg.get(key, default)
        if isinstance(default, bool):
            if parse_bool(val) != parse_bool(default):
                out.append(f"{key}={val!r}")
        elif str(val) != str(default):
            out.append(f"{key}={val!r}")
    return out


def check_train_eval_branch_arch(
    eval_model_cfg: Any,
    ckpt_path: str,
    eval_data_num_frames: int = 33,
    eval_data_num_extra_ref_frames: int = 0,
    allow_missing_train_config: bool = False,
) -> None:
    from mtwam.models.wan22._load_guards import check_branch_arch_consistency

    cfg_path, candidates = train_cfg_path_for(ckpt_path)
    if cfg_path is None:
        claimed = _nondefault_behavioral_keys(eval_model_cfg)
        allowed = allow_missing_train_config or os.environ.get(
            ALLOW_MISSING_TRAIN_CONFIG_ENV, ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        if claimed and (not allowed):
            raise ValueError(
                f"no training config.yaml next to the checkpoint, but eval composed non-default branch switches: {', '.join(claimed)}. Searched: {candidates}. Place the training config.yaml beside the checkpoint. Override: {ALLOW_MISSING_TRAIN_CONFIG_ENV}=1 or allow_missing_train_config=True."
            )
        if claimed:
            logger.warning(
                "Training config.yaml not found (searched %s); architecture check skipped by explicit override for %s.",
                candidates,
                ", ".join(claimed),
            )
        else:
            logger.info(
                "Training config.yaml not found (searched %s); using default branch settings.",
                candidates,
            )
        return
    train_cfg = OmegaConf.load(cfg_path)
    train_model = train_cfg.get("model", None) or {}
    train_data = (train_cfg.get("data", None) or {}).get("train", None) or {}
    errors = check_branch_arch_consistency(
        train_model,
        eval_model_cfg,
        train_data_k=train_data.get("num_extra_ref_frames", 0),
        eval_data_k=int(eval_data_num_extra_ref_frames),
        train_data_num_frames=train_data.get("num_frames", 33),
        eval_data_num_frames=int(eval_data_num_frames),
    )
    if errors:
        raise ValueError(
            "train/eval branch-arch mismatch:\n  - " + "\n  - ".join(errors)
        )
    logger.info("Branch-arch reconcile vs %s: OK", cfg_path)
