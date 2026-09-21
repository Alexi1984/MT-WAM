import logging
import os
import inspect
from pathlib import Path
import torch
from torch.utils.data import Subset
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image
import numpy as np
from einops import repeat
from omegaconf import OmegaConf
from .models.wan22._load_guards import strict_bool
from .utils.logging_config import get_logger, setup_logging
from .utils import misc

logger = get_logger(__name__)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    if not isinstance(mixed_precision, str):
        raise ValueError(f"`mixed_precision` must be str, got {type(mixed_precision)}")
    key = mixed_precision.strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def create_wan22_model(
    model_id: str,
    tokenizer_model_id: str,
    dit_config,
    tokenizer_max_len: int = 512,
    train_shift: float = 5.0,
    infer_shift: float = 5.0,
    num_train_timesteps: int = 1000,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.wan22.wan22 import Wan22Core

    if isinstance(dit_config, DictConfig):
        dit_config = OmegaConf.to_container(dit_config, resolve=True)
    if not isinstance(dit_config, dict):
        raise ValueError(f"`dit_config` must resolve to a dict, got {type(dit_config)}")
    return Wan22Core.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        redirect_common_files=bool(redirect_common_files),
        dit_config=dit_config,
        train_shift=float(train_shift),
        infer_shift=float(infer_shift),
        num_train_timesteps=int(num_train_timesteps),
    )


def resolve_branch_horizon(
    dynamic_branch_horizon,
    enable_dynamic_branch: bool,
    dynamic_branch_teacher_p_max=None,
):
    if dynamic_branch_horizon is not None:
        return int(dynamic_branch_horizon)
    if not enable_dynamic_branch:
        return None
    if dynamic_branch_teacher_p_max is not None:
        return int(dynamic_branch_teacher_p_max)
    raise ValueError(
        "dynamic_branch_horizon is unset and dynamic_branch_teacher_p_max is unavailable. Set an explicit horizon or call reconcile_branch_horizon(cfg) before constructing the model."
    )


def create_mtwam(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_dit_pretrained_path: str | None = None,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    enable_dynamic_branch: bool = True,
    dynamic_branch_num_layers: int = 10,
    dynamic_branch_read_text: bool = False,
    dynamic_branch_camera_layout: str = "horizontal",
    dynamic_branch_teacher_geometry: str | None = None,
    dynamic_branch_dual_f0: bool = True,
    dynamic_branch_ffn_moe: bool = True,
    dynamic_branch_ffn_moe_routing: str = "role",
    dynamic_branch_horizon: int = 2,
    dynamic_branch_teacher_p_max: int | None = None,
    num_extra_ref_frames: int = 0,
    num_frames: int | None = None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.wan22.mtwam import MTWAM

    if isinstance(video_dit_config, DictConfig):
        video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)
    if not isinstance(video_dit_config, dict):
        raise ValueError(
            f"`video_dit_config` must resolve to a dict, got {type(video_dit_config)}"
        )
    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if action_dit_config is None:
        action_dit_config = {}
    if not isinstance(action_dit_config, dict):
        raise ValueError(
            f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}"
        )
    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if video_scheduler is None:
        video_scheduler = {}
    if not isinstance(video_scheduler, dict):
        raise ValueError(
            f"`video_scheduler` must be dict-like, got {type(video_scheduler)}"
        )
    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if action_scheduler is None:
        raise ValueError("`action_scheduler` is required for MTWAM.")
    if not isinstance(action_scheduler, dict):
        raise ValueError(
            f"`action_scheduler` must be dict-like, got {type(action_scheduler)}"
        )
    required_action_scheduler_keys = {
        "train_shift",
        "infer_shift",
        "num_train_timesteps",
    }
    missing_keys = required_action_scheduler_keys - set(action_scheduler.keys())
    if missing_keys:
        raise ValueError(
            f"`action_scheduler` missing required keys: {sorted(missing_keys)}. Expected keys: train_shift, infer_shift, num_train_timesteps."
        )
    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)
    if loss is None:
        loss = {}
    if not isinstance(loss, dict):
        raise ValueError(f"`loss` must be dict-like, got {type(loss)}")
    return MTWAM.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=None if proprio_dim is None else int(proprio_dim),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        video_dit_pretrained_path=None
        if video_dit_pretrained_path in (None, "")
        else str(video_dit_pretrained_path),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        loss_lambda_traj=float(loss.get("lambda_traj", 0.5)),
        loss_lambda_tex=float(loss.get("lambda_tex", 0.25)),
        loss_traj_validity_mask=bool(loss.get("traj_validity_mask", True)),
        enable_dynamic_branch=strict_bool(
            enable_dynamic_branch, field="enable_dynamic_branch"
        ),
        dynamic_branch_num_layers=int(dynamic_branch_num_layers),
        dynamic_branch_read_text=strict_bool(
            dynamic_branch_read_text, field="dynamic_branch_read_text"
        ),
        dynamic_branch_camera_layout=str(dynamic_branch_camera_layout),
        dynamic_branch_teacher_geometry=None
        if dynamic_branch_teacher_geometry in (None, "")
        else str(dynamic_branch_teacher_geometry),
        dynamic_branch_dual_f0=strict_bool(
            dynamic_branch_dual_f0, field="dynamic_branch_dual_f0"
        ),
        dynamic_branch_ffn_moe=strict_bool(
            dynamic_branch_ffn_moe, field="dynamic_branch_ffn_moe"
        ),
        dynamic_branch_ffn_moe_routing=str(dynamic_branch_ffn_moe_routing),
        dynamic_branch_horizon=resolve_branch_horizon(
            dynamic_branch_horizon,
            strict_bool(enable_dynamic_branch, field="enable_dynamic_branch"),
            dynamic_branch_teacher_p_max,
        ),
        dynamic_branch_teacher_p_max=None
        if dynamic_branch_teacher_p_max is None
        else int(dynamic_branch_teacher_p_max),
        num_extra_ref_frames=int(num_extra_ref_frames),
        num_frames=None if num_frames is None else int(num_frames),
    )


def build_datasets(data_cfg: DictConfig):
    train_ds = instantiate(data_cfg.train)
    if data_cfg.get("val") is None:
        val_ds = train_ds
    else:
        train_stats_path = data_cfg.train.get("pretrained_norm_stats")
        default_stats_path = os.path.join(misc.get_work_dir(), "dataset_stats.json")
        val_stats_path = data_cfg.val.get("pretrained_norm_stats")
        pretrained_norm_stats = val_stats_path or train_stats_path or default_stats_path
        logger.info(
            "Building val dataset with pretrained_norm_stats: %s", pretrained_norm_stats
        )
        val_ds = instantiate(data_cfg.val, pretrained_norm_stats=pretrained_norm_stats)
    return (train_ds, val_ds)


def _resolve_train_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    device_count = torch.cuda.device_count()
    if device_count <= 1:
        return "cuda:0"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank < 0 or local_rank >= device_count:
        return "cuda:0"
    return f"cuda:{local_rank}"


def apply_train_subset(train_ds, train_subset_size):
    if train_subset_size is None:
        return train_ds
    n = int(train_subset_size)
    total = len(train_ds)
    if n > total:
        raise ValueError(
            f"train_subset_size={n} exceeds dataset size {total}; it must be <= the number of cached leading windows."
        )
    return Subset(train_ds, range(n))


def validate_branch_subset(enable_dynamic_branch, train_subset_size):
    if enable_dynamic_branch and train_subset_size is None:
        raise ValueError(
            "enable_dynamic_branch=True requires train_subset_size to be set to select windows with cached teacher features."
        )


def model_factory_extra_kwargs(
    target: str,
    camera_layout: str,
    k_data: int,
    teacher_geometry_override=None,
    dual_f0: bool = False,
    teacher_p_max=None,
    num_frames=None,
):
    from hydra.utils import get_method

    params = inspect.signature(get_method(str(target))).parameters
    extra = {}
    if "dynamic_branch_camera_layout" in params:
        extra["dynamic_branch_camera_layout"] = str(camera_layout)
    if teacher_p_max is not None and "dynamic_branch_teacher_p_max" in params:
        extra["dynamic_branch_teacher_p_max"] = int(teacher_p_max)
    if num_frames is not None and "num_frames" in params:
        extra["num_frames"] = int(num_frames)
    if "num_extra_ref_frames" in params:
        extra["num_extra_ref_frames"] = int(k_data)
    elif int(k_data):
        raise ValueError(
            f"num_extra_ref_frames={k_data} requested but model factory {target} does not accept it."
        )
    if teacher_geometry_override:
        if "dynamic_branch_teacher_geometry" in params:
            extra["dynamic_branch_teacher_geometry"] = str(teacher_geometry_override)
        else:
            raise ValueError(
                f"teacher_geometry_override={teacher_geometry_override!r} requested but model factory {target} does not accept dynamic_branch_teacher_geometry."
            )
    if dual_f0:
        if "dynamic_branch_dual_f0" in params:
            extra["dynamic_branch_dual_f0"] = True
        else:
            raise ValueError(
                f"dual_f0=True requested but model factory {target} does not accept dynamic_branch_dual_f0."
            )
    return extra


def validate_branch_dual_f0(enable_dynamic_branch, dual_f0, k_data):
    if not dual_f0:
        return
    if not enable_dynamic_branch:
        raise ValueError(
            "dynamic_branch_dual_f0=True requires enable_dynamic_branch=True."
        )
    if int(k_data or 0) > 0:
        raise ValueError(
            f"dynamic_branch_dual_f0=True supports K=0 only, got num_extra_ref_frames={k_data}."
        )


def window_p_max_from_cfg(data_cfg) -> int:
    from .datasets.lerobot.teacher_cache import teacher_p_max

    nf = int(data_cfg.get("num_frames", 33) or 33)
    ratio = int(data_cfg.get("action_video_freq_ratio", 4) or 4)
    k = int(data_cfg.get("num_extra_ref_frames", 0) or 0)
    num_sampled = (nf - 1) // ratio + 1
    anchor = k // ratio
    return teacher_p_max(num_sampled, anchor)


def validate_branch_horizon(enable_dynamic_branch, horizon, k_data, p_max=None):
    p_max = 2 if p_max is None else int(p_max)
    if int(horizon) == p_max:
        return
    if not enable_dynamic_branch:
        raise ValueError(
            f"dynamic_branch_horizon={horizon} (!= window P_max={p_max}) requires enable_dynamic_branch=True."
        )
    if int(k_data or 0) > 0:
        raise ValueError(
            f"dynamic_branch_horizon={horizon} (window P_max={p_max}) supports K=0 only, got num_extra_ref_frames={k_data}."
        )


def reconcile_branch_horizon(cfg) -> int:
    p_max = window_p_max_from_cfg(cfg.data.train)
    _hm = cfg.model.get("dynamic_branch_horizon", None)
    _hd = cfg.data.train.get("dynamic_branch_horizon", None)
    h_model = None if _hm is None else int(_hm)
    h_data = None if _hd is None else int(_hd)
    if h_model is not None and h_data is not None and (h_model != h_data):
        raise ValueError(
            f"dynamic_branch_horizon drift: model config says {h_model}, data config says {h_data}. Match the model and data settings."
        )
    h_eff = h_data if h_data is not None else h_model if h_model is not None else p_max
    if not 1 <= h_eff <= p_max:
        raise ValueError(
            f"dynamic_branch_horizon={h_eff} out of range [1, {p_max}] for this window (num_frames={cfg.data.train.get('num_frames', 33)}). Set a horizon within the available future latent frames."
        )
    OmegaConf.update(cfg, "model.dynamic_branch_horizon", h_eff, force_add=True)
    OmegaConf.update(cfg, "data.train.dynamic_branch_horizon", h_eff, force_add=True)
    if cfg.data.get("val") is not None:
        OmegaConf.update(cfg, "data.val.dynamic_branch_horizon", h_eff, force_add=True)
    return h_eff


def reconcile_teacher_geometry(cfg):
    tg_data = cfg.data.train.get("teacher_geometry_override", None) or None
    tg_model = cfg.model.get("dynamic_branch_teacher_geometry", None) or None
    if tg_model is not None and tg_data is not None and (str(tg_model) != str(tg_data)):
        raise ValueError(
            f"teacher_geometry_override drift: model config says {tg_model!r}, data config says {tg_data!r}. Match the model and data settings."
        )
    tg_effective = tg_data or tg_model
    if tg_effective is not None:
        OmegaConf.update(
            cfg,
            "model.dynamic_branch_teacher_geometry",
            str(tg_effective),
            force_add=True,
        )
        OmegaConf.update(
            cfg,
            "data.train.teacher_geometry_override",
            str(tg_effective),
            force_add=True,
        )
        if cfg.data.get("val") is not None:
            OmegaConf.update(
                cfg,
                "data.val.teacher_geometry_override",
                str(tg_effective),
                force_add=True,
            )
    return tg_effective


def validate_branch_eval_config(
    enable_dynamic_branch, lambda_traj, lambda_tex, eval_every, data_cfg=None
):
    if not (
        bool(enable_dynamic_branch)
        and (float(lambda_traj) > 0 or float(lambda_tex) > 0)
        and (int(eval_every) > 0)
    ):
        return
    val_cfg = None
    if data_cfg is not None:
        val_cfg = data_cfg.get("val")
        if val_cfg is None:
            val_cfg = data_cfg.get("train")
    if val_cfg is None:
        raise ValueError(
            f"eval_every={eval_every} with the dynamic branch enabled (lambda_traj={lambda_traj}, lambda_tex={lambda_tex}) requires a validation data config with teacher targets. Pass data_cfg or set eval_every=0."
        )
    missing = []
    if not bool(val_cfg.get("enable_dynamic_branch", False)):
        missing.append("enable_dynamic_branch=true")
    if not (
        bool(val_cfg.get("online_teacher", False))
        or val_cfg.get("teacher_feature_cache_dir")
    ):
        missing.append("online_teacher=true (or teacher_feature_cache_dir)")
    if missing:
        raise ValueError(
            f"eval_every={eval_every} with the dynamic branch enabled (lambda_traj={lambda_traj}, lambda_tex={lambda_tex}): validation data lacks {' + '.join(missing)}. Update the validation data configuration or set eval_every=0."
        )


def run_training(cfg: DictConfig):
    from .trainer import Wan22Trainer

    setup_logging(
        log_level=logging.INFO,
        is_main_process=torch.distributed.get_rank() == 0
        if torch.distributed.is_initialized()
        else True,
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    misc.register_work_dir(cfg.output_dir)
    horizon_effective = reconcile_branch_horizon(cfg)
    tg_effective = reconcile_teacher_geometry(cfg)
    config_payload = OmegaConf.to_container(cfg, resolve=True)
    with open(Path(cfg.output_dir) / "config.yaml", "w") as f:
        OmegaConf.save(config_payload, f)
    model_device = _resolve_train_device()
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    camera_layout = str(
        cfg.data.train.get("concat_multi_camera", "horizontal") or "horizontal"
    )
    k_data = int(cfg.data.train.get("num_extra_ref_frames", 0) or 0)
    k_model = int(cfg.model.get("num_extra_ref_frames", 0) or 0)
    if k_model not in (0, k_data):
        raise ValueError(
            f"num_extra_ref_frames drift: model config says {k_model}, data config says {k_data}. Match the model and data settings."
        )
    _loss_cfg = cfg.model.get("loss", None) or {}
    validate_branch_eval_config(
        cfg.model.get("enable_dynamic_branch", False),
        _loss_cfg.get("lambda_traj", 0.0),
        _loss_cfg.get("lambda_tex", 0.0),
        cfg.get("eval_every", 0),
        data_cfg=cfg.data,
    )
    _branch_on_cfg = strict_bool(
        cfg.model.get("enable_dynamic_branch", False), field="enable_dynamic_branch"
    )
    dual_f0 = strict_bool(
        cfg.model.get("dynamic_branch_dual_f0", False), field="dynamic_branch_dual_f0"
    )
    validate_branch_dual_f0(_branch_on_cfg, dual_f0, k_data)
    branch_p_max = window_p_max_from_cfg(cfg.data.train)
    validate_branch_horizon(
        cfg.model.get("enable_dynamic_branch", False),
        horizon_effective,
        k_data,
        p_max=branch_p_max,
    )
    model = instantiate(
        cfg.model,
        model_dtype=model_dtype,
        device=model_device,
        **model_factory_extra_kwargs(
            cfg.model._target_,
            camera_layout,
            k_data,
            teacher_geometry_override=tg_effective,
            dual_f0=dual_f0,
            teacher_p_max=branch_p_max,
            num_frames=int(cfg.data.train.get("num_frames", 33) or 33),
        ),
    )
    online_teacher = bool(cfg.data.train.get("online_teacher", False))
    if online_teacher:
        from mtwam.datasets.lerobot.teacher_extract import load_teacher_models

        dino_ckpt = cfg.get("dino_checkpoint", None)
        if dino_ckpt is None:
            raise ValueError(
                "online_teacher=True requires +dino_checkpoint=<DINOv2 ViT-B/14 .pth>."
            )
        cotracker_ckpt = cfg.get("cotracker_checkpoint", None)
        teacher_compile = cfg.get("teacher_compile", None)
        logger.info(
            "online_teacher: loading frozen CoTracker+DINO (cotracker=%s dino=%s) on %s (compile=%s)",
            cotracker_ckpt,
            dino_ckpt,
            model_device,
            teacher_compile,
        )
        model.teacher_models = load_teacher_models(
            cotracker_ckpt, dino_ckpt, device=model_device, compile_mode=teacher_compile
        )
    train_ds, val_ds = build_datasets(cfg.data)
    validate_branch_subset(
        _branch_on_cfg and (not online_teacher), cfg.get("train_subset_size")
    )
    train_ds = apply_train_subset(train_ds, cfg.get("train_subset_size"))
    trainer = Wan22Trainer(
        cfg=cfg, model=model, train_dataset=train_ds, val_dataset=val_ds
    )
    trainer.train()
