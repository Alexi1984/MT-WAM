from typing import Any, Iterable, List, Optional

BRANCH_KEY_PREFIX = "dynamic_branch."
_TRUE_TOKENS = frozenset({"true", "1", "yes", "on"})
_FALSE_TOKENS = frozenset({"false", "0", "no", "off"})


def parse_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value) if value in (0, 1) else None
    if isinstance(value, str):
        s = value.strip().lower()
        if s in _TRUE_TOKENS:
            return True
        if s in _FALSE_TOKENS:
            return False
    return None


def branch_fork_span(
    video_tokens_per_frame, num_ref_latents, dual_f0, video_seq_len=None
) -> int:
    ff = int(video_tokens_per_frame)
    if bool(dual_f0):
        return ff
    span = int(num_ref_latents) * ff
    return span if video_seq_len is None else min(span, int(video_seq_len))


def _nested_get(cfg: Any, *path: str) -> Any:
    cur = cfg
    for part in path:
        if cur is None or not hasattr(cur, "get"):
            return None
        cur = cur.get(part, None)
    return cur


def strict_bool(value: Any, *, field: str) -> bool:
    parsed = parse_bool(value)
    if parsed is None:
        raise ValueError(
            f"""{field}: cannot interpret {value!r} (type {type(value).__name__}) as a boolean. Accepted: real bools, 0/1, or the strings {sorted(_TRUE_TOKENS | _FALSE_TOKENS)} (case-insensitive)."""
        )
    return parsed


def check_mot_branch_load(
    payload_mot_keys: Iterable[str],
    model_mot_keys: Iterable[str],
    missing_keys: Iterable[str],
    unexpected_keys: Iterable[str],
    *,
    allow_uninitialized_branch: bool = False,
) -> List[str]:
    payload_branch = [k for k in payload_mot_keys if k.startswith(BRANCH_KEY_PREFIX)]
    model_branch = [k for k in model_mot_keys if k.startswith(BRANCH_KEY_PREFIX)]
    dropped = sorted((k for k in unexpected_keys if k.startswith(BRANCH_KEY_PREFIX)))
    missing_set = set(missing_keys)
    errors: List[str] = []
    if dropped or (payload_branch and (not model_branch)):
        errors.append(
            f"checkpoint carries {len(payload_branch)} `dynamic_branch.*` weights but the current model has no dynamic branch. Set enable_dynamic_branch=true. Sample dropped keys: {dropped[:3]}"
        )
    if model_branch and (not allow_uninitialized_branch):
        consumed = [k for k in model_branch if k not in missing_set]
        if not consumed:
            errors.append(
                f"model was built with a dynamic branch ({len(model_branch)} params) but the checkpoint provided none of its `dynamic_branch.*` weights. Load a checkpoint containing branch weights."
            )

    def _ffn_kind(branch_keys):
        if any((".ffn.experts." in k or ".ffn.gate." in k for k in branch_keys)):
            return "gated-MoE (experts.*/gate)"
        if any((".ffn.ffn_t." in k or ".ffn.ffn_s." in k for k in branch_keys)):
            return "role/scramble-MoE (ffn_t/ffn_s)"
        return "shared FFN"

    if payload_branch and model_branch:
        pk, mk = (_ffn_kind(payload_branch), _ffn_kind(model_branch))
        if pk != mk:
            errors.append(
                f"branch FFN structure mismatch: checkpoint is {pk} but the model is {mk}. Match dynamic_branch_ffn_moe and dynamic_branch_ffn_moe_routing to the checkpoint."
            )
    return errors


BEHAVIORAL_BRANCH_KEYS = {
    "enable_dynamic_branch": False,
    "dynamic_branch_num_layers": 10,
    "dynamic_branch_read_text": False,
    "dynamic_branch_action_reads_appearance": False,
    "dynamic_branch_ffn_moe": False,
    "dynamic_branch_ffn_moe_routing": "role",
    "dynamic_branch_num_experts": 4,
    "dynamic_branch_teacher_geometry": None,
    "dynamic_branch_dual_f0": False,
    "dynamic_branch_triple_f0": False,
    "dynamic_branch_horizon": 2,
}


def check_branch_arch_consistency(
    train_model_cfg,
    eval_model_cfg,
    train_data_k=None,
    eval_data_k=None,
    train_data_num_frames=None,
    eval_data_num_frames=None,
) -> List[str]:
    errors: List[str] = []
    for key, default in BEHAVIORAL_BRANCH_KEYS.items():
        t = train_model_cfg.get(key, default)
        e = eval_model_cfg.get(key, default)
        if isinstance(default, bool):
            tb, eb = (parse_bool(t), parse_bool(e))
            if tb is None or eb is None:
                bad = f"training={t!r}" if tb is None else f"eval={e!r}"
                errors.append(
                    f"{key}: cannot interpret {bad} as a boolean. Set model.{key} to a boolean."
                )
            elif tb != eb:
                errors.append(
                    f"{key}: training ran with {t!r} but eval composed {e!r}; configuration mismatch. Set model.{key}={tb} for evaluation."
                )
            continue
        if str(t) != str(e):
            errors.append(
                f"{key}: training ran with {t!r} but eval composed {e!r}; configuration mismatch. Set model.{key}={t} for evaluation."
            )
    t_mode = _nested_get(
        train_model_cfg, "video_dit_config", "video_attention_mask_mode"
    )
    e_mode = _nested_get(
        eval_model_cfg, "video_dit_config", "video_attention_mask_mode"
    )
    if t_mode is not None and e_mode is not None and (str(t_mode) != str(e_mode)):
        errors.append(
            f"video_attention_mask_mode: training ran with {t_mode!r} but eval composed {e_mode!r}. Set model.video_dit_config.video_attention_mask_mode={t_mode}."
        )
    train_loss = train_model_cfg.get("loss", None)
    if train_loss is not None and parse_bool(
        eval_model_cfg.get("enable_dynamic_branch", False)
    ):
        lt = float(train_loss.get("lambda_traj", 0.0) or 0.0)
        lx = float(train_loss.get("lambda_tex", 0.0) or 0.0)
        if lt <= 0.0 and lx <= 0.0:
            errors.append(
                "dynamic branch has no positive training loss weight: lambda_traj and lambda_tex were both non-positive. Use a checkpoint trained with a positive branch loss weight."
            )
    if train_data_k is not None or eval_data_k is not None:
        t_k, e_k = (int(train_data_k or 0), int(eval_data_k or 0))
        if t_k != e_k:
            errors.append(
                f"num_extra_ref_frames: training ran with K={t_k} but eval composed K={e_k}. Use the data configuration matching the checkpoint's training window."
            )
    if train_data_num_frames is not None or eval_data_num_frames is not None:
        t_nf, e_nf = (int(train_data_num_frames or 33), int(eval_data_num_frames or 33))
        if t_nf != e_nf:
            errors.append(
                f"num_frames: training ran with a {t_nf}-frame window but eval composed {e_nf}. Use the data configuration matching the checkpoint's training window."
            )
    return errors
