import os
from pathlib import Path
import pytest
from omegaconf import OmegaConf
from mtwam.utils.branch_arch_reconcile import (
    ALLOW_MISSING_TRAIN_CONFIG_ENV,
    check_train_eval_branch_arch,
    resolve_eval_branch_horizon,
    train_cfg_path_for,
)


def _run_with_ckpt(
    tmp_path, name: str, train_model: dict, train_data: dict | None = None
):
    run = tmp_path / name
    ckpt = run / "checkpoints" / "weights" / "step_100.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_text("")
    _write_yaml(
        run / "config.yaml",
        {
            "model": train_model,
            "data": {
                "train": train_data
                if train_data is not None
                else {"num_frames": 33, "num_extra_ref_frames": 0}
            },
        },
    )
    return ckpt


def _write_yaml(path: Path, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(cfg), path)


def test_train_cfg_path_for_run_dir_layout(tmp_path):
    run = tmp_path / "run_x"
    ckpt = run / "checkpoints" / "weights" / "step_100.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_text("")
    cfg_path = run / "config.yaml"
    _write_yaml(cfg_path, {"model": {}})
    got, candidates = train_cfg_path_for(str(ckpt))
    assert got == str(cfg_path)
    assert len(candidates) == 2


def test_train_cfg_path_for_archive_layout(tmp_path):
    arm = tmp_path / "checkpoint_fixture"
    arm.mkdir()
    ckpt = arm / "step_021370.pt"
    ckpt.write_text("")
    cfg_path = arm / "config.yaml"
    _write_yaml(cfg_path, {"model": {}})
    got, _ = train_cfg_path_for(str(ckpt))
    assert got == str(cfg_path)


def test_train_cfg_path_for_missing(tmp_path):
    ckpt = tmp_path / "orphan.pt"
    ckpt.write_text("")
    got, candidates = train_cfg_path_for(str(ckpt))
    assert got is None
    assert len(candidates) == 2


def test_missing_config_with_baseline_eval_still_skips(tmp_path):
    ckpt = tmp_path / "external_baseline.pt"
    ckpt.write_text("")
    check_train_eval_branch_arch(OmegaConf.create({}), str(ckpt))
    check_train_eval_branch_arch(
        OmegaConf.create({"enable_dynamic_branch": False}), str(ckpt)
    )


def test_missing_config_with_nondefault_switches_now_refuses(tmp_path):
    ckpt = tmp_path / "external_branch.pt"
    ckpt.write_text("")
    eval_cfg = OmegaConf.create(
        {"enable_dynamic_branch": True, "dynamic_branch_dual_f0": True}
    )
    with pytest.raises(ValueError, match="no training config.yaml"):
        check_train_eval_branch_arch(eval_cfg, str(ckpt))


def test_missing_config_bypass_is_explicit_not_default(tmp_path):
    ckpt = tmp_path / "external_bypass.pt"
    ckpt.write_text("")
    eval_cfg = OmegaConf.create(
        {"enable_dynamic_branch": True, "dynamic_branch_triple_f0": True}
    )
    check_train_eval_branch_arch(eval_cfg, str(ckpt), allow_missing_train_config=True)


def test_missing_config_bypass_via_env(tmp_path, monkeypatch):
    ckpt = tmp_path / "external_bypass_env.pt"
    ckpt.write_text("")
    eval_cfg = OmegaConf.create(
        {"enable_dynamic_branch": True, "dynamic_branch_triple_f0": True}
    )
    monkeypatch.setenv(ALLOW_MISSING_TRAIN_CONFIG_ENV, "1")
    check_train_eval_branch_arch(eval_cfg, str(ckpt))
    monkeypatch.setenv(ALLOW_MISSING_TRAIN_CONFIG_ENV, "0")
    with pytest.raises(ValueError, match="no training config.yaml"):
        check_train_eval_branch_arch(eval_cfg, str(ckpt))


def test_check_train_eval_branch_arch_ok(tmp_path):
    run = tmp_path / "run_ok"
    ckpt = run / "checkpoints" / "weights" / "step_100.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_text("")
    _write_yaml(
        run / "config.yaml",
        {
            "model": {"enable_dynamic_branch": True, "dynamic_branch_dual_f0": True},
            "data": {"train": {"num_frames": 33, "num_extra_ref_frames": 0}},
        },
    )
    eval_cfg = OmegaConf.create(
        {"enable_dynamic_branch": True, "dynamic_branch_dual_f0": True}
    )
    check_train_eval_branch_arch(
        eval_cfg, str(ckpt), eval_data_num_frames=33, eval_data_num_extra_ref_frames=0
    )


def test_check_train_eval_branch_arch_dual_f0_mismatch_raises(tmp_path):
    run = tmp_path / "run_mismatch"
    ckpt = run / "checkpoints" / "weights" / "step_100.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_text("")
    _write_yaml(
        run / "config.yaml",
        {
            "model": {"enable_dynamic_branch": True, "dynamic_branch_dual_f0": True},
            "data": {"train": {"num_frames": 33, "num_extra_ref_frames": 0}},
        },
    )
    eval_cfg = OmegaConf.create({"enable_dynamic_branch": True})
    with pytest.raises(ValueError, match="branch-arch mismatch"):
        check_train_eval_branch_arch(
            eval_cfg,
            str(ckpt),
            eval_data_num_frames=33,
            eval_data_num_extra_ref_frames=0,
        )


def test_unresolved_null_horizon_hard_fails_even_for_a_baseline_arm(tmp_path):
    ckpt = _run_with_ckpt(tmp_path, "run_null_h", {"dynamic_branch_horizon": 2})
    eval_cfg = OmegaConf.create({"dynamic_branch_horizon": None})
    with pytest.raises(ValueError, match="dynamic_branch_horizon"):
        check_train_eval_branch_arch(
            eval_cfg,
            str(ckpt),
            eval_data_num_frames=33,
            eval_data_num_extra_ref_frames=0,
        )


def test_resolve_before_reconcile_unblocks_the_deadlock(tmp_path):
    ckpt = _run_with_ckpt(tmp_path, "run_resolved", {"dynamic_branch_horizon": 2})
    eval_cfg = OmegaConf.create({"dynamic_branch_horizon": None})
    got = resolve_eval_branch_horizon(
        eval_cfg,
        str(ckpt),
        {"num_frames": 33, "action_video_freq_ratio": 4, "num_extra_ref_frames": 0},
    )
    assert got == 2 and eval_cfg.dynamic_branch_horizon == 2
    check_train_eval_branch_arch(
        eval_cfg, str(ckpt), eval_data_num_frames=33, eval_data_num_extra_ref_frames=0
    )


def test_resolve_prefers_training_config_over_window_derivation(tmp_path):
    ckpt = _run_with_ckpt(tmp_path, "run_w_arm", {"dynamic_branch_horizon": 1})
    eval_cfg = OmegaConf.create({"dynamic_branch_horizon": None})
    got = resolve_eval_branch_horizon(
        eval_cfg,
        str(ckpt),
        {"num_frames": 33, "action_video_freq_ratio": 4, "num_extra_ref_frames": 0},
    )
    assert got == 1, "Use the horizon saved in the checkpoint training configuration"


def test_resolve_explicit_override_wins_and_does_not_touch_cfg(tmp_path):
    ckpt = _run_with_ckpt(tmp_path, "run_explicit", {"dynamic_branch_horizon": 2})
    eval_cfg = OmegaConf.create({"dynamic_branch_horizon": 1})
    got = resolve_eval_branch_horizon(
        eval_cfg,
        str(ckpt),
        {"num_frames": 33, "action_video_freq_ratio": 4, "num_extra_ref_frames": 0},
    )
    assert got == 1 and eval_cfg.dynamic_branch_horizon == 1


@pytest.mark.parametrize("num_frames, expected", [(33, 2), (17, 1)])
def test_resolve_falls_back_to_window_for_external_ckpt(tmp_path, num_frames, expected):
    ckpt = tmp_path / f"external_{num_frames}.pt"
    ckpt.write_text("")
    eval_cfg = OmegaConf.create({"dynamic_branch_horizon": None})
    got = resolve_eval_branch_horizon(
        eval_cfg,
        str(ckpt),
        {
            "num_frames": num_frames,
            "action_video_freq_ratio": 4,
            "num_extra_ref_frames": 0,
        },
    )
    assert got == expected and eval_cfg.dynamic_branch_horizon == expected


def test_resolve_accepts_a_plain_dict_model_cfg(tmp_path):
    ckpt = _run_with_ckpt(tmp_path, "run_plain", {"dynamic_branch_horizon": 2})
    eval_cfg: dict = {"dynamic_branch_horizon": None}
    got = resolve_eval_branch_horizon(
        eval_cfg,
        str(ckpt),
        {"num_frames": 33, "action_video_freq_ratio": 4, "num_extra_ref_frames": 0},
    )
    assert got == 2 and eval_cfg["dynamic_branch_horizon"] == 2
