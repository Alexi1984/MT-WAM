import pytest
from mtwam.runtime import model_factory_extra_kwargs, validate_branch_eval_config

MTWAM = "mtwam.runtime.create_mtwam"
_SUPPLIED = {"enable_dynamic_branch": True, "online_teacher": True}


def test_injection_full_factory_gets_both():
    extra = model_factory_extra_kwargs(MTWAM, "robotwin", 16)
    assert extra == {
        "dynamic_branch_camera_layout": "robotwin",
        "num_extra_ref_frames": 16,
    }


def test_branch_eval_guard_unconstrained_quadrant():
    validate_branch_eval_config(True, 0.1, 0.01, 0)
    validate_branch_eval_config(False, 0.0, 0.0, 200)
    validate_branch_eval_config(True, 0.0, 0.0, 200)


def test_branch_eval_guard_complete_supply_passes():
    validate_branch_eval_config(
        True, 0.1, 0.01, 200, data_cfg={"train": dict(_SUPPLIED)}
    )
    validate_branch_eval_config(
        True, 0.1, 0.01, 200, data_cfg={"train": {}, "val": dict(_SUPPLIED)}
    )
    validate_branch_eval_config(
        True,
        0.1,
        0.01,
        200,
        data_cfg={
            "train": {
                "enable_dynamic_branch": True,
                "teacher_feature_cache_dir": "/x/teacher",
            }
        },
    )


def test_branch_eval_guard_incomplete_supply_refused():
    with pytest.raises(ValueError, match="online_teacher"):
        validate_branch_eval_config(
            True, 0.1, 0.01, 200, data_cfg={"train": {"enable_dynamic_branch": True}}
        )
    with pytest.raises(ValueError, match="enable_dynamic_branch"):
        validate_branch_eval_config(
            True, 0.1, 0.01, 200, data_cfg={"train": {"online_teacher": True}}
        )
    with pytest.raises(ValueError, match="enable_dynamic_branch"):
        validate_branch_eval_config(
            True, 0.1, 0.01, 200, data_cfg={"train": dict(_SUPPLIED), "val": {}}
        )
    with pytest.raises(ValueError, match="requires a validation data config"):
        validate_branch_eval_config(True, 0.1, 0.01, 200)
