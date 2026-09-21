from mtwam.models.wan22._load_guards import check_branch_arch_consistency

_TRAIN = {
    "enable_dynamic_branch": True,
    "dynamic_branch_num_layers": 10,
    "dynamic_branch_read_text": False,
    "dynamic_branch_ffn_moe": True,
    "dynamic_branch_ffn_moe_routing": "role",
}


def test_match_passes():
    assert check_branch_arch_consistency(_TRAIN, dict(_TRAIN)) == []


def test_routing_mismatch_caught():
    errs = check_branch_arch_consistency(
        _TRAIN, {**_TRAIN, "dynamic_branch_ffn_moe_routing": "scramble"}
    )
    assert (
        len(errs) == 1
        and "routing" in errs[0]
        and ("configuration mismatch" in errs[0])
    )


def test_read_text_mismatch_caught():
    errs = check_branch_arch_consistency(
        _TRAIN, {**_TRAIN, "dynamic_branch_read_text": True}
    )
    assert len(errs) == 1 and "read_text" in errs[0]


def test_num_layers_mismatch_caught():
    errs = check_branch_arch_consistency(
        _TRAIN, {**_TRAIN, "dynamic_branch_num_layers": 5}
    )
    assert (
        len(errs) == 1
        and "dynamic_branch_num_layers" in errs[0]
        and ("configuration mismatch" in errs[0])
    )
    for m in (5, 15, 20):
        cfg = {**_TRAIN, "dynamic_branch_num_layers": m}
        assert check_branch_arch_consistency(cfg, dict(cfg)) == []


def test_missing_keys_take_defaults():
    assert check_branch_arch_consistency({}, {"enable_dynamic_branch": False}) == []
    errs = check_branch_arch_consistency({}, {"dynamic_branch_ffn_moe": True})
    assert len(errs) == 1 and "ffn_moe" in errs[0]


def test_data_k_mismatch_caught():
    errs = check_branch_arch_consistency(
        _TRAIN, dict(_TRAIN), train_data_k=16, eval_data_k=0
    )
    assert len(errs) == 1 and "num_extra_ref_frames" in errs[0]
    assert (
        check_branch_arch_consistency(
            _TRAIN, dict(_TRAIN), train_data_k=16, eval_data_k=16
        )
        == []
    )
    assert check_branch_arch_consistency(_TRAIN, dict(_TRAIN)) == []
    assert (
        check_branch_arch_consistency(
            _TRAIN, dict(_TRAIN), train_data_k=0, eval_data_k=None
        )
        == []
    )
    errs = check_branch_arch_consistency(
        _TRAIN, dict(_TRAIN), train_data_k=16, eval_data_k=None
    )
    assert len(errs) == 1 and "num_extra_ref_frames" in errs[0]
    errs = check_branch_arch_consistency(
        _TRAIN, dict(_TRAIN), train_data_k=None, eval_data_k=16
    )
    assert len(errs) == 1 and "num_extra_ref_frames" in errs[0]


def test_data_num_frames_mismatch_caught():
    errs = check_branch_arch_consistency(
        _TRAIN, dict(_TRAIN), train_data_num_frames=17, eval_data_num_frames=33
    )
    assert len(errs) == 1 and "num_frames" in errs[0]
    assert (
        check_branch_arch_consistency(
            _TRAIN, dict(_TRAIN), train_data_num_frames=17, eval_data_num_frames=17
        )
        == []
    )
    assert check_branch_arch_consistency(_TRAIN, dict(_TRAIN)) == []
    assert (
        check_branch_arch_consistency(
            _TRAIN, dict(_TRAIN), train_data_num_frames=33, eval_data_num_frames=None
        )
        == []
    )
    errs = check_branch_arch_consistency(
        _TRAIN, dict(_TRAIN), train_data_num_frames=17, eval_data_num_frames=None
    )
    assert len(errs) == 1 and "num_frames" in errs[0]
    errs = check_branch_arch_consistency(
        _TRAIN, dict(_TRAIN), train_data_num_frames=None, eval_data_num_frames=17
    )
    assert len(errs) == 1 and "num_frames" in errs[0]


import pytest
from mtwam.models.wan22._load_guards import parse_bool, strict_bool


@pytest.mark.parametrize(
    "value, expected",
    [
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("  true  ", True),
        ("false", False),
        ("False", False),
        ("FALSE", False),
        ("1", True),
        ("0", False),
        ("yes", True),
        ("no", False),
        ("on", True),
        ("off", False),
    ],
)
def test_builder_and_reconcile_read_every_boolean_the_same_way(value, expected):
    assert parse_bool(value) is expected
    assert strict_bool(value, field="x") is expected


def test_the_exact_trap_value_quoted_False():
    assert bool("False") is True, (
        "Boolean conversion alone does not validate configuration types"
    )
    assert strict_bool("False", field="dynamic_branch_triple_f0") is False


@pytest.mark.parametrize("garbage", ["maybe", "", "2", "None", None, 7, 1.5, []])
def test_uninterpretable_values_fail_loud_instead_of_going_truthy(garbage):
    assert parse_bool(garbage) is None
    with pytest.raises(ValueError, match="dynamic_branch_triple_f0"):
        strict_bool(garbage, field="dynamic_branch_triple_f0")


def test_string_boolean_that_agrees_with_training_now_passes_consistently():
    train = {"enable_dynamic_branch": True, "dynamic_branch_triple_f0": False}
    ev = {"enable_dynamic_branch": True, "dynamic_branch_triple_f0": "False"}
    assert check_branch_arch_consistency(train, ev) == []
    assert strict_bool(ev["dynamic_branch_triple_f0"], field="k") is False


def test_string_boolean_that_disagrees_with_training_is_caught():
    train = {"enable_dynamic_branch": True, "dynamic_branch_triple_f0": True}
    ev = {"enable_dynamic_branch": True, "dynamic_branch_triple_f0": "false"}
    errs = check_branch_arch_consistency(train, ev)
    assert len(errs) == 1 and "dynamic_branch_triple_f0" in errs[0]


def test_uninterpretable_bool_is_reported_by_the_reconcile():
    train = {"enable_dynamic_branch": True}
    ev = {"enable_dynamic_branch": "maybe"}
    errs = check_branch_arch_consistency(train, ev)
    assert len(errs) == 1 and "cannot interpret" in errs[0]


def _with_mode(base: dict, mode) -> dict:
    out = dict(base)
    out["video_dit_config"] = {"video_attention_mask_mode": mode}
    return out


def test_mask_mode_mismatch_caught():
    errs = check_branch_arch_consistency(
        _with_mode(_TRAIN, "first_frame_causal"), _with_mode(_TRAIN, "per_frame_causal")
    )
    assert len(errs) == 1 and "video_attention_mask_mode" in errs[0]


def test_mask_mode_absent_on_either_side_is_skipped():
    assert (
        check_branch_arch_consistency(_TRAIN, _with_mode(_TRAIN, "first_frame_causal"))
        == []
    )
    assert (
        check_branch_arch_consistency(_with_mode(_TRAIN, "first_frame_causal"), _TRAIN)
        == []
    )


def test_never_trained_branch_is_caught():
    train = dict(_TRAIN, loss={"lambda_traj": 0.0, "lambda_tex": 0.0})
    errs = check_branch_arch_consistency(train, dict(_TRAIN))
    assert len(errs) == 1 and "no positive training loss weight" in errs[0]


@pytest.mark.parametrize("lt, lx", [(1.0, 0.0), (0.0, 0.25), (5.0, 0.4)])
def test_trained_branch_passes(lt, lx):
    train = dict(_TRAIN, loss={"lambda_traj": lt, "lambda_tex": lx})
    assert check_branch_arch_consistency(train, dict(_TRAIN)) == []


def test_never_trained_check_skipped_when_branch_is_off():
    train = dict(
        _TRAIN,
        enable_dynamic_branch=False,
        loss={"lambda_traj": 0.0, "lambda_tex": 0.0},
    )
    ev = dict(_TRAIN, enable_dynamic_branch=False)
    assert check_branch_arch_consistency(train, ev) == []


def test_never_trained_check_skipped_when_train_cfg_has_no_loss_node():
    assert check_branch_arch_consistency(dict(_TRAIN), dict(_TRAIN)) == []
