from mtwam.models.wan22._load_guards import check_mot_branch_load, BRANCH_KEY_PREFIX


def _mot_base_keys():
    return ["blocks.0.attn.q.weight", "blocks.0.ffn.0.weight", "head.weight"]


def _mot_branch_keys():
    return [
        f"{BRANCH_KEY_PREFIX}role_t",
        f"{BRANCH_KEY_PREFIX}role_s",
        f"{BRANCH_KEY_PREFIX}blocks.0.attn.q.weight",
        f"{BRANCH_KEY_PREFIX}traj_head.0.weight",
        f"{BRANCH_KEY_PREFIX}tex_head.0.weight",
    ]


def test_baseline_into_baseline_no_error():
    keys = _mot_base_keys()
    errors = check_mot_branch_load(
        payload_mot_keys=keys, model_mot_keys=keys, missing_keys=[], unexpected_keys=[]
    )
    assert errors == []


def test_full_into_full_no_error():
    keys = _mot_base_keys() + _mot_branch_keys()
    errors = check_mot_branch_load(
        payload_mot_keys=keys, model_mot_keys=keys, missing_keys=[], unexpected_keys=[]
    )
    assert errors == []


def test_full_ckpt_into_branchless_model_raises():
    model_keys = _mot_base_keys()
    payload_keys = _mot_base_keys() + _mot_branch_keys()
    unexpected = _mot_branch_keys()
    errors = check_mot_branch_load(
        payload_mot_keys=payload_keys,
        model_mot_keys=model_keys,
        missing_keys=[],
        unexpected_keys=unexpected,
    )
    assert len(errors) >= 1
    assert any(("branch-less" in e or "no dynamic branch" in e for e in errors))


def test_branch_model_but_ckpt_has_no_branch_raises():
    model_keys = _mot_base_keys() + _mot_branch_keys()
    payload_keys = _mot_base_keys()
    missing = _mot_branch_keys()
    errors = check_mot_branch_load(
        payload_mot_keys=payload_keys,
        model_mot_keys=model_keys,
        missing_keys=missing,
        unexpected_keys=[],
    )
    assert len(errors) >= 1
    assert any(("checkpoint provided none" in e.lower() for e in errors))


def test_branch_model_ckpt_no_branch_allow_override():
    model_keys = _mot_base_keys() + _mot_branch_keys()
    payload_keys = _mot_base_keys()
    missing = _mot_branch_keys()
    errors = check_mot_branch_load(
        payload_mot_keys=payload_keys,
        model_mot_keys=model_keys,
        missing_keys=missing,
        unexpected_keys=[],
        allow_uninitialized_branch=True,
    )
    assert errors == []


def test_partial_branch_consumed_no_error():
    model_keys = _mot_base_keys() + _mot_branch_keys()
    payload_keys = _mot_base_keys() + _mot_branch_keys()
    missing = [f"{BRANCH_KEY_PREFIX}tex_head.0.weight"]
    errors = check_mot_branch_load(
        payload_mot_keys=payload_keys,
        model_mot_keys=model_keys,
        missing_keys=missing,
        unexpected_keys=[],
    )
    assert errors == []


def _moe_branch_keys():
    return [
        f"{BRANCH_KEY_PREFIX}role_t",
        f"{BRANCH_KEY_PREFIX}role_s",
        f"{BRANCH_KEY_PREFIX}blocks.0.attn.q.weight",
        f"{BRANCH_KEY_PREFIX}blocks.0.ffn.ffn_t.0.weight",
        f"{BRANCH_KEY_PREFIX}blocks.0.ffn.ffn_s.0.weight",
    ]


def _shared_ffn_branch_keys():
    return [
        f"{BRANCH_KEY_PREFIX}role_t",
        f"{BRANCH_KEY_PREFIX}role_s",
        f"{BRANCH_KEY_PREFIX}blocks.0.attn.q.weight",
        f"{BRANCH_KEY_PREFIX}blocks.0.ffn.0.weight",
    ]


def test_moe_ckpt_into_shared_ffn_model_raises():
    model_keys = _mot_base_keys() + _shared_ffn_branch_keys()
    payload_keys = _mot_base_keys() + _moe_branch_keys()
    errors = check_mot_branch_load(
        payload_mot_keys=payload_keys,
        model_mot_keys=model_keys,
        missing_keys=[f"{BRANCH_KEY_PREFIX}blocks.0.ffn.0.weight"],
        unexpected_keys=[
            f"{BRANCH_KEY_PREFIX}blocks.0.ffn.ffn_t.0.weight",
            f"{BRANCH_KEY_PREFIX}blocks.0.ffn.ffn_s.0.weight",
        ],
    )
    assert any(("FFN structure mismatch" in e for e in errors))


def test_shared_ffn_ckpt_into_moe_model_raises():
    model_keys = _mot_base_keys() + _moe_branch_keys()
    payload_keys = _mot_base_keys() + _shared_ffn_branch_keys()
    errors = check_mot_branch_load(
        payload_mot_keys=payload_keys,
        model_mot_keys=model_keys,
        missing_keys=[
            f"{BRANCH_KEY_PREFIX}blocks.0.ffn.ffn_t.0.weight",
            f"{BRANCH_KEY_PREFIX}blocks.0.ffn.ffn_s.0.weight",
        ],
        unexpected_keys=[f"{BRANCH_KEY_PREFIX}blocks.0.ffn.0.weight"],
    )
    assert any(("FFN structure mismatch" in e for e in errors))


def test_moe_ckpt_into_moe_model_no_structure_error():
    keys = _mot_base_keys() + _moe_branch_keys()
    errors = check_mot_branch_load(
        payload_mot_keys=keys, model_mot_keys=keys, missing_keys=[], unexpected_keys=[]
    )
    assert errors == []


def _gated_branch_keys():
    return [
        f"{BRANCH_KEY_PREFIX}role_t",
        f"{BRANCH_KEY_PREFIX}role_s",
        f"{BRANCH_KEY_PREFIX}blocks.0.attn.q.weight",
        f"{BRANCH_KEY_PREFIX}blocks.0.ffn.gate.weight",
        f"{BRANCH_KEY_PREFIX}blocks.0.ffn.experts.0.0.weight",
        f"{BRANCH_KEY_PREFIX}blocks.0.ffn.experts.1.0.weight",
    ]


def test_gated_ckpt_into_shared_ffn_model_raises():
    errors = check_mot_branch_load(
        payload_mot_keys=_mot_base_keys() + _gated_branch_keys(),
        model_mot_keys=_mot_base_keys() + _shared_ffn_branch_keys(),
        missing_keys=[f"{BRANCH_KEY_PREFIX}blocks.0.ffn.0.weight"],
        unexpected_keys=[
            f"{BRANCH_KEY_PREFIX}blocks.0.ffn.gate.weight",
            f"{BRANCH_KEY_PREFIX}blocks.0.ffn.experts.0.0.weight",
        ],
    )
    assert any(("FFN structure mismatch" in e for e in errors))


def test_gated_ckpt_into_role_moe_model_raises():
    errors = check_mot_branch_load(
        payload_mot_keys=_mot_base_keys() + _gated_branch_keys(),
        model_mot_keys=_mot_base_keys() + _moe_branch_keys(),
        missing_keys=[
            f"{BRANCH_KEY_PREFIX}blocks.0.ffn.ffn_t.0.weight",
            f"{BRANCH_KEY_PREFIX}blocks.0.ffn.ffn_s.0.weight",
        ],
        unexpected_keys=[
            f"{BRANCH_KEY_PREFIX}blocks.0.ffn.gate.weight",
            f"{BRANCH_KEY_PREFIX}blocks.0.ffn.experts.0.0.weight",
        ],
    )
    assert any(("FFN structure mismatch" in e for e in errors))


def test_gated_ckpt_into_gated_model_no_structure_error():
    keys = _mot_base_keys() + _gated_branch_keys()
    errors = check_mot_branch_load(
        payload_mot_keys=keys, model_mot_keys=keys, missing_keys=[], unexpected_keys=[]
    )
    assert errors == []
