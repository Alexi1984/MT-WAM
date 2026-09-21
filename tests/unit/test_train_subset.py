import pytest
from mtwam.runtime import apply_train_subset, validate_branch_subset


def test_apply_train_subset_limits_to_leading_range():
    ds = list(range(10))
    sub = apply_train_subset(ds, 3)
    assert len(sub) == 3
    assert [sub[i] for i in range(3)] == [0, 1, 2]


def test_apply_train_subset_none_is_passthrough():
    ds = list(range(10))
    assert apply_train_subset(ds, None) is ds


def test_apply_train_subset_rejects_n_larger_than_dataset():
    ds = list(range(10))
    with pytest.raises(ValueError):
        apply_train_subset(ds, 11)


def test_validate_branch_subset_requires_subset_when_branch_on():
    with pytest.raises(ValueError):
        validate_branch_subset(True, None)


def test_validate_branch_subset_ok_when_branch_on_with_subset():
    validate_branch_subset(True, 2837)


def test_validate_branch_subset_ok_when_branch_off():
    validate_branch_subset(False, None)
