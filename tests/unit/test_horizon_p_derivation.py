import pytest
import torch
from mtwam.datasets.lerobot.teacher_cache import (
    TEACHER_HORIZON_P,
    prepare_teacher_frames_for_camera,
    teacher_p_max,
)


@pytest.mark.parametrize(
    "n_sampled,anchor,expected",
    [(9, 0, 2), (9, 0, 2), (13, 4, 2), (17, 8, 2), (5, 0, 1), (9, 4, 1)],
)
def test_teacher_p_max_truth_table(n_sampled, anchor, expected):
    assert teacher_p_max(n_sampled, anchor) == expected


def test_teacher_p_max_existing_configs_all_derive_2():
    for n_sampled, anchor in [(9, 0), (13, 4), (17, 8)]:
        assert teacher_p_max(n_sampled, anchor) == TEACHER_HORIZON_P


def test_teacher_p_max_too_short_raises():
    with pytest.raises(ValueError, match="teacher window too short"):
        teacher_p_max(5, 4)
    with pytest.raises(ValueError, match="teacher window too short"):
        teacher_p_max(1, 0)


VSI_Q = list(range(0, 17, 4))


def _tagged_window(t_obs):
    w = torch.zeros(t_obs, 3, 32, 32)
    for i in range(t_obs):
        w[i] = i / 100.0
    return w


def test_q_window_default_none_derives_pmax_1():
    reps = prepare_teacher_frames_for_camera(_tagged_window(17), VSI_Q, (32, 32))
    assert reps.shape[0] == 2
    assert torch.allclose(reps[0], torch.full_like(reps[0], 0 / 100.0), atol=1e-06)
    assert torch.allclose(reps[1], torch.full_like(reps[1], 16 / 100.0), atol=1e-06)
    assert not torch.allclose(reps[1], torch.full_like(reps[1], 32 / 100.0), atol=1e-06)


def test_q_window_explicit_horizon_1_equals_default():
    a = prepare_teacher_frames_for_camera(_tagged_window(17), VSI_Q, (32, 32))
    b = prepare_teacher_frames_for_camera(
        _tagged_window(17), VSI_Q, (32, 32), horizon_p=1
    )
    assert torch.allclose(a, b, atol=1e-06)


def test_q_window_horizon_2_exceeds_pmax_raises():
    with pytest.raises(ValueError, match="horizon_p must be in \\[1, 1\\]"):
        prepare_teacher_frames_for_camera(
            _tagged_window(17), VSI_Q, (32, 32), horizon_p=2
        )


def test_33_window_default_none_still_p2():
    VSI = list(range(0, 33, 4))
    reps_none = prepare_teacher_frames_for_camera(_tagged_window(33), VSI, (32, 32))
    reps_2 = prepare_teacher_frames_for_camera(
        _tagged_window(33), VSI, (32, 32), horizon_p=2
    )
    assert reps_none.shape[0] == 3
    assert torch.allclose(reps_none, reps_2, atol=1e-06)
