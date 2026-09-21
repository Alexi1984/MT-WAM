import torch
from torch.utils.data import default_collate
from mtwam.datasets.lerobot.teacher_cache import (
    TEACHER_GRID_TRAJ,
    TEACHER_GRID_DINO,
    TEACHER_HORIZON_P,
)


def _fake_sample(num_cameras=1):
    P = TEACHER_HORIZON_P
    return {
        "teacher_tracks": torch.zeros(num_cameras, P, TEACHER_GRID_TRAJ, 2),
        "teacher_track_vis": torch.ones(
            num_cameras, P, TEACHER_GRID_TRAJ, dtype=torch.bool
        ),
        "teacher_dino": torch.zeros(num_cameras, P, TEACHER_GRID_DINO, 768),
        "action": torch.zeros(32, 7),
    }


def test_default_collate_stacks_fixed_length_teacher():
    batch = default_collate([_fake_sample(), _fake_sample()])
    assert batch["teacher_tracks"].shape == (
        2,
        1,
        TEACHER_HORIZON_P,
        TEACHER_GRID_TRAJ,
        2,
    )
    assert batch["teacher_track_vis"].shape == (
        2,
        1,
        TEACHER_HORIZON_P,
        TEACHER_GRID_TRAJ,
    )
    assert batch["teacher_track_vis"].dtype == torch.bool
    assert batch["teacher_dino"].shape == (
        2,
        1,
        TEACHER_HORIZON_P,
        TEACHER_GRID_DINO,
        768,
    )
    assert batch["action"].shape == (2, 32, 7)
