import pytest
from mtwam.utils.eval_windows import build_val_window_set, window_seed

_PLAN = [
    {
        "plan_idx": 0,
        "dataset_index": 0,
        "repo_key": "suiteA_lerobot",
        "episode_index": 0,
        "global_start": 0,
        "n_windows": 100,
    },
    {
        "plan_idx": 1,
        "dataset_index": 0,
        "repo_key": "suiteA_lerobot",
        "episode_index": 1,
        "global_start": 100,
        "n_windows": 50,
    },
    {
        "plan_idx": 2,
        "dataset_index": 1,
        "repo_key": "suiteB_lerobot",
        "episode_index": 0,
        "global_start": 150,
        "n_windows": 200,
    },
]


def test_window_seed_pinned_crc32():
    assert window_seed("libero_object_lerobot", 3, 17) == 3717028715
    assert window_seed("libero_goal_lerobot", 0, 0) == 2035839264
    assert window_seed("libero_goal_lerobot", 0, 1) != window_seed(
        "libero_goal_lerobot", 0, 0
    )
    assert window_seed("libero_goal_lerobot", 1, 0) != window_seed(
        "libero_goal_lerobot", 0, 0
    )


def test_val_window_set_deterministic():
    a = build_val_window_set(_PLAN, n_per_repo=16, seed=20260705)
    b = build_val_window_set(_PLAN, n_per_repo=16, seed=20260705)
    assert a == b
    assert a != build_val_window_set(_PLAN, n_per_repo=16, seed=1)


def test_val_window_set_stratified_and_valid():
    ws = build_val_window_set(_PLAN, n_per_repo=16, seed=20260705)
    per_repo = {}
    for w in ws:
        per_repo.setdefault(w["repo_key"], []).append(w)
    assert set(per_repo) == {"suiteA_lerobot", "suiteB_lerobot"}
    assert all((len(v) == 16 for v in per_repo.values()))
    assert len({w["global_idx"] for w in ws}) == len(ws)
    for w in ws:
        ep = next(
            (
                p
                for p in _PLAN
                if p["repo_key"] == w["repo_key"]
                and p["episode_index"] == w["episode_index"]
            )
        )
        assert w["global_idx"] == ep["global_start"] + w["frame_index"]
        assert 0 <= w["frame_index"] < ep["n_windows"]
    keys = [(w["repo_key"], w["global_idx"]) for w in ws]
    assert keys == sorted(keys)


def test_val_window_set_repo_isolation():
    both = [
        w
        for w in build_val_window_set(_PLAN, 16, 20260705)
        if w["repo_key"] == "suiteA_lerobot"
    ]
    only_a = build_val_window_set(
        [p for p in _PLAN if p["repo_key"] == "suiteA_lerobot"], 16, 20260705
    )
    assert both == only_a


def test_val_window_set_insufficient_raises():
    with pytest.raises(ValueError, match="n_per_repo"):
        build_val_window_set(_PLAN, n_per_repo=151, seed=20260705)
    with pytest.raises(ValueError, match="empty"):
        build_val_window_set([], n_per_repo=1, seed=0)
