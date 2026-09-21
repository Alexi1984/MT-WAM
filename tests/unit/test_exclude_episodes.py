from mtwam.datasets.lerobot.base_lerobot_dataset import filter_excluded_episodes


def test_filter_excluded_drops_listed_episode():
    eps = {"goal": list(range(5))}
    out = filter_excluded_episodes(eps, [2])
    assert out["goal"] == [0, 1, 3, 4]


def test_filter_excluded_none_is_passthrough():
    eps = {"goal": [0, 1, 2]}
    assert filter_excluded_episodes(eps, None) == {"goal": [0, 1, 2]}


def test_filter_excluded_empty_is_passthrough():
    eps = {"goal": [0, 1, 2]}
    assert filter_excluded_episodes(eps, []) == {"goal": [0, 1, 2]}


def test_filter_excluded_list_applies_to_all_repos():
    eps = {"A": [0, 1, 2], "B": [0, 1, 2]}
    out = filter_excluded_episodes(eps, [1])
    assert out["A"] == [0, 2] and out["B"] == [0, 2]


def test_filter_excluded_dict_scopes_to_named_repo():
    eps = {
        ".../libero_goal_no_noops_lerobot": list(range(100)),
        ".../libero_object_no_noops_lerobot": list(range(100)),
        ".../libero_10_no_noops_lerobot": list(range(100)),
    }
    out = filter_excluded_episodes(eps, {".../libero_goal_no_noops_lerobot": [82]})
    assert 82 not in out[".../libero_goal_no_noops_lerobot"]
    assert 82 in out[".../libero_object_no_noops_lerobot"]
    assert 82 in out[".../libero_10_no_noops_lerobot"]
    assert len(out[".../libero_goal_no_noops_lerobot"]) == 99
    assert len(out[".../libero_object_no_noops_lerobot"]) == 100


def test_filter_excluded_dict_unlisted_repo_untouched():
    eps = {"goal": list(range(5)), "object": list(range(5))}
    out = filter_excluded_episodes(eps, {"goal": [2]})
    assert out["goal"] == [0, 1, 3, 4]
    assert out["object"] == [0, 1, 2, 3, 4]


def test_filter_excluded_dict_unknown_key_raises():
    eps = {"goal": [0, 1, 2], "object": [0, 1, 2]}
    try:
        filter_excluded_episodes(eps, {"libero_goal_typo": [82]})
    except ValueError as e:
        assert "libero_goal_typo" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown repo_id key")


def test_filter_excluded_dict_str_ints():
    eps = {"goal": [0, 1, 2], "object": [0, 1, 2]}
    out = filter_excluded_episodes(eps, {"goal": ["1"]})
    assert out["goal"] == [0, 2] and out["object"] == [0, 1, 2]


def test_filter_excluded_missing_index_is_noop():
    eps = {"goal": [0, 1, 2]}
    assert filter_excluded_episodes(eps, [82]) == {"goal": [0, 1, 2]}


def test_filter_excluded_multiple_indices():
    eps = {"goal": list(range(6))}
    out = filter_excluded_episodes(eps, [1, 4])
    assert out["goal"] == [0, 2, 3, 5]


def test_filter_excluded_accepts_str_ints():
    eps = {"goal": [0, 1, 2]}
    out = filter_excluded_episodes(eps, ["1"])
    assert out["goal"] == [0, 2]
