from mtwam.datasets.lerobot.base_lerobot_dataset import (
    filter_included_episodes,
    load_episode_ids_file,
)


def test_filter_included_keeps_only_listed():
    eps = {"robotwin": list(range(5))}
    out = filter_included_episodes(eps, [1, 3])
    assert out["robotwin"] == [1, 3]


def test_filter_included_none_is_passthrough():
    eps = {"robotwin": [0, 1, 2]}
    assert filter_included_episodes(eps, None) == {"robotwin": [0, 1, 2]}


def test_filter_included_empty_is_passthrough():
    eps = {"robotwin": [0, 1, 2]}
    assert filter_included_episodes(eps, []) == {"robotwin": [0, 1, 2]}


def test_filter_included_list_applies_to_all_repos():
    eps = {"A": [0, 1, 2], "B": [0, 1, 2]}
    out = filter_included_episodes(eps, [0, 2])
    assert out["A"] == [0, 2] and out["B"] == [0, 2]


def test_filter_included_order_follows_source_not_whitelist():
    eps = {"robotwin": [0, 1, 2, 3, 4]}
    out = filter_included_episodes(eps, [4, 1, 0])
    assert out["robotwin"] == [0, 1, 4]


def test_filter_included_dict_scopes_to_named_repo():
    eps = {"goal": list(range(5)), "object": list(range(5))}
    out = filter_included_episodes(eps, {"goal": [1, 2]})
    assert out["goal"] == [1, 2]
    assert out["object"] == [0, 1, 2, 3, 4]


def test_filter_included_dict_unknown_key_raises():
    eps = {"goal": [0, 1, 2], "object": [0, 1, 2]}
    try:
        filter_included_episodes(eps, {"robotwin_typo": [1]})
    except ValueError as e:
        assert "robotwin_typo" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown repo_id key")


def test_filter_included_accepts_str_ints():
    eps = {"robotwin": [0, 1, 2]}
    out = filter_included_episodes(eps, ["0", "2"])
    assert out["robotwin"] == [0, 2]


def test_filter_included_missing_index_is_skipped():
    eps = {"robotwin": [0, 1, 2]}
    out = filter_included_episodes(eps, [1, 99])
    assert out["robotwin"] == [1]


def test_filter_included_exact_subset_size():
    total, subset = (27500, sorted({i * 137 % 27500 for i in range(3350)}))
    eps = {"robotwin": list(range(total))}
    out = filter_included_episodes(eps, subset)
    assert out["robotwin"] == subset and len(out["robotwin"]) == len(subset)


def test_load_episode_ids_file(tmp_path):
    p = tmp_path / "subset.txt"
    p.write_text("# header comment\n2\n3\n\n4\n  12  \n# mid comment\n23\n")
    assert load_episode_ids_file(str(p)) == [2, 3, 4, 12, 23]


def test_load_episode_ids_file_roundtrips_into_whitelist(tmp_path):
    p = tmp_path / "s.txt"
    p.write_text("\n".join((str(e) for e in [0, 2, 4])) + "\n")
    ids = load_episode_ids_file(str(p))
    out = filter_included_episodes({"robotwin": list(range(5))}, ids)
    assert out["robotwin"] == [0, 2, 4]
