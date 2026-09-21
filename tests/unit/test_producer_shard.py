import os
import sys
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
)
from precompute_teacher_feats import shard_indices, _load_exclude_episodes_file


def test_shards_partition_range_exactly():
    n, N = (66984, 8)
    seen = []
    for k in range(N):
        seen.extend(list(shard_indices(n, k, N)))
    assert len(seen) == n
    assert len(set(seen)) == n
    assert sorted(seen) == list(range(n))


def test_shard_indices_strided():
    assert list(shard_indices(10, 0, 3)) == [0, 3, 6, 9]
    assert list(shard_indices(10, 1, 3)) == [1, 4, 7]
    assert list(shard_indices(10, 2, 3)) == [2, 5, 8]


def test_single_shard_is_full_range():
    assert list(shard_indices(5)) == [0, 1, 2, 3, 4]


def test_rejects_bad_shard_id():
    with pytest.raises(ValueError):
        shard_indices(10, 3, 3)
    with pytest.raises(ValueError):
        shard_indices(10, -1, 3)
    with pytest.raises(ValueError):
        shard_indices(10, 0, 0)


def test_load_exclude_episodes_file_parses_ints_skips_comments(tmp_path):
    p = tmp_path / "exclude.txt"
    p.write_text("# header\n1\n2\n\n  10  \n# mid comment\n100\n")
    assert _load_exclude_episodes_file(str(p)) == [1, 2, 10, 100]


def test_exclude_complement_yields_exact_subset(tmp_path):
    from mtwam.datasets.lerobot.base_lerobot_dataset import filter_excluded_episodes

    total = 20
    subset = {3, 7, 11, 15}
    complement = sorted(set(range(total)) - subset)
    p = tmp_path / "exclude.txt"
    p.write_text("# complement of subset\n" + "\n".join(map(str, complement)) + "\n")
    excl = _load_exclude_episodes_file(str(p))
    out = filter_excluded_episodes({"robotwin": list(range(total))}, excl)
    assert out["robotwin"] == sorted(subset)
