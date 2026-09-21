import json
import pytest
from mtwam.datasets.lerobot.robot_video_dataset import (
    DEFAULT_PROMPT,
    collect_prompts_for_episodes,
)


def _make_ds(tmp_path, episodes, name="ds"):
    root = tmp_path / name
    (root / "meta").mkdir(parents=True)
    with (root / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for idx, tasks in episodes.items():
            f.write(
                json.dumps({"episode_index": idx, "tasks": tasks, "length": 10}) + "\n"
            )
    seen, rows = (set(), [])
    for tasks in episodes.values():
        for t in tasks:
            if t not in seen:
                seen.add(t)
                rows.append(t)
    with (root / "meta" / "tasks.jsonl").open("w", encoding="utf-8") as f:
        for i, t in enumerate(rows):
            f.write(json.dumps({"task_index": i, "task": t}) + "\n")
    return str(root)


EPISODES = {
    0: ["lift the bottle", "grab the green bottle"],
    1: ["lift the bottle", "pick it up"],
    2: ["open the microwave"],
    3: ["stack the bowls", "stack two bowls"],
}


def test_whitelist_covers_exactly_those_episodes(tmp_path):
    ds = _make_ds(tmp_path, EPISODES)
    got = collect_prompts_for_episodes((ds_dirs := [ds]), include_episodes=[0, 2])
    assert set(got) == {
        DEFAULT_PROMPT.format(task=t)
        for t in ("lift the bottle", "grab the green bottle", "open the microwave")
    }
    assert len(got) == 3
    assert len(set(got)) == len(got)
    assert ds_dirs == [ds]


def test_none_whitelist_equals_full_task_scan(tmp_path):
    ds = _make_ds(tmp_path, EPISODES)
    from_episodes = collect_prompts_for_episodes([ds], include_episodes=None)
    with open(f"{ds}/meta/tasks.jsonl", encoding="utf-8") as f:
        from_tasks = [
            DEFAULT_PROMPT.format(task=json.loads(l)["task"]) for l in f if l.strip()
        ]
    assert set(from_episodes) == set(from_tasks)
    assert len(from_episodes) == len(from_tasks) == 6


def test_prompts_use_the_consumer_template(tmp_path):
    ds = _make_ds(tmp_path, {7: ["open the microwave"]})
    (got,) = collect_prompts_for_episodes([ds], include_episodes=[7])
    assert got == DEFAULT_PROMPT.format(task="open the microwave")
    assert "open the microwave" in got and got != "open the microwave"


def test_multiple_dataset_dirs_are_unioned(tmp_path):
    a = _make_ds(tmp_path, {0: ["task a"]}, name="a")
    b = _make_ds(tmp_path, {0: ["task b"]}, name="b")
    got = collect_prompts_for_episodes([a, b], include_episodes=[0])
    assert set(got) == {DEFAULT_PROMPT.format(task=t) for t in ("task a", "task b")}


def test_empty_whitelist_yields_nothing(tmp_path):
    ds = _make_ds(tmp_path, EPISODES)
    assert collect_prompts_for_episodes([ds], include_episodes=[]) == []


def test_whitelist_ids_absent_from_dataset_are_ignored(tmp_path):
    ds = _make_ds(tmp_path, EPISODES)
    got = collect_prompts_for_episodes([ds], include_episodes=[2, 99999])
    assert got == [DEFAULT_PROMPT.format(task="open the microwave")]


def test_missing_episodes_file_raises(tmp_path):
    (tmp_path / "empty" / "meta").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="episodes.jsonl"):
        collect_prompts_for_episodes([str(tmp_path / "empty")], include_episodes=[0])


def test_string_ids_are_accepted(tmp_path):
    ds = _make_ds(tmp_path, EPISODES)
    assert collect_prompts_for_episodes([ds], include_episodes=["2"]) == [
        DEFAULT_PROMPT.format(task="open the microwave")
    ]
