import json
import math
import re
from pathlib import Path


def read_tasks(path):
    tasks = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if (
            len(parts) != 2
            or re.fullmatch("[A-Za-z0-9_]+", parts[0]) is None
            or (not parts[1].isdigit())
        ):
            raise ValueError(f"Invalid task on line {number}: {line!r}")
        tasks.append((parts[0], int(parts[1])))
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError(
            "Task manifest must be nonempty and contain no duplicate tasks."
        )
    return tasks


def result_complete(result, suite, task_id, num_trials, seed, digest):
    if not isinstance(result, dict):
        return False
    if any(
        (
            result.get(key) != value
            for key, value in dict(
                task_suite=suite,
                task_id=task_id,
                total_episodes=num_trials,
                seed=seed,
                configuration_sha256=digest,
            ).items()
        )
    ):
        return False
    success = result.get("success_episodes")
    failure = result.get("failure_episodes")
    if not isinstance(success, list) or not isinstance(failure, list):
        return False
    episodes = success + failure
    if any((type(value) is not int for value in episodes)) or sorted(episodes) != list(
        range(num_trials)
    ):
        return False
    duration = result.get("duration")
    return (
        type(result.get("successes")) is int
        and result["successes"] == len(success)
        and isinstance(duration, (int, float))
        and math.isfinite(duration)
        and (duration > 0)
    )


def variant_result_done(output_dir, suite, task_id, num_trials, seed, digest):
    for path in sorted(
        (Path(output_dir) / suite).glob(f"gpu*_task{task_id}_results.json")
    ):
        try:
            result = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if result_complete(result, suite, task_id, num_trials, seed, digest):
            return True
    return False
