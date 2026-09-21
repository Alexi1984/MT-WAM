import zlib
import numpy as np


def window_seed(repo_key: str, episode_index: int, frame_index: int) -> int:
    return zlib.crc32(f"{repo_key}:{int(episode_index)}:{int(frame_index)}".encode())


def shard_round_robin(items, rank: int, world_size: int):
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError(f"bad shard spec: rank={rank}, world_size={world_size}")
    return items[rank::world_size]


def build_val_window_set(plan, n_per_repo: int, seed: int):
    by_repo = {}
    for ep in plan:
        by_repo.setdefault(ep["repo_key"], []).append(ep)
    if not by_repo:
        raise ValueError("empty episode plan — no repos to draw val windows from")
    out = []
    for repo_key in sorted(by_repo):
        windows = []
        for ep in by_repo[repo_key]:
            for off in range(int(ep["n_windows"])):
                windows.append(
                    (int(ep["global_start"]) + off, int(ep["episode_index"]), off)
                )
        if len(windows) < n_per_repo:
            raise ValueError(
                f"repo {repo_key} has only {len(windows)} windows < n_per_repo={n_per_repo}. Lower n_per_repo."
            )
        rng = np.random.default_rng([int(seed), zlib.crc32(repo_key.encode())])
        pick = rng.choice(len(windows), size=int(n_per_repo), replace=False)
        for j in sorted((int(x) for x in pick)):
            global_idx, episode_index, off = windows[j]
            out.append(
                {
                    "repo_key": repo_key,
                    "episode_index": episode_index,
                    "frame_index": off,
                    "global_idx": global_idx,
                }
            )
    return out
