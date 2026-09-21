import json
import os
import warnings
import torch

META_FILENAME = "meta.json"


def repo_key_for(ds_name: str) -> str:
    return os.path.basename(os.path.realpath(str(ds_name)).rstrip("/"))


def build_episode_plan(multi_ds):
    plan, offset = ([], 0)
    for d_idx, d in enumerate(multi_ds._datasets):
        frm = d.episode_data_index["from"]
        to = d.episode_data_index["to"]
        eps = getattr(d, "episodes", None)
        key = repo_key_for(multi_ds.ds_names[d_idx])
        for p in range(len(frm)):
            plan.append(
                {
                    "plan_idx": len(plan),
                    "dataset_index": d_idx,
                    "repo_key": key,
                    "episode_index": int(eps[p]) if eps is not None else p,
                    "global_start": offset + int(frm[p]),
                    "n_windows": int(to[p]) - int(frm[p]),
                }
            )
        offset += d.num_frames
    return plan


def shard_episodes(plan, shard_id=0, num_shards=1):
    if num_shards < 1 or not 0 <= shard_id < num_shards:
        raise ValueError(
            f"shard must be in [0, num_shards={num_shards}), got {shard_id}"
        )
    return [ep for ep in plan if ep["plan_idx"] % num_shards == shard_id]


def build_cache_meta(data_cfg, dataset_dirs, latent_shape=None):
    meta = {
        "layout": "per_repo",
        "num_frames": int(data_cfg["num_frames"]),
        "num_extra_ref_frames": int(data_cfg.get("num_extra_ref_frames", 0)),
        "action_video_freq_ratio": int(data_cfg["action_video_freq_ratio"]),
        "video_size": [int(x) for x in data_cfg["video_size"]],
        "concat_multi_camera": str(data_cfg.get("concat_multi_camera", "horizontal")),
        "dataset_dirs": [os.path.realpath(str(p)) for p in dataset_dirs],
        "producer": "scripts/precompute_latent_cache.py",
    }
    images = (data_cfg.get("shape_meta") or {}).get("images")
    if images:
        meta["image_keys"] = [str(x["key"]) for x in images]
    if latent_shape is not None:
        meta["latent_shape"] = [int(x) for x in latent_shape]
    return meta


def write_cache_meta(out_dir, meta):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, META_FILENAME)
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)
        clash = {
            k: (existing[k], meta[k])
            for k in meta
            if k in existing and existing[k] != meta[k]
        }
        if clash:
            raise RuntimeError(
                f"meta.json at {path} has a cache configuration mismatch: {clash}. Use a fresh out_dir."
            )
        legacy_missing = [k for k in meta if k not in existing]
        if legacy_missing:
            warnings.warn(
                f"meta.json at {path} is missing keys {legacy_missing}. Regenerate the cache to include them."
            )
        return path
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    os.replace(tmp, path)
    return path


def episode_cache_path(out_dir, repo_key, episode_index):
    return os.path.join(out_dir, repo_key, f"ep_{int(episode_index):06d}.safetensors")


def episode_is_complete(path, n_windows, expected_meta=None):
    if not os.path.exists(path):
        return False
    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt") as f:
            if int(f.get_slice("latents").get_shape()[0]) != int(n_windows):
                return False
            file_meta = f.metadata() or {}
    except Exception:
        return False
    if expected_meta:
        clash = {
            k: (file_meta[k], str(v))
            for k, v in expected_meta.items()
            if k in file_meta and file_meta[k] != str(v)
        }
        if clash:
            raise RuntimeError(
                f"latent cache file {path} __metadata__ mismatches the current recipe ({clash}). Use a fresh out_dir."
            )
    return True


def assert_window_identity(window_ids, ep, global_idx):
    if window_ids is None:
        return
    exp_fr = int(global_idx) - int(ep["global_start"])
    got_ep = int(window_ids["episode_index"])
    got_fr = int(window_ids["frame_index"])
    got_repo = repo_key_for(window_ids["repo_id"])
    if (
        got_ep != int(ep["episode_index"])
        or got_fr != exp_fr
        or got_repo != ep["repo_key"]
    ):
        raise RuntimeError(
            f"window identity drift at global idx {global_idx}: got (repo={got_repo}, ep={got_ep}, fr={got_fr}) expected (repo={ep['repo_key']}, ep={ep['episode_index']}, fr={exp_fr})."
        )


def save_episode_latents(path, latents, str_meta=None):
    from safetensors.torch import save_file

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    save_file(
        {"latents": latents.to(torch.bfloat16).contiguous()},
        tmp,
        metadata={k: str(v) for k, v in (str_meta or {}).items()},
    )
    os.replace(tmp, path)
