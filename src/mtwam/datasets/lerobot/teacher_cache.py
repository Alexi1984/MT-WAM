import hashlib
import os
import pathlib
import uuid
import torch


class DynamicBranchCacheMissing(FileNotFoundError):
    pass


def build_teacher_cache_key(
    repo_id, episode_index, frame_index, video_sample_indices, video_size, camera
) -> str:
    payload = "|".join(
        (
            str(x)
            for x in (
                repo_id,
                int(episode_index),
                int(frame_index),
                tuple((int(i) for i in video_sample_indices)),
                tuple((int(s) for s in video_size)),
                int(camera),
            )
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


TEACHER_GRID_TRAJ = 196
TEACHER_GRID_DINO = 256
TEACHER_HORIZON_P = 2


def teacher_p_max(num_sampled_frames, anchor_offset=0, vae_temporal_factor=4) -> int:
    p = (int(num_sampled_frames) - 1 - int(anchor_offset)) // int(vae_temporal_factor)
    if p < 1:
        raise ValueError(
            f"teacher window too short: {num_sampled_frames} sampled frames at anchor_offset={anchor_offset} give P_max={p} (< 1 future latent frame) — lengthen num_frames."
        )
    return p


def teacher_cache_path(cache_dir, key: str, horizon_p: int = TEACHER_HORIZON_P) -> str:
    return os.path.join(cache_dir, f"{key}.teacher_P{int(horizon_p)}.pt")


def atomic_save_teacher(payload: dict, output_path) -> None:
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.parent / f".{output_path.name}.tmp.{uuid.uuid4().hex}"
    torch.save(payload, str(tmp))
    os.replace(tmp, output_path)


def load_teacher_payload(
    path: str, traj_grid: int = TEACHER_GRID_TRAJ, dino_grid: int = TEACHER_GRID_DINO
) -> dict:
    if not os.path.exists(path):
        raise DynamicBranchCacheMissing(
            f"Missing teacher cache: {path}. Run scripts/precompute_teacher_feats.py first."
        )
    payload = torch.load(path, map_location="cpu")
    tracks = payload["teacher_tracks"]
    vis = payload["teacher_track_vis"]
    dino = payload["teacher_dino"]
    if tuple(tracks.shape[-2:]) != (traj_grid, 2):
        raise ValueError(
            f"teacher_tracks last dims must be ({traj_grid}, 2), got {tuple(tracks.shape)}"
        )
    if vis.shape[-1] != traj_grid:
        raise ValueError(
            f"teacher_track_vis last dim must be {traj_grid}, got {tuple(vis.shape)}"
        )
    if tuple(dino.shape[-2:]) != (dino_grid, 768):
        raise ValueError(
            f"teacher_dino last dims must be ({dino_grid}, 768), got {tuple(dino.shape)}"
        )
    return payload


def prepare_teacher_frames_for_camera(
    camera_window,
    video_sample_indices,
    video_size,
    horizon_p=None,
    vae_temporal_factor=4,
    anchor_offset=0,
):
    import torchvision.transforms.functional as TF

    vh, vw = (
        (video_size, video_size)
        if isinstance(video_size, int)
        else (int(video_size[0]), int(video_size[1]))
    )
    anchor = int(anchor_offset)
    if anchor < 0:
        raise ValueError(f"anchor_offset must be >= 0, got {anchor_offset}")
    sample_indices = list(video_sample_indices)
    p_max = teacher_p_max(len(sample_indices), anchor, vae_temporal_factor)
    if horizon_p is None:
        horizon_p = p_max
    if not 1 <= horizon_p <= p_max:
        raise ValueError(
            f"horizon_p must be in [1, {p_max}] (window-derived P_max for {len(sample_indices)} sampled frames at anchor_offset={anchor}), got {horizon_p}"
        )
    future_ks = range(p_max - horizon_p + 1, p_max + 1)
    rep_positions = [anchor] + [anchor + vae_temporal_factor * k for k in future_ks]
    if len(sample_indices) <= rep_positions[-1]:
        raise ValueError(
            f"teacher window too short: need > {rep_positions[-1]} sampled frames for P={horizon_p} at anchor_offset={anchor}, got {len(sample_indices)}"
        )
    window = camera_window
    if window.ndim == 3:
        window = window.unsqueeze(0)
    window = window[sample_indices]
    reps = window[rep_positions].float()
    reps = TF.resize(reps, [vh, vw], antialias=True)
    if not (float(reps.min()) >= -0.01 and float(reps.max()) <= 1.01):
        raise ValueError(
            f"teacher frames not in [0,1]: [{float(reps.min()):.3f},{float(reps.max()):.3f}] (producer/consumer must feed raw [0,1] frames, pre-Normalize)"
        )
    return reps


TEACHER_VIDEO_SIZE = (224, 224)


def _raw_id_int(raw: dict, column: str) -> int:
    if column not in raw:
        raise KeyError(
            f"raw lerobot row has no '{column}' column (present: {sorted(raw)}); cannot build teacher cache key — check the dataset parquet schema (DEFAULT_FEATURES merge)."
        )
    value = raw[column]
    return int(value.reshape(-1)[0]) if hasattr(value, "reshape") else int(value)


def derive_teacher_window_id_from_raw(ds, raw) -> dict:
    dataset_index = _raw_id_int(raw, "dataset_index")
    return {
        "repo_id": os.path.realpath(ds.multi_dataset.ds_names[dataset_index]),
        "episode_index": _raw_id_int(raw, "episode_index"),
        "frame_index": _raw_id_int(raw, "frame_index"),
        "dataset_index": dataset_index,
    }


def derive_teacher_window_id(ds, loaded_idx: int) -> dict:
    return derive_teacher_window_id_from_raw(ds, ds.multi_dataset[loaded_idx])


TEACHER_GEOMETRY = {
    "horizontal": {
        "video_size": (224, 224),
        "traj_grid_hw": (14, 14),
        "dino_grid_hw": (16, 16),
    },
    "robotwin": {
        "video_size": (224, 280),
        "traj_grid_hw": (16, 20),
        "dino_grid_hw": (16, 20),
    },
    "robotwin_dense": {
        "video_size": (224, 280),
        "traj_grid_hw": (28, 28),
        "dino_grid_hw": (20, 20),
        "dino_size": (280, 280),
    },
    "horizontal_dense": {
        "video_size": (224, 224),
        "traj_grid_hw": (28, 28),
        "dino_grid_hw": (20, 20),
        "dino_size": (280, 280),
    },
}
for _name, _g in TEACHER_GEOMETRY.items():
    _vh, _vw = _g["video_size"]
    _tgh, _tgw = _g["traj_grid_hw"]
    assert _vh % _tgh == 0 and _vw % _tgw == 0, (
        f"TEACHER_GEOMETRY[{_name}]: traj grid {(_tgh, _tgw)} must divide video_size {(_vh, _vw)}"
    )
    _dh, _dw = _g.get("dino_size", _g["video_size"])
    assert (
        _dh % 14 == 0
        and _dw % 14 == 0
        and ((_dh // 14, _dw // 14) == tuple(_g["dino_grid_hw"]))
    ), (
        f"TEACHER_GEOMETRY[{_name}]: dino grid {_g['dino_grid_hw']} must equal dino input {(_dh, _dw)}/14"
    )


def teacher_geometry(camera_layout: str = "horizontal"):
    if camera_layout not in TEACHER_GEOMETRY:
        raise ValueError(
            f"unknown camera_layout={camera_layout!r} (expected {sorted(TEACHER_GEOMETRY)})"
        )
    g = TEACHER_GEOMETRY[camera_layout]
    return (g["video_size"], g["traj_grid_hw"], g["dino_grid_hw"])


def teacher_dino_size(camera_layout: str = "horizontal"):
    if camera_layout not in TEACHER_GEOMETRY:
        raise ValueError(
            f"unknown camera_layout={camera_layout!r} (expected {sorted(TEACHER_GEOMETRY)})"
        )
    g = TEACHER_GEOMETRY[camera_layout]
    return g.get("dino_size", g["video_size"])


def teacher_key_for_camera(
    ds,
    loaded_idx: int,
    video_sample_indices,
    camera: int,
    video_size=TEACHER_VIDEO_SIZE,
) -> str:
    ids = derive_teacher_window_id(ds, loaded_idx)
    return build_teacher_cache_key(
        ids["repo_id"],
        ids["episode_index"],
        ids["frame_index"],
        video_sample_indices,
        video_size,
        camera,
    )
