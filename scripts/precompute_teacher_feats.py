import logging
import os
import torch
from mtwam.datasets.lerobot.teacher_cache import (
    TEACHER_GRID_DINO,
    TEACHER_GRID_TRAJ,
    TEACHER_HORIZON_P,
    TEACHER_VIDEO_SIZE,
    atomic_save_teacher,
    build_teacher_cache_key,
    derive_teacher_window_id_from_raw,
    prepare_teacher_frames_for_camera,
    teacher_cache_path,
    teacher_geometry,
    teacher_p_max,
)
from mtwam.datasets.lerobot.teacher_extract import (
    DEFAULT_VAE_UPSAMPLING_FACTOR,
    DINO_GRID_HW,
    DINO_MEAN,
    DINO_STD,
    TEACHER_IMG_SIZE,
    TRAJ_GRID_HW,
    TRAJ_PATCH,
    VAE_TEMPORAL_FACTOR,
    _grid_query_points,
    extract_teacher_for_camera,
    load_teacher_models,
)

logger = logging.getLogger(__name__)


def build_camera_payload(cam_frames, extractor) -> dict:
    tracks, vis, dino = extractor(cam_frames)
    return {"teacher_tracks": tracks, "teacher_track_vis": vis, "teacher_dino": dino}


def process_window(
    window: dict, cache_dir: str, extractor, overwrite: bool = False
) -> list:
    statuses = []
    for camera, cam_frames in enumerate(window["per_camera_frames"]):
        key = build_teacher_cache_key(
            window["repo_id"],
            window["episode_index"],
            window["frame_index"],
            window["video_sample_indices"],
            window["video_size"],
            camera,
        )
        path = teacher_cache_path(
            cache_dir, key, horizon_p=window.get("horizon_p", TEACHER_HORIZON_P)
        )
        existed = os.path.exists(path)
        if existed and (not overwrite):
            statuses.append("skip")
            continue
        payload = build_camera_payload(cam_frames, extractor)
        atomic_save_teacher(payload, path)
        statuses.append("overwrite" if existed else "new")
    return statuses


def shard_indices(n, shard_id=0, num_shards=1):
    if num_shards < 1 or not 0 <= shard_id < num_shards:
        raise ValueError(
            f"shard_id must be in [0, num_shards={num_shards}), got {shard_id}"
        )
    return range(shard_id, n, num_shards)


def build_dataset_windows(
    base_dataset,
    video_sample_indices,
    horizon_p=None,
    video_size=TEACHER_VIDEO_SIZE,
    max_samples=None,
    shard_id=0,
    num_shards=1,
):
    vh, vw = (
        (video_size, video_size)
        if isinstance(video_size, int)
        else (int(video_size[0]), int(video_size[1]))
    )
    sample_indices = list(video_sample_indices)
    horizon_p_eff = (
        teacher_p_max(len(sample_indices)) if horizon_p is None else int(horizon_p)
    )
    image_keys = [m["lerobot_key"] for m in base_dataset.image_meta]
    n = len(base_dataset.multi_dataset)
    if max_samples is not None:
        n = min(n, int(max_samples))
    for idx in shard_indices(n, shard_id, num_shards):
        raw = base_dataset.multi_dataset[idx]
        per_camera_frames = [
            prepare_teacher_frames_for_camera(
                raw[key],
                sample_indices,
                (vh, vw),
                horizon_p=horizon_p_eff,
                vae_temporal_factor=VAE_TEMPORAL_FACTOR,
            )
            for key in image_keys
        ]
        ids = derive_teacher_window_id_from_raw(base_dataset, raw)
        yield {
            "repo_id": ids["repo_id"],
            "episode_index": ids["episode_index"],
            "frame_index": ids["frame_index"],
            "video_sample_indices": tuple(sample_indices),
            "video_size": (vh, vw),
            "horizon_p": horizon_p_eff,
            "per_camera_frames": per_camera_frames,
        }


def _load_exclude_episodes_file(path):
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(int(line))
    return out


def main(cfg):
    from omegaconf import DictConfig, ListConfig, OmegaConf
    from tqdm import tqdm
    from mtwam.datasets.lerobot.base_lerobot_dataset import (
        BaseLerobotDataset,
        load_episode_ids_file,
    )

    logging.basicConfig(level=logging.INFO)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError(
            "Teacher precomputation requires WORLD_SIZE=1. Run one process per GPU with explicit shard settings."
        )
    if cfg.get("data") is None:
        raise ValueError("`cfg.data` is required.")
    options = cfg.teacher_cache
    overwrite = bool(options.overwrite)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cotracker_ckpt = cfg.get("cotracker_checkpoint", None)
    dino_ckpt = cfg.get("dino_checkpoint", None)
    if dino_ckpt is None:
        raise ValueError("`dino_checkpoint` is required (DINOv2 ViT-B/14 .pth).")
    cache_dir = options.out_dir
    if cache_dir is None or not str(cache_dir).strip():
        raise ValueError("`teacher_cache.out_dir` is required.")
    max_samples = options.max_samples
    shard_id = int(options.shard)
    num_shards = int(options.num_shards)
    if num_shards > 1:
        logger.info(
            "Sharded producer: shard %d/%d (strided windows)", shard_id, num_shards
        )
    exclude_file = options.exclude_episodes_file
    exclude_from_file = (
        _load_exclude_episodes_file(str(exclude_file)) if exclude_file else None
    )
    if exclude_from_file is not None:
        logger.info(
            "exclude_episodes_file=%s -> %d episodes excluded (flat, all repos)",
            exclude_file,
            len(exclude_from_file),
        )
    logger.info(
        "Loading teachers (cotracker=%s dino=%s) on %s",
        cotracker_ckpt,
        dino_ckpt,
        device,
    )
    models = load_teacher_models(cotracker_ckpt, dino_ckpt, device=device)

    def _iter_dataset_nodes(node):
        if isinstance(node, DictConfig):
            if node.get("dataset_dirs") is not None:
                yield node
            for value in node.values():
                yield from _iter_dataset_nodes(value)
        elif isinstance(node, ListConfig):
            for value in node:
                yield from _iter_dataset_nodes(value)

    node_list = list(_iter_dataset_nodes(cfg.data))
    if not node_list:
        raise ValueError("No dataset node with `dataset_dirs` found under cfg.data.")
    stats = {"new": 0, "overwrite": 0, "skip": 0}
    for node in node_list:
        num_frames = int(node.get("num_frames", 33))
        freq = int(node.get("action_video_freq_ratio", 1))
        if int(node.get("num_extra_ref_frames", 0)) != 0:
            raise ValueError("Teacher precomputation requires num_extra_ref_frames=0.")
        included = node.get("include_episodes")
        if not included and node.get("include_episodes_file"):
            included = load_episode_ids_file(node.include_episodes_file)
        base = BaseLerobotDataset(
            dataset_dirs=[str(d) for d in node["dataset_dirs"]],
            shape_meta=OmegaConf.to_container(node["shape_meta"], resolve=True),
            include_episodes=included,
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=float(node.get("val_set_proportion", 0.0)),
            is_training_set=bool(node.get("is_training_set", True)),
            global_sample_stride=int(node.get("global_sample_stride", 1)),
            exclude_episodes=exclude_from_file
            if exclude_from_file is not None
            else node.get("exclude_episodes", None),
        )
        base._set_return_images(True)
        layout = str(node.get("concat_multi_camera", "horizontal") or "horizontal")
        video_size, traj_grid_hw, dino_grid_hw = teacher_geometry(layout)
        logger.info(
            "Node layout=%s -> teacher crop=%s traj_grid=%s dino_grid=%s",
            layout,
            video_size,
            traj_grid_hw,
            dino_grid_hw,
        )

        def extractor(cam_frames, _tg=traj_grid_hw, _vs=video_size, _dg=dino_grid_hw):
            return extract_teacher_for_camera(
                cam_frames,
                models,
                device=device,
                traj_grid_hw=_tg,
                img_size=_vs,
                dino_grid_hw=_dg,
            )

        vsi = list(range(0, num_frames, freq))
        horizon_p_node = (len(vsi) - 1) // VAE_TEMPORAL_FACTOR
        windows = build_dataset_windows(
            base,
            vsi,
            horizon_p=horizon_p_node,
            video_size=video_size,
            max_samples=max_samples,
            shard_id=shard_id,
            num_shards=num_shards,
        )
        for window in tqdm(windows, desc="windows"):
            for s in process_window(
                window, str(cache_dir), extractor, overwrite=overwrite
            ):
                stats[s] += 1
    logger.info(
        "Finished teacher precompute. new=%d overwrite=%d skip=%d",
        stats["new"],
        stats["overwrite"],
        stats["skip"],
    )


if __name__ == "__main__":
    import hydra

    hydra.main(
        config_path="../configs", config_name="train_libero", version_base="1.3"
    )(main)()
