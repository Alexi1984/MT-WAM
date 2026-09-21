import hashlib
import json
import os
from typing import Optional
import time
import numpy as np
import traceback
import torch
import torchvision.transforms.functional as transforms_F
from contextlib import contextmanager
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from .teacher_cache import (
    DynamicBranchCacheMissing,
    TEACHER_HORIZON_P,
    derive_teacher_window_id,
    load_teacher_payload,
    prepare_teacher_frames_for_camera,
    teacher_cache_path,
    teacher_geometry,
    teacher_key_for_camera,
    teacher_p_max,
)
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from mtwam.utils.logging_config import get_logger
from mtwam.utils import misc, pytorch_utils
from accelerate import PartialState

logger = get_logger(__name__)
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


def collect_prompts_for_episodes(dataset_dirs, include_episodes=None):
    wanted = None if include_episodes is None else {int(e) for e in include_episodes}
    prompts = []
    seen = set()
    for ds_dir in dataset_dirs:
        episodes_path = os.path.join(str(ds_dir), "meta", "episodes.jsonl")
        if not os.path.exists(episodes_path):
            raise FileNotFoundError(f"Missing episodes file: {episodes_path}")
        with open(episodes_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "episode_index" not in record:
                    raise KeyError(
                        f"Missing `episode_index` at {episodes_path}:{line_idx}"
                    )
                if wanted is not None and int(record["episode_index"]) not in wanted:
                    continue
                for task in record.get("tasks") or []:
                    prompt = DEFAULT_PROMPT.format(task=str(task))
                    if prompt not in seen:
                        seen.add(prompt)
                        prompts.append(prompt)
    return prompts


def read_latent_window(cache_dir, episode_index, frame_index):
    from safetensors import safe_open

    path = os.path.join(cache_dir, f"ep_{int(episode_index):06d}.safetensors")
    if not os.path.exists(path):
        raise DynamicBranchCacheMissing(f"latent cache miss: {path}")
    fr = int(frame_index)
    with safe_open(path, framework="pt") as f:
        return f.get_slice("latents")[fr : fr + 1].squeeze(0)


LATENT_CACHE_META_FILENAME = "meta.json"


def validate_latent_cache_meta(
    cache_dir,
    *,
    num_frames,
    num_extra_ref_frames,
    action_video_freq_ratio,
    video_size=None,
    concat_multi_camera=None,
    image_keys=None,
):
    if not cache_dir:
        raise ValueError(
            "use_cached_latent=true requires latent_cache_dir to be set (got None/empty)."
        )
    meta_path = os.path.join(cache_dir, LATENT_CACHE_META_FILENAME)
    if not os.path.exists(meta_path):
        if int(num_extra_ref_frames) > 0:
            raise ValueError(
                f"use_cached_latent/latent_cache_dir is incompatible with num_extra_ref_frames={int(num_extra_ref_frames)}: this cache has no {LATENT_CACHE_META_FILENAME}. Regenerate the cache with matching settings or set use_cached_latent=false."
            )
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    want = {
        "num_frames": int(num_frames),
        "num_extra_ref_frames": int(num_extra_ref_frames),
        "action_video_freq_ratio": int(action_video_freq_ratio),
    }
    mismatches = {
        k: (meta.get(k), v) for k, v in want.items() if int(meta.get(k, 0)) != v
    }
    if video_size is not None and meta.get("video_size") is not None:
        if [int(x) for x in meta["video_size"]] != [int(x) for x in video_size]:
            mismatches["video_size"] = (meta["video_size"], list(video_size))
    if concat_multi_camera is not None and meta.get("concat_multi_camera") is not None:
        if str(meta["concat_multi_camera"]) != str(concat_multi_camera):
            mismatches["concat_multi_camera"] = (
                meta["concat_multi_camera"],
                concat_multi_camera,
            )
    if image_keys is not None and meta.get("image_keys") is not None:
        if [str(k) for k in meta["image_keys"]] != [str(k) for k in image_keys]:
            mismatches["image_keys"] = (meta["image_keys"], list(image_keys))
    if mismatches:
        detail = "; ".join(
            (f"{k}: cache={a} vs dataset={b}" for k, (a, b) in mismatches.items())
        )
        raise ValueError(
            f"latent cache at {cache_dir} does not match this dataset's window recipe ({detail}). Select a matching latent_cache_dir or regenerate the cache."
        )
    return meta


class RobotVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal",
        override_instruction: Optional[str] = None,
        enable_dynamic_branch: bool = False,
        teacher_feature_cache_dir=None,
        online_teacher: bool = False,
        exclude_episodes=None,
        include_episodes=None,
        include_episodes_file=None,
        latent_cache_dir=None,
        use_cached_latent: bool = False,
        num_extra_ref_frames: int = 0,
        teacher_geometry_override=None,
        dynamic_branch_horizon: int = 2,
    ):
        if include_episodes_file and (not include_episodes):
            from .base_lerobot_dataset import load_episode_ids_file

            include_episodes = load_episode_ids_file(include_episodes_file)
        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=num_frames - 1,
            past_obs_size=num_extra_ref_frames,
            past_action_size=num_extra_ref_frames,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
            exclude_episodes=exclude_episodes,
            include_episodes=include_episodes,
        )
        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio
        self.dynamic_branch_horizon = int(dynamic_branch_horizon)
        if self.dynamic_branch_horizon < 1:
            raise ValueError(
                f"dynamic_branch_horizon must be >= 1, got {dynamic_branch_horizon}"
            )
        assert (num_frames - 1) % self.action_video_freq_ratio == 0, (
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {self.action_video_freq_ratio}"
        )
        assert (num_frames - 1) // self.action_video_freq_ratio % 4 == 0, (
            f"video frames must be divisible by 4 for tokenization, got {(num_frames - 1) // self.action_video_freq_ratio}"
        )
        self.video_sample_indices = list(
            range(0, num_frames, self.action_video_freq_ratio)
        )
        self.num_extra_ref_frames = int(num_extra_ref_frames)
        raw_frames_per_latent = 4 * self.action_video_freq_ratio
        assert (
            self.num_extra_ref_frames >= 0
            and self.num_extra_ref_frames % raw_frames_per_latent == 0
        ), (
            f"num_extra_ref_frames must be a non-negative multiple of 4*action_video_freq_ratio={raw_frames_per_latent} (one latent frame), got {self.num_extra_ref_frames}"
        )
        if self.num_extra_ref_frames > 0:
            t_video = (num_frames - 1) // self.action_video_freq_ratio + 1
            t_lat = (t_video - 1) // 4 + 1
            num_ref_latents = 1 + self.num_extra_ref_frames // raw_frames_per_latent
            assert num_ref_latents < t_lat, (
                f"num_extra_ref_frames={self.num_extra_ref_frames} leaves no predicted latent: num_ref_latents={num_ref_latents} must be < T_lat={t_lat} (raise num_frames or lower K)"
            )
            if enable_dynamic_branch and (not online_teacher):
                raise ValueError(
                    f"num_extra_ref_frames={self.num_extra_ref_frames} with enable_dynamic_branch requires online_teacher=true."
                )
        _anchor_off = self.num_extra_ref_frames // self.action_video_freq_ratio
        self.teacher_p_max = teacher_p_max(len(self.video_sample_indices), _anchor_off)
        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(True)
        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction
        self.enable_dynamic_branch = enable_dynamic_branch
        self.teacher_feature_cache_dir = teacher_feature_cache_dir
        self.online_teacher = online_teacher
        self.teacher_geometry_override = (
            str(teacher_geometry_override) if teacher_geometry_override else None
        )
        self.latent_cache_dir = latent_cache_dir
        self.use_cached_latent = bool(use_cached_latent)
        self._latent_cache_layout = "flat"
        if self.use_cached_latent or self.latent_cache_dir:
            _sm = (
                shape_meta
                if isinstance(shape_meta, dict)
                else OmegaConf.to_container(shape_meta, resolve=True)
            )
            _cache_image_keys = [
                str(x["key"]) for x in (_sm or {}).get("images") or []
            ] or None
            cache_meta = validate_latent_cache_meta(
                self.latent_cache_dir,
                num_frames=num_frames,
                num_extra_ref_frames=self.num_extra_ref_frames,
                action_video_freq_ratio=self.action_video_freq_ratio,
                video_size=video_size,
                concat_multi_camera=concat_multi_camera,
                image_keys=_cache_image_keys,
            )
            if cache_meta is not None:
                self._latent_cache_layout = str(cache_meta.get("layout", "flat"))
        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]}
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]}
        )
        self.normalize_transform = Normalize(args={"mean": 0.5, "std": 0.5})
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError(
                        "pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them."
                    )
                if PartialState().is_main_process:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(
                        dataset_stats, os.path.join(work_dir, "dataset_stats.json")
                    )
                else:
                    dataset_stats = None
                if (
                    torch.distributed.is_available()
                    and torch.distributed.is_initialized()
                ):
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if PartialState().is_main_process:
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(
                        dataset_stats, os.path.join(work_dir, "dataset_stats.json")
                    )
            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)

    def __len__(self):
        return len(self.lerobot_dataset)

    def _get(self, idx, return_window_ids=False):
        sample_idx = idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]
            if not self.skip_padding_as_possible:
                break
            action_is_pad = sample["action_is_pad"]
            image_is_pad = sample["image_is_pad"]
            proprio_is_pad = sample["proprio_is_pad"]
            has_pad = False
            if bool(action_is_pad.any().item()):
                has_pad = True
            if bool(image_is_pad.any().item()):
                has_pad = True
            if bool(proprio_is_pad.any().item()):
                has_pad = True
            if not has_pad or attempt >= self.max_padding_retry:
                break
            sample_idx = np.random.randint(len(self.lerobot_dataset))
        image_is_pad = sample["image_is_pad"]
        video = sample["pixel_values"]
        num_cameras = 1
        if video.ndim == 5:
            video = video[:, self.video_sample_indices, :, :, :]
            num_cameras, T_video, C, H, W = video.shape
        else:
            assert video.ndim == 4, (
                f"Expected video to have shape [T, C, H, W], but got {video.shape}"
            )
            video = video[self.video_sample_indices, :, :, :]
            T_video, C, H, W = video.shape
        image_is_pad = image_is_pad[self.video_sample_indices]
        video = video.view(num_cameras, T_video, C, H, W)
        if self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
                )
            cam_top = transforms_F.resize(
                video[0],
                size=[256, 320],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            cam_left = transforms_F.resize(
                video[1],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            cam_right = transforms_F.resize(
                video[2],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            bottom = torch.cat([cam_left, cam_right], dim=-1)
            video = torch.cat([cam_top, bottom], dim=-2)
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. Expected one of: horizontal, vertical, robotwin."
                )
        else:
            video = video.squeeze(0)
        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)
        video = video.permute(1, 0, 2, 3)
        action = sample["action"]
        proprio = sample["proprio"][:-1, :]
        if video.shape[1] <= 1:
            raise ValueError(
                f"`video` must have at least 2 frames, got shape {tuple(video.shape)}"
            )
        if action.shape[0] % (video.shape[1] - 1) != 0:
            raise ValueError(
                f"`action` horizon must be divisible by `video` transitions, got {action.shape[0]} and {video.shape[1] - 1}"
            )
        task = sample["instruction"]
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)
        context, context_mask = self._get_cached_text_context(instruction)
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)
        data = {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
        }
        if getattr(self, "use_cached_latent", False):
            data["cached_latent"] = self._read_cached_latent(
                int(sample["idx"]), window_ids=sample.get("window_ids")
            )
        self._maybe_attach_teacher(data, sample)
        if return_window_ids:
            data["window_ids"] = sample.get("window_ids")
        return data

    def _maybe_attach_teacher(self, data, sample):
        if not self.enable_dynamic_branch:
            return data
        if getattr(self, "online_teacher", False):
            video_size, _, _ = teacher_geometry(
                getattr(self, "teacher_geometry_override", None)
                or getattr(self, "concat_multi_camera", "horizontal")
            )
            pixel_values = sample["pixel_values"]
            num_cameras = 1 if pixel_values.ndim == 4 else pixel_values.shape[0]
            anchor = int(getattr(self, "num_extra_ref_frames", 0))
            if anchor:
                anchor //= self.action_video_freq_ratio
            frames = [
                prepare_teacher_frames_for_camera(
                    pixel_values if pixel_values.ndim == 4 else pixel_values[c],
                    self.video_sample_indices,
                    video_size,
                    horizon_p=None,
                    anchor_offset=anchor,
                )
                for c in range(num_cameras)
            ]
            data["teacher_frames"] = torch.stack(frames, dim=0)
            return data
        loaded_idx = int(sample["idx"])
        pixel_values = sample["pixel_values"]
        num_cameras = 1 if pixel_values.ndim == 4 else pixel_values.shape[0]
        if getattr(self, "teacher_geometry_override", None):
            raise ValueError(
                f"teacher_geometry_override={self.teacher_geometry_override!r} requires online_teacher=true."
            )
        _hz = int(getattr(self, "dynamic_branch_horizon", 2))
        _p_max = int(getattr(self, "teacher_p_max", TEACHER_HORIZON_P))
        if _hz != _p_max:
            raise ValueError(
                f"dynamic_branch_horizon={_hz} != window P_max={_p_max} requires online_teacher=true; the offline cache uses P={_p_max}."
            )
        video_size, traj_grid_hw, dino_grid_hw = teacher_geometry(
            getattr(self, "concat_multi_camera", "horizontal")
        )
        traj_grid = int(traj_grid_hw[0]) * int(traj_grid_hw[1])
        dino_grid = int(dino_grid_hw[0]) * int(dino_grid_hw[1])
        tracks, vis, dino = ([], [], [])
        for camera in range(num_cameras):
            key = teacher_key_for_camera(
                self.lerobot_dataset,
                loaded_idx,
                self.video_sample_indices,
                camera,
                video_size=video_size,
            )
            payload = load_teacher_payload(
                teacher_cache_path(self.teacher_feature_cache_dir, key, horizon_p=_hz),
                traj_grid=traj_grid,
                dino_grid=dino_grid,
            )
            tracks.append(payload["teacher_tracks"])
            vis.append(payload["teacher_track_vis"])
            dino.append(payload["teacher_dino"])
        data["teacher_tracks"] = torch.stack(tracks, dim=0)
        data["teacher_track_vis"] = torch.stack(vis, dim=0)
        data["teacher_dino"] = torch.stack(dino, dim=0)
        return data

    def _read_cached_latent(self, loaded_idx, window_ids=None):
        ids = (
            window_ids
            if window_ids is not None
            else derive_teacher_window_id(self.lerobot_dataset, int(loaded_idx))
        )
        cache_dir = self.latent_cache_dir
        if getattr(self, "_latent_cache_layout", "flat") == "per_repo":
            cache_dir = os.path.join(
                cache_dir, os.path.basename(str(ids["repo_id"]).rstrip("/"))
            )
        return read_latent_window(cache_dir, ids["episode_index"], ids["frame_index"])

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(
            cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt"
        )
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. Run scripts/precompute_text_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )
        return (context, context_mask)

    def __getitem__(self, idx):
        try:
            data = self._get(idx)
        except DynamicBranchCacheMissing:
            raise
        except Exception as e:
            print(
                f"Error processing sample idx {idx}: {e}. Returning a random sample instead."
            )
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        return data
