import json
import inspect
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional
import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from experiments.libero.libero_utils import (
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    invert_gripper_action,
    quat2axisangle,
    save_rollout_video,
)
from mtwam.datasets.lerobot.processors.mtwam_processor import MTWAMProcessor
from mtwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from mtwam.utils.pytorch_utils import set_global_seed
from mtwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from experiments.libero.action_ensembler import ActionEnsembler
from mtwam.utils.libero_results import read_tasks, variant_result_done
from mtwam.utils.eval_config import evaluation_identity, find_dataset_stats

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_eval_device(cfg: DictConfig) -> str:
    eval_device = cfg.EVALUATION.get("device")
    if eval_device is not None:
        return str(eval_device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dataset_stats_path(cfg: DictConfig) -> Path:
    return find_dataset_stats(cfg.ckpt, cfg.EVALUATION.get("dataset_stats_path"))


def _resolve_eval_branch_horizon(cfg: DictConfig, ckpt_path: str) -> None:
    from mtwam.utils.branch_arch_reconcile import resolve_eval_branch_horizon

    resolve_eval_branch_horizon(cfg.model, ckpt_path, cfg.data.train)


def _check_train_eval_branch_arch(cfg: DictConfig, ckpt_path: str) -> None:
    from mtwam.utils.branch_arch_reconcile import check_train_eval_branch_arch

    check_train_eval_branch_arch(
        cfg.model,
        ckpt_path,
        eval_data_num_frames=int(cfg.data.train.get("num_frames", 33)),
        eval_data_num_extra_ref_frames=int(
            cfg.data.train.get("num_extra_ref_frames", 0)
        ),
    )


def _load_model_checkpoint(model: torch.nn.Module, ckpt: str) -> None:
    model.load_checkpoint(ckpt)
    logging.info("Loaded checkpoint via model.load_checkpoint: %s", ckpt)


def _center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil_image = Image.fromarray(image)
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize(
        (round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR
    )
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    cropped = resized.crop((left, top, left + width, top + height))
    return np.asarray(cropped, dtype=np.uint8)


def _normalize_proprio(proprio: np.ndarray, processor: MTWAMProcessor) -> torch.Tensor:
    state_meta = processor.shape_meta["state"]
    if len(state_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged state key in shape_meta['state']."
        )
    state_key = state_meta[0]["key"]
    state_batch = {
        "state": {state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)}
    }
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    return state_batch["state"][state_key]


def _obs_to_model_input(
    obs: dict,
    cfg: DictConfig,
    processor: MTWAMProcessor,
    width: int,
    height: int,
    device: str,
    dtype: torch.dtype,
):
    imgs = get_libero_image(obs)
    image_meta = processor.shape_meta["images"]
    if len(image_meta) < int(processor.num_output_cameras):
        raise ValueError(
            f"shape_meta.images has {len(image_meta)} entries, but num_output_cameras={processor.num_output_cameras}."
        )

    def _meta_to_hw(meta: dict, camera_idx: int) -> tuple[int, int]:
        shape = meta["shape"]
        if len(shape) != 3:
            raise ValueError(
                f"shape_meta.images[{camera_idx}].shape must be [C,H,W], got {shape}"
            )
        return (int(shape[1]), int(shape[2]))

    concatenation = cfg.data.train.get("concat_multi_camera", "horizontal")
    num_cameras = processor.num_output_cameras
    if num_cameras == 1:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        rgb = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
    elif num_cameras == 2:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        wrist_h, wrist_w = _meta_to_hw(image_meta[1], camera_idx=1)
        primary = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
        wrist = _center_crop_resize(imgs["wrist_image"], width=wrist_w, height=wrist_h)
        if concatenation == "horizontal":
            rgb = np.concatenate([primary, wrist], axis=1)
        elif concatenation == "vertical":
            rgb = np.concatenate([primary, wrist], axis=0)
        else:
            raise ValueError(f"Invalid concat_multi_camera: {concatenation}")
    else:
        raise ValueError(
            f"LIBERO eval currently supports num_output_cameras in [1, 2], got {num_cameras}."
        )
    actual_h, actual_w = (int(rgb.shape[0]), int(rgb.shape[1]))
    expected_h, expected_w = (int(height), int(width))
    image_shapes = [meta["shape"] for meta in image_meta]
    assert actual_h == expected_h and actual_w == expected_w, (
        f"Input image size mismatch after per-camera resize + concat: got (H,W)=({actual_h},{actual_w}), expected (H,W)=({expected_h},{expected_w}) from data.train.video_size={[expected_h, expected_w]}; shape_meta.images={image_shapes}, concat_multi_camera={concatenation}."
    )
    x = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    x = x * (2.0 / 255.0) - 1.0
    proprio = _normalize_proprio(_extract_sim_state(obs), processor)
    return (x, proprio, imgs)


def _extract_sim_state(obs: dict) -> np.ndarray:
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)
    return state


def _denormalize_action(action: torch.Tensor, processor: MTWAMProcessor) -> np.ndarray:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if action.ndim != 3:
        raise ValueError(f"Expected action tensor [B, T, D], got {tuple(action.shape)}")
    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged action key in shape_meta['action']."
        )
    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    action = action.to(dtype=torch.float32, device="cpu")
    denorm = normalizer.backward(action)
    return denorm.numpy()


def _predict_action_chunk(
    obs,
    task_description,
    model,
    processor,
    cfg,
    *,
    action_horizon,
    input_w,
    input_h,
    model_device,
):
    steps = cfg.EVALUATION.get("num_inference_steps", None)
    num_inference_steps = int(
        cfg.get("eval_num_inference_steps", 20) if steps is None else steps
    )
    prompt = DEFAULT_PROMPT.format(task=task_description)
    image, proprio, imgs = _obs_to_model_input(
        obs,
        cfg=cfg,
        processor=processor,
        width=input_w,
        height=input_h,
        device=model_device,
        dtype=model.torch_dtype,
    )
    with torch.no_grad():
        pred = model.infer_action(
            prompt=prompt,
            input_image=image,
            action_horizon=action_horizon,
            negative_prompt=str(cfg.EVALUATION.get("negative_prompt", "")),
            text_cfg_scale=float(cfg.EVALUATION.get("text_cfg_scale", 1.0)),
            num_inference_steps=num_inference_steps,
            proprio=proprio,
            sigma_shift=None
            if cfg.EVALUATION.get("sigma_shift") is None
            else float(cfg.EVALUATION.sigma_shift),
            seed=None if cfg.get("seed") is None else int(cfg.seed),
            rand_device=str(cfg.EVALUATION.get("rand_device", "cpu")),
            tiled=bool(cfg.EVALUATION.get("tiled", False)),
        )
    action = _denormalize_action(pred["action"], processor)[0]
    action[..., -1] = action[..., -1] * 2 - 1
    action = invert_gripper_action(action)
    if bool(cfg.EVALUATION.get("binarize_gripper", False)):
        action[..., -1] = np.sign(action[..., -1])
    return (action, imgs)


def _get_max_steps(task_suite_name: str) -> int:
    suite_steps = {
        "libero_spatial": 400,
        "libero_object": 400,
        "libero_goal": 400,
        "libero_10": 700,
        "libero_90": 700,
    }
    if task_suite_name not in suite_steps:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    return suite_steps[task_suite_name]


def run_single_episode(
    env,
    initial_state,
    task_description,
    model,
    processor,
    cfg,
    episode_idx,
    *,
    action_horizon,
    input_w,
    input_h,
    model_device,
):
    max_steps = _get_max_steps(cfg.EVALUATION.task_suite_name)
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    num_steps_wait = int(cfg.EVALUATION.get("num_steps_wait", 5))
    use_action_ensembler = bool(cfg.EVALUATION.get("use_action_ensembler", False))
    env.reset()
    obs = env.set_init_state(initial_state)
    if use_action_ensembler:
        ensembler = ActionEnsembler()
        ensembler.reset()
    replay_images = []
    pending_actions = []
    t = 0
    done = False
    pbar = tqdm(total=max_steps + num_steps_wait, desc=f"Episode {episode_idx + 1}")
    while t < max_steps + num_steps_wait:
        pbar.update(1)
        if t < num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action())
            t += 1
            continue
        if len(pending_actions) == 0:
            action_chunk, imgs = _predict_action_chunk(
                obs=obs,
                task_description=task_description,
                model=model,
                processor=processor,
                cfg=cfg,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
            )
            if use_action_ensembler:
                ensembler.add_actions(action_chunk, t)
                pending_actions = [
                    ensembler.get_action(ts).tolist()
                    for ts in range(t, t + replan_steps)
                ]
            else:
                pending_actions = action_chunk[:replan_steps].tolist()
            replay_images.append(imgs.copy())
        else:
            imgs = get_libero_image(obs)
            replay_images.append(imgs.copy())
        obs, _, done, _ = env.step(pending_actions.pop(0))
        if done:
            break
        t += 1
    pbar.close()
    return (bool(done), replay_images)


def run_single_task(
    task,
    initial_states,
    model: torch.nn.Module,
    processor: MTWAMProcessor,
    cfg: DictConfig,
    video_dir: Path,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
) -> dict:
    env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, cfg.get("seed"))
    results = {
        "successes": 0,
        "failure_episodes": [],
        "success_episodes": [],
        "task_description": task_description,
    }
    try:
        for trial_idx in range(int(cfg.EVALUATION.num_trials)):
            success, replay_images = run_single_episode(
                env=env,
                initial_state=initial_states[trial_idx],
                task_description=task_description,
                model=model,
                processor=processor,
                cfg=cfg,
                episode_idx=trial_idx,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
            )
            if success:
                results["successes"] += 1
                results["success_episodes"].append(trial_idx)
            else:
                results["failure_episodes"].append(trial_idx)
            save_rollout_video(
                video_dir,
                replay_images,
                f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                success=success,
                task_description=task_description,
            )
        return results
    finally:
        try:
            env.close()
        except Exception:
            pass


def _run_one_variant(
    cfg,
    suite,
    task_id,
    *,
    t_start,
    model,
    processor,
    action_horizon,
    input_h,
    input_w,
    model_device,
    local_log_dir,
):
    from libero.libero import benchmark

    cfg.EVALUATION.task_suite_name = suite
    cfg.EVALUATION.task_id = task_id
    video_dir = local_log_dir / suite / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite]()
    task = task_suite.get_task(task_id)
    initial_states = list(task_suite.get_task_init_states(task_id))
    if not initial_states:
        raise ValueError(f"No initial states available for {suite},{task_id}.")
    while len(initial_states) < int(cfg.EVALUATION.num_trials):
        initial_states.extend(
            initial_states[: int(cfg.EVALUATION.num_trials) - len(initial_states)]
        )
    results = {
        "task_suite": suite,
        "task_id": task_id,
        "task_description": None,
        "successes": 0,
        "total_episodes": int(cfg.EVALUATION.num_trials),
        "gpu_id": int(cfg.gpu_id),
        "success_episodes": [],
        "failure_episodes": [],
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": 0,
    }
    logging.info("Running LIBERO evaluation with env_num=1")
    task_results = run_single_task(
        task=task,
        initial_states=initial_states,
        model=model,
        processor=processor,
        cfg=cfg,
        video_dir=video_dir,
        action_horizon=action_horizon,
        input_w=input_w,
        input_h=input_h,
        model_device=model_device,
    )
    results.update(task_results)
    results["duration"] = time.time() - t_start
    results["seed"] = int(cfg.seed)
    results["configuration_sha256"] = evaluation_identity(cfg)
    output_dir = local_log_dir / suite
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"gpu{cfg.gpu_id}_task{task_id}_results.json"
    temporary = output_file.with_suffix(".json.tmp")
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)
    temporary.replace(output_file)
    for stale in output_dir.glob(f"gpu*_task{task_id}_results.json"):
        if stale != output_file:
            stale.unlink()
    print(
        f"Task {task_id} completed: {results['successes']}/{cfg.EVALUATION.num_trials} successes"
    )
    print(f"Time taken: {results['duration']:.2f} seconds")
    return results


@hydra.main(version_base="1.3", config_path="../../configs", config_name="eval_libero")
def eval_single_process(cfg: DictConfig):
    if cfg.ckpt is None:
        raise ValueError("Set ckpt to an MT-WAM checkpoint.")
    if not Path(str(cfg.ckpt)).expanduser().is_file():
        raise FileNotFoundError(f"Checkpoint not found: {cfg.ckpt}")
    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    if int(cfg.EVALUATION.num_trials) < 1:
        raise ValueError("EVALUATION.num_trials must be positive.")
    start_time = time.time()
    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)
    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")
    env_num = int(cfg.EVALUATION.get("env_num", 1))
    if env_num != 1:
        raise ValueError(
            "Only env_num=1 is supported in eval_libero_single.py. Use run_libero_manager.py for task parallelism."
        )
    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    k_hist_cfg = int(cfg.data.train.get("num_extra_ref_frames", 0) or 0)
    k_model_cfg = int(cfg.model.get("num_extra_ref_frames", 0) or 0)
    if k_model_cfg not in (0, k_hist_cfg):
        raise ValueError(
            f"num_extra_ref_frames drift: model config says {k_model_cfg}, data config says {k_hist_cfg}. Match the model and data settings."
        )
    _resolve_eval_branch_horizon(cfg, str(cfg.ckpt))
    _check_train_eval_branch_arch(cfg, str(cfg.ckpt))
    model = instantiate(
        cfg.model,
        model_dtype=model_dtype,
        device=model_device,
        num_extra_ref_frames=k_hist_cfg,
        num_frames=int(cfg.data.train.num_frames),
    )
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()
    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: MTWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    logging.info("Using dataset stats: %s", dataset_stats_path)
    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    if action_horizon_cfg is None:
        action_horizon = int(cfg.data.train.num_frames) - 1 - k_hist_cfg
    else:
        action_horizon = int(action_horizon_cfg)
    if action_horizon <= 0:
        raise ValueError(
            f"EVALUATION.action_horizon must be positive, got {action_horizon}"
        )
    _replan = int(cfg.EVALUATION.get("replan_steps", 5))
    if not 1 <= _replan <= action_horizon:
        raise ValueError(
            f"EVALUATION.replan_steps={_replan} exceeds action_horizon={action_horizon} — each predicted chunk is only action_horizon steps; set replan_steps <= action_horizon."
        )
    video_size = cfg.data.train.get("video_size", [224, 224])
    if len(video_size) != 2:
        raise ValueError(f"data.train.video_size must be [H, W], got {video_size}")
    input_h = int(video_size[0])
    input_w = int(video_size[1])
    concat_multi_camera = cfg.data.train.get("concat_multi_camera", None)
    shape_meta_images = [meta["shape"] for meta in processor.shape_meta["images"]]
    local_log_dir = Path(cfg.EVALUATION.output_dir)
    local_log_dir.mkdir(parents=True, exist_ok=True)
    worker_task_file = cfg.EVALUATION.get("worker_task_file", None)
    if worker_task_file:
        chunk_path = Path(os.path.expanduser(os.path.expandvars(str(worker_task_file))))
        variants = read_tasks(chunk_path)
        logging.info(
            "BATCH worker gpu_id=%s: %d variants from %s",
            cfg.gpu_id,
            len(variants),
            chunk_path,
        )
        done = 0
        resumed = 0
        skipped = []
        for i, (suite, task_id) in enumerate(variants):
            if variant_result_done(
                local_log_dir,
                suite,
                task_id,
                int(cfg.EVALUATION.num_trials),
                int(cfg.seed),
                evaluation_identity(cfg),
            ):
                resumed += 1
                print(
                    f"[batch] gpu_id={cfg.gpu_id} {i + 1}/{len(variants)} resume-skip {suite},{task_id} (done) resumed={resumed}",
                    flush=True,
                )
                continue
            try:
                _run_one_variant(
                    cfg,
                    suite,
                    task_id,
                    t_start=time.time(),
                    model=model,
                    processor=processor,
                    action_horizon=action_horizon,
                    input_h=input_h,
                    input_w=input_w,
                    model_device=model_device,
                    local_log_dir=local_log_dir,
                )
                done += 1
            except Exception as exc:
                logging.exception("BATCH skip %s,%s: %s", suite, task_id, exc)
                skipped.append(f"{suite},{task_id}")
            print(
                f"[batch] gpu_id={cfg.gpu_id} {i + 1}/{len(variants)} done={done} resumed={resumed} skipped={len(skipped)}",
                flush=True,
            )
        if skipped:
            skip_file = local_log_dir / f"skipped_{chunk_path.stem}.txt"
            skip_file.write_text("\n".join(skipped) + "\n", encoding="utf-8")
            logging.warning(
                "BATCH worker gpu_id=%s skipped %d variants -> %s",
                cfg.gpu_id,
                len(skipped),
                skip_file,
            )
        print(
            f"[batch] worker gpu_id={cfg.gpu_id} FINISHED: done={done} resumed={resumed} skipped={len(skipped)}"
        )
        if skipped:
            raise RuntimeError(
                f"{len(skipped)} LIBERO variants failed; see worker log."
            )
        return {"done": done, "resumed": resumed, "skipped": skipped}
    return _run_one_variant(
        cfg,
        cfg.EVALUATION.task_suite_name,
        int(cfg.EVALUATION.task_id),
        t_start=start_time,
        model=model,
        processor=processor,
        action_horizon=action_horizon,
        input_h=input_h,
        input_w=input_w,
        model_device=model_device,
        local_log_dir=local_log_dir,
    )


if __name__ == "__main__":
    eval_single_process()
