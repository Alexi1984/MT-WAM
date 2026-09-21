import logging
import json
import inspect
import os
import re
from math import ceil
from pathlib import Path
import time
import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from PIL import Image
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)
from torch.utils.data import DataLoader
from .utils.fs import ensure_dir
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler

logger = get_logger(__name__)


class Wan22Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.eval_windows_per_repo = int(cfg.get("eval_windows_per_repo", 16))
        self.eval_window_seed = int(cfg.get("eval_window_seed", 20260705))
        self._eval_fixed_batches = None
        self._eval_fixed_warned = False
        self._eval_visual_warned = False
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        self.resume = cfg.resume
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
        )
        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            self.accelerator.state.deepspeed_plugin.deepspeed_config.get(
                "zero_optimization", {}
            ).get("stage", "unknown"),
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")
        self._apply_dit_only_train_mode(self.model)
        trainable_params = list(self.model.dit.parameters())
        proprio_encoder = getattr(self.model, "proprio_encoder", None)
        if proprio_encoder is not None:
            trainable_params.extend(list(proprio_encoder.parameters()))
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        self.train_loader = self._build_loader(
            self.train_dataset, worker_init_fn=worker_init_fn
        )
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        warmup_steps = int(total_train_steps * 0.05)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")
        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)
        self.model, self.optimizer, self.train_loader, self.scheduler = (
            self.accelerator.prepare(
                self.model, self.optimizer, self.train_loader, self.scheduler
            )
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        self._resume_or_load_checkpoint()
        val_size = (
            len(self.val_dataset)
            if self.val_dataset is not None
            else len(self.train_dataset)
        )
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e
        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            group=None
            if self.cfg.wandb.group in (None, "null", "")
            else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
        )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def _finish_wandb(self):
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def _build_loader(self, dataset, worker_init_fn=None):
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
        )

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(
                f"`{dataset_name}` must implement __len__ for rank consistency checks."
            )
        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor(
                [local_length], device=self.accelerator.device, dtype=torch.int64
            )
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return
        if self.accelerator.is_main_process:
            print(
                f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:"
            )
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)
        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError(
                "`train_dataset` must implement __len__ when `max_steps` is None."
            )
        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(
            ceil(len(self.train_dataset) / global_batch_size), 1
        )
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps), 1
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(
        self, scheduler_type, total_train_steps: int, warmup_steps: int = 0
    ):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)
        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer, T_max=remaining_steps, eta_min=self.learning_rate * 0.01
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(
                self.optimizer, factor=1.0, total_iters=remaining_steps
            )
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. Expected one of: ['cosine', 'constant']."
            )
        if warmup_steps <= 0:
            return main_scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )

    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-06)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-09))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return (f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec)

    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            logger.info("Resuming full training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint only: %s", resume)
        self.accelerator.unwrap_model(self.model).load_checkpoint(
            str(resume_path), optimizer=None
        )
        logger.warning(
            "Loaded policy weights only; optimizer, scheduler and step were not restored."
        )

    def _set_dit_only_train_mode(self):
        logger.info("Setting DiT to train mode and freezing other model components.")
        model = self.accelerator.unwrap_model(self.model)
        self._apply_dit_only_train_mode(model)

    @staticmethod
    def _apply_dit_only_train_mode(model):
        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)
        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(
                f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}"
            )
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(
                f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}"
            )
        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(
                f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}"
            )
        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(
                    f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}"
                )
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(
                    f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, got {action.shape[1]}"
                )
            action_horizon = int(action.shape[1])
        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(
                    f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}"
                )
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(
                    f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}"
                )
        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError(
                    "`context` and `context_mask` must both exist in eval sample."
                )
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
        out = {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
        }
        for key in ("teacher_frames", "cached_latent"):
            v = sample.get(key, None)
            if isinstance(v, torch.Tensor):
                out[key] = v.unsqueeze(0)
        return out

    def _eval_fixed_window_losses(self, model):
        ds = self.val_dataset
        multi = getattr(getattr(ds, "lerobot_dataset", None), "multi_dataset", None)
        if multi is None or not hasattr(ds, "_get"):
            if not self._eval_fixed_warned:
                logger.warning(
                    "Validation dataset lacks multi_dataset/_get; using single-sample validation loss."
                )
                self._eval_fixed_warned = True
            return None
        from torch.utils.data import default_collate
        from .datasets.lerobot.latent_cache import build_episode_plan, repo_key_for
        from .utils.eval_windows import (
            build_val_window_set,
            shard_round_robin,
            window_seed,
        )

        if self._eval_fixed_batches is None:
            plan = build_episode_plan(multi)
            windows = build_val_window_set(
                plan, self.eval_windows_per_repo, self.eval_window_seed
            )
            mine = shard_round_robin(
                windows, self.accelerator.process_index, self.accelerator.num_processes
            )
            batches = []
            for w in mine:
                s = ds._get(int(w["global_idx"]), return_window_ids=True)
                wid = s.pop("window_ids", None)
                ok = (
                    wid is not None
                    and repo_key_for(wid["repo_id"]) == w["repo_key"]
                    and (int(wid["episode_index"]) == w["episode_index"])
                    and (int(wid["frame_index"]) == w["frame_index"])
                )
                if not ok:
                    logger.warning(
                        "fixed-window eval: identity drift at global idx %s — window skipped",
                        w["global_idx"],
                    )
                    continue
                batches.append((w, default_collate([s])))
            self._eval_fixed_batches = batches
            logger.info(
                "fixed-window eval: cached %d/%d windows on rank %d (n_per_repo=%d, seed=%d)",
                len(batches),
                len(mine),
                self.accelerator.process_index,
                self.eval_windows_per_repo,
                self.eval_window_seed,
            )
        keys = ("loss_action", "loss_video", "loss_traj", "loss_tex")
        dev = self.accelerator.device
        sums = torch.zeros(len(keys) + 1, device=dev, dtype=torch.float64)
        counts = torch.zeros(len(keys) + 1, device=dev, dtype=torch.float64)
        for w, batch in self._eval_fixed_batches:
            b = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in batch.items()}
            devices = [dev] if dev.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(
                    window_seed(w["repo_key"], w["episode_index"], w["frame_index"])
                )
                with self.accelerator.autocast():
                    loss, ld = model.training_loss(b)
            sums[0] += float(loss.float().item())
            counts[0] += 1.0
            for i, k in enumerate(keys):
                if k in ld:
                    sums[i + 1] += float(ld[k])
                    counts[i + 1] += 1.0
        sums = self.accelerator.reduce(sums, reduction="sum")
        counts = self.accelerator.reduce(counts, reduction="sum")
        if float(counts[0].item()) == 0.0:
            logger.warning(
                "No scoreable fixed windows; using single-sample validation loss."
            )
            return None
        out = {"val_loss": float((sums[0] / counts[0]).item())}
        for i, k in enumerate(keys):
            if float(counts[i + 1].item()) > 0:
                out["val_" + k] = float((sums[i + 1] / counts[i + 1]).item())
        return out

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None
        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        model.eval()
        rng = torch.Generator(device="cpu").manual_seed(
            self.global_step + self.accelerator.process_index
        )
        eval_index = torch.randint(0, len(self.val_dataset), (1,), generator=rng).item()
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])
        fixed_losses = self._eval_fixed_window_losses(model)
        if fixed_losses is None:
            with self.accelerator.autocast():
                val_loss, _ = model.training_loss(sample)
                val_loss = val_loss.float().item()
        else:
            val_loss = fixed_losses["val_loss"]
        if was_dit_training:
            self._set_dit_only_train_mode()
        result = {"val_loss": float(val_loss)}
        if fixed_losses is not None:
            for k, v in fixed_losses.items():
                if k != "val_loss":
                    result[k] = v
        return result

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        model.save_checkpoint(ckpt_path, optimizer=None, step=self.global_step)
        return ckpt_path

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"
        self.accelerator.wait_for_everyone()
        ckpt_path = None
        if self.accelerator.is_main_process:
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        self.accelerator.wait_for_everyone()
        state_path = os.path.join(self.state_dir, step_tag)
        ensure_dir(state_path)
        self.accelerator.save_state(output_dir=state_path)
        if self.accelerator.is_main_process:
            self._save_trainer_state(state_path)
        self.accelerator.wait_for_everyone()
        return {"weights_path": ckpt_path, "state_path": state_path}

    def load_training_state(self, state_dir: str):
        self.accelerator.load_state(input_dir=state_dir)
        state_file = Path(state_dir) / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.global_step = int(payload["global_step"])
            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self.train_sampler.set_epoch_offset(self.epoch)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch
                    * self.batch_size
                    * self.accelerator.num_processes,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return
        match = re.search("step[_-](\\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info(
            "Loaded accelerate training state from %s at step=%d",
            state_dir,
            self.global_step,
        )
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def train(self):
        self._set_dit_only_train_mode()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        if self.max_steps is None:
            raise ValueError(
                "`max_steps` must be set before entering the while-step training loop."
            )
        profile_on = os.environ.get("MTWAM_PROFILE") == "1"
        trace_dir = os.environ.get("MTWAM_PROFILE_TRACE")
        prof_sums, prof_steps = (
            {"data": 0.0, "fwd": 0.0, "bwd": 0.0, "opt": 0.0, "total": 0.0},
            0,
        )
        torch_profiler = None
        if trace_dir and self.accelerator.is_main_process:
            ensure_dir(trace_dir)
            torch_profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=15, warmup=2, active=3, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
                with_stack=False,
            )
            torch_profiler.start()

        def _prof_mark():
            if profile_on:
                torch.cuda.synchronize()
            return time.perf_counter()

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()
        while self.global_step < self.max_steps:
            t0 = _prof_mark() if profile_on else None
            try:
                sample = next(data_iter)
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                data_iter = iter(self.train_loader)
                continue
            t_data = _prof_mark() if profile_on else None
            with self.accelerator.accumulate(self.model):
                train_model = (
                    self.model
                    if hasattr(self.model, "training_loss")
                    else self.accelerator.unwrap_model(self.model)
                )
                with self.accelerator.autocast():
                    loss, loss_dict = train_model.training_loss(sample)
                t_fwd = _prof_mark() if profile_on else None
                self.accelerator.backward(loss)
                t_bwd = _prof_mark() if profile_on else None
                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()
                    if not self.accelerator.optimizer_step_was_skipped:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    if profile_on:
                        t_opt = _prof_mark()
                        prof_sums["data"] += t_data - t0
                        prof_sums["fwd"] += t_fwd - t_data
                        prof_sums["bwd"] += t_bwd - t_fwd
                        prof_sums["opt"] += t_opt - t_bwd
                        prof_sums["total"] += t_opt - t0
                        prof_steps += 1
                        if torch_profiler is not None:
                            torch_profiler.step()
                        if (
                            self.accelerator.is_main_process
                            and prof_steps % max(1, self.log_every) == 0
                        ):
                            means = {k: v / prof_steps for k, v in prof_sums.items()}
                            logger.info(
                                "[profile] step=%d means over %d steps: data=%.2fs fwd=%.2fs bwd=%.2fs opt+comm=%.2fs total=%.2fs",
                                self.global_step,
                                prof_steps,
                                means["data"],
                                means["fwd"],
                                means["bwd"],
                                means["opt"],
                                means["total"],
                            )
                    global_loss = float(
                        self.accelerator.gather(loss.detach().float().reshape(1))
                        .mean()
                        .item()
                    )
                    global_loss_metrics = {}
                    for key, value in loss_dict.items():
                        metric_tensor = torch.tensor(
                            float(value), device=loss.device, dtype=torch.float32
                        ).reshape(1)
                        global_loss_metrics[key] = float(
                            self.accelerator.gather(metric_tensor).mean().item()
                        )
                    grad_norm_tensor = torch.tensor(
                        grad_norm, device=loss.device, dtype=torch.float32
                    )
                    global_grad_norm = float(
                        self.accelerator.gather(grad_norm_tensor).mean().item()
                    )
                    current_lr = float(self.optimizer.param_groups[0]["lr"])
                    if (
                        self.log_every > 0
                        and self.global_step % self.log_every == 0
                        and self.accelerator.is_main_process
                    ):
                        eta_str, steps_per_sec = self._estimate_eta()
                        description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                            self.epoch,
                            self.global_step,
                            self.max_steps,
                            global_loss,
                        )
                        if global_loss_metrics:
                            detail_str = " ".join(
                                [
                                    f"{k}={v:.4f}"
                                    for k, v in sorted(global_loss_metrics.items())
                                ]
                            )
                            description += detail_str + " "
                        description += (
                            "lr=%.2e speed=%.2f step/s, %.2f samples/s eta=%s"
                            % (
                                current_lr,
                                steps_per_sec,
                                steps_per_sec
                                * self.batch_size
                                * self.accelerator.num_processes,
                                eta_str,
                            )
                        )
                        logger.info(description)
                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/lr": current_lr,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": steps_per_sec
                            * self.batch_size
                            * self.accelerator.num_processes,
                        }
                        for key, value in global_loss_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        self._wandb_log(wandb_payload)
                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and (self.global_step % self.eval_every == 0)
                    ):
                        metrics = self.evaluate()
                        self.accelerator.wait_for_everyone()
                        if metrics is not None and self.accelerator.is_main_process:
                            description = "[eval] step=%d val_loss=%.4f" % (
                                self.global_step,
                                metrics["val_loss"],
                            )
                            if "psnr_rd" in metrics:
                                description += " infer_psnr=%.4f infer_ssim=%.4f" % (
                                    metrics["psnr_rd"],
                                    metrics["ssim_rd"],
                                )
                            for key in (
                                "val_loss_action",
                                "val_loss_video",
                                "val_loss_traj",
                                "val_loss_tex",
                            ):
                                if key in metrics:
                                    description += " %s=%.4f" % (
                                        key.replace("val_loss_", ""),
                                        metrics[key],
                                    )
                            if "action_l2" in metrics:
                                description += " action_l2=%.4f" % metrics["action_l2"]
                            if "action_l1" in metrics:
                                description += " action_l1=%.4f" % metrics["action_l1"]
                            logger.info(description)
                            eval_payload = {
                                f"eval/{key}": float(value)
                                for key, value in metrics.items()
                                if key != "video_path"
                            }
                            self._wandb_log(eval_payload)
                    if self.save_every > 0 and self.global_step % self.save_every == 0:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[ckpt] step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )
                    if self.global_step >= self.max_steps:
                        if torch_profiler is not None:
                            torch_profiler.stop()
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[done] max_steps reached step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )
                        return
        if torch_profiler is not None:
            torch_profiler.stop()
        ckpt_info = self.save_checkpoint()
        if self.accelerator.is_main_process:
            logger.info(
                "[done] training finished step=%d weights=%s state=%s",
                self.global_step,
                ckpt_info["weights_path"],
                ckpt_info["state_path"],
            )
