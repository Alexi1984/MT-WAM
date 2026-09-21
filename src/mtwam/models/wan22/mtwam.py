from typing import Any, Optional, Sequence, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function
from PIL import Image
from mtwam.utils.logging_config import get_logger
from ._load_guards import branch_fork_span, check_mot_branch_load, strict_bool
from .action_dit import ActionDiT
from .branch_losses import (
    assemble_branch_loss,
    derive_branch_frame_pad,
    predict_branch_heads_per_camera,
)
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

logger = get_logger(__name__)


class MTWAM(torch.nn.Module):
    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        loss_lambda_traj: float = 0.0,
        loss_lambda_tex: float = 0.0,
        loss_traj_validity_mask: bool = True,
        num_extra_ref_frames: int = 0,
        num_frames: int | None = None,
    ):
        super().__init__()
        self.num_extra_ref_frames = int(num_extra_ref_frames)
        if self.num_extra_ref_frames != 0:
            raise ValueError("MT-WAM supports num_extra_ref_frames=0 only.")
        self.num_frames = None if num_frames is None else int(num_frames)
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        self.dit = self.mot
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError(
                    "`text_dim` is required when `text_encoder` is not loaded."
                )
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(
                torch_dtype
            )
        else:
            self.proprio_encoder = None
        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps, shift=video_train_shift
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps, shift=video_infer_shift
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps, shift=action_train_shift
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps, shift=action_infer_shift
        )
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self.loss_lambda_traj = float(loss_lambda_traj)
        self.loss_lambda_tex = float(loss_lambda_tex)
        self.loss_traj_validity_mask = bool(loss_traj_validity_mask)
        self.dynamic_branch_camera_layout = "horizontal"
        self.dynamic_branch_teacher_geometry = None
        self.dynamic_branch_dual_f0 = False
        self.teacher_models = None
        self.to(self.device)
        if getattr(self.mot, "dynamic_branch", None) is not None:
            self.mot.dynamic_branch.to(self.torch_dtype)

    @classmethod
    def _check_dual_f0_bounds(
        cls, enable_dynamic_branch, dynamic_branch_dual_f0, num_extra_ref_frames
    ):
        if not bool(dynamic_branch_dual_f0):
            return
        if not bool(enable_dynamic_branch):
            raise ValueError(
                "dynamic_branch_dual_f0=True requires enable_dynamic_branch=True."
            )
        if int(num_extra_ref_frames) > 0:
            raise ValueError(
                f"dynamic_branch_dual_f0=True supports K=0 only, got num_extra_ref_frames={num_extra_ref_frames}."
            )

    @classmethod
    def _check_f0_slot_mask_mode(cls, video_dit_config, dynamic_branch_dual_f0):
        if not strict_bool(dynamic_branch_dual_f0, field="dynamic_branch_dual_f0"):
            return
        mode = str(video_dit_config.get("video_attention_mask_mode", "bidirectional"))
        if mode != "first_frame_causal":
            knob = "dynamic_branch_dual_f0"
            raise ValueError(
                f"{knob}=True requires video_attention_mask_mode='first_frame_causal', got {mode!r}."
            )

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        video_dit_pretrained_path: str | None = None,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        loss_lambda_traj: float = 0.5,
        loss_lambda_tex: float = 0.25,
        loss_traj_validity_mask: bool = True,
        enable_dynamic_branch: bool = True,
        dynamic_branch_num_layers: int = 10,
        dynamic_branch_read_text: bool = False,
        dynamic_branch_camera_layout: str = "horizontal",
        dynamic_branch_teacher_geometry: str | None = None,
        dynamic_branch_dual_f0: bool = True,
        dynamic_branch_ffn_moe: bool = True,
        dynamic_branch_ffn_moe_routing: str = "role",
        dynamic_branch_horizon: int = 2,
        dynamic_branch_teacher_p_max: int | None = None,
        num_extra_ref_frames: int = 0,
        num_frames: int | None = None,
    ):
        if int(num_extra_ref_frames) != 0:
            raise ValueError("MT-WAM supports num_extra_ref_frames=0 only.")
        required_flags = {
            "enable_dynamic_branch": enable_dynamic_branch,
            "dynamic_branch_dual_f0": dynamic_branch_dual_f0,
            "dynamic_branch_ffn_moe": dynamic_branch_ffn_moe,
        }
        for name, value in required_flags.items():
            if not strict_bool(value, field=name):
                raise ValueError(f"MT-WAM requires {name}=True.")
        if strict_bool(dynamic_branch_read_text, field="dynamic_branch_read_text"):
            raise ValueError("MT-WAM requires dynamic_branch_read_text=False.")
        if str(dynamic_branch_ffn_moe_routing) != "role":
            raise ValueError("MT-WAM requires dynamic_branch_ffn_moe_routing=role.")
        if int(dynamic_branch_num_layers) != 10 or int(dynamic_branch_horizon) != 2:
            raise ValueError(
                "MT-WAM requires dynamic_branch_num_layers=10 and dynamic_branch_horizon=2."
            )
        if (
            dynamic_branch_teacher_p_max is not None
            and int(dynamic_branch_teacher_p_max) != 2
        ):
            raise ValueError("MT-WAM requires two future teacher frames.")
        if dynamic_branch_camera_layout not in {"horizontal", "robotwin"}:
            raise ValueError("MT-WAM supports horizontal and robotwin camera layouts.")
        if dynamic_branch_teacher_geometry not in {
            None,
            "",
            dynamic_branch_camera_layout,
        }:
            raise ValueError(
                "MT-WAM requires teacher geometry matching its camera layout."
            )
        if video_dit_config is None:
            raise ValueError(
                "`video_dit_config` is required for MTWAM.from_wan22_pretrained()."
            )
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for MTWAM.")
        cls._check_f0_slot_mask_mode(video_dit_config, dynamic_branch_dual_f0)
        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
            video_dit_pretrained_path=video_dit_pretrained_path,
        )
        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError(
                "ActionDiT `num_heads` must match video expert for MoT mixed attention."
            )
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError(
                "ActionDiT `attn_head_dim` must match video expert for MoT mixed attention."
            )
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")
        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )
        if enable_dynamic_branch:
            from .dynamic_branch import DynamicBranch
            from ...datasets.lerobot.teacher_cache import (
                TEACHER_HORIZON_P,
                teacher_geometry,
            )

            _cap = (
                TEACHER_HORIZON_P
                if dynamic_branch_teacher_p_max is None
                else int(dynamic_branch_teacher_p_max)
            )
            if not 1 <= int(dynamic_branch_horizon) <= _cap:
                raise ValueError(
                    f"dynamic_branch_horizon must be in [1, {_cap}] (window-derived P_max), got {dynamic_branch_horizon}"
                )
            _geom_name = (
                str(dynamic_branch_teacher_geometry)
                if dynamic_branch_teacher_geometry
                else dynamic_branch_camera_layout
            )
            _, _traj_grid_hw, _dino_grid_hw = teacher_geometry(_geom_name)
            with torch.random.fork_rng(devices=[]):
                mot.dynamic_branch = DynamicBranch(
                    video_expert=video_expert,
                    num_layers=int(dynamic_branch_num_layers),
                    build_heads=True,
                    grid_traj=tuple(_traj_grid_hw),
                    grid_tex=tuple(_dino_grid_hw),
                    read_text=bool(dynamic_branch_read_text),
                    ffn_moe=bool(dynamic_branch_ffn_moe),
                    ffn_moe_routing=str(dynamic_branch_ffn_moe_routing),
                    horizon=int(dynamic_branch_horizon),
                )
        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            loss_lambda_traj=loss_lambda_traj,
            loss_lambda_tex=loss_lambda_tex,
            loss_traj_validity_mask=loss_traj_validity_mask,
            num_extra_ref_frames=num_extra_ref_frames,
            num_frames=num_frames,
        )
        model.dynamic_branch_camera_layout = str(dynamic_branch_camera_layout)
        model.dynamic_branch_teacher_geometry = (
            str(dynamic_branch_teacher_geometry)
            if dynamic_branch_teacher_geometry
            else None
        )
        cls._check_dual_f0_bounds(
            enable_dynamic_branch, dynamic_branch_dual_f0, num_extra_ref_frames
        )
        model.dynamic_branch_dual_f0 = bool(dynamic_branch_dual_f0)
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": "SKIPPED_PRETRAIN"
            if skip_dit_load_from_pretrain
            else action_dit_pretrained_path,
        }
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return (height, width, num_frames)

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return (prompt_emb.to(device=self.device), mask)

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return (context, context_mask)
        if proprio.ndim != 2:
            raise ValueError(
                f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}"
            )
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype)
        proprio_mask = torch.ones(
            (context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device
        )
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    @torch.no_grad()
    def _encode_video_latents(
        self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)
    ):
        z = self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return z

    @torch.no_grad()
    def _encode_input_image_latents_tensor(
        self,
        input_image: torch.Tensor,
        tiled=False,
        tile_size=(30, 52),
        tile_stride=(15, 26),
    ):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if (
            input_image.ndim != 4
            or input_image.shape[0] != 1
            or input_image.shape[1] != 3
        ):
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        z = self.vae.encode(
            [image],
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    def _attach_teacher_to_inputs(self, sample, inputs, device, dtype, enable):
        if not enable:
            return inputs
        teacher_frames = sample.get("teacher_frames", None)
        if teacher_frames is not None:
            if getattr(self, "teacher_models", None) is None:
                raise RuntimeError(
                    "online_teacher: sample has teacher_frames but self.teacher_models is None — the trainer must inject frozen CoTracker+DINO (load_teacher_models) before training_loss."
                )
            from ...datasets.lerobot.teacher_cache import (
                teacher_dino_size,
                teacher_geometry,
            )
            from ...datasets.lerobot.teacher_extract import extract_teacher_batch

            _geom_name = (
                getattr(self, "dynamic_branch_teacher_geometry", None)
                or self.dynamic_branch_camera_layout
            )
            video_size, traj_grid_hw, dino_grid_hw = teacher_geometry(_geom_name)
            tr, vi, di = extract_teacher_batch(
                teacher_frames,
                self.teacher_models,
                traj_grid_hw,
                video_size,
                dino_grid_hw,
                device=device,
                dino_size=teacher_dino_size(_geom_name),
            )
            _branch = getattr(getattr(self, "mot", None), "dynamic_branch", None)
            _h = (
                int(_branch.traj_head.horizon)
                if _branch is not None
                else int(tr.shape[2])
            )
            if int(tr.shape[2]) < _h:
                raise ValueError(
                    f"branch head horizon={_h} exceeds the window's teacher P_max={int(tr.shape[2])} future latent frames — the window is too short for this head. Check that num_frames and dynamic_branch_horizon reconcile."
                )
            if int(tr.shape[2]) != _h:
                tr, vi, di = (tr[:, :, -_h:], vi[:, :, -_h:], di[:, :, -_h:])
            inputs["teacher_tracks"] = tr.to(
                device=device, dtype=dtype, non_blocking=True
            )
            inputs["teacher_dino"] = di.to(
                device=device, dtype=dtype, non_blocking=True
            )
            inputs["teacher_track_vis"] = vi.to(
                device=device, dtype=torch.bool, non_blocking=True
            )
            return inputs
        inputs["teacher_tracks"] = sample["teacher_tracks"].to(
            device=device, dtype=dtype, non_blocking=True
        )
        inputs["teacher_dino"] = sample["teacher_dino"].to(
            device=device, dtype=dtype, non_blocking=True
        )
        inputs["teacher_track_vis"] = sample["teacher_track_vis"].to(
            device=device, dtype=torch.bool, non_blocking=True
        )
        return inputs

    def build_inputs(self, sample, tiled: bool = False):
        if self.num_extra_ref_frames != 0:
            raise ValueError("MT-WAM supports num_extra_ref_frames=0 only.")
        video = sample["video"]
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError(
                "MTWAM training requires `sample['context']` and `sample['context_mask']`."
            )
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio", None)
        if video.ndim != 5:
            raise ValueError(
                f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}"
            )
        if video.shape[1] != 3:
            raise ValueError(
                f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}"
            )
        batch_size, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
            )
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(
                f"Video T must be > 1 for action-conditioned training, got T={num_frames}"
            )
        if "action" not in sample:
            raise ValueError("`sample['action']` is required for MTWAM training.")
        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(
                f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}"
            )
        action_horizon = int(action.shape[1])
        if action_horizon % (num_frames - 1) != 0:
            raise ValueError(
                f"`sample['action']` temporal dimension must be divisible by video transitions ({num_frames - 1}), got {action_horizon}"
            )
        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}"
                )
            if (
                action_is_pad.shape[0] != batch_size
                or action_is_pad.shape[1] != action_horizon
            ):
                raise ValueError(
                    f"`sample['action_is_pad']` shape mismatch: got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action_horizon})"
                )
        k_raw = self.num_extra_ref_frames
        num_ref_latents = 1
        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            if image_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
                )
            if (
                image_is_pad.shape[0] != batch_size
                or image_is_pad.shape[1] != num_frames
            ):
                raise ValueError(
                    f"`sample['image_is_pad']` shape mismatch: got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {num_frames})"
                )
        cached_latent = sample.get("cached_latent")
        if cached_latent is not None:
            input_latents = cached_latent.to(
                device=self.device, dtype=self.torch_dtype, non_blocking=True
            )
        else:
            input_video = video.to(
                device=self.device, dtype=self.torch_dtype, non_blocking=True
            )
            with record_function("mtwam/vae_encode"):
                input_latents = self._encode_video_latents(input_video, tiled=tiled)
        reference_latents = None
        fuse_flag = False
        if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
            if num_ref_latents >= input_latents.shape[2]:
                raise ValueError(
                    f"num_ref_latents={num_ref_latents} leaves no predicted latent frame (T_lat={input_latents.shape[2]}); raise num_frames or lower num_extra_ref_frames"
                )
            reference_latents = input_latents[:, :, 0:num_ref_latents]
            fuse_flag = True
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        context_mask = context_mask.to(
            device=self.device, dtype=torch.bool, non_blocking=True
        )
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError(
                    "`sample['proprio']` is required when `proprio_dim` is enabled."
                )
            if proprio.ndim != 3:
                raise ValueError(
                    f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}"
                )
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
                )
            if k_raw >= proprio.shape[1]:
                raise ValueError(
                    f"`sample['proprio']` window ({proprio.shape[1]}) too short for num_extra_ref_frames={k_raw} (needs the CURRENT-frame index K)"
                )
            proprio = proprio[:, k_raw, :]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )
        action = action.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )
        out = {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "reference_latents": reference_latents,
            "num_ref_latents": num_ref_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
        }
        with record_function("mtwam/online_teacher"):
            self._attach_teacher_to_inputs(
                sample,
                out,
                self.device,
                self.torch_dtype,
                enable=getattr(self.mot, "dynamic_branch", None) is not None
                and (self.loss_lambda_traj > 0 or self.loss_lambda_tex > 0),
            )
        return out

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
        traj_seq_len: int = 0,
        num_ref_latents: int = 1,
        dual_f0: bool = False,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len + traj_seq_len
        mask = torch.zeros(
            (total_seq_len, total_seq_len), dtype=torch.bool, device=device
        )
        a0 = video_seq_len
        a1 = video_seq_len + action_seq_len
        mask[:video_seq_len, :video_seq_len] = (
            self.video_expert.build_video_to_video_mask(
                video_seq_len=video_seq_len,
                video_tokens_per_frame=video_tokens_per_frame,
                device=device,
                num_ref_latents=num_ref_latents,
                **{"dual_f0_branch_slot": True} if dual_f0 else {},
            )
        )
        mask[a0:a1, a0:a1] = True
        ref_tokens = min(num_ref_latents * video_tokens_per_frame, video_seq_len)
        ff = video_tokens_per_frame
        if dual_f0:
            if num_ref_latents != 2:
                raise ValueError(
                    f"dual_f0 requires num_ref_latents == 2 (slot 0 = branch copy, slot 1 = trunk anchor), got {num_ref_latents}"
                )
            mask[a0:a1, ff : 2 * ff] = True
            branch_ref_spans = ((0, ff), (0, ff))
        else:
            mask[a0:a1, :ref_tokens] = True
            branch_ref_spans = ((0, ref_tokens), (0, ref_tokens))
        branch_ref_tokens = branch_fork_span(
            ff, num_ref_latents, dual_f0, video_seq_len=video_seq_len
        )
        if traj_seq_len > 0:
            if traj_seq_len != 2 * branch_ref_tokens:
                raise ValueError(
                    f"branch length mismatch: traj_seq_len={traj_seq_len} must equal 2*branch_ref_tokens={2 * branch_ref_tokens} (two copies of the branch reference span)"
                )
            st_f = traj_seq_len // 2
            t0, t1 = (a1, a1 + st_f)
            s0, s1 = (t1, a1 + traj_seq_len)
            mask[a0:a1, t0:t1] = True
            for (lo, hi), (rlo, rhi) in zip(((t0, t1), (s0, s1)), branch_ref_spans):
                mask[lo:hi, rlo:rhi] = True
                mask[lo:hi, lo:hi] = True
        return mask

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_video_step: bool,
        num_ref_latents: int = 1,
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(
            pred_video.float(), target_video.float(), reduction="none"
        ).mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)
        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(
                f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}."
            )
        if image_is_pad.shape[1] < 1:
            raise ValueError("`image_is_pad` must contain at least one frame.")
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                f"Cannot align `image_is_pad` with video latent steps: num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )
        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(
            image_is_pad.shape[0], -1, temporal_factor
        ).all(dim=2)
        if include_initial_video_step:
            video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        else:
            video_is_pad = latent_tail_is_pad[:, num_ref_latents - 1 :]
        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                f"Video-loss mask shape mismatch: mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )
        valid = (~video_is_pad).to(
            device=video_loss_token.device, dtype=video_loss_token.dtype
        )
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]
        dual_f0 = bool(getattr(self, "dynamic_branch_dual_f0", False))
        n_f0_copies = 1 if dual_f0 else 0
        num_ref_latents_data = int(inputs["num_ref_latents"])
        if n_f0_copies:
            _knob = "dynamic_branch_dual_f0"
            if inputs["reference_latents"] is None:
                raise ValueError(
                    f"{_knob}=True requires pinned reference latents (inputs['reference_latents'] is None)."
                )
            if num_ref_latents_data != 1:
                raise ValueError(
                    f"{_knob}=True supports K=0 only (num_ref_latents_data=1), got {num_ref_latents_data} — f0-slot expansion x history conditioning is rejected."
                )
            input_latents = torch.cat(
                [input_latents[:, :, 0:1]] * n_f0_copies + [input_latents], dim=2
            )
        num_ref_latents = num_ref_latents_data + n_f0_copies
        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size, device=self.device, dtype=input_latents.dtype
        )
        latents = self.train_video_scheduler.add_noise(
            input_latents, noise_video, timestep_video
        )
        target_video = self.train_video_scheduler.training_target(
            input_latents, noise_video, timestep_video
        )
        if inputs["reference_latents"] is not None:
            reference_latents = inputs["reference_latents"]
            if n_f0_copies:
                reference_latents = torch.cat(
                    [reference_latents[:, :, 0:1]] * n_f0_copies + [reference_latents],
                    dim=2,
                )
            latents[:, :, 0:num_ref_latents] = reference_latents
        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size, device=self.device, dtype=action.dtype
        )
        noisy_action = self.train_action_scheduler.add_noise(
            action, noise_action, timestep_action
        )
        target_action = self.train_action_scheduler.training_target(
            action, noise_action, timestep_action
        )
        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
            num_ref_latents=num_ref_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]
        branch = getattr(self.mot, "dynamic_branch", None)
        branch_on = branch is not None and (
            self.loss_lambda_traj > 0 or self.loss_lambda_tex > 0
        )
        ff = int(video_pre["meta"]["tokens_per_frame"])
        traj_seq_len = 0
        if branch_on:
            traj_seq_len = 2 * branch_fork_span(ff, num_ref_latents, dual_f0)
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=ff,
            device=video_tokens.device,
            traj_seq_len=traj_seq_len,
            num_ref_latents=num_ref_latents,
            dual_f0=dual_f0,
        )
        with record_function("mtwam/mot_forward"):
            tokens_out = self.mot(
                embeds_all={"video": video_tokens, "action": action_tokens},
                attention_mask=attention_mask,
                freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
                context_all={
                    "video": {
                        "context": video_pre["context"],
                        "mask": video_pre["context_mask"],
                    },
                    "action": {
                        "context": action_pre["context"],
                        "mask": action_pre["context_mask"],
                    },
                },
                t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
                dynamic_branch_payload={
                    "enable": True,
                    "first_frame_tokens": ff,
                    "num_ref_latents": num_ref_latents,
                    "dual_f0": dual_f0,
                }
                if branch_on
                else None,
            )
        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        include_initial_video_step = inputs["reference_latents"] is None
        if inputs["reference_latents"] is not None:
            pred_video = pred_video[:, :, num_ref_latents:]
            target_video = target_video[:, :, num_ref_latents:]
        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
            num_ref_latents=num_ref_latents_data,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()
        action_loss_token = F.mse_loss(
            pred_action.float(), target_action.float(), reduction="none"
        ).mean(dim=2)
        if action_is_pad is not None:
            valid = (~action_is_pad).to(
                device=action_loss_token.device, dtype=action_loss_token.dtype
            )
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()
        loss_total = (
            self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        )
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        if branch_on:
            missing_teacher = [
                k
                for k in ("teacher_tracks", "teacher_track_vis", "teacher_dino")
                if k not in inputs
            ]
            if missing_teacher:
                raise ValueError(
                    f"dynamic branch loss enabled (lambda_traj/tex > 0) but teacher targets {missing_teacher} absent in inputs. Enable online_teacher or prepare the configured teacher feature cache."
                )
            bh = self.mot._last_branch_hidden
            n_branch, branch_dev = (bh["traj"].shape[0], bh["traj"].device)
            horizon = int(branch.traj_head.horizon)
            frame_pad = derive_branch_frame_pad(
                image_is_pad,
                n_branch,
                horizon,
                self.vae.temporal_downsample_factor,
                branch_dev,
                num_ref_latents=num_ref_latents_data,
            )
            teacher = {
                "tracks": inputs["teacher_tracks"],
                "track_vis": inputs["teacher_track_vis"],
                "dino": inputs["teacher_dino"],
                "frame_pad": frame_pad,
            }
            _, grid_h, grid_w = video_pre["meta"]["grid_size"]
            branch_total, branch_dict = assemble_branch_loss(
                branch,
                self.mot._last_branch_hidden,
                teacher,
                lambda_traj=self.loss_lambda_traj,
                lambda_tex=self.loss_lambda_tex,
                validity_mask=self.loss_traj_validity_mask,
                grid_hw=(int(grid_h), int(grid_w)),
                camera_layout=self.dynamic_branch_camera_layout,
                num_ref_latents=num_ref_latents_data,
            )
            loss_total = loss_total + branch_total
            loss_dict.update(branch_dict)
        return (loss_total, loss_dict)

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        prefix_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        prefix_seq_len: int,
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            prefix_kv_cache=prefix_kv_cache,
            attention_mask=attention_mask,
            prefix_seq_len=prefix_seq_len,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        if self.num_extra_ref_frames != 0:
            raise ValueError("MT-WAM supports num_extra_ref_frames=0 only.")
        if action_horizon is None:
            raise ValueError(
                "infer_action requires action_horizon matching the training window (num_frames - 1)."
            )
        if self.num_frames is not None:
            expected_chunk = self.num_frames - 1 - self.num_extra_ref_frames
            if action_horizon != expected_chunk:
                raise ValueError(
                    f"action_horizon={action_horizon} != training chunk {expected_chunk} (num_frames={self.num_frames} - 1 - K={self.num_extra_ref_frames}). Derive action_horizon from the model's training window, not a hardcoded value."
                )
        self.eval()
        if (
            str(getattr(self.video_expert, "video_attention_mask_mode", ""))
            != "first_frame_causal"
        ):
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )
        k_raw = self.num_extra_ref_frames
        if input_image is None:
            raise ValueError("`input_image` is required when num_extra_ref_frames=0.")
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if (
            input_image.ndim != 4
            or input_image.shape[0] != 1
            or input_image.shape[1] != 3
        ):
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"input frames must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError(
                    "`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled."
                )
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(
                    f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}"
                )
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(
                    f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
                )
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)
        generator = (
            None
            if seed is None
            else torch.Generator(device=rand_device).manual_seed(seed)
        )
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        with record_function("mtwam/infer_vae_encode"):
            input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
            reference_latents = self._encode_input_image_latents_tensor(
                input_image=input_image, tiled=tiled
            )
            num_ref_latents = 1
        dual_f0 = bool(getattr(self, "dynamic_branch_dual_f0", False))
        n_f0_copies = 1 if dual_f0 else 0
        if n_f0_copies:
            _knob = "dynamic_branch_dual_f0"
            if num_ref_latents != 1:
                raise ValueError(
                    f"{_knob}=True supports K=0 only (one reference latent), got num_ref_latents={num_ref_latents}."
                )
            reference_latents = torch.cat(
                [reference_latents[:, :, 0:1]] * n_f0_copies + [reference_latents],
                dim=2,
            )
            num_ref_latents = 1 + n_f0_copies
        fuse_flag = bool(
            getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)
        )
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError(
                "`prompt` and `context/context_mask` are mutually exclusive."
            )
        if not use_prompt and (not use_context):
            raise ValueError(
                "Either `prompt` or both `context/context_mask` must be provided."
            )
        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError(
                    "`context` and `context_mask` must be both provided together."
                )
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(
                device=self.device, dtype=self.torch_dtype, non_blocking=True
            )
            context_mask = context_mask.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context, context_mask=context_mask, proprio=proprio
            )
        timestep_video = torch.zeros(
            (reference_latents.shape[0],),
            dtype=reference_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=reference_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
            num_ref_latents=num_ref_latents,
        )
        prefix_seq_len = int(video_pre["tokens"].shape[1])
        ff = int(video_pre["meta"]["tokens_per_frame"])
        branch_on = getattr(self.mot, "dynamic_branch", None) is not None
        traj_seq_len = 0
        if branch_on:
            traj_seq_len = 2 * branch_fork_span(ff, num_ref_latents, dual_f0)
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=prefix_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=ff,
            device=video_pre["tokens"].device,
            traj_seq_len=traj_seq_len,
            num_ref_latents=num_ref_latents,
            dual_f0=dual_f0,
        )
        if branch_on:
            sa, st = (int(latents_action.shape[1]), traj_seq_len)
            non_action = list(range(prefix_seq_len)) + list(
                range(prefix_seq_len + sa, prefix_seq_len + sa + st)
            )
            pidx = torch.tensor(non_action, device=attention_mask.device)
            prefix_attention_mask = attention_mask[pidx][:, pidx]
            branch_payload = {
                "enable": True,
                "first_frame_tokens": ff,
                "num_ref_latents": num_ref_latents,
                "dual_f0": dual_f0,
            }
        else:
            prefix_attention_mask = attention_mask[:prefix_seq_len, :prefix_seq_len]
            branch_payload = None
        with record_function("mtwam/infer_prefill"):
            prefill_kwargs = {
                "prefix_tokens": video_pre["tokens"],
                "prefix_freqs": video_pre["freqs"],
                "prefix_t_mod": video_pre["t_mod"],
                "prefix_context_payload": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "prefix_attention_mask": prefix_attention_mask,
                "dynamic_branch_payload": branch_payload,
            }
            prefill_result = self.mot.prefill_video_cache(**prefill_kwargs)
        prefix_kv_cache = prefill_result
        infer_timesteps_action, infer_deltas_action = (
            self.infer_action_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=self.device,
                dtype=latents_action.dtype,
                shift_override=sigma_shift,
            )
        )
        with record_function("mtwam/infer_denoise"):
            for step_t_action, step_delta_action in zip(
                infer_timesteps_action, infer_deltas_action
            ):
                with record_function("mtwam/infer_denoise_step"):
                    timestep_action = step_t_action.unsqueeze(0).to(
                        dtype=latents_action.dtype, device=self.device
                    )
                    pred_action_posi = self._predict_action_noise_with_cache(
                        latents_action=latents_action,
                        timestep_action=timestep_action,
                        context=context,
                        context_mask=context_mask,
                        prefix_kv_cache=prefix_kv_cache,
                        attention_mask=attention_mask,
                        prefix_seq_len=prefix_seq_len,
                    )
                    pred_action = pred_action_posi
                    latents_action = self.infer_action_scheduler.step(
                        pred_action, step_delta_action, latents_action
                    )
        result = {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        }
        return result

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or "mot" not in payload:
            raise ValueError(
                f"Checkpoint must contain a complete `mot` state dictionary: {path}"
            )

        def validate(name, expected, supplied):
            if not isinstance(supplied, dict):
                raise ValueError(f"Checkpoint `{name}` must be a state dictionary.")
            missing = sorted(set(expected) - set(supplied))
            unexpected = sorted(set(supplied) - set(expected))
            if missing or unexpected:
                raise ValueError(
                    f"Checkpoint `{name}` keys do not match the model: missing={missing[:8]}, unexpected={unexpected[:8]}"
                )
            for key, tensor in supplied.items():
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tensor.shape != expected[key].shape
                ):
                    shape = getattr(tensor, "shape", None)
                    raise ValueError(
                        f"Checkpoint `{name}.{key}` has shape {shape}; expected {tuple(expected[key].shape)}."
                    )

        validate("mot", self.mot.state_dict(), payload["mot"])
        if self.proprio_encoder is not None:
            if "proprio_encoder" not in payload:
                raise ValueError(
                    "Checkpoint is missing required `proprio_encoder` weights."
                )
            validate(
                "proprio_encoder",
                self.proprio_encoder.state_dict(),
                payload["proprio_encoder"],
            )
        elif "proprio_encoder" in payload:
            raise ValueError(
                "Checkpoint carries `proprio_encoder` weights but the model has proprio_dim=None."
            )
        self.mot.load_state_dict(payload["mot"], strict=True)
        if self.proprio_encoder is not None:
            self.proprio_encoder.load_state_dict(
                payload["proprio_encoder"], strict=True
            )
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
