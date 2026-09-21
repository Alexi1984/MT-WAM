from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn as nn
from ._load_guards import branch_fork_span
from .wan_video_dit import flash_attention, modulate, rope_apply
from mtwam.utils.logging_config import get_logger

logger = get_logger(__name__)


class MoT(nn.Module):
    def __init__(
        self, mixtures: Dict[str, nn.Module], mot_checkpoint_mixed_attn: bool = True
    ):
        super().__init__()
        if not mixtures:
            raise ValueError("`mixtures` cannot be empty.")
        if "video" not in mixtures or "action" not in mixtures:
            raise ValueError(
                "`mixtures` must include both 'video' and 'action' experts."
            )
        self.mixtures = nn.ModuleDict(mixtures)
        self.expert_order = list(self.mixtures.keys())
        self.mot_checkpoint_mixed_attn = mot_checkpoint_mixed_attn
        if mot_checkpoint_mixed_attn:
            logger.info(
                "Using gradient checkpointing for mixture attention. This will save memory but use more computation."
            )
        first_expert = self.mixtures[self.expert_order[0]]
        self.num_layers = len(first_expert.blocks)
        self.num_heads = first_expert.num_heads
        self.attn_head_dim = first_expert.attn_head_dim
        for name in self.expert_order[1:]:
            expert = self.mixtures[name]
            if len(expert.blocks) != self.num_layers:
                raise ValueError(
                    f"All experts must have same number of layers; got {self.num_layers} and {len(expert.blocks)}"
                )
            if expert.num_heads != self.num_heads:
                raise ValueError(
                    f"All experts must have same num_heads; got {self.num_heads} and {expert.num_heads}"
                )
            if expert.attn_head_dim != self.attn_head_dim:
                raise ValueError(
                    f"All experts must have same attn_head_dim; got {self.attn_head_dim} and {expert.attn_head_dim}"
                )
        logger.info(
            f"Initialized MoT with experts: {self.expert_order}, num_layers={self.num_layers}"
        )
        for name in self.expert_order:
            expert = self.mixtures[name]
            logger.info(
                f"  Expert '{name}': num_params={sum((p.numel() for p in expert.parameters())) / 1000000000.0:.2f} B"
            )

    @staticmethod
    def _split_modulation(block, t_mod: torch.Tensor):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        base_mod = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            base_mod + t_mod
        ).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        return (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)

    def _mixed_attention(
        self,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_mask = attention_mask.to(device=q_cat.device)

        def _forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return flash_attention(
                q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=attn_mask
            )

        if self.mot_checkpoint_mixed_attn and self.training:
            return torch.utils.checkpoint.checkpoint(
                _forward, q_cat, k_cat, v_cat, use_reentrant=False
            )
        return _forward(q_cat, k_cat, v_cat)

    @staticmethod
    def _branch_context_payload(payload, st):
        if payload is None:
            return None
        mask = payload.get("mask")
        if mask is not None and mask.dim() == 3 and (mask.shape[1] != st):
            mask = mask[:, :1, :].expand(-1, st, -1)
        return {"context": payload.get("context"), "mask": mask}

    @staticmethod
    def _apply_expert_post_block(
        block,
        residual_x: torch.Tensor,
        mixed_attn_out: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        x = block.gate(residual_x, gate_msa, block.self_attn.o(mixed_attn_out))
        if context_payload is not None:
            context = context_payload.get("context")
            if context is not None:
                context_mask = context_payload.get("mask")
                if context_mask is not None and context_mask.dim() == 3:
                    context_mask = context_mask.unsqueeze(1)
                x = x + block.cross_attn(block.norm3(x), context, ctx_mask=context_mask)
        mlp_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
        x = block.gate(x, gate_mlp, block.ffn(mlp_input))
        return x

    def _build_expert_attention_io(
        self, expert, block, x: torch.Tensor, freqs: torch.Tensor, t_mod: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        bool,
    ]:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self._split_modulation(block, t_mod)
        )
        attn_input = modulate(block.norm1(x), shift_msa, scale_msa)
        q = block.self_attn.norm_q(block.self_attn.q(attn_input))
        k = block.self_attn.norm_k(block.self_attn.k(attn_input))
        v = block.self_attn.v(attn_input)
        q = rope_apply(q, freqs, block.num_heads)
        k = rope_apply(k, freqs, block.num_heads)
        use_gradient_checkpointing = bool(
            getattr(expert, "use_gradient_checkpointing", False)
        )
        return (
            q,
            k,
            v,
            x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            use_gradient_checkpointing,
        )

    def _apply_post_with_optional_checkpoint(
        self,
        block,
        residual_x: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        use_gradient_checkpointing: bool,
        mixed_slice: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        def _post_fn(
            _mixed_slice: torch.Tensor,
            _x: torch.Tensor,
            _gate_msa: torch.Tensor,
            _shift_mlp: torch.Tensor,
            _scale_mlp: torch.Tensor,
            _gate_mlp: torch.Tensor,
            _block=block,
            _context_payload=context_payload,
        ) -> torch.Tensor:
            return self._apply_expert_post_block(
                block=_block,
                residual_x=_x,
                mixed_attn_out=_mixed_slice,
                gate_msa=_gate_msa,
                shift_mlp=_shift_mlp,
                scale_mlp=_scale_mlp,
                gate_mlp=_gate_mlp,
                context_payload=_context_payload,
            )

        if use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                _post_fn,
                mixed_slice,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_reentrant=False,
            )
        return _post_fn(
            mixed_slice, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp
        )

    def prefill_video_cache(
        self,
        prefix_tokens: torch.Tensor,
        prefix_freqs: torch.Tensor,
        prefix_t_mod: torch.Tensor,
        prefix_context_payload: Optional[dict],
        prefix_attention_mask: torch.Tensor,
        dynamic_branch_payload: Optional[dict] = None,
    ) -> list[dict[str, torch.Tensor]]:
        if dynamic_branch_payload and dynamic_branch_payload.get("triple_f0", False):
            raise ValueError("MT-WAM does not support a triple-f0 branch payload.")
        if "video" not in self.mixtures:
            raise ValueError("MoT requires `video` expert for `prefill_video_cache`.")
        if prefix_attention_mask.ndim != 2:
            raise ValueError(
                f"`prefix_attention_mask` must be 2D [S,S], got shape {tuple(prefix_attention_mask.shape)}"
            )
        if prefix_attention_mask.shape[0] != prefix_attention_mask.shape[1]:
            raise ValueError(
                f"`prefix_attention_mask` must be square, got shape {tuple(prefix_attention_mask.shape)}"
            )
        sv = int(prefix_tokens.shape[1])
        branch_on = (
            dynamic_branch_payload is not None
            and dynamic_branch_payload.get("enable", False)
            and (getattr(self, "dynamic_branch", None) is not None)
        )
        if branch_on:
            fork_depth = self.num_layers - self.dynamic_branch.num_layers
            ff = int(dynamic_branch_payload["first_frame_tokens"])
            dual_f0 = bool(dynamic_branch_payload.get("dual_f0", False))
            fork_span = branch_fork_span(
                ff, dynamic_branch_payload.get("num_ref_latents", 1), dual_f0
            )
            branch_freqs = torch.cat(
                [prefix_freqs[:fork_span], prefix_freqs[:fork_span]], dim=0
            )
            branch_tmod = torch.cat(
                [prefix_t_mod[:, :fork_span], prefix_t_mod[:, :fork_span]], dim=1
            )
            expected_prefix = sv + 2 * fork_span
            x_branch = None
        else:
            expected_prefix = sv
        if prefix_attention_mask.shape[0] != expected_prefix:
            raise ValueError(
                f"`prefix_attention_mask` seq length mismatch: mask={prefix_attention_mask.shape[0]} vs expected_prefix={expected_prefix}"
            )
        expert = self.mixtures["video"]
        x = prefix_tokens
        kv_cache: list[dict[str, torch.Tensor]] = []
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            (
                q,
                k,
                v,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert, block=block, x=x, freqs=prefix_freqs, t_mod=prefix_t_mod
            )
            is_tail = branch_on and layer_idx >= fork_depth
            if is_tail:
                if layer_idx == fork_depth:
                    ref = x[:, :fork_span, :]
                    x_branch = torch.cat(
                        [
                            ref + self.dynamic_branch.role_t,
                            ref + self.dynamic_branch.role_s,
                        ],
                        dim=1,
                    )
                branch_block = self.dynamic_branch.blocks[layer_idx - fork_depth]
                (
                    bq,
                    bk,
                    bv,
                    b_residual,
                    b_gate_msa,
                    b_shift_mlp,
                    b_scale_mlp,
                    b_gate_mlp,
                    b_use_gc,
                ) = self._build_expert_attention_io(
                    expert=self.dynamic_branch,
                    block=branch_block,
                    x=x_branch,
                    freqs=branch_freqs,
                    t_mod=branch_tmod,
                )
                k_cat = torch.cat([k, bk], dim=1)
                v_cat = torch.cat([v, bv], dim=1)
                mixed = self._mixed_attention(
                    q_cat=torch.cat([q, bq], dim=1),
                    k_cat=k_cat,
                    v_cat=v_cat,
                    attention_mask=prefix_attention_mask,
                )
                x = self._apply_post_with_optional_checkpoint(
                    block=block,
                    residual_x=residual_x,
                    gate_msa=gate_msa,
                    shift_mlp=shift_mlp,
                    scale_mlp=scale_mlp,
                    gate_mlp=gate_mlp,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                    mixed_slice=mixed[:, :sv, :],
                    context_payload=prefix_context_payload,
                )
                x_branch = self._apply_post_with_optional_checkpoint(
                    block=branch_block,
                    residual_x=b_residual,
                    gate_msa=b_gate_msa,
                    shift_mlp=b_shift_mlp,
                    scale_mlp=b_scale_mlp,
                    gate_mlp=b_gate_mlp,
                    use_gradient_checkpointing=b_use_gc,
                    mixed_slice=mixed[:, sv:, :],
                    context_payload=self._branch_context_payload(
                        prefix_context_payload, 2 * fork_span
                    )
                    if getattr(self.dynamic_branch, "read_text", False)
                    else None,
                )
                kv_cache.append({"k": k_cat, "v": v_cat})
            else:
                mixed = self._mixed_attention(
                    q_cat=q,
                    k_cat=k,
                    v_cat=v,
                    attention_mask=prefix_attention_mask[:sv, :sv],
                )
                x = self._apply_post_with_optional_checkpoint(
                    block=block,
                    residual_x=residual_x,
                    gate_msa=gate_msa,
                    shift_mlp=shift_mlp,
                    scale_mlp=scale_mlp,
                    gate_mlp=gate_mlp,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                    mixed_slice=mixed,
                    context_payload=prefix_context_payload,
                )
                kv_cache.append({"k": k, "v": v})
        return kv_cache

    def forward_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        prefix_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        prefix_seq_len: int,
    ) -> torch.Tensor:
        if "action" not in self.mixtures:
            raise ValueError(
                "MoT requires `action` expert for `forward_action_with_video_cache`."
            )
        if len(prefix_kv_cache) != self.num_layers:
            raise ValueError(
                f"`prefix_kv_cache` must contain {self.num_layers} layers, got {len(prefix_kv_cache)}."
            )
        if attention_mask.ndim != 2:
            raise ValueError(
                f"`attention_mask` must be 2D [S,S], got shape {tuple(attention_mask.shape)}"
            )
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(
                f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}"
            )
        action_seq_len = int(action_tokens.shape[1])
        sv = int(prefix_seq_len)
        joint_total = int(attention_mask.shape[0])
        st = joint_total - sv - action_seq_len
        if st < 0:
            raise ValueError(
                f"`attention_mask` too small for prefix+action: mask={joint_total} vs prefix_seq_len(Sv)={sv} + action={action_seq_len}"
            )
        action_rows = attention_mask[sv : sv + action_seq_len, :]
        expert = self.mixtures["action"]
        x = action_tokens
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            (
                q_action,
                k_action,
                v_action,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert, block=block, x=x, freqs=action_freqs, t_mod=action_t_mod
            )
            layer_cache = prefix_kv_cache[layer_idx]
            if "k" not in layer_cache or "v" not in layer_cache:
                raise ValueError(
                    f"`prefix_kv_cache[{layer_idx}]` must contain `k` and `v`."
                )
            k_prefix = layer_cache["k"]
            v_prefix = layer_cache["v"]
            traj_len = int(k_prefix.shape[1]) - sv
            if traj_len < 0 or traj_len > st or v_prefix.shape[1] != k_prefix.shape[1]:
                raise ValueError(
                    f"`prefix_kv_cache[{layer_idx}]` len {tuple(k_prefix.shape)} inconsistent with Sv={sv}, St={st}."
                )
            k_video, k_traj = (k_prefix[:, :sv], k_prefix[:, sv:])
            v_video, v_traj = (v_prefix[:, :sv], v_prefix[:, sv:])
            k_cat = torch.cat([k_video, k_action, k_traj], dim=1)
            v_cat = torch.cat([v_video, v_action, v_traj], dim=1)
            layer_action_mask = action_rows[:, : sv + action_seq_len + traj_len]
            mixed = self._mixed_attention(
                q_cat=q_action,
                k_cat=k_cat,
                v_cat=v_cat,
                attention_mask=layer_action_mask,
            )
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=action_context_payload,
            )
        return x

    def forward(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
        dynamic_branch_payload: Optional[dict] = None,
    ):
        if dynamic_branch_payload and dynamic_branch_payload.get("triple_f0", False):
            raise ValueError("MT-WAM does not support a triple-f0 branch payload.")
        missing = [k for k in self.expert_order if k not in embeds_all]
        if missing:
            raise ValueError(f"Missing expert tokens for {missing}")
        missing = [k for k in self.expert_order if k not in freqs_all]
        if missing:
            raise ValueError(f"Missing expert freqs for {missing}")
        missing = [k for k in self.expert_order if k not in t_mod_all]
        if missing:
            raise ValueError(f"Missing expert t_mod for {missing}")
        if attention_mask.ndim != 2:
            raise ValueError(
                f"`attention_mask` must be 2D [S, S], got shape {tuple(attention_mask.shape)}"
            )
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(
                f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}"
            )
        branch_on = (
            dynamic_branch_payload is not None
            and dynamic_branch_payload.get("enable", False)
            and (getattr(self, "dynamic_branch", None) is not None)
        )
        if branch_on:
            fork_depth = self.num_layers - self.dynamic_branch.num_layers
            ff = int(dynamic_branch_payload["first_frame_tokens"])
            dual_f0 = bool(dynamic_branch_payload.get("dual_f0", False))
            fork_span = branch_fork_span(
                ff, dynamic_branch_payload.get("num_ref_latents", 1), dual_f0
            )
            v_freqs = freqs_all["video"]
            v_tmod = t_mod_all["video"]
            branch_freqs = torch.cat([v_freqs[:fork_span], v_freqs[:fork_span]], dim=0)
            branch_tmod = torch.cat(
                [v_tmod[:, :fork_span], v_tmod[:, :fork_span]], dim=1
            )
            x_branch = None
            self._last_branch_hidden = None
        tokens_all = {k: v for k, v in embeds_all.items()}
        for layer_idx in range(self.num_layers):
            is_tail = branch_on and layer_idx >= fork_depth
            q_chunks = []
            k_chunks = []
            v_chunks = []
            cached = {}
            seq_lens = []
            for name in self.expert_order:
                expert = self.mixtures[name]
                block = expert.blocks[layer_idx]
                x = tokens_all[name]
                freqs = freqs_all[name]
                t_mod = t_mod_all[name]
                (
                    q,
                    k,
                    v,
                    residual_x,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                    use_gradient_checkpointing,
                ) = self._build_expert_attention_io(
                    expert=expert, block=block, x=x, freqs=freqs, t_mod=t_mod
                )
                q_chunks.append(q)
                k_chunks.append(k)
                v_chunks.append(v)
                seq_lens.append(x.shape[1])
                cached[name] = {
                    "block": block,
                    "residual_x": residual_x,
                    "gate_msa": gate_msa,
                    "shift_mlp": shift_mlp,
                    "scale_mlp": scale_mlp,
                    "gate_mlp": gate_mlp,
                    "use_gradient_checkpointing": use_gradient_checkpointing,
                }
            base_seq = sum(seq_lens)
            branch_cache = None
            if is_tail:
                if layer_idx == fork_depth:
                    ref = tokens_all["video"][:, :fork_span, :]
                    x_branch = torch.cat(
                        [
                            ref + self.dynamic_branch.role_t,
                            ref + self.dynamic_branch.role_s,
                        ],
                        dim=1,
                    )
                branch_block = self.dynamic_branch.blocks[layer_idx - fork_depth]
                (
                    bq,
                    bk,
                    bv,
                    b_residual,
                    b_gate_msa,
                    b_shift_mlp,
                    b_scale_mlp,
                    b_gate_mlp,
                    b_use_gc,
                ) = self._build_expert_attention_io(
                    expert=self.dynamic_branch,
                    block=branch_block,
                    x=x_branch,
                    freqs=branch_freqs,
                    t_mod=branch_tmod,
                )
                q_chunks.append(bq)
                k_chunks.append(bk)
                v_chunks.append(bv)
                branch_cache = {
                    "block": branch_block,
                    "residual_x": b_residual,
                    "gate_msa": b_gate_msa,
                    "shift_mlp": b_shift_mlp,
                    "scale_mlp": b_scale_mlp,
                    "gate_mlp": b_gate_mlp,
                    "use_gradient_checkpointing": b_use_gc,
                }
                active_mask = attention_mask
                active_seq = base_seq + x_branch.shape[1]
            else:
                active_mask = (
                    attention_mask[:base_seq, :base_seq]
                    if branch_on
                    else attention_mask
                )
                active_seq = base_seq
            q_cat = torch.cat(q_chunks, dim=1)
            k_cat = torch.cat(k_chunks, dim=1)
            v_cat = torch.cat(v_chunks, dim=1)
            if active_mask.shape[0] != active_seq:
                raise ValueError(
                    f"Attention mask seq length mismatch: mask={active_mask.shape[0]} vs tokens={active_seq}"
                )
            mixed = self._mixed_attention(
                q_cat=q_cat, k_cat=k_cat, v_cat=v_cat, attention_mask=active_mask
            )
            start = 0
            for name, seq_len in zip(self.expert_order, seq_lens):
                end = start + seq_len
                mixed_slice = mixed[:, start:end, :]
                cached_expert = cached[name]
                block = cached_expert["block"]
                context_payload = context_all.get(name)
                updated_tokens = self._apply_post_with_optional_checkpoint(
                    block=block,
                    residual_x=cached_expert["residual_x"],
                    gate_msa=cached_expert["gate_msa"],
                    shift_mlp=cached_expert["shift_mlp"],
                    scale_mlp=cached_expert["scale_mlp"],
                    gate_mlp=cached_expert["gate_mlp"],
                    use_gradient_checkpointing=cached_expert[
                        "use_gradient_checkpointing"
                    ],
                    mixed_slice=mixed_slice,
                    context_payload=context_payload,
                )
                tokens_all[name] = updated_tokens
                start = end
            if is_tail:
                end = start + x_branch.shape[1]
                x_branch = self._apply_post_with_optional_checkpoint(
                    block=branch_cache["block"],
                    residual_x=branch_cache["residual_x"],
                    gate_msa=branch_cache["gate_msa"],
                    shift_mlp=branch_cache["shift_mlp"],
                    scale_mlp=branch_cache["scale_mlp"],
                    gate_mlp=branch_cache["gate_mlp"],
                    use_gradient_checkpointing=branch_cache[
                        "use_gradient_checkpointing"
                    ],
                    mixed_slice=mixed[:, start:end, :],
                    context_payload=self._branch_context_payload(
                        context_all.get("video"), x_branch.shape[1]
                    )
                    if getattr(self.dynamic_branch, "read_text", False)
                    else None,
                )
                start = end
        if branch_on and x_branch is not None:
            half = x_branch.shape[1] // 2
            self._last_branch_hidden = {
                "traj": x_branch[:, :half],
                "tex": x_branch[:, half:],
            }
        return tokens_all
