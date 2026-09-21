import torch
import torch.nn as nn
from mtwam.models.wan22.wan_video_dit import WanVideoDiT
from mtwam.models.wan22.action_dit import ActionDiT
from mtwam.models.wan22.mot import MoT

_NUM_LAYERS = 4
_NUM_HEADS = 2
_ATTN_HEAD_DIM = 8
_HIDDEN = _NUM_HEADS * _ATTN_HEAD_DIM


def make_tiny_video_expert(num_layers=_NUM_LAYERS):
    return WanVideoDiT(
        hidden_dim=_HIDDEN,
        in_dim=48,
        ffn_dim=32,
        out_dim=48,
        text_dim=32,
        freq_dim=256,
        eps=1e-06,
        patch_size=(1, 2, 2),
        num_heads=_NUM_HEADS,
        attn_head_dim=_ATTN_HEAD_DIM,
        num_layers=num_layers,
        has_image_input=False,
        seperated_timestep=True,
    )


def make_tiny_action_expert(num_layers=_NUM_LAYERS):
    return ActionDiT(
        hidden_dim=_HIDDEN,
        action_dim=7,
        ffn_dim=32,
        text_dim=32,
        freq_dim=256,
        eps=1e-06,
        num_heads=_NUM_HEADS,
        attn_head_dim=_ATTN_HEAD_DIM,
        num_layers=num_layers,
    )


def make_tiny_mismatched_expert():
    return ActionDiT(
        hidden_dim=8,
        action_dim=7,
        ffn_dim=16,
        text_dim=32,
        freq_dim=256,
        eps=1e-06,
        num_heads=2,
        attn_head_dim=4,
        num_layers=_NUM_LAYERS,
    )


def make_tiny_mot(
    enable_dynamic_branch=False,
    dynamic_branch_num_layers=2,
    num_layers=_NUM_LAYERS,
    ffn_moe=False,
    ffn_moe_routing="role",
):
    mot = MoT(
        mixtures={
            "video": make_tiny_video_expert(num_layers=num_layers),
            "action": make_tiny_action_expert(num_layers=num_layers),
        },
        mot_checkpoint_mixed_attn=False,
    )
    if enable_dynamic_branch:
        from mtwam.models.wan22.dynamic_branch import DynamicBranch

        mot.dynamic_branch = DynamicBranch(
            video_expert=mot.mixtures["video"],
            num_layers=dynamic_branch_num_layers,
            ffn_moe=ffn_moe,
            ffn_moe_routing=ffn_moe_routing,
        )
    return mot


class DummyVAE(nn.Module):
    temporal_downsample_factor = 4
    upsampling_factor = 8

    def to(self, *args, **kwargs):
        return self


def make_tiny_mtwam(
    enable_dynamic_branch=False,
    dynamic_branch_num_layers=1,
    head_decoder_dim=8,
    dynamic_branch_horizon=2,
    proprio_dim=None,
):
    from mtwam.models.wan22.mtwam import MTWAM
    from mtwam.models.wan22.dynamic_branch import DynamicBranch

    video = make_tiny_video_expert()
    action = make_tiny_action_expert()
    mot = MoT(
        mixtures={"video": video, "action": action}, mot_checkpoint_mixed_attn=False
    )
    if enable_dynamic_branch:
        with torch.random.fork_rng(devices=[]):
            mot.dynamic_branch = DynamicBranch(
                video_expert=mot.mixtures["video"],
                num_layers=dynamic_branch_num_layers,
                build_heads=True,
                head_decoder_dim=head_decoder_dim,
                grid_traj=196,
                grid_tex=256,
                horizon=dynamic_branch_horizon,
            )
    return MTWAM(
        video_expert=video,
        action_expert=action,
        mot=mot,
        vae=DummyVAE(),
        text_dim=32,
        device="cpu",
        proprio_dim=proprio_dim,
    )


def build_tiny_payload(mot, T=2, H=2, W=4, Sa=3, L=2, seed=0):
    torch.manual_seed(seed)
    video = mot.mixtures["video"]
    action = mot.mixtures["action"]
    ctx = torch.randn(1, L, 32)
    vpre = video.pre_dit(
        x=torch.randn(1, 48, T, H, W),
        timestep=torch.tensor([500.0]),
        context=ctx,
        context_mask=None,
        action=None,
        fuse_vae_embedding_in_latents=True,
    )
    apre = action.pre_dit(
        action_tokens=torch.randn(1, Sa, 7),
        timestep=torch.tensor([500.0]),
        context=ctx,
        context_mask=None,
    )
    meta = {
        "Sv": vpre["tokens"].shape[1],
        "Sa": apre["tokens"].shape[1],
        "tpf": int(vpre["meta"]["tokens_per_frame"]),
    }
    return (vpre, apre, meta)


def assemble_mot_kwargs(mot, vpre, apre, attention_mask):
    return dict(
        embeds_all={"video": vpre["tokens"], "action": apre["tokens"]},
        attention_mask=attention_mask,
        freqs_all={"video": vpre["freqs"], "action": apre["freqs"]},
        context_all={
            "video": {"context": vpre["context"], "mask": vpre["context_mask"]},
            "action": {"context": apre["context"], "mask": apre["context_mask"]},
        },
        t_mod_all={"video": vpre["t_mod"], "action": apre["t_mod"]},
    )
