import copy
import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block


class RoleMoEFFN(nn.Module):
    def __init__(self, donor_ffn: nn.Module, routing: str = "role"):
        super().__init__()
        if routing != "role":
            raise ValueError(f"RoleMoEFFN supports routing=role only, got {routing!r}.")
        self.routing = routing
        self.ffn_t = copy.deepcopy(donor_ffn)
        self.ffn_s = copy.deepcopy(donor_ffn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        st = x.shape[1]
        if st % 2 != 0:
            raise ValueError(
                f"RoleMoEFFN expects an even [t|s] branch seq (St=2*ff), got St={st}"
            )
        h = st // 2
        return torch.cat([self.ffn_t(x[:, :h]), self.ffn_s(x[:, h:])], dim=1)


class DynamicBranch(nn.Module):
    def __init__(
        self,
        video_expert,
        num_layers: int,
        _donor_blocks=None,
        build_heads: bool = False,
        head_decoder_dim: int = 512,
        grid_traj: int = 196,
        grid_tex: int = 256,
        horizon: int = 2,
        read_text: bool = False,
        ffn_moe: bool = False,
        ffn_moe_routing: str = "role",
    ):
        super().__init__()
        if ffn_moe_routing != "role":
            raise ValueError("MT-WAM supports role routing only.")
        if int(horizon) < 1:
            raise ValueError(f"dynamic branch horizon must be >= 1, got {horizon}")
        self.read_text = read_text
        N = len(video_expert.blocks)
        if num_layers <= 0 or num_layers > N:
            raise ValueError(
                f"dynamic_branch_num_layers must be in [1, {N}] (video tower depth), got {num_layers}"
            )
        donor = (
            _donor_blocks
            if _donor_blocks is not None
            else video_expert.blocks[N - num_layers : N]
        )
        if len(donor) < num_layers:
            raise ValueError(
                f"donor must provide >= {num_layers} blocks, got {len(donor)}"
            )
        ref_sd = video_expert.blocks[N - 1].state_dict()
        donor_sd = donor[0].state_dict()
        if set(donor_sd) != set(ref_sd) or any(
            (donor_sd[k].shape != ref_sd[k].shape for k in ref_sd)
        ):
            raise ValueError(
                "dynamic branch donor geometry must match the video tower."
            )
        self.num_layers = num_layers
        self.blocks = nn.ModuleList(
            [
                copy.deepcopy(video_expert.blocks[N - num_layers + i])
                for i in range(num_layers)
            ]
        )
        self.ffn_moe = ffn_moe
        self.ffn_moe_routing = ffn_moe_routing
        if ffn_moe:
            _hidden = int(video_expert.hidden_dim)
            for blk in self.blocks:
                blk.ffn = RoleMoEFFN(blk.ffn, routing=ffn_moe_routing)
        hidden = int(video_expert.hidden_dim)
        self.role_t = nn.Parameter(torch.empty(hidden))
        self.role_s = nn.Parameter(torch.empty(hidden))
        nn.init.normal_(self.role_t, std=0.02)
        nn.init.normal_(self.role_s, std=0.02)
        with torch.no_grad():
            self.role_s.add_(0.02)
        self.traj_head = None
        self.tex_head = None
        if build_heads:
            self.traj_head = TrajMAEHead(
                in_dim=hidden,
                decoder_dim=head_decoder_dim,
                grid=grid_traj,
                horizon=horizon,
            )
            self.tex_head = TexMAEHead(
                in_dim=hidden,
                decoder_dim=head_decoder_dim,
                grid=grid_tex,
                horizon=horizon,
            )


def _get_2d_sincos_pos_embed(embed_dim, grid_h, grid_w=None):
    if embed_dim % 4 != 0:
        raise ValueError(
            f"decoder_dim must be divisible by 4 for 2D sincos, got {embed_dim}"
        )
    if grid_w is None:
        grid_w = grid_h
    rows = np.arange(grid_h, dtype=np.float32)
    cols = np.arange(grid_w, dtype=np.float32)
    grid = np.stack(np.meshgrid(cols, rows), axis=0).reshape([2, 1, grid_h, grid_w])

    def _1d(dim, pos):
        omega = 1.0 / 10000 ** (np.arange(dim // 2, dtype=np.float32) / (dim / 2.0))
        out = np.einsum("m,d->md", pos.reshape(-1), omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    return np.concatenate(
        [_1d(embed_dim // 2, grid[0]), _1d(embed_dim // 2, grid[1])], axis=1
    )


class _BranchMAEHead(nn.Module):
    def __init__(self, in_dim, decoder_dim, grid, horizon, out_dim, num_heads=None):
        super().__init__()
        if isinstance(grid, (tuple, list)):
            gh, gw = (int(grid[0]), int(grid[1]))
            grid_flat = gh * gw
        else:
            grid_flat = int(grid)
            gh = gw = int(round(grid_flat**0.5))
            if gh * gw != grid_flat:
                raise ValueError(
                    f"square grid must be a perfect square, got {grid_flat}"
                )
        if num_heads is None:
            num_heads = max(1, decoder_dim // 32)
        self.grid, self.horizon, self.out_dim = (grid_flat, horizon, out_dim)
        self.obs_proj = nn.Linear(in_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.frame_emb = nn.Parameter(torch.zeros(horizon, 1, decoder_dim))
        nn.init.normal_(self.frame_emb, std=0.02)
        self.register_buffer(
            "mask_pos",
            torch.from_numpy(_get_2d_sincos_pos_embed(decoder_dim, gh, gw))
            .float()
            .unsqueeze(0),
            persistent=False,
        )
        self.decoder = nn.Sequential(
            Block(
                decoder_dim,
                num_heads=num_heads,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
            ),
            Block(
                decoder_dim,
                num_heads=num_heads,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
            ),
        )
        self.norm = nn.LayerNorm(decoder_dim)
        self.pred = nn.Linear(decoder_dim, out_dim)

    def forward(self, x):
        B, st_f = (x.shape[0], x.shape[1])
        obs = self.obs_proj(x)
        mask = (self.mask_token + self.mask_pos).unsqueeze(1)
        mask = (mask + self.frame_emb.unsqueeze(0)).expand(B, -1, -1, -1)
        mask = mask.reshape(B, self.horizon * self.grid, -1)
        seq = self.norm(self.decoder(torch.cat([obs, mask], dim=1)))
        out = self.pred(seq[:, st_f:, :])
        return out.reshape(B, self.horizon, self.grid, self.out_dim)


class TrajMAEHead(_BranchMAEHead):
    def __init__(self, in_dim, decoder_dim, grid, horizon, num_heads=None):
        super().__init__(
            in_dim, decoder_dim, grid, horizon, out_dim=2, num_heads=num_heads
        )


class TexMAEHead(_BranchMAEHead):
    def __init__(
        self, in_dim, decoder_dim, grid, horizon, feat_dim=768, num_heads=None
    ):
        super().__init__(
            in_dim, decoder_dim, grid, horizon, out_dim=feat_dim, num_heads=num_heads
        )
