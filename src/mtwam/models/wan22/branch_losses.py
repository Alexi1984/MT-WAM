import torch
import torch.nn.functional as F


def _frame_valid(frame_is_pad):
    return (~frame_is_pad).float()


def derive_branch_frame_pad(
    image_is_pad,
    n_branch,
    horizon,
    temporal_downsample_factor,
    device,
    num_ref_latents=1,
):
    if image_is_pad is None:
        return torch.zeros(n_branch, horizon, dtype=torch.bool, device=device)
    tf = int(temporal_downsample_factor)
    latent_tail_pad = image_is_pad[:, 1:].reshape(n_branch, -1, tf).all(dim=2)
    latent_future_pad = latent_tail_pad[:, int(num_ref_latents) - 1 :]
    n_future = int(latent_future_pad.shape[1])
    if n_future < horizon:
        raise ValueError(
            f"dynamic branch needs >= horizon={horizon} future latent frames for frame_pad, but image_is_pad yields only {n_future} (raw frames={image_is_pad.shape[1]}, num_ref_latents={num_ref_latents}, temporal_downsample_factor={tf}). Lengthen the video window or shorten the branch horizon."
        )
    return latent_future_pad[:, -horizon:].to(device)


def compute_traj_loss(pred, target, vis, frame_is_pad, validity_mask: bool):
    if (
        pred.shape != target.shape
        or vis.shape != pred.shape[:-1]
        or frame_is_pad.shape[1] != pred.shape[2]
    ):
        raise ValueError(
            f"branch traj P/h mismatch: pred {tuple(pred.shape)} vs teacher {tuple(target.shape)} vs vis {tuple(vis.shape)} vs frame_is_pad {tuple(frame_is_pad.shape)}. Match dynamic_branch_horizon to the teacher frame count."
        )
    per_pt = F.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=-1)
    w = vis.float() if validity_mask else torch.ones_like(per_pt)
    fw = _frame_valid(frame_is_pad)[:, None, :, None]
    w = w * fw
    num = (per_pt * w).sum(dim=(2, 3))
    den = w.sum(dim=(2, 3)).clamp(min=1.0)
    return (num / den).mean()


def compute_tex_loss(pred, target, frame_is_pad):
    if pred.shape != target.shape or frame_is_pad.shape[1] != pred.shape[2]:
        raise ValueError(
            f"branch tex P/h mismatch: pred {tuple(pred.shape)} vs teacher {tuple(target.shape)} vs frame_is_pad {tuple(frame_is_pad.shape)}. Match dynamic_branch_horizon to the teacher frame count."
        )
    cos = 1.0 - F.cosine_similarity(pred.float(), target.float(), dim=-1)
    fw = _frame_valid(frame_is_pad)[:, None, :, None]
    num = (cos * fw).sum(dim=(2, 3))
    den = fw.expand_as(cos).sum(dim=(2, 3)).clamp(min=1.0)
    return (num / den).mean()


def split_branch_cameras(hidden, grid_hw, num_cameras):
    B, st_f, Hd = hidden.shape
    h, w = (int(grid_hw[0]), int(grid_hw[1]))
    if h * w != st_f:
        raise ValueError(
            f"grid_hw {(h, w)} (h*w={h * w}) must match branch St_f={st_f}"
        )
    if w % num_cameras != 0:
        raise ValueError(
            f"horizontal camera split needs grid width w={w} divisible by num_cameras={num_cameras}"
        )
    x = hidden.reshape(B, h, num_cameras, w // num_cameras, Hd)
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    return x.reshape(B * num_cameras, h * (w // num_cameras), Hd)


def split_branch_cameras_robotwin(hidden, grid_hw):
    B, st_f, Hd = hidden.shape
    h, w = (int(grid_hw[0]), int(grid_hw[1]))
    if h * w != st_f:
        raise ValueError(
            f"grid_hw {(h, w)} (h*w={h * w}) must match branch St_f={st_f}"
        )
    if h % 3 != 0 or w % 2 != 0:
        raise ValueError(
            f"robotwin camera split needs grid h divisible by 3 (cam_high=2h/3) and w by 2 (left/right at w/2), got grid_hw={(h, w)}"
        )
    grid = hidden.reshape(B, h, w, Hd)
    top = h * 2 // 3
    mid = w // 2
    cam_high = grid[:, :top, :, :].reshape(B, top * w, Hd)
    cam_left = grid[:, top:, :mid, :].reshape(B, (h - top) * mid, Hd)
    cam_right = grid[:, top:, mid:, :].reshape(B, (h - top) * (w - mid), Hd)
    return [cam_high, cam_left, cam_right]


def _run_head_per_camera(
    head, hidden, num_cameras, grid_hw, camera_layout="horizontal"
):
    B = hidden.shape[0]
    if num_cameras == 1:
        return head(hidden).unsqueeze(1)
    if grid_hw is None:
        raise ValueError(
            "per-camera (C>1) branch loss needs grid_hw=(h,w) to split cameras"
        )
    if camera_layout == "robotwin":
        cams = split_branch_cameras_robotwin(hidden, grid_hw)
        if num_cameras != len(cams):
            raise ValueError(
                f"robotwin camera layout expects {len(cams)} cameras, got num_cameras={num_cameras}"
            )
        outs = [head(c) for c in cams]
        return torch.stack(outs, dim=1)
    if camera_layout != "horizontal":
        raise ValueError(
            f"unknown camera_layout={camera_layout!r} (expected 'horizontal'/'robotwin')"
        )
    per_cam = split_branch_cameras(hidden, grid_hw, num_cameras)
    out = head(per_cam)
    return out.reshape(B, num_cameras, *out.shape[1:])


def predict_branch_heads_per_camera(
    branch, branch_hidden, num_cameras, grid_hw, camera_layout="horizontal"
):
    if branch is None or branch.traj_head is None or branch.tex_head is None:
        raise ValueError(
            "dynamic branch prediction requires both trajectory and appearance heads"
        )
    if set(branch_hidden) != {"traj", "tex"}:
        raise ValueError("branch_hidden must contain exactly {'traj', 'tex'}")
    motion = _run_head_per_camera(
        branch.traj_head,
        branch_hidden["traj"],
        int(num_cameras),
        grid_hw,
        camera_layout,
    )
    appearance = _run_head_per_camera(
        branch.tex_head, branch_hidden["tex"], int(num_cameras), grid_hw, camera_layout
    )
    return (motion, appearance)


def assemble_branch_loss(
    branch,
    branch_hidden,
    teacher,
    lambda_traj,
    lambda_tex,
    validity_mask,
    grid_hw=None,
    camera_layout="horizontal",
    num_ref_latents=1,
):
    nrl = int(num_ref_latents)
    if nrl > 1:
        st_f = int(branch_hidden["traj"].shape[1])
        if st_f % nrl != 0:
            raise ValueError(
                f"branch path length St_f={st_f} not divisible by num_ref_latents={nrl}"
            )
        frame_tokens = st_f // nrl
        branch_hidden = {k: v[:, -frame_tokens:] for k, v in branch_hidden.items()}
    total = 0.0
    loss_dict = {"loss_traj": 0.0, "loss_tex": 0.0}
    if lambda_traj > 0:
        pred = _run_head_per_camera(
            branch.traj_head,
            branch_hidden["traj"],
            int(teacher["tracks"].shape[1]),
            grid_hw,
            camera_layout,
        )
        lt = compute_traj_loss(
            pred,
            teacher["tracks"],
            teacher["track_vis"],
            teacher["frame_pad"],
            validity_mask,
        )
        total = total + lambda_traj * lt
        loss_dict["loss_traj"] = lambda_traj * float(lt.detach())
    if lambda_tex > 0:
        pred = _run_head_per_camera(
            branch.tex_head,
            branch_hidden["tex"],
            int(teacher["dino"].shape[1]),
            grid_hw,
            camera_layout,
        )
        lx = compute_tex_loss(pred, teacher["dino"], teacher["frame_pad"])
        total = total + lambda_tex * lx
        loss_dict["loss_tex"] = lambda_tex * float(lx.detach())
    return (total, loss_dict)
