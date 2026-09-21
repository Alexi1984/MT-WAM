import logging
import os
from pathlib import Path
import torch
from mtwam.datasets.lerobot.teacher_cache import TEACHER_GRID_DINO, TEACHER_GRID_TRAJ

TEACHER_IMG_SIZE = 224
TRAJ_GRID_HW = 14
TRAJ_PATCH = TEACHER_IMG_SIZE // TRAJ_GRID_HW
DINO_GRID_HW = 16
DEFAULT_VAE_UPSAMPLING_FACTOR = 16
VAE_TEMPORAL_FACTOR = 4
DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)
assert TRAJ_GRID_HW * TRAJ_GRID_HW == TEACHER_GRID_TRAJ, (
    "traj grid must square to TEACHER_GRID_TRAJ"
)
assert DINO_GRID_HW * DINO_GRID_HW == TEACHER_GRID_DINO, (
    "dino grid must square to TEACHER_GRID_DINO"
)


def _grid_query_points(grid_hw, img_size, device) -> torch.Tensor:
    gh, gw = (
        (grid_hw, grid_hw)
        if isinstance(grid_hw, int)
        else (int(grid_hw[0]), int(grid_hw[1]))
    )
    H, W = (
        (img_size, img_size)
        if isinstance(img_size, int)
        else (int(img_size[0]), int(img_size[1]))
    )
    patch_h, patch_w = (H // gh, W // gw)
    centers = [
        (patch_w // 2 + j * patch_w, patch_h // 2 + i * patch_h)
        for i in range(gh)
        for j in range(gw)
    ]
    xy = torch.tensor(centers, dtype=torch.float32, device=device)
    t = torch.zeros(xy.shape[0], 1, dtype=torch.float32, device=device)
    return torch.cat([t, xy], dim=1).unsqueeze(0)


def load_teacher_models(
    cotracker_ckpt, dino_ckpt, device="cuda", compile_mode=None
) -> dict:
    for name, path in (("CoTracker", cotracker_ckpt), ("DINOv2", dino_ckpt)):
        if not path or not Path(path).expanduser().is_file():
            raise FileNotFoundError(f"{name} checkpoint not found: {path}")
    from cotracker.predictor import CoTrackerPredictor

    cotracker = CoTrackerPredictor(
        checkpoint=str(Path(cotracker_ckpt).expanduser()), offline=True, window_len=60
    )
    cotracker = cotracker.to(device).eval()
    from dinov2.hub.backbones import dinov2_vitb14

    dino = dinov2_vitb14(pretrained=False)
    dino.load_state_dict(
        torch.load(
            str(Path(dino_ckpt).expanduser()), map_location="cpu", weights_only=True
        ),
        strict=True,
    )
    dino = dino.to(device).eval()
    if compile_mode:
        dino.forward = torch.compile(dino.forward, mode=compile_mode, dynamic=False)
        try:
            cotracker.forward = torch.compile(
                cotracker.forward, mode=compile_mode, dynamic=False
            )
        except Exception as err:
            logging.getLogger(__name__).warning(
                "teacher compile: CoTracker torch.compile failed (%s); running it eager (DINO stays compiled).",
                err,
            )
    return {"cotracker": cotracker, "dino": dino}


@torch.no_grad()
def extract_teacher_for_camera(
    cam_frames,
    models,
    vae_factor=DEFAULT_VAE_UPSAMPLING_FACTOR,
    device="cuda",
    traj_grid_hw=(TRAJ_GRID_HW, TRAJ_GRID_HW),
    img_size=(TEACHER_IMG_SIZE, TEACHER_IMG_SIZE),
    dino_grid_hw=(DINO_GRID_HW, DINO_GRID_HW),
    dino_size=None,
):
    P = cam_frames.shape[0] - 1
    frames = cam_frames.to(device)
    video = frames.unsqueeze(0) * 255.0
    queries = _grid_query_points(traj_grid_hw, img_size, device)
    pred_tracks, pred_vis = models["cotracker"](video, queries=queries)
    pred_tracks = pred_tracks[0]
    pred_vis = pred_vis[0]
    tracks = (pred_tracks[1:] - pred_tracks[0:1]) / float(vae_factor)
    vis = pred_vis[1:].bool()
    mean = torch.tensor(DINO_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(DINO_STD, device=device).view(1, 3, 1, 1)
    dino_frames = frames[1:]
    if dino_size is not None and tuple(dino_size) != tuple(dino_frames.shape[-2:]):
        dino_frames = torch.nn.functional.interpolate(
            dino_frames,
            size=tuple(dino_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    dino_in = (dino_frames - mean) / std
    out = models["dino"](dino_in, is_training=True)
    assert isinstance(out, dict) and "x_norm_patchtokens" in out, (
        f"DINOv2(is_training=True) must return a dict with 'x_norm_patchtokens', got {type(out)} — a different dinov2 version may return a pooled tensor (GPU-verify, dino_api review)"
    )
    dino = out["x_norm_patchtokens"]
    n_traj = int(traj_grid_hw[0]) * int(traj_grid_hw[1])
    n_dino = int(dino_grid_hw[0]) * int(dino_grid_hw[1])
    assert tuple(tracks.shape) == (P, n_traj, 2), (
        f"bad tracks shape {tuple(tracks.shape)} (expect P={P},{n_traj},2)"
    )
    assert tuple(dino.shape) == (P, n_dino, 768), (
        f"bad dino shape {tuple(dino.shape)} (expect P={P},{n_dino},768)"
    )
    return (tracks.float().cpu(), vis.cpu(), dino.float().cpu())


def _extract_teacher_batch_loop(
    teacher_frames, models, traj_grid_hw, img_size, dino_grid_hw, device="cuda"
):
    B, C = (int(teacher_frames.shape[0]), int(teacher_frames.shape[1]))
    tracks_b, vis_b, dino_b = ([], [], [])
    for b in range(B):
        tracks_c, vis_c, dino_c = ([], [], [])
        for c in range(C):
            t, v, d = extract_teacher_for_camera(
                teacher_frames[b, c],
                models,
                device=device,
                traj_grid_hw=traj_grid_hw,
                img_size=img_size,
                dino_grid_hw=dino_grid_hw,
            )
            tracks_c.append(t)
            vis_c.append(v)
            dino_c.append(d)
        tracks_b.append(torch.stack(tracks_c, dim=0))
        vis_b.append(torch.stack(vis_c, dim=0))
        dino_b.append(torch.stack(dino_c, dim=0))
    return (
        torch.stack(tracks_b, dim=0),
        torch.stack(vis_b, dim=0),
        torch.stack(dino_b, dim=0),
    )


@torch.no_grad()
def extract_teacher_batch(
    teacher_frames,
    models,
    traj_grid_hw,
    img_size,
    dino_grid_hw,
    device="cuda",
    vae_factor=DEFAULT_VAE_UPSAMPLING_FACTOR,
    dino_size=None,
):
    B, C = (int(teacher_frames.shape[0]), int(teacher_frames.shape[1]))
    P = int(teacher_frames.shape[2]) - 1
    n_traj = int(traj_grid_hw[0]) * int(traj_grid_hw[1])
    n_dino = int(dino_grid_hw[0]) * int(dino_grid_hw[1])
    flat = teacher_frames.reshape(B * C, *teacher_frames.shape[2:]).to(device)
    chunk = int(os.environ.get("MTWAM_TEACHER_BATCH_CHUNK", "0") or 0)
    if chunk <= 0:
        chunk = flat.shape[0]
    queries_one = _grid_query_points(traj_grid_hw, img_size, device)
    mean = torch.tensor(DINO_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(DINO_STD, device=device).view(1, 3, 1, 1)
    tracks_parts, vis_parts, dino_parts = ([], [], [])
    for start in range(0, flat.shape[0], chunk):
        part = flat[start : start + chunk]
        n = part.shape[0]
        video = part * 255.0
        queries = queries_one.expand(n, -1, -1).contiguous()
        pred_tracks, pred_vis = models["cotracker"](video, queries=queries)
        tracks_parts.append(
            (pred_tracks[:, 1:] - pred_tracks[:, 0:1]) / float(vae_factor)
        )
        vis_parts.append(pred_vis[:, 1:].bool())
        dino_frames = part[:, 1:].reshape(n * P, *part.shape[2:])
        if dino_size is not None and tuple(dino_size) != tuple(dino_frames.shape[-2:]):
            dino_frames = torch.nn.functional.interpolate(
                dino_frames,
                size=tuple(dino_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        dino_in = (dino_frames - mean) / std
        out = models["dino"](dino_in, is_training=True)
        assert isinstance(out, dict) and "x_norm_patchtokens" in out, (
            f"DINOv2(is_training=True) must return a dict with 'x_norm_patchtokens', got {type(out)}"
        )
        dino_parts.append(out["x_norm_patchtokens"].reshape(n, P, n_dino, 768))
    tracks = torch.cat(tracks_parts, dim=0).reshape(B, C, P, n_traj, 2).float()
    vis = torch.cat(vis_parts, dim=0).reshape(B, C, P, n_traj)
    dino = torch.cat(dino_parts, dim=0).reshape(B, C, P, n_dino, 768).float()
    assert tuple(tracks.shape) == (B, C, P, n_traj, 2), (
        f"bad tracks shape {tuple(tracks.shape)}"
    )
    assert tuple(dino.shape) == (B, C, P, n_dino, 768), (
        f"bad dino shape {tuple(dino.shape)}"
    )
    return (tracks, vis, dino)
