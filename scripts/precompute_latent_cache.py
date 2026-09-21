import json
import logging
import torch
from mtwam.datasets.lerobot.latent_cache import (
    assert_window_identity,
    build_cache_meta,
    build_episode_plan,
    episode_cache_path,
    episode_is_complete,
    repo_key_for,
    save_episode_latents,
    shard_episodes,
    write_cache_meta,
)

logger = logging.getLogger(__name__)


def encode_episode(train_ds, model, ep, batch_size, device, dtype):
    from mtwam.datasets.lerobot.teacher_cache import derive_teacher_window_id

    ids = derive_teacher_window_id(train_ds.lerobot_dataset, ep["global_start"])
    if int(ids["episode_index"]) != ep["episode_index"] or int(ids["frame_index"]) != 0:
        raise RuntimeError(
            f"episode plan drift at global idx {ep['global_start']}: plan says (ep={ep['episode_index']}, fr=0) but raw row says (ep={ids['episode_index']}, fr={ids['frame_index']}). Check the episode selection settings."
        )
    if repo_key_for(ids["repo_id"]) != ep["repo_key"]:
        raise RuntimeError(
            f"repo drift at global idx {ep['global_start']}: plan={ep['repo_key']} raw={repo_key_for(ids['repo_id'])}"
        )
    chunks, buf = ([], [])
    for i in range(ep["global_start"], ep["global_start"] + ep["n_windows"]):
        w = train_ds._get(int(i), return_window_ids=True)
        wid = w.get("window_ids")
        if wid is None:
            raise RuntimeError("window_ids missing from _get(return_window_ids=True).")
        assert_window_identity(wid, ep, i)
        buf.append(w["video"])
        if len(buf) == batch_size:
            chunks.append(_encode_batch(model, buf, device, dtype))
            buf = []
    if buf:
        chunks.append(_encode_batch(model, buf, device, dtype))
    return torch.cat(chunks, dim=0)


def _encode_batch(model, videos, device, dtype):
    vb = torch.stack(videos, dim=0).to(device=device, dtype=dtype, non_blocking=True)
    with torch.no_grad():
        z = model._encode_video_latents(vb)
    return z.to(device="cpu", dtype=torch.bfloat16)


class _VaeOnlyEncoder:
    def __init__(self, vae, device):
        self.vae = vae
        self.device = torch.device(device)

    def eval(self):
        self.vae.eval()
        return self

    def _encode_video_latents(
        self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)
    ):
        return self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )


def main():
    import hydra
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    @hydra.main(config_path="../configs", config_name="train_libero", version_base=None)
    def _run(cfg):
        logging.basicConfig(level=logging.INFO)
        lc = cfg.get("latent_cache", {})
        out_dir = lc.get("out_dir")
        if not out_dir:
            raise ValueError("+latent_cache.out_dir=<dir> is required.")
        shard = int(lc.get("shard", 0))
        num_shards = int(lc.get("num_shards", 1))
        batch_size = int(lc.get("batch_size", 8))
        max_episodes = lc.get("max_episodes")
        vsp = float(cfg.data.train.get("val_set_proportion", 0.05))
        if vsp != 0.0:
            raise ValueError(
                f"Latent precomputation requires data.train.val_set_proportion=0, got {vsp}."
            )
        if bool(cfg.data.train.get("skip_padding_as_possible", False)):
            raise ValueError(
                "Latent precomputation requires data.train.skip_padding_as_possible=false to preserve window identities."
            )
        import os as _os
        from mtwam.utils import misc as _misc

        _misc.register_work_dir(cfg.output_dir)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16
        if bool(lc.get("vae_only", False)):
            from mtwam.models.wan22.helpers.loader import load_wan22_vae_only

            logger.info("Building VAE only (vae_only=true) on %s ...", device)
            vae = load_wan22_vae_only(
                device=device,
                torch_dtype=dtype,
                model_id=cfg.model.model_id,
                tokenizer_model_id=cfg.model.tokenizer_model_id,
                redirect_common_files=bool(
                    cfg.model.get("redirect_common_files", True)
                ),
            )
            model = _VaeOnlyEncoder(vae, device)
        else:
            logger.info("Building model (for its VAE) on %s ...", device)
            model = instantiate(cfg.model, model_dtype=dtype, device=device)
        model.eval()
        logger.info("Building train dataset ...")
        train_ds = instantiate(
            cfg.data.train,
            use_cached_latent=False,
            latent_cache_dir=None,
            enable_dynamic_branch=False,
            online_teacher=False,
        )
        base = train_ds.lerobot_dataset
        data_cfg = OmegaConf.to_container(cfg.data.train, resolve=True)
        plan = build_episode_plan(base.multi_dataset)
        mine = shard_episodes(plan, shard, num_shards)
        if max_episodes is not None:
            mine = mine[: int(max_episodes)]
        total_windows = sum((ep["n_windows"] for ep in mine))
        logger.info(
            "shard %d/%d: %d episodes / %d windows -> %s",
            shard,
            num_shards,
            len(mine),
            total_windows,
            out_dir,
        )
        image_keys_list = build_cache_meta(data_cfg, base.multi_dataset.ds_names).get(
            "image_keys"
        )
        image_keys_meta = json.dumps(image_keys_list) if image_keys_list else None
        meta_written = False
        done_windows = 0
        for ep in mine:
            path = episode_cache_path(out_dir, ep["repo_key"], ep["episode_index"])
            ep_str_meta = {
                "episode_index": ep["episode_index"],
                "repo_key": ep["repo_key"],
                "num_frames": data_cfg["num_frames"],
                "num_extra_ref_frames": data_cfg.get("num_extra_ref_frames", 0),
            }
            if image_keys_meta is not None:
                ep_str_meta["image_keys"] = image_keys_meta
            if episode_is_complete(path, ep["n_windows"], expected_meta=ep_str_meta):
                done_windows += ep["n_windows"]
                continue
            latents = encode_episode(train_ds, model, ep, batch_size, device, dtype)
            if latents.shape[0] != ep["n_windows"]:
                raise RuntimeError(
                    f"slot count mismatch for {path}: {latents.shape[0]} vs {ep['n_windows']}"
                )
            if not meta_written:
                write_cache_meta(
                    out_dir,
                    build_cache_meta(
                        data_cfg,
                        base.multi_dataset.ds_names,
                        latent_shape=latents.shape[1:],
                    ),
                )
                meta_written = True
            save_episode_latents(path, latents, str_meta=ep_str_meta)
            done_windows += ep["n_windows"]
            logger.info("wrote %s  [%d/%d windows]", path, done_windows, total_windows)
        if not meta_written:
            write_cache_meta(
                out_dir, build_cache_meta(data_cfg, base.multi_dataset.ds_names)
            )
        logger.info("shard %d/%d DONE: %d windows.", shard, num_shards, done_windows)

    _run()


if __name__ == "__main__":
    main()
