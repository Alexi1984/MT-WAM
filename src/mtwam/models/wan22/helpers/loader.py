from dataclasses import dataclass
import glob
import inspect
from pathlib import Path
from typing import Any
import torch
import time
from .io import ModelConfig, hash_model_file, load_state_dict
from .state_dict_converters import wan_video_vae_state_dict_converter
from ..wan_video_dit import WanVideoDiT
from ..wan_video_text_encoder import HuggingfaceTokenizer, WanTextEncoder
from ..wan_video_vae import WanVideoVAE38
from mtwam.utils.logging_config import get_logger

logger = get_logger(__name__)
SKIPPED_PRETRAIN_SENTINEL = "SKIPPED_PRETRAIN"


@dataclass
class Wan22LoadedComponents:
    dit: WanVideoDiT
    vae: WanVideoVAE38
    text_encoder: WanTextEncoder | None
    tokenizer: HuggingfaceTokenizer | None
    dit_path: str
    vae_path: str
    text_encoder_path: str | None
    tokenizer_path: str | None


WAN22_MODEL_REGISTRY = [
    {
        "model_hash": "9c8818c2cbea55eca56c7b447df170da",
        "model_name": "wan_video_text_encoder",
        "model_class": WanTextEncoder,
    },
    {
        "model_hash": "1f5ab7703c6fc803fdded85ff040c316",
        "model_name": "wan_video_dit",
        "model_class": WanVideoDiT,
    },
    {
        "model_hash": "e1de6c02cdac79f8b739f4d3698cd216",
        "model_name": "wan_video_vae",
        "model_class": WanVideoVAE38,
        "state_dict_converter": wan_video_vae_state_dict_converter,
    },
]


def _validate_dit_config(dit_config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(dit_config, dict):
        raise ValueError(f"`dit_config` must be a dict, got {type(dit_config)}")
    validated = dict(dit_config)
    signature = inspect.signature(WanVideoDiT.__init__)
    allowed_keys = set()
    required_keys = set()
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        allowed_keys.add(name)
        if param.default is inspect.Signature.empty:
            required_keys.add(name)
    unknown_keys = sorted(set(validated) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            f"Unknown keys in `dit_config`: {unknown_keys}. Allowed keys: {sorted(allowed_keys)}"
        )
    missing_keys = sorted(required_keys - set(validated))
    if missing_keys:
        raise ValueError(
            f"Missing required keys in `dit_config`: {missing_keys}. Please specify all required WanVideoDiT constructor args."
        )
    return validated


def _load_registered_model(
    path,
    model_name: str,
    torch_dtype: torch.dtype,
    device: str,
    model_kwargs_override: dict[str, Any] | None = None,
):
    model_hash = hash_model_file(path)
    matched_config = None
    for config in WAN22_MODEL_REGISTRY:
        if config["model_hash"] == model_hash and config["model_name"] == model_name:
            matched_config = config
            break
    if matched_config is None:
        raise ValueError(
            f"Cannot detect model type for {model_name}. File: {path}. Model hash: {model_hash}. This standalone package follows DiffSynth hash-based loading."
        )
    model_class = matched_config["model_class"]
    model_kwargs = dict(matched_config.get("extra_kwargs", {}))
    if model_kwargs_override is not None:
        model_kwargs.update(model_kwargs_override)
    state_dict_converter = matched_config.get("state_dict_converter")
    model = model_class(**model_kwargs)
    state_dict = load_state_dict(path, torch_dtype=torch_dtype, device="cpu")
    if state_dict_converter is not None:
        state_dict = state_dict_converter(state_dict)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device=device, dtype=torch_dtype)
    return model


def _normalize_video_dit_state_dict(state_dict: dict[str, torch.Tensor]):
    keys = list(state_dict.keys())
    if not keys:
        raise ValueError("video_dit_pretrained_path file contains an empty state dict.")
    for prefix in ("dit.", "model."):
        if all((k.startswith(prefix) for k in keys)):
            return ({k[len(prefix) :]: v for k, v in state_dict.items()}, prefix)
    if "blocks.0.self_attn.q.weight" in state_dict:
        return (state_dict, "")
    raise ValueError(
        f"Unrecognized video DiT state-dict layout for `video_dit_pretrained_path`: {len(keys)} keys carry no uniform 'dit.'/'model.' prefix and the native probe 'blocks.0.self_attn.q.weight' is absent. First keys: {sorted(keys)[:20]}"
    )


def _load_video_dit_from_path(
    video_dit_pretrained_path: str,
    validated_dit_config: dict[str, Any],
    torch_dtype: torch.dtype,
    device: str,
):
    p = Path(str(video_dit_pretrained_path))
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[5] / p
    matches = sorted(glob.glob(str(p)))
    if not matches:
        raise FileNotFoundError(f"`video_dit_pretrained_path` matched no files: {p}")
    path = matches if len(matches) > 1 else matches[0]
    model = WanVideoDiT(**validated_dit_config)
    state_dict = load_state_dict(path, torch_dtype=torch_dtype, device="cpu")
    state_dict, stripped_prefix = _normalize_video_dit_state_dict(state_dict)
    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(
            f"`video_dit_pretrained_path` load is not exact: missing={len(result.missing_keys)} {sorted(result.missing_keys)[:10]}, unexpected={len(result.unexpected_keys)} {sorted(result.unexpected_keys)[:10]} (file: {path}, stripped_prefix={stripped_prefix!r})"
        )
    logger.info(
        "Loaded video DiT warm-start weights from %s (stripped_prefix=%r, %d tensors).",
        path,
        stripped_prefix,
        len(state_dict),
    )
    return (model.to(device=device, dtype=torch_dtype), str(path))


def _resolve_configs(
    model_id: str, tokenizer_model_id: str, redirect_common_files: bool = True
):
    dit_config = ModelConfig(
        model_id=model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors"
    )
    text_config = ModelConfig(
        model_id=model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"
    )
    vae_config = ModelConfig(model_id=model_id, origin_file_pattern="Wan2.2_VAE.pth")
    tokenizer_config = ModelConfig(
        model_id=tokenizer_model_id, origin_file_pattern="google/umt5-xxl/"
    )
    if redirect_common_files:
        redirect_dict = {
            "models_t5_umt5-xxl-enc-bf16.pth": (
                "DiffSynth-Studio/Wan-Series-Converted-Safetensors",
                "models_t5_umt5-xxl-enc-bf16.safetensors",
            ),
            "Wan2.2_VAE.pth": (
                "DiffSynth-Studio/Wan-Series-Converted-Safetensors",
                "Wan2.2_VAE.safetensors",
            ),
        }
        text_config.model_id, text_config.origin_file_pattern = redirect_dict[
            text_config.origin_file_pattern
        ]
        vae_config.model_id, vae_config.origin_file_pattern = redirect_dict[
            vae_config.origin_file_pattern
        ]
    return (dit_config, text_config, vae_config, tokenizer_config)


def load_wan22_ti2v_5b_components(
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
    tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
    tokenizer_max_len: int = 512,
    redirect_common_files: bool = True,
    dit_config: dict[str, Any] | None = None,
    skip_dit_load_from_pretrain: bool = False,
    load_text_encoder: bool = True,
    video_dit_pretrained_path: str | None = None,
):
    logger.info("Loading Wan2.2-TI2V-5B components...")
    start = time.time()
    if video_dit_pretrained_path and skip_dit_load_from_pretrain:
        raise ValueError(
            "`video_dit_pretrained_path` and `skip_dit_load_from_pretrain=True` are mutually exclusive."
        )
    if dit_config is None:
        raise ValueError("`dit_config` is required for Wan2.2-TI2V-5B loading.")
    validated_dit_config = _validate_dit_config(dit_config)
    dit_model_config, text_config, vae_config, tokenizer_config = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=redirect_common_files,
    )
    vae_config.download_if_necessary()
    if load_text_encoder:
        text_config.download_if_necessary()
        tokenizer_config.download_if_necessary()
    if skip_dit_load_from_pretrain:
        logger.info(
            "Skipping pretrained video DiT load (`skip_dit_load_from_pretrain=True`); initializing video expert randomly and expecting checkpoint override."
        )
        dit: WanVideoDiT = WanVideoDiT(**validated_dit_config).to(
            device=device, dtype=torch_dtype
        )
        dit_path = SKIPPED_PRETRAIN_SENTINEL
    elif video_dit_pretrained_path:
        dit, dit_path = _load_video_dit_from_path(
            video_dit_pretrained_path,
            validated_dit_config,
            torch_dtype=torch_dtype,
            device=device,
        )
    else:
        dit_model_config.download_if_necessary()
        dit = _load_registered_model(
            dit_model_config.path,
            "wan_video_dit",
            torch_dtype=torch_dtype,
            device=device,
            model_kwargs_override=validated_dit_config,
        )
        dit_path = str(dit_model_config.path)
    text_encoder: WanTextEncoder | None = None
    tokenizer: HuggingfaceTokenizer | None = None
    text_encoder_path: str | None = None
    tokenizer_path: str | None = None
    if load_text_encoder:
        text_encoder = _load_registered_model(
            text_config.path,
            "wan_video_text_encoder",
            torch_dtype=torch_dtype,
            device=device,
        )
        tokenizer = HuggingfaceTokenizer(
            name=tokenizer_config.path,
            seq_len=int(tokenizer_max_len),
            clean="whitespace",
        )
        text_encoder_path = str(text_config.path)
        tokenizer_path = str(tokenizer_config.path)
    else:
        logger.info(
            "Skipping pretrained text encoder/tokenizer load (`load_text_encoder=False`); training must provide cached `context/context_mask`."
        )
    vae: WanVideoVAE38 = _load_registered_model(
        vae_config.path, "wan_video_vae", torch_dtype=torch_dtype, device=device
    )
    logger.info(
        "Finished loading Wan2.2-TI2V-5B components in %.2f seconds.",
        time.time() - start,
    )
    return Wan22LoadedComponents(
        dit=dit,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        dit_path=dit_path,
        vae_path=str(vae_config.path),
        text_encoder_path=text_encoder_path,
        tokenizer_path=tokenizer_path,
    )


def load_wan22_vae_only(
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
    tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
    redirect_common_files: bool = True,
) -> WanVideoVAE38:
    _, _, vae_config, _ = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=redirect_common_files,
    )
    vae_config.download_if_necessary()
    vae: WanVideoVAE38 = _load_registered_model(
        vae_config.path, "wan_video_vae", torch_dtype=torch_dtype, device=device
    )
    logger.info(
        "Loaded Wan2.2 VAE only (no video DiT / ActionDiT / text encoder) from %s",
        vae_config.path,
    )
    return vae
