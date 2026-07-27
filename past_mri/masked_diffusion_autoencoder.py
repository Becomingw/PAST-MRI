from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from past_mri.model_factory import build_cond_naf_unet_from_config
from past_mri.models import CondNAFUNet


MDAE_TRANSFER_PREFIXES = {
    "encoder_middle_core": ("stem.", "encoders.", "downs.", "middle."),
    "full_core": ("stem.", "encoders.", "downs.", "middle.", "ups.", "decoders."),
}
MDAE_TRANSFER_EXCLUDED_PARTS = (
    ".film1.",
    ".film2.",
    ".pw1_lora.",
    ".pw2_lora.",
    ".ffn1_lora.",
    ".ffn2_lora.",
)


def sinusoidal_timestep_embedding(timestep: torch.Tensor, dimension: int, max_period: int = 10_000) -> torch.Tensor:
    if timestep.ndim != 1:
        raise ValueError(f"timestep must have shape [B], got {tuple(timestep.shape)}")
    if dimension < 4:
        raise ValueError("time embedding dimension must be at least 4")
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(float(max_period))
        * torch.arange(half, device=timestep.device, dtype=torch.float32)
        / max(1, half - 1)
    )
    arguments = timestep.float()[:, None] * frequencies[None]
    embedding = torch.cat([torch.cos(arguments), torch.sin(arguments)], dim=1)
    if embedding.shape[1] < dimension:
        embedding = F.pad(embedding, (0, dimension - embedding.shape[1]))
    return embedding


class MDAE2p5DModel(nn.Module):
    """Training-only MDAE wrapper around the CondNAFUNet visual trunk."""

    def __init__(self, backbone: CondNAFUNet, *, time_embedding_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.time_embedding_dim = int(time_embedding_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embedding_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.reconstruction_head = nn.Conv2d(backbone.stem.out_channels, 1, kernel_size=3, padding=1)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MDAE2p5DModel":
        model_cfg = dict(config.get("model", {}) or {})
        cond_dim = int(model_cfg.get("cond_dim", 256))
        return cls(
            build_cond_naf_unet_from_config(config),
            time_embedding_dim=int(model_cfg.get("time_embedding_dim", 256)),
            cond_dim=cond_dim,
        )

    def forward(self, input_stack: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        time_embedding = sinusoidal_timestep_embedding(timestep, self.time_embedding_dim)
        cond = self.time_mlp(time_embedding.to(dtype=input_stack.dtype))
        x = self.backbone.stem(input_stack)
        skips: list[torch.Tensor] = []
        for encoder, down in zip(self.backbone.encoders, self.backbone.downs):
            x = encoder(x, cond)
            skips.append(x)
            x = down(x)
        x = self.backbone.middle(x, cond)
        for up, decoder in zip(self.backbone.ups, self.backbone.decoders):
            x = up(x)
            skip = skips.pop()
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(x + skip, cond)
        return torch.sigmoid(self.reconstruction_head(x))


def visible_noise_weight(
    sigma: torch.Tensor,
    sigma_data: torch.Tensor,
    *,
    mode: str,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    if sigma.shape != sigma_data.shape:
        raise ValueError("sigma and sigma_data must have the same shape")
    if mode == "uniform":
        return torch.ones_like(sigma)
    if mode == "bounded_inverse_variance":
        ratio = sigma / sigma_data.clamp_min(float(epsilon))
        return 1.0 / (1.0 + ratio.square())
    raise ValueError(f"Unsupported visible noise weighting: {mode}")


def _region_mse(error_squared: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
    if error_squared.shape != region.shape:
        raise ValueError(f"error/region shape mismatch: {error_squared.shape} vs {region.shape}")
    denominator = region.flatten(1).sum(dim=1)
    if torch.any(denominator < 1.0):
        raise ValueError("Every sample must have at least one supervised voxel in each region")
    return (error_squared * region).flatten(1).sum(dim=1) / denominator


def masked_diffusion_autoencoder_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    masked_region: torch.Tensor,
    visible_region: torch.Tensor,
    sigma: torch.Tensor,
    sigma_data: torch.Tensor,
    *,
    masked_weight: float = 1.0,
    visible_weight: float = 1.0,
    visible_weighting: str = "bounded_inverse_variance",
) -> dict[str, torch.Tensor]:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction/target shape mismatch: {prediction.shape} vs {target.shape}")
    error_squared = (prediction - target).square()
    masked_per_sample = _region_mse(error_squared, masked_region)
    visible_per_sample = _region_mse(error_squared, visible_region)
    noise_weight = visible_noise_weight(sigma, sigma_data, mode=visible_weighting)
    weighted_visible_per_sample = noise_weight * visible_per_sample
    loss = float(masked_weight) * masked_per_sample.mean() + float(visible_weight) * weighted_visible_per_sample.mean()
    return {
        "loss": loss,
        "masked_mse": masked_per_sample.mean(),
        "visible_mse": visible_per_sample.mean(),
        "weighted_visible_mse": weighted_visible_per_sample.mean(),
        "visible_noise_weight": noise_weight.mean(),
        "masked_mse_per_sample": masked_per_sample,
        "visible_mse_per_sample": visible_per_sample,
        "weighted_visible_mse_per_sample": weighted_visible_per_sample,
    }


def _checkpoint_model_state(checkpoint: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint does not contain a model state dictionary")
    return state


def extract_mdae_transfer_state(
    checkpoint: Mapping[str, Any],
    *,
    transfer_mode: str,
) -> dict[str, torch.Tensor]:
    if transfer_mode not in MDAE_TRANSFER_PREFIXES:
        raise ValueError(f"Unsupported MDAE transfer mode: {transfer_mode}")
    prefixes = MDAE_TRANSFER_PREFIXES[transfer_mode]
    transfer_state: dict[str, torch.Tensor] = {}
    for key, value in _checkpoint_model_state(checkpoint).items():
        normalized = str(key)
        if normalized.startswith("module."):
            normalized = normalized[len("module.") :]
        if not normalized.startswith("backbone."):
            continue
        target_key = normalized[len("backbone.") :]
        if not any(target_key.startswith(prefix) for prefix in prefixes):
            continue
        if any(part in target_key for part in MDAE_TRANSFER_EXCLUDED_PARTS):
            continue
        transfer_state[target_key] = value.detach().cpu()
    if not transfer_state:
        raise ValueError("no transferable CondNAFUNet core keys found in MDAE checkpoint")
    return transfer_state


def export_mdae_initializer(
    source_checkpoint: str | Path,
    output_checkpoint: str | Path,
    *,
    transfer_mode: str,
) -> dict[str, Any]:
    source_checkpoint = Path(source_checkpoint)
    output_checkpoint = Path(output_checkpoint)
    checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    state = extract_mdae_transfer_state(checkpoint, transfer_mode=transfer_mode)
    payload = {
        "format": "task3_mdae2p5d_initializer_v1",
        "model": state,
        "step": int(checkpoint.get("step", 0)) if isinstance(checkpoint, Mapping) else 0,
        "source_checkpoint": str(source_checkpoint),
        "transfer_mode": transfer_mode,
        "transfer_policy": "core visual weights only; reset translation conditioning, FiLM, decoder/head as specified",
    }
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_checkpoint.with_name(output_checkpoint.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output_checkpoint)
    return {
        "source_checkpoint": str(source_checkpoint),
        "output_checkpoint": str(output_checkpoint),
        "step": payload["step"],
        "transfer_mode": transfer_mode,
        "transferred_keys": len(state),
        "transferred_parameters": sum(int(value.numel()) for value in state.values()),
        "excluded_parts": list(MDAE_TRANSFER_EXCLUDED_PARTS),
    }
