from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

from past_mri.constants import CONTRAST_TO_ID, FIELD_TO_ID
from past_mri.data.masked_context_restoration_dataset import (
    MaskedContextRestorationDataset,
    make_block_mask,
)
from past_mri.io import stack_slices


def foreground_robust_scale(
    clean_stack: np.ndarray,
    foreground: np.ndarray,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if foreground.ndim != 2 or clean_stack.ndim != 3:
        raise ValueError("clean_stack must be [D,H,W] and foreground must be [H,W]")
    values = clean_stack[:, foreground]
    if values.size < 2:
        return float(minimum)
    q16, q84 = np.quantile(values.astype(np.float64, copy=False), [0.16, 0.84])
    robust_std = 0.5 * float(q84 - q16)
    if not np.isfinite(robust_std):
        robust_std = float(np.std(values.astype(np.float64, copy=False)))
    return float(np.clip(robust_std, minimum, maximum))


def add_independent_gaussian_noise(
    clean_stack: np.ndarray,
    *,
    sigma: float,
    seed: int,
) -> np.ndarray:
    if clean_stack.ndim != 3:
        raise ValueError("clean_stack must have shape [D,H,W]")
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")
    noise_rng = np.random.RandomState(int(seed))
    noise = noise_rng.normal(0.0, float(sigma), size=clean_stack.shape).astype(np.float32)
    return clean_stack.astype(np.float32, copy=False) + noise


class MaskedDiffusionAutoencoderDataset(MaskedContextRestorationDataset):
    """Deterministic 2.5D adaptation of MDAE's noise-then-mask corruption."""

    def __init__(
        self,
        manifest_path: str,
        *,
        stack_depth: int = 5,
        patch_size: int = 256,
        virtual_samples: int = 1_000_000,
        seed: int = 20260718,
        cache_size: int = 16,
        mask_config: dict[str, Any] | None = None,
        noise_config: dict[str, Any] | None = None,
        foreground_threshold: float = 0.01,
        minimum_foreground_fraction: float = 0.03,
        crop_attempts: int = 8,
    ) -> None:
        super().__init__(
            manifest_path,
            stack_depth=stack_depth,
            patch_size=patch_size,
            virtual_samples=virtual_samples,
            seed=seed,
            cache_size=cache_size,
            mask_config=mask_config,
            corruption_config={},
            foreground_threshold=foreground_threshold,
            minimum_foreground_fraction=minimum_foreground_fraction,
            crop_attempts=crop_attempts,
        )
        self.noise_config = dict(noise_config or {})

    def _sample_region_masks(
        self,
        target: np.ndarray,
        rng: random.Random,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        ratio_min = float(self.mask_config.get("ratio_min", 0.01))
        ratio_max = float(self.mask_config.get("ratio_max", 0.99))
        block_sizes = [int(value) for value in self.mask_config.get("block_sizes", [16])]
        attempts = max(1, int(self.mask_config.get("foreground_attempts", 32)))
        minimum_pixels = max(1, int(self.mask_config.get("minimum_foreground_pixels_per_region", 64)))
        foreground = target[0] > self.foreground_threshold

        best: tuple[float, np.ndarray, np.ndarray, np.ndarray, float] | None = None
        for _ in range(attempts):
            sampled_ratio = rng.uniform(ratio_min, ratio_max)
            mask = make_block_mask(
                target.shape[-2],
                target.shape[-1],
                ratio=sampled_ratio,
                block_size=rng.choice(block_sizes),
                rng=rng,
            )[None]
            masked_region = mask * foreground[None]
            visible_region = (1.0 - mask) * foreground[None]
            masked_pixels = float(masked_region.sum())
            visible_pixels = float(visible_region.sum())
            score = min(masked_pixels, visible_pixels)
            if best is None or score > best[0]:
                best = (score, mask, masked_region, visible_region, sampled_ratio)
            if masked_pixels >= minimum_pixels and visible_pixels >= minimum_pixels:
                break

        assert best is not None
        score, mask, masked_region, visible_region, sampled_ratio = best
        if score < 1.0:
            raise ValueError("Unable to construct non-empty masked and visible foreground regions")
        return (
            mask.astype(np.float32, copy=False),
            masked_region.astype(np.float32, copy=False),
            visible_region.astype(np.float32, copy=False),
            float(sampled_ratio),
        )

    def __getitem__(self, token: int) -> dict[str, Any]:
        token = int(token)
        rng = self._rng_for_token(token)
        (contrast, field), row = self._select_row(token, rng)
        volume = self._load_cached(str(row["path"]))
        z_start = max(0, int(row.get("z_start", 0)))
        z_end = min(volume.shape[2], int(row.get("z_end", volume.shape[2])))
        if z_end <= z_start:
            raise ValueError(f"Invalid z range [{z_start}, {z_end}) for {row['path']}")
        z = rng.randrange(z_start, z_end)
        clean_stack = np.clip(
            self._crop_stack(stack_slices(volume, z, self.stack_depth), rng),
            0.0,
            1.0,
        ).astype(np.float32, copy=False)
        center = self.stack_depth // 2
        target = clean_stack[center : center + 1].copy()
        foreground = target[0] > self.foreground_threshold

        noise_cfg = self.noise_config
        sigma_data = foreground_robust_scale(
            clean_stack,
            foreground,
            minimum=float(noise_cfg.get("sigma_data_min", 0.05)),
            maximum=float(noise_cfg.get("sigma_data_max", 0.30)),
        )
        timestep = rng.uniform(float(noise_cfg.get("t_min", 0.0)), float(noise_cfg.get("t_max", 1.0)))
        sigma = timestep * float(noise_cfg.get("sigma_max_multiplier", 1.0)) * sigma_data
        noise_seed = rng.randrange(0, 2**32 - 1)

        # The paper applies diffusion corruption before spatial masking.
        noisy_stack = add_independent_gaussian_noise(clean_stack, sigma=sigma, seed=noise_seed)
        mask, masked_region, visible_region, sampled_ratio = self._sample_region_masks(target, rng)
        fill_value = float(self.mask_config.get("fill_value", 0.0))
        input_stack = noisy_stack * (1.0 - mask) + fill_value * mask

        return {
            "input_stack": torch.from_numpy(input_stack.copy()).float(),
            "target": torch.from_numpy(target.copy()).float(),
            "mask": torch.from_numpy(mask.copy()).float(),
            "masked_loss_mask": torch.from_numpy(masked_region.copy()).float(),
            "visible_loss_mask": torch.from_numpy(visible_region.copy()).float(),
            "timestep": torch.tensor(timestep, dtype=torch.float32),
            "sigma": torch.tensor(sigma, dtype=torch.float32),
            "sigma_data": torch.tensor(sigma_data, dtype=torch.float32),
            "source_field_id": torch.tensor(FIELD_TO_ID[field], dtype=torch.long),
            "target_field_id": torch.tensor(FIELD_TO_ID[field], dtype=torch.long),
            "contrast_id": torch.tensor(CONTRAST_TO_ID[contrast], dtype=torch.long),
            "meta": {
                "token": token,
                "path": str(row["path"]),
                "subject_id": str(row.get("subject_id", "")),
                "contrast": contrast,
                "field": field,
                "z": z,
                "timestep": float(timestep),
                "sigma": float(sigma),
                "sigma_data": float(sigma_data),
                "noise_seed": int(noise_seed),
                "sampled_mask_ratio": float(sampled_ratio),
                "actual_mask_ratio": float(mask.mean()),
                "masked_foreground_pixels": float(masked_region.sum()),
                "visible_foreground_pixels": float(visible_region.sum()),
            },
        }
