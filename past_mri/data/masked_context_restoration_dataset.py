from __future__ import annotations

import json
import math
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from past_mri.constants import CONTRAST_TO_ID, FIELD_TO_ID
from past_mri.io import load_volume, stack_slices


def make_block_mask(
    height: int,
    width: int,
    *,
    ratio: float,
    block_size: int,
    rng: random.Random,
) -> np.ndarray:
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"mask ratio must be in (0, 1), got {ratio}")
    if block_size < 1:
        raise ValueError(f"block_size must be positive, got {block_size}")
    blocks_y = math.ceil(height / block_size)
    blocks_x = math.ceil(width / block_size)
    coordinates = [(y, x) for y in range(blocks_y) for x in range(blocks_x)]
    rng.shuffle(coordinates)
    masked_blocks = max(1, min(len(coordinates) - 1, round(ratio * len(coordinates))))
    mask = np.zeros((height, width), dtype=np.float32)
    for block_y, block_x in coordinates[:masked_blocks]:
        y0 = block_y * block_size
        x0 = block_x * block_size
        mask[y0 : min(height, y0 + block_size), x0 : min(width, x0 + block_size)] = 1.0
    return mask


class MaskedContextRestorationDataset(Dataset):
    """Deterministic token-indexed 2.5D MRI context-restoration dataset.

    Each integer index is a complete sampling token. It deterministically selects
    a contrast/field bucket, volume, slice, crop, corruption, and mask. This makes
    the sample stream exactly recoverable from the training global step.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        stack_depth: int = 5,
        patch_size: int = 256,
        virtual_samples: int = 1_000_000,
        seed: int = 20260714,
        cache_size: int = 16,
        mask_config: dict[str, Any] | None = None,
        corruption_config: dict[str, Any] | None = None,
        foreground_threshold: float = 0.01,
        minimum_foreground_fraction: float = 0.03,
        crop_attempts: int = 8,
    ) -> None:
        if stack_depth < 1 or stack_depth % 2 != 1:
            raise ValueError("stack_depth must be a positive odd integer")
        if patch_size < 8:
            raise ValueError("patch_size must be at least 8")
        if virtual_samples < 1:
            raise ValueError("virtual_samples must be positive")
        self.manifest_path = Path(manifest_path)
        self.rows = self._load_rows(self.manifest_path)
        if not self.rows:
            raise ValueError(f"Manifest has no rows: {self.manifest_path}")
        self.stack_depth = int(stack_depth)
        self.patch_size = int(patch_size)
        self.virtual_samples = int(virtual_samples)
        self.seed = int(seed)
        self.cache_size = max(0, int(cache_size))
        self.mask_config = dict(mask_config or {})
        self.corruption_config = dict(corruption_config or {})
        self.foreground_threshold = float(foreground_threshold)
        self.minimum_foreground_fraction = float(minimum_foreground_fraction)
        self.crop_attempts = max(1, int(crop_attempts))
        self._volume_cache: OrderedDict[str, np.ndarray] = OrderedDict()

        self._buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in self.rows:
            path = str(row.get("path", ""))
            contrast = str(row.get("contrast", ""))
            field = str(row.get("field", ""))
            if not path or contrast not in CONTRAST_TO_ID or field not in FIELD_TO_ID:
                raise ValueError(f"Invalid MCR manifest row: {row}")
            self._buckets.setdefault((contrast, field), []).append(row)
        self.bucket_keys = sorted(self._buckets)

    @staticmethod
    def _load_rows(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Manifest line {line_number} is not an object")
                rows.append(row)
        return rows

    def __len__(self) -> int:
        return self.virtual_samples

    def _load_cached(self, path: str) -> np.ndarray:
        if self.cache_size == 0:
            volume, _, _ = load_volume(path)
            return volume
        if path not in self._volume_cache:
            volume, _, _ = load_volume(path)
            self._volume_cache[path] = volume
            while len(self._volume_cache) > self.cache_size:
                self._volume_cache.popitem(last=False)
        else:
            self._volume_cache.move_to_end(path)
        return self._volume_cache[path]

    def _rng_for_token(self, token: int) -> random.Random:
        mixed = (self.seed * 0x9E3779B185EBCA87 + int(token) * 0xC2B2AE3D27D4EB4F) & ((1 << 64) - 1)
        return random.Random(mixed)

    def _select_row(self, token: int, rng: random.Random) -> tuple[tuple[str, str], dict[str, Any]]:
        bucket = self.bucket_keys[int(token) % len(self.bucket_keys)]
        return bucket, rng.choice(self._buckets[bucket])

    def _crop_stack(self, stack: np.ndarray, rng: random.Random) -> np.ndarray:
        _, height, width = stack.shape
        crop_height = min(height, self.patch_size)
        crop_width = min(width, self.patch_size)
        best: tuple[float, int, int] | None = None
        center = stack[self.stack_depth // 2]
        for _ in range(self.crop_attempts):
            y0 = rng.randrange(0, height - crop_height + 1) if height > crop_height else 0
            x0 = rng.randrange(0, width - crop_width + 1) if width > crop_width else 0
            foreground = float(
                (center[y0 : y0 + crop_height, x0 : x0 + crop_width] > self.foreground_threshold).mean()
            )
            if best is None or foreground > best[0]:
                best = (foreground, y0, x0)
            if foreground >= self.minimum_foreground_fraction:
                break
        assert best is not None
        _, y0, x0 = best
        cropped = stack[:, y0 : y0 + crop_height, x0 : x0 + crop_width]
        if crop_height == self.patch_size and crop_width == self.patch_size:
            return cropped.astype(np.float32, copy=False)
        pad_top = (self.patch_size - crop_height) // 2
        pad_bottom = self.patch_size - crop_height - pad_top
        pad_left = (self.patch_size - crop_width) // 2
        pad_right = self.patch_size - crop_width - pad_left
        return np.pad(
            cropped,
            ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0.0,
        ).astype(np.float32, copy=False)

    @staticmethod
    def _gaussian_blur(stack: np.ndarray, sigma: float) -> np.ndarray:
        if sigma <= 0.0:
            return stack
        radius = max(1, int(math.ceil(3.0 * sigma)))
        coordinates = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (coordinates / sigma).pow(2))
        kernel = kernel / kernel.sum()
        tensor = torch.from_numpy(stack)[None]
        channels = tensor.shape[1]
        horizontal = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
        vertical = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
        tensor = F.pad(tensor, (radius, radius, 0, 0), mode="reflect")
        tensor = F.conv2d(tensor, horizontal, groups=channels)
        tensor = F.pad(tensor, (0, 0, radius, radius), mode="reflect")
        tensor = F.conv2d(tensor, vertical, groups=channels)
        return tensor[0].numpy().astype(np.float32, copy=False)

    def _apply_corruptions(self, stack: np.ndarray, rng: random.Random) -> np.ndarray:
        cfg = self.corruption_config
        output = np.clip(stack.astype(np.float32, copy=True), 0.0, 1.0)

        if rng.random() < float(cfg.get("gamma_prob", 0.0)):
            gamma = rng.uniform(float(cfg.get("gamma_min", 0.8)), float(cfg.get("gamma_max", 1.2)))
            output = np.power(output, gamma).astype(np.float32, copy=False)

        if rng.random() < float(cfg.get("bias_field_prob", 0.0)):
            strength = float(cfg.get("bias_field_strength", 0.2))
            height, width = output.shape[-2:]
            yy = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
            xx = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
            ax = rng.uniform(-strength, strength)
            ay = rng.uniform(-strength, strength)
            axy = rng.uniform(-0.5 * strength, 0.5 * strength)
            field = np.clip(1.0 + ax * xx + ay * yy + axy * xx * yy, 0.5, 1.5)
            output = output * field[None]

        if rng.random() < float(cfg.get("blur_prob", 0.0)):
            sigma = rng.uniform(float(cfg.get("blur_sigma_min", 0.3)), float(cfg.get("blur_sigma_max", 1.2)))
            output = self._gaussian_blur(output.astype(np.float32, copy=False), sigma)

        if rng.random() < float(cfg.get("noise_prob", 0.0)):
            noise_std = rng.uniform(0.0, float(cfg.get("noise_std_max", 0.03)))
            noise_rng = np.random.RandomState(rng.randrange(0, 2**32 - 1))
            noise = noise_rng.normal(0.0, noise_std, size=output.shape).astype(np.float32)
            output = output + noise

        return np.clip(output, 0.0, 1.0).astype(np.float32, copy=False)

    def _sample_mask(self, target: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
        min_ratio = float(self.mask_config.get("ratio_min", 0.4))
        max_ratio = float(self.mask_config.get("ratio_max", 0.6))
        block_sizes = [int(value) for value in self.mask_config.get("block_sizes", [16, 32])]
        if not block_sizes:
            raise ValueError("mask.block_sizes must not be empty")
        foreground = (target[0] > self.foreground_threshold).astype(np.float32)
        attempts = max(1, int(self.mask_config.get("foreground_attempts", 6)))
        minimum_pixels = max(1, int(self.mask_config.get("minimum_foreground_pixels", 64)))
        best_mask: np.ndarray | None = None
        best_loss_mask: np.ndarray | None = None
        best_pixels = -1.0
        for _ in range(attempts):
            ratio = rng.uniform(min_ratio, max_ratio)
            block_size = rng.choice(block_sizes)
            mask = make_block_mask(
                target.shape[-2],
                target.shape[-1],
                ratio=ratio,
                block_size=block_size,
                rng=rng,
            )[None]
            loss_mask = mask * foreground[None]
            pixels = float(loss_mask.sum())
            if pixels > best_pixels:
                best_mask = mask
                best_loss_mask = loss_mask
                best_pixels = pixels
            if pixels >= minimum_pixels:
                break
        assert best_mask is not None and best_loss_mask is not None
        if best_pixels < 1.0:
            best_loss_mask = best_mask.copy()
        return best_mask.astype(np.float32), best_loss_mask.astype(np.float32)

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
        target = clean_stack[self.stack_depth // 2 : self.stack_depth // 2 + 1].copy()
        input_stack = self._apply_corruptions(clean_stack, rng)
        mask, loss_mask = self._sample_mask(target, rng)
        fill_value = float(self.mask_config.get("fill_value", 0.0))
        input_stack = input_stack * (1.0 - mask) + fill_value * mask

        return {
            "input_stack": torch.from_numpy(input_stack.copy()).float(),
            "target": torch.from_numpy(target.copy()).float(),
            "mask": torch.from_numpy(mask.copy()).float(),
            "loss_mask": torch.from_numpy(loss_mask.copy()).float(),
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
                "mask_ratio": float(mask.mean()),
                "masked_foreground_pixels": float(loss_mask.sum()),
            },
        }
