from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np


def load_volume(path: str | Path) -> tuple[np.ndarray, np.ndarray | None, nib.Nifti1Header | None]:
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32), None, None
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(data, 0.0, 1.0).astype(np.float32), img.affine, img.header


def save_nifti_like(
    data: np.ndarray,
    path: str | Path,
    affine: np.ndarray | None = None,
    header: nib.Nifti1Header | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if affine is None:
        affine = np.eye(4, dtype=np.float32)
    out = nib.Nifti1Image(np.clip(data, 0.0, 1.0).astype(np.float32), affine, header=header)
    nib.save(out, str(path))


def save_array(path: str | Path, data: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.clip(data, 0.0, 1.0).astype(np.float32))


def stack_slices(volume: np.ndarray, center_z: int, depth: int) -> np.ndarray:
    radius = depth // 2
    slices = []
    max_z = volume.shape[2] - 1
    for dz in range(-radius, radius + 1):
        z = min(max(center_z + dz, 0), max_z)
        slices.append(volume[:, :, z])
    return np.stack(slices, axis=0).astype(np.float32)


def pad_to_multiple(array: np.ndarray, multiple: int = 16) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = array.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return array, (0, 0)
    padded = np.pad(array, [(0, 0)] * (array.ndim - 2) + [(0, pad_h), (0, pad_w)], mode="edge")
    return padded, (pad_h, pad_w)


def crop_padding(array: np.ndarray, padding: tuple[int, int]) -> np.ndarray:
    pad_h, pad_w = padding
    h_end = None if pad_h == 0 else -pad_h
    w_end = None if pad_w == 0 else -pad_w
    return array[..., :h_end, :w_end]
