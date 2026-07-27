#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from past_mri.calibration import calibrate_source_like, load_target_stats
from past_mri.config import load_yaml_config
from past_mri.constants import CONTRAST_TO_ID, DEFAULT_Z_END, DEFAULT_Z_START, FIELD_TO_ID, TASK3_PAIRS, TASK3_VALIDATION_IDS
from past_mri.io import crop_padding, pad_to_multiple, save_nifti_like, stack_slices
from past_mri.model_factory import build_cond_naf_unet_from_config
from past_mri.models import CondNAFUNet


def load_config(path: str | Path) -> dict:
    return load_yaml_config(path)


def load_model(config: dict, checkpoint: Path, device: torch.device) -> CondNAFUNet:
    model = build_cond_naf_unet_from_config(config)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model


def output_fusion_weights(channels: int, mode: str) -> np.ndarray:
    if channels < 1 or channels % 2 != 1:
        raise ValueError("output channels must be a positive odd number")
    mode = mode.strip().lower().replace("-", "_")
    if mode in ("center", "center_only"):
        weights = np.zeros((channels,), dtype=np.float32)
        weights[channels // 2] = 1.0
        return weights
    if mode == "uniform":
        return np.ones((channels,), dtype=np.float32)
    if mode in ("center_weighted", "triangular"):
        radius = channels // 2
        return np.asarray([radius + 1 - abs(index - radius) for index in range(channels)], dtype=np.float32)
    raise ValueError(f"unknown output fusion mode: {mode}")


def parse_pair_indices(spec: str, pair_count: int) -> list[int]:
    indices = [int(token.strip()) for token in spec.split(",") if token.strip()]
    if not indices:
        raise ValueError("--pair-indices must contain at least one index")
    if len(indices) != len(set(indices)):
        raise ValueError("--pair-indices contains duplicate indices")
    invalid = [index for index in indices if index < 0 or index >= pair_count]
    if invalid:
        raise ValueError(f"--pair-indices out of range: {invalid}; valid range is 0..{pair_count - 1}")
    return indices


def apply_prediction_slab_override(
    prediction: np.ndarray,
    full_image: nib.spatialimages.SpatialImage,
    override_path: Path,
    prediction_z_start: int,
    override_z_start: int,
    override_z_end: int,
) -> np.ndarray:
    local_start = override_z_start - prediction_z_start
    local_end = override_z_end - prediction_z_start
    if local_start < 0 or local_end > prediction.shape[2] or local_end <= local_start:
        raise ValueError(
            f"override range [{override_z_start}, {override_z_end}) is outside prediction range "
            f"[{prediction_z_start}, {prediction_z_start + prediction.shape[2]})"
        )
    override_image = nib.load(str(override_path))
    expected_shape = (*prediction.shape[:2], override_z_end - override_z_start)
    if override_image.shape != expected_shape:
        raise ValueError(
            f"override shape mismatch for {override_path}: got {override_image.shape}, expected {expected_shape}"
        )
    expected_affine = full_image.slicer[:, :, override_z_start:override_z_end].affine
    if not np.allclose(override_image.affine, expected_affine, atol=1e-4, rtol=0.0):
        raise ValueError(f"override affine mismatch for {override_path}")
    override = np.asarray(override_image.dataobj, dtype=np.float32)
    if not np.isfinite(override).all():
        raise ValueError(f"override contains non-finite values: {override_path}")
    result = prediction.copy()
    result[:, :, local_start:local_end] = override
    return result


@torch.no_grad()
def infer_slab(
    model: CondNAFUNet,
    volume: np.ndarray,
    source_field: str,
    target_field: str,
    contrast: str,
    z_start: int,
    z_end: int,
    device: torch.device,
    base_calibration: str = "source-copy",
    base_calibration_by_contrast: dict[str, str] | None = None,
    target_stats: dict | None = None,
    output_fusion: str = "uniform",
) -> np.ndarray:
    method = (base_calibration_by_contrast or {}).get(contrast, base_calibration)
    output_channels = int(getattr(model, "output_channels", 1))
    weights = output_fusion_weights(output_channels, output_fusion if output_channels > 1 else "center")
    radius = output_channels // 2
    depth = z_end - z_start
    if depth <= 0:
        raise ValueError("z_end must be greater than z_start")
    accumulated = np.zeros((*volume.shape[:2], depth), dtype=np.float32)
    accumulated_weight = np.zeros((depth,), dtype=np.float32)

    for z in range(z_start, z_end):
        stack = stack_slices(volume, z, model.in_channels)
        if output_channels == 1:
            base = stack[model.in_channels // 2 : model.in_channels // 2 + 1]
        else:
            base = stack
        if method not in ("", "none", "source-copy"):
            if target_stats is None:
                raise ValueError("target_stats is required when base_calibration is enabled")
            base = np.stack(
                [
                    calibrate_source_like(
                        source_slice,
                        contrast=contrast,
                        target_field=target_field,
                        target_stats_table=target_stats,
                        method=method,
                    )
                    for source_slice in base
                ],
                axis=0,
            ).astype(np.float32)
        padded, padding = pad_to_multiple(stack, 16)
        padded_base, _ = pad_to_multiple(base, 16)
        tensor = torch.from_numpy(padded)[None].float().to(device)
        base_tensor = torch.from_numpy(padded_base)[None].float().to(device)
        out = model(
            tensor,
            torch.tensor([FIELD_TO_ID[source_field]], device=device),
            torch.tensor([FIELD_TO_ID[target_field]], device=device),
            torch.tensor([CONTRAST_TO_ID[contrast]], device=device),
            base_center=base_tensor,
        ).pred.squeeze(0).cpu().numpy()
        out = crop_padding(out, padding).astype(np.float32)
        if out.shape[0] != output_channels:
            raise RuntimeError(f"model emitted {out.shape[0]} channels; expected {output_channels}")
        for channel, weight in enumerate(weights):
            target_z = z + channel - radius
            if weight <= 0.0 or target_z < z_start or target_z >= z_end:
                continue
            index = target_z - z_start
            accumulated[:, :, index] += float(weight) * out[channel]
            accumulated_weight[index] += float(weight)

    if np.any(accumulated_weight <= 0.0):
        missing = np.flatnonzero(accumulated_weight <= 0.0).tolist()
        raise RuntimeError(f"overlap fusion left output slices uncovered: {missing}")
    return (accumulated / accumulated_weight[None, None, :]).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PAST-MRI multi-field translation inference.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--contrast", default="T2W", choices=["T1W", "T2W", "T2FLAIR"])
    parser.add_argument("--z-start", type=int, default=DEFAULT_Z_START)
    parser.add_argument("--z-end", type=int, default=DEFAULT_Z_END)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-pairs", type=int, default=None)
    parser.add_argument("--pair-offset", type=int, default=0, help="Skip this many Task3 field-pairs before inference.")
    parser.add_argument("--pair-count", type=int, default=None, help="Infer at most this many field-pairs after pair-offset.")
    parser.add_argument(
        "--pair-indices",
        default=None,
        help="Comma-separated zero-based Task3 pair indices. Mutually exclusive with pair slicing options.",
    )
    parser.add_argument(
        "--override-slab-root",
        type=Path,
        default=None,
        help="Optional existing task3 prediction root whose slab values replace the matching full-volume range.",
    )
    parser.add_argument("--override-z-start", type=int, default=DEFAULT_Z_START)
    parser.add_argument("--override-z-end", type=int, default=DEFAULT_Z_END)
    parser.add_argument("--skip-existing", action="store_true", help="Do not rewrite prediction files that already exist.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    base_calib_cfg = cfg.get("base_calibration", {})
    base_calibration = str(base_calib_cfg.get("method", "source-copy"))
    base_calibration_by_contrast = {
        str(k): str(v)
        for k, v in base_calib_cfg.get("methods_by_contrast", {}).items()
    }
    target_stats_path = base_calib_cfg.get("target_stats_path")
    target_stats = load_target_stats(target_stats_path) if target_stats_path else None
    output_fusion = str((cfg.get("inference", {}) or {}).get("output_fusion", "uniform"))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, args.checkpoint, device)
    if args.pair_indices is not None:
        if args.limit_pairs is not None or args.pair_offset != 0 or args.pair_count is not None:
            raise ValueError("--pair-indices is mutually exclusive with --limit-pairs/--pair-offset/--pair-count")
        selected_indices = parse_pair_indices(args.pair_indices, len(TASK3_PAIRS))
        pairs = [TASK3_PAIRS[index] for index in selected_indices]
    else:
        pairs = TASK3_PAIRS[: args.limit_pairs] if args.limit_pairs else TASK3_PAIRS
        if args.pair_offset < 0:
            raise ValueError("--pair-offset must be non-negative")
        if args.pair_count is not None and args.pair_count < 0:
            raise ValueError("--pair-count must be non-negative")
        pairs = pairs[args.pair_offset :]
        if args.pair_count is not None:
            pairs = pairs[: args.pair_count]
    count = 0
    skipped = 0
    for source_field, target_field in pairs:
        pair = f"{source_field}_to_{target_field}"
        for sid in TASK3_VALIDATION_IDS[source_field]:
            out_path = args.out_root / "task3" / args.contrast / pair / "pred" / f"P_{args.contrast}_{target_field}_{sid}.nii.gz"
            if args.skip_existing and out_path.is_file():
                skipped += 1
                continue
            src_path = args.input_root / args.contrast / source_field / f"P_{args.contrast}_{source_field}_{sid}.nii.gz"
            img = nib.load(str(src_path))
            volume = np.clip(np.nan_to_num(img.get_fdata(dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
            prediction = infer_slab(
                model,
                volume,
                source_field,
                target_field,
                args.contrast,
                args.z_start,
                args.z_end,
                device,
                base_calibration=base_calibration,
                base_calibration_by_contrast=base_calibration_by_contrast,
                target_stats=target_stats,
                output_fusion=output_fusion,
            )
            if args.override_slab_root is not None:
                override_path = (
                    args.override_slab_root
                    / "task3"
                    / args.contrast
                    / pair
                    / "pred"
                    / f"P_{args.contrast}_{target_field}_{sid}.nii.gz"
                )
                if not override_path.is_file():
                    raise FileNotFoundError(override_path)
                prediction = apply_prediction_slab_override(
                    prediction,
                    img,
                    override_path,
                    args.z_start,
                    args.override_z_start,
                    args.override_z_end,
                )
            save_nifti_like(
                prediction,
                out_path,
                affine=img.slicer[:, :, args.z_start : args.z_end].affine,
                header=img.header,
            )
            count += 1
    print(f"wrote {args.contrast} predictions: {count}; skipped existing: {skipped}; pair_offset={args.pair_offset}; pair_count={args.pair_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
