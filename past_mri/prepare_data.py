from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from past_mri.calibration import DEFAULT_PERCENTILES
from past_mri.constants import CONTRASTS, FIELDS, TASK3_PAIRS


SUBJECT_PATTERN = re.compile(r"_(\d{4})\.nii\.gz$")


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def subject_id(path: Path) -> str:
    match = SUBJECT_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse four-digit subject id from {path}")
    return match.group(1)


def load_clipped(path: Path) -> np.ndarray:
    image = nib.load(str(path))
    data = image.get_fdata(dtype=np.float32)
    return np.clip(
        np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0),
        0.0,
        1.0,
    ).astype(np.float32)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def stable_score(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def split_unpaired_rows(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    val_fraction: float,
    min_val_per_bucket: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    subjects = sorted({str(row["subject_id"]) for row in rows})
    if len(subjects) < 2:
        raise RuntimeError("At least two subjects are required for a subject-disjoint split")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between zero and one")
    ordered_subjects = sorted(subjects, key=lambda item: stable_score(item, seed))
    val_count = min(max(round(len(subjects) * val_fraction), 1), len(subjects) - 1)
    val_subjects = set(ordered_subjects[:val_count])
    for contrast in CONTRASTS:
        for field in FIELDS:
            bucket = sorted(
                {
                    str(row["subject_id"])
                    for row in rows
                    if row["contrast"] == contrast and row["field"] == field
                },
                key=lambda item: stable_score(item, seed),
            )
            selected = sum(item in val_subjects for item in bucket)
            for item in bucket:
                if selected >= min_val_per_bucket:
                    break
                if item not in val_subjects:
                    val_subjects.add(item)
                    selected += 1
    if len(val_subjects) == len(subjects):
        bucket_subjects = {
            (contrast, field): {
                str(row["subject_id"])
                for row in rows
                if row["contrast"] == contrast and row["field"] == field
            }
            for contrast in CONTRASTS
            for field in FIELDS
        }
        removable = [
            subject
            for subject in reversed(ordered_subjects)
            if all(
                len(subjects_in_bucket & (val_subjects - {subject})) >= min_val_per_bucket
                for subjects_in_bucket in bucket_subjects.values()
            )
        ]
        if not removable:
            raise RuntimeError(
                "Per-bucket validation requirements leave no subject for training"
            )
        val_subjects.remove(removable[0])
    train = [row for row in rows if str(row["subject_id"]) not in val_subjects]
    val = [row for row in rows if str(row["subject_id"]) in val_subjects]
    overlap = {str(row["subject_id"]) for row in train} & {str(row["subject_id"]) for row in val}
    if not train or not val or overlap:
        raise RuntimeError(f"Invalid subject split: train={len(train)} val={len(val)} overlap={sorted(overlap)}")
    return train, val


def cache_unpaired(
    input_root: Path,
    output_root: Path,
    *,
    z_start: int,
    z_end: int,
    stack_margin: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    slab_start = z_start - stack_margin
    slab_end = z_end + stack_margin
    if slab_start < 0 or slab_end <= slab_start:
        raise ValueError("Invalid unpaired slab range")
    for contrast in CONTRASTS:
        for field in FIELDS:
            paths = sorted((input_root / contrast / field).glob("*.nii.gz"))
            if not paths:
                raise FileNotFoundError(f"No NIfTI files under {input_root / contrast / field}")
            for source in paths:
                sid = subject_id(source)
                volume = load_clipped(source)
                if volume.shape[2] < slab_end:
                    raise ValueError(f"{source} has only {volume.shape[2]} slices; require {slab_end}")
                destination = (
                    output_root
                    / "unpaired_slabs"
                    / contrast
                    / field
                    / f"R_{contrast}_{field}_{sid}_z{slab_start}_{slab_end}.npy"
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                slab = volume[:, :, slab_start:slab_end]
                np.save(destination, slab)
                rows.append(
                    {
                        "path": str(destination.resolve()),
                        "subject_id": sid,
                        "contrast": contrast,
                        "field": field,
                        "z_start": 0,
                        "z_end": int(slab.shape[2]),
                        "shape": list(slab.shape),
                        "dtype": str(slab.dtype),
                        "bytes": int(destination.stat().st_size),
                    }
                )
    return rows


def build_target_stats(
    rows: list[dict[str, Any]],
    *,
    max_voxels_per_file: int,
    mask_eps: float,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["contrast"]), str(row["field"])), []).append(row)
    stats: dict[str, dict[str, Any]] = {}
    for contrast in CONTRASTS:
        stats[contrast] = {}
        for field in FIELDS:
            samples: list[np.ndarray] = []
            bucket_rows = grouped.get((contrast, field), [])
            if not bucket_rows:
                raise ValueError(f"No rows for target statistics: {contrast}/{field}")
            for row in bucket_rows:
                data = np.load(row["path"], mmap_mode="r")
                values = np.asarray(data[data > mask_eps], dtype=np.float32)
                if values.size > max_voxels_per_file:
                    rng = np.random.default_rng(stable_score(str(row["path"]), 20260718))
                    indices = rng.choice(values.size, size=max_voxels_per_file, replace=False)
                    values = values[indices]
                samples.append(values)
            values = np.concatenate(samples)
            quantiles = np.percentile(values, DEFAULT_PERCENTILES)
            stats[contrast][field] = {
                "files": len(bucket_rows),
                "sampled_voxels": int(values.size),
                "percentiles": {
                    f"{percentile:g}": float(value)
                    for percentile, value in zip(DEFAULT_PERCENTILES, quantiles)
                },
            }
    return {
        "mask_eps": mask_eps,
        "max_voxels_per_file": max_voxels_per_file,
        "percentiles": list(DEFAULT_PERCENTILES),
        "stats": stats,
    }


def cache_paired(
    input_root: Path,
    output_root: Path,
    *,
    subjects: list[str],
    z_start: int,
    z_end: int,
    stack_margin: int,
    repeat: int,
) -> list[dict[str, Any]]:
    if repeat < 1:
        raise ValueError("repeat must be positive")
    source_start = z_start - stack_margin
    source_end = z_end + stack_margin
    local_start = z_start - source_start
    base_rows: list[dict[str, Any]] = []
    source_paths: dict[tuple[str, str, str], Path] = {}
    target_paths: dict[tuple[str, str, str], Path] = {}

    for contrast in CONTRASTS:
        for sid in subjects:
            for field in FIELDS:
                source = input_root / contrast / field / f"P_{contrast}_{field}_{sid}.nii.gz"
                if not source.is_file():
                    raise FileNotFoundError(source)
                volume = load_clipped(source)
                if source_start < 0 or volume.shape[2] < source_end:
                    raise ValueError(f"{source} cannot provide z=[{source_start},{source_end})")
                source_out = (
                    output_root
                    / "paired_slabs"
                    / "source"
                    / contrast
                    / field
                    / f"P_{contrast}_{field}_{sid}_z{source_start}_{source_end}.npy"
                )
                target_out = (
                    output_root
                    / "paired_slabs"
                    / "target"
                    / contrast
                    / field
                    / f"P_{contrast}_{field}_{sid}_z{z_start}_{z_end}.npy"
                )
                source_out.parent.mkdir(parents=True, exist_ok=True)
                target_out.parent.mkdir(parents=True, exist_ok=True)
                np.save(source_out, volume[:, :, source_start:source_end])
                np.save(target_out, volume[:, :, z_start:z_end])
                source_paths[(contrast, field, sid)] = source_out
                target_paths[(contrast, field, sid)] = target_out

            for source_field, target_field in TASK3_PAIRS:
                base_rows.append(
                    {
                        "contrast": contrast,
                        "subject_id": sid,
                        "source_field": source_field,
                        "target_field": target_field,
                        "source_path": str(source_paths[(contrast, source_field, sid)].resolve()),
                        "target_path": str(target_paths[(contrast, target_field, sid)].resolve()),
                        "z_start": local_start,
                        "z_end": local_start + (z_end - z_start),
                        "source_z_start": source_start,
                        "source_z_end": source_end,
                        "target_z_start": z_start,
                        "target_z_end": z_end,
                        "supervision": "paired_gt",
                    }
                )
    rows: list[dict[str, Any]] = []
    for repeat_index in range(repeat):
        for row in base_rows:
            rows.append({**row, "repeat_index": repeat_index})
    return rows


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    manifests = output_root / "manifests"
    unpaired_rows = cache_unpaired(
        args.unpaired_root,
        output_root,
        z_start=args.unpaired_z_start,
        z_end=args.unpaired_z_end,
        stack_margin=args.stack_margin,
    )
    stage1_train, stage1_val = split_unpaired_rows(
        unpaired_rows,
        seed=args.split_seed,
        val_fraction=args.val_fraction,
        min_val_per_bucket=args.min_val_per_bucket,
    )
    write_jsonl(manifests / "stage1_train.jsonl", stage1_train)
    write_jsonl(manifests / "stage1_val.jsonl", stage1_val)

    target_stats = build_target_stats(
        unpaired_rows,
        max_voxels_per_file=args.max_voxels_per_file,
        mask_eps=args.mask_eps,
    )
    (output_root / "target_stats.json").write_text(json.dumps(target_stats, indent=2, sort_keys=True) + "\n")

    stage2_rows = cache_paired(
        args.paired_root,
        output_root,
        subjects=parse_csv(args.paired_subjects),
        z_start=args.paired_z_start,
        z_end=args.paired_z_end,
        stack_margin=args.stack_margin,
        repeat=args.paired_repeat,
    )
    write_jsonl(manifests / "stage2_train.jsonl", stage2_rows)

    train_subjects = {str(row["subject_id"]) for row in stage1_train}
    val_subjects = {str(row["subject_id"]) for row in stage1_val}
    summary = {
        "output_root": str(output_root),
        "stage1_train_rows": len(stage1_train),
        "stage1_val_rows": len(stage1_val),
        "stage1_subject_overlap": sorted(train_subjects & val_subjects),
        "stage1_bucket_counts": dict(Counter(f"{row['contrast']}/{row['field']}" for row in unpaired_rows)),
        "stage2_rows": len(stage2_rows),
        "stage2_subjects": parse_csv(args.paired_subjects),
        "stage2_repeat": args.paired_repeat,
        "target_stats": str(output_root / "target_stats.json"),
    }
    (output_root / "prepare_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare unpaired Stage1 and paired Stage2 PAST-MRI data.")
    parser.add_argument("--unpaired-root", type=Path, required=True)
    parser.add_argument("--paired-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--paired-subjects", default="0006,0007,0009")
    parser.add_argument("--unpaired-z-start", type=int, default=150)
    parser.add_argument("--unpaired-z-end", type=int, default=180)
    parser.add_argument("--paired-z-start", type=int, default=56)
    parser.add_argument("--paired-z-end", type=int, default=281)
    parser.add_argument("--stack-margin", type=int, default=2)
    parser.add_argument("--paired-repeat", type=int, default=48)
    parser.add_argument("--split-seed", type=int, default=20260714)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--min-val-per-bucket", type=int, default=3)
    parser.add_argument("--max-voxels-per-file", type=int, default=20000)
    parser.add_argument("--mask-eps", type=float, default=1e-4)
    args = parser.parse_args()
    print(json.dumps(prepare(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
