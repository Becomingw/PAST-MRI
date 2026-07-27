from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np

from past_mri.constants import CONTRASTS, FIELDS
from past_mri.prepare_data import (
    build_target_stats,
    cache_paired,
    cache_unpaired,
    split_unpaired_rows,
)


def save_volume(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((8, 8, 12), value, dtype=np.float32)
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)


def test_prepare_unpaired_and_paired_manifests(tmp_path: Path) -> None:
    unpaired_root = tmp_path / "Training_retrospective"
    paired_root = tmp_path / "Training_prospective"
    for contrast_index, contrast in enumerate(CONTRASTS):
        for field_index, field in enumerate(FIELDS):
            save_volume(
                unpaired_root / contrast / field / f"R_{contrast}_{field}_0001.nii.gz",
                0.1 + 0.01 * contrast_index + 0.001 * field_index,
            )
            save_volume(
                unpaired_root / contrast / field / f"R_{contrast}_{field}_0002.nii.gz",
                0.2 + 0.01 * contrast_index + 0.001 * field_index,
            )
            save_volume(
                paired_root / contrast / field / f"P_{contrast}_{field}_0006.nii.gz",
                0.3 + 0.01 * contrast_index + 0.001 * field_index,
            )

    processed = tmp_path / "processed"
    unpaired_rows = cache_unpaired(
        unpaired_root,
        processed,
        z_start=2,
        z_end=8,
        stack_margin=1,
    )
    train, val = split_unpaired_rows(
        unpaired_rows,
        seed=7,
        val_fraction=0.5,
        min_val_per_bucket=1,
    )
    assert len(unpaired_rows) == 30
    assert {row["subject_id"] for row in train}.isdisjoint({row["subject_id"] for row in val})

    target_stats = build_target_stats(unpaired_rows, max_voxels_per_file=16, mask_eps=1e-4)
    assert set(target_stats["stats"]) == set(CONTRASTS)
    assert set(target_stats["stats"]["T1W"]) == set(FIELDS)
    json.dumps(target_stats)

    paired_rows = cache_paired(
        paired_root,
        processed,
        subjects=["0006"],
        z_start=2,
        z_end=9,
        stack_margin=1,
        repeat=2,
    )
    assert len(paired_rows) == len(CONTRASTS) * 20 * 2
    assert all(Path(row["source_path"]).is_file() for row in paired_rows)
    assert all(Path(row["target_path"]).is_file() for row in paired_rows)
