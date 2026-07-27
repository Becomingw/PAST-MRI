#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from past_mri.constants import CONTRASTS, DEFAULT_Z_START, TASK3_PAIRS, TASK3_VALIDATION_IDS  # noqa: E402


@dataclass
class Task3ZipReport:
    zip_path: str
    entries: int
    nifti_files: int
    top_prefixes: list[str]
    missing_files: list[str]
    extra_nifti_files: list[str]
    duplicate_entries: list[str]
    duplicate_content_groups: list[list[str]]
    bad_task3_task3_prefix: int
    seg_entries: int
    zip_integrity: str
    checked_nifti_files: int
    bad_shapes: list[str]
    nonfinite_files: list[str]
    out_of_range_files: list[str]
    affine_checked_files: int
    missing_reference_files: list[str]
    bad_affines: list[str]
    first_entry: str

    @property
    def ok(self) -> bool:
        return (
            self.top_prefixes == ["task3"]
            and not self.missing_files
            and not self.extra_nifti_files
            and not self.duplicate_entries
            and not self.duplicate_content_groups
            and self.bad_task3_task3_prefix == 0
            and self.seg_entries == 0
            and self.zip_integrity == "OK"
            and not self.bad_shapes
            and not self.nonfinite_files
            and not self.out_of_range_files
            and not self.missing_reference_files
            and not self.bad_affines
        )


def expected_task3_nifti_names(
    *,
    contrasts: Sequence[str] = CONTRASTS,
    pairs: Sequence[tuple[str, str]] = TASK3_PAIRS,
    validation_ids: Mapping[str, Sequence[str]] = TASK3_VALIDATION_IDS,
) -> set[str]:
    names: set[str] = set()
    for contrast in contrasts:
        for source_field, target_field in pairs:
            pair = f"{source_field}_to_{target_field}"
            for sid in validation_ids[source_field]:
                names.add(f"task3/{contrast}/{pair}/pred/P_{contrast}_{target_field}_{sid}.nii.gz")
    return names


def _check_nifti_entry(
    zf: zipfile.ZipFile,
    name: str,
    *,
    expected_shape: tuple[int, int, int],
    check_range: bool,
    range_eps: float,
) -> tuple[str | None, str | None, str | None, np.ndarray, str]:
    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tmp:
        tmp.write(zf.read(name))
        tmp.flush()
        img = nib.load(tmp.name)
        data = img.get_fdata(dtype=np.float32)
        affine = np.asarray(img.affine, dtype=np.float64).copy()
    bad_shape = None if data.shape == expected_shape else f"{name}: {tuple(data.shape)}"
    nonfinite = None if np.isfinite(data).all() else name
    out_of_range = None
    if check_range and (float(np.nanmin(data)) < -range_eps or float(np.nanmax(data)) > 1.0 + range_eps):
        out_of_range = f"{name}: {float(np.nanmin(data)):.6g}/{float(np.nanmax(data)):.6g}"
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(data).tobytes())
    digest.update(np.ascontiguousarray(affine).tobytes())
    return bad_shape, nonfinite, out_of_range, affine, digest.hexdigest()


def _reference_input_path(reference_input_root: Path, name: str) -> Path:
    parts = name.split("/")
    if len(parts) != 5 or parts[0] != "task3" or parts[3] != "pred":
        raise ValueError(f"Cannot map Task3 entry to reference input: {name}")
    contrast = parts[1]
    pair = parts[2]
    if "_to_" not in pair:
        raise ValueError(f"Malformed Task3 pair: {pair}")
    source_field, _ = pair.split("_to_", 1)
    filename = parts[4]
    suffix = ".nii.gz"
    if not filename.endswith(suffix):
        raise ValueError(f"Malformed Task3 filename: {filename}")
    sid = filename[: -len(suffix)].rsplit("_", 1)[-1]
    return reference_input_root / contrast / source_field / f"P_{contrast}_{source_field}_{sid}.nii.gz"


def verify_task3_zip(
    zip_path: Path,
    *,
    expected_shape: tuple[int, int, int] = (364, 436, 30),
    check_nifti: bool = True,
    check_range: bool = True,
    check_content_uniqueness: bool = True,
    range_eps: float = 1e-5,
    reference_input_root: Path | None = None,
    reference_z_start: int = DEFAULT_Z_START,
    affine_atol: float = 1e-5,
    affine_rtol: float = 1e-6,
    contrasts: Sequence[str] = CONTRASTS,
    pairs: Sequence[tuple[str, str]] = TASK3_PAIRS,
    validation_ids: Mapping[str, Sequence[str]] = TASK3_VALIDATION_IDS,
) -> Task3ZipReport:
    expected = expected_task3_nifti_names(contrasts=contrasts, pairs=pairs, validation_ids=validation_ids)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        duplicate_entries = sorted(name for name, count in Counter(names).items() if count > 1)
        nifti = sorted(name for name in names if name.endswith(".nii.gz"))
        name_set = set(nifti)
        bad = zf.testzip()
        missing = sorted(expected - name_set)
        extra = sorted(name_set - expected)
        bad_shapes: list[str] = []
        nonfinite_files: list[str] = []
        out_of_range_files: list[str] = []
        content_names: dict[str, list[str]] = defaultdict(list)
        missing_reference_files: list[str] = []
        bad_affines: list[str] = []
        affine_checked = 0
        checked = 0
        if check_nifti:
            for name in sorted(expected & name_set):
                bad_shape, nonfinite, out_of_range, affine, content_digest = _check_nifti_entry(
                    zf,
                    name,
                    expected_shape=expected_shape,
                    check_range=check_range,
                    range_eps=range_eps,
                )
                checked += 1
                if bad_shape:
                    bad_shapes.append(bad_shape)
                if nonfinite:
                    nonfinite_files.append(nonfinite)
                if out_of_range:
                    out_of_range_files.append(out_of_range)
                if check_content_uniqueness:
                    content_names[content_digest].append(name)
                if reference_input_root is not None:
                    reference_path = _reference_input_path(reference_input_root, name)
                    if not reference_path.is_file():
                        missing_reference_files.append(str(reference_path))
                    else:
                        reference_img = nib.load(str(reference_path))
                        reference_z_end = reference_z_start + expected_shape[2]
                        reference_affine = np.asarray(
                            reference_img.slicer[:, :, reference_z_start:reference_z_end].affine,
                            dtype=np.float64,
                        )
                        affine_checked += 1
                        if not np.allclose(affine, reference_affine, atol=affine_atol, rtol=affine_rtol):
                            max_abs = float(np.max(np.abs(affine - reference_affine)))
                            bad_affines.append(f"{name}: reference={reference_path}, max_abs={max_abs:.6g}")
        duplicate_content_groups = sorted(
            sorted(group) for group in content_names.values() if len(group) > 1
        )
        return Task3ZipReport(
            zip_path=str(zip_path),
            entries=len(names),
            nifti_files=len(nifti),
            top_prefixes=sorted({name.split("/")[0] for name in names if name}),
            missing_files=missing,
            extra_nifti_files=extra,
            duplicate_entries=duplicate_entries,
            duplicate_content_groups=duplicate_content_groups,
            bad_task3_task3_prefix=sum(name.startswith("task3/task3/") for name in names),
            seg_entries=sum("/seg/" in name for name in names),
            zip_integrity="OK" if bad is None else bad,
            checked_nifti_files=checked,
            bad_shapes=bad_shapes,
            nonfinite_files=nonfinite_files,
            out_of_range_files=out_of_range_files,
            affine_checked_files=affine_checked,
            missing_reference_files=sorted(missing_reference_files),
            bad_affines=sorted(bad_affines),
            first_entry=names[0] if names else "",
        )


def _parse_shape(text: str) -> tuple[int, int, int]:
    parts = tuple(int(part) for part in text.split(","))
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be formatted as X,Y,Z")
    return parts


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a Task3 submission zip uses the official task3/... layout.")
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("--expected-shape", type=_parse_shape, default=(364, 436, 30))
    parser.add_argument("--skip-nifti", action="store_true", help="only validate zip layout and names")
    parser.add_argument("--skip-range", action="store_true", help="do not enforce [0, 1] intensity range")
    parser.add_argument("--skip-content-uniqueness", action="store_true")
    parser.add_argument("--reference-input-root", type=Path, default=None)
    parser.add_argument("--reference-z-start", type=int, default=DEFAULT_Z_START)
    parser.add_argument("--affine-atol", type=float, default=1e-5)
    parser.add_argument("--affine-rtol", type=float, default=1e-6)
    parser.add_argument("--range-eps", type=float, default=1e-5)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args()

    report = verify_task3_zip(
        args.zip_path,
        expected_shape=args.expected_shape,
        check_nifti=not args.skip_nifti,
        check_range=not args.skip_range,
        check_content_uniqueness=not args.skip_content_uniqueness,
        range_eps=args.range_eps,
        reference_input_root=args.reference_input_root,
        reference_z_start=args.reference_z_start,
        affine_atol=args.affine_atol,
        affine_rtol=args.affine_rtol,
    )
    payload = asdict(report) | {"ok": report.ok}
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
