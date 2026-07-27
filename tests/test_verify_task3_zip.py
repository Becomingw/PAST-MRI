from __future__ import annotations

import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np

from scripts.verify_submission import verify_task3_zip


def _write_pred(
    root: Path,
    value: float,
    shape=(4, 5, 2),
    *,
    sid: str = "0001",
    affine: np.ndarray | None = None,
) -> Path:
    path = root / "task3" / "T1W" / "0.1T_to_1.5T" / "pred" / f"P_T1W_1.5T_{sid}.nii.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(
            np.full(shape, value, dtype=np.float32),
            np.eye(4) if affine is None else affine,
        ),
        path,
    )
    return path


def _write_reference(root: Path, sid: str, affine: np.ndarray) -> Path:
    path = root / "T1W" / "0.1T" / f"P_T1W_0.1T_{sid}.nii.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.zeros((4, 5, 2), dtype=np.float32), affine), path)
    return path


def _zip_dir(task3_parent: Path, out_zip: Path) -> None:
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted((task3_parent / "task3").rglob("*")):
            zf.write(path, path.relative_to(task3_parent))


def test_verify_task3_zip_accepts_official_layout(tmp_path):
    _write_pred(tmp_path / "work", value=0.25)
    zip_path = tmp_path / "task3.zip"
    _zip_dir(tmp_path / "work", zip_path)

    report = verify_task3_zip(
        zip_path,
        expected_shape=(4, 5, 2),
        contrasts=("T1W",),
        pairs=(("0.1T", "1.5T"),),
        validation_ids={"0.1T": ("0001",)},
    )

    assert report.ok
    assert report.top_prefixes == ["task3"]
    assert report.nifti_files == 1
    assert report.checked_nifti_files == 1


def test_verify_task3_zip_rejects_missing_task3_root(tmp_path):
    root = tmp_path / "badroot"
    pred = root / "T1W" / "0.1T_to_1.5T" / "pred" / "P_T1W_1.5T_0001.nii.gz"
    pred.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.zeros((4, 5, 2), dtype=np.float32), np.eye(4)), pred)
    zip_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(pred, pred.relative_to(root))

    report = verify_task3_zip(
        zip_path,
        expected_shape=(4, 5, 2),
        contrasts=("T1W",),
        pairs=(("0.1T", "1.5T"),),
        validation_ids={"0.1T": ("0001",)},
    )

    assert not report.ok
    assert report.top_prefixes == ["T1W"]
    assert report.missing_files == ["task3/T1W/0.1T_to_1.5T/pred/P_T1W_1.5T_0001.nii.gz"]


def test_verify_task3_zip_rejects_bad_shape_and_range(tmp_path):
    _write_pred(tmp_path / "work", value=1.5, shape=(4, 5, 3))
    zip_path = tmp_path / "task3.zip"
    _zip_dir(tmp_path / "work", zip_path)

    report = verify_task3_zip(
        zip_path,
        expected_shape=(4, 5, 2),
        contrasts=("T1W",),
        pairs=(("0.1T", "1.5T"),),
        validation_ids={"0.1T": ("0001",)},
    )

    assert not report.ok
    assert report.bad_shapes
    assert report.out_of_range_files


def test_verify_task3_zip_checks_reference_affine(tmp_path):
    affine = np.diag([1.1, 1.2, 1.3, 1.0])
    _write_pred(tmp_path / "work", value=0.25, affine=affine)
    reference_root = tmp_path / "reference"
    _write_reference(reference_root, "0001", affine)
    zip_path = tmp_path / "task3.zip"
    _zip_dir(tmp_path / "work", zip_path)

    report = verify_task3_zip(
        zip_path,
        expected_shape=(4, 5, 2),
        reference_input_root=reference_root,
        reference_z_start=0,
        contrasts=("T1W",),
        pairs=(("0.1T", "1.5T"),),
        validation_ids={"0.1T": ("0001",)},
    )

    assert report.ok
    assert report.affine_checked_files == 1


def test_verify_task3_zip_rejects_wrong_affine(tmp_path):
    _write_pred(tmp_path / "work", value=0.25, affine=np.eye(4))
    reference_root = tmp_path / "reference"
    _write_reference(reference_root, "0001", np.diag([2.0, 1.0, 1.0, 1.0]))
    zip_path = tmp_path / "task3.zip"
    _zip_dir(tmp_path / "work", zip_path)

    report = verify_task3_zip(
        zip_path,
        expected_shape=(4, 5, 2),
        reference_input_root=reference_root,
        reference_z_start=0,
        contrasts=("T1W",),
        pairs=(("0.1T", "1.5T"),),
        validation_ids={"0.1T": ("0001",)},
    )

    assert not report.ok
    assert report.bad_affines


def test_verify_task3_zip_rejects_duplicate_prediction_content(tmp_path):
    _write_pred(tmp_path / "work", value=0.25, sid="0001")
    _write_pred(tmp_path / "work", value=0.25, sid="0002")
    zip_path = tmp_path / "task3.zip"
    _zip_dir(tmp_path / "work", zip_path)

    report = verify_task3_zip(
        zip_path,
        expected_shape=(4, 5, 2),
        contrasts=("T1W",),
        pairs=(("0.1T", "1.5T"),),
        validation_ids={"0.1T": ("0001", "0002")},
    )

    assert not report.ok
    assert report.duplicate_content_groups == [[
        "task3/T1W/0.1T_to_1.5T/pred/P_T1W_1.5T_0001.nii.gz",
        "task3/T1W/0.1T_to_1.5T/pred/P_T1W_1.5T_0002.nii.gz",
    ]]


def test_verify_task3_zip_rejects_duplicate_entries(tmp_path):
    pred = _write_pred(tmp_path / "work", value=0.25)
    zip_path = tmp_path / "task3.zip"
    arcname = "task3/T1W/0.1T_to_1.5T/pred/P_T1W_1.5T_0001.nii.gz"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(pred, arcname)
        zf.write(pred, arcname)

    report = verify_task3_zip(
        zip_path,
        expected_shape=(4, 5, 2),
        contrasts=("T1W",),
        pairs=(("0.1T", "1.5T"),),
        validation_ids={"0.1T": ("0001",)},
    )

    assert not report.ok
    assert report.duplicate_entries == [arcname]
