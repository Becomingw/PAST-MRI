#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

from scripts.verify_submission import verify_task3_zip


def main() -> int:
    parser = argparse.ArgumentParser(description="Package and verify PAST-MRI predictions.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-input-root", type=Path, default=None)
    args = parser.parse_args()

    task3_root = args.predictions / "task3"
    if not task3_root.is_dir():
        raise FileNotFoundError(task3_root)
    expected_files = 180
    files = sorted(task3_root.rglob("*.nii.gz"))
    if len(files) != expected_files:
        raise ValueError(f"Expected {expected_files} predictions, found {len(files)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(path, path.relative_to(args.predictions))
        shutil.move(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)

    report = verify_task3_zip(
        args.output,
        reference_input_root=args.reference_input_root,
    )
    if not report.ok:
        raise RuntimeError(f"Submission verification failed: {report}")
    print(f"wrote {args.output} with {report.nifti_files} verified NIfTI files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

