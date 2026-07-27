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

from past_mri.calibration import calibrate_source_like, load_target_stats
from past_mri.constants import CONTRAST_TO_ID, FIELD_TO_ID
from past_mri.io import load_volume, stack_slices
from past_mri.method_features import METHOD_FEATURE_NAMES, method_feature_vector


METHOD_BANK_FAMILY_TO_ID = {
    "source-copy": 0,
    "paired-quantile": 1,
    "targetcohort": 2,
    "source-lfbias": 3,
    "other": 4,
}

EXECUTOR_FAMILY_TARGET_NAMES: dict[str, list[str]] = {
    "family4": ["source-copy", "paired-quantile", "targetcohort", "source-lfbias"],
    "binary_source_vs_calib": ["source_like", "calibration_like"],
    "ternary_source_quantile_cohort": ["source_like", "paired_quantile", "cohort_like"],
}

EXECUTOR_FAMILY_TARGET_MAP: dict[str, list[int]] = {
    "family4": [0, 1, 2, 3],
    "binary_source_vs_calib": [0, 1, 1, 0],
    "ternary_source_quantile_cohort": [0, 1, 2, 0],
}


def method_bank_family_id(method: str) -> int:
    if method.startswith("source-copy"):
        return METHOD_BANK_FAMILY_TO_ID["source-copy"]
    if method.startswith("paired-quantile"):
        return METHOD_BANK_FAMILY_TO_ID["paired-quantile"]
    if method.startswith("targetcohort"):
        return METHOD_BANK_FAMILY_TO_ID["targetcohort"]
    if method.startswith("source-lfbias"):
        return METHOD_BANK_FAMILY_TO_ID["source-lfbias"]
    return METHOD_BANK_FAMILY_TO_ID["other"]


def remap_executor_family_targets(
    soft_targets: np.ndarray,
    hard_target_id: int,
    target_mode: str,
) -> tuple[np.ndarray, int]:
    if target_mode not in EXECUTOR_FAMILY_TARGET_NAMES:
        raise ValueError(f"Unknown executor_family_target_mode: {target_mode}")
    soft_targets = np.asarray(soft_targets, dtype=np.float32)
    if target_mode == "family4":
        return soft_targets, int(hard_target_id)
    if soft_targets.shape != (4,):
        raise ValueError(
            "coarse executor family target modes require 4-way input soft targets, "
            f"got shape {soft_targets.shape}"
        )
    mapping = EXECUTOR_FAMILY_TARGET_MAP[target_mode]
    out = np.zeros((len(EXECUTOR_FAMILY_TARGET_NAMES[target_mode]),), dtype=np.float32)
    for src_idx, dst_idx in enumerate(mapping):
        out[dst_idx] += float(soft_targets[src_idx])
    if hard_target_id >= 0:
        if hard_target_id >= len(mapping):
            raise ValueError(f"executor family hard target {hard_target_id} is outside 4-way family range")
        hard_target_id = mapping[int(hard_target_id)]
    return out, int(hard_target_id)


class PairedTranslationDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        stack_depth: int = 5,
        output_depth: int = 1,
        patch_size: int | None = 256,
        identity_prob: float = 0.2,
        pair_balanced: bool = True,
        cache_size: int = 16,
        mmap_npy: bool = False,
        seed: int = 20260614,
        base_calibration: str = "source-copy",
        base_calibration_by_contrast: dict[str, str] | None = None,
        target_stats_path: str | Path | None = None,
        load_method_bank_candidates: bool = True,
        executor_family_target_mode: str = "family4",
        augmentation_config: dict[str, Any] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.rows = self._load_rows(self.manifest_path)
        if not self.rows:
            raise ValueError(f"Manifest has no rows: {self.manifest_path}")
        self.stack_depth = stack_depth
        self.output_depth = int(output_depth)
        if self.output_depth < 1 or self.output_depth % 2 != 1:
            raise ValueError("output_depth must be a positive odd number")
        if self.output_depth > self.stack_depth:
            raise ValueError("output_depth must not exceed stack_depth")
        self.patch_size = patch_size
        self.identity_prob = identity_prob
        self.pair_balanced = pair_balanced
        self.cache_size = max(0, cache_size)
        self.mmap_npy = bool(mmap_npy)
        self.rng = random.Random(seed)
        self._volume_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.base_calibration = base_calibration
        self.base_calibration_by_contrast = dict(base_calibration_by_contrast or {})
        self.target_stats = load_target_stats(target_stats_path) if target_stats_path else None
        self.load_method_bank_candidates = bool(load_method_bank_candidates)
        self.augmentation_config = dict(augmentation_config or {})
        self.augmentation_enabled = bool(self.augmentation_config.get("enabled", False))
        if executor_family_target_mode not in EXECUTOR_FAMILY_TARGET_NAMES:
            raise ValueError(f"Unknown executor_family_target_mode: {executor_family_target_mode}")
        self.executor_family_target_mode = executor_family_target_mode
        enabled_methods = [self.base_calibration, *self.base_calibration_by_contrast.values()]
        if any(method not in ("", "none", "source-copy") for method in enabled_methods) and self.target_stats is None:
            raise ValueError("target_stats_path is required when base_calibration is enabled")

        self._pairs: dict[tuple[str, str], list[int]] = {}
        self._label_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.method_bank_candidate_count = max(
            (
                max(
                    len(row.get("method_bank_candidate_paths", []) or []),
                    len(row.get("method_bank_metric_soft_targets", []) or []),
                    len(row.get("method_bank_candidate_methods", []) or []),
                )
                for row in self.rows
            ),
            default=0,
        )
        if self.executor_family_target_mode == "family4":
            self.executor_family_count = max(
                (len(row.get("executor_family_soft_targets", []) or []) for row in self.rows),
                default=0,
            )
        else:
            self.executor_family_count = len(EXECUTOR_FAMILY_TARGET_NAMES[self.executor_family_target_mode])
        self.executor_param_count = max(
            (
                max(
                    len(row.get("executor_param_soft_targets", []) or []),
                    len(row.get("executor_param_hard_targets", []) or []),
                )
                for row in self.rows
            ),
            default=0,
        )
        for idx, row in enumerate(self.rows):
            key = (row["source_field"], row["target_field"])
            self._pairs.setdefault(key, []).append(idx)
        self._pair_keys = sorted(self._pairs)

    @staticmethod
    def _load_rows(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def _load_cached(self, path: str) -> np.ndarray:
        def load() -> np.ndarray:
            if self.mmap_npy and Path(path).suffix == ".npy":
                volume = np.load(path, mmap_mode="r")
                if volume.dtype != np.float32:
                    volume = volume.astype(np.float32)
                return volume
            volume, _, _ = load_volume(path)
            return volume

        if self.cache_size == 0:
            return load()
        if path not in self._volume_cache:
            self._volume_cache[path] = load()
            while len(self._volume_cache) > self.cache_size:
                self._volume_cache.popitem(last=False)
        else:
            self._volume_cache.move_to_end(path)
        return self._volume_cache[path]

    def _load_label_cached(self, path: str) -> np.ndarray:
        if self.cache_size == 0:
            return np.load(path).astype(np.int64, copy=False)
        if path not in self._label_cache:
            label = np.load(path).astype(np.int64, copy=False)
            self._label_cache[path] = label
            while len(self._label_cache) > self.cache_size:
                self._label_cache.popitem(last=False)
        else:
            self._label_cache.move_to_end(path)
        return self._label_cache[path]

    def _select_row(self, index: int) -> dict[str, Any]:
        if self.pair_balanced:
            pair = self._pair_keys[index % len(self._pair_keys)]
            row_idx = self.rng.choice(self._pairs[pair])
            return self.rows[row_idx]
        return self.rows[index % len(self.rows)]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._select_row(index)
        source = self._load_cached(row["source_path"])
        z_start = int(row.get("z_start", 150))
        z_end = int(row.get("z_end", source.shape[2]))
        z = self.rng.randrange(z_start, z_end)
        source_z = min(max(z, 0), source.shape[2] - 1)
        source_stack = stack_slices(source, source_z, self.stack_depth)
        source_center = source_stack[self.stack_depth // 2 : self.stack_depth // 2 + 1]
        output_radius = self.output_depth // 2
        stack_center = self.stack_depth // 2
        source_target = source_stack[stack_center - output_radius : stack_center + output_radius + 1]

        is_identity = self.rng.random() < self.identity_prob
        control_features = np.zeros(4, dtype=np.float32)
        has_control_features = False
        method_bank_target_index = -1
        has_method_bank_target = False
        method_bank_soft_targets = np.zeros((self.method_bank_candidate_count,), dtype=np.float32)
        method_bank_candidate_family_ids = np.full((self.method_bank_candidate_count,), -1, dtype=np.int64)
        has_method_bank_soft_targets = False
        executor_family_soft_targets = np.zeros((self.executor_family_count,), dtype=np.float32)
        executor_param_soft_targets = np.zeros((self.executor_param_count,), dtype=np.float32)
        executor_param_hard_targets = np.zeros((self.executor_param_count,), dtype=np.float32)
        executor_family_target_id = -1
        has_executor_targets = False
        method_bank_candidates = np.zeros(
            (self.method_bank_candidate_count, self.output_depth, source.shape[0], source.shape[1]),
            dtype=np.float32,
        )
        has_method_bank_candidates = False
        residual_target = np.zeros_like(source_target, dtype=np.float32)
        has_residual_target = False
        if is_identity:
            target = source_target
            target_field = row["source_field"]
            policy_features = np.zeros(len(METHOD_FEATURE_NAMES), dtype=np.float32)
            has_policy_features = False
        else:
            # Accept the legacy key so manifests generated before the public
            # release remain usable; both keys identify paired target slabs.
            target_path = row.get("target_path", row.get("teacher_pred_path"))
            if not target_path:
                raise KeyError("target_path missing for paired translation sample")
            target_volume = self._load_cached(target_path)
            target_z = min(max(z - z_start, 0), target_volume.shape[2] - 1)
            target = stack_slices(target_volume, target_z, self.output_depth).astype(np.float32)
            target_field = row["target_field"]
            anchor_path = row.get("anchor_pred_path")
            if anchor_path:
                anchor = self._load_cached(str(anchor_path))
                anchor_z = min(max(z - z_start, 0), anchor.shape[2] - 1)
                anchor_slice = stack_slices(anchor, anchor_z, self.output_depth).astype(np.float32)
                if anchor_slice.shape != target.shape:
                    raise ValueError(f"anchor/target shape mismatch: {anchor_slice.shape} vs {target.shape}")
                residual_target = (target - anchor_slice).astype(np.float32, copy=False)
                has_residual_target = True
            control_feature_override = row.get("control_feature_override")
            if control_feature_override is not None:
                control_features = np.asarray(control_feature_override, dtype=np.float32)
                if control_features.shape != (4,):
                    raise ValueError(
                        "control_feature_override must have 4 values "
                        f"(map_beta, lowfreq_alpha, blur_strength, source_floor), got shape {control_features.shape}"
                    )
                has_control_features = True
            policy_feature_override = row.get("policy_feature_override")
            if policy_feature_override is not None:
                policy_features = np.asarray(policy_feature_override, dtype=np.float32)
                if policy_features.shape != (len(METHOD_FEATURE_NAMES),):
                    raise ValueError(
                        "policy_feature_override must have "
                        f"{len(METHOD_FEATURE_NAMES)} values, got shape {policy_features.shape}"
                    )
                has_policy_features = True
            elif shared_policy_method := row.get("shared_policy_method"):
                policy_features = method_feature_vector(str(shared_policy_method))
                has_policy_features = True
            else:
                policy_features = np.zeros(len(METHOD_FEATURE_NAMES), dtype=np.float32)
                has_policy_features = False
            if "method_bank_target_index" in row:
                method_bank_target_index = int(row["method_bank_target_index"])
                has_method_bank_target = method_bank_target_index >= 0
            metric_soft_targets = row.get("method_bank_metric_soft_targets")
            if metric_soft_targets is not None:
                method_bank_soft_targets = np.asarray(metric_soft_targets, dtype=np.float32)
                if method_bank_soft_targets.shape != (self.method_bank_candidate_count,):
                    raise ValueError(
                        "method_bank_metric_soft_targets length mismatch: "
                        f"got {method_bank_soft_targets.shape}, expected ({self.method_bank_candidate_count},)"
                    )
                total = float(method_bank_soft_targets.sum())
                if total <= 0.0:
                    raise ValueError("method_bank_metric_soft_targets must have positive sum")
                method_bank_soft_targets = (method_bank_soft_targets / total).astype(np.float32, copy=False)
                has_method_bank_soft_targets = True
            executor_family_soft_override = row.get("executor_family_soft_targets")
            if executor_family_soft_override is not None:
                raw_executor_family_soft_targets = np.asarray(executor_family_soft_override, dtype=np.float32)
                if self.executor_family_target_mode == "family4" and raw_executor_family_soft_targets.shape != (
                    self.executor_family_count,
                ):
                    raise ValueError(
                        "executor_family_soft_targets length mismatch: "
                        f"got {raw_executor_family_soft_targets.shape}, expected ({self.executor_family_count},)"
                    )
                executor_family_target_id = int(row.get("metric_target_executor_family_id", -1))
                if executor_family_target_id < 0:
                    executor_family_target_id = int(np.argmax(raw_executor_family_soft_targets))
                executor_family_soft_targets, executor_family_target_id = remap_executor_family_targets(
                    raw_executor_family_soft_targets,
                    executor_family_target_id,
                    self.executor_family_target_mode,
                )
                if executor_family_soft_targets.shape != (self.executor_family_count,):
                    raise ValueError(
                        "remapped executor_family_soft_targets length mismatch: "
                        f"got {executor_family_soft_targets.shape}, expected ({self.executor_family_count},)"
                    )
                total = float(executor_family_soft_targets.sum())
                if total <= 0.0:
                    raise ValueError("executor_family_soft_targets must have positive sum")
                executor_family_soft_targets = (executor_family_soft_targets / total).astype(np.float32, copy=False)
                has_executor_targets = True
            executor_param_soft_override = row.get("executor_param_soft_targets")
            if executor_param_soft_override is not None:
                executor_param_soft_targets = np.asarray(executor_param_soft_override, dtype=np.float32)
                if executor_param_soft_targets.shape != (self.executor_param_count,):
                    raise ValueError(
                        "executor_param_soft_targets length mismatch: "
                        f"got {executor_param_soft_targets.shape}, expected ({self.executor_param_count},)"
                    )
                has_executor_targets = True
            executor_param_hard_override = row.get("executor_param_hard_targets")
            if executor_param_hard_override is not None:
                executor_param_hard_targets = np.asarray(executor_param_hard_override, dtype=np.float32)
                if executor_param_hard_targets.shape != (self.executor_param_count,):
                    raise ValueError(
                        "executor_param_hard_targets length mismatch: "
                        f"got {executor_param_hard_targets.shape}, expected ({self.executor_param_count},)"
                    )
                has_executor_targets = True
            candidate_paths = list(row.get("method_bank_candidate_paths", []) or [])
            candidate_methods = list(row.get("method_bank_candidate_methods", []) or [])
            if candidate_methods:
                if len(candidate_methods) != self.method_bank_candidate_count:
                    raise ValueError(
                        "method_bank_candidate_methods length mismatch: "
                        f"got {len(candidate_methods)}, expected {self.method_bank_candidate_count}"
                    )
                method_bank_candidate_family_ids = np.asarray(
                    [method_bank_family_id(str(method)) for method in candidate_methods],
                    dtype=np.int64,
                )
            if candidate_paths and self.load_method_bank_candidates:
                if len(candidate_paths) != self.method_bank_candidate_count:
                    raise ValueError(
                        "method_bank_candidate_paths length mismatch: "
                        f"got {len(candidate_paths)}, expected {self.method_bank_candidate_count}"
                    )
                candidate_slices: list[np.ndarray] = []
                for candidate_path in candidate_paths:
                    candidate = self._load_cached(str(candidate_path))
                    candidate_z = min(max(z - z_start, 0), candidate.shape[2] - 1)
                    candidate_slice = stack_slices(candidate, candidate_z, self.output_depth).astype(np.float32)
                    if candidate_slice.shape != target.shape:
                        raise ValueError(
                            "method-bank candidate/target shape mismatch: "
                            f"{candidate_slice.shape} vs {target.shape}"
                        )
                    candidate_slices.append(candidate_slice)
                method_bank_candidates = np.stack(candidate_slices, axis=0).astype(np.float32, copy=False)
                has_method_bank_candidates = True

        base_center = self._make_base_center(source_target, row, target_field, is_identity)
        tissue_target = np.zeros((1, target.shape[1], target.shape[2]), dtype=np.int64)
        has_tissue_target = False
        tissue_path = row.get("synthseg_region_path") or row.get("synthseg_tissue_teacher_path")
        if tissue_path:
            tissue = self._load_label_cached(str(tissue_path))
            tissue_z = min(max(z - z_start, 0), tissue.shape[2] - 1)
            tissue_target = tissue[:, :, tissue_z][None, ...].astype(np.int64, copy=False)
            if tissue_target.shape[1:] != target.shape[1:]:
                raise ValueError(f"tissue/target shape mismatch: {tissue_target.shape[1:]} vs {target.shape[1:]}")
            has_tissue_target = True

        if self.augmentation_enabled:
            (
                source_stack,
                source_center,
                target,
                base_center,
                tissue_target,
                residual_target,
                method_bank_candidates,
            ) = self._apply_augmentation(
                source_stack,
                source_center,
                target,
                base_center,
                tissue_target,
                residual_target,
                method_bank_candidates,
                row=row,
                target_field=target_field,
                is_identity=is_identity,
            )

        (
            source_stack,
            source_center,
            target,
            base_center,
            tissue_target,
            residual_target,
            method_bank_candidates,
        ) = self._crop_arrays(
            source_stack,
            source_center,
            target,
            base_center,
            tissue_target,
            residual_target,
            method_bank_candidates,
        )
        source_target = source_stack[stack_center - output_radius : stack_center + output_radius + 1]

        source_field_id = FIELD_TO_ID[row["source_field"]]
        target_field_id = FIELD_TO_ID[target_field]
        contrast_id = CONTRAST_TO_ID.get(row.get("contrast", "T2W"), CONTRAST_TO_ID["T2W"])

        return {
            "source_stack": torch.from_numpy(source_stack.copy()).float(),
            "source_center": torch.from_numpy(source_center.copy()).float(),
            "source_target": torch.from_numpy(source_target.copy()).float(),
            "base_center": torch.from_numpy(base_center.copy()).float(),
            "target": torch.from_numpy(target.copy()).float(),
            "source_field_id": torch.tensor(source_field_id, dtype=torch.long),
            "target_field_id": torch.tensor(target_field_id, dtype=torch.long),
            "contrast_id": torch.tensor(contrast_id, dtype=torch.long),
            "is_identity": torch.tensor(is_identity, dtype=torch.bool),
            "policy_features": torch.from_numpy(policy_features.copy()).float(),
            "has_policy_features": torch.tensor(has_policy_features, dtype=torch.bool),
            "control_features": torch.from_numpy(control_features.copy()).float(),
            "has_control_features": torch.tensor(has_control_features, dtype=torch.bool),
            "method_bank_target_index": torch.tensor(method_bank_target_index, dtype=torch.long),
            "has_method_bank_target": torch.tensor(has_method_bank_target, dtype=torch.bool),
            "method_bank_soft_targets": torch.from_numpy(method_bank_soft_targets.copy()).float(),
            "method_bank_candidate_family_ids": torch.from_numpy(method_bank_candidate_family_ids.copy()).long(),
            "has_method_bank_soft_targets": torch.tensor(has_method_bank_soft_targets, dtype=torch.bool),
            "executor_family_soft_targets": torch.from_numpy(executor_family_soft_targets.copy()).float(),
            "executor_param_soft_targets": torch.from_numpy(executor_param_soft_targets.copy()).float(),
            "executor_param_hard_targets": torch.from_numpy(executor_param_hard_targets.copy()).float(),
            "executor_family_target_id": torch.tensor(executor_family_target_id, dtype=torch.long),
            "has_executor_targets": torch.tensor(has_executor_targets, dtype=torch.bool),
            "method_bank_candidates": torch.from_numpy(method_bank_candidates.copy()).float(),
            "has_method_bank_candidates": torch.tensor(has_method_bank_candidates, dtype=torch.bool),
            "tissue_target": torch.from_numpy(tissue_target[0].copy()).long(),
            "has_tissue_target": torch.tensor(has_tissue_target, dtype=torch.bool),
            "residual_target": torch.from_numpy(residual_target.copy()).float(),
            "has_residual_target": torch.tensor(has_residual_target, dtype=torch.bool),
            "meta": row,
        }

    def _make_base_center(
        self,
        source_center: np.ndarray,
        row: dict[str, Any],
        target_field: str,
        is_identity: bool,
    ) -> np.ndarray:
        method = self.base_calibration_by_contrast.get(row.get("contrast", "T2W"), self.base_calibration)
        if is_identity or method in ("", "none", "source-copy"):
            return source_center.astype(np.float32, copy=False)
        calibrated = [
            calibrate_source_like(
                source_slice,
                contrast=row.get("contrast", "T2W"),
                target_field=target_field,
                target_stats_table=self.target_stats,
                method=method,
            )
            for source_slice in source_center
        ]
        return np.stack(calibrated, axis=0).astype(np.float32)

    def _apply_augmentation(
        self,
        source_stack: np.ndarray,
        source_center: np.ndarray,
        target: np.ndarray,
        base_center: np.ndarray,
        tissue_target: np.ndarray,
        residual_target: np.ndarray,
        method_bank_candidates: np.ndarray,
        *,
        row: dict[str, Any],
        target_field: str,
        is_identity: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.augmentation_config
        if self.rng.random() < float(cfg.get("spatial_prob", 1.0)):
            grid = self._make_spatial_augmentation_grid(
                height=int(source_stack.shape[-2]),
                width=int(source_stack.shape[-1]),
                cfg=cfg,
            )
            padding_mode = str(cfg.get("padding_mode", "border"))
            source_stack = self._warp_array(source_stack, grid, mode="bilinear", padding_mode=padding_mode)
            source_center = self._warp_array(source_center, grid, mode="bilinear", padding_mode=padding_mode)
            target = self._warp_array(target, grid, mode="bilinear", padding_mode=padding_mode)
            base_center = self._warp_array(base_center, grid, mode="bilinear", padding_mode=padding_mode)
            residual_target = self._warp_array(residual_target, grid, mode="bilinear", padding_mode=padding_mode)
            method_bank_candidates = self._warp_array(
                method_bank_candidates,
                grid,
                mode="bilinear",
                padding_mode=padding_mode,
            )
            tissue_target = self._warp_array(tissue_target, grid, mode="nearest", padding_mode=padding_mode).astype(
                np.int64,
                copy=False,
            )

        intensity_cfg = dict(cfg.get("source_intensity", {}) or {})
        if intensity_cfg.get("enabled", False) and self.rng.random() < float(intensity_cfg.get("prob", 1.0)):
            protect_identity = bool(intensity_cfg.get("protect_identity", True))
            if not (protect_identity and is_identity):
                source_stack, source_center = self._apply_source_intensity_augmentation(
                    source_stack,
                    source_center,
                    intensity_cfg,
                )
                if bool(intensity_cfg.get("recompute_base_after_source_intensity", True)):
                    radius = self.output_depth // 2
                    center = self.stack_depth // 2
                    source_target = source_stack[center - radius : center + radius + 1]
                    base_center = self._make_base_center(source_target, row, target_field, is_identity)

        return (
            source_stack.astype(np.float32, copy=False),
            source_center.astype(np.float32, copy=False),
            target.astype(np.float32, copy=False),
            base_center.astype(np.float32, copy=False),
            tissue_target.astype(np.int64, copy=False),
            residual_target.astype(np.float32, copy=False),
            method_bank_candidates.astype(np.float32, copy=False),
        )

    def _make_spatial_augmentation_grid(
        self,
        *,
        height: int,
        width: int,
        cfg: dict[str, Any],
    ) -> torch.Tensor:
        rotate_deg = float(cfg.get("rotate_deg", 0.0))
        angle = math.radians(self.rng.uniform(-rotate_deg, rotate_deg)) if rotate_deg > 0.0 else 0.0
        scale_min = float(cfg.get("scale_min", 1.0))
        scale_max = float(cfg.get("scale_max", scale_min))
        scale = self.rng.uniform(scale_min, scale_max) if scale_max != scale_min else scale_min
        scale = max(scale, 1e-3)
        translate_px = float(cfg.get("translate_px", 0.0))
        tx = self.rng.uniform(-translate_px, translate_px) * 2.0 / max(1.0, float(width))
        ty = self.rng.uniform(-translate_px, translate_px) * 2.0 / max(1.0, float(height))
        cos_a = math.cos(angle) / scale
        sin_a = math.sin(angle) / scale
        theta = torch.tensor(
            [[[cos_a, -sin_a, tx], [sin_a, cos_a, ty]]],
            dtype=torch.float32,
        )
        grid = F.affine_grid(theta, size=(1, 1, height, width), align_corners=False)

        if self.rng.random() < float(cfg.get("elastic_prob", 0.0)):
            grid_size = max(2, int(cfg.get("elastic_grid_size", 5)))
            max_px = float(cfg.get("elastic_max_px", 0.0))
            if max_px > 0.0:
                disp_x = np.asarray(
                    [self.rng.uniform(-max_px, max_px) * 2.0 / max(1.0, float(width)) for _ in range(grid_size * grid_size)],
                    dtype=np.float32,
                ).reshape(grid_size, grid_size)
                disp_y = np.asarray(
                    [self.rng.uniform(-max_px, max_px) * 2.0 / max(1.0, float(height)) for _ in range(grid_size * grid_size)],
                    dtype=np.float32,
                ).reshape(grid_size, grid_size)
                disp = torch.from_numpy(np.stack([disp_x, disp_y], axis=0))[None, ...]
                disp = F.interpolate(disp, size=(height, width), mode="bicubic", align_corners=True)
                grid = grid + disp.permute(0, 2, 3, 1)
        return grid.clamp(-1.25, 1.25)

    @staticmethod
    def _warp_array(
        array: np.ndarray,
        grid: torch.Tensor,
        *,
        mode: str,
        padding_mode: str,
    ) -> np.ndarray:
        if array.size == 0:
            return array
        original_shape = array.shape
        height, width = original_shape[-2:]
        flat = array.reshape(-1, height, width)
        tensor = torch.from_numpy(flat.astype(np.float32, copy=False))[None, ...]
        warped = F.grid_sample(
            tensor,
            grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=False,
        )[0]
        return warped.numpy().reshape(original_shape)

    def _apply_source_intensity_augmentation(
        self,
        source_stack: np.ndarray,
        source_center: np.ndarray,
        cfg: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        gamma_min = float(cfg.get("gamma_min", 1.0))
        gamma_max = float(cfg.get("gamma_max", gamma_min))
        gamma = self.rng.uniform(gamma_min, gamma_max) if gamma_max != gamma_min else gamma_min
        gain_min = float(cfg.get("gain_min", 1.0))
        gain_max = float(cfg.get("gain_max", gain_min))
        gain = self.rng.uniform(gain_min, gain_max) if gain_max != gain_min else gain_min
        bias = self.rng.uniform(-float(cfg.get("bias_abs", 0.0)), float(cfg.get("bias_abs", 0.0)))
        noise_std = float(cfg.get("noise_std", 0.0))

        def augment(array: np.ndarray) -> np.ndarray:
            out = np.clip(array.astype(np.float32, copy=True), 0.0, 1.0)
            if gamma > 0.0 and gamma != 1.0:
                out = np.power(out, gamma).astype(np.float32, copy=False)
            out = out * gain + bias
            if noise_std > 0.0:
                noise_seed = self.rng.randrange(0, 2**32 - 1)
                noise = np.random.RandomState(noise_seed).normal(0.0, noise_std, size=out.shape).astype(np.float32)
                out = out + noise
            return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

        source_stack = augment(source_stack)
        source_center = source_stack[self.stack_depth // 2 : self.stack_depth // 2 + 1]
        return source_stack, source_center

    def _crop_arrays(
        self,
        *arrays: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        if self.patch_size is None:
            return tuple(arrays)
        source_stack = arrays[0]
        _, h, w = source_stack.shape
        crop_h = min(self.patch_size, h)
        crop_w = min(self.patch_size, w)
        y0 = self.rng.randrange(0, h - crop_h + 1) if h > crop_h else 0
        x0 = self.rng.randrange(0, w - crop_w + 1) if w > crop_w else 0
        ys = slice(y0, y0 + crop_h)
        xs = slice(x0, x0 + crop_w)
        cropped = tuple(array[..., ys, xs] for array in arrays)
        if crop_h == self.patch_size and crop_w == self.patch_size:
            return cropped

        pad_top = (self.patch_size - crop_h) // 2
        pad_bottom = self.patch_size - crop_h - pad_top
        pad_left = (self.patch_size - crop_w) // 2
        pad_right = self.patch_size - crop_w - pad_left
        return tuple(
            np.pad(
                array,
                [(0, 0)] * (array.ndim - 2) + [(pad_top, pad_bottom), (pad_left, pad_right)],
                mode="constant",
                constant_values=0,
            )
            for array in cropped
        )
