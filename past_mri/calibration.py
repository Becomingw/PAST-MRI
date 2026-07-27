from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

EPS = 1e-6
ROBUST_KEYS = (5.0, 50.0, 95.0)
DEFAULT_PERCENTILES = (0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0, 99.5)


def masked_values(data: np.ndarray, mask_eps: float = 1e-4) -> np.ndarray:
    values = data[np.isfinite(data) & (data > mask_eps)]
    if values.size == 0:
        values = data[np.isfinite(data)]
    return values.astype(np.float32, copy=False).reshape(-1)


def percentile_stats(
    data: np.ndarray,
    percentiles: Sequence[float] = DEFAULT_PERCENTILES,
    mask_eps: float = 1e-4,
) -> dict[str, float]:
    values = masked_values(np.asarray(data, dtype=np.float32), mask_eps=mask_eps)
    if values.size == 0:
        raise ValueError("Cannot compute percentiles from an empty array")
    qs = np.percentile(values, percentiles)
    return {f"{p:g}": float(v) for p, v in zip(percentiles, qs)}


def load_target_stats(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text())
    if "stats" not in payload:
        raise KeyError(f"target stats file missing top-level 'stats': {path}")
    return payload["stats"]


def pct(stats: Mapping[str, float], key: float) -> float:
    return float(stats[f"{key:g}"])


def robust_zscore_array(
    x: np.ndarray,
    source_stats: Mapping[str, float],
    target_stats: Mapping[str, float],
) -> np.ndarray:
    s5, s50, s95 = (pct(source_stats, key) for key in ROBUST_KEYS)
    t5, t50, t95 = (pct(target_stats, key) for key in ROBUST_KEYS)
    out = (x.astype(np.float32) - s50) / max(s95 - s5, EPS) * max(t95 - t5, EPS) + t50
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def percentile_match_array(
    x: np.ndarray,
    source_stats: Mapping[str, float],
    target_stats: Mapping[str, float],
) -> np.ndarray:
    keys = sorted(float(key) for key in source_stats)
    xs = np.asarray([pct(source_stats, key) for key in keys], dtype=np.float32)
    ys = np.asarray([pct(target_stats, key) for key in keys], dtype=np.float32)
    xs = np.maximum.accumulate(xs)
    for idx in range(1, xs.size):
        if xs[idx] <= xs[idx - 1]:
            xs[idx] = xs[idx - 1] + EPS
    return np.clip(np.interp(x.astype(np.float32), xs, ys), 0.0, 1.0).astype(np.float32)


def split_beta_method(method: str) -> tuple[str, float | None]:
    method = method.strip()
    if "_b" not in method:
        return method, None
    base_method, beta_token = method.rsplit("_b", 1)
    if not base_method:
        return method, None
    try:
        beta = float(beta_token) / 100.0 if beta_token.isdigit() else float(beta_token)
    except ValueError as exc:
        raise ValueError(f"Invalid calibration beta in method: {method}") from exc
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"Calibration beta must be in [0, 1], got {beta} from method: {method}")
    return base_method, beta


def calibrate_array(
    x: np.ndarray,
    source_stats: Mapping[str, float],
    target_stats: Mapping[str, float],
    method: str,
) -> np.ndarray:
    source = np.clip(x.astype(np.float32), 0.0, 1.0).astype(np.float32)
    base_method, beta = split_beta_method(method)
    if base_method in ("", "source-copy", "none"):
        calibrated = source
    elif base_method == "robust-zscore":
        calibrated = robust_zscore_array(x, source_stats, target_stats)
    elif base_method in ("percentile", "percentile-matching", "quantile"):
        calibrated = percentile_match_array(x, source_stats, target_stats)
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    if beta is None:
        return calibrated
    return np.clip((1.0 - beta) * source + beta * calibrated, 0.0, 1.0).astype(np.float32)



def calibrate_source_like(
    source: np.ndarray,
    contrast: str,
    target_field: str,
    target_stats_table: Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]],
    method: str,
    mask_eps: float = 1e-4,
) -> np.ndarray:
    source_stats = percentile_stats(source, mask_eps=mask_eps)
    try:
        target_stats = target_stats_table[contrast][target_field]["percentiles"]
    except KeyError as exc:
        raise KeyError(f"Missing target stats for {contrast}/{target_field}") from exc
    return calibrate_array(source, source_stats, target_stats, method=method)
