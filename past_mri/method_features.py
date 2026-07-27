from __future__ import annotations

import re

import numpy as np


METHOD_FEATURE_NAMES = (
    "family_source_copy",
    "family_quantile",
    "family_targetcohort",
    "family_other",
    "beta",
    "lowfreq_alpha",
    "lowfreq_sigma_xy",
    "lowfreq_sigma_z",
    "source_texture",
    "texture_blur_sigma_xy",
    "texture_blur_sigma_z",
    "texture_unsharp_amount",
    "texture_median_kernel_xy",
    "texture_median_kernel_z",
)

METHOD_FEATURE_TO_INDEX = {name: idx for idx, name in enumerate(METHOD_FEATURE_NAMES)}


def _parse_triplet(text: str) -> tuple[float, float, float]:
    parts = [float(part) for part in text.split("x")]
    if len(parts) != 3:
        raise ValueError(f"Expected triplet, got {text}")
    return parts[0], parts[1], parts[2]


def _find_scaled_int(pattern: str, method: str) -> float:
    match = re.search(pattern, method)
    if not match:
        return 0.0
    return float(match.group(1)) / 100.0


def parse_method_features(method: str) -> dict[str, float]:
    features = {name: 0.0 for name in METHOD_FEATURE_NAMES}
    if method.startswith(("source-copy", "source-lfbias")):
        features["family_source_copy"] = 1.0
    elif method.startswith(("paired-quantile", "slice-quantile")):
        features["family_quantile"] = 1.0
    elif method.startswith("targetcohort"):
        features["family_targetcohort"] = 1.0
    else:
        features["family_other"] = 1.0

    features["beta"] = _find_scaled_int(r"_b(\d{3})(?:_|$)", f"_{method}")

    lfbias = re.search(r"(?:^source-|_)lfbias_s([0-9.]+x[0-9.]+x[0-9.]+)_a(\d{3})(?:_|$)", method)
    if lfbias:
        sx, sy, sz = _parse_triplet(lfbias.group(1))
        features["lowfreq_sigma_xy"] = (sx + sy) / 2.0
        features["lowfreq_sigma_z"] = sz
        features["lowfreq_alpha"] = float(lfbias.group(2)) / 100.0

    blur = re.search(r"_blur_s([0-9.]+x[0-9.]+x[0-9.]+)(?:_|$)", method)
    median = re.search(r"_median_s([0-9.]+x[0-9.]+x[0-9.]+)(?:_|$)", method)
    unsharp = re.search(r"_unsharp_s([0-9.]+x[0-9.]+x[0-9.]+)_a(\d{3})(?:_|$)", method)
    if blur:
        sx, sy, sz = _parse_triplet(blur.group(1))
        features["texture_blur_sigma_xy"] = (sx + sy) / 2.0
        features["texture_blur_sigma_z"] = sz
    if median:
        sx, sy, sz = _parse_triplet(median.group(1))
        features["texture_median_kernel_xy"] = (sx + sy) / 2.0
        features["texture_median_kernel_z"] = sz
    if unsharp:
        sx, sy, sz = _parse_triplet(unsharp.group(1))
        features["texture_blur_sigma_xy"] = (sx + sy) / 2.0
        features["texture_blur_sigma_z"] = sz
        features["texture_unsharp_amount"] = float(unsharp.group(2)) / 100.0
        if method.startswith("source-copy"):
            features["source_texture"] = 1.0
    return features


def method_feature_vector(method: str) -> np.ndarray:
    features = parse_method_features(method)
    return np.asarray([features[name] for name in METHOD_FEATURE_NAMES], dtype=np.float32)
