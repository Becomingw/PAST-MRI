#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data._utils.collate import default_collate

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from past_mri.config import load_yaml_config
from past_mri.data import PairedTranslationDataset
from past_mri.ema import ModelEMA
from past_mri.losses import Task3DistillLoss, tissue_ce_loss
from past_mri.constants import CONTRAST_TO_ID, FIELD_TO_ID
from past_mri.model_factory import build_cond_naf_unet_from_config
from past_mri.policy_supervision import (
    direct_guard_feature_loss,
    executor_family_hard_target_loss,
    executor_family_soft_target_loss,
    executor_param_target_loss,
    guard_feature_loss,
    method_bank_candidate_blend_loss,
    method_bank_candidate_masked_nrmse_loss,
    method_bank_entropy_loss,
    method_bank_family_soft_target_loss,
    method_bank_soft_target_loss,
    method_bank_target_loss,
    policy_feature_loss,
    primitive_weight_target_loss,
    resolve_policy_feature_scales,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collate_translation_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    meta = [item.get("meta", {}) for item in batch]
    tensor_items = [{key: value for key, value in item.items() if key != "meta"} for item in batch]
    collated = default_collate(tensor_items)
    collated["meta"] = meta
    return collated


def load_config(path: str | Path) -> dict:
    return load_yaml_config(path)


def cosine_lr(base_lr: float, step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * float(step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def scheduled_lr(base_lr: float, step: int, total_steps: int, warmup_steps: int, optim_cfg: dict) -> float:
    scheduler = str(optim_cfg.get("scheduler", "cosine")).lower()
    if step < warmup_steps:
        return base_lr * float(step + 1) / max(1, warmup_steps)
    if scheduler in {"constant", "constant_after_warmup", "flat"}:
        return base_lr
    if scheduler in {"cosine_floor", "cosine_with_floor"}:
        floor = float(optim_cfg.get("min_lr_ratio", 0.25))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return base_lr * (floor + (1.0 - floor) * cosine)
    return cosine_lr(base_lr, step, total_steps, warmup_steps)


def parse_assert_power(items: list[str] | None) -> dict[str, str]:
    assertions: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--assert-power entries must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        assertions[key.strip()] = value.strip()
    return assertions


def bool_arg(value: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Expected boolean value, got {value!r}")


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return bool_arg(value)


def configured_seed_count(cfg: dict) -> int:
    for key in ("seed_set", "seeds", "replicate_seeds"):
        values = cfg.get(key)
        if isinstance(values, list):
            return len(values)
    return 1 if "seed" in cfg else 0


def build_power_self_check(cfg: dict, manifest_path: Path | None = None) -> dict[str, Any]:
    optim_cfg = dict(cfg.get("optim", {}) or {})
    logging_cfg = dict(cfg.get("logging", {}) or {})
    early_cfg = dict(cfg.get("early_stop", {}) or cfg.get("convergence", {}).get("early_stop", {}) or {})
    batch_size = int(cfg.get("batch_size", 1))
    grad_accum = int(
        optim_cfg.get(
            "gradient_accumulation_steps",
            cfg.get("gradient_accumulation_steps", cfg.get("grad_accum_steps", 1)),
        )
    )
    scheduler = str(optim_cfg.get("scheduler", "cosine")).lower()
    min_lr_ratio = float(optim_cfg.get("min_lr_ratio", 0.0))
    lr_no_early_decay = bool(
        optim_cfg.get("lr_no_early_decay", False)
        or scheduler in {"constant", "constant_after_warmup", "flat"}
        or (scheduler in {"cosine_floor", "cosine_with_floor"} and min_lr_ratio >= 0.25)
    )
    manifest_rows = None
    if manifest_path is not None and manifest_path.exists():
        with manifest_path.open() as f:
            manifest_rows = sum(1 for line in f if line.strip())
    return {
        "effective_batch": int(batch_size * max(1, grad_accum)),
        "batch_size": batch_size,
        "gradient_accumulation_steps": max(1, grad_accum),
        "lr": float(optim_cfg.get("lr", 0.0)),
        "scheduler": scheduler,
        "min_lr_ratio": min_lr_ratio,
        "lr_no_early_decay": lr_no_early_decay,
        "warmup_steps": int(optim_cfg.get("warmup_steps", 0)),
        "total_steps": int(optim_cfg.get("total_steps", 0)),
        "val_interval": int(logging_cfg.get("val_interval", 0)),
        "converge_on_val": bool(early_cfg.get("enabled", False)) and int(logging_cfg.get("val_interval", 0)) > 0,
        "early_stop": early_cfg,
        "ema_enabled": bool((cfg.get("ema", {}) or {}).get("enabled", True)),
        "ema_decay": float((cfg.get("ema", {}) or {}).get("decay", 0.0)),
        "seed_count": configured_seed_count(cfg),
        "seeds": cfg.get("seed_set", cfg.get("seeds", [cfg.get("seed")] if "seed" in cfg else [])),
        "manifest_rows": manifest_rows,
    }


def assert_power_self_check(power: dict[str, Any], assertions: dict[str, str]) -> list[str]:
    failures: list[str] = []
    if "batch_ge" in assertions and int(power["effective_batch"]) < int(assertions["batch_ge"]):
        failures.append(f"effective_batch={power['effective_batch']} < {assertions['batch_ge']}")
    if "seeds_ge" in assertions and int(power["seed_count"]) < int(assertions["seeds_ge"]):
        failures.append(f"seed_count={power['seed_count']} < {assertions['seeds_ge']}")
    if "lr_no_early_decay" in assertions and bool(power["lr_no_early_decay"]) != bool_arg(assertions["lr_no_early_decay"]):
        failures.append(f"lr_no_early_decay={power['lr_no_early_decay']} != {assertions['lr_no_early_decay']}")
    if "converge_on_val" in assertions and bool(power["converge_on_val"]) != bool_arg(assertions["converge_on_val"]):
        failures.append(f"converge_on_val={power['converge_on_val']} != {assertions['converge_on_val']}")
    if "ema_decay" in assertions and abs(float(power.get("ema_decay", 0.0)) - float(assertions["ema_decay"])) > 1e-12:
        failures.append(f"ema_decay={power.get('ema_decay')} != {assertions['ema_decay']}")
    return failures


def capture_rng_state(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch_state = state["torch"]
        if torch.is_tensor(torch_state):
            torch_state = torch_state.detach().cpu()
        torch.set_rng_state(torch_state)
    if "torch_cuda_all" in state and torch.cuda.is_available():
        cuda_states = [
            value.detach().cpu() if torch.is_tensor(value) else value
            for value in state["torch_cuda_all"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)


def dataset_rng_state(
    train_dataset: PairedTranslationDataset,
    val_dataset: PairedTranslationDataset,
) -> dict[str, Any]:
    return {
        "train_dataset": train_dataset.rng.getstate(),
        "val_dataset": val_dataset.rng.getstate(),
    }


def restore_dataset_rng_state(
    state: dict[str, Any] | None,
    train_dataset: PairedTranslationDataset,
    val_dataset: PairedTranslationDataset,
) -> None:
    if not state:
        return
    if "train_dataset" in state:
        train_dataset.rng.setstate(state["train_dataset"])
    if "val_dataset" in state:
        val_dataset.rng.setstate(state["val_dataset"])


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer,
    step: int,
    ema: ModelEMA | None = None,
    *,
    scaler: torch.cuda.amp.GradScaler | None = None,
    config: dict | None = None,
    device: torch.device | None = None,
    dataset_rng: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    if ema is not None:
        payload["ema"] = ema.module.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if config is not None:
        payload["config"] = config
    if device is not None:
        payload["rng_state"] = capture_rng_state(device)
    if dataset_rng is not None:
        payload["dataset_rng_state"] = dataset_rng
    atomic_torch_save(payload, path)


def load_model_weights(
    model: torch.nn.Module,
    checkpoint: Path,
    device: torch.device,
    *,
    strict: bool = True,
    skip_prefixes: tuple[str, ...] = (),
) -> dict:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    step = int(ckpt.get("step", 0)) if isinstance(ckpt, dict) else 0
    skip_prefixes = tuple(prefix for prefix in skip_prefixes if prefix)
    if strict and skip_prefixes:
        raise ValueError("skip_prefixes requires strict=False")
    if strict:
        model.load_state_dict(state)
        return {
            "step": step,
            "strict": True,
            "loaded_keys": len(state),
            "missing_model_keys": [],
            "skipped_unexpected_keys": [],
            "skipped_shape_mismatch": [],
            "skipped_by_prefix": [],
            "expanded_input_keys": [],
        }

    model_state = model.state_dict()
    compatible_state = {}
    skipped_unexpected_keys = []
    skipped_shape_mismatch = []
    skipped_by_prefix = []
    expanded_input_keys = []
    for key, value in state.items():
        if any(key.startswith(prefix) for prefix in skip_prefixes):
            skipped_by_prefix.append(key)
            continue
        if key not in model_state:
            skipped_unexpected_keys.append(key)
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            target_value = model_state[key]
            if (
                value.ndim == 2
                and target_value.ndim == 2
                and target_value.shape[0] == value.shape[0]
                and target_value.shape[1] > value.shape[1]
            ):
                expanded = torch.zeros_like(target_value)
                expanded[:, : value.shape[1]] = value.to(device=target_value.device, dtype=target_value.dtype)
                compatible_state[key] = expanded
                expanded_input_keys.append(
                    {
                        "key": key,
                        "checkpoint_shape": list(value.shape),
                        "model_shape": list(target_value.shape),
                    }
                )
                continue
            skipped_shape_mismatch.append(
                {
                    "key": key,
                    "checkpoint_shape": list(value.shape),
                    "model_shape": list(model_state[key].shape),
                }
            )
            continue
        compatible_state[key] = value

    if not compatible_state:
        raise RuntimeError(f"No compatible weights found in {checkpoint}")

    merged_state = dict(model_state)
    merged_state.update(compatible_state)
    model.load_state_dict(merged_state)
    missing_model_keys = [key for key in model_state.keys() if key not in compatible_state]
    return {
        "step": step,
        "strict": False,
        "loaded_keys": len(compatible_state),
        "missing_model_keys": missing_model_keys,
        "skipped_unexpected_keys": skipped_unexpected_keys,
        "skipped_shape_mismatch": skipped_shape_mismatch,
        "skipped_by_prefix": skipped_by_prefix,
        "expanded_input_keys": expanded_input_keys,
    }


def residual_reg_weight_from_contrast(
    contrast_id: torch.Tensor,
    multipliers: dict[str, float],
) -> torch.Tensor | None:
    return sample_weight_from_contrast_id(
        contrast_id,
        multipliers,
        config_name="residual_reg_multipliers",
    )


def sample_weight_from_contrast_id(
    contrast_id: torch.Tensor,
    multipliers: dict[str, float],
    *,
    config_name: str,
) -> torch.Tensor | None:
    if not multipliers:
        return None
    weights = torch.ones_like(contrast_id, dtype=torch.float32)
    for contrast, multiplier in multipliers.items():
        if contrast not in CONTRAST_TO_ID:
            raise ValueError(f"Unknown contrast in {config_name}: {contrast}")
        weights = torch.where(
            contrast_id == CONTRAST_TO_ID[contrast],
            weights.new_tensor(float(multiplier)),
            weights,
        )
    return weights


def sample_weight_from_field_id(
    field_id: torch.Tensor,
    multipliers: dict[str, float],
    *,
    config_name: str,
) -> torch.Tensor | None:
    if not multipliers:
        return None
    weights = torch.ones_like(field_id, dtype=torch.float32)
    for field, multiplier in multipliers.items():
        if field not in FIELD_TO_ID:
            raise ValueError(f"Unknown field in {config_name}: {field}")
        weights = torch.where(
            field_id == FIELD_TO_ID[field],
            weights.new_tensor(float(multiplier)),
            weights,
        )
    return weights


def _multiply_optional_weights(*weights: torch.Tensor | None) -> torch.Tensor | None:
    out: torch.Tensor | None = None
    for weight in weights:
        if weight is None:
            continue
        out = weight if out is None else out.to(device=weight.device, dtype=weight.dtype) * weight
    return out


def _parse_target_field_multipliers(raw: dict | None) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for term, mapping in dict(raw or {}).items():
        out[str(term)] = {str(field): float(multiplier) for field, multiplier in dict(mapping or {}).items()}
    return out


def _parse_contrast_term_multipliers(raw: dict | None) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for term, mapping in dict(raw or {}).items():
        out[str(term)] = {str(contrast): float(multiplier) for contrast, multiplier in dict(mapping or {}).items()}
    return out


def target_field_term_weights(
    target_field_id: torch.Tensor,
    target_field_multipliers: dict[str, dict[str, float]],
) -> dict[str, torch.Tensor]:
    weights: dict[str, torch.Tensor] = {}
    for term, multipliers in target_field_multipliers.items():
        if term in {"residual_reg", "calibration_lowfreq"}:
            continue
        weight = sample_weight_from_field_id(
            target_field_id,
            multipliers,
            config_name=f"target_field_multipliers.{term}",
        )
        if weight is not None:
            weights[term] = weight
    return weights


def contrast_term_weights(
    contrast_id: torch.Tensor,
    contrast_term_multipliers: dict[str, dict[str, float]],
) -> dict[str, torch.Tensor]:
    weights: dict[str, torch.Tensor] = {}
    for term, multipliers in contrast_term_multipliers.items():
        if term in {"residual_reg", "calibration_lowfreq"}:
            continue
        weight = sample_weight_from_contrast_id(
            contrast_id,
            multipliers,
            config_name=f"contrast_term_multipliers.{term}",
        )
        if weight is not None:
            weights[term] = weight
    return weights


def merge_term_weights(*groups: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    merged: dict[str, torch.Tensor] = {}
    for group in groups:
        for term, weight in group.items():
            merged[term] = _multiply_optional_weights(merged.get(term), weight)
    return merged


def configure_trainable_parameters(model: torch.nn.Module, train_cfg: dict | None = None) -> dict[str, object]:
    train_cfg = dict(train_cfg or {})
    contains = [str(item) for item in train_cfg.get("trainable_name_contains", []) if str(item)]
    if not contains:
        trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
        return {
            "mode": "all",
            "tokens": [],
            "trainable_names": trainable_names,
            "trainable_params": sum(param.numel() for param in model.parameters() if param.requires_grad),
            "frozen_params": sum(param.numel() for param in model.parameters() if not param.requires_grad),
        }

    trainable_names: list[str] = []
    frozen_names: list[str] = []
    for name, param in model.named_parameters():
        keep = any(token in name for token in contains)
        param.requires_grad_(keep)
        if keep:
            trainable_names.append(name)
        else:
            frozen_names.append(name)
    if not trainable_names:
        raise ValueError(f"trainable_name_contains matched no parameters: {contains}")
    return {
        "mode": "name_contains",
        "tokens": contains,
        "trainable_names": trainable_names,
        "frozen_names": frozen_names,
        "trainable_params": sum(param.numel() for param in model.parameters() if param.requires_grad),
        "frozen_params": sum(param.numel() for param in model.parameters() if not param.requires_grad),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the paired PAST-MRI five-slice translation model.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--init-model", type=Path, default=None, help="Initialize model weights only, without optimizer state.")
    parser.add_argument("--init-model-partial", action="store_true", help="Load only matching init-model weights by key and shape.")
    parser.add_argument(
        "--init-model-skip-prefix",
        action="append",
        default=[],
        help="With --init-model-partial, skip checkpoint keys starting with this prefix. Can be repeated.",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed for an otherwise identical run.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config/manifest/power settings without training.")
    parser.add_argument(
        "--assert-power",
        nargs="*",
        default=[],
        help="Power self-check assertions as key=value, e.g. batch_ge=16 lr_no_early_decay=true.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    cfg["runtime_provenance"] = {
        "config_path": str(args.config.resolve()),
        "config_sha256": file_sha256(args.config),
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "execution_seed": int(cfg.get("seed", 20260614)),
        "actual_seed_count": 1,
    }
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False) + "\n")
    log_path = args.work_dir / "train.log"
    power_self_check = build_power_self_check(cfg, args.manifest)
    power_assertions = parse_assert_power(args.assert_power)
    power_failures = assert_power_self_check(power_self_check, power_assertions)
    power_payload = {
        "event": "power_self_check",
        "config": str(args.config),
        "manifest": str(args.manifest),
        "work_dir": str(args.work_dir),
        "assertions": power_assertions,
        "failures": power_failures,
        "passed": not power_failures,
        **power_self_check,
    }
    (args.work_dir / "power_self_check.json").write_text(json.dumps(power_payload, indent=2, sort_keys=True) + "\n")
    if args.dry_run:
        print(json.dumps({"dry_run": True, **power_payload}, indent=2, sort_keys=True))
        return 0 if not power_failures else 2
    if power_failures:
        print(json.dumps(power_payload, indent=2, sort_keys=True), file=sys.stderr)
        return 2
    torch.manual_seed(int(cfg.get("seed", 20260614)))
    base_calib_cfg = cfg.get("base_calibration", {})

    sampling_cfg = cfg.get("sampling", {})
    dataset_kwargs = {
        "manifest_path": args.manifest,
        "stack_depth": int(cfg.get("stack_depth", 5)),
        "output_depth": int(cfg.get("model", {}).get("output_channels", 1)),
        "patch_size": int(cfg.get("patch_size", 256)),
        "identity_prob": float(sampling_cfg.get("identity_prob", 0.2)),
        "pair_balanced": bool(sampling_cfg.get("pair_balanced", True)),
        "cache_size": int(sampling_cfg.get("cache_size", 16)),
        "mmap_npy": bool(sampling_cfg.get("mmap_npy", False)),
        "seed": int(cfg.get("seed", 20260614)),
        "base_calibration": str(base_calib_cfg.get("method", "source-copy")),
        "base_calibration_by_contrast": {
            str(k): str(v)
            for k, v in base_calib_cfg.get("methods_by_contrast", {}).items()
        },
        "target_stats_path": base_calib_cfg.get("target_stats_path"),
        "load_method_bank_candidates": bool(sampling_cfg.get("load_method_bank_candidates", True)),
        "executor_family_target_mode": str(sampling_cfg.get("executor_family_target_mode", "family4")),
    }
    train_dataset = PairedTranslationDataset(
        **dataset_kwargs,
        augmentation_config=sampling_cfg.get("augmentation", {"enabled": False}),
    )
    val_dataset_kwargs = dict(dataset_kwargs)
    val_dataset_kwargs["seed"] = int(cfg.get("seed", 20260614)) + 104729
    val_dataset = PairedTranslationDataset(
        **val_dataset_kwargs,
        augmentation_config={"enabled": False},
    )
    if len(train_dataset) != len(val_dataset):
        raise RuntimeError(f"train/val dataset length mismatch: {len(train_dataset)} vs {len(val_dataset)}")
    dataset = train_dataset
    val_len = max(1, min(len(dataset) // 20, 128))
    train_len = len(dataset) - val_len
    split_gen = torch.Generator().manual_seed(int(cfg.get("seed", 20260614)) + 17)
    split_indices = torch.randperm(len(dataset), generator=split_gen).tolist()
    val_indices = split_indices[:val_len]
    train_indices = split_indices[val_len:]
    train_ds = Subset(train_dataset, train_indices)
    val_ds = Subset(val_dataset, val_indices)
    num_workers = int(cfg.get("num_workers", 8))
    persistent_workers = bool(cfg.get("persistent_workers", True)) and num_workers > 0
    loader = DataLoader(
        train_ds,
        batch_size=int(cfg.get("batch_size", 8)),
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        persistent_workers=persistent_workers,
        collate_fn=collate_translation_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.get("batch_size", 8)),
        num_workers=0,
        collate_fn=collate_translation_batch,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_cond_naf_unet_from_config(cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    trainable_report = configure_trainable_parameters(model, cfg.get("training", {}))
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters remain after applying training config")

    optim_cfg = cfg.get("optim", {})
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(optim_cfg.get("lr", 2e-4)),
        weight_decay=float(optim_cfg.get("weight_decay", 1e-4)),
        betas=tuple(optim_cfg.get("betas", [0.9, 0.99])),
    )
    loss_cfg = cfg.get("loss", {})
    residual_reg_multipliers = {
        str(k): float(v)
        for k, v in loss_cfg.get("residual_reg_multipliers", {}).items()
    }
    calibration_lowfreq_multipliers = {
        str(k): float(v)
        for k, v in loss_cfg.get("calibration_lowfreq_multipliers", {}).items()
    }
    target_field_multipliers = _parse_target_field_multipliers(loss_cfg.get("target_field_multipliers"))
    contrast_term_multipliers = _parse_contrast_term_multipliers(loss_cfg.get("contrast_term_multipliers"))
    criterion = Task3DistillLoss(
        charbonnier=float(loss_cfg.get("charbonnier", 1.0)),
        msssim=float(loss_cfg.get("msssim", 0.3)),
        lpips=float(loss_cfg.get("lpips", 0.05)),
        gradient=float(loss_cfg.get("gradient", 0.05)),
        laplacian=float(loss_cfg.get("laplacian", 0.0)),
        masked_nrmse=float(loss_cfg.get("masked_nrmse", 0.0)),
        relative_l2=float(loss_cfg.get("relative_l2", 0.0)),
        relative_l2_eps=float(loss_cfg.get("relative_l2_eps", 1e-8)),
        percentile=float(loss_cfg.get("percentile", 0.0)),
        residual_reg=float(loss_cfg.get("residual_reg", 0.01)),
        calibration_lowfreq=float(loss_cfg.get("calibration_lowfreq", 0.0)),
        calibration_lowfreq_kernel_size=int(loss_cfg.get("calibration_lowfreq_kernel_size", 9)),
        extra_residual=float(loss_cfg.get("extra_residual", 0.0)),
        extra_residual_kernel_size=int(loss_cfg.get("extra_residual_kernel_size", 1)),
        source_highfreq_hinge=float(loss_cfg.get("source_highfreq_hinge", 0.0)),
        source_highfreq_kernel_size=int(loss_cfg.get("source_highfreq_kernel_size", 9)),
        source_highfreq_margin=float(loss_cfg.get("source_highfreq_margin", 0.0)),
        target_highfreq=float(loss_cfg.get("target_highfreq", 0.0)),
        target_highfreq_kernel_size=int(loss_cfg.get("target_highfreq_kernel_size", 9)),
        fourier_highfreq=float(loss_cfg.get("fourier_highfreq", 0.0)),
        fourier_highfreq_cutoff=float(loss_cfg.get("fourier_highfreq_cutoff", 0.18)),
        ssim_data_range_mode=str(loss_cfg.get("ssim_data_range_mode", "unit")),
        ssim_data_range_eps=float(loss_cfg.get("ssim_data_range_eps", 1e-6)),
    ).to(device)
    if float(loss_cfg.get("lpips", 0.05)) > 0.0 and criterion.lpips_model is None:
        raise RuntimeError("LPIPS loss is enabled but lpips.LPIPS(net='alex') failed to load")
    policy_feature_weight = float(loss_cfg.get("policy_feature", 0.0))
    policy_feature_scales = resolve_policy_feature_scales(loss_cfg.get("policy_feature_scales"), device)
    guard_feature_weight = float(loss_cfg.get("guard_feature", 0.0))
    guard_feature_weights = loss_cfg.get("guard_feature_weights")
    if guard_feature_weights is not None:
        guard_feature_weights = torch.as_tensor(guard_feature_weights, dtype=torch.float32, device=device)
    direct_guard_feature_weight = float(loss_cfg.get("direct_guard_feature", 0.0))
    direct_guard_feature_weights = loss_cfg.get("direct_guard_feature_weights", loss_cfg.get("guard_feature_weights"))
    if direct_guard_feature_weights is not None:
        direct_guard_feature_weights = torch.as_tensor(direct_guard_feature_weights, dtype=torch.float32, device=device)
    primitive_weight_weight = float(loss_cfg.get("primitive_weight", 0.0))
    primitive_weight_smoothing = float(loss_cfg.get("primitive_weight_smoothing", 0.0))
    method_bank_weight = float(loss_cfg.get("method_bank", 0.0))
    method_bank_smoothing = float(loss_cfg.get("method_bank_smoothing", 0.0))
    method_bank_soft_target_weight = float(loss_cfg.get("method_bank_soft_target", 0.0))
    method_bank_soft_target_temperature = float(loss_cfg.get("method_bank_soft_target_temperature", 1.0))
    method_bank_family_soft_target_weight = float(loss_cfg.get("method_bank_family_soft_target", 0.0))
    method_bank_family_soft_target_temperature = float(
        loss_cfg.get("method_bank_family_soft_target_temperature", method_bank_soft_target_temperature)
    )
    method_bank_candidate_blend_weight = float(loss_cfg.get("method_bank_candidate_blend", 0.0))
    method_bank_candidate_temperature = float(loss_cfg.get("method_bank_candidate_temperature", 1.0))
    method_bank_candidate_nrmse_weight = float(loss_cfg.get("method_bank_candidate_nrmse", 0.0))
    method_bank_candidate_nrmse_temperature = float(
        loss_cfg.get("method_bank_candidate_nrmse_temperature", method_bank_candidate_temperature)
    )
    method_bank_candidate_nrmse_mask_eps = float(loss_cfg.get("method_bank_candidate_nrmse_mask_eps", 1e-4))
    method_bank_entropy_weight = float(loss_cfg.get("method_bank_entropy", 0.0))
    method_bank_entropy_temperature = float(loss_cfg.get("method_bank_entropy_temperature", method_bank_candidate_temperature))
    executor_family_soft_target_weight = float(loss_cfg.get("executor_family_soft_target", 0.0))
    executor_family_soft_target_temperature = float(loss_cfg.get("executor_family_soft_target_temperature", 1.0))
    executor_family_hard_target_weight = float(loss_cfg.get("executor_family_hard_target", 0.0))
    executor_family_hard_target_smoothing = float(loss_cfg.get("executor_family_hard_target_smoothing", 0.0))
    executor_family_class_weights = loss_cfg.get("executor_family_class_weights")
    if executor_family_class_weights is not None:
        executor_family_class_weights = torch.as_tensor(executor_family_class_weights, dtype=torch.float32, device=device)
    executor_param_soft_target_weight = float(loss_cfg.get("executor_param_soft_target", 0.0))
    executor_param_hard_target_weight = float(loss_cfg.get("executor_param_hard_target", 0.0))
    executor_param_weights = loss_cfg.get("executor_param_weights")
    if executor_param_weights is not None:
        executor_param_weights = torch.as_tensor(executor_param_weights, dtype=torch.float32, device=device)
    tissue_ce_weight = float(loss_cfg.get("tissue_ce", 0.0))

    ema_cfg = cfg.get("ema", {})
    ema = ModelEMA(model, decay=float(ema_cfg.get("decay", 0.999))) if ema_cfg.get("enabled", True) else None
    ema_start = int(ema_cfg.get("start_step", 5000))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.get("amp", True)) and device.type == "cuda")
    total_steps = int(args.max_steps or optim_cfg.get("total_steps", 100000))
    warmup_steps = int(optim_cfg.get("warmup_steps", 2000))
    base_lr = float(optim_cfg.get("lr", 2e-4))
    log_interval = int(cfg.get("logging", {}).get("log_interval", 50))
    val_interval = int(cfg.get("logging", {}).get("val_interval", 2000))
    save_interval = int(cfg.get("logging", {}).get("save_interval", 5000))
    skip_validation = env_bool("T3_SKIP_VAL", False)
    early_cfg = dict(cfg.get("early_stop", {}) or cfg.get("convergence", {}).get("early_stop", {}) or {})
    early_enabled = bool(early_cfg.get("enabled", False))
    early_min_steps = int(early_cfg.get("min_steps", 0))
    early_patience = int(early_cfg.get("patience", 0))
    early_min_delta = float(early_cfg.get("min_delta", 0.0))

    start_step = 0
    if args.init_model:
        init_report = load_model_weights(
            model,
            args.init_model,
            device,
            strict=not args.init_model_partial,
            skip_prefixes=tuple(args.init_model_skip_prefix or ()),
        )
        if ema is not None:
            ema.module.load_state_dict(model.state_dict())
        print(
            "initialized model from "
            f"{args.init_model} "
            f"(checkpoint_step={init_report['step']} "
            f"strict={init_report['strict']} "
            f"loaded_keys={init_report['loaded_keys']} "
            f"missing_model_keys={len(init_report['missing_model_keys'])} "
            f"skipped_unexpected_keys={len(init_report['skipped_unexpected_keys'])} "
            f"skipped_shape_mismatch={len(init_report['skipped_shape_mismatch'])} "
            f"skipped_by_prefix={len(init_report['skipped_by_prefix'])} "
            f"expanded_input_keys={len(init_report['expanded_input_keys'])})"
        )
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))
        if ema is not None and "ema" in ckpt:
            ema.module.load_state_dict(ckpt["ema"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        restore_rng_state(ckpt.get("rng_state"))
        restore_dataset_rng_state(ckpt.get("dataset_rng_state"), train_dataset, val_dataset)

    with log_path.open("a") as log:
        start_record = {
            "event": "start",
            "params": param_count,
            "trainable_params": trainable_report["trainable_params"],
            "frozen_params": trainable_report["frozen_params"],
            "trainable_mode": trainable_report["mode"],
            "trainable_tokens": trainable_report["tokens"],
            "trainable_names": trainable_report["trainable_names"],
            "dataset": len(dataset),
            "power_self_check": power_self_check,
            "skip_validation": skip_validation,
        }
        log.write(json.dumps(start_record, sort_keys=True) + "\n")
        print(
            f"params={param_count} "
            f"trainable={trainable_report['trainable_params']} "
            f"frozen={trainable_report['frozen_params']} "
            f"dataset={len(dataset)} work_dir={args.work_dir}"
        )
        step = start_step
        loader_iter = iter(loader)
        best_val = float("inf")
        no_improve_vals = 0
        stop_reason = "max_steps"
        while step < total_steps:
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)

            lr = scheduled_lr(base_lr, step, total_steps, warmup_steps, optim_cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                output = model(
                    batch["source_stack"],
                    batch["source_field_id"],
                    batch["target_field_id"],
                    batch["contrast_id"],
                    base_center=batch["base_center"],
                )
                residual_weight = _multiply_optional_weights(
                    residual_reg_weight_from_contrast(batch["contrast_id"], residual_reg_multipliers),
                    sample_weight_from_field_id(
                        batch["target_field_id"],
                        target_field_multipliers.get("residual_reg", {}),
                        config_name="target_field_multipliers.residual_reg",
                    ),
                )
                calibration_lowfreq_weight = _multiply_optional_weights(
                    residual_reg_weight_from_contrast(batch["contrast_id"], calibration_lowfreq_multipliers),
                    sample_weight_from_field_id(
                        batch["target_field_id"],
                        target_field_multipliers.get("calibration_lowfreq", {}),
                        config_name="target_field_multipliers.calibration_lowfreq",
                    ),
                )
                term_weights = merge_term_weights(
                    target_field_term_weights(batch["target_field_id"], target_field_multipliers),
                    contrast_term_weights(batch["contrast_id"], contrast_term_multipliers),
                )
                loss, terms = criterion(
                    output,
                    batch["target"],
                    source=batch["source_target"],
                    residual_target=batch["residual_target"],
                    has_residual_target=batch["has_residual_target"],
                    residual_reg_weight=residual_weight,
                    calibration_lowfreq_weight=calibration_lowfreq_weight,
                    term_weights=term_weights,
                )
                if policy_feature_weight:
                    loss_policy_feature = policy_feature_loss(
                        output,
                        batch["policy_features"],
                        batch["has_policy_features"],
                        feature_scales=policy_feature_scales,
                    )
                    loss = loss + policy_feature_weight * loss_policy_feature
                    terms["loss_policy_feature"] = float(loss_policy_feature.detach().cpu())
                if guard_feature_weight:
                    loss_guard_feature = guard_feature_loss(
                        output,
                        batch["policy_features"],
                        batch["has_policy_features"],
                        feature_weights=guard_feature_weights,
                    )
                    loss = loss + guard_feature_weight * loss_guard_feature
                    terms["loss_guard_feature"] = float(loss_guard_feature.detach().cpu())
                if direct_guard_feature_weight:
                    loss_direct_guard_feature = direct_guard_feature_loss(
                        output,
                        batch["control_features"],
                        batch["has_control_features"],
                        feature_weights=direct_guard_feature_weights,
                    )
                    loss = loss + direct_guard_feature_weight * loss_direct_guard_feature
                    terms["loss_direct_guard_feature"] = float(loss_direct_guard_feature.detach().cpu())
                if primitive_weight_weight:
                    loss_primitive_weight = primitive_weight_target_loss(
                        output,
                        batch["policy_features"],
                        batch["has_policy_features"],
                        smoothing=primitive_weight_smoothing,
                    )
                    loss = loss + primitive_weight_weight * loss_primitive_weight
                    terms["loss_primitive_weight"] = float(loss_primitive_weight.detach().cpu())
                if method_bank_weight:
                    loss_method_bank = method_bank_target_loss(
                        output,
                        batch["method_bank_target_index"],
                        batch["has_method_bank_target"],
                        smoothing=method_bank_smoothing,
                    )
                    loss = loss + method_bank_weight * loss_method_bank
                    terms["loss_method_bank"] = float(loss_method_bank.detach().cpu())
                if method_bank_soft_target_weight:
                    loss_method_bank_soft_target = method_bank_soft_target_loss(
                        output,
                        batch["method_bank_soft_targets"],
                        batch["has_method_bank_soft_targets"],
                        temperature=method_bank_soft_target_temperature,
                    )
                    loss = loss + method_bank_soft_target_weight * loss_method_bank_soft_target
                    terms["loss_method_bank_soft_target"] = float(loss_method_bank_soft_target.detach().cpu())
                if method_bank_family_soft_target_weight:
                    loss_method_bank_family_soft_target = method_bank_family_soft_target_loss(
                        output,
                        batch["method_bank_candidate_family_ids"],
                        batch["method_bank_soft_targets"],
                        batch["has_method_bank_soft_targets"],
                        temperature=method_bank_family_soft_target_temperature,
                    )
                    loss = loss + method_bank_family_soft_target_weight * loss_method_bank_family_soft_target
                    terms["loss_method_bank_family_soft_target"] = float(
                        loss_method_bank_family_soft_target.detach().cpu()
                    )
                if method_bank_candidate_blend_weight:
                    loss_method_bank_candidate_blend = method_bank_candidate_blend_loss(
                        output,
                        batch["method_bank_candidates"],
                        batch["target"],
                        batch["has_method_bank_candidates"],
                        temperature=method_bank_candidate_temperature,
                    )
                    loss = loss + method_bank_candidate_blend_weight * loss_method_bank_candidate_blend
                    terms["loss_method_bank_candidate_blend"] = float(
                        loss_method_bank_candidate_blend.detach().cpu()
                    )
                if method_bank_candidate_nrmse_weight:
                    loss_method_bank_candidate_nrmse = method_bank_candidate_masked_nrmse_loss(
                        output,
                        batch["method_bank_candidates"],
                        batch["target"],
                        batch["has_method_bank_candidates"],
                        temperature=method_bank_candidate_nrmse_temperature,
                        mask_eps=method_bank_candidate_nrmse_mask_eps,
                    )
                    loss = loss + method_bank_candidate_nrmse_weight * loss_method_bank_candidate_nrmse
                    terms["loss_method_bank_candidate_nrmse"] = float(
                        loss_method_bank_candidate_nrmse.detach().cpu()
                    )
                if method_bank_entropy_weight:
                    loss_method_bank_entropy = method_bank_entropy_loss(
                        output,
                        batch["has_method_bank_candidates"],
                        temperature=method_bank_entropy_temperature,
                    )
                    loss = loss + method_bank_entropy_weight * loss_method_bank_entropy
                    terms["loss_method_bank_entropy"] = float(loss_method_bank_entropy.detach().cpu())
                if executor_family_soft_target_weight:
                    loss_executor_family_soft_target = executor_family_soft_target_loss(
                        output,
                        batch["executor_family_soft_targets"],
                        batch["has_executor_targets"],
                        temperature=executor_family_soft_target_temperature,
                    )
                    loss = loss + executor_family_soft_target_weight * loss_executor_family_soft_target
                    terms["loss_executor_family_soft_target"] = float(
                        loss_executor_family_soft_target.detach().cpu()
                    )
                if executor_family_hard_target_weight:
                    loss_executor_family_hard_target = executor_family_hard_target_loss(
                        output,
                        batch["executor_family_target_id"],
                        batch["has_executor_targets"],
                        class_weights=executor_family_class_weights,
                        smoothing=executor_family_hard_target_smoothing,
                    )
                    loss = loss + executor_family_hard_target_weight * loss_executor_family_hard_target
                    terms["loss_executor_family_hard_target"] = float(
                        loss_executor_family_hard_target.detach().cpu()
                    )
                if executor_param_soft_target_weight:
                    loss_executor_param_soft_target = executor_param_target_loss(
                        output,
                        batch["executor_param_soft_targets"],
                        batch["has_executor_targets"],
                        feature_weights=executor_param_weights,
                    )
                    loss = loss + executor_param_soft_target_weight * loss_executor_param_soft_target
                    terms["loss_executor_param_soft_target"] = float(
                        loss_executor_param_soft_target.detach().cpu()
                    )
                if executor_param_hard_target_weight:
                    loss_executor_param_hard_target = executor_param_target_loss(
                        output,
                        batch["executor_param_hard_targets"],
                        batch["has_executor_targets"],
                        feature_weights=executor_param_weights,
                    )
                    loss = loss + executor_param_hard_target_weight * loss_executor_param_hard_target
                    terms["loss_executor_param_hard_target"] = float(
                        loss_executor_param_hard_target.detach().cpu()
                    )
                if tissue_ce_weight:
                    loss_tissue_ce = tissue_ce_loss(output, batch["tissue_target"], batch["has_tissue_target"])
                    loss = loss + tissue_ce_weight * loss_tissue_ce
                    terms["loss_tissue_ce"] = float(loss_tissue_ce.detach().cpu())
                terms["loss_total"] = float(loss.detach().cpu())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if ema is not None and step >= ema_start:
                ema.update(model)
            step += 1

            if step % log_interval == 0 or step == 1:
                tissue_delta = output.tissue_delta.detach() if output.tissue_delta is not None else None
                tissue_coefficients = (
                    output.tissue_coefficients.detach() if output.tissue_coefficients is not None else None
                )
                record = {
                    "event": "train",
                    "step": step,
                    "lr": lr,
                    "ema_enabled": ema is not None and step >= ema_start,
                    "alpha_mean": float(output.alpha.detach().mean().cpu()),
                    "alpha_std": float(output.alpha.detach().std().cpu()),
                    "gate_mean": float(output.gate.detach().mean().cpu()),
                    "gate_std": float(output.gate.detach().std().cpu()),
                    "pred_mean": float(output.pred.detach().mean().cpu()),
                    "pred_std": float(output.pred.detach().std().cpu()),
                    "target_mean": float(batch["target"].detach().mean().cpu()),
                    "target_std": float(batch["target"].detach().std().cpu()),
                    "identity_count": int(batch["is_identity"].sum().cpu()),
                    "translation_count": int((~batch["is_identity"]).sum().cpu()),
                    "tissue_count": int(batch["has_tissue_target"].sum().cpu()),
                    "tissue_delta_mean": float(tissue_delta.mean().cpu()) if tissue_delta is not None else 0.0,
                    "tissue_delta_std": float(tissue_delta.std().cpu()) if tissue_delta is not None else 0.0,
                    "tissue_delta_absmax": float(tissue_delta.abs().max().cpu()) if tissue_delta is not None else 0.0,
                    "tissue_coeff_absmax": (
                        float(tissue_coefficients.abs().max().cpu()) if tissue_coefficients is not None else 0.0
                    ),
                    "residual_reg_weight_mean": float(residual_weight.mean().detach().cpu()) if residual_weight is not None else 1.0,
                    "calibration_lowfreq_weight_mean": (
                        float(calibration_lowfreq_weight.mean().detach().cpu())
                        if calibration_lowfreq_weight is not None
                        else 1.0
                    ),
                    **{
                        f"{term}_weight_mean": float(weight.mean().detach().cpu())
                        for term, weight in sorted(term_weights.items())
                    },
                    **terms,
                }
                line = json.dumps(record, sort_keys=True)
                print(line)
                log.write(line + "\n")
                log.flush()
            if not skip_validation and val_interval > 0 and step % val_interval == 0:
                val_loss = run_validation(
                    model if ema is None or step < ema_start else ema.module,
                    criterion,
                    val_loader,
                    device,
                    residual_reg_multipliers,
                    calibration_lowfreq_multipliers,
                    target_field_multipliers,
                    contrast_term_multipliers,
                    policy_feature_weight=policy_feature_weight,
                    policy_feature_scales=policy_feature_scales,
                    primitive_weight_weight=primitive_weight_weight,
                    primitive_weight_smoothing=primitive_weight_smoothing,
                    method_bank_weight=method_bank_weight,
                    method_bank_smoothing=method_bank_smoothing,
                    method_bank_soft_target_weight=method_bank_soft_target_weight,
                    method_bank_soft_target_temperature=method_bank_soft_target_temperature,
                    method_bank_family_soft_target_weight=method_bank_family_soft_target_weight,
                    method_bank_family_soft_target_temperature=method_bank_family_soft_target_temperature,
                    method_bank_candidate_blend_weight=method_bank_candidate_blend_weight,
                    method_bank_candidate_temperature=method_bank_candidate_temperature,
                    method_bank_candidate_nrmse_weight=method_bank_candidate_nrmse_weight,
                    method_bank_candidate_nrmse_temperature=method_bank_candidate_nrmse_temperature,
                    method_bank_candidate_nrmse_mask_eps=method_bank_candidate_nrmse_mask_eps,
                    method_bank_entropy_weight=method_bank_entropy_weight,
                    method_bank_entropy_temperature=method_bank_entropy_temperature,
                    executor_family_soft_target_weight=executor_family_soft_target_weight,
                    executor_family_soft_target_temperature=executor_family_soft_target_temperature,
                    executor_family_hard_target_weight=executor_family_hard_target_weight,
                    executor_family_hard_target_smoothing=executor_family_hard_target_smoothing,
                    executor_family_class_weights=executor_family_class_weights,
                    executor_param_soft_target_weight=executor_param_soft_target_weight,
                    executor_param_hard_target_weight=executor_param_hard_target_weight,
                    executor_param_weights=executor_param_weights,
                    guard_feature_weight=guard_feature_weight,
                    guard_feature_weights=guard_feature_weights,
                    direct_guard_feature_weight=direct_guard_feature_weight,
                    direct_guard_feature_weights=direct_guard_feature_weights,
                    tissue_ce_weight=tissue_ce_weight,
                )
                record = {"event": "val", "step": step, "loss_total": val_loss}
                print(json.dumps(record, sort_keys=True))
                log.write(json.dumps(record, sort_keys=True) + "\n")
                if val_loss < best_val - early_min_delta:
                    best_val = val_loss
                    no_improve_vals = 0
                    save_checkpoint(
                        args.work_dir / "best_val_proxy.ckpt",
                        model,
                        optimizer,
                        step,
                        ema,
                        scaler=scaler,
                        config=cfg,
                        device=device,
                        dataset_rng=dataset_rng_state(train_dataset, val_dataset),
                    )
                    if ema is not None:
                        atomic_torch_save(
                            {"model": ema.module.state_dict(), "step": step},
                            args.work_dir / "ema_best.ckpt",
                        )
                else:
                    no_improve_vals += 1
                if early_enabled and step >= early_min_steps and early_patience > 0 and no_improve_vals >= early_patience:
                    stop_reason = "val_converged_early_stop"
                    save_checkpoint(
                        args.work_dir / "last.ckpt",
                        model,
                        optimizer,
                        step,
                        ema,
                        scaler=scaler,
                        config=cfg,
                        device=device,
                        dataset_rng=dataset_rng_state(train_dataset, val_dataset),
                    )
                    if ema is not None:
                        atomic_torch_save(
                            {"model": ema.module.state_dict(), "step": step},
                            args.work_dir / "ema_last.ckpt",
                        )
                    break
            if step % save_interval == 0 or step == total_steps:
                save_checkpoint(
                    args.work_dir / "last.ckpt",
                    model,
                    optimizer,
                    step,
                    ema,
                    scaler=scaler,
                    config=cfg,
                    device=device,
                    dataset_rng=dataset_rng_state(train_dataset, val_dataset),
                )
                save_checkpoint(
                    args.work_dir / f"ckpt_step_{step:06d}.ckpt",
                    model,
                    optimizer,
                    step,
                    ema,
                    scaler=scaler,
                    config=cfg,
                    device=device,
                    dataset_rng=dataset_rng_state(train_dataset, val_dataset),
                )
                if ema is not None:
                    atomic_torch_save(
                        {"model": ema.module.state_dict(), "step": step},
                        args.work_dir / "ema_last.ckpt",
                    )
                    atomic_torch_save(
                        {"model": ema.module.state_dict(), "step": step},
                        args.work_dir / f"ema_step_{step:06d}.ckpt",
                    )
        done_record = {
            "event": "done",
            "step": step,
            "best_val_proxy": best_val,
            "stop_reason": stop_reason,
            "checkpoint": str(args.work_dir / "last.ckpt"),
            "ema_checkpoint": str(args.work_dir / "ema_last.ckpt") if ema is not None else "",
        }
        print(json.dumps(done_record, sort_keys=True))
        log.write(json.dumps(done_record, sort_keys=True) + "\n")
    return 0


@torch.no_grad()
def run_validation(
    model: torch.nn.Module,
    criterion: Task3DistillLoss,
    loader: DataLoader,
    device: torch.device,
    residual_reg_multipliers: dict[str, float] | None = None,
    calibration_lowfreq_multipliers: dict[str, float] | None = None,
    target_field_multipliers: dict[str, dict[str, float]] | None = None,
    contrast_term_multipliers: dict[str, dict[str, float]] | None = None,
    policy_feature_weight: float = 0.0,
    policy_feature_scales: torch.Tensor | None = None,
    guard_feature_weight: float = 0.0,
    guard_feature_weights: torch.Tensor | None = None,
    direct_guard_feature_weight: float = 0.0,
    direct_guard_feature_weights: torch.Tensor | None = None,
    primitive_weight_weight: float = 0.0,
    primitive_weight_smoothing: float = 0.0,
    method_bank_weight: float = 0.0,
    method_bank_smoothing: float = 0.0,
    method_bank_soft_target_weight: float = 0.0,
    method_bank_soft_target_temperature: float = 1.0,
    method_bank_family_soft_target_weight: float = 0.0,
    method_bank_family_soft_target_temperature: float = 1.0,
    method_bank_candidate_blend_weight: float = 0.0,
    method_bank_candidate_temperature: float = 1.0,
    method_bank_candidate_nrmse_weight: float = 0.0,
    method_bank_candidate_nrmse_temperature: float = 1.0,
    method_bank_candidate_nrmse_mask_eps: float = 1e-4,
    method_bank_entropy_weight: float = 0.0,
    method_bank_entropy_temperature: float = 1.0,
    executor_family_soft_target_weight: float = 0.0,
    executor_family_soft_target_temperature: float = 1.0,
    executor_family_hard_target_weight: float = 0.0,
    executor_family_hard_target_smoothing: float = 0.0,
    executor_family_class_weights: torch.Tensor | None = None,
    executor_param_soft_target_weight: float = 0.0,
    executor_param_hard_target_weight: float = 0.0,
    executor_param_weights: torch.Tensor | None = None,
    tissue_ce_weight: float = 0.0,
) -> float:
    model.eval()
    vals = []
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        output = model(
            batch["source_stack"],
            batch["source_field_id"],
            batch["target_field_id"],
            batch["contrast_id"],
            base_center=batch["base_center"],
        )
        target_field_multipliers = target_field_multipliers or {}
        residual_weight = _multiply_optional_weights(
            residual_reg_weight_from_contrast(batch["contrast_id"], residual_reg_multipliers or {}),
            sample_weight_from_field_id(
                batch["target_field_id"],
                target_field_multipliers.get("residual_reg", {}),
                config_name="target_field_multipliers.residual_reg",
            ),
        )
        calibration_lowfreq_weight = _multiply_optional_weights(
            residual_reg_weight_from_contrast(batch["contrast_id"], calibration_lowfreq_multipliers or {}),
            sample_weight_from_field_id(
                batch["target_field_id"],
                target_field_multipliers.get("calibration_lowfreq", {}),
                config_name="target_field_multipliers.calibration_lowfreq",
            ),
        )
        term_weights = merge_term_weights(
            target_field_term_weights(batch["target_field_id"], target_field_multipliers),
            contrast_term_weights(batch["contrast_id"], contrast_term_multipliers or {}),
        )
        loss, _ = criterion(
            output,
            batch["target"],
            source=batch["source_target"],
            residual_target=batch["residual_target"],
            has_residual_target=batch["has_residual_target"],
            residual_reg_weight=residual_weight,
            calibration_lowfreq_weight=calibration_lowfreq_weight,
            term_weights=term_weights,
        )
        if policy_feature_weight:
            loss = loss + policy_feature_weight * policy_feature_loss(
                output,
                batch["policy_features"],
                batch["has_policy_features"],
                feature_scales=policy_feature_scales,
            )
        if guard_feature_weight:
            loss = loss + guard_feature_weight * guard_feature_loss(
                output,
                batch["policy_features"],
                batch["has_policy_features"],
                feature_weights=guard_feature_weights,
            )
        if direct_guard_feature_weight:
            loss = loss + direct_guard_feature_weight * direct_guard_feature_loss(
                output,
                batch["control_features"],
                batch["has_control_features"],
                feature_weights=direct_guard_feature_weights,
            )
        if primitive_weight_weight:
            loss = loss + primitive_weight_weight * primitive_weight_target_loss(
                output,
                batch["policy_features"],
                batch["has_policy_features"],
                smoothing=primitive_weight_smoothing,
            )
        if method_bank_weight:
            loss = loss + method_bank_weight * method_bank_target_loss(
                output,
                batch["method_bank_target_index"],
                batch["has_method_bank_target"],
                smoothing=method_bank_smoothing,
            )
        if method_bank_soft_target_weight:
            loss = loss + method_bank_soft_target_weight * method_bank_soft_target_loss(
                output,
                batch["method_bank_soft_targets"],
                batch["has_method_bank_soft_targets"],
                temperature=method_bank_soft_target_temperature,
            )
        if method_bank_family_soft_target_weight:
            loss = loss + method_bank_family_soft_target_weight * method_bank_family_soft_target_loss(
                output,
                batch["method_bank_candidate_family_ids"],
                batch["method_bank_soft_targets"],
                batch["has_method_bank_soft_targets"],
                temperature=method_bank_family_soft_target_temperature,
            )
        if method_bank_candidate_blend_weight:
            loss = loss + method_bank_candidate_blend_weight * method_bank_candidate_blend_loss(
                output,
                batch["method_bank_candidates"],
                batch["target"],
                batch["has_method_bank_candidates"],
                temperature=method_bank_candidate_temperature,
            )
        if method_bank_candidate_nrmse_weight:
            loss = loss + method_bank_candidate_nrmse_weight * method_bank_candidate_masked_nrmse_loss(
                output,
                batch["method_bank_candidates"],
                batch["target"],
                batch["has_method_bank_candidates"],
                temperature=method_bank_candidate_nrmse_temperature,
                mask_eps=method_bank_candidate_nrmse_mask_eps,
            )
        if method_bank_entropy_weight:
            loss = loss + method_bank_entropy_weight * method_bank_entropy_loss(
                output,
                batch["has_method_bank_candidates"],
                temperature=method_bank_entropy_temperature,
            )
        if executor_family_soft_target_weight:
            loss = loss + executor_family_soft_target_weight * executor_family_soft_target_loss(
                output,
                batch["executor_family_soft_targets"],
                batch["has_executor_targets"],
                temperature=executor_family_soft_target_temperature,
            )
        if executor_family_hard_target_weight:
            loss = loss + executor_family_hard_target_weight * executor_family_hard_target_loss(
                output,
                batch["executor_family_target_id"],
                batch["has_executor_targets"],
                class_weights=executor_family_class_weights,
                smoothing=executor_family_hard_target_smoothing,
            )
        if executor_param_soft_target_weight:
            loss = loss + executor_param_soft_target_weight * executor_param_target_loss(
                output,
                batch["executor_param_soft_targets"],
                batch["has_executor_targets"],
                feature_weights=executor_param_weights,
            )
        if executor_param_hard_target_weight:
            loss = loss + executor_param_hard_target_weight * executor_param_target_loss(
                output,
                batch["executor_param_hard_targets"],
                batch["has_executor_targets"],
                feature_weights=executor_param_weights,
            )
        if tissue_ce_weight:
            loss = loss + tissue_ce_weight * tissue_ce_loss(output, batch["tissue_target"], batch["has_tissue_target"])
        vals.append(float(loss.cpu()))
    model.train()
    return float(sum(vals) / max(1, len(vals)))


if __name__ == "__main__":
    raise SystemExit(main())
