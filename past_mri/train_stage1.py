#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from past_mri.data import MaskedDiffusionAutoencoderDataset
from past_mri.losses import ssim_loss
from past_mri.masked_diffusion_autoencoder import (
    MDAE2p5DModel,
    export_mdae_initializer,
    masked_diffusion_autoencoder_loss,
)
from past_mri.training_utils import (
    StepTokenBatchSampler,
    amp_optimizer_step,
    atomic_torch_save,
    capture_rng_state,
    cosine_floor_lambda,
    file_sha256,
    load_config,
    restore_rng_state,
    save_panel,
    seed_everything,
)


SUPPORTED_VISIBLE_WEIGHTING = {"uniform", "bounded_inverse_variance"}
SUPPORTED_TRANSFER_MODES = {"encoder_middle_core", "full_core"}


def validate_config(config: dict[str, Any]) -> None:
    for section in ("data", "mask", "noise", "model", "loss", "optim", "logging", "transfer"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"config.{section} must be a mapping")
    stack_depth = int(config.get("stack_depth", 0))
    patch_size = int(config.get("patch_size", 0))
    batch_size = int(config.get("batch_size", 0))
    model_cfg = config["model"]
    if stack_depth < 1 or stack_depth % 2 != 1:
        raise ValueError("stack_depth must be a positive odd integer")
    if int(model_cfg.get("in_channels", -1)) != stack_depth:
        raise ValueError("model.in_channels must equal stack_depth")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    enc_blocks = list(model_cfg.get("enc_blocks", []))
    dec_blocks = list(model_cfg.get("dec_blocks", []))
    if not enc_blocks or len(enc_blocks) != len(dec_blocks):
        raise ValueError("model.enc_blocks and model.dec_blocks must have equal non-zero lengths")
    required_multiple = 2 ** len(enc_blocks)
    if patch_size < required_multiple or patch_size % required_multiple != 0:
        raise ValueError(f"patch_size must be divisible by {required_multiple}")
    if int(model_cfg.get("time_embedding_dim", 0)) < 4:
        raise ValueError("model.time_embedding_dim must be at least 4")
    if bool(model_cfg.get("use_hyper_trunk", False)):
        raise ValueError("MDAE pretraining requires use_hyper_trunk=false for clean core transfer")

    data_cfg = config["data"]
    for key in ("train_manifest", "val_manifest"):
        if not str(data_cfg.get(key, "")):
            raise ValueError(f"data.{key} must be set")

    mask_cfg = config["mask"]
    ratio_min = float(mask_cfg.get("ratio_min", 0.0))
    ratio_max = float(mask_cfg.get("ratio_max", 0.0))
    if not 0.0 < ratio_min <= ratio_max < 1.0:
        raise ValueError("mask ratios must satisfy 0 < ratio_min <= ratio_max < 1")
    block_sizes = list(mask_cfg.get("block_sizes", []))
    if not block_sizes or any(int(value) < 1 for value in block_sizes):
        raise ValueError("mask.block_sizes must contain positive integers")
    if not bool(mask_cfg.get("shared_across_all_slices", False)):
        raise ValueError("MDAE 2.5D requires one spatial mask shared across all adjacent slices")

    noise_cfg = config["noise"]
    if str(noise_cfg.get("schedule")) != "ve_linear":
        raise ValueError("noise.schedule must be ve_linear")
    if str(noise_cfg.get("scale_mode")) != "foreground_robust_std":
        raise ValueError("noise.scale_mode must be foreground_robust_std for [0,1] Task3 MRI")
    if bool(noise_cfg.get("clip_noisy_input", True)):
        raise ValueError("noise.clip_noisy_input must be false to preserve Gaussian corruption")
    if not bool(noise_cfg.get("independent_per_slice", False)):
        raise ValueError("noise.independent_per_slice must be true to prevent adjacent-slice shortcuts")
    t_min = float(noise_cfg.get("t_min", -1.0))
    t_max = float(noise_cfg.get("t_max", -1.0))
    if not 0.0 <= t_min < t_max <= 1.0:
        raise ValueError("noise timestep bounds must satisfy 0 <= t_min < t_max <= 1")
    if float(noise_cfg.get("sigma_max_multiplier", 0.0)) <= 0.0:
        raise ValueError("noise.sigma_max_multiplier must be positive")

    loss_cfg = config["loss"]
    if str(loss_cfg.get("visible_weighting")) not in SUPPORTED_VISIBLE_WEIGHTING:
        raise ValueError(f"loss.visible_weighting must be one of {sorted(SUPPORTED_VISIBLE_WEIGHTING)}")
    if float(loss_cfg.get("masked_weight", -1.0)) < 0.0 or float(loss_cfg.get("visible_weight", -1.0)) < 0.0:
        raise ValueError("loss weights must be non-negative")

    optim_cfg = config["optim"]
    total_steps = int(optim_cfg.get("total_steps", 0))
    warmup_steps = int(optim_cfg.get("warmup_steps", 0))
    if total_steps < 1 or not 0 <= warmup_steps < total_steps:
        raise ValueError("optim requires total_steps > 0 and 0 <= warmup_steps < total_steps")
    if not 0.0 <= float(optim_cfg.get("min_lr_ratio", 0.0)) <= 1.0:
        raise ValueError("optim.min_lr_ratio must be in [0,1]")

    transfer_cfg = config["transfer"]
    if str(transfer_cfg.get("selection")) != "terminal_step":
        raise ValueError("transfer.selection must be terminal_step; pretext validation cannot select downstream value")
    primary_mode = str(transfer_cfg.get("primary_mode"))
    diagnostic_mode = str(transfer_cfg.get("diagnostic_mode"))
    if primary_mode not in SUPPORTED_TRANSFER_MODES or diagnostic_mode not in SUPPORTED_TRANSFER_MODES:
        raise ValueError(f"transfer modes must be one of {sorted(SUPPORTED_TRANSFER_MODES)}")
    if primary_mode != "encoder_middle_core":
        raise ValueError("transfer.primary_mode must be encoder_middle_core to match the paper's downstream use")


def build_dataset(
    config: dict[str, Any],
    manifest: Path,
    *,
    seed: int,
    virtual_samples: int,
) -> MaskedDiffusionAutoencoderDataset:
    data_cfg = dict(config.get("data", {}) or {})
    return MaskedDiffusionAutoencoderDataset(
        str(manifest),
        stack_depth=int(config.get("stack_depth", 5)),
        patch_size=int(config.get("patch_size", 256)),
        virtual_samples=virtual_samples,
        seed=seed,
        cache_size=int(data_cfg.get("cache_size", 16)),
        mask_config=dict(config.get("mask", {}) or {}),
        noise_config=dict(config.get("noise", {}) or {}),
        foreground_threshold=float(data_cfg.get("foreground_threshold", 0.01)),
        minimum_foreground_fraction=float(data_cfg.get("minimum_foreground_fraction", 0.03)),
        crop_attempts=int(data_cfg.get("crop_attempts", 8)),
    )


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    keys = (
        "input_stack",
        "target",
        "mask",
        "masked_loss_mask",
        "visible_loss_mask",
        "timestep",
        "sigma",
        "sigma_data",
    )
    return {key: batch[key].to(device, non_blocking=True) for key in keys}


@torch.inference_mode()
def validate(
    model: MDAE2p5DModel,
    loader: DataLoader,
    device: torch.device,
    *,
    loss_config: dict[str, Any],
    amp_enabled: bool,
) -> tuple[dict[str, float], dict[str, torch.Tensor] | None]:
    model.eval()
    names = (
        "loss",
        "masked_mse",
        "visible_mse",
        "weighted_visible_mse",
        "visible_noise_weight",
        "ssim",
        "baseline_loss",
        "baseline_ssim",
    )
    totals = {name: 0.0 for name in names}
    timestep_bins = {
        "low": {"sum": 0.0, "count": 0},
        "mid": {"sum": 0.0, "count": 0},
        "high": {"sum": 0.0, "count": 0},
    }
    samples = 0
    panel: dict[str, torch.Tensor] | None = None
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            prediction = model(batch["input_stack"], batch["timestep"])
            losses = masked_diffusion_autoencoder_loss(
                prediction,
                batch["target"],
                batch["masked_loss_mask"],
                batch["visible_loss_mask"],
                batch["sigma"],
                batch["sigma_data"],
                masked_weight=float(loss_config.get("masked_weight", 1.0)),
                visible_weight=float(loss_config.get("visible_weight", 1.0)),
                visible_weighting=str(loss_config.get("visible_weighting", "bounded_inverse_variance")),
            )
        baseline = batch["input_stack"][:, batch["input_stack"].shape[1] // 2 : batch["input_stack"].shape[1] // 2 + 1]
        baseline = baseline.clamp(0.0, 1.0)
        baseline_losses = masked_diffusion_autoencoder_loss(
            baseline,
            batch["target"],
            batch["masked_loss_mask"],
            batch["visible_loss_mask"],
            batch["sigma"],
            batch["sigma_data"],
            masked_weight=float(loss_config.get("masked_weight", 1.0)),
            visible_weight=float(loss_config.get("visible_weight", 1.0)),
            visible_weighting=str(loss_config.get("visible_weighting", "bounded_inverse_variance")),
        )
        batch_size = int(batch["target"].shape[0])
        for name in ("loss", "masked_mse", "visible_mse", "weighted_visible_mse", "visible_noise_weight"):
            totals[name] += float(losses[name].item()) * batch_size
        totals["ssim"] += float((1.0 - ssim_loss(prediction.float(), batch["target"].float())).item()) * batch_size
        totals["baseline_loss"] += float(baseline_losses["loss"].item()) * batch_size
        totals["baseline_ssim"] += float((1.0 - ssim_loss(baseline.float(), batch["target"].float())).item()) * batch_size
        per_sample = losses["masked_mse_per_sample"] + losses["weighted_visible_mse_per_sample"]
        for value, timestep in zip(per_sample.detach().cpu().tolist(), batch["timestep"].detach().cpu().tolist()):
            bin_name = "low" if timestep < 1.0 / 3.0 else "mid" if timestep < 2.0 / 3.0 else "high"
            timestep_bins[bin_name]["sum"] += float(value)
            timestep_bins[bin_name]["count"] += 1
        samples += batch_size
        if panel is None:
            panel = {
                "input_center": baseline[:4].detach().cpu(),
                "prediction": prediction[:4].detach().cpu(),
                "target": batch["target"][:4].detach().cpu(),
                "mask": batch["mask"][:4].detach().cpu(),
            }
    if samples == 0:
        raise ValueError("validation loader produced no samples")
    metrics = {name: total / samples for name, total in totals.items()}
    metrics["loss_gain_vs_corrupted_zero_fill"] = metrics["baseline_loss"] - metrics["loss"]
    for bin_name, values in timestep_bins.items():
        count = int(values["count"])
        metrics[f"{bin_name}_t_loss"] = float(values["sum"]) / count if count else float("nan")
        metrics[f"{bin_name}_t_samples"] = float(count)
    metrics["samples"] = float(samples)
    return metrics, panel


def critical_config(config: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "seed",
        "deterministic",
        "stack_depth",
        "patch_size",
        "batch_size",
        "amp",
        "data",
        "model",
        "mask",
        "noise",
        "loss",
        "optim",
        "transfer",
    )
    return {key: config.get(key) for key in keys}


def checkpoint_payload(
    *,
    model: MDAE2p5DModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    step: int,
    best_val_loss: float,
    config: dict[str, Any],
    config_path: Path,
    train_manifest: Path,
    val_manifest: Path,
    run_id: str,
    amp_overflow_count: int,
    consecutive_amp_overflows: int,
) -> dict[str, Any]:
    return {
        "format": "task3_mdae2p5d_full_resume_v1",
        "run_id": run_id,
        "step": int(step),
        "best_val_loss": float(best_val_loss),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng_state": capture_rng_state(),
        "config": config,
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "train_manifest": str(train_manifest),
        "train_manifest_sha256": file_sha256(train_manifest),
        "val_manifest": str(val_manifest),
        "val_manifest_sha256": file_sha256(val_manifest),
        "data_stream": "step_token_v1",
        "amp_overflow_count": int(amp_overflow_count),
        "consecutive_amp_overflows": int(consecutive_amp_overflows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Pretrain CondNAFUNet with task-adapted 2.5D MDAE.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, default=None)
    parser.add_argument("--val-manifest", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--stop-after-step", type=int, default=None)
    parser.add_argument("--allow-config-mismatch", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    validate_config(config)
    data_cfg = dict(config.get("data", {}) or {})
    train_manifest = args.train_manifest or Path(str(data_cfg.get("train_manifest", "")))
    val_manifest = args.val_manifest or Path(str(data_cfg.get("val_manifest", "")))
    if not train_manifest.exists():
        raise FileNotFoundError(f"Training manifest not found: {train_manifest}")
    if not val_manifest.exists():
        raise FileNotFoundError(f"Validation manifest not found: {val_manifest}")
    work_dir = args.work_dir.resolve()
    project_root = REPO_ROOT.resolve()
    if work_dir == project_root or project_root in work_dir.parents:
        raise ValueError(f"work-dir must be outside the code workspace: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config.get("seed", 20260718))
    seed_everything(seed)
    deterministic = bool(config.get("deterministic", False))
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    optim_cfg = dict(config.get("optim", {}) or {})
    logging_cfg = dict(config.get("logging", {}) or {})
    loss_cfg = dict(config.get("loss", {}) or {})
    transfer_cfg = dict(config.get("transfer", {}) or {})
    total_steps = int(optim_cfg.get("total_steps", 40000))
    stop_after_step = int(args.stop_after_step or total_steps)
    if not 0 < stop_after_step <= total_steps:
        raise ValueError("stop-after-step must be in (0,total_steps]")
    batch_size = int(config.get("batch_size", 4))
    amp_enabled = bool(config.get("amp", True)) and device.type == "cuda"

    model = MDAE2p5DModel.from_config(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optim_cfg.get("lr", 1e-4)),
        betas=tuple(float(value) for value in optim_cfg.get("betas", [0.9, 0.999])),
        weight_decay=float(optim_cfg.get("weight_decay", 1e-4)),
    )
    warmup_steps = int(optim_cfg.get("warmup_steps", 1000))
    min_lr_ratio = float(optim_cfg.get("min_lr_ratio", 0.1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_floor_lambda(
            step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio,
        ),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    start_step = 0
    best_val_loss = float("inf")
    amp_overflow_count = 0
    consecutive_amp_overflows = 0
    run_id = str(config.get("experiment_name", "task3_mdae2p5d"))

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("format") != "task3_mdae2p5d_full_resume_v1":
            raise ValueError(f"Unsupported resume format: {checkpoint.get('format')}")
        if not args.allow_config_mismatch and critical_config(checkpoint["config"]) != critical_config(config):
            raise ValueError("Critical config differs from resume checkpoint")
        if checkpoint.get("train_manifest_sha256") != file_sha256(train_manifest):
            raise ValueError("Training manifest changed since checkpoint creation")
        if checkpoint.get("val_manifest_sha256") != file_sha256(val_manifest):
            raise ValueError("Validation manifest changed since checkpoint creation")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng_state(checkpoint.get("rng_state"))
        start_step = int(checkpoint["step"])
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        amp_overflow_count = int(checkpoint.get("amp_overflow_count", 0))
        consecutive_amp_overflows = int(checkpoint.get("consecutive_amp_overflows", 0))
        if start_step >= stop_after_step:
            raise ValueError(f"resume step {start_step} is not before stop step {stop_after_step}")

    train_dataset = build_dataset(config, train_manifest, seed=seed, virtual_samples=total_steps * batch_size)
    val_samples = int(data_cfg.get("val_samples_per_bucket", 8)) * len(train_dataset.bucket_keys)
    val_dataset = build_dataset(config, val_manifest, seed=seed + 1_000_003, virtual_samples=val_samples)
    num_workers = int(config.get("num_workers", 4))
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=StepTokenBatchSampler(start_step=start_step, total_steps=stop_after_step, batch_size=batch_size),
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(data_cfg.get("val_batch_size", batch_size)),
        shuffle=False,
        num_workers=min(num_workers, 2),
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )

    (work_dir / "config.snapshot.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    environment = {
        "run_id": run_id,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "seed": seed,
        "start_step": start_step,
        "stop_after_step": stop_after_step,
        "total_steps": total_steps,
        "train_manifest": str(train_manifest),
        "train_manifest_sha256": file_sha256(train_manifest),
        "val_manifest": str(val_manifest),
        "val_manifest_sha256": file_sha256(val_manifest),
        "train_bucket_counts": {f"{c}/{f}": len(train_dataset._buckets[(c, f)]) for c, f in train_dataset.bucket_keys},
        "val_bucket_counts": {f"{c}/{f}": len(val_dataset._buckets[(c, f)]) for c, f in val_dataset.bucket_keys},
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "code_state": "workspace has incomplete .git metadata; file hashes are authoritative",
    }
    (work_dir / "run_manifest.json").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n")
    metrics_path = work_dir / "metrics.jsonl"
    log_interval = int(logging_cfg.get("log_interval", 50))
    val_interval = int(logging_cfg.get("val_interval", 2000))
    save_interval = int(logging_cfg.get("save_interval", 5000))
    grad_clip = float(optim_cfg.get("grad_clip_norm", 1.0))
    max_overflows = int(logging_cfg.get("max_consecutive_amp_overflows", 8))
    keep_periodic = bool(logging_cfg.get("keep_periodic", True))
    save_visuals = bool(logging_cfg.get("save_validation_panels", True))

    def save_full_checkpoint(path: Path, step: int) -> None:
        atomic_torch_save(
            checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                step=step,
                best_val_loss=best_val_loss,
                config=config,
                config_path=args.config,
                train_manifest=train_manifest,
                val_manifest=val_manifest,
                run_id=run_id,
                amp_overflow_count=amp_overflow_count,
                consecutive_amp_overflows=consecutive_amp_overflows,
            ),
            path,
        )

    latest_val: dict[str, float] | None = None
    completed_step = start_step
    last_step_time = time.monotonic()
    with metrics_path.open("a") as metrics_log:
        metrics_log.write(json.dumps({"event": "start", **environment, "resume": str(args.resume or "")}, sort_keys=True) + "\n")
        metrics_log.flush()
        for raw_batch in train_loader:
            data_ready_time = time.monotonic()
            step = completed_step + 1
            batch = move_batch(raw_batch, device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                prediction = model(batch["input_stack"], batch["timestep"])
                losses = masked_diffusion_autoencoder_loss(
                    prediction,
                    batch["target"],
                    batch["masked_loss_mask"],
                    batch["visible_loss_mask"],
                    batch["sigma"],
                    batch["sigma_data"],
                    masked_weight=float(loss_cfg.get("masked_weight", 1.0)),
                    visible_weight=float(loss_cfg.get("visible_weight", 1.0)),
                    visible_weighting=str(loss_cfg.get("visible_weighting", "bounded_inverse_variance")),
                )
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError(f"Non-finite training loss at step {step}: {losses['loss'].item()}")
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            applied, consecutive_amp_overflows, scale_before, scale_after = amp_optimizer_step(
                scaler=scaler,
                optimizer=optimizer,
                scheduler=scheduler,
                grad_norm=grad_norm,
                consecutive_overflows=consecutive_amp_overflows,
                max_consecutive_overflows=max_overflows,
            )
            if not applied:
                amp_overflow_count += 1
                record = {
                    "event": "amp_overflow",
                    "step": step,
                    "grad_norm": float(grad_norm.item()),
                    "scale_before": scale_before,
                    "scale_after": scale_after,
                    "consecutive_overflows": consecutive_amp_overflows,
                    "total_overflows": amp_overflow_count,
                    "optimizer_step_applied": False,
                }
                metrics_log.write(json.dumps(record, sort_keys=True) + "\n")
                metrics_log.flush()
                print(json.dumps(record, sort_keys=True), flush=True)
            completed_step = step
            now = time.monotonic()
            if step == 1 or step % log_interval == 0:
                record = {
                    "event": "train",
                    "step": step,
                    "loss": float(losses["loss"].item()),
                    "masked_mse": float(losses["masked_mse"].item()),
                    "visible_mse": float(losses["visible_mse"].item()),
                    "weighted_visible_mse": float(losses["weighted_visible_mse"].item()),
                    "visible_noise_weight": float(losses["visible_noise_weight"].item()),
                    "mean_timestep": float(batch["timestep"].mean().item()),
                    "mean_sigma": float(batch["sigma"].mean().item()),
                    "grad_norm": float(grad_norm.item()),
                    "optimizer_step_applied": applied,
                    "amp_scale": scale_after,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "data_time": float(data_ready_time - last_step_time),
                    "step_time": float(now - data_ready_time),
                    "samples_per_second": float(batch_size / max(now - data_ready_time, 1e-8)),
                    "gpu_memory_allocated": int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0,
                }
                metrics_log.write(json.dumps(record, sort_keys=True) + "\n")
                metrics_log.flush()
                print(json.dumps(record, sort_keys=True), flush=True)
            last_step_time = now

            if val_interval > 0 and (step % val_interval == 0 or step == stop_after_step):
                latest_val, panel = validate(model, val_loader, device, loss_config=loss_cfg, amp_enabled=amp_enabled)
                val_record = {"event": "validation", "step": step, **latest_val}
                metrics_log.write(json.dumps(val_record, sort_keys=True) + "\n")
                metrics_log.flush()
                print(json.dumps(val_record, sort_keys=True), flush=True)
                if save_visuals and panel is not None:
                    save_panel(panel, work_dir / "figures" / f"validation_step_{step:06d}.png")
                if latest_val["loss"] < best_val_loss:
                    best_val_loss = latest_val["loss"]
                    save_full_checkpoint(work_dir / "best.ckpt", step)
                    for mode in SUPPORTED_TRANSFER_MODES:
                        report = export_mdae_initializer(
                            work_dir / "best.ckpt",
                            work_dir / f"best_{mode}_diagnostic_initializer.ckpt",
                            transfer_mode=mode,
                        )
                        (work_dir / f"best_{mode}_diagnostic_initializer.json").write_text(
                            json.dumps(report, indent=2, sort_keys=True) + "\n"
                        )

            if save_interval > 0 and (step % save_interval == 0 or step == stop_after_step):
                save_full_checkpoint(work_dir / "last.ckpt", step)
                if keep_periodic:
                    save_full_checkpoint(work_dir / f"ckpt_step_{step:06d}.ckpt", step)

    if completed_step == start_step:
        raise RuntimeError("training loop completed without consuming a batch")
    if latest_val is None:
        latest_val, panel = validate(model, val_loader, device, loss_config=loss_cfg, amp_enabled=amp_enabled)
        if save_visuals and panel is not None:
            save_panel(panel, work_dir / "figures" / f"validation_step_{completed_step:06d}.png")
    save_full_checkpoint(work_dir / "last.ckpt", completed_step)

    terminal_initializers: dict[str, str] = {}
    for mode in (str(transfer_cfg["primary_mode"]), str(transfer_cfg["diagnostic_mode"])):
        output = work_dir / f"terminal_{mode}_initializer.ckpt"
        report = export_mdae_initializer(work_dir / "last.ckpt", output, transfer_mode=mode)
        (work_dir / f"terminal_{mode}_initializer.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        terminal_initializers[mode] = str(output)
    summary = {
        "run_id": run_id,
        "status": "complete" if completed_step == total_steps else "planned_stop",
        "step": completed_step,
        "total_steps": total_steps,
        "selection_rule": "terminal_step",
        "best_val_loss_diagnostic_only": best_val_loss,
        "amp_overflow_count": amp_overflow_count,
        "consecutive_amp_overflows": consecutive_amp_overflows,
        "latest_validation": latest_val,
        "last_checkpoint": str(work_dir / "last.ckpt"),
        "terminal_initializers": terminal_initializers,
    }
    (work_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
