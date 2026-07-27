from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from past_mri.data.masked_diffusion_autoencoder_dataset import (
    MaskedDiffusionAutoencoderDataset,
    add_independent_gaussian_noise,
    foreground_robust_scale,
)
from past_mri.masked_diffusion_autoencoder import (
    MDAE2p5DModel,
    export_mdae_initializer,
    masked_diffusion_autoencoder_loss,
    sinusoidal_timestep_embedding,
    visible_noise_weight,
)
from past_mri.model_factory import build_cond_naf_unet_from_config
from past_mri.train_stage1 import validate_config


def _tiny_config() -> dict:
    return {
        "stack_depth": 5,
        "patch_size": 32,
        "batch_size": 2,
        "data": {"train_manifest": "train.jsonl", "val_manifest": "val.jsonl"},
        "mask": {
            "ratio_min": 0.01,
            "ratio_max": 0.99,
            "block_sizes": [8],
            "shared_across_all_slices": True,
        },
        "noise": {
            "schedule": "ve_linear",
            "scale_mode": "foreground_robust_std",
            "t_min": 0.0,
            "t_max": 1.0,
            "sigma_max_multiplier": 1.0,
            "independent_per_slice": True,
            "clip_noisy_input": False,
        },
        "model": {
            "name": "CondNAFUNet",
            "in_channels": 5,
            "width": 4,
            "enc_blocks": [1, 1, 1],
            "mid_blocks": 1,
            "dec_blocks": [1, 1, 1],
            "cond_dim": 16,
            "time_embedding_dim": 16,
            "output_mode": "direct_output",
            "use_calibration_head": False,
            "use_hyper_residual_head": False,
            "use_hyper_trunk": False,
        },
        "loss": {
            "masked_weight": 1.0,
            "visible_weight": 1.0,
            "visible_weighting": "bounded_inverse_variance",
        },
        "optim": {"total_steps": 10, "warmup_steps": 1, "min_lr_ratio": 0.1},
        "logging": {},
        "transfer": {
            "selection": "terminal_step",
            "primary_mode": "encoder_middle_core",
            "diagnostic_mode": "full_core",
        },
    }


def _write_volume(path: Path) -> None:
    yy, xx = np.mgrid[-1:1:64j, -1:1:64j]
    brain = ((xx * xx + yy * yy) < 0.75).astype(np.float32)
    volume = np.stack([brain * (0.35 + 0.04 * z) for z in range(7)], axis=2)
    np.save(path, volume.astype(np.float32))


def test_dual_corruption_dataset_is_deterministic_and_leakage_safe(tmp_path: Path) -> None:
    volume = tmp_path / "R_T1W_1.5T_0001.npy"
    _write_volume(volume)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"path": str(volume), "subject_id": "0001", "contrast": "T1W", "field": "1.5T"})
        + "\n"
    )
    dataset = MaskedDiffusionAutoencoderDataset(
        str(manifest),
        stack_depth=5,
        patch_size=32,
        virtual_samples=4,
        seed=19,
        cache_size=0,
        mask_config={
            "ratio_min": 0.25,
            "ratio_max": 0.75,
            "block_sizes": [8],
            "foreground_attempts": 16,
            "minimum_foreground_pixels_per_region": 8,
        },
        noise_config={
            "t_min": 0.2,
            "t_max": 0.8,
            "sigma_max_multiplier": 1.0,
            "sigma_data_min": 0.05,
            "sigma_data_max": 0.30,
        },
    )
    first = dataset[0]
    repeated = dataset[0]
    assert torch.equal(first["input_stack"], repeated["input_stack"])
    assert torch.equal(first["mask"], repeated["mask"])
    masked = first["mask"][0].bool()
    assert torch.count_nonzero(first["input_stack"][:, masked]) == 0
    assert first["masked_loss_mask"].sum() > 0
    assert first["visible_loss_mask"].sum() > 0
    assert torch.count_nonzero(first["masked_loss_mask"] * first["visible_loss_mask"]) == 0
    supervised = first["masked_loss_mask"] + first["visible_loss_mask"]
    assert torch.all(supervised <= 1.0)
    assert 0.0 < float(first["timestep"]) < 1.0
    assert float(first["sigma"]) > 0.0


def test_noise_is_independent_across_slices_and_scale_is_bounded() -> None:
    clean = np.full((5, 8, 8), 0.5, dtype=np.float32)
    noisy = add_independent_gaussian_noise(clean, sigma=0.1, seed=7)
    residual = noisy - clean
    assert not np.array_equal(residual[0], residual[1])
    foreground = np.ones((8, 8), dtype=bool)
    assert foreground_robust_scale(clean, foreground, minimum=0.05, maximum=0.30) == pytest.approx(0.05)


def test_model_loss_and_time_embedding_are_finite() -> None:
    torch.manual_seed(0)
    model = MDAE2p5DModel.from_config(_tiny_config())
    input_stack = torch.randn(2, 5, 32, 32)
    timestep = torch.tensor([0.1, 0.9])
    prediction = model(input_stack, timestep)
    target = torch.rand(2, 1, 32, 32)
    mask = torch.zeros_like(target)
    mask[:, :, :16] = 1.0
    visible = 1.0 - mask
    losses = masked_diffusion_autoencoder_loss(
        prediction,
        target,
        mask,
        visible,
        torch.tensor([0.01, 0.10]),
        torch.tensor([0.10, 0.10]),
        visible_weighting="bounded_inverse_variance",
    )
    losses["loss"].backward()
    assert prediction.shape == target.shape
    assert torch.isfinite(losses["loss"])
    assert model.reconstruction_head.weight.grad is not None
    embedding = sinusoidal_timestep_embedding(timestep, 17)
    assert embedding.shape == (2, 17)
    weights = visible_noise_weight(
        torch.tensor([0.0, 1.0]),
        torch.ones(2),
        mode="bounded_inverse_variance",
    )
    assert torch.allclose(weights, torch.tensor([1.0, 0.5]))


def test_transfer_exports_reset_conditioning_and_separate_decoder_arm(tmp_path: Path) -> None:
    config = _tiny_config()
    model = MDAE2p5DModel.from_config(config)
    source = tmp_path / "mdae.ckpt"
    torch.save({"model": model.state_dict(), "step": 11}, source)
    encoder_output = tmp_path / "encoder.ckpt"
    full_output = tmp_path / "full.ckpt"
    encoder_report = export_mdae_initializer(source, encoder_output, transfer_mode="encoder_middle_core")
    full_report = export_mdae_initializer(source, full_output, transfer_mode="full_core")
    encoder_state = torch.load(encoder_output, map_location="cpu", weights_only=False)["model"]
    full_state = torch.load(full_output, map_location="cpu", weights_only=False)["model"]

    assert encoder_report["step"] == 11
    assert full_report["transferred_keys"] > encoder_report["transferred_keys"]
    assert any(key.startswith("encoders.") for key in encoder_state)
    assert not any(key.startswith("decoders.") for key in encoder_state)
    assert any(key.startswith("decoders.") for key in full_state)
    for state in (encoder_state, full_state):
        assert not any("film1" in key or "film2" in key for key in state)
        assert not any(key.startswith("field_embed") for key in state)
        assert not any(key.startswith("condition_mlp") for key in state)
        downstream = build_cond_naf_unet_from_config(config)
        incompatible = downstream.load_state_dict(state, strict=False)
        assert not incompatible.unexpected_keys


def test_production_config_and_contract_validate() -> None:
    path = Path("configs/stage1_pretrain.yaml")
    config = yaml.safe_load(path.read_text())
    validate_config(config)
    broken = dict(config)
    broken["noise"] = {**config["noise"], "clip_noisy_input": True}
    with pytest.raises(ValueError, match="clip_noisy_input must be false"):
        validate_config(broken)


def test_trainer_strict_resume_and_terminal_exports(tmp_path: Path) -> None:
    volume = tmp_path / "R_T1W_1.5T_0001.npy"
    _write_volume(volume)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"path": str(volume), "subject_id": "0001", "contrast": "T1W", "field": "1.5T"})
        + "\n"
    )
    config = _tiny_config()
    config.update({"experiment_name": "tiny-mdae-resume", "seed": 23, "num_workers": 0, "amp": False})
    config["data"] = {
        "train_manifest": str(manifest),
        "val_manifest": str(manifest),
        "cache_size": 0,
        "val_samples_per_bucket": 1,
        "val_batch_size": 1,
        "foreground_threshold": 0.01,
        "minimum_foreground_fraction": 0.03,
        "crop_attempts": 2,
    }
    config["mask"].update(
        {
            "foreground_attempts": 8,
            "minimum_foreground_pixels_per_region": 4,
            "fill_value": 0.0,
        }
    )
    config["noise"].update({"sigma_data_min": 0.05, "sigma_data_max": 0.30})
    config["optim"] = {
        "name": "AdamW",
        "lr": 1e-3,
        "betas": [0.9, 0.999],
        "weight_decay": 0.0,
        "warmup_steps": 0,
        "total_steps": 4,
        "min_lr_ratio": 0.1,
        "grad_clip_norm": 1.0,
    }
    config["logging"] = {
        "log_interval": 1,
        "val_interval": 2,
        "save_interval": 2,
        "keep_periodic": False,
        "save_validation_panels": False,
        "max_consecutive_amp_overflows": 2,
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    work_dir = tmp_path / "run"
    command = [
        sys.executable,
        "-m",
        "past_mri.train_stage1",
        "--config",
        str(config_path),
        "--work-dir",
        str(work_dir),
        "--device",
        "cpu",
    ]
    subprocess.run([*command, "--stop-after-step", "2"], check=True, capture_output=True, text=True)
    first = torch.load(work_dir / "last.ckpt", map_location="cpu", weights_only=False)
    assert first["format"] == "task3_mdae2p5d_full_resume_v1"
    assert first["step"] == 2
    assert {"model", "optimizer", "scheduler", "scaler", "rng_state"}.issubset(first)
    subprocess.run(
        [*command, "--resume", str(work_dir / "last.ckpt"), "--stop-after-step", "4"],
        check=True,
        capture_output=True,
        text=True,
    )
    resumed = torch.load(work_dir / "last.ckpt", map_location="cpu", weights_only=False)
    summary = json.loads((work_dir / "summary.json").read_text())
    assert resumed["step"] == 4
    assert summary["status"] == "complete"
    assert Path(summary["terminal_initializers"]["encoder_middle_core"]).is_file()
    assert Path(summary["terminal_initializers"]["full_core"]).is_file()
