#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from past_mri.config import load_yaml_config
from past_mri.masked_diffusion_autoencoder import MDAE2p5DModel
from past_mri.model_factory import build_cond_naf_unet_from_config
from past_mri.train_stage2 import load_model_weights


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify PAST-MRI initializer and inference checkpoints.")
    parser.add_argument("--stage", choices=("stage1", "stage2"), default="stage2")
    parser.add_argument("--config", type=Path, default=Path("configs/stage2_translation.yaml"))
    parser.add_argument("--initializer", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    model = (
        MDAE2p5DModel.from_config(config)
        if args.stage == "stage1"
        else build_cond_naf_unet_from_config(config)
    )
    report: dict[str, object] = {
        "model": type(model).__name__,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "state_keys": len(model.state_dict()),
    }
    if args.initializer is not None:
        if args.stage != "stage2":
            parser.error("--initializer is only valid for --stage stage2")
        initialization = load_model_weights(
            model,
            args.initializer,
            torch.device("cpu"),
            strict=False,
        )
        report["initializer"] = {
            "loaded_keys": initialization["loaded_keys"],
            "unexpected": len(initialization["skipped_unexpected_keys"]),
            "shape_mismatch": len(initialization["skipped_shape_mismatch"]),
        }
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state = checkpoint.get("model", checkpoint)
        model.load_state_dict(state, strict=True)
        report["checkpoint"] = {
            "step": int(checkpoint.get("step", -1)),
            "strict_load": True,
            "state_keys": len(state),
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
