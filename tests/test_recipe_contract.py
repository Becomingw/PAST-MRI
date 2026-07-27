from __future__ import annotations

from pathlib import Path
import json

import yaml

from past_mri.calibration import DEFAULT_PERCENTILES
from past_mri.constants import CONTRASTS, FIELDS
from past_mri.model_factory import build_cond_naf_unet_from_config


ROOT = Path(__file__).resolve().parents[1]


def test_stage1_public_recipe_contract() -> None:
    config = yaml.safe_load((ROOT / "configs/stage1_pretrain.yaml").read_text())
    assert config["model"]["name"] == "CondNAFUNet"
    assert config["model"]["width"] == 128
    assert config["optim"]["total_steps"] == 40000
    assert config["transfer"]["primary_mode"] == "encoder_middle_core"


def test_stage2_public_recipe_contract(tmp_path: Path) -> None:
    config = yaml.safe_load((ROOT / "configs/stage2_translation.yaml").read_text())
    percentiles = {f"{value:g}": float(value / 100.0) for value in DEFAULT_PERCENTILES}
    stats = {
        contrast: {
            field: {"percentiles": percentiles, "files": 1, "sampled_voxels": 1}
            for field in FIELDS
        }
        for contrast in CONTRASTS
    }
    stats_path = tmp_path / "target_stats.json"
    stats_path.write_text(json.dumps({"stats": stats}))
    config["base_calibration"]["target_stats_path"] = str(stats_path)
    config["model"]["target_stats_path"] = str(stats_path)
    model = build_cond_naf_unet_from_config(config)
    assert type(model).__name__ == "CondNAFUNet5x5"
    assert model.in_channels == 5
    assert model.output_channels == 5
    assert sum(parameter.numel() for parameter in model.parameters()) == 145_546_501
    assert config["optim"]["total_steps"] == 40000
    assert config["ema"]["decay"] == 0.9999
    assert config["inference"]["output_fusion"] == "uniform"
