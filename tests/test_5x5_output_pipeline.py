from __future__ import annotations

import json
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
import torch


def test_cond_naf_unet_5x5_emits_five_supervised_slices():
    from past_mri.models import CondNAFUNet5x5

    model = CondNAFUNet5x5(
        in_channels=5,
        output_channels=5,
        width=8,
        enc_blocks=[1],
        mid_blocks=1,
        dec_blocks=[1],
        cond_dim=32,
        use_calibration_head=False,
        use_hyper_residual_head=True,
        hyper_residual_mode="full_kernel_delta",
    )
    source = torch.rand(2, 5, 24, 32)
    ids = (
        torch.tensor([0, 1]),
        torch.tensor([4, 3]),
        torch.tensor([0, 2]),
    )

    output = model(source, *ids, base_center=source)

    assert output.pred.shape == source.shape
    assert output.residual.shape == source.shape
    assert output.gate.shape == source.shape
    assert output.alpha.shape == (2, 5, 1, 1)
    assert torch.isfinite(output.pred).all()


def test_5x1_checkpoint_uses_explicit_trunk_only_migration(tmp_path):
    from past_mri.models import CondNAFUNet, CondNAFUNet5x5
    from past_mri.train_stage2 import load_model_weights

    common = {
        "in_channels": 5,
        "width": 8,
        "enc_blocks": [1],
        "mid_blocks": 1,
        "dec_blocks": [1],
        "cond_dim": 32,
        "use_calibration_head": False,
        "use_hyper_residual_head": True,
        "hyper_residual_mode": "full_kernel_delta",
    }
    source = CondNAFUNet(**common)
    target = CondNAFUNet5x5(output_channels=5, **common)
    with torch.no_grad():
        source.stem.weight.fill_(0.125)
    target_head_before = target.residual_head.weight.detach().clone()
    checkpoint = tmp_path / "five_to_one.ckpt"
    torch.save({"model": source.state_dict(), "step": 40000}, checkpoint)

    strict_target = CondNAFUNet5x5(output_channels=5, **common)
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_model_weights(strict_target, checkpoint, torch.device("cpu"), strict=True)

    report = load_model_weights(target, checkpoint, torch.device("cpu"), strict=False)

    assert report["step"] == 40000
    assert report["strict"] is False
    assert torch.all(target.stem.weight == 0.125)
    assert torch.equal(target.residual_head.weight, target_head_before)
    mismatch_keys = {item["key"] for item in report["skipped_shape_mismatch"]}
    assert "residual_head.weight" in mismatch_keys
    assert "gate_head.weight" in mismatch_keys
    assert any(key.startswith("alpha_mlp.") and key.endswith(".weight") for key in mismatch_keys)


def test_paired_dataset_builds_aligned_five_slice_targets(tmp_path):
    from past_mri.data.paired_translation_dataset import PairedTranslationDataset

    source = np.zeros((8, 8, 9), dtype=np.float32)
    target = np.zeros_like(source)
    for z in range(source.shape[2]):
        source[:, :, z] = z / 20.0
        target[:, :, z] = 0.4 + z / 100.0
    source_path = tmp_path / "source.npy"
    target_path = tmp_path / "target.npy"
    np.save(source_path, source)
    np.save(target_path, target)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "source_path": str(source_path),
                "target_path": str(target_path),
                "source_field": "0.1T",
                "target_field": "7T",
                "contrast": "T2W",
                "subject_id": "synthetic",
                "z_start": 2,
                "z_end": 7,
            }
        )
        + "\n"
    )
    dataset = PairedTranslationDataset(
        manifest,
        stack_depth=5,
        output_depth=5,
        patch_size=None,
        identity_prob=0.0,
        pair_balanced=False,
        cache_size=0,
        seed=3,
    )

    sample = dataset[0]

    assert sample["source_stack"].shape == (5, 8, 8)
    assert sample["source_target"].shape == (5, 8, 8)
    assert sample["target"].shape == (5, 8, 8)
    assert sample["base_center"].shape == (5, 8, 8)
    assert torch.equal(sample["source_stack"], sample["source_target"])
    target_values = sample["target"][:, 0, 0]
    assert torch.all(target_values[1:] >= target_values[:-1])


class _IdentityFiveSliceModel(torch.nn.Module):
    in_channels = 5
    output_channels = 5

    def forward(self, source_stack, *_ids, base_center=None):
        return SimpleNamespace(pred=source_stack)


def test_overlap_fusion_preserves_physical_slice_mapping_at_boundaries():
    from past_mri.infer import infer_slab

    volume = np.zeros((16, 16, 11), dtype=np.float32)
    for z in range(volume.shape[2]):
        volume[:, :, z] = z / 10.0
    expected = volume[:, :, 2:9]
    model = _IdentityFiveSliceModel()

    uniform = infer_slab(
        model,
        volume,
        "0.1T",
        "7T",
        "T2W",
        2,
        9,
        torch.device("cpu"),
        output_fusion="uniform",
    )
    weighted = infer_slab(
        model,
        volume,
        "0.1T",
        "7T",
        "T2W",
        2,
        9,
        torch.device("cpu"),
        output_fusion="center_weighted",
    )

    np.testing.assert_allclose(uniform, expected, atol=1e-6)
    np.testing.assert_allclose(weighted, expected, atol=1e-6)


def test_pair_index_selection_is_explicit_and_rejects_ambiguous_specs():
    from past_mri.infer import parse_pair_indices

    assert parse_pair_indices("0,1,2,3,7,11,15", 20) == [0, 1, 2, 3, 7, 11, 15]
    with pytest.raises(ValueError, match="duplicate"):
        parse_pair_indices("0,3,3", 20)
    with pytest.raises(ValueError, match="out of range"):
        parse_pair_indices("20", 20)


def test_full_volume_slab_override_preserves_exact_submitted_values(tmp_path):
    from past_mri.infer import apply_prediction_slab_override

    affine = np.eye(4, dtype=np.float64)
    full_image = nib.Nifti1Image(np.zeros((8, 10, 12), dtype=np.float32), affine)
    prediction = np.zeros((8, 10, 8), dtype=np.float32)
    override_data = np.full((8, 10, 3), 0.625, dtype=np.float32)
    override_image = full_image.slicer[:, :, 5:8]
    override_path = tmp_path / "submitted_slab.nii.gz"
    nib.save(nib.Nifti1Image(override_data, override_image.affine, override_image.header), override_path)

    result = apply_prediction_slab_override(
        prediction,
        full_image,
        override_path,
        prediction_z_start=2,
        override_z_start=5,
        override_z_end=8,
    )

    np.testing.assert_array_equal(result[:, :, 3:6], override_data)
    assert np.count_nonzero(result[:, :, :3]) == 0
    assert np.count_nonzero(result[:, :, 6:]) == 0


def test_multislice_nrmse_is_mean_of_slice_metrics():
    from past_mri.losses import _slice_batch, masked_nrmse_loss

    target = torch.stack([torch.ones(4, 4), 2.0 * torch.ones(4, 4)], dim=0)[None]
    pred = torch.stack([torch.zeros(4, 4), torch.ones(4, 4)], dim=0)[None]
    loss = masked_nrmse_loss(_slice_batch(pred), _slice_batch(target))
    expected = torch.tensor((1.0 + 0.5) / 2.0)

    assert torch.allclose(loss, expected, atol=1e-5)
