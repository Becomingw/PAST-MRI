from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from past_mri.models.cond_naf_unet import ModelOutput


def _weighted_mean(per_sample: torch.Tensor, weight: torch.Tensor | None, *, name: str) -> torch.Tensor:
    per_sample = per_sample.flatten()
    if weight is None:
        return per_sample.mean()
    weight = weight.to(device=per_sample.device, dtype=per_sample.dtype).flatten()
    if weight.numel() != per_sample.numel():
        raise ValueError(f"{name} weight must have one value per sample: got {weight.shape}, batch={per_sample.numel()}")
    return (per_sample * weight).mean()


def _slice_batch(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"expected BCHW tensor, got {tuple(x.shape)}")
    return x.reshape(x.shape[0] * x.shape[1], 1, x.shape[2], x.shape[3])


def _slice_weight(weight: torch.Tensor | None, channels: int) -> torch.Tensor | None:
    if weight is None or channels == 1:
        return weight
    return weight.flatten().repeat_interleave(channels)


def charbonnier_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-3,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    per_sample = torch.sqrt((pred - target).pow(2) + eps * eps).flatten(1).mean(dim=1)
    return _weighted_mean(per_sample, weight, name="charbonnier")


def gradient_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    tgt_dx = target[..., :, 1:] - target[..., :, :-1]
    tgt_dy = target[..., 1:, :] - target[..., :-1, :]
    per_sample = (pred_dx - tgt_dx).abs().flatten(1).mean(dim=1) + (pred_dy - tgt_dy).abs().flatten(1).mean(dim=1)
    return _weighted_mean(per_sample, weight, name="gradient")


def laplacian_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    kernel = pred.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
    pred_lap = F.conv2d(pred, kernel, padding=1)
    target_lap = F.conv2d(target, kernel, padding=1)
    per_sample = (pred_lap - target_lap).abs().flatten(1).mean(dim=1)
    return _weighted_mean(per_sample, weight, name="laplacian")


def highfreq(x: torch.Tensor, kernel_size: int = 9) -> torch.Tensor:
    if kernel_size <= 1:
        return x
    if kernel_size % 2 == 0:
        raise ValueError("highfreq kernel_size must be odd")
    pad = kernel_size // 2
    low = F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), kernel_size, stride=1)
    return x - low


def source_highfreq_hinge_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    source: torch.Tensor,
    kernel_size: int = 9,
    margin: float = 0.0,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    pred_hf = highfreq(pred, kernel_size)
    target_hf = highfreq(target, kernel_size)
    source_hf = highfreq(source, kernel_size)
    pred_err = (pred_hf - source_hf).abs().flatten(1).mean(dim=1)
    target_err = (target_hf - source_hf).abs().flatten(1).mean(dim=1)
    per_sample = torch.relu(pred_err - target_err - float(margin))
    return _weighted_mean(per_sample, weight, name="source_highfreq_hinge")


def target_highfreq_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    kernel_size: int = 9,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    pred_hf = highfreq(pred, kernel_size)
    target_hf = highfreq(target, kernel_size)
    per_sample = (pred_hf - target_hf).abs().flatten(1).mean(dim=1)
    return _weighted_mean(per_sample, weight, name="target_highfreq")


def fourier_highfreq_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    cutoff: float = 0.18,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if not 0.0 <= cutoff < 1.0:
        raise ValueError("fourier_highfreq cutoff must be in [0, 1)")
    if pred.shape != target.shape:
        raise ValueError(f"fourier_highfreq shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")
    height, width = pred.shape[-2:]
    fy = torch.fft.fftfreq(height, device=pred.device, dtype=pred.dtype).view(height, 1)
    fx = torch.fft.rfftfreq(width, device=pred.device, dtype=pred.dtype).view(1, width // 2 + 1)
    radius = torch.sqrt(fy.square() + fx.square())
    norm_radius = radius / radius.max().clamp_min(1e-6)
    high_mask = norm_radius >= float(cutoff)
    if not torch.any(high_mask):
        return pred.new_tensor(0.0)
    pred_mag = torch.log1p(torch.fft.rfft2(pred, norm="ortho").abs())
    target_mag = torch.log1p(torch.fft.rfft2(target, norm="ortho").abs())
    per_sample = (pred_mag[..., high_mask] - target_mag[..., high_mask]).abs().flatten(1).mean(dim=1)
    return _weighted_mean(per_sample, weight, name="fourier_highfreq")


def tissue_ce_loss(
    output: ModelOutput,
    tissue_target: torch.Tensor,
    has_tissue_target: torch.Tensor,
) -> torch.Tensor:
    if output.tissue_logits is None:
        return tissue_target.new_tensor(0.0, dtype=torch.float32)
    logits = output.tissue_logits
    target = tissue_target.to(device=logits.device, dtype=torch.long)
    has_target = has_tissue_target.to(device=logits.device, dtype=torch.bool).flatten()
    if target.ndim != 3:
        raise ValueError(f"tissue_target must be [B,H,W], got {tuple(target.shape)}")
    if target.shape[0] != logits.shape[0] or target.shape[-2:] != logits.shape[-2:]:
        raise ValueError(f"tissue/logits shape mismatch: target={tuple(target.shape)}, logits={tuple(logits.shape)}")
    per_pixel = F.cross_entropy(logits, target.clamp_min(0), reduction="none")
    per_sample = per_pixel.flatten(1).mean(dim=1)
    if has_target.numel() != per_sample.numel():
        raise ValueError(
            f"has_tissue_target must have one value per sample: got {has_target.shape}, batch={per_sample.numel()}"
        )
    if not torch.any(has_target):
        return logits.new_tensor(0.0)
    return per_sample[has_target].mean()


def masked_nrmse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask_eps: float = 1e-4,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    mask = (target > mask_eps).to(dtype=pred.dtype)
    count = mask.flatten(1).sum(dim=1).clamp_min(1.0)
    diff_mse = ((pred - target).pow(2) * mask).flatten(1).sum(dim=1) / count
    target_power = (target.pow(2) * mask).flatten(1).sum(dim=1) / count
    per_sample = torch.sqrt(diff_mse + 1e-8) / torch.sqrt(target_power + 1e-8)
    return _weighted_mean(per_sample, weight, name="masked_nrmse")


def relative_l2_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    diff_norm = (pred - target).pow(2).flatten(1).sum(dim=1).clamp_min(eps).sqrt()
    target_norm = target.pow(2).flatten(1).sum(dim=1).clamp_min(eps).sqrt()
    per_sample = diff_norm / target_norm
    return _weighted_mean(per_sample, weight, name="relative_l2")


def percentile_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    quantiles = pred.new_tensor([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    pred_q = torch.quantile(pred.flatten(1).clamp(0, 1), quantiles, dim=1).T
    target_q = torch.quantile(target.flatten(1).clamp(0, 1), quantiles, dim=1).T
    per_sample = (pred_q - target_q).abs().mean(dim=1)
    return _weighted_mean(per_sample, weight, name="percentile")


def ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    *,
    data_range_mode: str = "unit",
    data_range_eps: float = 1e-6,
) -> torch.Tensor:
    if data_range_mode == "unit":
        c1 = pred.new_tensor(0.01**2)
        c2 = pred.new_tensor(0.03**2)
    elif data_range_mode in {"target", "target_dynamic"}:
        target_flat = target.flatten(1)
        data_range = (target_flat.max(dim=1).values - target_flat.min(dim=1).values).clamp_min(data_range_eps)
        data_range = data_range.view(-1, 1, 1, 1)
        c1 = (0.01 * data_range).pow(2)
        c2 = (0.03 * data_range).pow(2)
    else:
        raise ValueError(f"Unknown ssim_data_range_mode: {data_range_mode}")
    mu_x = F.avg_pool2d(pred, 7, stride=1, padding=3)
    mu_y = F.avg_pool2d(target, 7, stride=1, padding=3)
    sigma_x = F.avg_pool2d(pred * pred, 7, stride=1, padding=3) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, 7, stride=1, padding=3) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, 7, stride=1, padding=3) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2) + 1e-8
    )
    per_sample = 1.0 - ssim.clamp(0, 1).flatten(1).mean(dim=1)
    return _weighted_mean(per_sample, weight, name="msssim")


def lowfreq_calibration_delta_loss(
    output: ModelOutput,
    target: torch.Tensor,
    kernel_size: int = 9,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if output.calibration_anchor is None or output.calibration_delta is None:
        return target.new_tensor(0.0)
    anchor = output.calibration_anchor
    delta = output.calibration_delta
    if kernel_size > 1:
        pad = kernel_size // 2
        anchor = F.avg_pool2d(F.pad(anchor, (pad, pad, pad, pad), mode="reflect"), kernel_size, stride=1)
        delta = F.avg_pool2d(F.pad(delta, (pad, pad, pad, pad), mode="reflect"), kernel_size, stride=1)
        target = F.avg_pool2d(F.pad(target, (pad, pad, pad, pad), mode="reflect"), kernel_size, stride=1)
    per_sample = (delta - (target - anchor)).abs().flatten(1).mean(dim=1)
    if weight is not None:
        weight = weight.to(device=target.device, dtype=target.dtype).flatten()
        if weight.numel() != per_sample.numel():
            raise ValueError(
                "calibration_lowfreq_weight must have one value per sample: "
                f"got {weight.shape}, batch={per_sample.numel()}"
            )
        per_sample = per_sample * weight
    return per_sample.mean()


def extra_residual_delta_loss(
    output: ModelOutput,
    residual_target: torch.Tensor,
    has_residual_target: torch.Tensor,
    kernel_size: int = 1,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if output.calibration_extra_delta is None:
        return residual_target.new_tensor(0.0)
    delta = output.calibration_extra_delta
    target = residual_target.to(device=delta.device, dtype=delta.dtype)
    has_target = has_residual_target.to(device=delta.device, dtype=torch.bool).flatten()
    if target.shape != delta.shape:
        raise ValueError(f"residual_target shape mismatch: target={tuple(target.shape)}, delta={tuple(delta.shape)}")
    if has_target.numel() != delta.shape[0]:
        raise ValueError(
            "has_residual_target must have one value per sample: "
            f"got {has_target.shape}, batch={delta.shape[0]}"
        )
    if not torch.any(has_target):
        return delta.new_tensor(0.0)
    if kernel_size > 1:
        if kernel_size % 2 == 0:
            raise ValueError("extra_residual_kernel_size must be odd")
        pad = kernel_size // 2
        delta = F.avg_pool2d(F.pad(delta, (pad, pad, pad, pad), mode="reflect"), kernel_size, stride=1)
        target = F.avg_pool2d(F.pad(target, (pad, pad, pad, pad), mode="reflect"), kernel_size, stride=1)
    per_sample = (delta - target).abs().flatten(1).mean(dim=1)
    if weight is not None:
        weight = weight.to(device=delta.device, dtype=delta.dtype).flatten()
        if weight.numel() != per_sample.numel():
            raise ValueError(
                "extra_residual_weight must have one value per sample: "
                f"got {weight.shape}, batch={per_sample.numel()}"
            )
        per_sample = per_sample * weight
    return per_sample[has_target].mean()


class Task3DistillLoss(nn.Module):
    def __init__(
        self,
        charbonnier: float = 1.0,
        msssim: float = 0.3,
        lpips: float = 0.05,
        gradient: float = 0.05,
        laplacian: float = 0.0,
        masked_nrmse: float = 0.0,
        relative_l2: float = 0.0,
        relative_l2_eps: float = 1e-8,
        percentile: float = 0.0,
        residual_reg: float = 0.01,
        calibration_lowfreq: float = 0.0,
        calibration_lowfreq_kernel_size: int = 9,
        extra_residual: float = 0.0,
        extra_residual_kernel_size: int = 1,
        source_highfreq_hinge: float = 0.0,
        source_highfreq_kernel_size: int = 9,
        source_highfreq_margin: float = 0.0,
        target_highfreq: float = 0.0,
        target_highfreq_kernel_size: int = 9,
        fourier_highfreq: float = 0.0,
        fourier_highfreq_cutoff: float = 0.18,
        ssim_data_range_mode: str = "unit",
        ssim_data_range_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if calibration_lowfreq_kernel_size < 1:
            raise ValueError("calibration_lowfreq_kernel_size must be >= 1")
        if calibration_lowfreq_kernel_size % 2 == 0:
            raise ValueError("calibration_lowfreq_kernel_size must be odd")
        if extra_residual_kernel_size < 1:
            raise ValueError("extra_residual_kernel_size must be >= 1")
        if extra_residual_kernel_size % 2 == 0:
            raise ValueError("extra_residual_kernel_size must be odd")
        if source_highfreq_kernel_size < 1:
            raise ValueError("source_highfreq_kernel_size must be >= 1")
        if source_highfreq_kernel_size % 2 == 0:
            raise ValueError("source_highfreq_kernel_size must be odd")
        if target_highfreq_kernel_size < 1:
            raise ValueError("target_highfreq_kernel_size must be >= 1")
        if target_highfreq_kernel_size % 2 == 0:
            raise ValueError("target_highfreq_kernel_size must be odd")
        self.weights = {
            "charbonnier": charbonnier,
            "msssim": msssim,
            "lpips": lpips,
            "gradient": gradient,
            "laplacian": laplacian,
            "masked_nrmse": masked_nrmse,
            "relative_l2": relative_l2,
            "percentile": percentile,
            "residual_reg": residual_reg,
            "calibration_lowfreq": calibration_lowfreq,
            "extra_residual": extra_residual,
            "source_highfreq_hinge": source_highfreq_hinge,
            "target_highfreq": target_highfreq,
            "fourier_highfreq": fourier_highfreq,
        }
        self.calibration_lowfreq_kernel_size = int(calibration_lowfreq_kernel_size)
        self.extra_residual_kernel_size = int(extra_residual_kernel_size)
        self.source_highfreq_kernel_size = int(source_highfreq_kernel_size)
        self.source_highfreq_margin = float(source_highfreq_margin)
        self.target_highfreq_kernel_size = int(target_highfreq_kernel_size)
        self.fourier_highfreq_cutoff = float(fourier_highfreq_cutoff)
        self.relative_l2_eps = float(relative_l2_eps)
        self.ssim_data_range_mode = str(ssim_data_range_mode)
        self.ssim_data_range_eps = float(ssim_data_range_eps)
        self.lpips_model = None
        if lpips > 0:
            try:
                import lpips as lpips_lib

                self.lpips_model = lpips_lib.LPIPS(net="alex")
                for param in self.lpips_model.parameters():
                    param.requires_grad_(False)
                self.lpips_model.eval()
            except Exception:
                self.lpips_model = None

    def forward(
        self,
        output: ModelOutput,
        target: torch.Tensor,
        source: torch.Tensor | None = None,
        residual_target: torch.Tensor | None = None,
        has_residual_target: torch.Tensor | None = None,
        residual_reg_weight: torch.Tensor | None = None,
        calibration_lowfreq_weight: torch.Tensor | None = None,
        term_weights: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        pred = output.pred
        if pred.shape != target.shape:
            raise ValueError(f"prediction/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
        channels = int(pred.shape[1])
        pred_slices = _slice_batch(pred)
        target_slices = _slice_batch(target)
        source_slices = _slice_batch(source) if source is not None else None
        term_weights = term_weights or {}
        slice_weights = {name: _slice_weight(weight, channels) for name, weight in term_weights.items()}
        terms_t: dict[str, torch.Tensor] = {}
        terms_t["loss_charbonnier"] = charbonnier_loss(
            pred_slices,
            target_slices,
            weight=slice_weights.get("charbonnier"),
        )
        terms_t["loss_msssim"] = (
            ssim_loss(
                pred_slices,
                target_slices,
                weight=slice_weights.get("msssim"),
                data_range_mode=self.ssim_data_range_mode,
                data_range_eps=self.ssim_data_range_eps,
            )
            if self.weights["msssim"]
            else pred.new_tensor(0.0)
        )
        if self.weights["lpips"] and self.lpips_model is not None:
            lp = self.lpips_model.to(pred.device)
            pred_lp = pred_slices.repeat(1, 3, 1, 1) * 2.0 - 1.0
            target_lp = target_slices.repeat(1, 3, 1, 1) * 2.0 - 1.0
            lp_per_sample = lp(pred_lp, target_lp).flatten(1).mean(dim=1)
            terms_t["loss_lpips"] = _weighted_mean(lp_per_sample, slice_weights.get("lpips"), name="lpips")
        else:
            terms_t["loss_lpips"] = pred.new_tensor(0.0)
        terms_t["loss_gradient"] = (
            gradient_loss(pred_slices, target_slices, weight=slice_weights.get("gradient"))
            if self.weights["gradient"]
            else pred.new_tensor(0.0)
        )
        terms_t["loss_laplacian"] = (
            laplacian_loss(pred_slices, target_slices, weight=slice_weights.get("laplacian"))
            if self.weights["laplacian"]
            else pred.new_tensor(0.0)
        )
        terms_t["loss_masked_nrmse"] = (
            masked_nrmse_loss(pred_slices, target_slices, weight=slice_weights.get("masked_nrmse"))
            if self.weights["masked_nrmse"]
            else pred.new_tensor(0.0)
        )
        terms_t["loss_relative_l2"] = (
            relative_l2_loss(pred_slices, target_slices, eps=self.relative_l2_eps, weight=slice_weights.get("relative_l2"))
            if self.weights["relative_l2"]
            else pred.new_tensor(0.0)
        )
        terms_t["loss_percentile"] = (
            percentile_loss(pred_slices, target_slices, weight=slice_weights.get("percentile"))
            if self.weights["percentile"]
            else pred.new_tensor(0.0)
        )
        terms_t["loss_calibration_lowfreq"] = (
            lowfreq_calibration_delta_loss(
                output,
                target,
                self.calibration_lowfreq_kernel_size,
                calibration_lowfreq_weight,
            )
            if self.weights["calibration_lowfreq"]
            else pred.new_tensor(0.0)
        )
        terms_t["loss_extra_residual"] = (
            extra_residual_delta_loss(
                output,
                residual_target,
                has_residual_target,
                self.extra_residual_kernel_size,
                term_weights.get("extra_residual"),
            )
            if self.weights["extra_residual"] and residual_target is not None and has_residual_target is not None
            else pred.new_tensor(0.0)
        )
        terms_t["loss_source_highfreq_hinge"] = (
            source_highfreq_hinge_loss(
                pred_slices,
                target_slices,
                source_slices,
                self.source_highfreq_kernel_size,
                self.source_highfreq_margin,
                weight=slice_weights.get("source_highfreq_hinge"),
            )
            if self.weights["source_highfreq_hinge"] and source is not None
            else pred.new_tensor(0.0)
        )
        terms_t["loss_target_highfreq"] = (
            target_highfreq_loss(
                pred_slices,
                target_slices,
                self.target_highfreq_kernel_size,
                weight=slice_weights.get("target_highfreq"),
            )
            if self.weights["target_highfreq"]
            else pred.new_tensor(0.0)
        )
        terms_t["loss_fourier_highfreq"] = (
            fourier_highfreq_loss(
                pred_slices,
                target_slices,
                self.fourier_highfreq_cutoff,
                weight=slice_weights.get("fourier_highfreq"),
            )
            if self.weights["fourier_highfreq"]
            else pred.new_tensor(0.0)
        )
        residual_per_sample = output.residual.abs().flatten(1).mean(dim=1)
        if residual_reg_weight is not None:
            residual_reg_weight = residual_reg_weight.to(device=pred.device, dtype=pred.dtype).flatten()
            if residual_reg_weight.numel() != residual_per_sample.numel():
                raise ValueError(
                    "residual_reg_weight must have one value per sample: "
                    f"got {residual_reg_weight.shape}, batch={residual_per_sample.numel()}"
                )
            residual_per_sample = residual_per_sample * residual_reg_weight
        terms_t["loss_residual_reg"] = residual_per_sample.mean()

        total = (
            self.weights["charbonnier"] * terms_t["loss_charbonnier"]
            + self.weights["msssim"] * terms_t["loss_msssim"]
            + self.weights["lpips"] * terms_t["loss_lpips"]
            + self.weights["gradient"] * terms_t["loss_gradient"]
            + self.weights["laplacian"] * terms_t["loss_laplacian"]
            + self.weights["masked_nrmse"] * terms_t["loss_masked_nrmse"]
            + self.weights["relative_l2"] * terms_t["loss_relative_l2"]
            + self.weights["percentile"] * terms_t["loss_percentile"]
            + self.weights["calibration_lowfreq"] * terms_t["loss_calibration_lowfreq"]
            + self.weights["extra_residual"] * terms_t["loss_extra_residual"]
            + self.weights["source_highfreq_hinge"] * terms_t["loss_source_highfreq_hinge"]
            + self.weights["target_highfreq"] * terms_t["loss_target_highfreq"]
            + self.weights["fourier_highfreq"] * terms_t["loss_fourier_highfreq"]
            + self.weights["residual_reg"] * terms_t["loss_residual_reg"]
        )
        terms = {k: float(v.detach().cpu()) for k, v in terms_t.items()}
        terms["loss_total"] = float(total.detach().cpu())
        return total, terms
