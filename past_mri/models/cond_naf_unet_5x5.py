from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .cond_naf_unet import CondNAFUNet, ModelOutput


class CondNAFUNet5x5(CondNAFUNet):
    """True five-slice output variant of CondNAFUNet with a shared 2D trunk."""

    def __init__(self, *args, output_channels: int = 5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.output_channels = int(output_channels)
        if self.output_channels != self.in_channels or self.output_channels % 2 != 1:
            raise ValueError("CondNAFUNet5x5 requires odd output_channels matching in_channels")
        if self.use_tissue_bridge or self.use_tissue_local_bridge:
            raise ValueError("Tissue bridges are not supported by CondNAFUNet5x5")

        width = self.residual_head.in_channels
        self.residual_head = nn.Conv2d(width, self.output_channels, 3, padding=1)
        self.gate_head = nn.Conv2d(width, self.output_channels, 3, padding=1)
        self.alpha_mlp[-1] = nn.Linear(self.alpha_mlp[-1].in_features, self.output_channels)

        if self.use_hyper_residual_head:
            params_per_output = width + 1 if self.hyper_residual_mode == "adapter" else width * 9 + 1
            self.hyper_residual_mlp[-1] = nn.Linear(
                self.hyper_residual_mlp[-1].in_features,
                self.output_channels * params_per_output,
            )
            nn.init.zeros_(self.hyper_residual_mlp[-1].weight)
            nn.init.zeros_(self.hyper_residual_mlp[-1].bias)

        for name in ("local_lowfreq_field_head", "local_lowfreq_gain_head", "local_lowfreq_bias_head"):
            if hasattr(self, name):
                old = getattr(self, name)
                head = nn.Conv2d(old.in_channels, self.output_channels, old.kernel_size, padding=old.padding)
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
                setattr(self, name, head)

        if hasattr(self, "lowres_calibration_field_head"):
            old = self.lowres_calibration_field_head[-1]
            channels = self.output_channels if self.lowres_field_mode == "additive" else 2 * self.output_channels
            head = nn.Conv2d(old.in_channels, channels, old.kernel_size, padding=old.padding)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            self.lowres_calibration_field_head[-1] = head

    def _apply_hyper_residual_adapter(
        self,
        feature_map: torch.Tensor,
        cond: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_hyper_residual_head:
            return residual
        params = self.hyper_residual_mlp(cond)
        channels = int(feature_map.shape[1])
        if self.hyper_residual_mode == "full_kernel_delta":
            params = params.reshape(-1, self.output_channels, channels * 9 + 1)
            weights = params[:, :, :-1].reshape(-1, self.output_channels, channels, 3, 3)
            bias = params[:, :, -1]
            dynamic = torch.cat(
                [
                    F.conv2d(
                        feature_map[idx : idx + 1],
                        weights[idx] / math.sqrt(float(channels * 9)),
                        bias=bias[idx],
                        padding=1,
                    )
                    for idx in range(feature_map.shape[0])
                ],
                dim=0,
            )
        else:
            params = params.reshape(-1, self.output_channels, channels + 1)
            weights = params[:, :, :-1]
            bias = params[:, :, -1]
            dynamic = torch.einsum("bchw,boc->bohw", feature_map, weights)
            dynamic = dynamic / math.sqrt(float(channels)) + bias[:, :, None, None]
        return residual + self.hyper_residual_scale * dynamic

    def forward(
        self,
        source_stack: torch.Tensor,
        source_field_id: torch.Tensor,
        target_field_id: torch.Tensor,
        contrast_id: torch.Tensor,
        base_center: torch.Tensor | None = None,
    ) -> ModelOutput:
        cond = self.encode_condition(source_field_id, target_field_id, contrast_id)
        self._last_calibration_extra_delta = None
        self._last_primitive_candidates = None
        self._last_primitive_blend = None
        self._last_lowres_field = None
        self._last_lowres_field_upsampled = None
        self._last_lowres_field_shape = None
        self._last_lowres_full_shape = None
        source_center = source_stack[:, self.in_channels // 2 : self.in_channels // 2 + 1]
        if base_center is None:
            base_center = source_stack
        if base_center.shape[1] != self.output_channels:
            raise ValueError(
                f"base_center must have {self.output_channels} channels, got {base_center.shape[1]}"
            )

        x = self.stem(source_stack)
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x, cond)
            skips.append(x)
            x = down(x)
        x = self.middle(x, cond)
        for up, decoder in zip(self.ups, self.decoders):
            x = up(x)
            skip = skips.pop()
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(x + skip, cond)

        residual = self._apply_hyper_residual_adapter(x, cond, self.residual_head(x))
        gate = torch.sigmoid(self.gate_head(x))
        tissue_logits = self.tissue_head(x) if self.tissue_head is not None else None
        alpha = self.alpha_max * torch.sigmoid(self.alpha_mlp(cond))[:, :, None, None]
        if self.output_mode == "direct_output":
            raw_pred = torch.sigmoid(residual)
        elif self.output_mode == "anchor_unbounded_residual":
            raw_pred = base_center + residual
        else:
            raw_pred = base_center + alpha * gate * torch.tanh(residual)

        stats_source = source_center if self.source_stats_mode == "legacy7" else source_stack
        stats = self._source_stats(stats_source, mode=self.source_stats_mode)
        (
            pred,
            knots,
            primitive_weights,
            primitive_base,
            policy_features,
            calibration_bias,
            calibration_anchor,
            calibration_delta,
        ) = self._apply_calibration(
            raw_pred,
            cond,
            stats,
            source_stack,
            base_center,
            x,
            source_field_id,
            target_field_id,
            contrast_id,
        )
        features = torch.cat([cond, stats], dim=1)
        method_bank_features = self._method_bank_features(features, source_field_id, target_field_id)
        method_bank_logits = self._predict_method_bank_logits(method_bank_features)
        executor_family_logits = self.executor_family_mlp(features) if self.executor_family_size > 0 else None
        executor_params = torch.sigmoid(self.executor_param_mlp(features)) if self.executor_param_dim > 0 else None
        return ModelOutput(
            pred=pred,
            raw_pred=raw_pred,
            residual=residual,
            gate=gate,
            alpha=alpha,
            calibrated=pred,
            calibration_knots=knots,
            calibration_bias=calibration_bias,
            calibration_anchor=calibration_anchor,
            calibration_delta=calibration_delta,
            calibration_extra_delta=getattr(self, "_last_calibration_extra_delta", None),
            primitive_weights=primitive_weights,
            primitive_base=primitive_base,
            primitive_candidates=getattr(self, "_last_primitive_candidates", None),
            primitive_blend=getattr(self, "_last_primitive_blend", None),
            policy_features=policy_features,
            method_bank_logits=method_bank_logits,
            executor_family_logits=executor_family_logits,
            executor_params=executor_params,
            guard_features=getattr(self, "_last_guard_features", None),
            tissue_logits=tissue_logits,
            tissue_delta=None,
            tissue_coefficients=None,
            lowres_field=getattr(self, "_last_lowres_field", None),
            lowres_field_upsampled=getattr(self, "_last_lowres_field_upsampled", None),
        )
