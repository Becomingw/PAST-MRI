from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from past_mri.calibration import DEFAULT_PERCENTILES
from past_mri.constants import CONTRASTS, FIELD_VALUES, FIELDS
from past_mri.method_features import METHOD_FEATURE_TO_INDEX


@dataclass
class ModelOutput:
    pred: torch.Tensor
    raw_pred: torch.Tensor
    residual: torch.Tensor
    gate: torch.Tensor
    alpha: torch.Tensor
    calibrated: torch.Tensor | None = None
    calibration_knots: torch.Tensor | None = None
    calibration_bias: torch.Tensor | None = None
    calibration_anchor: torch.Tensor | None = None
    calibration_delta: torch.Tensor | None = None
    calibration_extra_delta: torch.Tensor | None = None
    primitive_weights: torch.Tensor | None = None
    primitive_base: torch.Tensor | None = None
    primitive_candidates: torch.Tensor | None = None
    primitive_blend: torch.Tensor | None = None
    policy_features: torch.Tensor | None = None
    method_bank_logits: torch.Tensor | None = None
    executor_family_logits: torch.Tensor | None = None
    executor_params: torch.Tensor | None = None
    guard_features: torch.Tensor | None = None
    tissue_logits: torch.Tensor | None = None
    tissue_delta: torch.Tensor | None = None
    tissue_coefficients: torch.Tensor | None = None
    lowres_field: torch.Tensor | None = None
    lowres_field_upsampled: torch.Tensor | None = None


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class CondFiLM(nn.Module):
    def __init__(self, cond_dim: int, channels: int) -> None:
        super().__init__()
        self.to_gamma_beta = nn.Linear(cond_dim, channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return x * (1.0 + gamma) + beta


class CondLoRADelta(nn.Module):
    """Condition-generated low-rank delta for a convolutional projection.

    The base convolution remains a normal nn.Conv2d so old checkpoints can still
    partially initialize shared weights. This module only adds a zero-init
    condition-dependent residual branch.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        *,
        rank: int = 12,
        hidden: int = 256,
        scale: float = 0.05,
        kernel_size: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.rank = int(rank)
        self.scale = float(scale)
        self.kernel_size = int(kernel_size)
        self.padding = int(padding)
        if self.rank < 1:
            raise ValueError("hyper_trunk_rank must be >= 1")
        if int(hidden) < 1:
            raise ValueError("hyper_trunk_hidden must be >= 1")
        if self.scale < 0.0:
            raise ValueError("hyper_trunk_scale must be >= 0")
        if self.kernel_size < 1 or self.kernel_size % 2 != 1:
            raise ValueError("CondLoRADelta kernel_size must be positive and odd")

        fan_in = self.in_channels * self.kernel_size * self.kernel_size
        self.lora_a = nn.Parameter(torch.empty(self.rank, fan_in))
        self.lora_b = nn.Parameter(torch.empty(self.out_channels, self.rank))
        self.condition_mlp = nn.Sequential(
            nn.Linear(cond_dim, int(hidden)),
            nn.SiLU(),
            nn.Linear(int(hidden), self.rank),
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.normal_(self.lora_b, mean=0.0, std=0.02)
        nn.init.zeros_(self.condition_mlp[-1].weight)
        nn.init.zeros_(self.condition_mlp[-1].bias)

    def condition_vector(self, cond: torch.Tensor) -> torch.Tensor:
        return self.condition_mlp(cond)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        coeff = self.condition_vector(cond)
        rows = []
        norm = math.sqrt(float(self.rank * self.in_channels * self.kernel_size * self.kernel_size))
        for idx in range(x.shape[0]):
            basis = self.lora_b * coeff[idx][None, :]
            weight = (basis @ self.lora_a).view(
                self.out_channels,
                self.in_channels,
                self.kernel_size,
                self.kernel_size,
            )
            rows.append(
                F.conv2d(
                    x[idx : idx + 1],
                    self.scale * weight / norm,
                    padding=self.padding,
                )
            )
        return torch.cat(rows, dim=0)


class CondNAFBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        cond_dim: int,
        expansion: int = 2,
        *,
        use_hyper_trunk: bool = False,
        hyper_trunk_rank: int = 12,
        hyper_trunk_hidden: int = 256,
        hyper_trunk_scale: float = 0.05,
    ) -> None:
        super().__init__()
        hidden = channels * expansion
        self.norm1 = LayerNorm2d(channels)
        self.film1 = CondFiLM(cond_dim, channels)
        self.pw1 = nn.Conv2d(channels, hidden * 2, 1)
        self.dw = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2)
        self.pw2 = nn.Conv2d(hidden, channels, 1)
        self.norm2 = LayerNorm2d(channels)
        self.film2 = CondFiLM(cond_dim, channels)
        self.ffn1 = nn.Conv2d(channels, hidden * 2, 1)
        self.ffn2 = nn.Conv2d(hidden, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.use_hyper_trunk = bool(use_hyper_trunk)
        if self.use_hyper_trunk:
            kwargs = {
                "rank": int(hyper_trunk_rank),
                "hidden": int(hyper_trunk_hidden),
                "scale": float(hyper_trunk_scale),
            }
            self.pw1_lora = CondLoRADelta(channels, hidden * 2, cond_dim, **kwargs)
            self.pw2_lora = CondLoRADelta(hidden, channels, cond_dim, **kwargs)
            self.ffn1_lora = CondLoRADelta(channels, hidden * 2, cond_dim, **kwargs)
            self.ffn2_lora = CondLoRADelta(hidden, channels, cond_dim, **kwargs)

    @staticmethod
    def _simple_gate(x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return a * b

    def _project(self, name: str, conv: nn.Conv2d, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        y = conv(x)
        if self.use_hyper_trunk:
            y = y + getattr(self, f"{name}_lora")(x, cond)
        return y

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        y = self.film1(self.norm1(x), cond)
        y = self._project("pw1", self.pw1, y, cond)
        y = self.dw(y)
        y = self._simple_gate(y)
        y = self._project("pw2", self.pw2, y, cond)
        x = x + self.beta * y

        y = self.film2(self.norm2(x), cond)
        y = self._project("ffn1", self.ffn1, y, cond)
        y = self._simple_gate(y)
        y = self._project("ffn2", self.ffn2, y, cond)
        return x + self.gamma * y


class CondPlainResBlock(nn.Module):
    """Plain conditional residual block for the matched-backbone ablation."""

    def __init__(self, channels: int, cond_dim: int) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.film1 = CondFiLM(cond_dim, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.SiLU()
        self.norm2 = LayerNorm2d(channels)
        self.film2 = CondFiLM(cond_dim, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.residual_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        y = self.conv1(self.film1(self.norm1(x), cond))
        y = self.act(y)
        y = self.conv2(self.film2(self.norm2(y), cond))
        return x + self.residual_scale * y


class CondStage(nn.Module):
    def __init__(
        self,
        channels: int,
        cond_dim: int,
        blocks: int,
        *,
        use_hyper_trunk: bool = False,
        hyper_trunk_rank: int = 12,
        hyper_trunk_hidden: int = 256,
        hyper_trunk_scale: float = 0.05,
        block_family: str = "naf",
    ) -> None:
        super().__init__()
        family = str(block_family).strip().lower().replace("-", "_")
        if family in ("naf", "nafnet", "cond_naf"):
            family = "naf"
        elif family in ("plain", "plain_conv", "plain_residual"):
            family = "plain"
        else:
            raise ValueError(f"Unknown block_family: {block_family}; expected naf or plain")
        if family == "plain" and use_hyper_trunk:
            raise ValueError("Plain residual blocks do not support use_hyper_trunk")
        self.block_family = family
        modules = []
        for _ in range(blocks):
            if family == "naf":
                module = CondNAFBlock(
                    channels,
                    cond_dim,
                    use_hyper_trunk=use_hyper_trunk,
                    hyper_trunk_rank=hyper_trunk_rank,
                    hyper_trunk_hidden=hyper_trunk_hidden,
                    hyper_trunk_scale=hyper_trunk_scale,
                )
            else:
                module = CondPlainResBlock(channels, cond_dim)
            modules.append(module)
        self.blocks = nn.ModuleList(modules)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, cond)
        return x


class WindowAttentionResidualBlock(nn.Module):
    def __init__(self, channels: int, heads: int, window_size: int, expansion: int = 2) -> None:
        super().__init__()
        if heads < 1:
            raise ValueError("swin_residual_heads must be >= 1")
        if channels % heads != 0:
            raise ValueError("width must be divisible by swin_residual_heads")
        if window_size < 1:
            raise ValueError("swin_residual_window_size must be >= 1")
        self.window_size = int(window_size)
        self.norm1 = LayerNorm2d(channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm2 = LayerNorm2d(channels)
        hidden = channels * int(expansion)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1),
        )
        self.attn_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.ffn_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def _window_attention(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        window = self.window_size
        pad_h = (window - height % window) % window
        pad_w = (window - width % window) % window
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        padded_h, padded_w = x.shape[-2:]
        num_h = padded_h // window
        num_w = padded_w // window
        tokens = (
            x.view(batch, channels, num_h, window, num_w, window)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(batch * num_h * num_w, window * window, channels)
        )
        tokens = self.attn(tokens, tokens, tokens, need_weights=False)[0]
        y = (
            tokens.view(batch, num_h, num_w, window, window, channels)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(batch, channels, padded_h, padded_w)
        )
        return y[:, :, :height, :width]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._window_attention(self.norm1(x))
        x = x + self.attn_scale * y
        y = self.ffn(self.norm2(x))
        return x + self.ffn_scale * y


class SwinResidualAdapter(nn.Module):
    def __init__(self, channels: int, depth: int, heads: int, window_size: int) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("swin_residual_depth must be >= 1")
        self.blocks = nn.ModuleList(
            [
                WindowAttentionResidualBlock(
                    channels=channels,
                    heads=heads,
                    window_size=window_size,
                )
                for _ in range(int(depth))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class CondNAFUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        width: int = 48,
        enc_blocks: list[int] | tuple[int, ...] = (2, 2, 4),
        mid_blocks: int = 8,
        dec_blocks: list[int] | tuple[int, ...] = (2, 2, 2),
        cond_dim: int = 256,
        alpha_max: float = 1.0,
        output_mode: str = "bounded_residual",
        num_contrasts: int = 3,
        use_calibration_head: bool = True,
        calibration_head: str = "affine",
        calibration_knots: int = 8,
        calibration_knot_delta_max: float = 0.05,
        calibration_relation_map_init_logit: float = -4.0,
        calibration_relation_lowfreq_init_logit: float = -4.0,
        calibration_relation_source_floor_init_logit: float = -6.0,
        calibration_lowfreq_kernel_size: int = 9,
        calibration_blend_max: float = 1.0,
        calibration_texture_blend_max: float = 1.0,
        calibration_gain_scale_max: float = 0.25,
        calibration_bias_scale_max: float = 0.06,
        calibration_guard_mode: str = "none",
        calibration_guard_extreme_scale: float = 0.25,
        calibration_source_gate_mode: str = "none",
        calibration_source_gate_floor: float = 0.25,
        calibration_source_gate_strength: float = 1.0,
        calibration_guard_v3_downfield_boost: float = 0.0,
        calibration_guard_v3_blur_boost: float = 0.0,
        primitive_bank_blend_max: float = 1.0,
        primitive_bank_mode: str = "basic",
        primitive_bank_init: str = "uniform",
        primitive_bank_init_strength: float = 4.0,
        policy_feature_dim: int = 0,
        policy_feature_bias_init: list[float] | tuple[float, ...] | None = None,
        policy_feature_bias_zero_weight: bool = False,
        method_bank_size: int = 0,
        method_bank_feature_mode: str = "default",
        executor_family_size: int = 0,
        executor_param_dim: int = 0,
        primitive_bank_use_policy_features: bool = False,
        target_stats_path: str | None = None,
        source_stats_mode: str = "legacy7",
        direct_control_init_values: list[float] | tuple[float, ...] | None = None,
        direct_control_init_weight_std: float = 0.0,
        use_tissue_head: bool = False,
        tissue_classes: int = 7,
        use_tissue_bridge: bool = False,
        tissue_bridge_max: float = 0.02,
        tissue_bridge_kernel_size: int = 9,
        use_tissue_local_bridge: bool = False,
        tissue_local_bridge_max: float = 0.02,
        tissue_local_bridge_hidden: int = 16,
        tissue_local_bridge_kernel_size: int = 9,
        use_hyper_residual_head: bool = False,
        hyper_residual_hidden: int = 128,
        hyper_residual_scale: float = 0.05,
        hyper_residual_mode: str = "adapter",
        use_hyper_trunk: bool = False,
        hyper_trunk_blocks: list[str] | tuple[str, ...] | str = ("mid",),
        hyper_trunk_rank: int = 12,
        hyper_trunk_hidden: int = 256,
        hyper_trunk_scale: float = 0.05,
        calibration_lowrank_components: int = 4,
        calibration_lowrank_delta_max: float = 0.02,
        calibration_lowrank_coeff_init: float = 0.1,
        lowres_field_downsample: int = 8,
        lowres_field_mode: str = "additive",
        swin_residual_depth: int = 2,
        swin_residual_heads: int = 2,
        swin_residual_window_size: int = 8,
        block_family: str = "naf",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.cond_dim = cond_dim
        self.alpha_max = alpha_max
        self.block_family = str(block_family).strip().lower().replace("-", "_")
        if self.block_family in ("nafnet", "cond_naf"):
            self.block_family = "naf"
        elif self.block_family in ("plain_conv", "plain_residual"):
            self.block_family = "plain"
        if self.block_family not in ("naf", "plain"):
            raise ValueError(f"Unknown block_family: {block_family}; expected naf or plain")
        self.output_mode = output_mode.strip().lower().replace("-", "_")
        if self.output_mode in ("relaxed_residual", "bounded", "bounded_residual"):
            self.output_mode = "bounded_residual"
        elif self.output_mode in ("direct", "direct_output", "direct_sigmoid"):
            self.output_mode = "direct_output"
        elif self.output_mode in ("unbounded_residual", "anchor_unbounded_residual", "linear_residual"):
            self.output_mode = "anchor_unbounded_residual"
        else:
            raise ValueError(
                "Unknown output_mode: "
                f"{output_mode}; expected bounded_residual, direct_output, or anchor_unbounded_residual"
            )
        self.use_calibration_head = use_calibration_head
        self.calibration_head = "none" if not use_calibration_head else calibration_head.strip().lower()
        if self.calibration_head == "lut":
            self.calibration_head = "monotonic_lut"
        if self.calibration_head == "source_lut":
            self.calibration_head = "source_monotonic_lut"
        if self.calibration_head == "affine_plus_lowfreq_bias":
            self.calibration_head = "affine_lowfreq_bias"
        if self.calibration_head not in (
            "none",
            "identity",
            "affine",
            "affine_gamma",
            "monotonic_lut",
            "source_monotonic_lut",
            "lowfreq_bias",
            "affine_lowfreq_bias",
            "source_stat_robust",
            "source_stat_percentile",
            "source_lut_lowfreq_texture",
            "primitive_bank",
            "policy_param_lut_lowfreq",
            "direct_param_source_safe_v1",
            "direct_param_target_percentile_v1",
            "direct_param_target_percentile_guarded_v2",
            "direct_param_target_percentile_guarded_v3",
            "direct_control_target_percentile_v1",
            "direct_control_gain_bias_field_v1",
            "target_percentile_relation_v1",
            "source_anchor_percentile_relation_v1",
            "base_anchor_percentile_relation_v1",
            "source_preserving_monotonic_lowfreq_v1",
            "calibrated_anchor_detail_v1",
            "target_percentile_primitive_bank_v1",
            "local_lowfreq_field_v1",
            "target_stat_lowfreq_field_v1",
            "target_stat_anchor_detail_v1",
            "target_stat_anchor_lowrank_residual_v1",
            "target_stat_anchor_fine_residual_v1",
            "target_stat_anchor_swin_residual_v1",
            "target_stat_gain_bias_field_v1",
            "target_stat_gain_bias_lowres_delta_v1",
            "lowres_calibration_field_v1",
        ):
            raise ValueError(f"Unknown calibration_head: {calibration_head}")
        if self.calibration_head in (
            "source_stat_robust",
            "source_stat_percentile",
            "direct_param_target_percentile_v1",
            "direct_param_target_percentile_guarded_v2",
            "direct_param_target_percentile_guarded_v3",
            "direct_control_target_percentile_v1",
            "direct_control_gain_bias_field_v1",
            "target_percentile_relation_v1",
            "source_anchor_percentile_relation_v1",
            "base_anchor_percentile_relation_v1",
            "source_preserving_monotonic_lowfreq_v1",
            "calibrated_anchor_detail_v1",
            "target_percentile_primitive_bank_v1",
            "target_stat_lowfreq_field_v1",
            "target_stat_anchor_detail_v1",
            "target_stat_anchor_lowrank_residual_v1",
            "target_stat_anchor_fine_residual_v1",
            "target_stat_anchor_swin_residual_v1",
            "target_stat_gain_bias_field_v1",
            "target_stat_gain_bias_lowres_delta_v1",
            "lowres_calibration_field_v1",
        ) and not target_stats_path:
            raise ValueError(f"{self.calibration_head} calibration_head requires target_stats_path")
        self.calibration_knots = int(calibration_knots)
        self.calibration_knot_delta_max = float(calibration_knot_delta_max)
        if self.calibration_knot_delta_max < 0.0:
            raise ValueError("calibration_knot_delta_max must be >= 0")
        self.calibration_relation_map_init_logit = float(calibration_relation_map_init_logit)
        self.calibration_relation_lowfreq_init_logit = float(calibration_relation_lowfreq_init_logit)
        self.calibration_relation_source_floor_init_logit = float(calibration_relation_source_floor_init_logit)
        self.direct_control_init_values = (
            tuple(float(value) for value in direct_control_init_values)
            if direct_control_init_values is not None
            else None
        )
        if self.direct_control_init_values is not None:
            if len(self.direct_control_init_values) != 4:
                raise ValueError("direct_control_init_values must contain 4 values")
            if any(value <= 0.0 or value >= 1.0 for value in self.direct_control_init_values):
                raise ValueError("direct_control_init_values must be strictly between 0 and 1")
        self.direct_control_init_weight_std = float(direct_control_init_weight_std)
        if self.direct_control_init_weight_std < 0.0:
            raise ValueError("direct_control_init_weight_std must be >= 0")
        self.use_tissue_head = bool(use_tissue_head)
        self.tissue_classes = int(tissue_classes)
        if self.tissue_classes < 2:
            raise ValueError("tissue_classes must be >= 2")
        self.use_tissue_bridge = bool(use_tissue_bridge)
        if self.use_tissue_bridge and not self.use_tissue_head:
            raise ValueError("use_tissue_bridge requires use_tissue_head")
        self.tissue_bridge_max = float(tissue_bridge_max)
        if self.tissue_bridge_max < 0.0:
            raise ValueError("tissue_bridge_max must be >= 0")
        self.tissue_bridge_kernel_size = int(tissue_bridge_kernel_size)
        if self.tissue_bridge_kernel_size < 1:
            raise ValueError("tissue_bridge_kernel_size must be >= 1")
        if self.tissue_bridge_kernel_size % 2 == 0:
            raise ValueError("tissue_bridge_kernel_size must be odd")
        self.use_tissue_local_bridge = bool(use_tissue_local_bridge)
        if self.use_tissue_local_bridge and not self.use_tissue_head:
            raise ValueError("use_tissue_local_bridge requires use_tissue_head")
        self.tissue_local_bridge_max = float(tissue_local_bridge_max)
        if self.tissue_local_bridge_max < 0.0:
            raise ValueError("tissue_local_bridge_max must be >= 0")
        self.tissue_local_bridge_hidden = int(tissue_local_bridge_hidden)
        if self.tissue_local_bridge_hidden < 1:
            raise ValueError("tissue_local_bridge_hidden must be >= 1")
        self.tissue_local_bridge_kernel_size = int(tissue_local_bridge_kernel_size)
        if self.tissue_local_bridge_kernel_size < 1:
            raise ValueError("tissue_local_bridge_kernel_size must be >= 1")
        if self.tissue_local_bridge_kernel_size % 2 == 0:
            raise ValueError("tissue_local_bridge_kernel_size must be odd")
        self.use_hyper_residual_head = bool(use_hyper_residual_head)
        self.hyper_residual_hidden = int(hyper_residual_hidden)
        if self.hyper_residual_hidden < 1:
            raise ValueError("hyper_residual_hidden must be >= 1")
        self.hyper_residual_scale = float(hyper_residual_scale)
        if self.hyper_residual_scale < 0.0:
            raise ValueError("hyper_residual_scale must be >= 0")
        self.hyper_residual_mode = str(hyper_residual_mode).strip().lower()
        if self.hyper_residual_mode in ("", "legacy"):
            self.hyper_residual_mode = "adapter"
        if self.hyper_residual_mode not in ("adapter", "full_kernel_delta"):
            raise ValueError("hyper_residual_mode must be adapter or full_kernel_delta")
        self.use_hyper_trunk = bool(use_hyper_trunk)
        if self.block_family == "plain" and self.use_hyper_trunk:
            raise ValueError("Plain residual blocks require use_hyper_trunk=false")
        if isinstance(hyper_trunk_blocks, str):
            trunk_tokens = [
                token.strip().lower()
                for token in hyper_trunk_blocks.replace(",", " ").split()
                if token.strip()
            ]
        else:
            trunk_tokens = [str(token).strip().lower() for token in hyper_trunk_blocks if str(token).strip()]
        trunk_aliases = {
            "middle": "mid",
            "decoder_inner": "dec_inner",
            "decoder_outer": "dec_outer",
            "decoder_all": "dec_all",
            "encoder_all": "enc_all",
            "none": "",
        }
        trunk_blocks = tuple(trunk_aliases.get(token, token) for token in trunk_tokens)
        if self.use_hyper_trunk and not trunk_blocks:
            trunk_blocks = ("mid",)
        allowed_trunk_blocks = {"mid", "dec_inner", "dec_outer", "dec_all", "enc_all", ""}
        allowed_trunk_blocks.update(f"dec{idx}" for idx in range(len(dec_blocks)))
        allowed_trunk_blocks.update(f"enc{idx}" for idx in range(len(enc_blocks)))
        unknown_trunk_blocks = sorted(set(trunk_blocks) - allowed_trunk_blocks)
        if unknown_trunk_blocks:
            raise ValueError(f"Unknown hyper_trunk_blocks entries: {unknown_trunk_blocks}")
        self.hyper_trunk_blocks = tuple(token for token in trunk_blocks if token)
        self.hyper_trunk_rank = int(hyper_trunk_rank)
        if self.hyper_trunk_rank < 1:
            raise ValueError("hyper_trunk_rank must be >= 1")
        self.hyper_trunk_hidden = int(hyper_trunk_hidden)
        if self.hyper_trunk_hidden < 1:
            raise ValueError("hyper_trunk_hidden must be >= 1")
        self.hyper_trunk_scale = float(hyper_trunk_scale)
        if self.hyper_trunk_scale < 0.0:
            raise ValueError("hyper_trunk_scale must be >= 0")
        if (
            self.calibration_head
            in (
                "monotonic_lut",
                "source_monotonic_lut",
                "source_lut_lowfreq_texture",
                "policy_param_lut_lowfreq",
                "direct_param_source_safe_v1",
                "direct_param_target_percentile_v1",
                "direct_param_target_percentile_guarded_v2",
                "direct_param_target_percentile_guarded_v3",
            )
            and self.calibration_knots < 3
        ):
            raise ValueError("calibration_knots must be >= 3 for monotonic_lut")
        self.calibration_lowfreq_kernel_size = int(calibration_lowfreq_kernel_size)
        if self.calibration_head in (
            "lowfreq_bias",
            "affine_lowfreq_bias",
            "source_lut_lowfreq_texture",
            "direct_param_source_safe_v1",
            "direct_param_target_percentile_v1",
            "direct_param_target_percentile_guarded_v2",
            "direct_param_target_percentile_guarded_v3",
            "calibrated_anchor_detail_v1",
            "target_percentile_primitive_bank_v1",
            "local_lowfreq_field_v1",
            "target_stat_lowfreq_field_v1",
            "target_stat_anchor_detail_v1",
            "target_stat_anchor_lowrank_residual_v1",
            "target_stat_anchor_fine_residual_v1",
            "target_stat_anchor_swin_residual_v1",
            "target_stat_gain_bias_field_v1",
            "target_stat_gain_bias_lowres_delta_v1",
            "lowres_calibration_field_v1",
            "direct_control_gain_bias_field_v1",
        ):
            if self.calibration_lowfreq_kernel_size < 1:
                raise ValueError("calibration_lowfreq_kernel_size must be >= 1")
            if self.calibration_lowfreq_kernel_size % 2 == 0:
                raise ValueError("calibration_lowfreq_kernel_size must be odd")
        self.calibration_blend_max = float(calibration_blend_max)
        self.calibration_texture_blend_max = float(calibration_texture_blend_max)
        self.calibration_lowrank_components = int(calibration_lowrank_components)
        if self.calibration_lowrank_components < 1:
            raise ValueError("calibration_lowrank_components must be >= 1")
        self.calibration_lowrank_delta_max = float(calibration_lowrank_delta_max)
        if self.calibration_lowrank_delta_max < 0.0:
            raise ValueError("calibration_lowrank_delta_max must be >= 0")
        self.calibration_lowrank_coeff_init = float(calibration_lowrank_coeff_init)
        self.lowres_field_downsample = int(lowres_field_downsample)
        if self.lowres_field_downsample < 1:
            raise ValueError("lowres_field_downsample must be >= 1")
        self.lowres_field_mode = str(lowres_field_mode).strip().lower()
        if self.lowres_field_mode not in ("additive", "gain_bias"):
            raise ValueError("lowres_field_mode must be additive or gain_bias")
        self.swin_residual_depth = int(swin_residual_depth)
        if self.swin_residual_depth < 1:
            raise ValueError("swin_residual_depth must be >= 1")
        self.swin_residual_heads = int(swin_residual_heads)
        if self.swin_residual_heads < 1:
            raise ValueError("swin_residual_heads must be >= 1")
        if width % self.swin_residual_heads != 0:
            raise ValueError("width must be divisible by swin_residual_heads")
        self.swin_residual_window_size = int(swin_residual_window_size)
        if self.swin_residual_window_size < 1:
            raise ValueError("swin_residual_window_size must be >= 1")
        self.calibration_gain_scale_max = float(calibration_gain_scale_max)
        if self.calibration_gain_scale_max < 0.0:
            raise ValueError("calibration_gain_scale_max must be >= 0")
        self.calibration_bias_scale_max = float(calibration_bias_scale_max)
        if self.calibration_bias_scale_max < 0.0:
            raise ValueError("calibration_bias_scale_max must be >= 0")
        self.calibration_guard_mode = calibration_guard_mode.strip().lower()
        if self.calibration_guard_mode not in ("none", "field_extreme_v1"):
            raise ValueError(f"Unknown calibration_guard_mode: {calibration_guard_mode}")
        self.calibration_guard_extreme_scale = float(calibration_guard_extreme_scale)
        if not 0.0 <= self.calibration_guard_extreme_scale <= 1.0:
            raise ValueError("calibration_guard_extreme_scale must be in [0, 1]")
        self.calibration_source_gate_mode = calibration_source_gate_mode.strip().lower()
        if self.calibration_source_gate_mode not in ("none", "source_edge_v1"):
            raise ValueError(f"Unknown calibration_source_gate_mode: {calibration_source_gate_mode}")
        self.calibration_source_gate_floor = float(calibration_source_gate_floor)
        if not 0.0 <= self.calibration_source_gate_floor <= 1.0:
            raise ValueError("calibration_source_gate_floor must be in [0, 1]")
        self.calibration_source_gate_strength = float(calibration_source_gate_strength)
        if self.calibration_source_gate_strength < 0.0:
            raise ValueError("calibration_source_gate_strength must be >= 0")
        self.calibration_guard_v3_downfield_boost = float(calibration_guard_v3_downfield_boost)
        if self.calibration_guard_v3_downfield_boost < 0.0:
            raise ValueError("calibration_guard_v3_downfield_boost must be >= 0")
        self.calibration_guard_v3_blur_boost = float(calibration_guard_v3_blur_boost)
        if self.calibration_guard_v3_blur_boost < 0.0:
            raise ValueError("calibration_guard_v3_blur_boost must be >= 0")
        self.source_stats_mode = source_stats_mode.strip().lower()
        if self.source_stats_mode not in ("legacy7", "masked_basic13"):
            raise ValueError(f"Unknown source_stats_mode: {source_stats_mode}")
        self.source_stats_dim = 7 if self.source_stats_mode == "legacy7" else 13
        self.primitive_bank_blend_max = float(primitive_bank_blend_max)
        self.primitive_bank_mode = primitive_bank_mode.strip().lower()
        if self.primitive_bank_mode not in (
            "basic",
            "source_safe_v1",
            "affine_source_safe_v1",
            "policy_affine_source_safe_v1",
            "policy_lowfreq_source_safe_v1",
            "policy_lut_lowfreq_source_safe_v1",
            "policy_lut_lowfreq_median_source_safe_v1",
        ):
            raise ValueError(f"Unknown primitive_bank_mode: {primitive_bank_mode}")
        self.primitive_bank_init = primitive_bank_init.strip().lower()
        if self.primitive_bank_init not in ("uniform", "raw", "affine"):
            raise ValueError(f"Unknown primitive_bank_init: {primitive_bank_init}")
        self.primitive_bank_init_strength = float(primitive_bank_init_strength)
        if self.primitive_bank_init_strength < 0.0:
            raise ValueError("primitive_bank_init_strength must be >= 0")
        if (
            self.primitive_bank_init == "affine"
            and self.calibration_head != "target_percentile_primitive_bank_v1"
            and self.primitive_bank_mode not in (
            "affine_source_safe_v1",
            "policy_affine_source_safe_v1",
            "policy_lowfreq_source_safe_v1",
            "policy_lut_lowfreq_source_safe_v1",
            "policy_lut_lowfreq_median_source_safe_v1",
            )
        ):
            raise ValueError(
                "primitive_bank_init='affine' requires an affine source-safe primitive bank mode"
            )
        self.policy_feature_dim = int(policy_feature_dim)
        self.method_bank_size = int(method_bank_size)
        if self.method_bank_size < 0:
            raise ValueError("method_bank_size must be >= 0")
        self.executor_family_size = int(executor_family_size)
        if self.executor_family_size < 0:
            raise ValueError("executor_family_size must be >= 0")
        self.executor_param_dim = int(executor_param_dim)
        if self.executor_param_dim < 0:
            raise ValueError("executor_param_dim must be >= 0")
        self.method_bank_feature_mode = method_bank_feature_mode.strip().lower()
        if self.method_bank_feature_mode not in ("default", "highfield_v1"):
            raise ValueError(f"Unknown method_bank_feature_mode: {method_bank_feature_mode}")
        self.policy_feature_bias_zero_weight = bool(policy_feature_bias_zero_weight)
        self.primitive_bank_use_policy_features = bool(primitive_bank_use_policy_features)
        if policy_feature_bias_init is not None:
            if self.policy_feature_dim <= 0:
                raise ValueError("policy_feature_bias_init requires policy_feature_dim > 0")
            if len(policy_feature_bias_init) != self.policy_feature_dim:
                raise ValueError(
                    f"policy_feature_bias_init length must match policy_feature_dim "
                    f"({len(policy_feature_bias_init)} != {self.policy_feature_dim})"
                )
        if self.primitive_bank_mode in (
            "policy_affine_source_safe_v1",
            "policy_lowfreq_source_safe_v1",
            "policy_lut_lowfreq_source_safe_v1",
            "policy_lut_lowfreq_median_source_safe_v1",
        ) and self.policy_feature_dim <= 0:
            raise ValueError(f"{self.primitive_bank_mode} requires policy_feature_dim > 0")
        if self.calibration_head == "policy_param_lut_lowfreq" and self.policy_feature_dim <= 0:
            raise ValueError("policy_param_lut_lowfreq requires policy_feature_dim > 0")
        if self.calibration_head == "direct_param_source_safe_v1" and self.policy_feature_dim <= 0:
            raise ValueError("direct_param_source_safe_v1 requires policy_feature_dim > 0")
        if self.calibration_head in (
            "direct_param_target_percentile_v1",
            "direct_param_target_percentile_guarded_v2",
            "direct_param_target_percentile_guarded_v3",
        ) and self.policy_feature_dim <= 0:
            raise ValueError(f"{self.calibration_head} requires policy_feature_dim > 0")
        if self.calibration_head == "source_stat_robust":
            self.register_buffer(
                "target_robust_stats",
                self._load_target_robust_stats(target_stats_path, num_contrasts),
            )
        if self.calibration_head in (
            "source_stat_percentile",
            "direct_param_target_percentile_v1",
            "direct_param_target_percentile_guarded_v2",
            "direct_param_target_percentile_guarded_v3",
            "direct_control_target_percentile_v1",
            "direct_control_gain_bias_field_v1",
            "target_percentile_relation_v1",
            "source_anchor_percentile_relation_v1",
            "base_anchor_percentile_relation_v1",
            "source_preserving_monotonic_lowfreq_v1",
            "calibrated_anchor_detail_v1",
            "target_percentile_primitive_bank_v1",
            "target_stat_lowfreq_field_v1",
            "target_stat_anchor_detail_v1",
            "target_stat_anchor_lowrank_residual_v1",
            "target_stat_anchor_fine_residual_v1",
            "target_stat_anchor_swin_residual_v1",
            "target_stat_gain_bias_field_v1",
            "target_stat_gain_bias_lowres_delta_v1",
            "lowres_calibration_field_v1",
        ):
            self.register_buffer(
                "source_stat_percentile_levels",
                torch.tensor([level / 100.0 for level in DEFAULT_PERCENTILES], dtype=torch.float32),
            )
            self.register_buffer(
                "target_percentile_stats",
                self._load_target_percentile_stats(target_stats_path, num_contrasts),
            )
        self.field_embed = nn.Embedding(len(FIELDS), 32)
        self.contrast_embed = nn.Embedding(num_contrasts, 16)
        cond_input_dim = 32 + 32 + 16 + 6
        self.condition_mlp = nn.Sequential(
            nn.Linear(cond_input_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.stem = nn.Conv2d(in_channels, width, 3, padding=1)
        channels = width
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        for encoder_idx, blocks in enumerate(enc_blocks):
            use_encoder_hyper = self.use_hyper_trunk and (
                "enc_all" in self.hyper_trunk_blocks or f"enc{encoder_idx}" in self.hyper_trunk_blocks
            )
            self.encoders.append(
                CondStage(
                    channels,
                    cond_dim,
                    int(blocks),
                    use_hyper_trunk=use_encoder_hyper,
                    hyper_trunk_rank=self.hyper_trunk_rank,
                    hyper_trunk_hidden=self.hyper_trunk_hidden,
                    hyper_trunk_scale=self.hyper_trunk_scale,
                    block_family=self.block_family,
                )
            )
            self.downs.append(nn.Conv2d(channels, channels * 2, 2, stride=2))
            channels *= 2

        self.middle = CondStage(
            channels,
            cond_dim,
            int(mid_blocks),
            use_hyper_trunk=self.use_hyper_trunk and "mid" in self.hyper_trunk_blocks,
            hyper_trunk_rank=self.hyper_trunk_rank,
            hyper_trunk_hidden=self.hyper_trunk_hidden,
            hyper_trunk_scale=self.hyper_trunk_scale,
            block_family=self.block_family,
        )

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for decoder_idx, blocks in enumerate(dec_blocks):
            self.ups.append(nn.ConvTranspose2d(channels, channels // 2, 2, stride=2))
            channels //= 2
            use_decoder_hyper = self.use_hyper_trunk and (
                "dec_all" in self.hyper_trunk_blocks
                or ("dec_inner" in self.hyper_trunk_blocks and decoder_idx == 0)
                or ("dec_outer" in self.hyper_trunk_blocks and decoder_idx == len(dec_blocks) - 1)
                or f"dec{decoder_idx}" in self.hyper_trunk_blocks
            )
            self.decoders.append(
                CondStage(
                    channels,
                    cond_dim,
                    int(blocks),
                    use_hyper_trunk=use_decoder_hyper,
                    hyper_trunk_rank=self.hyper_trunk_rank,
                    hyper_trunk_hidden=self.hyper_trunk_hidden,
                    hyper_trunk_scale=self.hyper_trunk_scale,
                    block_family=self.block_family,
                )
            )

        self.residual_head = nn.Conv2d(width, 1, 3, padding=1)
        if self.use_hyper_residual_head:
            hyper_out_dim = width + 1 if self.hyper_residual_mode == "adapter" else width * 9 + 1
            self.hyper_residual_mlp = nn.Sequential(
                nn.Linear(cond_dim, self.hyper_residual_hidden),
                nn.SiLU(),
                nn.Linear(self.hyper_residual_hidden, hyper_out_dim),
            )
            nn.init.zeros_(self.hyper_residual_mlp[-1].weight)
            nn.init.zeros_(self.hyper_residual_mlp[-1].bias)
        self.gate_head = nn.Conv2d(width, 1, 3, padding=1)
        self.tissue_head = nn.Conv2d(width, self.tissue_classes, 3, padding=1) if self.use_tissue_head else None
        self.alpha_mlp = nn.Sequential(nn.Linear(cond_dim, cond_dim // 2), nn.SiLU(), nn.Linear(cond_dim // 2, 1))
        features_dim = cond_dim + self.source_stats_dim
        if self.use_tissue_bridge:
            self.tissue_bridge_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, self.tissue_classes),
            )
            nn.init.zeros_(self.tissue_bridge_mlp[-1].weight)
            nn.init.zeros_(self.tissue_bridge_mlp[-1].bias)
        if self.use_tissue_local_bridge:
            self.tissue_local_bridge_net = nn.Sequential(
                nn.Conv2d(self.tissue_classes + 3, self.tissue_local_bridge_hidden, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(self.tissue_local_bridge_hidden, 1, 3, padding=1),
            )
            nn.init.zeros_(self.tissue_local_bridge_net[-1].weight)
            nn.init.zeros_(self.tissue_local_bridge_net[-1].bias)
        if self.policy_feature_dim > 0:
            self.policy_feature_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, self.policy_feature_dim),
            )
            if policy_feature_bias_init is not None:
                final = self.policy_feature_mlp[-1]
                with torch.no_grad():
                    final.bias.copy_(torch.as_tensor(policy_feature_bias_init, dtype=final.bias.dtype))
                    if self.policy_feature_bias_zero_weight:
                        final.weight.zero_()
        self.method_bank_extra_feature_dim = 0 if self.method_bank_feature_mode == "default" else 8
        if self.method_bank_size > 0:
            self.method_bank_mlp = nn.Sequential(
                nn.Linear(features_dim + self.method_bank_extra_feature_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, self.method_bank_size),
            )
        if self.executor_family_size > 0:
            self.executor_family_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, self.executor_family_size),
            )
        if self.executor_param_dim > 0:
            self.executor_param_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, self.executor_param_dim),
            )
        if self.calibration_head in ("affine", "affine_gamma"):
            self.calibration_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, 2),
            )
            if self.calibration_head == "affine_gamma":
                self.calibration_gamma_mlp = nn.Sequential(
                    nn.Linear(features_dim, cond_dim // 2),
                    nn.SiLU(),
                    nn.Linear(cond_dim // 2, 1),
                )
                nn.init.zeros_(self.calibration_gamma_mlp[-1].weight)
                nn.init.zeros_(self.calibration_gamma_mlp[-1].bias)
        elif self.calibration_head == "affine_lowfreq_bias":
            self.calibration_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, 2),
            )
            self.calibration_bias_head = nn.Conv2d(width, 1, 3, padding=1)
            nn.init.zeros_(self.calibration_bias_head.weight)
            nn.init.zeros_(self.calibration_bias_head.bias)
            self.calibration_blend_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, 1),
            )
        elif self.calibration_head in ("monotonic_lut", "source_monotonic_lut"):
            self.calibration_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, self.calibration_knots - 1),
            )
            self.calibration_blend_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, 1),
            )
        elif self.calibration_head == "lowfreq_bias":
            self.calibration_bias_head = nn.Conv2d(width, 1, 3, padding=1)
            nn.init.zeros_(self.calibration_bias_head.weight)
            nn.init.zeros_(self.calibration_bias_head.bias)
            self.calibration_blend_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, 1),
            )
        elif self.calibration_head == "local_lowfreq_field_v1":
            self.local_lowfreq_field_head = nn.Conv2d(width, 1, 3, padding=1)
            nn.init.zeros_(self.local_lowfreq_field_head.weight)
            nn.init.zeros_(self.local_lowfreq_field_head.bias)
        elif self.calibration_head == "target_stat_lowfreq_field_v1":
            self.target_stat_lowfreq_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, 4),
            )
            nn.init.zeros_(self.target_stat_lowfreq_mlp[-1].weight)
            nn.init.zeros_(self.target_stat_lowfreq_mlp[-1].bias)
            with torch.no_grad():
                final_bias = self.target_stat_lowfreq_mlp[-1].bias
                final_bias[0] = self.calibration_relation_map_init_logit
                final_bias[1] = self.calibration_relation_lowfreq_init_logit
                final_bias[2] = -4.0
                final_bias[3] = self.calibration_relation_source_floor_init_logit
            self.local_lowfreq_field_head = nn.Conv2d(width, 1, 3, padding=1)
            nn.init.zeros_(self.local_lowfreq_field_head.weight)
            nn.init.zeros_(self.local_lowfreq_field_head.bias)
        elif self.calibration_head == "target_stat_anchor_detail_v1":
            self.calibration_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, 2),
            )
            self.target_stat_anchor_detail_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, 4),
            )
            nn.init.zeros_(self.target_stat_anchor_detail_mlp[-1].weight)
            nn.init.zeros_(self.target_stat_anchor_detail_mlp[-1].bias)
            with torch.no_grad():
                final_bias = self.target_stat_anchor_detail_mlp[-1].bias
                final_bias[0] = self.calibration_relation_map_init_logit
                final_bias[1] = 2.0
                final_bias[2] = -4.0
                final_bias[3] = self.calibration_relation_source_floor_init_logit
            self.local_lowfreq_field_head = nn.Conv2d(width, 1, 3, padding=1)
            nn.init.zeros_(self.local_lowfreq_field_head.weight)
            nn.init.zeros_(self.local_lowfreq_field_head.bias)
        elif self.calibration_head == "target_stat_anchor_lowrank_residual_v1":
            self.calibration_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, 2),
            )
            self.target_stat_anchor_detail_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, 4),
            )
            nn.init.zeros_(self.target_stat_anchor_detail_mlp[-1].weight)
            nn.init.zeros_(self.target_stat_anchor_detail_mlp[-1].bias)
            with torch.no_grad():
                final_bias = self.target_stat_anchor_detail_mlp[-1].bias
                final_bias[0] = self.calibration_relation_map_init_logit
                final_bias[1] = 2.0
                final_bias[2] = -4.0
                final_bias[3] = self.calibration_relation_source_floor_init_logit
            self.local_lowfreq_field_head = nn.Conv2d(width, 1, 3, padding=1)
            nn.init.zeros_(self.local_lowfreq_field_head.weight)
            nn.init.zeros_(self.local_lowfreq_field_head.bias)
            self.lowrank_residual_basis_head = nn.Conv2d(width, self.calibration_lowrank_components, 3, padding=1)
            self.lowrank_residual_coeff_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, self.calibration_lowrank_components),
            )
            nn.init.zeros_(self.lowrank_residual_basis_head.weight)
            nn.init.zeros_(self.lowrank_residual_basis_head.bias)
            nn.init.zeros_(self.lowrank_residual_coeff_mlp[-1].weight)
            nn.init.constant_(self.lowrank_residual_coeff_mlp[-1].bias, self.calibration_lowrank_coeff_init)
        elif self.calibration_head == "target_stat_anchor_fine_residual_v1":
            self.calibration_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, 2),
            )
            self.target_stat_anchor_detail_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, 4),
            )
            nn.init.zeros_(self.target_stat_anchor_detail_mlp[-1].weight)
            nn.init.zeros_(self.target_stat_anchor_detail_mlp[-1].bias)
            with torch.no_grad():
                final_bias = self.target_stat_anchor_detail_mlp[-1].bias
                final_bias[0] = self.calibration_relation_map_init_logit
                final_bias[1] = 2.0
                final_bias[2] = -4.0
                final_bias[3] = self.calibration_relation_source_floor_init_logit
            self.local_lowfreq_field_head = nn.Conv2d(width, 1, 3, padding=1)
            nn.init.zeros_(self.local_lowfreq_field_head.weight)
            nn.init.zeros_(self.local_lowfreq_field_head.bias)
            self.fine_residual_field_head = nn.Conv2d(width, 1, 3, padding=1)
            self.fine_residual_coeff_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, 1),
            )
            nn.init.zeros_(self.fine_residual_field_head.weight)
            nn.init.zeros_(self.fine_residual_field_head.bias)
            nn.init.zeros_(self.fine_residual_coeff_mlp[-1].weight)
            nn.init.constant_(self.fine_residual_coeff_mlp[-1].bias, self.calibration_lowrank_coeff_init)
        elif self.calibration_head == "target_stat_anchor_swin_residual_v1":
            self.calibration_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, 2),
            )
            self.target_stat_anchor_detail_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, 4),
            )
            nn.init.zeros_(self.target_stat_anchor_detail_mlp[-1].weight)
            nn.init.zeros_(self.target_stat_anchor_detail_mlp[-1].bias)
            with torch.no_grad():
                final_bias = self.target_stat_anchor_detail_mlp[-1].bias
                final_bias[0] = self.calibration_relation_map_init_logit
                final_bias[1] = 2.0
                final_bias[2] = -4.0
                final_bias[3] = self.calibration_relation_source_floor_init_logit
            self.local_lowfreq_field_head = nn.Conv2d(width, 1, 3, padding=1)
            nn.init.zeros_(self.local_lowfreq_field_head.weight)
            nn.init.zeros_(self.local_lowfreq_field_head.bias)
            self.swin_residual_adapter = SwinResidualAdapter(
                channels=width,
                depth=self.swin_residual_depth,
                heads=self.swin_residual_heads,
                window_size=self.swin_residual_window_size,
            )
            self.swin_residual_field_head = nn.Conv2d(width, 1, 3, padding=1)
            self.swin_residual_coeff_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, 1),
            )
            nn.init.zeros_(self.swin_residual_field_head.weight)
            nn.init.zeros_(self.swin_residual_field_head.bias)
            nn.init.zeros_(self.swin_residual_coeff_mlp[-1].weight)
            nn.init.constant_(self.swin_residual_coeff_mlp[-1].bias, self.calibration_lowrank_coeff_init)
        elif self.calibration_head in ("target_stat_gain_bias_field_v1", "target_stat_gain_bias_lowres_delta_v1"):
            self.calibration_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, 2),
            )
            self.target_stat_gain_bias_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, 6),
            )
            nn.init.zeros_(self.target_stat_gain_bias_mlp[-1].weight)
            nn.init.zeros_(self.target_stat_gain_bias_mlp[-1].bias)
            with torch.no_grad():
                final_bias = self.target_stat_gain_bias_mlp[-1].bias
                final_bias[0] = self.calibration_relation_map_init_logit
                final_bias[1] = 2.0
                final_bias[2] = -4.0
                final_bias[3] = self.calibration_relation_source_floor_init_logit
                final_bias[4] = -4.0
                final_bias[5] = -4.0
            self.local_lowfreq_field_head = nn.Conv2d(width, 1, 3, padding=1)
            self.local_lowfreq_gain_head = nn.Conv2d(width, 1, 3, padding=1)
            self.local_lowfreq_bias_head = nn.Conv2d(width, 1, 3, padding=1)
            for head in (
                self.local_lowfreq_field_head,
                self.local_lowfreq_gain_head,
                self.local_lowfreq_bias_head,
            ):
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
            if self.calibration_head == "target_stat_gain_bias_lowres_delta_v1":
                out_channels = 1 if self.lowres_field_mode == "additive" else 2
                self.lowres_calibration_field_head = nn.Sequential(
                    nn.Conv2d(width, width, 3, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(width, out_channels, 1),
                )
                nn.init.zeros_(self.lowres_calibration_field_head[-1].weight)
                nn.init.zeros_(self.lowres_calibration_field_head[-1].bias)
        elif self.calibration_head == "lowres_calibration_field_v1":
            self.calibration_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, 2),
            )
            self.target_stat_gain_bias_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, 6),
            )
            nn.init.zeros_(self.target_stat_gain_bias_mlp[-1].weight)
            nn.init.zeros_(self.target_stat_gain_bias_mlp[-1].bias)
            with torch.no_grad():
                final_bias = self.target_stat_gain_bias_mlp[-1].bias
                final_bias[0] = self.calibration_relation_map_init_logit
                final_bias[1] = 2.0
                final_bias[2] = -4.0
                final_bias[3] = self.calibration_relation_source_floor_init_logit
                final_bias[4] = -4.0
                final_bias[5] = -4.0
            out_channels = 1 if self.lowres_field_mode == "additive" else 2
            self.lowres_calibration_field_head = nn.Sequential(
                nn.Conv2d(width, width, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(width, out_channels, 1),
            )
            nn.init.zeros_(self.lowres_calibration_field_head[-1].weight)
            nn.init.zeros_(self.lowres_calibration_field_head[-1].bias)
        elif self.calibration_head == "source_stat_percentile":
            self.calibration_blend_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, 1),
            )
            nn.init.zeros_(self.calibration_blend_mlp[-1].weight)
        elif self.calibration_head == "policy_param_lut_lowfreq":
            self.primitive_lut_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, self.calibration_knots - 1),
            )
            nn.init.zeros_(self.primitive_lut_mlp[-1].weight)
            nn.init.zeros_(self.primitive_lut_mlp[-1].bias)
        elif self.calibration_head == "direct_param_source_safe_v1":
            self.calibration_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, 2),
            )
            self.primitive_lut_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, self.calibration_knots - 1),
            )
            nn.init.zeros_(self.primitive_lut_mlp[-1].weight)
            nn.init.zeros_(self.primitive_lut_mlp[-1].bias)
        elif self.calibration_head in (
            "direct_param_target_percentile_v1",
            "direct_param_target_percentile_guarded_v2",
            "direct_param_target_percentile_guarded_v3",
        ):
            self.calibration_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, 2),
            )
            if self.calibration_head in (
                "direct_param_target_percentile_guarded_v2",
                "direct_param_target_percentile_guarded_v3",
            ):
                guard_outputs = 4 if self.calibration_head == "direct_param_target_percentile_guarded_v3" else 2
                self.percentile_guard_mlp = nn.Sequential(
                    nn.Linear(features_dim, cond_dim // 2),
                    nn.SiLU(),
                    nn.Linear(cond_dim // 2, guard_outputs),
                )
                nn.init.zeros_(self.percentile_guard_mlp[-1].weight)
                nn.init.zeros_(self.percentile_guard_mlp[-1].bias)
        elif self.calibration_head in ("direct_control_target_percentile_v1", "direct_control_gain_bias_field_v1"):
            self.calibration_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, 2),
            )
            self.percentile_control_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, 4),
            )
            nn.init.zeros_(self.percentile_control_mlp[-1].weight)
            nn.init.zeros_(self.percentile_control_mlp[-1].bias)
            with torch.no_grad():
                final_weight = self.percentile_control_mlp[-1].weight
                if self.direct_control_init_weight_std > 0.0:
                    nn.init.normal_(final_weight, mean=0.0, std=self.direct_control_init_weight_std)
                final_bias = self.percentile_control_mlp[-1].bias
                if self.direct_control_init_values is None:
                    final_bias[0] = -4.0
                    final_bias[1] = -4.0
                    final_bias[2] = -4.0
                    final_bias[3] = -6.0
                else:
                    init_values = torch.as_tensor(
                        self.direct_control_init_values,
                        dtype=final_bias.dtype,
                        device=final_bias.device,
                    )
                    final_bias.copy_(torch.logit(init_values))
            if self.calibration_head == "direct_control_gain_bias_field_v1":
                self.local_lowfreq_field_head = nn.Conv2d(width, 1, 3, padding=1)
                self.local_lowfreq_gain_head = nn.Conv2d(width, 1, 3, padding=1)
                self.local_lowfreq_bias_head = nn.Conv2d(width, 1, 3, padding=1)
                for head in (
                    self.local_lowfreq_field_head,
                    self.local_lowfreq_gain_head,
                    self.local_lowfreq_bias_head,
                ):
                    nn.init.zeros_(head.weight)
                    nn.init.zeros_(head.bias)
        elif self.calibration_head in (
            "target_percentile_relation_v1",
            "source_anchor_percentile_relation_v1",
            "base_anchor_percentile_relation_v1",
        ):
            self.calibration_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, 2),
            )
            relation_outputs = len(DEFAULT_PERCENTILES) + 3
            self.percentile_relation_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, relation_outputs),
            )
            nn.init.zeros_(self.percentile_relation_mlp[-1].weight)
            nn.init.zeros_(self.percentile_relation_mlp[-1].bias)
            with torch.no_grad():
                final_bias = self.percentile_relation_mlp[-1].bias
                final_bias[len(DEFAULT_PERCENTILES)] = self.calibration_relation_map_init_logit
                final_bias[len(DEFAULT_PERCENTILES) + 1] = self.calibration_relation_lowfreq_init_logit
                final_bias[len(DEFAULT_PERCENTILES) + 2] = self.calibration_relation_source_floor_init_logit
        elif self.calibration_head == "source_preserving_monotonic_lowfreq_v1":
            self.source_preserving_control_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, 4),
            )
            nn.init.zeros_(self.source_preserving_control_mlp[-1].weight)
            nn.init.zeros_(self.source_preserving_control_mlp[-1].bias)
            with torch.no_grad():
                final_bias = self.source_preserving_control_mlp[-1].bias
                final_bias[0] = self.calibration_relation_map_init_logit
                final_bias[1] = self.calibration_relation_lowfreq_init_logit
                final_bias[2] = -20.0
                final_bias[3] = -self.calibration_relation_source_floor_init_logit
        elif self.calibration_head == "calibrated_anchor_detail_v1":
            self.calibration_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, 2),
            )
            self.calibrated_anchor_detail_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, 4),
            )
            nn.init.zeros_(self.calibrated_anchor_detail_mlp[-1].weight)
            nn.init.zeros_(self.calibrated_anchor_detail_mlp[-1].bias)
            with torch.no_grad():
                final_bias = self.calibrated_anchor_detail_mlp[-1].bias
                final_bias[0] = self.calibration_relation_map_init_logit
                final_bias[1] = self.calibration_relation_lowfreq_init_logit
                final_bias[2] = -20.0
                final_bias[3] = -self.calibration_relation_source_floor_init_logit
        elif self.calibration_head == "target_percentile_primitive_bank_v1":
            self.calibration_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, 2),
            )
            primitive_count = 9
            self.primitive_weight_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, primitive_count),
            )
            self.primitive_blend_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, 1),
            )
            final = self.primitive_weight_mlp[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(self.primitive_blend_mlp[-1].weight)
            nn.init.zeros_(self.primitive_blend_mlp[-1].bias)
            with torch.no_grad():
                if self.primitive_bank_init == "uniform":
                    final.bias.zero_()
                else:
                    strength = self.primitive_bank_init_strength
                    final.bias.fill_(-strength)
                    final.bias[1 if self.primitive_bank_init == "raw" else 0] = strength
        elif self.calibration_head == "source_lut_lowfreq_texture":
            self.calibration_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, 2),
            )
            self.primitive_lut_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, self.calibration_knots - 1),
            )
            self.calibration_blend_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, 1),
            )
            self.calibration_texture_blend_mlp = nn.Sequential(
                nn.Linear(features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, 1),
            )
            nn.init.zeros_(self.primitive_lut_mlp[-1].weight)
            nn.init.zeros_(self.primitive_lut_mlp[-1].bias)
            nn.init.zeros_(self.calibration_mlp[-1].weight)
            nn.init.zeros_(self.calibration_mlp[-1].bias)
            nn.init.zeros_(self.calibration_blend_mlp[-1].weight)
            nn.init.zeros_(self.calibration_texture_blend_mlp[-1].weight)
            nn.init.constant_(self.calibration_blend_mlp[-1].bias, -4.0)
            nn.init.constant_(self.calibration_texture_blend_mlp[-1].bias, -4.0)
        elif self.calibration_head == "primitive_bank":
            if self.primitive_bank_mode == "basic":
                primitive_count = 4
            elif self.primitive_bank_mode == "source_safe_v1":
                primitive_count = 5
            elif self.primitive_bank_mode == "affine_source_safe_v1":
                primitive_count = 6
                self.calibration_mlp = nn.Sequential(
                    nn.Linear(features_dim, cond_dim // 2),
                    nn.SiLU(),
                    nn.Linear(cond_dim // 2, 2),
                )
            elif self.primitive_bank_mode == "policy_affine_source_safe_v1":
                primitive_count = 7
                self.calibration_mlp = nn.Sequential(
                    nn.Linear(features_dim, cond_dim // 2),
                    nn.SiLU(),
                    nn.Linear(cond_dim // 2, 2),
                )
            elif self.primitive_bank_mode == "policy_lowfreq_source_safe_v1":
                primitive_count = 8
                self.calibration_mlp = nn.Sequential(
                    nn.Linear(features_dim, cond_dim // 2),
                    nn.SiLU(),
                    nn.Linear(cond_dim // 2, 2),
                )
            else:
                primitive_count = 10 if self.primitive_bank_mode == "policy_lut_lowfreq_median_source_safe_v1" else 9
                self.calibration_mlp = nn.Sequential(
                    nn.Linear(features_dim, cond_dim // 2),
                    nn.SiLU(),
                    nn.Linear(cond_dim // 2, 2),
                )
                self.primitive_lut_mlp = nn.Sequential(
                    nn.Linear(features_dim, cond_dim // 2),
                    nn.SiLU(),
                    nn.Linear(cond_dim // 2, self.calibration_knots - 1),
                )
                nn.init.zeros_(self.primitive_lut_mlp[-1].weight)
                nn.init.zeros_(self.primitive_lut_mlp[-1].bias)
            primitive_features_dim = features_dim
            if self.primitive_bank_use_policy_features:
                primitive_features_dim += self.policy_feature_dim
            self.primitive_weight_mlp = nn.Sequential(
                nn.Linear(primitive_features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, primitive_count),
            )
            self.primitive_blend_mlp = nn.Sequential(
                nn.Linear(primitive_features_dim, cond_dim // 2),
                nn.SiLU(),
                nn.Linear(cond_dim // 2, 1),
            )
            if self.primitive_bank_init in ("raw", "affine"):
                final = self.primitive_weight_mlp[-1]
                nn.init.zeros_(final.weight)
                with torch.no_grad():
                    strength = self.primitive_bank_init_strength
                    final.bias.fill_(-strength)
                    if self.primitive_bank_init == "affine":
                        final.bias[0] = strength
                    elif self.primitive_bank_mode in (
                        "affine_source_safe_v1",
                        "policy_affine_source_safe_v1",
                        "policy_lowfreq_source_safe_v1",
                    ):
                        final.bias[1] = strength
                    else:
                        final.bias[0] = strength

    @staticmethod
    def _target_percentile_value(percentiles: dict, key: float) -> float:
        for token in (f"{key:g}", str(int(key)), f"{key:.1f}"):
            if token in percentiles:
                return float(percentiles[token])
        raise KeyError(f"Missing percentile {key:g}")

    @classmethod
    def _load_target_robust_stats(cls, target_stats_path: str | None, num_contrasts: int) -> torch.Tensor:
        if target_stats_path is None:
            raise ValueError("target_stats_path is required")
        payload = json.loads(Path(target_stats_path).read_text())
        stats_table = payload.get("stats", payload)
        stats = torch.full((num_contrasts, len(FIELDS), 3), float("nan"), dtype=torch.float32)
        for contrast_idx, contrast in enumerate(CONTRASTS[:num_contrasts]):
            contrast_stats = stats_table.get(contrast, {})
            for field_idx, field in enumerate(FIELDS):
                field_stats = contrast_stats.get(field, {})
                percentiles = field_stats.get("percentiles", field_stats)
                if not percentiles:
                    continue
                stats[contrast_idx, field_idx, 0] = cls._target_percentile_value(percentiles, 5.0)
                stats[contrast_idx, field_idx, 1] = cls._target_percentile_value(percentiles, 50.0)
                stats[contrast_idx, field_idx, 2] = cls._target_percentile_value(percentiles, 95.0)
        return stats

    @classmethod
    def _load_target_percentile_stats(cls, target_stats_path: str | None, num_contrasts: int) -> torch.Tensor:
        if target_stats_path is None:
            raise ValueError("target_stats_path is required")
        payload = json.loads(Path(target_stats_path).read_text())
        stats_table = payload.get("stats", payload)
        stats = torch.full(
            (num_contrasts, len(FIELDS), len(DEFAULT_PERCENTILES)),
            float("nan"),
            dtype=torch.float32,
        )
        for contrast_idx, contrast in enumerate(CONTRASTS[:num_contrasts]):
            contrast_stats = stats_table.get(contrast, {})
            for field_idx, field in enumerate(FIELDS):
                field_stats = contrast_stats.get(field, {})
                percentiles = field_stats.get("percentiles", field_stats)
                if not percentiles:
                    continue
                for level_idx, level in enumerate(DEFAULT_PERCENTILES):
                    stats[contrast_idx, field_idx, level_idx] = cls._target_percentile_value(percentiles, level)
        return stats

    def encode_condition(
        self,
        source_field_id: torch.Tensor,
        target_field_id: torch.Tensor,
        contrast_id: torch.Tensor,
    ) -> torch.Tensor:
        src = self.field_embed(source_field_id)
        tgt = self.field_embed(target_field_id)
        con = self.contrast_embed(contrast_id)
        values = torch.tensor([FIELD_VALUES[name] for name in FIELDS], device=source_field_id.device)
        src_b0 = values[source_field_id].clamp_min(1e-6)
        tgt_b0 = values[target_field_id].clamp_min(1e-6)
        delta = tgt_b0 - src_b0
        numeric = torch.stack(
            [
                torch.log(src_b0),
                torch.log(tgt_b0),
                torch.log(tgt_b0 / src_b0),
                delta,
                torch.sign(delta),
                delta.abs(),
            ],
            dim=1,
        )
        return self.condition_mlp(torch.cat([src, tgt, con, numeric], dim=1))

    def _apply_hyper_residual_adapter(
        self,
        feature_map: torch.Tensor,
        cond: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_hyper_residual_head:
            return residual
        params = self.hyper_residual_mlp(cond)
        bias = params[:, -1:]
        if self.hyper_residual_mode == "full_kernel_delta":
            channels = int(feature_map.shape[1])
            weights = params[:, :-1].reshape(-1, 1, channels, 3, 3)
            dynamic_parts = [
                F.conv2d(
                    feature_map[idx : idx + 1],
                    weights[idx] / math.sqrt(float(channels * 9)),
                    bias=bias[idx],
                    padding=1,
                )
                for idx in range(feature_map.shape[0])
            ]
            dynamic = torch.cat(dynamic_parts, dim=0)
        else:
            weights = params[:, :-1]
            dynamic = (feature_map * weights[:, :, None, None]).sum(dim=1, keepdim=True)
            dynamic = dynamic / math.sqrt(float(feature_map.shape[1]))
            dynamic = dynamic + bias[:, :, None, None]
        return residual + self.hyper_residual_scale * dynamic

    @staticmethod
    def _legacy_source_stats(source_center: torch.Tensor) -> torch.Tensor:
        flat = source_center.flatten(1)
        mask = flat > 0
        safe = torch.where(mask, flat, torch.zeros_like(flat))
        count = mask.sum(dim=1).clamp_min(1)
        mean = safe.sum(dim=1) / count
        var = torch.where(mask, (flat - mean[:, None]).pow(2), torch.zeros_like(flat)).sum(dim=1) / count
        std = torch.sqrt(var + 1e-6)
        quantiles = torch.quantile(
            flat.clamp(0, 1),
            torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], device=flat.device),
            dim=1,
        ).T
        return torch.cat([mean[:, None], std[:, None], quantiles], dim=1)

    @staticmethod
    def _masked_basic13_source_stats(source: torch.Tensor, mask_eps: float = 1e-4) -> torch.Tensor:
        x = source.clamp(0.0, 1.0)
        flat = x.flatten(1)
        mask = flat > float(mask_eps)
        quantile_levels = torch.tensor([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99], device=x.device, dtype=x.dtype)
        rows = []
        for idx in range(flat.shape[0]):
            values = flat[idx][mask[idx]]
            if values.numel() == 0:
                rows.append(flat.new_zeros(11))
                continue
            percentiles = torch.quantile(values, quantile_levels)
            mean = values.mean()
            std = values.std(unbiased=False).clamp_min(0.0)
            iqr_95_5 = percentiles[5] - percentiles[1]
            occupancy = mask[idx].to(dtype=x.dtype).mean()
            rows.append(torch.cat([mean[None], std[None], percentiles, iqr_95_5[None], occupancy[None]]))
        intensity_stats = torch.stack(rows, dim=0)

        dx = x[..., :, 1:] - x[..., :, :-1]
        dx = F.pad(dx.abs(), (1, 0, 0, 0))
        dy = x[..., 1:, :] - x[..., :-1, :]
        dy = F.pad(dy.abs(), (0, 0, 1, 0))
        grad = torch.sqrt(dx.pow(2) + dy.pow(2) + 1e-12).flatten(1)
        grad_rows = []
        for idx in range(flat.shape[0]):
            grad_values = grad[idx][mask[idx]]
            if grad_values.numel() == 0:
                grad_rows.append(flat.new_zeros(2))
                continue
            grad_rows.append(torch.stack([grad_values.mean(), torch.quantile(grad_values, grad_values.new_tensor(0.95))]))
        grad_stats = torch.stack(grad_rows, dim=0)
        return torch.cat([intensity_stats, grad_stats], dim=1)

    @staticmethod
    def _source_stats(source_center: torch.Tensor, mode: str = "legacy7") -> torch.Tensor:
        mode = mode.strip().lower()
        if mode == "legacy7":
            return CondNAFUNet._legacy_source_stats(source_center)
        if mode == "masked_basic13":
            return CondNAFUNet._masked_basic13_source_stats(source_center)
        raise ValueError(f"Unknown source_stats_mode: {mode}")

    @staticmethod
    def _masked_robust_source_stats(source_center: torch.Tensor, mask_eps: float = 1e-4) -> torch.Tensor:
        flat = source_center.flatten(1).clamp(0.0, 1.0)
        quantile_levels = torch.tensor([0.05, 0.5, 0.95], device=flat.device, dtype=flat.dtype)
        rows = []
        for idx in range(flat.shape[0]):
            values = flat[idx][flat[idx] > float(mask_eps)]
            if values.numel() == 0:
                values = flat[idx]
            rows.append(torch.quantile(values, quantile_levels))
        return torch.stack(rows, dim=0)

    @staticmethod
    def _masked_percentile_source_stats(
        source_center: torch.Tensor,
        quantile_levels: torch.Tensor,
        mask_eps: float = 1e-4,
    ) -> torch.Tensor:
        flat = source_center.flatten(1).clamp(0.0, 1.0)
        levels = quantile_levels.to(device=flat.device, dtype=flat.dtype)
        rows = []
        for idx in range(flat.shape[0]):
            values = flat[idx][flat[idx] > float(mask_eps)]
            if values.numel() == 0:
                values = flat[idx]
            rows.append(torch.quantile(values, levels))
        return torch.stack(rows, dim=0)

    @staticmethod
    def _interpolate_lut(x: torch.Tensor, y_knots: torch.Tensor) -> torch.Tensor:
        knots = y_knots.shape[1]
        scaled = x.clamp(0.0, 1.0) * float(knots - 1)
        idx0 = scaled.floor().long().clamp(0, knots - 2)
        idx1 = idx0 + 1
        frac = scaled - idx0.to(dtype=scaled.dtype)

        y0 = torch.gather(y_knots, 1, idx0.flatten(1)).view_as(x)
        y1 = torch.gather(y_knots, 1, idx1.flatten(1)).view_as(x)
        return y0 + (y1 - y0) * frac

    @staticmethod
    def _interpolate_piecewise(x: torch.Tensor, x_knots: torch.Tensor, y_knots: torch.Tensor) -> torch.Tensor:
        rows = []
        for idx in range(x.shape[0]):
            xs = torch.cummax(x_knots[idx].to(device=x.device, dtype=x.dtype), dim=0).values.clone()
            for knot_idx in range(1, xs.shape[0]):
                if xs[knot_idx] <= xs[knot_idx - 1]:
                    xs[knot_idx] = xs[knot_idx - 1] + 1e-6
            ys = y_knots[idx].to(device=x.device, dtype=x.dtype)
            values = x[idx].flatten().clamp(0.0, 1.0)
            upper = torch.bucketize(values, xs).clamp(1, xs.shape[0] - 1)
            lower = upper - 1
            x0 = xs[lower]
            x1 = xs[upper]
            y0 = ys[lower]
            y1 = ys[upper]
            frac = ((values - x0) / (x1 - x0).clamp_min(1e-6)).clamp(0.0, 1.0)
            rows.append((y0 + (y1 - y0) * frac).view_as(x[idx]))
        return torch.stack(rows, dim=0).clamp(0.0, 1.0)

    @staticmethod
    def _smooth(x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        if kernel_size <= 1:
            return x
        pad = kernel_size // 2
        padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        return F.avg_pool2d(padded, kernel_size=kernel_size, stride=1)

    @staticmethod
    def _median_smooth(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        if kernel_size <= 1:
            return x
        if kernel_size % 2 == 0:
            raise ValueError("median kernel_size must be odd")
        pad = kernel_size // 2
        padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        patches = F.unfold(padded, kernel_size=kernel_size, stride=1)
        patches = patches.view(x.shape[0], x.shape[1], kernel_size * kernel_size, x.shape[2], x.shape[3])
        return patches.median(dim=2).values

    @staticmethod
    def _source_edge_gate(
        source_center: torch.Tensor,
        floor: float = 0.25,
        strength: float = 1.0,
    ) -> torch.Tensor:
        dx = (source_center[..., :, 1:] - source_center[..., :, :-1]).abs()
        dx = F.pad(dx, (1, 0, 0, 0))
        dy = (source_center[..., 1:, :] - source_center[..., :-1, :]).abs()
        dy = F.pad(dy, (0, 0, 1, 0))
        grad = torch.sqrt(dx.pow(2) + dy.pow(2) + 1e-12)
        scale = grad.flatten(1).mean(dim=1).clamp_min(1e-6)[:, None, None, None]
        edge = grad / (grad + scale)
        gate = 1.0 - float(strength) * edge
        return gate.clamp(float(floor), 1.0)

    def _apply_source_gate(self, delta: torch.Tensor, source_center: torch.Tensor) -> torch.Tensor:
        if self.calibration_source_gate_mode == "none":
            return delta
        if self.calibration_source_gate_mode == "source_edge_v1":
            gate = self._source_edge_gate(
                source_center,
                floor=self.calibration_source_gate_floor,
                strength=self.calibration_source_gate_strength,
            )
            return delta * gate
        raise RuntimeError(f"Unhandled calibration_source_gate_mode: {self.calibration_source_gate_mode}")

    def _apply_tissue_bridge(
        self,
        calibrated: torch.Tensor,
        tissue_logits: torch.Tensor | None,
        features: torch.Tensor,
        source_center: torch.Tensor,
        base_center: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if not self.use_tissue_bridge and not self.use_tissue_local_bridge:
            return calibrated, None, None
        if tissue_logits is None:
            raise RuntimeError("tissue bridge requires tissue_logits")
        probs = torch.softmax(tissue_logits, dim=1)
        delta = calibrated.new_zeros(calibrated.shape)
        coefficients = None
        if self.use_tissue_bridge:
            coefficients = self.tissue_bridge_max * torch.tanh(self.tissue_bridge_mlp(features))
            scalar_delta = (probs * coefficients[:, :, None, None]).sum(dim=1, keepdim=True)
            scalar_delta = self._smooth(scalar_delta, self.tissue_bridge_kernel_size)
            delta = delta + scalar_delta
        if self.use_tissue_local_bridge:
            source_low = self._smooth(source_center.clamp(0.0, 1.0), self.tissue_local_bridge_kernel_size)
            anchor_low = self._smooth(calibrated.clamp(0.0, 1.0), self.tissue_local_bridge_kernel_size)
            base_low = self._smooth(base_center.clamp(0.0, 1.0), self.tissue_local_bridge_kernel_size)
            local_state = torch.cat(
                [
                    probs,
                    source_low,
                    anchor_low,
                    (anchor_low - base_low).clamp(-1.0, 1.0),
                ],
                dim=1,
            )
            local_delta = self.tissue_local_bridge_max * torch.tanh(self.tissue_local_bridge_net(local_state))
            local_delta = self._smooth(local_delta, self.tissue_local_bridge_kernel_size)
            delta = delta + local_delta
        delta = self._apply_source_gate(delta, source_center)
        return (calibrated + delta).clamp(0.0, 1.0), delta, coefficients

    def _apply_affine_head(self, raw_pred: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        a_raw, b_raw = self.calibration_mlp(features).chunk(2, dim=1)
        a = (1.0 + 0.5 * torch.tanh(a_raw))[:, :, None, None]
        b = (0.2 * torch.tanh(b_raw))[:, :, None, None]
        return torch.clamp(a * raw_pred + b, 0.0, 1.0)

    def _apply_affine_gamma_head(self, raw_pred: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        a_raw, b_raw = self.calibration_mlp(features).chunk(2, dim=1)
        gamma_raw = self.calibration_gamma_mlp(features)
        a = (1.0 + 0.5 * torch.tanh(a_raw))[:, :, None, None]
        b = (0.2 * torch.tanh(b_raw))[:, :, None, None]
        gamma = torch.exp(0.6931471805599453 * torch.tanh(gamma_raw))[:, :, None, None]
        return torch.clamp(a * raw_pred.clamp(0.0, 1.0).pow(gamma) + b, 0.0, 1.0)

    def _apply_source_lut_primitive(self, source_center: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        interval_logits = self.primitive_lut_mlp(features)
        intervals = F.softplus(interval_logits) + 1e-4
        cumulative = torch.cumsum(intervals, dim=1)
        normalized = cumulative / cumulative[:, -1:].clamp_min(1e-6)
        zero = normalized.new_zeros(normalized.shape[0], 1)
        y_knots = torch.cat([zero, normalized], dim=1).clamp(0.0, 1.0)
        return self._interpolate_lut(source_center, y_knots)

    def _apply_local_lowfreq_field_head(
        self,
        source_center: torch.Tensor,
        feature_map: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if feature_map is None:
            raise RuntimeError("local_lowfreq_field_v1 calibration requires decoder feature_map")
        source = source_center.clamp(0.0, 1.0)
        field = torch.tanh(self.local_lowfreq_field_head(feature_map))
        field = self._smooth(field, self.calibration_lowfreq_kernel_size)
        delta = float(self.calibration_blend_max) * field
        delta = self._apply_source_gate(delta, source)
        calibrated = (source + delta).clamp(0.0, 1.0)
        return calibrated, field, source, calibrated - source

    def _apply_target_stat_lowfreq_field_head(
        self,
        source_center: torch.Tensor,
        feature_map: torch.Tensor | None,
        features: torch.Tensor,
        contrast_id: torch.Tensor,
        target_field_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._last_guard_features = None
        if feature_map is None:
            raise RuntimeError("target_stat_lowfreq_field_v1 calibration requires decoder feature_map")
        if not hasattr(self, "target_percentile_stats"):
            raise RuntimeError("target_stat_lowfreq_field_v1 requires target_percentile_stats buffer")
        target_knots = self.target_percentile_stats.to(device=source_center.device, dtype=source_center.dtype)[
            contrast_id, target_field_id
        ]
        if not torch.isfinite(target_knots).all():
            raise RuntimeError("Missing target percentile stats for requested contrast/target field")

        source = source_center.clamp(0.0, 1.0)
        anchor = source
        source_knots = self._masked_percentile_source_stats(
            source,
            self.source_stat_percentile_levels,
        )

        controls = self.target_stat_lowfreq_mlp(features)
        control_map = torch.sigmoid(controls[:, 0:1])
        control_stat = torch.sigmoid(controls[:, 1:2])
        control_field = torch.sigmoid(controls[:, 2:3])
        source_floor = torch.sigmoid(controls[:, 3:4])

        map_beta = min(max(self.calibration_blend_max, 0.0), 1.0) * control_map
        stat_alpha = min(max(self.calibration_blend_max, 0.0), 1.0) * control_stat
        learned_alpha = max(self.calibration_texture_blend_max, 0.0) * control_field

        y_knots = source_knots + map_beta * (target_knots - source_knots)
        y_knots = torch.cummax(y_knots.clamp(0.0, 1.0), dim=1).values
        mapped = self._interpolate_piecewise(source, source_knots, y_knots).clamp(0.0, 1.0)

        source_low = self._smooth(source, self.calibration_lowfreq_kernel_size)
        mapped_low = self._smooth(mapped, self.calibration_lowfreq_kernel_size)
        stat_delta = stat_alpha[:, :, None, None] * (mapped_low - source_low)

        field = torch.tanh(self.local_lowfreq_field_head(feature_map))
        field = self._smooth(field, self.calibration_lowfreq_kernel_size)
        learned_delta = learned_alpha[:, :, None, None] * field

        delta = self._apply_source_gate(stat_delta + learned_delta, source)
        calibrated = (anchor + delta).clamp(0.0, 1.0)
        calibrated = (source_floor[:, :, None, None] * source + (1.0 - source_floor[:, :, None, None]) * calibrated).clamp(
            0.0,
            1.0,
        )

        self._last_guard_features = torch.cat(
            [
                map_beta.clamp(0.0, 1.0),
                stat_alpha.clamp(0.0, 1.0),
                learned_alpha.clamp(0.0, 1.0),
                source_floor.clamp(0.0, 1.0),
            ],
            dim=1,
        )
        return calibrated, y_knots, field, anchor, calibrated - anchor

    def _apply_target_stat_anchor_detail_head(
        self,
        raw_pred: torch.Tensor,
        source_center: torch.Tensor,
        feature_map: torch.Tensor | None,
        features: torch.Tensor,
        contrast_id: torch.Tensor,
        target_field_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._last_guard_features = None
        if feature_map is None:
            raise RuntimeError("target_stat_anchor_detail_v1 calibration requires decoder feature_map")
        if not hasattr(self, "target_percentile_stats"):
            raise RuntimeError("target_stat_anchor_detail_v1 requires target_percentile_stats buffer")
        target_knots = self.target_percentile_stats.to(device=raw_pred.device, dtype=raw_pred.dtype)[
            contrast_id, target_field_id
        ]
        if not torch.isfinite(target_knots).all():
            raise RuntimeError("Missing target percentile stats for requested contrast/target field")

        source = source_center.clamp(0.0, 1.0)
        base_anchor = self._apply_affine_head(raw_pred, features).clamp(0.0, 1.0)
        source_knots = self._masked_percentile_source_stats(
            source,
            self.source_stat_percentile_levels,
        )
        y_knots = torch.cummax(target_knots.clamp(0.0, 1.0), dim=1).values
        mapped = self._interpolate_piecewise(source, source_knots, y_knots).clamp(0.0, 1.0)

        controls = self.target_stat_anchor_detail_mlp(features)
        anchor_beta = min(max(self.calibration_blend_max, 0.0), 1.0) * torch.sigmoid(controls[:, 0:1])
        detail_alpha = min(max(self.calibration_texture_blend_max, 0.0), 1.0) * torch.sigmoid(controls[:, 1:2])
        learned_alpha = min(max(self.calibration_texture_blend_max, 0.0), 1.0) * torch.sigmoid(controls[:, 2:3])
        base_floor = torch.sigmoid(controls[:, 3:4])

        source_low = self._smooth(source, self.calibration_lowfreq_kernel_size)
        mapped_low = self._smooth(mapped, self.calibration_lowfreq_kernel_size)
        source_detail = source - source_low
        stat_detail_anchor = (mapped_low + detail_alpha[:, :, None, None] * source_detail).clamp(0.0, 1.0)
        target_anchor = ((1.0 - anchor_beta[:, :, None, None]) * base_anchor + anchor_beta[:, :, None, None] * stat_detail_anchor).clamp(
            0.0,
            1.0,
        )
        anchor = (base_floor[:, :, None, None] * base_anchor + (1.0 - base_floor[:, :, None, None]) * target_anchor).clamp(
            0.0,
            1.0,
        )

        field = torch.tanh(self.local_lowfreq_field_head(feature_map))
        field = self._smooth(field, self.calibration_lowfreq_kernel_size)
        delta = learned_alpha[:, :, None, None] * field
        delta = self._apply_source_gate(delta, source)
        delta = (1.0 - base_floor[:, :, None, None]) * delta
        calibrated = (anchor + delta).clamp(0.0, 1.0)

        self._last_guard_features = torch.cat(
            [
                anchor_beta.clamp(0.0, 1.0),
                detail_alpha.clamp(0.0, 1.0),
                learned_alpha.clamp(0.0, 1.0),
                base_floor.clamp(0.0, 1.0),
            ],
            dim=1,
        )
        return calibrated, y_knots, field, anchor, calibrated - anchor

    def _apply_target_stat_gain_bias_field_head(
        self,
        raw_pred: torch.Tensor,
        source_center: torch.Tensor,
        feature_map: torch.Tensor | None,
        features: torch.Tensor,
        contrast_id: torch.Tensor,
        target_field_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._last_guard_features = None
        if feature_map is None:
            raise RuntimeError("target_stat_gain_bias_field_v1 calibration requires decoder feature_map")
        if not hasattr(self, "target_percentile_stats"):
            raise RuntimeError("target_stat_gain_bias_field_v1 requires target_percentile_stats buffer")
        target_knots = self.target_percentile_stats.to(device=raw_pred.device, dtype=raw_pred.dtype)[
            contrast_id, target_field_id
        ]
        if not torch.isfinite(target_knots).all():
            raise RuntimeError("Missing target percentile stats for requested contrast/target field")

        source = source_center.clamp(0.0, 1.0)
        base_anchor = self._apply_affine_head(raw_pred, features).clamp(0.0, 1.0)
        source_knots = self._masked_percentile_source_stats(
            source,
            self.source_stat_percentile_levels,
        )
        y_knots = torch.cummax(target_knots.clamp(0.0, 1.0), dim=1).values
        mapped = self._interpolate_piecewise(source, source_knots, y_knots).clamp(0.0, 1.0)

        controls = self.target_stat_gain_bias_mlp(features)
        anchor_beta = min(max(self.calibration_blend_max, 0.0), 1.0) * torch.sigmoid(controls[:, 0:1])
        detail_alpha = min(max(self.calibration_texture_blend_max, 0.0), 1.0) * torch.sigmoid(controls[:, 1:2])
        learned_alpha = min(max(self.calibration_texture_blend_max, 0.0), 1.0) * torch.sigmoid(controls[:, 2:3])
        base_floor = torch.sigmoid(controls[:, 3:4])
        gain_strength = self.calibration_gain_scale_max * torch.sigmoid(controls[:, 4:5])
        bias_strength = self.calibration_bias_scale_max * torch.sigmoid(controls[:, 5:6])

        source_low = self._smooth(source, self.calibration_lowfreq_kernel_size)
        mapped_low = self._smooth(mapped, self.calibration_lowfreq_kernel_size)
        source_detail = source - source_low
        stat_detail_anchor = (mapped_low + detail_alpha[:, :, None, None] * source_detail).clamp(0.0, 1.0)
        target_anchor = ((1.0 - anchor_beta[:, :, None, None]) * base_anchor + anchor_beta[:, :, None, None] * stat_detail_anchor).clamp(
            0.0,
            1.0,
        )
        anchor = (base_floor[:, :, None, None] * base_anchor + (1.0 - base_floor[:, :, None, None]) * target_anchor).clamp(
            0.0,
            1.0,
        )

        gain_field = torch.tanh(self.local_lowfreq_gain_head(feature_map))
        gain_field = self._smooth(gain_field, self.calibration_lowfreq_kernel_size)
        bias_field = torch.tanh(self.local_lowfreq_bias_head(feature_map))
        bias_field = self._smooth(bias_field, self.calibration_lowfreq_kernel_size)
        gain = torch.exp(gain_strength[:, :, None, None] * gain_field).clamp(0.5, 2.0)
        bias = bias_strength[:, :, None, None] * bias_field
        bias = self._apply_source_gate(bias, source)

        field = torch.tanh(self.local_lowfreq_field_head(feature_map))
        field = self._smooth(field, self.calibration_lowfreq_kernel_size)
        residual_delta = learned_alpha[:, :, None, None] * field
        residual_delta = self._apply_source_gate(residual_delta, source)
        residual_delta = (1.0 - base_floor[:, :, None, None]) * residual_delta

        calibrated = (anchor * gain + bias + residual_delta).clamp(0.0, 1.0)

        self._last_guard_features = torch.cat(
            [
                anchor_beta.clamp(0.0, 1.0),
                detail_alpha.clamp(0.0, 1.0),
                learned_alpha.clamp(0.0, 1.0),
                base_floor.clamp(0.0, 1.0),
                torch.sigmoid(controls[:, 4:5]).clamp(0.0, 1.0),
                torch.sigmoid(controls[:, 5:6]).clamp(0.0, 1.0),
            ],
            dim=1,
        )
        field_stack = torch.cat([gain_field, bias_field, field], dim=1)
        return calibrated, y_knots, field_stack, anchor, calibrated - anchor

    def _target_stat_gain_bias_anchor(
        self,
        raw_pred: torch.Tensor,
        source_center: torch.Tensor,
        features: torch.Tensor,
        contrast_id: torch.Tensor,
        target_field_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not hasattr(self, "target_percentile_stats"):
            raise RuntimeError(f"{self.calibration_head} requires target_percentile_stats buffer")
        target_knots = self.target_percentile_stats.to(device=raw_pred.device, dtype=raw_pred.dtype)[
            contrast_id, target_field_id
        ]
        if not torch.isfinite(target_knots).all():
            raise RuntimeError("Missing target percentile stats for requested contrast/target field")

        source = source_center.clamp(0.0, 1.0)
        base_anchor = self._apply_affine_head(raw_pred, features).clamp(0.0, 1.0)
        source_knots = self._masked_percentile_source_stats(
            source,
            self.source_stat_percentile_levels,
        )
        y_knots = torch.cummax(target_knots.clamp(0.0, 1.0), dim=1).values
        mapped = self._interpolate_piecewise(source, source_knots, y_knots).clamp(0.0, 1.0)

        controls = self.target_stat_gain_bias_mlp(features)
        anchor_beta = min(max(self.calibration_blend_max, 0.0), 1.0) * torch.sigmoid(controls[:, 0:1])
        detail_alpha = min(max(self.calibration_texture_blend_max, 0.0), 1.0) * torch.sigmoid(controls[:, 1:2])
        base_floor = torch.sigmoid(controls[:, 3:4])

        source_low = self._smooth(source, self.calibration_lowfreq_kernel_size)
        mapped_low = self._smooth(mapped, self.calibration_lowfreq_kernel_size)
        source_detail = source - source_low
        stat_detail_anchor = (mapped_low + detail_alpha[:, :, None, None] * source_detail).clamp(0.0, 1.0)
        target_anchor = ((1.0 - anchor_beta[:, :, None, None]) * base_anchor + anchor_beta[:, :, None, None] * stat_detail_anchor).clamp(
            0.0,
            1.0,
        )
        anchor = (base_floor[:, :, None, None] * base_anchor + (1.0 - base_floor[:, :, None, None]) * target_anchor).clamp(
            0.0,
            1.0,
        )
        return source, anchor, y_knots, controls, anchor_beta, detail_alpha

    def _apply_lowres_calibration_field_head(
        self,
        raw_pred: torch.Tensor,
        source_center: torch.Tensor,
        feature_map: torch.Tensor | None,
        features: torch.Tensor,
        contrast_id: torch.Tensor,
        target_field_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._last_guard_features = None
        self._last_lowres_field = None
        self._last_lowres_field_upsampled = None
        self._last_lowres_field_shape = None
        self._last_lowres_full_shape = None
        if feature_map is None:
            raise RuntimeError("lowres_calibration_field_v1 calibration requires decoder feature_map")

        source, anchor, y_knots, controls, anchor_beta, detail_alpha = self._target_stat_gain_bias_anchor(
            raw_pred,
            source_center,
            features,
            contrast_id,
            target_field_id,
        )
        if self.lowres_field_downsample > 1:
            pooled = F.avg_pool2d(
                feature_map,
                kernel_size=self.lowres_field_downsample,
                stride=self.lowres_field_downsample,
                ceil_mode=True,
            )
        else:
            pooled = feature_map
        lowres = torch.tanh(self.lowres_calibration_field_head(pooled))
        field = F.interpolate(lowres, size=anchor.shape[-2:], mode="bilinear", align_corners=False)

        self._last_lowres_field = lowres
        self._last_lowres_field_upsampled = field
        self._last_lowres_field_shape = tuple(int(value) for value in lowres.shape)
        self._last_lowres_full_shape = tuple(int(value) for value in field.shape)

        learned_alpha = min(max(self.calibration_texture_blend_max, 0.0), 1.0) * torch.sigmoid(controls[:, 2:3])
        base_floor = torch.sigmoid(controls[:, 3:4])
        if self.lowres_field_mode == "additive":
            delta = learned_alpha[:, :, None, None] * field
            delta = self._apply_source_gate(delta, source)
        else:
            gain_field, bias_field = field.chunk(2, dim=1)
            gain_strength = self.calibration_gain_scale_max * torch.sigmoid(controls[:, 4:5])
            bias_strength = self.calibration_bias_scale_max * torch.sigmoid(controls[:, 5:6])
            gain = torch.exp(gain_strength[:, :, None, None] * gain_field).clamp(0.5, 2.0)
            bias = bias_strength[:, :, None, None] * bias_field
            bias = self._apply_source_gate(bias, source)
            delta = anchor * (gain - 1.0) + bias
        delta = (1.0 - base_floor[:, :, None, None]) * delta
        calibrated = (anchor + delta).clamp(0.0, 1.0)

        self._last_guard_features = torch.cat(
            [
                anchor_beta.clamp(0.0, 1.0),
                detail_alpha.clamp(0.0, 1.0),
                learned_alpha.clamp(0.0, 1.0),
                base_floor.clamp(0.0, 1.0),
                torch.sigmoid(controls[:, 4:5]).clamp(0.0, 1.0),
                torch.sigmoid(controls[:, 5:6]).clamp(0.0, 1.0),
            ],
            dim=1,
        )
        return calibrated, y_knots, field, anchor, calibrated - anchor

    def _apply_target_stat_gain_bias_lowres_delta_head(
        self,
        raw_pred: torch.Tensor,
        source_center: torch.Tensor,
        feature_map: torch.Tensor | None,
        features: torch.Tensor,
        contrast_id: torch.Tensor,
        target_field_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        calibrated, y_knots, inherited_field, anchor, _ = self._apply_target_stat_gain_bias_field_head(
            raw_pred,
            source_center,
            feature_map,
            features,
            contrast_id,
            target_field_id,
        )
        if feature_map is None:
            raise RuntimeError("target_stat_gain_bias_lowres_delta_v1 calibration requires decoder feature_map")

        if self.lowres_field_downsample > 1:
            pooled = F.avg_pool2d(
                feature_map,
                kernel_size=self.lowres_field_downsample,
                stride=self.lowres_field_downsample,
                ceil_mode=True,
            )
        else:
            pooled = feature_map
        lowres = torch.tanh(self.lowres_calibration_field_head(pooled))
        field = F.interpolate(lowres, size=calibrated.shape[-2:], mode="bilinear", align_corners=False)

        self._last_lowres_field = lowres
        self._last_lowres_field_upsampled = field
        self._last_lowres_field_shape = tuple(int(value) for value in lowres.shape)
        self._last_lowres_full_shape = tuple(int(value) for value in field.shape)

        source = source_center.clamp(0.0, 1.0)
        controls = self.target_stat_gain_bias_mlp(features)
        learned_alpha = min(max(self.calibration_texture_blend_max, 0.0), 1.0) * torch.sigmoid(controls[:, 2:3])
        base_floor = torch.sigmoid(controls[:, 3:4])
        if self.lowres_field_mode == "additive":
            lowres_delta = learned_alpha[:, :, None, None] * field
            lowres_delta = self._apply_source_gate(lowres_delta, source)
        else:
            gain_field, bias_field = field.chunk(2, dim=1)
            gain_strength = self.calibration_gain_scale_max * torch.sigmoid(controls[:, 4:5])
            bias_strength = self.calibration_bias_scale_max * torch.sigmoid(controls[:, 5:6])
            gain = torch.exp(gain_strength[:, :, None, None] * gain_field).clamp(0.5, 2.0)
            bias = bias_strength[:, :, None, None] * bias_field
            bias = self._apply_source_gate(bias, source)
            lowres_delta = calibrated * (gain - 1.0) + bias
        lowres_delta = (1.0 - base_floor[:, :, None, None]) * lowres_delta
        pred = (calibrated + lowres_delta).clamp(0.0, 1.0)

        guard_features = getattr(self, "_last_guard_features", None)
        lowres_guard = torch.cat(
            [
                learned_alpha.clamp(0.0, 1.0),
                base_floor.clamp(0.0, 1.0),
                lowres.detach().abs().mean(dim=(1, 2, 3), keepdim=False)[:, None].clamp(0.0, 1.0),
            ],
            dim=1,
        )
        self._last_guard_features = lowres_guard if guard_features is None else torch.cat([guard_features, lowres_guard], dim=1)
        self._last_calibration_extra_delta = lowres_delta
        field_stack = torch.cat([inherited_field, field], dim=1)
        return pred, y_knots, field_stack, anchor, pred - anchor

    def _apply_target_stat_anchor_lowrank_residual_head(
        self,
        raw_pred: torch.Tensor,
        source_center: torch.Tensor,
        feature_map: torch.Tensor | None,
        features: torch.Tensor,
        contrast_id: torch.Tensor,
        target_field_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        calibrated, y_knots, field, anchor, anchor_delta = self._apply_target_stat_anchor_detail_head(
            raw_pred,
            source_center,
            feature_map,
            features,
            contrast_id,
            target_field_id,
        )
        if feature_map is None:
            raise RuntimeError("target_stat_anchor_lowrank_residual_v1 requires decoder feature_map")

        basis = torch.tanh(self.lowrank_residual_basis_head(feature_map))
        basis = self._smooth(basis, self.calibration_lowfreq_kernel_size)
        coeff = torch.tanh(self.lowrank_residual_coeff_mlp(features))
        lowrank_delta = (basis * coeff[:, :, None, None]).sum(dim=1, keepdim=True)
        lowrank_delta = self.calibration_lowrank_delta_max * lowrank_delta
        lowrank_delta = lowrank_delta.clamp(-self.calibration_lowrank_delta_max, self.calibration_lowrank_delta_max)
        lowrank_delta = self._apply_source_gate(lowrank_delta, source_center.clamp(0.0, 1.0))
        self._last_calibration_extra_delta = lowrank_delta
        pred = (calibrated + lowrank_delta).clamp(0.0, 1.0)

        if getattr(self, "_last_guard_features", None) is not None:
            coeff_abs_mean = coeff.abs().mean(dim=1, keepdim=True).clamp(0.0, 1.0)
            self._last_guard_features = torch.cat([self._last_guard_features, coeff_abs_mean], dim=1)
        field_stack = torch.cat([field, lowrank_delta], dim=1)
        return pred, y_knots, field_stack, anchor, anchor_delta + lowrank_delta

    def _apply_target_stat_anchor_fine_residual_head(
        self,
        raw_pred: torch.Tensor,
        source_center: torch.Tensor,
        feature_map: torch.Tensor | None,
        features: torch.Tensor,
        contrast_id: torch.Tensor,
        target_field_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        calibrated, y_knots, field, anchor, anchor_delta = self._apply_target_stat_anchor_detail_head(
            raw_pred,
            source_center,
            feature_map,
            features,
            contrast_id,
            target_field_id,
        )
        if feature_map is None:
            raise RuntimeError("target_stat_anchor_fine_residual_v1 requires decoder feature_map")

        fine_field = torch.tanh(self.fine_residual_field_head(feature_map))
        fine_field = self._smooth(fine_field, self.calibration_lowfreq_kernel_size)
        coeff = torch.tanh(self.fine_residual_coeff_mlp(features))[:, :, None, None]
        fine_delta = self.calibration_lowrank_delta_max * coeff * fine_field
        fine_delta = fine_delta.clamp(-self.calibration_lowrank_delta_max, self.calibration_lowrank_delta_max)
        fine_delta = self._apply_source_gate(fine_delta, source_center.clamp(0.0, 1.0))
        self._last_calibration_extra_delta = fine_delta
        pred = (calibrated + fine_delta).clamp(0.0, 1.0)

        if getattr(self, "_last_guard_features", None) is not None:
            self._last_guard_features = torch.cat(
                [self._last_guard_features, coeff.flatten(1).abs().clamp(0.0, 1.0)],
                dim=1,
            )
        field_stack = torch.cat([field, fine_field, fine_delta], dim=1)
        return pred, y_knots, field_stack, anchor, anchor_delta + fine_delta

    def _apply_target_stat_anchor_swin_residual_head(
        self,
        raw_pred: torch.Tensor,
        source_center: torch.Tensor,
        feature_map: torch.Tensor | None,
        features: torch.Tensor,
        contrast_id: torch.Tensor,
        target_field_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        calibrated, y_knots, field, anchor, anchor_delta = self._apply_target_stat_anchor_detail_head(
            raw_pred,
            source_center,
            feature_map,
            features,
            contrast_id,
            target_field_id,
        )
        if feature_map is None:
            raise RuntimeError("target_stat_anchor_swin_residual_v1 requires decoder feature_map")

        adapted = self.swin_residual_adapter(feature_map)
        residual_field = torch.tanh(self.swin_residual_field_head(adapted))
        coeff = torch.tanh(self.swin_residual_coeff_mlp(features))[:, :, None, None]
        residual_delta = self.calibration_lowrank_delta_max * coeff * residual_field
        residual_delta = residual_delta.clamp(
            -self.calibration_lowrank_delta_max,
            self.calibration_lowrank_delta_max,
        )
        residual_delta = self._apply_source_gate(residual_delta, source_center.clamp(0.0, 1.0))
        self._last_calibration_extra_delta = residual_delta
        pred = (calibrated + residual_delta).clamp(0.0, 1.0)

        if getattr(self, "_last_guard_features", None) is not None:
            self._last_guard_features = torch.cat(
                [self._last_guard_features, coeff.flatten(1).abs().clamp(0.0, 1.0)],
                dim=1,
            )
        field_stack = torch.cat([field, residual_field, residual_delta], dim=1)
        return pred, y_knots, field_stack, anchor, anchor_delta + residual_delta

    def _predict_policy_features(self, features: torch.Tensor) -> torch.Tensor | None:
        if self.policy_feature_dim <= 0:
            return None
        return self.policy_feature_mlp(features)

    def _predict_method_bank_logits(self, features: torch.Tensor) -> torch.Tensor | None:
        if self.method_bank_size <= 0:
            return None
        return self.method_bank_mlp(features)

    def _method_bank_features(
        self,
        features: torch.Tensor,
        source_field_id: torch.Tensor,
        target_field_id: torch.Tensor,
    ) -> torch.Tensor:
        if self.method_bank_feature_mode == "default":
            return features
        values = torch.tensor([FIELD_VALUES[name] for name in FIELDS], device=features.device, dtype=features.dtype)
        src_b0 = values[source_field_id].clamp_min(1e-6)
        tgt_b0 = values[target_field_id].clamp_min(1e-6)
        ratio = tgt_b0 / src_b0
        delta = tgt_b0 - src_b0
        extra = torch.stack(
            [
                src_b0 / 7.0,
                tgt_b0 / 7.0,
                torch.log(ratio).clamp(-6.0, 6.0) / 6.0,
                delta / 7.0,
                (tgt_b0 >= 7.0).to(dtype=features.dtype),
                (tgt_b0 >= 5.0).to(dtype=features.dtype),
                ((tgt_b0 >= 7.0) & (src_b0 < 7.0)).to(dtype=features.dtype),
                ((tgt_b0 >= 7.0) & (src_b0 >= 1.5)).to(dtype=features.dtype),
            ],
            dim=1,
        )
        return torch.cat([features, extra], dim=1)

    @staticmethod
    def _policy_feature(
        policy_features: torch.Tensor,
        name: str,
        min_value: float = 0.0,
        max_value: float = 1.0,
    ) -> torch.Tensor:
        feature_idx = METHOD_FEATURE_TO_INDEX[name]
        if feature_idx >= policy_features.shape[1]:
            return policy_features.new_zeros((policy_features.shape[0], 1))
        value = policy_features[:, feature_idx : feature_idx + 1]
        return value.clamp(float(min_value), float(max_value))

    def _apply_policy_param_lut_lowfreq_head(
        self,
        raw_pred: torch.Tensor,
        source_center: torch.Tensor,
        features: torch.Tensor,
        policy_features: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if policy_features is None:
            raise RuntimeError("policy_param_lut_lowfreq requires predicted policy_features")
        source = source_center.clamp(0.0, 1.0)
        anchor = raw_pred.clamp(0.0, 1.0)
        source_lut = self._apply_source_lut_primitive(source, features).clamp(0.0, 1.0)

        beta = self._policy_feature(policy_features, "beta")[:, :, None, None]
        lowfreq_alpha = self._policy_feature(policy_features, "lowfreq_alpha")[:, :, None, None]
        source_texture = self._policy_feature(policy_features, "source_texture")[:, :, None, None]
        texture_blur = self._policy_feature(policy_features, "texture_blur_sigma_xy", 0.0, 6.0)[:, :, None, None]
        median_kernel = self._policy_feature(policy_features, "texture_median_kernel_xy", 0.0, 7.0)[:, :, None, None]
        texture_unsharp = self._policy_feature(policy_features, "texture_unsharp_amount")[:, :, None, None]
        blur_strength = (texture_blur / 6.0).clamp(0.0, 1.0)
        median_strength = (median_kernel / 3.0).clamp(0.0, 1.0)
        texture_strength = torch.maximum(source_texture, texture_unsharp)

        target_like = ((1.0 - beta) * source + beta * source_lut).clamp(0.0, 1.0)
        source_low = self._smooth(source)
        target_low = self._smooth(target_like)
        lowfreq = (anchor + lowfreq_alpha * (target_low - source_low)).clamp(0.0, 1.0)
        texture_on_target = (target_low + source - source_low).clamp(0.0, 1.0)
        textured = (lowfreq + texture_strength * (texture_on_target - lowfreq)).clamp(0.0, 1.0)
        smoothed = target_low
        full_calibrated = (textured + blur_strength * (smoothed - textured)).clamp(0.0, 1.0)
        medianed = self._median_smooth(full_calibrated, kernel_size=3).clamp(0.0, 1.0)
        full_calibrated = (full_calibrated + median_strength * (medianed - full_calibrated)).clamp(0.0, 1.0)
        blend = min(max(self.calibration_blend_max, 0.0), 1.0)
        blended_lowfreq = (anchor + blend * (lowfreq - anchor)).clamp(0.0, 1.0)
        calibrated = (anchor + blend * (full_calibrated - anchor)).clamp(0.0, 1.0)
        return calibrated, blended_lowfreq, calibrated - blended_lowfreq

    def _apply_direct_param_source_safe_head(
        self,
        raw_pred: torch.Tensor,
        source_center: torch.Tensor,
        features: torch.Tensor,
        policy_features: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if policy_features is None:
            raise RuntimeError("direct_param_source_safe_v1 requires predicted policy_features")
        source = source_center.clamp(0.0, 1.0)
        anchor = self._apply_affine_head(raw_pred, features).clamp(0.0, 1.0)
        source_lut = self._apply_source_lut_primitive(source, features).clamp(0.0, 1.0)

        beta = self._policy_feature(policy_features, "beta")[:, :, None, None]
        lowfreq_alpha = self._policy_feature(policy_features, "lowfreq_alpha")[:, :, None, None]
        blur_sigma = self._policy_feature(policy_features, "texture_blur_sigma_xy", 0.0, 6.0)[:, :, None, None]
        median_kernel = self._policy_feature(policy_features, "texture_median_kernel_xy", 0.0, 7.0)[:, :, None, None]
        unsharp_amount = self._policy_feature(policy_features, "texture_unsharp_amount")[:, :, None, None]
        source_texture = self._policy_feature(policy_features, "source_texture")[:, :, None, None]

        source_low = self._smooth(source, self.calibration_lowfreq_kernel_size)
        target_like = ((1.0 - beta) * source + beta * source_lut).clamp(0.0, 1.0)
        target_low = self._smooth(target_like, self.calibration_lowfreq_kernel_size)

        lowfreq_delta = lowfreq_alpha * (target_low - source_low)
        lowfreq_delta = self._apply_source_gate(lowfreq_delta, source)
        lowfreq_blend = min(max(self.calibration_blend_max, 0.0), 1.0)
        calibrated = (anchor + lowfreq_blend * lowfreq_delta).clamp(0.0, 1.0)

        texture_base = (target_low + source - source_low).clamp(0.0, 1.0)
        texture_strength = torch.maximum(source_texture, unsharp_amount).clamp(0.0, 1.0)
        calibrated = (
            calibrated
            + self.calibration_texture_blend_max * texture_strength * (texture_base - calibrated)
        ).clamp(0.0, 1.0)

        blur_strength = (blur_sigma / 6.0).clamp(0.0, 1.0)
        blurred = self._smooth(calibrated, kernel_size=3)
        calibrated = (
            calibrated
            + self.calibration_texture_blend_max * blur_strength * (blurred - calibrated)
        ).clamp(0.0, 1.0)

        median_strength = (median_kernel / 5.0).clamp(0.0, 1.0)
        medianed = self._median_smooth(calibrated, kernel_size=3)
        calibrated = (
            calibrated
            + self.calibration_texture_blend_max * median_strength * (medianed - calibrated)
        ).clamp(0.0, 1.0)
        return calibrated, anchor, calibrated - anchor

    def _apply_direct_param_target_percentile_head(
        self,
        raw_pred: torch.Tensor,
        source_center: torch.Tensor,
        features: torch.Tensor,
        policy_features: torch.Tensor | None,
        contrast_id: torch.Tensor,
        target_field_id: torch.Tensor,
        source_field_id: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._last_guard_features = None
        if policy_features is None:
            raise RuntimeError(f"{self.calibration_head} requires predicted policy_features")
        if not hasattr(self, "target_percentile_stats"):
            raise RuntimeError(f"{self.calibration_head} requires target_percentile_stats buffer")
        target_knots = self.target_percentile_stats.to(device=raw_pred.device, dtype=raw_pred.dtype)[
            contrast_id, target_field_id
        ]
        if not torch.isfinite(target_knots).all():
            raise RuntimeError("Missing target percentile stats for requested contrast/target field")
        source = source_center.clamp(0.0, 1.0)
        anchor = self._apply_affine_head(raw_pred, features).clamp(0.0, 1.0)
        source_knots = self._masked_percentile_source_stats(
            source,
            self.source_stat_percentile_levels,
        )
        target_like = self._interpolate_piecewise(source, source_knots, target_knots).clamp(0.0, 1.0)

        family_quantile = self._policy_feature(policy_features, "family_quantile")[:, :, None, None]
        family_targetcohort = self._policy_feature(policy_features, "family_targetcohort")[:, :, None, None]
        family_source_copy = self._policy_feature(policy_features, "family_source_copy")[:, :, None, None]
        beta = self._policy_feature(policy_features, "beta")[:, :, None, None]
        lowfreq_alpha = self._policy_feature(policy_features, "lowfreq_alpha")[:, :, None, None]
        blur_sigma = self._policy_feature(policy_features, "texture_blur_sigma_xy", 0.0, 6.0)[:, :, None, None]
        median_kernel = self._policy_feature(policy_features, "texture_median_kernel_xy", 0.0, 7.0)[:, :, None, None]
        unsharp_amount = self._policy_feature(policy_features, "texture_unsharp_amount")[:, :, None, None]
        source_texture = self._policy_feature(policy_features, "source_texture")[:, :, None, None]

        percentile_strength = torch.maximum(family_quantile, family_targetcohort).clamp(0.0, 1.0)
        map_beta = (beta * percentile_strength).clamp(0.0, 1.0)
        source_floor = map_beta.new_zeros(map_beta.shape)
        if self.calibration_head == "direct_param_target_percentile_guarded_v2":
            guard_logits = self.percentile_guard_mlp(features)
            guard = torch.exp(0.5 * torch.tanh(guard_logits))[:, :, None, None]
            map_beta = (map_beta * guard[:, 0:1]).clamp(0.0, 1.0)
            lowfreq_alpha = (lowfreq_alpha * guard[:, 1:2]).clamp(0.0, 1.0)
        elif self.calibration_head == "direct_param_target_percentile_guarded_v3":
            guard_logits = self.percentile_guard_mlp(features)
            guard = torch.exp(0.5 * torch.tanh(guard_logits[:, 0:2]))[:, :, None, None]
            regime_boost = map_beta.new_ones(map_beta.shape)
            if source_field_id is not None:
                field_values = torch.tensor(
                    [FIELD_VALUES[name] for name in FIELDS],
                    device=source_field_id.device,
                    dtype=map_beta.dtype,
                )
                src_b0 = field_values[source_field_id].to(device=map_beta.device)
                tgt_b0 = field_values[target_field_id].to(device=map_beta.device)
                downfield = (src_b0 > tgt_b0).to(dtype=map_beta.dtype)
                target_low = (tgt_b0 <= 0.11).to(dtype=map_beta.dtype)
                source_high = (src_b0 >= 3.0).to(dtype=map_beta.dtype)
                downfield_ultra = (downfield * target_low * source_high)[:, None, None, None]
                regime_boost = 1.0 + self.calibration_guard_v3_downfield_boost * downfield_ultra
                blur_sigma = (
                    blur_sigma + self.calibration_guard_v3_blur_boost * downfield_ultra * percentile_strength
                ).clamp(0.0, 6.0)
            map_beta = (map_beta * guard[:, 0:1] * regime_boost).clamp(0.0, 1.0)
            lowfreq_alpha = (lowfreq_alpha * guard[:, 1:2] * regime_boost).clamp(0.0, 1.0)
            learned_source_floor = torch.sigmoid(guard_logits[:, 2:3] - 6.0)[:, :, None, None]
            source_floor = (family_source_copy * learned_source_floor).clamp(0.0, 1.0)
        if self.calibration_head in (
            "direct_param_target_percentile_guarded_v2",
            "direct_param_target_percentile_guarded_v3",
        ):
            self._last_guard_features = torch.cat(
                [
                    map_beta.flatten(1),
                    lowfreq_alpha.clamp(0.0, 1.0).flatten(1),
                    (blur_sigma / 6.0).clamp(0.0, 1.0).flatten(1),
                    source_floor.flatten(1),
                ],
                dim=1,
            )
        target_like = ((1.0 - map_beta) * source + map_beta * target_like).clamp(0.0, 1.0)

        source_low = self._smooth(source, self.calibration_lowfreq_kernel_size)
        target_low = self._smooth(target_like, self.calibration_lowfreq_kernel_size)
        lowfreq_delta = lowfreq_alpha * (target_low - source_low)
        lowfreq_delta = self._apply_source_gate(lowfreq_delta, source)
        lowfreq_blend = min(max(self.calibration_blend_max, 0.0), 1.0)
        calibrated = (anchor + lowfreq_blend * lowfreq_delta).clamp(0.0, 1.0)

        texture_base = (target_low + source - source_low).clamp(0.0, 1.0)
        texture_strength = torch.maximum(source_texture, unsharp_amount).clamp(0.0, 1.0)
        calibrated = (
            calibrated
            + self.calibration_texture_blend_max * texture_strength * (texture_base - calibrated)
        ).clamp(0.0, 1.0)

        blur_strength = (blur_sigma / 6.0).clamp(0.0, 1.0)
        blurred = self._smooth(calibrated, kernel_size=3)
        calibrated = (
            calibrated
            + self.calibration_texture_blend_max * blur_strength * (blurred - calibrated)
        ).clamp(0.0, 1.0)

        median_strength = (median_kernel / 5.0).clamp(0.0, 1.0)
        medianed = self._median_smooth(calibrated, kernel_size=3)
        calibrated = (
            calibrated
            + self.calibration_texture_blend_max * median_strength * (medianed - calibrated)
        ).clamp(0.0, 1.0)
        if self.calibration_head == "direct_param_target_percentile_guarded_v3":
            calibrated = (calibrated + source_floor * (source - calibrated)).clamp(0.0, 1.0)
        return calibrated, anchor, calibrated - anchor

    def _apply_direct_control_target_percentile_head(
        self,
        raw_pred: torch.Tensor,
        source_center: torch.Tensor,
        base_center: torch.Tensor,
        features: torch.Tensor,
        contrast_id: torch.Tensor,
        target_field_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._last_guard_features = None
        if not hasattr(self, "target_percentile_stats"):
            raise RuntimeError("direct_control_target_percentile_v1 requires target_percentile_stats buffer")
        target_knots = self.target_percentile_stats.to(device=raw_pred.device, dtype=raw_pred.dtype)[
            contrast_id, target_field_id
        ]
        if not torch.isfinite(target_knots).all():
            raise RuntimeError("Missing target percentile stats for requested contrast/target field")

        source = source_center.clamp(0.0, 1.0)
        anchor = self._apply_affine_head(raw_pred, features).clamp(0.0, 1.0)
        source_knots = self._masked_percentile_source_stats(
            source,
            self.source_stat_percentile_levels,
        )
        target_like = self._interpolate_piecewise(source, source_knots, target_knots).clamp(0.0, 1.0)

        control_logits = self.percentile_control_mlp(features)
        map_beta = torch.sigmoid(control_logits[:, 0:1])[:, :, None, None]
        lowfreq_alpha = torch.sigmoid(control_logits[:, 1:2])[:, :, None, None]
        blur_strength = torch.sigmoid(control_logits[:, 2:3])[:, :, None, None]
        source_floor = torch.sigmoid(control_logits[:, 3:4])[:, :, None, None]
        self._last_guard_features = torch.cat(
            [
                map_beta.flatten(1),
                lowfreq_alpha.flatten(1),
                blur_strength.flatten(1),
                source_floor.flatten(1),
            ],
            dim=1,
        )

        target_like = ((1.0 - map_beta) * source + map_beta * target_like).clamp(0.0, 1.0)
        source_low = self._smooth(source, self.calibration_lowfreq_kernel_size)
        target_low = self._smooth(target_like, self.calibration_lowfreq_kernel_size)
        lowfreq_delta = lowfreq_alpha * (target_low - source_low)
        lowfreq_delta = self._apply_source_gate(lowfreq_delta, source)
        lowfreq_blend = min(max(self.calibration_blend_max, 0.0), 1.0)
        calibrated = (anchor + lowfreq_blend * lowfreq_delta).clamp(0.0, 1.0)

        blurred = self._smooth(calibrated, kernel_size=3)
        calibrated = (
            calibrated
            + self.calibration_texture_blend_max * blur_strength * (blurred - calibrated)
        ).clamp(0.0, 1.0)
        calibrated = (calibrated + source_floor * (source - calibrated)).clamp(0.0, 1.0)
        return calibrated, anchor, calibrated - anchor

    def _apply_direct_control_gain_bias_field_head(
        self,
        raw_pred: torch.Tensor,
        source_center: torch.Tensor,
        feature_map: torch.Tensor | None,
        features: torch.Tensor,
        contrast_id: torch.Tensor,
        target_field_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._last_guard_features = None
        if feature_map is None:
            raise RuntimeError("direct_control_gain_bias_field_v1 calibration requires decoder feature_map")
        if not hasattr(self, "target_percentile_stats"):
            raise RuntimeError("direct_control_gain_bias_field_v1 requires target_percentile_stats buffer")
        target_knots = self.target_percentile_stats.to(device=raw_pred.device, dtype=raw_pred.dtype)[
            contrast_id, target_field_id
        ]
        if not torch.isfinite(target_knots).all():
            raise RuntimeError("Missing target percentile stats for requested contrast/target field")

        source = source_center.clamp(0.0, 1.0)
        base_anchor = self._apply_affine_head(raw_pred, features).clamp(0.0, 1.0)
        source_knots = self._masked_percentile_source_stats(
            source,
            self.source_stat_percentile_levels,
        )
        y_knots = torch.cummax(target_knots.clamp(0.0, 1.0), dim=1).values
        target_like = self._interpolate_piecewise(source, source_knots, y_knots).clamp(0.0, 1.0)

        control_logits = self.percentile_control_mlp(features)
        controls = torch.sigmoid(control_logits).clamp(0.0, 1.0)
        map_beta = controls[:, 0:1]
        lowfreq_alpha = controls[:, 1:2]
        blur_strength = controls[:, 2:3]
        source_floor = controls[:, 3:4]
        self._last_guard_features = controls

        map_beta_4d = map_beta[:, :, None, None]
        lowfreq_alpha_4d = lowfreq_alpha[:, :, None, None]
        blur_strength_4d = blur_strength[:, :, None, None]
        source_floor_4d = source_floor[:, :, None, None]

        target_like = ((1.0 - map_beta_4d) * source + map_beta_4d * target_like).clamp(0.0, 1.0)
        source_low = self._smooth(source, self.calibration_lowfreq_kernel_size)
        target_low = self._smooth(target_like, self.calibration_lowfreq_kernel_size)
        source_detail = source - source_low
        controlled_low = (
            source_low
            + self.calibration_blend_max * lowfreq_alpha_4d * (target_low - source_low)
        )
        detail_keep = (1.0 - self.calibration_texture_blend_max * blur_strength_4d).clamp(0.0, 1.0)
        stat_anchor = (controlled_low + detail_keep * source_detail).clamp(0.0, 1.0)
        anchor = ((1.0 - map_beta_4d) * base_anchor + map_beta_4d * stat_anchor).clamp(0.0, 1.0)
        anchor = (source_floor_4d * source + (1.0 - source_floor_4d) * anchor).clamp(0.0, 1.0)

        gain_field = torch.tanh(self.local_lowfreq_gain_head(feature_map))
        gain_field = self._smooth(gain_field, self.calibration_lowfreq_kernel_size)
        bias_field = torch.tanh(self.local_lowfreq_bias_head(feature_map))
        bias_field = self._smooth(bias_field, self.calibration_lowfreq_kernel_size)
        field = torch.tanh(self.local_lowfreq_field_head(feature_map))
        field = self._smooth(field, self.calibration_lowfreq_kernel_size)

        local_strength = (1.0 - source_floor_4d) * lowfreq_alpha_4d
        # Keep inherited local fields subordinate to the explicit source-preserving control anchor.
        gain = torch.exp(0.25 * self.calibration_gain_scale_max * local_strength * gain_field).clamp(0.75, 1.35)
        bias = 0.25 * self.calibration_bias_scale_max * map_beta_4d * bias_field
        bias = self._apply_source_gate(bias, source)
        residual_delta = 0.25 * self.calibration_texture_blend_max * local_strength * field
        residual_delta = self._apply_source_gate(residual_delta, source)

        calibrated = (anchor * gain + bias + residual_delta).clamp(0.0, 1.0)
        field_stack = torch.cat([gain_field, bias_field, field], dim=1)
        return calibrated, y_knots, field_stack, anchor, calibrated - anchor

    def _apply_target_percentile_relation_head(
        self,
        raw_pred: torch.Tensor,
        source_center: torch.Tensor,
        base_center: torch.Tensor,
        features: torch.Tensor,
        contrast_id: torch.Tensor,
        target_field_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._last_guard_features = None
        if not hasattr(self, "target_percentile_stats"):
            raise RuntimeError(f"{self.calibration_head} requires target_percentile_stats buffer")
        target_knots = self.target_percentile_stats.to(device=raw_pred.device, dtype=raw_pred.dtype)[
            contrast_id, target_field_id
        ]
        if not torch.isfinite(target_knots).all():
            raise RuntimeError("Missing target percentile stats for requested contrast/target field")

        source = source_center.clamp(0.0, 1.0)
        if self.calibration_head == "source_anchor_percentile_relation_v1":
            anchor = source
        elif self.calibration_head == "base_anchor_percentile_relation_v1":
            anchor = base_center.clamp(0.0, 1.0)
        else:
            anchor = self._apply_affine_head(raw_pred, features).clamp(0.0, 1.0)
        source_knots = self._masked_percentile_source_stats(
            source,
            self.source_stat_percentile_levels,
        )
        controls = self.percentile_relation_mlp(features)
        knot_delta_raw = controls[:, : len(DEFAULT_PERCENTILES)]
        map_raw, lowfreq_raw, source_floor_raw = controls[:, -3:].chunk(3, dim=1)

        map_beta = self.calibration_blend_max * torch.sigmoid(map_raw)
        lowfreq_alpha = self.calibration_texture_blend_max * torch.sigmoid(lowfreq_raw)
        source_floor = torch.sigmoid(source_floor_raw)

        base_knots = source_knots + map_beta * (target_knots - source_knots)
        knot_delta = self.calibration_knot_delta_max * torch.tanh(knot_delta_raw)
        y_knots = (base_knots + knot_delta).clamp(0.0, 1.0)
        y_knots = torch.cummax(y_knots, dim=1).values
        mapped = self._interpolate_piecewise(source, source_knots, y_knots).clamp(0.0, 1.0)

        source_low = self._smooth(source, self.calibration_lowfreq_kernel_size)
        mapped_low = self._smooth(mapped, self.calibration_lowfreq_kernel_size)
        lowfreq_delta = lowfreq_alpha[:, :, None, None] * (mapped_low - source_low)
        lowfreq_delta = self._apply_source_gate(lowfreq_delta, source)
        calibrated = (anchor + lowfreq_delta).clamp(0.0, 1.0)
        calibrated = (calibrated + source_floor[:, :, None, None] * (source - calibrated)).clamp(0.0, 1.0)
        self._last_guard_features = torch.cat(
            [
                map_beta.clamp(0.0, 1.0),
                lowfreq_alpha.clamp(0.0, 1.0),
                knot_delta.abs().mean(dim=1, keepdim=True).clamp(0.0, 1.0),
                source_floor.clamp(0.0, 1.0),
            ],
            dim=1,
        )
        return calibrated, y_knots, anchor, calibrated - anchor

    def _apply_source_preserving_monotonic_lowfreq_head(
        self,
        raw_pred: torch.Tensor,
        source_center: torch.Tensor,
        features: torch.Tensor,
        contrast_id: torch.Tensor,
        target_field_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._last_guard_features = None
        if not hasattr(self, "target_percentile_stats"):
            raise RuntimeError("source_preserving_monotonic_lowfreq_v1 requires target_percentile_stats buffer")
        target_knots = self.target_percentile_stats.to(device=raw_pred.device, dtype=raw_pred.dtype)[
            contrast_id, target_field_id
        ]
        if not torch.isfinite(target_knots).all():
            raise RuntimeError("Missing target percentile stats for requested contrast/target field")

        source = source_center.clamp(0.0, 1.0)
        anchor = source
        source_knots = self._masked_percentile_source_stats(
            source,
            self.source_stat_percentile_levels,
        )
        y_knots = torch.cummax(target_knots.clamp(0.0, 1.0), dim=1).values
        mapped = self._interpolate_piecewise(source, source_knots, y_knots).clamp(0.0, 1.0)

        controls = self.source_preserving_control_mlp(features)
        map_beta = torch.sigmoid(controls[:, 0:1])[:, :, None, None]
        lowfreq_alpha = torch.sigmoid(controls[:, 1:2])[:, :, None, None]
        texture_blend = torch.sigmoid(controls[:, 2:3])[:, :, None, None]
        source_floor = torch.sigmoid(controls[:, 3:4])[:, :, None, None]

        target_like = ((1.0 - map_beta) * source + map_beta * mapped).clamp(0.0, 1.0)
        source_low = self._smooth(source, self.calibration_lowfreq_kernel_size)
        target_low = self._smooth(target_like, self.calibration_lowfreq_kernel_size)
        lowfreq_delta = lowfreq_alpha * (target_low - source_low)
        lowfreq_delta = self._apply_source_gate(lowfreq_delta, source)
        lowfreq_delta = min(max(self.calibration_blend_max, 0.0), 1.0) * lowfreq_delta
        lowfreq = (anchor + lowfreq_delta).clamp(0.0, 1.0)

        texture_candidate = (target_low + source - source_low).clamp(0.0, 1.0)
        calibrated = (
            lowfreq
            + min(max(self.calibration_texture_blend_max, 0.0), 1.0)
            * texture_blend
            * (texture_candidate - lowfreq)
        ).clamp(0.0, 1.0)
        calibrated = (source_floor * source + (1.0 - source_floor) * calibrated).clamp(0.0, 1.0)

        self._last_guard_features = torch.cat(
            [
                map_beta.flatten(1),
                lowfreq_alpha.flatten(1),
                texture_blend.flatten(1),
                source_floor.flatten(1),
            ],
            dim=1,
        )
        return calibrated, y_knots, anchor, calibrated - anchor

    def _apply_calibrated_anchor_detail_head(
        self,
        raw_pred: torch.Tensor,
        source_center: torch.Tensor,
        features: torch.Tensor,
        contrast_id: torch.Tensor,
        target_field_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._last_guard_features = None
        if not hasattr(self, "target_percentile_stats"):
            raise RuntimeError("calibrated_anchor_detail_v1 requires target_percentile_stats buffer")
        target_knots = self.target_percentile_stats.to(device=raw_pred.device, dtype=raw_pred.dtype)[
            contrast_id, target_field_id
        ]
        if not torch.isfinite(target_knots).all():
            raise RuntimeError("Missing target percentile stats for requested contrast/target field")

        source = source_center.clamp(0.0, 1.0)
        anchor = self._apply_affine_head(raw_pred, features).clamp(0.0, 1.0)
        source_knots = self._masked_percentile_source_stats(
            source,
            self.source_stat_percentile_levels,
        )
        y_knots = torch.cummax(target_knots.clamp(0.0, 1.0), dim=1).values
        mapped = self._interpolate_piecewise(source, source_knots, y_knots).clamp(0.0, 1.0)

        controls = self.calibrated_anchor_detail_mlp(features)
        map_beta = torch.sigmoid(controls[:, 0:1])[:, :, None, None]
        lowfreq_alpha = torch.sigmoid(controls[:, 1:2])[:, :, None, None]
        detail_alpha = torch.sigmoid(controls[:, 2:3])[:, :, None, None]
        anchor_floor = torch.sigmoid(controls[:, 3:4])[:, :, None, None]

        target_like = ((1.0 - map_beta) * anchor + map_beta * mapped).clamp(0.0, 1.0)
        anchor_low = self._smooth(anchor, self.calibration_lowfreq_kernel_size)
        target_low = self._smooth(target_like, self.calibration_lowfreq_kernel_size)
        lowfreq_delta = lowfreq_alpha * (target_low - anchor_low)
        lowfreq_delta = self._apply_source_gate(lowfreq_delta, source)
        lowfreq = (anchor + lowfreq_delta).clamp(0.0, 1.0)

        source_low = self._smooth(source, self.calibration_lowfreq_kernel_size)
        detail_candidate = (target_low + source - source_low).clamp(0.0, 1.0)
        calibrated = (
            lowfreq
            + min(max(self.calibration_texture_blend_max, 0.0), 1.0)
            * detail_alpha
            * (detail_candidate - lowfreq)
        ).clamp(0.0, 1.0)
        calibrated = (anchor_floor * anchor + (1.0 - anchor_floor) * calibrated).clamp(0.0, 1.0)

        self._last_guard_features = torch.cat(
            [
                map_beta.flatten(1),
                lowfreq_alpha.flatten(1),
                detail_alpha.flatten(1),
                anchor_floor.flatten(1),
            ],
            dim=1,
        )
        return calibrated, y_knots, anchor, calibrated - anchor

    def _apply_target_percentile_primitive_bank_head(
        self,
        raw_pred: torch.Tensor,
        source_center: torch.Tensor,
        base_center: torch.Tensor,
        features: torch.Tensor,
        contrast_id: torch.Tensor,
        target_field_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not hasattr(self, "target_percentile_stats"):
            raise RuntimeError("target_percentile_primitive_bank_v1 requires target_percentile_stats buffer")
        target_knots = self.target_percentile_stats.to(device=raw_pred.device, dtype=raw_pred.dtype)[
            contrast_id, target_field_id
        ]
        if not torch.isfinite(target_knots).all():
            raise RuntimeError("Missing target percentile stats for requested contrast/target field")

        raw = raw_pred.clamp(0.0, 1.0)
        source = source_center.clamp(0.0, 1.0)
        base = base_center.clamp(0.0, 1.0)
        anchor = self._apply_affine_head(raw_pred, features).clamp(0.0, 1.0)
        source_knots = self._masked_percentile_source_stats(
            source,
            self.source_stat_percentile_levels,
        )
        y_knots = torch.cummax(target_knots.clamp(0.0, 1.0), dim=1).values
        mapped = self._interpolate_piecewise(source, source_knots, y_knots).clamp(0.0, 1.0)

        source_low = self._smooth(source, self.calibration_lowfreq_kernel_size)
        mapped_low = self._smooth(mapped, self.calibration_lowfreq_kernel_size)
        anchor_low = self._smooth(anchor, self.calibration_lowfreq_kernel_size)
        lowfreq_delta = self._apply_source_gate(mapped_low - source_low, source)
        lowfreq_on_anchor = (anchor + lowfreq_delta).clamp(0.0, 1.0)
        source_texture_on_anchor = (anchor_low + source - source_low).clamp(0.0, 1.0)
        target_texture = (mapped_low + source - source_low).clamp(0.0, 1.0)
        mid_mapped = (0.5 * source + 0.5 * mapped).clamp(0.0, 1.0)

        primitives = torch.stack(
            [
                anchor,
                raw,
                base,
                source,
                mid_mapped,
                mapped,
                lowfreq_on_anchor,
                source_texture_on_anchor,
                target_texture,
            ],
            dim=1,
        )
        weights = torch.softmax(self.primitive_weight_mlp(features), dim=1)
        primitive_base = (primitives * weights[:, :, None, None, None]).sum(dim=1)
        blend = self.primitive_bank_blend_max * torch.sigmoid(self.primitive_blend_mlp(features))[:, :, None, None]
        calibrated = (anchor + blend * (primitive_base - anchor)).clamp(0.0, 1.0)
        return calibrated, y_knots, weights, primitive_base, anchor

    def _apply_source_lut_lowfreq_texture_head(
        self,
        raw_pred: torch.Tensor,
        source_center: torch.Tensor,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source = source_center.clamp(0.0, 1.0)
        anchor = self._apply_affine_head(raw_pred, features)
        source_lut = self._apply_source_lut_primitive(source, features).clamp(0.0, 1.0)
        source_low = self._smooth(source, self.calibration_lowfreq_kernel_size)
        target_low = self._smooth(source_lut, self.calibration_lowfreq_kernel_size)

        lowfreq_blend = self.calibration_blend_max * torch.sigmoid(self.calibration_blend_mlp(features))[
            :, :, None, None
        ]
        lowfreq_delta = self._apply_source_gate(lowfreq_blend * (target_low - source_low), source)
        lowfreq = (anchor + lowfreq_delta).clamp(0.0, 1.0)

        texture_candidate = (target_low + source - source_low).clamp(0.0, 1.0)
        texture_blend = self.calibration_texture_blend_max * torch.sigmoid(
            self.calibration_texture_blend_mlp(features)
        )[:, :, None, None]
        calibrated = (lowfreq + texture_blend * (texture_candidate - lowfreq)).clamp(0.0, 1.0)
        return calibrated, lowfreq, calibrated - lowfreq

    def _apply_source_stat_robust_head(
        self,
        raw_pred: torch.Tensor,
        source_center: torch.Tensor,
        stats: torch.Tensor,
        contrast_id: torch.Tensor,
        target_field_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not hasattr(self, "target_robust_stats"):
            raise RuntimeError("source_stat_robust requires target_robust_stats buffer")
        target_stats = self.target_robust_stats.to(device=raw_pred.device, dtype=raw_pred.dtype)[
            contrast_id, target_field_id
        ]
        if not torch.isfinite(target_stats).all():
            raise RuntimeError("Missing target robust stats for requested contrast/target field")
        source_robust = self._masked_robust_source_stats(source_center)
        source_p5 = source_robust[:, 0:1, None, None]
        source_p50 = source_robust[:, 1:2, None, None]
        source_p95 = source_robust[:, 2:3, None, None]
        target_p5 = target_stats[:, 0:1, None, None]
        target_p50 = target_stats[:, 1:2, None, None]
        target_p95 = target_stats[:, 2:3, None, None]
        source_iqr = (source_p95 - source_p5).clamp_min(1e-6)
        target_iqr = (target_p95 - target_p5).clamp_min(1e-6)
        robust = ((source_center - source_p50) / source_iqr * target_iqr + target_p50).clamp(0.0, 1.0)
        blend_value = min(max(self.calibration_blend_max, 0.0), 1.0)
        blend = raw_pred.new_full((raw_pred.shape[0], 1, 1, 1), blend_value)
        if self.calibration_guard_mode == "field_extreme_v1":
            target_is_7t = target_field_id == (len(FIELDS) - 1)
            guard = 1.0 + target_is_7t.to(dtype=blend.dtype)[:, None, None, None] * (
                self.calibration_guard_extreme_scale - 1.0
            )
            blend = blend * guard
        delta = self._apply_source_gate(blend * (robust - raw_pred), source_center)
        calibrated = (raw_pred + delta).clamp(0.0, 1.0)
        return calibrated, robust, delta

    def _apply_source_stat_percentile_head(
        self,
        raw_pred: torch.Tensor,
        source_center: torch.Tensor,
        features: torch.Tensor,
        contrast_id: torch.Tensor,
        target_field_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not hasattr(self, "target_percentile_stats"):
            raise RuntimeError("source_stat_percentile requires target_percentile_stats buffer")
        target_knots = self.target_percentile_stats.to(device=raw_pred.device, dtype=raw_pred.dtype)[
            contrast_id, target_field_id
        ]
        if not torch.isfinite(target_knots).all():
            raise RuntimeError("Missing target percentile stats for requested contrast/target field")
        source_knots = self._masked_percentile_source_stats(
            source_center,
            self.source_stat_percentile_levels,
        )
        mapped = self._interpolate_piecewise(source_center, source_knots, target_knots)
        blend = self.calibration_blend_max * torch.sigmoid(self.calibration_blend_mlp(features))[:, :, None, None]
        if self.calibration_guard_mode == "field_extreme_v1":
            target_is_7t = target_field_id == (len(FIELDS) - 1)
            guard = 1.0 + target_is_7t.to(dtype=blend.dtype)[:, None, None, None] * (
                self.calibration_guard_extreme_scale - 1.0
            )
            blend = blend * guard
        delta = self._apply_source_gate(blend * (mapped - raw_pred), source_center)
        calibrated = (raw_pred + delta).clamp(0.0, 1.0)
        return calibrated, mapped, delta, target_knots

    def _primitive_condition_features(
        self,
        features: torch.Tensor,
        policy_features: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.primitive_bank_use_policy_features:
            return features
        if policy_features is None:
            raise RuntimeError("primitive_bank_use_policy_features requires predicted policy_features")
        return torch.cat([features, policy_features], dim=1)

    def _build_primitive_bank(
        self,
        raw_pred: torch.Tensor,
        source_center: torch.Tensor,
        base_center: torch.Tensor,
        affine_pred: torch.Tensor | None = None,
        policy_features: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raw = torch.clamp(raw_pred, 0.0, 1.0)
        source = torch.clamp(source_center, 0.0, 1.0)
        base = torch.clamp(base_center, 0.0, 1.0)
        if self.primitive_bank_mode == "basic":
            return torch.stack(
                [
                    raw,
                    source,
                    self._smooth(raw),
                    self._smooth(source),
                ],
                dim=1,
            )
        base_low = self._smooth(base)
        source_texture_on_base = torch.clamp(base_low + source - self._smooth(source), 0.0, 1.0)
        if self.primitive_bank_mode == "affine_source_safe_v1":
            if affine_pred is None:
                raise RuntimeError("affine_source_safe_v1 requires affine_pred")
            affine = torch.clamp(affine_pred, 0.0, 1.0)
            return torch.stack(
                [
                    affine,
                    raw,
                    base,
                    source,
                    base_low,
                    source_texture_on_base,
                ],
                dim=1,
            )
        if self.primitive_bank_mode == "policy_affine_source_safe_v1":
            if affine_pred is None:
                raise RuntimeError("policy_affine_source_safe_v1 requires affine_pred")
            if policy_features is None:
                raise RuntimeError("policy_affine_source_safe_v1 requires policy_features")
            affine = torch.clamp(affine_pred, 0.0, 1.0)
            beta = policy_features[:, METHOD_FEATURE_TO_INDEX["beta"] : METHOD_FEATURE_TO_INDEX["beta"] + 1].clamp(
                0.0, 1.0
            )
            beta = beta[:, :, None, None]
            beta_affine_source = torch.clamp((1.0 - beta) * source + beta * affine, 0.0, 1.0)
            return torch.stack(
                [
                    affine,
                    raw,
                    base,
                    source,
                    beta_affine_source,
                    base_low,
                    source_texture_on_base,
                ],
                dim=1,
            )
        if self.primitive_bank_mode == "policy_lowfreq_source_safe_v1":
            if affine_pred is None:
                raise RuntimeError("policy_lowfreq_source_safe_v1 requires affine_pred")
            if policy_features is None:
                raise RuntimeError("policy_lowfreq_source_safe_v1 requires policy_features")
            affine = torch.clamp(affine_pred, 0.0, 1.0)
            beta = policy_features[:, METHOD_FEATURE_TO_INDEX["beta"] : METHOD_FEATURE_TO_INDEX["beta"] + 1].clamp(
                0.0, 1.0
            )
            beta = beta[:, :, None, None]
            lowfreq_alpha = policy_features[
                :, METHOD_FEATURE_TO_INDEX["lowfreq_alpha"] : METHOD_FEATURE_TO_INDEX["lowfreq_alpha"] + 1
            ].clamp(0.0, 1.0)
            lowfreq_alpha = lowfreq_alpha[:, :, None, None]
            affine_low = self._smooth(affine)
            source_low = self._smooth(source)
            beta_affine_source = torch.clamp((1.0 - beta) * source + beta * affine, 0.0, 1.0)
            policy_lowfreq_base = torch.clamp(base + lowfreq_alpha * (affine_low - base_low), 0.0, 1.0)
            policy_lowfreq_source_texture = torch.clamp(
                source + lowfreq_alpha * (affine_low - source_low),
                0.0,
                1.0,
            )
            return torch.stack(
                [
                    affine,
                    raw,
                    base,
                    source,
                    beta_affine_source,
                    policy_lowfreq_base,
                    policy_lowfreq_source_texture,
                    source_texture_on_base,
                ],
                dim=1,
            )
        if self.primitive_bank_mode in (
            "policy_lut_lowfreq_source_safe_v1",
            "policy_lut_lowfreq_median_source_safe_v1",
        ):
            if affine_pred is None:
                raise RuntimeError("policy_lut_lowfreq_source_safe_v1 requires affine_pred")
            if policy_features is None:
                raise RuntimeError("policy_lut_lowfreq_source_safe_v1 requires policy_features")
            if features is None:
                raise RuntimeError("policy_lut_lowfreq_source_safe_v1 requires condition features")
            affine = torch.clamp(affine_pred, 0.0, 1.0)
            beta = policy_features[:, METHOD_FEATURE_TO_INDEX["beta"] : METHOD_FEATURE_TO_INDEX["beta"] + 1].clamp(
                0.0, 1.0
            )
            beta = beta[:, :, None, None]
            lowfreq_alpha = policy_features[
                :, METHOD_FEATURE_TO_INDEX["lowfreq_alpha"] : METHOD_FEATURE_TO_INDEX["lowfreq_alpha"] + 1
            ].clamp(0.0, 1.0)
            lowfreq_alpha = lowfreq_alpha[:, :, None, None]
            source_lut = torch.clamp(self._apply_source_lut_primitive(source, features), 0.0, 1.0)
            source_low = self._smooth(source)
            source_lut_low = self._smooth(source_lut)
            beta_affine_source = torch.clamp((1.0 - beta) * source + beta * affine, 0.0, 1.0)
            policy_lowfreq_source_lut = torch.clamp(
                source + lowfreq_alpha * (source_lut_low - source_low),
                0.0,
                1.0,
            )
            source_texture_on_lut = torch.clamp(source_lut_low + source - source_low, 0.0, 1.0)
            candidates = [
                affine,
                raw,
                base,
                source,
                beta_affine_source,
                source_lut,
                policy_lowfreq_source_lut,
                source_texture_on_base,
                source_texture_on_lut,
            ]
            if self.primitive_bank_mode == "policy_lut_lowfreq_median_source_safe_v1":
                candidates.append(self._median_smooth(source, kernel_size=3).clamp(0.0, 1.0))
            return torch.stack(candidates, dim=1)
        return torch.stack(
            [
                raw,
                base,
                source,
                base_low,
                source_texture_on_base,
            ],
            dim=1,
        )

    def _apply_calibration(
        self,
        raw_pred: torch.Tensor,
        cond: torch.Tensor,
        stats: torch.Tensor,
        source_center: torch.Tensor,
        base_center: torch.Tensor,
        feature_map: torch.Tensor | None,
        source_field_id: torch.Tensor,
        target_field_id: torch.Tensor,
        contrast_id: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        features = torch.cat([cond, stats], dim=1)
        policy_features = self._predict_policy_features(features)
        if self.calibration_head in ("none", "identity"):
            return torch.clamp(raw_pred, 0.0, 1.0), None, None, None, policy_features, None, None, None
        if self.calibration_head == "affine":
            return self._apply_affine_head(raw_pred, features), None, None, None, policy_features, None, None, None
        if self.calibration_head == "affine_gamma":
            return self._apply_affine_gamma_head(raw_pred, features), None, None, None, policy_features, None, None, None
        if self.calibration_head == "affine_lowfreq_bias":
            if feature_map is None:
                raise RuntimeError("affine_lowfreq_bias calibration requires decoder feature_map")
            affine_pred = self._apply_affine_head(raw_pred, features)
            bias = torch.tanh(self.calibration_bias_head(feature_map))
            bias = self._smooth(bias, self.calibration_lowfreq_kernel_size)
            blend = self.calibration_blend_max * torch.sigmoid(self.calibration_blend_mlp(features))[:, :, None, None]
            delta = self._apply_source_gate(blend * bias, source_center)
            calibrated = affine_pred + delta
            return torch.clamp(calibrated, 0.0, 1.0), None, None, None, policy_features, bias, affine_pred, delta
        if self.calibration_head in ("monotonic_lut", "source_monotonic_lut"):
            interval_logits = self.calibration_mlp(features)
            intervals = F.softplus(interval_logits) + 1e-4
            cumulative = torch.cumsum(intervals, dim=1)
            normalized = cumulative / cumulative[:, -1:].clamp_min(1e-6)
            zero = normalized.new_zeros(normalized.shape[0], 1)
            y_knots = torch.cat([zero, normalized], dim=1).clamp(0.0, 1.0)
            lut_input = source_center if self.calibration_head == "source_monotonic_lut" else raw_pred
            mapped = self._interpolate_lut(lut_input, y_knots)
            blend = self.calibration_blend_max * torch.sigmoid(self.calibration_blend_mlp(features))[:, :, None, None]
            if self.calibration_guard_mode == "field_extreme_v1":
                source_is_7t = source_field_id == (len(FIELDS) - 1)
                target_is_7t = target_field_id == (len(FIELDS) - 1)
                extreme = (source_is_7t | target_is_7t).to(dtype=blend.dtype, device=blend.device)
                guard = 1.0 + extreme[:, None, None, None] * (self.calibration_guard_extreme_scale - 1.0)
                blend = blend * guard
            calibrated = raw_pred + blend * (mapped - raw_pred)
            return torch.clamp(calibrated, 0.0, 1.0), y_knots, None, None, policy_features, None, None, None
        if self.calibration_head == "lowfreq_bias":
            if feature_map is None:
                raise RuntimeError("lowfreq_bias calibration requires decoder feature_map")
            bias = torch.tanh(self.calibration_bias_head(feature_map))
            bias = self._smooth(bias, self.calibration_lowfreq_kernel_size)
            blend = self.calibration_blend_max * torch.sigmoid(self.calibration_blend_mlp(features))[:, :, None, None]
            delta = self._apply_source_gate(blend * bias, source_center)
            calibrated = raw_pred + delta
            return torch.clamp(calibrated, 0.0, 1.0), None, None, None, policy_features, bias, raw_pred, delta
        if self.calibration_head == "local_lowfreq_field_v1":
            calibrated, field, anchor, delta = self._apply_local_lowfreq_field_head(
                source_center,
                feature_map,
            )
            return calibrated, None, None, None, policy_features, field, anchor, delta
        if self.calibration_head == "target_stat_lowfreq_field_v1":
            calibrated, knots, field, anchor, delta = self._apply_target_stat_lowfreq_field_head(
                source_center,
                feature_map,
                features,
                contrast_id,
                target_field_id,
            )
            return calibrated, knots, None, None, policy_features, field, anchor, delta
        if self.calibration_head == "target_stat_anchor_detail_v1":
            calibrated, knots, field, anchor, delta = self._apply_target_stat_anchor_detail_head(
                raw_pred,
                source_center,
                feature_map,
                features,
                contrast_id,
                target_field_id,
            )
            return calibrated, knots, None, None, policy_features, field, anchor, delta
        if self.calibration_head == "target_stat_anchor_lowrank_residual_v1":
            calibrated, knots, field, anchor, delta = self._apply_target_stat_anchor_lowrank_residual_head(
                raw_pred,
                source_center,
                feature_map,
                features,
                contrast_id,
                target_field_id,
            )
            return calibrated, knots, None, None, policy_features, field, anchor, delta
        if self.calibration_head == "target_stat_anchor_fine_residual_v1":
            calibrated, knots, field, anchor, delta = self._apply_target_stat_anchor_fine_residual_head(
                raw_pred,
                source_center,
                feature_map,
                features,
                contrast_id,
                target_field_id,
            )
            return calibrated, knots, None, None, policy_features, field, anchor, delta
        if self.calibration_head == "target_stat_anchor_swin_residual_v1":
            calibrated, knots, field, anchor, delta = self._apply_target_stat_anchor_swin_residual_head(
                raw_pred,
                source_center,
                feature_map,
                features,
                contrast_id,
                target_field_id,
            )
            return calibrated, knots, None, None, policy_features, field, anchor, delta
        if self.calibration_head == "target_stat_gain_bias_field_v1":
            calibrated, knots, field, anchor, delta = self._apply_target_stat_gain_bias_field_head(
                raw_pred,
                source_center,
                feature_map,
                features,
                contrast_id,
                target_field_id,
            )
            return calibrated, knots, None, None, policy_features, field, anchor, delta
        if self.calibration_head == "target_stat_gain_bias_lowres_delta_v1":
            calibrated, knots, field, anchor, delta = self._apply_target_stat_gain_bias_lowres_delta_head(
                raw_pred,
                source_center,
                feature_map,
                features,
                contrast_id,
                target_field_id,
            )
            return calibrated, knots, None, None, policy_features, field, anchor, delta
        if self.calibration_head == "lowres_calibration_field_v1":
            calibrated, knots, field, anchor, delta = self._apply_lowres_calibration_field_head(
                raw_pred,
                source_center,
                feature_map,
                features,
                contrast_id,
                target_field_id,
            )
            return calibrated, knots, None, None, policy_features, field, anchor, delta
        if self.calibration_head == "source_stat_robust":
            calibrated, robust, delta = self._apply_source_stat_robust_head(
                raw_pred,
                source_center,
                stats,
                contrast_id,
                target_field_id,
            )
            return calibrated, None, None, None, policy_features, None, robust, delta
        if self.calibration_head == "source_stat_percentile":
            calibrated, mapped, delta, target_knots = self._apply_source_stat_percentile_head(
                raw_pred,
                source_center,
                features,
                contrast_id,
                target_field_id,
            )
            return calibrated, target_knots, None, None, policy_features, None, mapped, delta
        if self.calibration_head == "policy_param_lut_lowfreq":
            calibrated, lowfreq, delta = self._apply_policy_param_lut_lowfreq_head(
                raw_pred,
                source_center,
                features,
                policy_features,
            )
            return calibrated, None, None, None, policy_features, None, lowfreq, delta
        if self.calibration_head == "direct_param_source_safe_v1":
            calibrated, anchor, delta = self._apply_direct_param_source_safe_head(
                raw_pred,
                source_center,
                features,
                policy_features,
            )
            return calibrated, None, None, None, policy_features, None, anchor, delta
        if self.calibration_head in (
            "direct_param_target_percentile_v1",
            "direct_param_target_percentile_guarded_v2",
            "direct_param_target_percentile_guarded_v3",
        ):
            calibrated, anchor, delta = self._apply_direct_param_target_percentile_head(
                raw_pred,
                source_center,
                features,
                policy_features,
                contrast_id,
                target_field_id,
                source_field_id,
            )
            return calibrated, None, None, None, policy_features, None, anchor, delta
        if self.calibration_head == "direct_control_target_percentile_v1":
            calibrated, anchor, delta = self._apply_direct_control_target_percentile_head(
                raw_pred,
                source_center,
                base_center,
                features,
                contrast_id,
                target_field_id,
            )
            return calibrated, None, None, None, policy_features, None, anchor, delta
        if self.calibration_head == "direct_control_gain_bias_field_v1":
            calibrated, knots, field, anchor, delta = self._apply_direct_control_gain_bias_field_head(
                raw_pred,
                source_center,
                feature_map,
                features,
                contrast_id,
                target_field_id,
            )
            return calibrated, knots, None, None, policy_features, field, anchor, delta
        if self.calibration_head in (
            "target_percentile_relation_v1",
            "source_anchor_percentile_relation_v1",
            "base_anchor_percentile_relation_v1",
        ):
            calibrated, knots, anchor, delta = self._apply_target_percentile_relation_head(
                raw_pred,
                source_center,
                base_center,
                features,
                contrast_id,
                target_field_id,
            )
            return calibrated, knots, None, None, policy_features, None, anchor, delta
        if self.calibration_head == "source_preserving_monotonic_lowfreq_v1":
            calibrated, knots, anchor, delta = self._apply_source_preserving_monotonic_lowfreq_head(
                raw_pred,
                source_center,
                features,
                contrast_id,
                target_field_id,
            )
            return calibrated, knots, None, None, policy_features, None, anchor, delta
        if self.calibration_head == "calibrated_anchor_detail_v1":
            calibrated, knots, anchor, delta = self._apply_calibrated_anchor_detail_head(
                raw_pred,
                source_center,
                features,
                contrast_id,
                target_field_id,
            )
            return calibrated, knots, None, None, policy_features, None, anchor, delta
        if self.calibration_head == "target_percentile_primitive_bank_v1":
            calibrated, knots, weights, primitive_base, anchor = self._apply_target_percentile_primitive_bank_head(
                raw_pred,
                source_center,
                base_center,
                features,
                contrast_id,
                target_field_id,
            )
            return calibrated, knots, weights, primitive_base, policy_features, None, anchor, calibrated - anchor
        if self.calibration_head == "source_lut_lowfreq_texture":
            calibrated, lowfreq, delta = self._apply_source_lut_lowfreq_texture_head(
                raw_pred,
                source_center,
                features,
            )
            return calibrated, None, None, None, policy_features, None, lowfreq, delta
        if self.calibration_head == "primitive_bank":
            raw = torch.clamp(raw_pred, 0.0, 1.0)
            affine_pred = None
            anchor = raw
            if self.primitive_bank_mode in (
                "affine_source_safe_v1",
                "policy_affine_source_safe_v1",
                "policy_lowfreq_source_safe_v1",
                "policy_lut_lowfreq_source_safe_v1",
                "policy_lut_lowfreq_median_source_safe_v1",
            ):
                affine_pred = self._apply_affine_head(raw_pred, features)
                anchor = affine_pred
            primitives = self._build_primitive_bank(
                raw_pred,
                source_center,
                base_center,
                affine_pred=affine_pred,
                policy_features=policy_features,
                features=features,
            )
            self._last_primitive_candidates = primitives
            primitive_features = self._primitive_condition_features(features, policy_features)
            weights = torch.softmax(self.primitive_weight_mlp(primitive_features), dim=1)
            primitive_base = (primitives * weights[:, :, None, None, None]).sum(dim=1)
            blend = self.primitive_bank_blend_max * torch.sigmoid(self.primitive_blend_mlp(primitive_features))[
                :, :, None, None
            ]
            self._last_primitive_blend = blend
            calibrated = anchor + blend * (primitive_base - anchor)
            return torch.clamp(calibrated, 0.0, 1.0), None, weights, primitive_base, policy_features, None, anchor, calibrated - anchor
        raise RuntimeError(f"Unhandled calibration_head: {self.calibration_head}")

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
            base_center = source_center
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
            x = x + skip
            x = decoder(x, cond)

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
            source_center,
            base_center,
            x,
            source_field_id,
            target_field_id,
            contrast_id,
        )
        features = torch.cat([cond, stats], dim=1)
        method_bank_features = self._method_bank_features(features, source_field_id, target_field_id)
        method_bank_logits = self._predict_method_bank_logits(method_bank_features)
        executor_family_logits = (
            self.executor_family_mlp(features) if self.executor_family_size > 0 else None
        )
        executor_params = (
            torch.sigmoid(self.executor_param_mlp(features)) if self.executor_param_dim > 0 else None
        )
        pred, tissue_delta, tissue_coefficients = self._apply_tissue_bridge(
            pred,
            tissue_logits,
            features,
            source_center,
            base_center,
        )
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
            tissue_delta=tissue_delta,
            tissue_coefficients=tissue_coefficients,
            lowres_field=getattr(self, "_last_lowres_field", None),
            lowres_field_upsampled=getattr(self, "_last_lowres_field_upsampled", None),
        )


class PlainCondUNet(CondNAFUNet):
    """CondNAFUNet scaffold with plain convolutional residual blocks."""

    def __init__(self, *args, **kwargs) -> None:
        requested_family = kwargs.pop("block_family", "plain")
        normalized_family = str(requested_family).strip().lower().replace("-", "_")
        if normalized_family not in ("plain", "plain_conv", "plain_residual"):
            raise ValueError("PlainCondUNet requires block_family=plain")
        super().__init__(*args, block_family="plain", **kwargs)
