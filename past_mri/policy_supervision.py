from __future__ import annotations

import torch
import torch.nn.functional as F

from past_mri.method_features import METHOD_FEATURE_TO_INDEX, METHOD_FEATURE_NAMES
from past_mri.models.cond_naf_unet import ModelOutput


DEFAULT_POLICY_FEATURE_SCALES = torch.tensor(
    [
        1.0,  # family_source_copy
        1.0,  # family_quantile
        1.0,  # family_targetcohort
        1.0,  # family_other
        1.0,  # beta
        1.0,  # lowfreq_alpha
        6.0,  # lowfreq_sigma_xy
        2.0,  # lowfreq_sigma_z
        1.0,  # source_texture
        2.0,  # texture_blur_sigma_xy
        2.0,  # texture_blur_sigma_z
        1.0,  # texture_unsharp_amount
        5.0,  # texture_median_kernel_xy
        1.0,  # texture_median_kernel_z
    ],
    dtype=torch.float32,
)


def resolve_policy_feature_scales(
    value: object,
    device: torch.device | str,
) -> torch.Tensor | None:
    if value is None or value is False:
        return None
    if isinstance(value, str):
        if value.strip().lower() in ("", "none", "false"):
            return None
        if value.strip().lower() == "method-default":
            return DEFAULT_POLICY_FEATURE_SCALES.to(device=device)
        raise ValueError(f"Unknown policy_feature_scales preset: {value}")
    scales = torch.as_tensor(value, dtype=torch.float32, device=device)
    if scales.numel() != len(METHOD_FEATURE_NAMES):
        raise ValueError(
            f"policy_feature_scales must have {len(METHOD_FEATURE_NAMES)} values, got {scales.numel()}"
        )
    return scales.view(-1)


def policy_feature_loss(
    output: ModelOutput,
    target_features: torch.Tensor,
    has_target: torch.Tensor,
    feature_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    if output.policy_features is None:
        return target_features.new_zeros(())
    mask = has_target.bool().view(-1)
    if not bool(mask.any()):
        return target_features.new_zeros(())
    pred = output.policy_features[mask]
    target = target_features[mask].to(device=pred.device, dtype=pred.dtype)
    if target.shape[1] != pred.shape[1]:
        shared_dim = min(target.shape[1], pred.shape[1])
        pred = pred[:, :shared_dim]
        target = target[:, :shared_dim]
    if feature_scales is not None:
        scales = torch.as_tensor(feature_scales, device=pred.device, dtype=pred.dtype).view(1, -1)
        if scales.shape[1] < pred.shape[1]:
            raise ValueError(f"feature_scales dim {scales.shape[1]} does not cover policy dim {pred.shape[1]}")
        scales = scales[:, : pred.shape[1]]
        scales = scales.clamp_min(1e-6)
        pred = pred / scales
        target = target / scales
    return (pred - target).pow(2).mean()


def guard_feature_targets(target_features: torch.Tensor) -> torch.Tensor:
    features = target_features
    idx = METHOD_FEATURE_TO_INDEX
    family_quantile = features[:, idx["family_quantile"]].clamp(0.0, 1.0)
    family_targetcohort = features[:, idx["family_targetcohort"]].clamp(0.0, 1.0)
    family_source_copy = features[:, idx["family_source_copy"]].clamp(0.0, 1.0)
    percentile_strength = torch.maximum(family_quantile, family_targetcohort).clamp(0.0, 1.0)
    map_beta = (features[:, idx["beta"]].clamp(0.0, 1.0) * percentile_strength).clamp(0.0, 1.0)
    lowfreq_alpha = features[:, idx["lowfreq_alpha"]].clamp(0.0, 1.0)
    blur_strength = (features[:, idx["texture_blur_sigma_xy"]].clamp(0.0, 6.0) / 6.0).clamp(0.0, 1.0)
    source_floor = family_source_copy
    return torch.stack([map_beta, lowfreq_alpha, blur_strength, source_floor], dim=1)


def guard_feature_loss(
    output: ModelOutput,
    target_features: torch.Tensor,
    has_target: torch.Tensor,
    feature_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if output.guard_features is None:
        return target_features.new_zeros(())
    mask = has_target.bool().view(-1)
    if not bool(mask.any()):
        return target_features.new_zeros(())
    pred = output.guard_features[mask]
    target = guard_feature_targets(target_features[mask].to(device=pred.device, dtype=pred.dtype))
    shared_dim = min(pred.shape[1], target.shape[1])
    diff = (pred[:, :shared_dim] - target[:, :shared_dim]).pow(2)
    if feature_weights is not None:
        weights = torch.as_tensor(feature_weights, device=pred.device, dtype=pred.dtype).view(1, -1)
        if weights.shape[1] < shared_dim:
            raise ValueError(f"guard feature weights dim {weights.shape[1]} does not cover guard dim {shared_dim}")
        weights = weights[:, :shared_dim].clamp_min(0.0)
        denom = weights.sum().clamp_min(1e-6)
        return (diff * weights).sum() / (diff.shape[0] * denom)
    return diff.mean()


def direct_guard_feature_loss(
    output: ModelOutput,
    target_controls: torch.Tensor,
    has_target: torch.Tensor,
    feature_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if output.guard_features is None:
        return target_controls.new_zeros(())
    mask = has_target.bool().view(-1)
    if not bool(mask.any()):
        return target_controls.new_zeros(())
    pred = output.guard_features[mask]
    target = target_controls[mask].to(device=pred.device, dtype=pred.dtype)
    shared_dim = min(pred.shape[1], target.shape[1])
    diff = (pred[:, :shared_dim] - target[:, :shared_dim]).pow(2)
    if feature_weights is not None:
        weights = torch.as_tensor(feature_weights, device=pred.device, dtype=pred.dtype).view(1, -1)
        if weights.shape[1] < shared_dim:
            raise ValueError(f"direct guard weights dim {weights.shape[1]} does not cover guard dim {shared_dim}")
        weights = weights[:, :shared_dim].clamp_min(0.0)
        denom = weights.sum().clamp_min(1e-6)
        return (diff * weights).sum() / (diff.shape[0] * denom)
    return diff.mean()


def primitive_target_indices(target_features: torch.Tensor, primitive_count: int | None = None) -> torch.Tensor:
    features = target_features
    idx = METHOD_FEATURE_TO_INDEX
    target = torch.zeros(features.shape[0], device=features.device, dtype=torch.long)

    is_source = features[:, idx["family_source_copy"]] > 0.5
    is_quantile = features[:, idx["family_quantile"]] > 0.5
    is_targetcohort = features[:, idx["family_targetcohort"]] > 0.5
    has_beta = features[:, idx["beta"]] > 1e-6
    has_lowfreq = features[:, idx["lowfreq_alpha"]] > 1e-6
    has_texture = (
        (features[:, idx["source_texture"]] > 0.5)
        | (features[:, idx["texture_blur_sigma_xy"]] > 1e-6)
        | (features[:, idx["texture_blur_sigma_z"]] > 1e-6)
        | (features[:, idx["texture_median_kernel_xy"]] > 1e-6)
        | (features[:, idx["texture_median_kernel_z"]] > 1e-6)
        | (features[:, idx["texture_unsharp_amount"]] > 1e-6)
    )
    has_median = (
        (features[:, idx["texture_median_kernel_xy"]] > 1e-6)
        | (features[:, idx["texture_median_kernel_z"]] > 1e-6)
    )

    target[is_source] = 3
    target[is_source & has_texture] = 7
    if primitive_count is not None and primitive_count >= 10:
        target[is_source & has_median] = 9
    target[is_quantile] = 5
    target[is_quantile & has_lowfreq] = 6
    target[is_quantile & has_texture] = 8
    target[is_targetcohort | has_beta] = 4
    return target


def primitive_weight_target_loss(
    output: ModelOutput,
    target_features: torch.Tensor,
    has_target: torch.Tensor,
    smoothing: float = 0.0,
) -> torch.Tensor:
    if output.primitive_weights is None:
        return target_features.new_zeros(())
    weights = output.primitive_weights
    if weights.shape[1] < 9:
        raise ValueError("primitive_weight_target_loss currently expects policy_lut_lowfreq_source_safe_v1 with 9 candidates")
    mask = has_target.bool().view(-1)
    if not bool(mask.any()):
        return target_features.new_zeros(())
    weights = weights[mask].clamp_min(1e-8)
    target = primitive_target_indices(
        target_features[mask].to(device=weights.device, dtype=weights.dtype),
        primitive_count=weights.shape[1],
    )
    log_weights = weights.log()
    if smoothing <= 0.0:
        return -log_weights.gather(1, target[:, None]).mean()
    smoothing = min(max(float(smoothing), 0.0), 1.0)
    target_dist = torch.full_like(weights, smoothing / weights.shape[1])
    target_dist.scatter_(1, target[:, None], 1.0 - smoothing + smoothing / weights.shape[1])
    return -(target_dist * log_weights).sum(dim=1).mean()


def method_bank_target_loss(
    output: ModelOutput,
    target_indices: torch.Tensor,
    has_target: torch.Tensor,
    smoothing: float = 0.0,
) -> torch.Tensor:
    if output.method_bank_logits is None:
        return target_indices.new_zeros((), dtype=torch.float32)
    logits = output.method_bank_logits
    mask = has_target.bool().view(-1)
    if not bool(mask.any()):
        return logits.new_zeros(())
    target = target_indices[mask].to(device=logits.device, dtype=torch.long)
    if bool((target < 0).any()) or bool((target >= logits.shape[1]).any()):
        raise ValueError(
            f"method_bank target index must be in [0, {logits.shape[1] - 1}] "
            f"for active samples"
        )
    return F.cross_entropy(logits[mask], target, label_smoothing=float(smoothing))


def method_bank_soft_target_loss(
    output: ModelOutput,
    target_probs: torch.Tensor,
    has_target: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    if output.method_bank_logits is None:
        return target_probs.new_zeros((), dtype=torch.float32)
    logits = output.method_bank_logits
    mask = has_target.bool().view(-1)
    if not bool(mask.any()):
        return logits.new_zeros(())
    target = target_probs[mask].to(device=logits.device, dtype=logits.dtype)
    if target.ndim != 2 or target.shape != logits[mask].shape:
        raise ValueError(
            "method_bank soft target/logits shape mismatch: "
            f"target={tuple(target.shape)}, logits={tuple(logits[mask].shape)}"
        )
    target_sum = target.sum(dim=1, keepdim=True).clamp_min(1e-8)
    target = target.clamp_min(0.0) / target_sum
    temperature = max(float(temperature), 1e-6)
    log_probs = F.log_softmax(logits[mask] / temperature, dim=1)
    return -(target * log_probs).sum(dim=1).mean()


def method_bank_family_soft_target_loss(
    output: ModelOutput,
    candidate_family_ids: torch.Tensor,
    target_probs: torch.Tensor,
    has_target: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    if output.method_bank_logits is None:
        return target_probs.new_zeros((), dtype=torch.float32)
    logits = output.method_bank_logits
    mask = has_target.bool().view(-1)
    if not bool(mask.any()):
        return logits.new_zeros(())
    family_ids = candidate_family_ids[mask].to(device=logits.device, dtype=torch.long)
    target = target_probs[mask].to(device=logits.device, dtype=logits.dtype)
    active_logits = logits[mask]
    if family_ids.shape != active_logits.shape or target.shape != active_logits.shape:
        raise ValueError(
            "method_bank family/logit/target shape mismatch: "
            f"family={tuple(family_ids.shape)}, target={tuple(target.shape)}, logits={tuple(active_logits.shape)}"
        )
    valid = family_ids >= 0
    if not bool(valid.any()):
        return logits.new_zeros(())
    family_count = int(family_ids[valid].max().item()) + 1
    family_count = max(family_count, 1)
    temperature = max(float(temperature), 1e-6)
    probs = F.softmax(active_logits / temperature, dim=1)
    target_sum = target.sum(dim=1, keepdim=True).clamp_min(1e-8)
    target = target.clamp_min(0.0) / target_sum
    family_ids_safe = family_ids.clamp_min(0)
    pred_family = probs.new_zeros((active_logits.shape[0], family_count))
    target_family = target.new_zeros((active_logits.shape[0], family_count))
    pred_family.scatter_add_(1, family_ids_safe, probs * valid.to(dtype=probs.dtype))
    target_family.scatter_add_(1, family_ids_safe, target * valid.to(dtype=target.dtype))
    target_family = target_family / target_family.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return -(target_family * pred_family.clamp_min(1e-12).log()).sum(dim=1).mean()


def executor_family_soft_target_loss(
    output: ModelOutput,
    target_probs: torch.Tensor,
    has_target: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    if output.executor_family_logits is None:
        return target_probs.new_zeros((), dtype=torch.float32)
    logits = output.executor_family_logits
    mask = has_target.bool().view(-1)
    if not bool(mask.any()):
        return logits.new_zeros(())
    target = target_probs[mask].to(device=logits.device, dtype=logits.dtype)
    active_logits = logits[mask]
    if target.ndim != 2 or target.shape != active_logits.shape:
        raise ValueError(
            "executor family target/logits shape mismatch: "
            f"target={tuple(target.shape)}, logits={tuple(active_logits.shape)}"
        )
    target = target.clamp_min(0.0)
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-8)
    temperature = max(float(temperature), 1e-6)
    log_probs = F.log_softmax(active_logits / temperature, dim=1)
    return -(target * log_probs).sum(dim=1).mean()


def executor_family_hard_target_loss(
    output: ModelOutput,
    target_indices: torch.Tensor,
    has_target: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    smoothing: float = 0.0,
) -> torch.Tensor:
    if output.executor_family_logits is None:
        return target_indices.new_zeros((), dtype=torch.float32)
    logits = output.executor_family_logits
    mask = has_target.bool().view(-1)
    if not bool(mask.any()):
        return logits.new_zeros(())
    target = target_indices[mask].to(device=logits.device, dtype=torch.long)
    active_logits = logits[mask]
    if bool((target < 0).any()) or bool((target >= active_logits.shape[1]).any()):
        raise ValueError(
            f"executor family target index must be in [0, {active_logits.shape[1] - 1}] "
            "for active samples"
        )
    weight = None
    if class_weights is not None:
        weight = torch.as_tensor(class_weights, device=logits.device, dtype=logits.dtype).view(-1)
        if weight.numel() != active_logits.shape[1]:
            raise ValueError(
                f"executor family class_weights dim {weight.numel()} "
                f"does not match family count {active_logits.shape[1]}"
            )
    return F.cross_entropy(
        active_logits,
        target,
        weight=weight,
        label_smoothing=float(smoothing),
    )


def executor_param_target_loss(
    output: ModelOutput,
    target_params: torch.Tensor,
    has_target: torch.Tensor,
    feature_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if output.executor_params is None:
        return target_params.new_zeros((), dtype=torch.float32)
    pred = output.executor_params
    mask = has_target.bool().view(-1)
    if not bool(mask.any()):
        return pred.new_zeros(())
    target = target_params[mask].to(device=pred.device, dtype=pred.dtype)
    pred = pred[mask]
    if target.ndim != 2 or pred.ndim != 2:
        raise ValueError(
            "executor param target/pred tensors must be 2D: "
            f"target={tuple(target.shape)}, pred={tuple(pred.shape)}"
        )
    shared_dim = min(pred.shape[1], target.shape[1])
    if shared_dim == 0:
        return pred.new_zeros(())
    diff = (pred[:, :shared_dim] - target[:, :shared_dim]).pow(2)
    if feature_weights is not None:
        weights = torch.as_tensor(feature_weights, device=pred.device, dtype=pred.dtype).view(1, -1)
        if weights.shape[1] < shared_dim:
            raise ValueError(
                f"executor_param feature_weights dim {weights.shape[1]} "
                f"does not cover param dim {shared_dim}"
            )
        weights = weights[:, :shared_dim].clamp_min(0.0)
        denom = weights.sum().clamp_min(1e-6)
        return (diff * weights).sum() / (diff.shape[0] * denom)
    return diff.mean()


def method_bank_candidate_blend(
    output: ModelOutput,
    candidates: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    if output.method_bank_logits is None:
        raise ValueError("method_bank_candidate_blend requires output.method_bank_logits")
    logits = output.method_bank_logits
    if candidates.ndim != 5:
        raise ValueError(f"method_bank candidates must be [B,K,1,H,W], got {tuple(candidates.shape)}")
    if logits.shape[0] != candidates.shape[0] or logits.shape[1] != candidates.shape[1]:
        raise ValueError(
            "method_bank logits/candidates shape mismatch: "
            f"logits={tuple(logits.shape)}, candidates={tuple(candidates.shape)}"
        )
    temperature = max(float(temperature), 1e-6)
    weights = F.softmax(logits / temperature, dim=1).to(dtype=candidates.dtype)
    return (weights[:, :, None, None, None] * candidates).sum(dim=1)


def method_bank_candidate_blend_loss(
    output: ModelOutput,
    candidates: torch.Tensor,
    target: torch.Tensor,
    has_candidates: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    if output.method_bank_logits is None:
        return target.new_zeros(())
    mask = has_candidates.bool().view(-1)
    if not bool(mask.any()):
        return target.new_zeros(())
    candidates = candidates.to(device=target.device, dtype=target.dtype)
    blend = method_bank_candidate_blend(output, candidates, temperature=temperature)
    target = target.to(device=blend.device, dtype=blend.dtype)
    if blend.shape != target.shape:
        raise ValueError(f"method_bank blend/target shape mismatch: {tuple(blend.shape)} vs {tuple(target.shape)}")
    return (blend[mask] - target[mask]).abs().flatten(1).mean()


def method_bank_candidate_masked_nrmse_loss(
    output: ModelOutput,
    candidates: torch.Tensor,
    target: torch.Tensor,
    has_candidates: torch.Tensor,
    *,
    temperature: float = 1.0,
    mask_eps: float = 1e-4,
) -> torch.Tensor:
    if output.method_bank_logits is None:
        return target.new_zeros(())
    mask = has_candidates.bool().view(-1)
    if not bool(mask.any()):
        return target.new_zeros(())
    candidates = candidates.to(device=target.device, dtype=target.dtype)
    blend = method_bank_candidate_blend(output, candidates, temperature=temperature)
    target = target.to(device=blend.device, dtype=blend.dtype)
    if blend.shape != target.shape:
        raise ValueError(f"method_bank blend/target shape mismatch: {tuple(blend.shape)} vs {tuple(target.shape)}")
    blend = blend[mask]
    target = target[mask]
    valid = (target > float(mask_eps)).to(dtype=blend.dtype)
    count = valid.flatten(1).sum(dim=1).clamp_min(1.0)
    mse = ((blend - target).pow(2) * valid).flatten(1).sum(dim=1) / count
    target_power = (target.pow(2) * valid).flatten(1).sum(dim=1) / count
    return (torch.sqrt(mse + 1e-8) / torch.sqrt(target_power + 1e-8)).mean()


def method_bank_entropy_loss(
    output: ModelOutput,
    has_target: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    if output.method_bank_logits is None:
        return has_target.new_zeros((), dtype=torch.float32)
    logits = output.method_bank_logits
    mask = has_target.bool().view(-1)
    if not bool(mask.any()):
        return logits.new_zeros(())
    temperature = max(float(temperature), 1e-6)
    probs = F.softmax(logits[mask] / temperature, dim=1).clamp_min(1e-12)
    return -(probs * probs.log()).sum(dim=1).mean()
