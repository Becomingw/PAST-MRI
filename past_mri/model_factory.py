from __future__ import annotations

from past_mri.models import CondNAFUNet, CondNAFUNet5x5


def build_cond_naf_unet_from_config(config: dict) -> CondNAFUNet:
    model_cfg = config.get("model", {})
    base_calib_cfg = config.get("base_calibration", {})
    model_name = str(model_cfg.get("name", "CondNAFUNet")).strip().lower().replace("-", "_")
    if model_name in ("condnafunet", "cond_naf_unet"):
        model_cls = CondNAFUNet
        expected_family = "naf"
    elif model_name in ("condnafunet5x5", "cond_naf_unet_5x5"):
        model_cls = CondNAFUNet5x5
        expected_family = "naf"
    else:
        raise ValueError(f"Unknown model.name: {model_cfg.get('name')!r}")
    block_family = str(model_cfg.get("block_family", expected_family)).strip().lower().replace("-", "_")
    if expected_family == "naf" and block_family not in ("naf", "nafnet", "cond_naf"):
        raise ValueError("CondNAFUNet requires model.block_family=naf")
    return model_cls(
        in_channels=int(model_cfg.get("in_channels", 5)),
        width=int(model_cfg.get("width", 48)),
        enc_blocks=list(model_cfg.get("enc_blocks", [2, 2, 4])),
        mid_blocks=int(model_cfg.get("mid_blocks", 8)),
        dec_blocks=list(model_cfg.get("dec_blocks", [2, 2, 2])),
        cond_dim=int(model_cfg.get("cond_dim", 256)),
        alpha_max=float(model_cfg.get("alpha_max", 1.0)),
        output_mode=str(model_cfg.get("output_mode", "bounded_residual")),
        use_calibration_head=bool(model_cfg.get("use_calibration_head", True)),
        calibration_head=str(model_cfg.get("calibration_head", "affine")),
        calibration_knots=int(model_cfg.get("calibration_knots", 8)),
        calibration_knot_delta_max=float(model_cfg.get("calibration_knot_delta_max", 0.05)),
        calibration_relation_map_init_logit=float(model_cfg.get("calibration_relation_map_init_logit", -4.0)),
        calibration_relation_lowfreq_init_logit=float(model_cfg.get("calibration_relation_lowfreq_init_logit", -4.0)),
        calibration_relation_source_floor_init_logit=float(
            model_cfg.get("calibration_relation_source_floor_init_logit", -6.0)
        ),
        calibration_lowfreq_kernel_size=int(model_cfg.get("calibration_lowfreq_kernel_size", 9)),
        calibration_blend_max=float(model_cfg.get("calibration_blend_max", 1.0)),
        calibration_texture_blend_max=float(model_cfg.get("calibration_texture_blend_max", 1.0)),
        calibration_gain_scale_max=float(model_cfg.get("calibration_gain_scale_max", 0.25)),
        calibration_bias_scale_max=float(model_cfg.get("calibration_bias_scale_max", 0.06)),
        calibration_guard_mode=str(model_cfg.get("calibration_guard_mode", "none")),
        calibration_guard_extreme_scale=float(model_cfg.get("calibration_guard_extreme_scale", 0.25)),
        calibration_guard_v3_downfield_boost=float(model_cfg.get("calibration_guard_v3_downfield_boost", 0.0)),
        calibration_guard_v3_blur_boost=float(model_cfg.get("calibration_guard_v3_blur_boost", 0.0)),
        calibration_source_gate_mode=str(model_cfg.get("calibration_source_gate_mode", "none")),
        calibration_source_gate_floor=float(model_cfg.get("calibration_source_gate_floor", 0.25)),
        calibration_source_gate_strength=float(model_cfg.get("calibration_source_gate_strength", 1.0)),
        primitive_bank_blend_max=float(model_cfg.get("primitive_bank_blend_max", 1.0)),
        primitive_bank_mode=str(model_cfg.get("primitive_bank_mode", "basic")),
        primitive_bank_init=str(model_cfg.get("primitive_bank_init", "uniform")),
        primitive_bank_init_strength=float(model_cfg.get("primitive_bank_init_strength", 4.0)),
        policy_feature_dim=int(model_cfg.get("policy_feature_dim", 0)),
        policy_feature_bias_init=model_cfg.get("policy_feature_bias_init"),
        policy_feature_bias_zero_weight=bool(model_cfg.get("policy_feature_bias_zero_weight", False)),
        method_bank_size=int(model_cfg.get("method_bank_size", 0)),
        method_bank_feature_mode=str(model_cfg.get("method_bank_feature_mode", "default")),
        executor_family_size=int(model_cfg.get("executor_family_size", 0)),
        executor_param_dim=int(model_cfg.get("executor_param_dim", 0)),
        primitive_bank_use_policy_features=bool(model_cfg.get("primitive_bank_use_policy_features", False)),
        target_stats_path=model_cfg.get("target_stats_path", base_calib_cfg.get("target_stats_path")),
        source_stats_mode=str(model_cfg.get("source_stats_mode", "legacy7")),
        direct_control_init_values=model_cfg.get("direct_control_init_values"),
        direct_control_init_weight_std=float(model_cfg.get("direct_control_init_weight_std", 0.0)),
        use_tissue_head=bool(model_cfg.get("use_tissue_head", False)),
        tissue_classes=int(model_cfg.get("tissue_classes", 7)),
        use_tissue_bridge=bool(model_cfg.get("use_tissue_bridge", False)),
        tissue_bridge_max=float(model_cfg.get("tissue_bridge_max", 0.02)),
        tissue_bridge_kernel_size=int(model_cfg.get("tissue_bridge_kernel_size", 9)),
        use_tissue_local_bridge=bool(model_cfg.get("use_tissue_local_bridge", False)),
        tissue_local_bridge_max=float(model_cfg.get("tissue_local_bridge_max", 0.02)),
        tissue_local_bridge_hidden=int(model_cfg.get("tissue_local_bridge_hidden", 16)),
        tissue_local_bridge_kernel_size=int(model_cfg.get("tissue_local_bridge_kernel_size", 9)),
        use_hyper_residual_head=bool(model_cfg.get("use_hyper_residual_head", False)),
        hyper_residual_hidden=int(model_cfg.get("hyper_residual_hidden", 128)),
        hyper_residual_scale=float(model_cfg.get("hyper_residual_scale", 0.05)),
        hyper_residual_mode=str(model_cfg.get("hyper_residual_mode", "adapter")),
        use_hyper_trunk=bool(model_cfg.get("use_hyper_trunk", False)),
        hyper_trunk_blocks=model_cfg.get("hyper_trunk_blocks", ["mid"]),
        hyper_trunk_rank=int(model_cfg.get("hyper_trunk_rank", 12)),
        hyper_trunk_hidden=int(model_cfg.get("hyper_trunk_hidden", 256)),
        hyper_trunk_scale=float(model_cfg.get("hyper_trunk_scale", 0.05)),
        calibration_lowrank_components=int(model_cfg.get("calibration_lowrank_components", 4)),
        calibration_lowrank_delta_max=float(model_cfg.get("calibration_lowrank_delta_max", 0.02)),
        calibration_lowrank_coeff_init=float(model_cfg.get("calibration_lowrank_coeff_init", 0.1)),
        lowres_field_downsample=int(model_cfg.get("lowres_field_downsample", 8)),
        lowres_field_mode=str(model_cfg.get("lowres_field_mode", "additive")),
        swin_residual_depth=int(model_cfg.get("swin_residual_depth", 2)),
        swin_residual_heads=int(model_cfg.get("swin_residual_heads", 2)),
        swin_residual_window_size=int(model_cfg.get("swin_residual_window_size", 8)),
        block_family=block_family,
        **(
            {"output_channels": int(model_cfg.get("output_channels", model_cfg.get("in_channels", 5)))}
            if model_cls is CondNAFUNet5x5
            else {}
        ),
    )
