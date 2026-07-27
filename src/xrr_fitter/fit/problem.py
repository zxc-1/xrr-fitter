"""Compilation of immutable single-dataset fitting problems."""

from __future__ import annotations

from math import isfinite

import numpy as np

from xrr_fitter.evaluation import assign_fit_regions, region_weights
from xrr_fitter.fit.parameters import (
    apply_parameter_settings,
    default_parameter_definitions,
    stage_parameter_settings,
    validate_compiled_definitions,
    validate_instrument_modes,
)
from xrr_fitter.model.data import PreparedData
from xrr_fitter.model.fitting import FitConfig, FitEvaluationContext
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import (
    ParameterCoordinate,
    ParameterDefinition,
    ParameterSetting,
)
from xrr_fitter.model.structure import StructureSpec
from xrr_fitter.model.structure import LayerSpec, PeriodicBlock

def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _valid_worker_count(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, np.integer)) and value >= 1


def _valid_scale_prior(config: FitConfig) -> bool:
    return (
        isinstance(config.scale_prior_enabled, bool)
        and isfinite(config.scale_prior_tau_decades)
        and config.scale_prior_tau_decades > 0.0
    )


def _validate_config(config: FitConfig) -> None:
    if (config.objective_name, config.objective_version) != ("robust_log_soft_l1", "1"):
        raise ValueError("unsupported objective configuration")
    if config.final_seed_count != 4 or not isfinite(config.c_decades) or config.c_decades <= 0.0:
        raise ValueError("invalid standard fit configuration: c_decades")
    if not _valid_worker_count(config.local_workers):
        raise ValueError("invalid standard fit configuration")
    if not _valid_scale_prior(config):
        raise ValueError("invalid scale-prior configuration")


def _validate_data_mode(data: PreparedData, instrument: InstrumentSpec) -> None:
    if not data.fit_ready:
        raise ValueError("current fit mask is not fit-ready")
    if instrument.resolution_domain == "theta" and data.resolution_raw is not None:
        raise ValueError("per-point resolution columns are unsupported in theta-domain mode")


def _variables(
    definitions: tuple[ParameterDefinition, ...],
) -> tuple[ParameterCoordinate, ...]:
    return tuple(
        ParameterCoordinate(index, definition.name, definition.transform)
        for index, definition in enumerate(definitions)
        if not definition.locked
    )


def _require_explicit_expert_density(
    structure: StructureSpec,
    parameter_settings: tuple[ParameterSetting, ...],
) -> None:
    configured = {setting.name for setting in parameter_settings}
    for component_index, component in enumerate(structure.components):
        if isinstance(component, LayerSpec):
            layers = ((f"component.{component_index}", component),)
        elif isinstance(component, PeriodicBlock):
            layers = tuple(
                (f"component.{component_index}.layer.{layer_index}", layer)
                for layer_index, layer in enumerate(component.layers)
            )
        else:
            layers = ()
        for prefix, layer in layers:
            name = f"{prefix}.density_scale"
            if not 0.5 <= layer.density_scale <= 1.1 and name not in configured:
                raise ValueError(f"initial outside compiled bounds: {name}")


def _region_layout(data: PreparedData) -> tuple[np.ndarray, np.ndarray]:
    fit_labels = assign_fit_regions(data.qz_a_inv[data.fit_mask])
    fit_weights = region_weights(fit_labels)
    labels = np.full(data.qz_a_inv.shape, -1, dtype=int)
    weights = np.zeros(data.qz_a_inv.shape, dtype=float)
    labels[data.fit_mask] = fit_labels
    weights[data.fit_mask] = fit_weights
    return _readonly(labels), _readonly(weights)


def _plateau_scale_estimate(
    data: PreparedData,
    instrument: InstrumentSpec,
) -> tuple[float | None, str | None]:
    from xrr_fitter.fit.initialization import (
        critical_edge_candidates,
        ramp_inflection_estimate_deg,
    )

    mask = data.fit_mask
    theta = data.two_theta_deg[mask] / 2.0 + data.import_angle_offset_deg
    observed = data.intensity_normalized[mask]
    edges = critical_edge_candidates(theta, observed)
    if not edges:
        return None, "未识别可靠临界边，尺度弱先验已关闭"
    if instrument.footprint_mode == "fit" and ramp_inflection_estimate_deg(data) is not None:
        return None, "低角足迹爬坡未锁定，尺度弱先验已关闭"
    lower = (
        1.05 * instrument.footprint_spill_angle_deg
        if instrument.footprint_mode == "geometry"
        else 0.0
    )
    plateau = (theta > lower) & (theta <= 0.8 * min(edges)) & (observed > 0.0)
    if np.count_nonzero(plateau) < 10:
        return None, "全反射平台点不足，尺度弱先验已关闭"
    x_values = theta[plateau]
    y_values = np.log10(np.maximum(observed[plateau], data.r_floor))
    if np.ptp(x_values) <= 0.0:
        return None, "全反射平台不平坦，尺度弱先验已关闭"
    slope = float(np.polyfit(x_values, y_values, 1)[0])
    if np.ptp(y_values) > 0.10 or abs(slope) * np.ptp(x_values) > 0.05:
        return None, "全反射平台不平坦，尺度弱先验已关闭"
    center = float(np.clip(np.percentile(observed[plateau], 95), 1e-3, 1e3))
    return center, None


def _scale_prior_state(
    data: PreparedData,
    instrument: InstrumentSpec,
    config: FitConfig,
) -> tuple[float | None, str | None]:
    if config.scale_prior_enabled:
        return _plateau_scale_estimate(data, instrument)
    return None, "专家配置已关闭尺度弱先验"


def compile_fit_problem(
    data: PreparedData,
    structure: StructureSpec,
    instrument: InstrumentSpec,
    config: FitConfig,
    parameter_settings: tuple[ParameterSetting, ...] = (),
) -> FitEvaluationContext:
    _validate_config(config)
    _validate_data_mode(data, instrument)
    _require_explicit_expert_density(structure, tuple(parameter_settings))
    definitions = apply_parameter_settings(
        default_parameter_definitions(data, structure, instrument, config),
        tuple(parameter_settings),
    )
    validate_compiled_definitions(definitions)
    validate_instrument_modes(definitions, instrument)
    labels, weights = _region_layout(data)
    center, reason = _scale_prior_state(data, instrument, config)
    return FitEvaluationContext(
        data=data,
        structure=structure,
        instrument=instrument,
        config=config,
        parameter_definitions=definitions,
        variables=_variables(definitions),
        region_labels=labels,
        weights=weights,
        scale_prior_center=center,
        scale_prior_tau_decades=config.scale_prior_tau_decades,
        scale_prior_reason=reason,
        warnings=() if reason is None else (reason,),
    )


def compile_stage_problem(
    problem: FitEvaluationContext,
    stage: str,
    current_values: dict[str, float],
) -> FitEvaluationContext:
    settings = stage_parameter_settings(problem.parameter_definitions, stage, current_values)
    return compile_fit_problem(
        problem.data,
        problem.structure,
        problem.instrument,
        problem.config,
        settings,
    )


def compile_fixed_parameter_problem(
    problem: FitEvaluationContext,
    parameter_name: str,
    value: float,
) -> FitEvaluationContext:
    definitions = {item.name: item for item in problem.parameter_definitions}
    selected = definitions.get(parameter_name)
    if selected is None:
        raise ValueError(f"unknown parameter: {parameter_name}")
    if selected.locked:
        raise ValueError(f"parameter is already locked: {parameter_name}")
    settings = tuple(
        ParameterSetting(
            definition.name,
            value if definition.name == parameter_name else definition.initial,
            value if definition.name == parameter_name else definition.lower,
            value if definition.name == parameter_name else definition.upper,
            locked=True if definition.name == parameter_name else definition.locked,
        )
        for definition in problem.parameter_definitions
    )
    return compile_fit_problem(
        problem.data,
        problem.structure,
        problem.instrument,
        problem.config,
        settings,
    )
