"""Compilation of immutable single-dataset fitting problems."""

from __future__ import annotations

from dataclasses import replace
from math import isfinite

import numpy as np

from xrr_fitter.evaluation import assign_fit_regions, region_weights
from xrr_fitter.fit.drift import DRIFT_DATASET, drift_constraint_rules, rebind_drift_rules
from xrr_fitter.fit.parameters import (
    apply_parameter_settings,
    default_parameter_definitions,
    stage_parameter_settings,
    validate_compiled_definitions,
    validate_instrument_modes,
    validate_transition_modes,
)
from xrr_fitter.model.data import PreparedData
from xrr_fitter.model.fitting import FitConfig, FitEvaluationContext
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import (
    ConstraintRule,
    ParameterCoordinate,
    ParameterDefinition,
    ParameterReference,
    ParameterSetting,
    _iter_references,
    constraint_cycle_path,
    validate_constraint_stage_split,
)
from xrr_fitter.model.structure import LayerSpec, PeriodicBlock, StructureSpec


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
        if not (definition.locked or definition.constrained)
    )


def _mark_constrained(
    definitions: tuple[ParameterDefinition, ...],
    constraint_rules: tuple[ConstraintRule, ...],
) -> tuple[ParameterDefinition, ...]:
    """Flag every constraint target so it leaves the free-variable layout.

    A derived target keeps its declared bounds for the runtime domain check but
    must not receive its own unit axis: its physical value comes from the rule
    expression. Marking here is also what shifts the checkpoint fingerprint, so
    an unconstrained compile stays byte-identical to the pre-feature layout.
    """
    if not constraint_rules:
        return definitions
    targets = {rule.target.parameter_name for rule in constraint_rules}
    known = {definition.name for definition in definitions}
    unknown = targets - known
    if unknown:
        raise ValueError(f"constraint target not in dataset: {sorted(unknown)}")
    return tuple(
        replace(definition, constrained=True) if definition.name in targets else definition
        for definition in definitions
    )


def _validate_local_constraints(
    constraint_rules: tuple[ConstraintRule, ...],
) -> str | None:
    """Keep the single-dataset compiler from erasing dataset identity.

    ``FitEvaluationContext`` indexes physical values by parameter name only.
    Cross-dataset expressions therefore need ``compile_joint_problem``, whose
    scatter layer owns the full ``ParameterReference`` namespace.  Accepting
    such a rule here would silently resolve a foreign reference against a local
    parameter with the same name.
    """
    if any(not isinstance(rule, ConstraintRule) for rule in constraint_rules):
        raise TypeError("constraint_rules must contain ConstraintRule values")
    dataset_ids = _constraint_dataset_ids(constraint_rules)
    local_dataset_ids = dataset_ids - {DRIFT_DATASET}
    if len(local_dataset_ids) > 1:
        raise ValueError(
            "cross-dataset constraints require the joint compiler; single-dataset compilation accepts local rules only"
        )
    targets = tuple(rule.target.parameter_name for rule in constraint_rules)
    if len(targets) != len(set(targets)):
        raise ValueError("constraint targets must be unique")
    cycle = constraint_cycle_path(constraint_rules)
    if cycle:
        raise ValueError(f"constraint rules form a dependency cycle: {' -> '.join(cycle)}")
    return next(iter(local_dataset_ids), next(iter(dataset_ids), None))


def _constraint_dataset_ids(
    constraint_rules: tuple[ConstraintRule, ...],
) -> frozenset[str]:
    return frozenset(
        dataset_id
        for rule in constraint_rules
        for dataset_id in (
            rule.target.dataset_id,
            *(reference.dataset_id for reference in _iter_references(rule.expression)),
        )
    )


def _validate_local_constraint_definitions(
    constraint_rules: tuple[ConstraintRule, ...],
    namespace: str | None,
    definitions: tuple[ParameterDefinition, ...],
) -> None:
    if namespace is None:
        return
    namespaces = _constraint_dataset_ids(constraint_rules) | frozenset((namespace,))
    by_reference = {
        ParameterReference(dataset_id, definition.name): definition
        for dataset_id in namespaces
        for definition in definitions
    }
    validate_constraint_stage_split(constraint_rules, by_reference)
    targets = {rule.target for rule in constraint_rules}
    if any(definition.prior is not None for reference, definition in by_reference.items() if reference in targets):
        raise ValueError("constraint target must not also have a parameter prior")


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
    lower = 1.05 * instrument.footprint_spill_angle_deg if instrument.footprint_mode == "geometry" else 0.0
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


def _validate_constraint_rule_values(rules: tuple[ConstraintRule, ...]) -> None:
    if any(not isinstance(rule, ConstraintRule) for rule in rules):
        raise TypeError("constraint_rules must contain ConstraintRule values")


def _compiled_constraint_rules(
    structure: StructureSpec,
    constraint_rules: tuple[ConstraintRule, ...],
) -> tuple[ConstraintRule, ...]:
    provided_rules = tuple(constraint_rules)
    _validate_constraint_rule_values(provided_rules)
    generated_sentinel = drift_constraint_rules(structure)
    if any(
        DRIFT_DATASET in _constraint_dataset_ids((rule,)) and rule not in generated_sentinel for rule in provided_rules
    ):
        raise ValueError(f"{DRIFT_DATASET!r} is reserved for generated drift rules")
    incoming_dataset_ids = _constraint_dataset_ids(provided_rules) - {DRIFT_DATASET}
    generated = generated_sentinel
    if len(incoming_dataset_ids) == 1:
        generated = rebind_drift_rules(generated, next(iter(incoming_dataset_ids)))
    # Regenerate drift rules on every compile so staged recompiles remain
    # idempotent, but only discard rules that exactly match a prior generated
    # rule. A user-authored repeat target must never disappear silently.
    generated_variants = set(generated_sentinel) | set(generated)
    incoming = tuple(rule for rule in provided_rules if rule not in generated_variants)
    conflicts = sorted(
        rule.target.parameter_name
        for rule in incoming
        if rule.target.parameter_name in {item.target.parameter_name for item in generated}
    )
    if conflicts:
        raise ValueError(f"user constraint target conflicts with generated drift rule: {conflicts}")
    return incoming + generated


def compile_fit_problem(
    data: PreparedData,
    structure: StructureSpec,
    instrument: InstrumentSpec,
    config: FitConfig,
    parameter_settings: tuple[ParameterSetting, ...] = (),
    constraint_rules: tuple[ConstraintRule, ...] = (),
) -> FitEvaluationContext:
    _validate_config(config)
    _validate_data_mode(data, instrument)
    _require_explicit_expert_density(structure, tuple(parameter_settings))
    rules = _compiled_constraint_rules(structure, tuple(constraint_rules))
    namespace = _validate_local_constraints(rules)
    definitions = apply_parameter_settings(
        default_parameter_definitions(data, structure, instrument, config),
        tuple(parameter_settings),
    )
    _validate_local_constraint_definitions(rules, namespace, definitions)
    definitions = _mark_constrained(definitions, rules)
    validate_compiled_definitions(definitions)
    validate_instrument_modes(definitions, instrument)
    validate_transition_modes(definitions, structure)
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
        constraint_rules=rules,
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
        problem.constraint_rules,
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
    if selected.constrained:
        raise ValueError(f"cannot fix constrained parameter: {parameter_name}")
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
        problem.constraint_rules,
    )
