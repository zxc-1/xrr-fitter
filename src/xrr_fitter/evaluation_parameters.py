"""Internal parameters implementation."""

from __future__ import annotations

from functools import partial
from math import isfinite

import numpy as np

from xrr_fitter.evaluation_geometry import (
    _active_upper_tangent as _active_upper_tangent,
)
from xrr_fitter.evaluation_geometry import (
    _add_latent_roughness_cap_jacobians as _add_latent_roughness_cap_jacobians,
)
from xrr_fitter.evaluation_geometry import (
    _add_latent_roughness_caps as _add_latent_roughness_caps,
)
from xrr_fitter.evaluation_geometry import (
    _allowed_missing_roughness_names as _allowed_missing_roughness_names,
)
from xrr_fitter.evaluation_geometry import (
    _definition_name as _definition_name,
)
from xrr_fitter.evaluation_geometry import (
    _expand_structure_with_jacobian as _expand_structure_with_jacobian,
)
from xrr_fitter.evaluation_geometry import (
    _fill_missing_roughness_caps as _fill_missing_roughness_caps,
)
from xrr_fitter.evaluation_geometry import (
    _fill_missing_roughness_jacobians as _fill_missing_roughness_jacobians,
)
from xrr_fitter.evaluation_geometry import (
    _gradient_slab_counts as _gradient_slab_counts,
)
from xrr_fitter.evaluation_geometry import (
    _interface_neighbor_indices as _interface_neighbor_indices,
)
from xrr_fitter.evaluation_geometry import (
    _interface_upper as _interface_upper,
)
from xrr_fitter.evaluation_geometry import (
    _is_public_interface as _is_public_interface,
)
from xrr_fitter.evaluation_geometry import (
    _is_roughness_definition as _is_roughness_definition,
)
from xrr_fitter.evaluation_geometry import (
    _latent_periodic_roughness_names as _latent_periodic_roughness_names,
)
from xrr_fitter.evaluation_geometry import (
    _record_active_upper as _record_active_upper,
)
from xrr_fitter.evaluation_geometry import (
    _roughness_definition_map as _roughness_definition_map,
)
from xrr_fitter.evaluation_geometry import (
    _roughness_definitions as _roughness_definitions,
)
from xrr_fitter.evaluation_geometry import (
    _roughness_dynamic_upper_jacobians as _roughness_dynamic_upper_jacobians,
)
from xrr_fitter.evaluation_geometry import (
    _roughness_dynamic_uppers as _roughness_dynamic_uppers,
)
from xrr_fitter.evaluation_geometry import (
    _roughness_geometry_context as _roughness_geometry_context,
)
from xrr_fitter.evaluation_geometry import (
    _zero_roughness_values as _zero_roughness_values,
)
from xrr_fitter.model.constraint_resolution import (
    ConstraintResolutionError,
    apply_constraint_values,
    constraint_value_jacobians,
    geometry_constraint_targets,
)
from xrr_fitter.model.fitting import FitEvaluationContext
from xrr_fitter.model.instrument import PhysicsDiagnostic
from xrr_fitter.model.parameters import (
    ParameterDefinition,
    _log_interval_width,
    physical_to_unit,
    unit_to_physical,
)


def _readonly_vector(value: object) -> np.ndarray:
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _validated_unit(problem: object, unit_vector: np.ndarray) -> np.ndarray:
    unit = np.asarray(unit_vector, dtype=float)
    valid = all(
        (
            unit.shape == (len(problem.variables),),
            np.all(np.isfinite(unit)),
        )
    )
    if not valid:
        raise ValueError("unit vector length does not match free variables")
    if np.any((unit < 0.0) | (unit > 1.0)):
        raise ValueError("unit vector lies outside [0,1]")
    return unit


def _unit_derivative(definition: ParameterDefinition, value: float) -> float:
    """Differentiate a nonrough physical value with respect to its unit axis.

    Static declared bounds define this derivative. Roughness is handled
    separately because its active upper can itself depend on thickness.
    """
    if definition.transform == "log":
        # Subtract logs instead of forming upper/lower first; very wide but
        # finite declarations can overflow that ratio even though both logs
        # and the physical derivative remain representable.
        try:
            log_span = _log_interval_width(definition.lower, definition.upper)
        except (ValueError, OverflowError) as error:
            raise FloatingPointError("nonfinite log parameter derivative") from error
        derivative = value * log_span
        if not np.isfinite(derivative):
            raise FloatingPointError("nonfinite log parameter derivative")
        return float(derivative)
    if definition.transform == "linear":
        span = definition.upper - definition.lower
        if not isfinite(span):
            raise FloatingPointError(f"linear parameter span is not representable: {definition.name}")
        return span
    raise ValueError(f"unknown transform: {definition.transform}")


def _roughness_value_jacobian(
    definition: ParameterDefinition,
    unit_value: float,
    unit_index: int,
    dynamic_upper: float,
    dynamic_jacobian: np.ndarray,
) -> np.ndarray:
    """Chain affine roughness decoding through a geometry-dependent upper.

    The dynamic tangent contributes only when geometry is tighter than the
    declared static bound. The coordinate's own axis always receives the active
    interval width directly.
    """
    upper = min(definition.upper, dynamic_upper)
    span = upper - definition.lower
    if not isfinite(span):
        raise FloatingPointError(f"roughness parameter span is not representable: {definition.name}")
    upper_jacobian = np.zeros_like(dynamic_jacobian) if definition.upper <= dynamic_upper else dynamic_jacobian
    jacobian = np.array(unit_value * upper_jacobian, dtype=float, copy=True)
    jacobian[unit_index] += span
    if np.any(~np.isfinite(jacobian)):
        raise FloatingPointError(f"nonfinite roughness parameter Jacobian: {definition.name}")
    return jacobian


def _initial_parameter_pair(
    definition: ParameterDefinition,
) -> tuple[str, float]:
    return definition.name, definition.initial


def _declared_values(problem: object) -> dict[str, float]:
    return dict(map(_initial_parameter_pair, problem.parameter_definitions))


def _zero_jacobian_pair(
    definition: ParameterDefinition,
    parameter_count: int,
) -> tuple[str, np.ndarray]:
    return definition.name, np.zeros(parameter_count, dtype=float)


def _decode_nonrough_values(
    problem: object,
    unit: np.ndarray,
    values: dict[str, float],
    *,
    continuous_only: bool,
) -> tuple[
    tuple[tuple[int, ParameterDefinition], ...],
    tuple[tuple[int, ParameterDefinition, float], ...],
]:
    """Decode geometry first and postpone bounds that depend on that geometry.

    All roughness coordinates must observe one identical reconstructed geometry
    snapshot. Continuous-only mode rejects integer topology coordinates before
    an analytic tangent can be requested for them.
    """
    postponed: list[tuple[int, ParameterDefinition]] = []
    decoded: list[tuple[int, ParameterDefinition, float]] = []
    for unit_index, variable in enumerate(problem.variables):
        definition = problem.parameter_definitions[variable.parameter_index]
        if (continuous_only, definition.integer) == (True, True):
            raise ValueError("analytic Jacobian requires continuous free parameters")
        if definition.transform == "roughness_fraction":
            postponed.append((unit_index, definition))
        else:
            value = unit_to_physical(definition, unit[unit_index])
            values[definition.name] = value
            decoded.append((unit_index, definition, value))
    return tuple(postponed), tuple(decoded)


def _apply_constraint_values(
    problem: FitEvaluationContext,
    values: dict[str, float],
    *,
    roughness: bool,
    dynamic_uppers: dict[str, float] | None = None,
    target_names: set[str] | None = None,
) -> None:
    try:
        apply_constraint_values(
            problem,
            values,
            roughness=roughness,
            dynamic_uppers=dynamic_uppers,
            target_names=target_names,
        )
    except ConstraintResolutionError as error:
        raise EvaluationConstraintError(error.reason) from error


def _apply_constraint_jacobians(
    problem: FitEvaluationContext,
    value_jacobians: dict[str, np.ndarray],
    values: dict[str, float],
    *,
    roughness: bool,
) -> None:
    try:
        constraint_value_jacobians(
            problem,
            value_jacobians,
            values,
            roughness=roughness,
        )
    except ConstraintResolutionError as error:
        raise EvaluationConstraintError(error.reason) from error


def values_and_jacobians(
    problem: FitEvaluationContext,
    unit_vector: np.ndarray,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Decode physical values and their analytic unit-coordinate tangents.

    Nonrough declarations are differentiated first, then their geometry builds
    the dynamic roughness values and upper tangents. Every returned declaration,
    including locked ones, owns a vector aligned with ``problem.variables``.
    """
    unit = _validated_unit(problem, unit_vector)
    parameter_count = len(problem.variables)
    values = _declared_values(problem)
    value_jacobians = dict(
        map(
            partial(_zero_jacobian_pair, parameter_count=parameter_count),
            problem.parameter_definitions,
        )
    )
    postponed, decoded = _decode_nonrough_values(
        problem,
        unit,
        values,
        continuous_only=True,
    )
    # Nonrough direct axes are independent at decode time; roughness is not.
    # Its upper tangent is chained only after provisional geometry is complete.
    for unit_index, definition, value in decoded:
        value_jacobians[definition.name][unit_index] = _unit_derivative(
            definition,
            value,
        )
    # 修正3 第一趟：非粗糙度约束目标须在几何重建（下方 dynamic upper）之前落值，
    # 其 Jacobian 也须在 dynamic-jacobian 合成之前链好。
    _apply_constraint_values(problem, values, roughness=False)
    _apply_constraint_jacobians(problem, value_jacobians, values, roughness=False)
    dynamic_values = _roughness_dynamic_uppers(problem, values)
    dynamic_jacobians = _roughness_dynamic_upper_jacobians(
        problem,
        values,
        value_jacobians,
    )
    for unit_index, definition in postponed:
        dynamic_upper = dynamic_values[definition.name]
        values[definition.name] = unit_to_physical(
            definition,
            unit[unit_index],
            dynamic_upper=dynamic_upper,
        )
        value_jacobians[definition.name] = _roughness_value_jacobian(
            definition,
            unit[unit_index],
            unit_index,
            dynamic_upper,
            dynamic_jacobians[definition.name][1],
        )
    # 修正3 第二趟：粗糙度约束目标在 postponed 粗糙度解算之后落值与链导，
    # 上界用刚算出的 dynamic_values 收紧。
    _apply_constraint_values(problem, values, roughness=True, dynamic_uppers=dynamic_values)
    _apply_constraint_jacobians(problem, value_jacobians, values, roughness=True)
    return values, value_jacobians


def values_by_name(
    problem: FitEvaluationContext,
    unit_vector: np.ndarray,
) -> dict[str, float]:
    """Decode nonrough coordinates before geometry-dependent roughness values.

    Locked initial values seed the complete mapping. Free coordinates overwrite
    them in stable variable order before one postponed roughness pass.
    """
    unit = _validated_unit(problem, unit_vector)
    values = _declared_values(problem)
    postponed, _decoded = _decode_nonrough_values(
        problem,
        unit,
        values,
        continuous_only=False,
    )
    # 修正3 第一趟：非粗糙度约束目标先落值，喂入下方几何重建。
    _apply_constraint_values(problem, values, roughness=False)
    dynamic = _roughness_dynamic_uppers(problem, values)
    for unit_index, definition in postponed:
        values[definition.name] = unit_to_physical(
            definition,
            unit[unit_index],
            dynamic_upper=dynamic[definition.name],
        )
    # 修正3 第二趟：粗糙度约束目标在 postponed 之后落值，上界用 dynamic 收紧。
    _apply_constraint_values(problem, values, roughness=True, dynamic_uppers=dynamic)
    return values


def roughness_dynamic_uppers(
    problem: FitEvaluationContext,
    unit_vector: np.ndarray,
) -> dict[str, float]:
    """Return geometry-dependent roughness caps for one local unit vector."""
    unit = _validated_unit(problem, unit_vector)
    values = _declared_values(problem)
    _decode_nonrough_values(
        problem,
        unit,
        values,
        continuous_only=False,
    )
    _apply_constraint_values(
        problem,
        values,
        roughness=False,
        target_names=geometry_constraint_targets(problem),
    )
    return _roughness_dynamic_uppers(problem, values)


def _physical_parameter_pair(
    definition: ParameterDefinition,
    physical_values: dict[str, float],
) -> tuple[str, float]:
    return definition.name, physical_values.get(definition.name, definition.initial)


def encode_physical_vector(
    problem: FitEvaluationContext,
    physical_values: dict[str, float],
) -> np.ndarray:
    """Encode a partial physical mapping with declaration defaults for omissions.

    Partial maps support stage transitions and candidate construction. Unknown
    extra keys are ignored, while the returned positional vector is an owned,
    read-only snapshot suitable for solver and checkpoint publication.
    """
    values = dict(
        map(
            partial(_physical_parameter_pair, physical_values=physical_values),
            problem.parameter_definitions,
        )
    )
    _apply_constraint_values(
        problem,
        values,
        roughness=False,
        target_names=geometry_constraint_targets(problem),
    )
    dynamic = _roughness_dynamic_uppers(problem, values)
    encoded: list[float] = []
    for variable in problem.variables:
        definition = problem.parameter_definitions[variable.parameter_index]
        upper = dynamic.get(definition.name) if definition.transform == "roughness_fraction" else None
        encoded.append(
            physical_to_unit(
                definition,
                values[definition.name],
                dynamic_upper=upper,
            )
        )
    return _readonly_vector(encoded)


class EvaluationConstraintError(Exception):
    """An expected physical-domain failure for one candidate vector.

    Diagnostics accumulated before the failure remain attached. This explicit
    type is the only boundary fit objectives may convert into an invalid result;
    unexpected implementation errors continue to propagate.
    """

    def __init__(
        self,
        reason: str,
        diagnostics: tuple[PhysicsDiagnostic, ...] = (),
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.diagnostics = tuple(diagnostics)
