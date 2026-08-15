"""Shared numerical evaluation primitives for fitting and analysis.

This module is the only boundary allowed to translate fitted coordinates into
physical structures and then chain model derivatives back to those coordinates.
Fit and analysis consume the same functions so likelihood, residual, and
Jacobian conventions cannot drift between the two domains.

The implementation preserves several frozen numerical contracts. Logarithmic
and affine transforms decode their endpoints exactly. Roughness is decoded only
after all geometry and uses the strictest expanded-interface cap. Periodic cells
share declared coordinates while retaining every repeated tangent occurrence.
Mixed K-alpha stacks are expanded separately by wavelength, then recombined in
the same order as the primal instrument model. The analytic path consumes the
physics tangents directly and never re-evaluates the primal candidate.

Expected physical-domain failures cross the boundary as
``EvaluationConstraintError``. Programming errors and unsupported layouts are
left visible to callers rather than converted into invalid candidates.

Parameter scalar transforms and immutable structure reconstruction are owned by
their lower-level model and physics modules. They are imported here as part of
this boundary so fit and analysis continue to consume one coordinate contract.
Neither domain reaches around evaluation to assemble candidate physics itself.

SciPy least-squares integration retains data residuals and the optional scale
prior as separate rows. Regional weights remain outside residual values and are
applied exactly once by the custom three-row loss. Invalid physical candidates
produce a fixed residual sentinel and a zero data Jacobian with stable axes;
unexpected callback errors remain visible.

The analytic optimizer path uses the same unit decoding and full model Jacobian
as ordinary evaluation. Only expected physical constraints, floating-point
failures, and the declared nonpositive-angle derivative boundary become a zero
Jacobian. Other ``ValueError`` instances identify unsupported layouts or defects
and are re-raised.

Analysis MCMC consumes a robust pseudo-posterior derived from the same soft-L1
data loss. Unit vectors outside the compiled box and expected physical failures
have negative-infinite density. The optional scale row contributes its explicit
Gaussian log prior and is not folded into regional data weighting.

These optimizer-facing functions accept an evaluator override only for narrow
unit testing of the boundary. Production callers use the module's primal and
analytic evaluators directly; no persistent callable collection is stored in a
problem, request, result, or checkpoint value.

SciPy requests residual and Jacobian through two callbacks, normally at the
same accepted parameter vector. ``least_squares_system`` obtains both from the
analytic traversal, whose primal return already contains the modeled values
needed for the residual. A one-entry callback cache binds that pair to the last
immutable copy of the unit vector. Trial vectors still trigger fresh physics,
while the immediately following Jacobian callback reuses only the matching
pair. Cache state is isolated per calling thread and is never global, shared
between fits, serialized, or used as a fallback for a different vector.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from functools import lru_cache, partial
from math import exp, isfinite, log
from threading import local

import numpy as np

from xrr_fitter.model.constraint_resolution import (
    ConstraintResolutionError,
    apply_constraint_values,
    constraint_value_jacobians,
    geometry_constraint_targets,
)
from xrr_fitter.model.fitting import FitEvaluationContext, ModelEvaluation
from xrr_fitter.model.instrument import PhysicsDiagnostic, resolution_to_sigma_q
from xrr_fitter.model.parameters import (
    ParameterDefinition,
    ParameterValue,
    PhysicalValueError,
    PriorSpec,
    physical_to_unit,
    unit_to_physical,
)
from xrr_fitter.model.structure import (
    SlabStack,
    StructureSpec,
)
from xrr_fitter.physics.derivatives import (
    parratt_reflectivity_jacobian,
    smear_with_widths_jacobian,
)
from xrr_fitter.physics.geometry import (
    GRADIENT_INTERNAL_INTERFACE,
    DifferentiableStack,
)
from xrr_fitter.physics.geometry import (
    GeometryExpansion as _GeometryExpansion,
)
from xrr_fitter.physics.geometry import (
    expand_geometry as _expand_geometry,
)
from xrr_fitter.physics.geometry import (
    expand_structure_with_jacobian as _expand_physical_structure_with_jacobian,
)
from xrr_fitter.physics.parratt import parratt_reflectivity
from xrr_fitter.physics.reflectivity import instrument_reflectivity, qz_from_theta_deg
from xrr_fitter.physics.stack import expand_structure, rebuild_structure


def _float_vectors(
    first: np.ndarray,
    second: np.ndarray,
    message: str,
) -> tuple[np.ndarray, np.ndarray]:
    left, right = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if left.shape != right.shape:
        raise ValueError(message)
    return left, right


def region_weights(labels: np.ndarray) -> np.ndarray:
    """Give every present region equal total squared residual weight.

    Region cardinalities may differ, so each point receives the square root of
    the inverse regional mass. This makes the squared weights of every region
    sum to the same value while preserving the full fitted-point count.
    """
    values = np.asarray(labels)
    if values.ndim != 1:
        raise ValueError("region labels must be a finite integer vector")
    valid = all(
        (
            np.all(np.isfinite(values)),
            np.all(values == np.floor(values)),
        )
    )
    if not valid:
        raise ValueError("region labels must be a finite integer vector")
    integer = values.astype(int)
    present = np.unique(integer)
    weights = np.empty(integer.size, dtype=float)
    # Squared point weights, not point weights themselves, define residual mass.
    # Multiplying by the global fitted count keeps the all-region sum unchanged.
    # The normalization therefore remains comparable to an ordinary mean loss.
    for label in present:
        selected = integer == label
        weights[selected] = np.sqrt(integer.size / (present.size * selected.sum()))
    return weights


def _validated_qz(qz_a_inv: np.ndarray) -> np.ndarray:
    qz = np.asarray(qz_a_inv, dtype=float)
    valid = all((qz.ndim == 1, qz.size > 0, np.all(np.isfinite(qz))))
    if not valid:
        raise ValueError("qz_a_inv must be a nonempty finite vector")
    return qz


def assign_fit_regions(
    qz_a_inv: np.ndarray,
    critical_candidates: tuple[tuple[float, float], ...] = (),
    bragg_candidates: tuple[tuple[float, float], ...] = (),
) -> np.ndarray:
    """Assign stable region labels from declared intervals or q quartiles.

    Explicit critical and Bragg intervals are consumed in caller order and do
    not overwrite points already claimed by an earlier interval. Unclaimed
    points receive deterministic equal-width labels. With no features, the
    frozen four-quartile fallback spans the complete q range directly.
    """
    qz = _validated_qz(qz_a_inv)
    if np.ptp(qz) == 0.0:
        return np.zeros(qz.size, dtype=int)
    intervals = tuple(critical_candidates) + tuple(bragg_candidates)
    if not intervals:
        edges = np.linspace(qz.min(), qz.max(), 5)
        return np.searchsorted(edges[1:-1], qz, side="right").astype(int)
    labels = np.full(qz.size, -1, dtype=int)
    next_label = 0
    # First-match ownership makes overlapping feature intervals deterministic.
    # Empty intervals do not consume a label, so region IDs remain contiguous.
    # Source q order is never sorted or otherwise changed during assignment.
    for lower, upper in intervals:
        valid = all((np.isfinite(lower), np.isfinite(upper), lower < upper))
        if not valid:
            raise ValueError("fit-region intervals must be finite and increasing")
        selected = (qz >= lower) & (qz <= upper) & (labels < 0)
        labels[selected] = next_label
        next_label += int(np.any(selected))
    remaining = labels < 0
    if np.any(remaining):
        subset = qz[remaining]
        edges = np.linspace(subset.min(), subset.max(), min(4, subset.size) + 1)
        labels[remaining] = next_label + np.searchsorted(edges[1:-1], subset, side="right").astype(int)
    return labels


def log_residuals(
    model: np.ndarray,
    observed: np.ndarray,
    r_floor: float,
) -> np.ndarray:
    """Return unweighted fitted residuals in base-10 reflectivity decades.

    The same positive floor is added to model and observation before either
    logarithm. Weighting remains outside this function so optimizers, profile
    likelihoods, and diagnostics can share one unweighted residual definition.
    """
    model, observed = _float_vectors(
        model,
        observed,
        "model and observed must have equal shapes",
    )
    if not all((isfinite(r_floor), r_floor > 0.0)):
        raise ValueError("r_floor must be positive and finite")
    if not all((np.all(np.isfinite(model)), np.all(np.isfinite(observed)))):
        raise ValueError("model and observed must be finite")
    if not all((np.all(model + r_floor > 0.0), np.all(observed + r_floor > 0.0))):
        raise ValueError("reflectivity plus floor must be positive")
    return np.log10(model + r_floor) - np.log10(observed + r_floor)


def robust_log_cost(
    delta: np.ndarray,
    weights: np.ndarray,
    c: float = 0.05,
) -> float:
    """Average the weighted soft-L1 loss with weights outside the loss.

    The frozen policy squares region weights after evaluating the soft-L1
    expression. Invalid numeric inputs produce infinity because this scalar is
    a search-policy value, not a structural decoding boundary.
    """
    delta, weights = _float_vectors(
        delta,
        weights,
        "delta and weights must have equal shapes",
    )
    valid = all(
        (
            np.isfinite(c),
            c > 0.0,
            delta.size != 0,
            np.all(np.isfinite(delta)),
            np.all(np.isfinite(weights)),
            np.all(weights > 0.0),
        )
    )
    if not valid:
        return float("inf")
    # This algebraic form is stable near zero and retains the frozen factor two.
    # Region weights sit outside the robust loss and are squared exactly once.
    loss = 2.0 * c**2 * (np.sqrt(1.0 + (delta / c) ** 2) - 1.0)
    return float(np.mean(weights**2 * loss))


def scale_prior_penalty(
    scale: float,
    scale_hat: float | None,
    tau_s: float,
    n: int,
) -> float:
    """Return the per-point weak prior on logarithmic instrument scale.

    A missing plateau estimate disables the prior exactly. Active evidence uses
    the fitted-point count in the denominator, keeping its contribution stable
    relative to the mean data loss as a fit mask changes size.
    """
    if scale_hat is None:
        return 0.0
    valid_type = (
        isinstance(n, (int, np.integer)),
        isinstance(n, bool),
    ) == (True, False)
    if not valid_type:
        raise ValueError("scale prior requires positive and finite scale, Ŝ, τ_S and N")
    values = np.asarray((scale, scale_hat, tau_s), dtype=float)
    valid = all((n > 0, np.all(np.isfinite(values) & (values > 0.0))))
    if not valid:
        raise ValueError("scale prior requires positive and finite scale, Ŝ, τ_S and N")
    # Work in the same base-10 coordinate as fitted log residuals.
    # Dividing by N makes this additive term compatible with a mean data loss.
    standardized = (np.log10(scale) - np.log10(scale_hat)) / tau_s
    return float(standardized**2 / n)


def _interface_neighbor_indices(
    interface_index: int,
    final_medium: int,
) -> tuple[int, ...]:
    """Return finite slab indices adjacent to one expanded interface.

    Incident and backing media are semi-infinite and do not constrain roughness.
    An interior interface may therefore expose one or two finite neighbors.
    """
    return tuple(
        range(
            max(1, interface_index),
            min(final_medium, interface_index + 2),
        )
    )


def _interface_upper(
    thickness_a: np.ndarray,
    neighbor_indices: tuple[int, ...],
) -> float:
    """Apply the strict 49-percent cap or the bare-interface 50 A limit.

    ``nextafter`` keeps the decoded upper strictly legal under stack expansion.
    The bare two-medium case has no finite geometry and uses its explicit cap.
    """
    if not neighbor_indices:
        return 50.0
    minimum = float(np.min(thickness_a[np.asarray(neighbor_indices, dtype=int)]))
    return float(np.nextafter(0.49 * minimum, 0.0))


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


def _is_roughness_definition(definition: ParameterDefinition) -> bool:
    return definition.transform == "roughness_fraction"


def _definition_name(definition: ParameterDefinition) -> str:
    return definition.name


def _zero_roughness_values(problem: object) -> dict[str, float]:
    """Neutralize every declared roughness while retaining expanded topology."""
    definitions = filter(_is_roughness_definition, problem.parameter_definitions)
    return dict.fromkeys(map(_definition_name, definitions), 0.0)


def _is_public_interface(item: tuple[int, str]) -> bool:
    return item[1] != GRADIENT_INTERNAL_INTERFACE


def _roughness_dynamic_uppers(
    problem: object,
    nonrough_values: dict[str, float],
) -> dict[str, float]:
    """Tighten each named roughness against all repeated interface instances.

    Roughness is zeroed only for this provisional topology expansion. Candidate
    thickness remains active, and repeated periodic names reduce by the strictest
    local neighborhood rather than whichever occurrence is visited last.
    """
    zero_roughness = _zero_roughness_values(problem)
    # Right-hand union deliberately overrides any supplied roughness coordinate.
    # Geometry remains candidate-specific because all thickness values are kept.
    provisional_values = nonrough_values | zero_roughness
    provisional = rebuild_structure(
        problem.structure,
        provisional_values,
    )
    geometry = _expand_geometry(provisional)
    dynamic: dict[str, float] = {}
    final_medium = geometry.thickness_a.size - 1
    # Repeated source names reduce by minimum, independent of occurrence order.
    # Internal gradient interfaces never receive a public parameter definition.
    for interface, name in filter(_is_public_interface, enumerate(geometry.interface_names)):
        neighbors = _interface_neighbor_indices(interface, final_medium)
        upper = _interface_upper(geometry.thickness_a, neighbors)
        dynamic[name] = min(dynamic.get(name, np.inf), upper)
    return dynamic


def _expand_structure_with_jacobian(
    problem: object,
    values: dict[str, float],
    value_jacobians: dict[str, np.ndarray],
    wavelength_a: float,
) -> DifferentiableStack:
    """Adapt the typed evaluation context to the physical stack expander."""
    return _expand_physical_structure_with_jacobian(
        problem.structure,
        values,
        value_jacobians,
        wavelength_a,
        len(problem.variables),
    )


def _roughness_geometry_context(
    problem: object,
    nonrough_values: dict[str, float],
    value_jacobians: dict[str, np.ndarray],
) -> _GeometryExpansion:
    """Build one zero-roughness geometry expansion with thickness tangents."""
    zero_roughness = _zero_roughness_values(problem)
    provisional_values = nonrough_values | zero_roughness
    provisional = rebuild_structure(
        problem.structure,
        provisional_values,
    )
    return _expand_geometry(
        provisional,
        parameter_count=len(problem.variables),
        value_jacobians=value_jacobians,
    )


def _active_upper_tangent(
    provisional: _GeometryExpansion,
    neighbors: tuple[int, ...],
    parameter_count: int,
) -> tuple[float, np.ndarray]:
    """Return the strict upper and symmetric tangent at an exact minimum tie.

    The cap follows the thinnest finite adjacent slab. At an exact equality,
    symmetric finite differences select the arithmetic mean subgradient across
    all active neighbors; near ties retain the single active derivative.
    """
    if not neighbors:
        return 50.0, np.zeros(parameter_count, dtype=float)
    indices = np.asarray(neighbors, dtype=int)
    assert provisional.thickness_jacobian is not None
    thickness = provisional.thickness_a[indices]
    minimum = float(np.min(thickness))
    active = indices[thickness == minimum]
    # Exact equality intentionally selects the symmetric minimum subgradient.
    # No tolerance widens this nonsmooth branch to merely nearby thicknesses.
    tangent = 0.49 * np.mean(
        provisional.thickness_jacobian[active],
        axis=0,
    )
    return float(np.nextafter(0.49 * minimum, 0.0)), tangent


def _record_active_upper(
    dynamic: dict[str, tuple[float, np.ndarray]],
    tie_counts: dict[str, int],
    name: str,
    upper: float,
    tangent: np.ndarray,
) -> None:
    """Reduce repeated parameter names by minimum, averaging exact ties.

    A lower occurrence atomically replaces value and tangent. Equal active
    occurrences update an incremental arithmetic mean, making periodic sharing
    independent of repetition count while preserving the frozen traversal.
    """
    previous = dynamic.get(name, (np.inf, tangent))
    # Replacement publishes value and tangent atomically at a stricter cap.
    # Equal caps retain all active occurrences through incremental averaging.
    if upper < previous[0]:
        dynamic[name] = (upper, tangent.copy())
        tie_counts[name] = 1
        return
    if upper == previous[0]:
        count = tie_counts[name]
        dynamic[name] = (
            upper,
            (count * previous[1] + tangent) / (count + 1),
        )
        tie_counts[name] = count + 1


def _roughness_dynamic_upper_jacobians(
    problem: object,
    nonrough_values: dict[str, float],
    value_jacobians: dict[str, np.ndarray],
) -> dict[str, tuple[float, np.ndarray]]:
    """Return candidate-specific roughness bounds and thickness tangents.

    Internal gradient interfaces are skipped because they have no public
    coordinate. Locked definitions remain represented so encoding and analytic
    decoding observe the same candidate geometry.
    """
    provisional = _roughness_geometry_context(
        problem,
        nonrough_values,
        value_jacobians,
    )
    dynamic: dict[str, tuple[float, np.ndarray]] = {}
    tie_counts: dict[str, int] = {}
    parameter_count = len(problem.variables)
    final_medium = provisional.thickness_a.size - 1
    for interface, name in filter(
        _is_public_interface,
        enumerate(provisional.interface_names),
    ):
        upper, tangent = _active_upper_tangent(
            provisional,
            _interface_neighbor_indices(interface, final_medium),
            parameter_count,
        )
        _record_active_upper(dynamic, tie_counts, name, upper, tangent)
    return dynamic


def _unit_derivative(definition: ParameterDefinition, value: float) -> float:
    """Differentiate a nonrough physical value with respect to its unit axis.

    Static declared bounds define this derivative. Roughness is handled
    separately because its active upper can itself depend on thickness.
    """
    if definition.transform == "log":
        return value * np.log(definition.upper / definition.lower)
    if definition.transform == "linear":
        return definition.upper - definition.lower
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
    upper_jacobian = np.zeros_like(dynamic_jacobian) if definition.upper <= dynamic_upper else dynamic_jacobian
    jacobian = np.array(unit_value * upper_jacobian, dtype=float, copy=True)
    jacobian[unit_index] += upper - definition.lower
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


def _primary_wavelength(problem: object) -> float:
    beam = problem.data.beam
    # Monochromatic data names its only wavelength directly; mixed data names
    # the primary channel explicitly and reserves effective wavelength for q.
    return {
        "monochromatic": beam.wavelength_a,
        "mixed_kalpha": beam.wavelength_1_a,
    }[beam.kind]


def _expanded_stacks(
    problem: object,
    rebuilt: StructureSpec,
) -> tuple[float, SlabStack, SlabStack | None]:
    """Expand primary and optional secondary stacks at their true wavelengths.

    Material SLD is wavelength dependent, so a mixed beam cannot reuse one stack
    even though both expansions share the same rebuilt physical geometry. Any
    drifted periodic block is already baked into per-copy layers by
    ``rebuild_structure``, so expansion is a pure function of the rebuilt geometry.
    """
    primary_wavelength = _primary_wavelength(problem)
    primary = expand_structure(rebuilt, primary_wavelength)
    secondary = (
        expand_structure(rebuilt, problem.data.beam.wavelength_2_a)
        if problem.data.beam.kind == "mixed_kalpha"
        else None
    )
    return primary_wavelength, primary, secondary


def _angle_layout(
    problem: object,
    values: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recompute incident angle, model rows, source indices, and effective q.

    The fitted angle offset acts on source two-theta values rather than shifting
    a previously prepared q array. Nonpositive angles remain outside the compact
    physics model and keep NaN q entries in full source order.
    """
    theta = problem.data.two_theta_deg / 2.0 + values["instrument.angle_offset_deg"]
    model_mask = np.isfinite(theta) & (theta > 0.0)
    model_indices = np.flatnonzero(model_mask)
    # Full source layout is retained for result publication and diagnostic rows.
    # Only the compact positive-angle subset enters trigonometric physics.
    qz = np.full(problem.data.qz_a_inv.shape, np.nan, dtype=float)
    qz[model_mask] = 4.0 * np.pi * np.sin(np.deg2rad(theta[model_mask])) / problem.data.beam.effective_wavelength_a
    return theta, model_mask, model_indices, qz


def _point_resolution_for_wavelength(
    problem: object,
    angle_offset_deg: float,
    wavelength_a: float,
) -> np.ndarray | None:
    """Resolve an imported per-row width in the active wavelength's q domain.

    Q-domain columns are already prepared and wavelength independent. Angular
    columns must be converted after applying the candidate angle offset. Theta
    convolution owns one scalar instrument width and forbids per-row columns.
    """
    data = problem.data
    if problem.instrument.resolution_domain == "theta":
        if data.resolution_raw is not None:
            raise ValueError("per-point resolution columns are unsupported in theta-domain mode")
        return None
    if data.resolution_raw is None:
        return None
    kind = data.column_mapping.resolution_kind
    if kind in {"sigma_q_a_inv", "fwhm_q_a_inv"}:
        if data.sigma_q_a_inv is None:
            raise ValueError("q-domain point resolution was not prepared")
        return data.sigma_q_a_inv
    if kind not in {"sigma_two_theta_deg", "fwhm_two_theta_deg"}:
        raise ValueError("unsupported point-resolution mapping")
    return resolution_to_sigma_q(
        data.two_theta_deg,
        data.resolution_raw,
        kind,
        wavelength_a,
        angle_offset_deg,
    )


def _parameter_values(
    problem: object,
    values: dict[str, float],
) -> tuple[ParameterValue, ...]:
    def parameter_value(definition: ParameterDefinition) -> ParameterValue:
        return ParameterValue(
            definition.name,
            values[definition.name],
            definition.lower,
            definition.upper,
        )

    return tuple(map(parameter_value, problem.parameter_definitions))


def _parameter_pair(parameter: ParameterValue) -> tuple[str, float]:
    return parameter.name, parameter.value


def _instrument_values(parameters: tuple[ParameterValue, ...]) -> dict[str, float]:
    """Project complete parameter values into the physics instrument keywords.

    The explicit key list documents the closed primal instrument contract and
    prevents structure coordinates from leaking into the physics call.
    """
    values = dict(map(_parameter_pair, parameters))
    return {
        "scale": values["instrument.scale"],
        "background": values["instrument.background"],
        "linear_background_per_a_inv": values["instrument.linear_background_per_a_inv"],
        "powerlaw_background_amplitude": values["instrument.powerlaw_background_amplitude"],
        "powerlaw_background_exponent": values["instrument.powerlaw_background_exponent"],
        "footprint_spill_angle_deg": values["instrument.footprint_spill_angle_deg"],
        "relative_sigma": values["instrument.relative_sigma"],
        "absolute_sigma_a_inv": values["instrument.absolute_sigma_a_inv"],
        "sigma_theta_deg": values["instrument.sigma_theta_deg"],
    }


def _nevot_croce_affected(stack: SlabStack) -> tuple[int, ...]:
    """Locate interfaces beyond one third of their thinnest finite neighbor.

    Semi-infinite outer media are represented by infinite comparison values.
    The warning threshold is diagnostic only and never clamps reflectivity or
    changes candidate validity.
    """
    left, right = stack.thickness_a[:-1].copy(), stack.thickness_a[1:].copy()
    left[0], right[-1] = np.inf, np.inf
    minimum = np.minimum(left, right)
    affected = np.isfinite(minimum) & (stack.roughness_a > minimum / 3.0)
    return tuple(np.flatnonzero(affected).tolist())


def _append_nevot_croce_diagnostic(
    stack: SlabStack,
    diagnostics: list[PhysicsDiagnostic],
) -> None:
    """Record all Nevot-Croce applicability interfaces as one diagnostic.

    Expanded interface indices are retained in the message because periodic and
    gradient structures may map several physical interfaces to one source value.
    """
    affected = _nevot_croce_affected(stack)
    if affected:
        diagnostics.append(
            PhysicsDiagnostic(
                "nevot_croce_applicability_exceeded",
                "粗糙度超出 Nevot–Croce 适用范围；受影响界面索引：" + ",".join(map(str, affected)),
            )
        )


def _append_ideal_reflectivity_diagnostic(
    stack: SlabStack,
    wavelength_a: float,
    theta: np.ndarray,
    model_indices: np.ndarray,
    channel: str,
    diagnostics: list[PhysicsDiagnostic],
) -> None:
    """Record full source rows whose unsmeared ideal reflectivity exceeds one.

    This check intentionally precedes resolution, scale, footprint, and
    background so it diagnoses only ideal stack physics. The fixed tolerance
    excludes roundoff at the physical ceiling.
    """
    qz = 4.0 * np.pi * np.sin(np.deg2rad(theta)) / wavelength_a
    # Evaluate raw stack reflectivity before instrument corrections by design.
    # ``model_indices`` restores original rows after the positive-angle mask.
    above = parratt_reflectivity(qz, stack) > 1.0 + 1e-6
    if np.any(above):
        diagnostics.append(
            PhysicsDiagnostic(
                "ideal_reflectivity_above_one",
                f"{channel} 理想反射率超过 1+1e-6",
                tuple(map(int, model_indices[above])),
            )
        )


def _append_stack_diagnostics(
    primary_stack: SlabStack,
    secondary_stack: SlabStack | None,
    primary_wavelength: float,
    secondary_wavelength: float,
    theta: np.ndarray,
    model_indices: np.ndarray,
    diagnostics: list[PhysicsDiagnostic],
) -> None:
    """Append primary applicability and per-wavelength ideal diagnostics.

    Nevot-Croce geometry is wavelength independent and is reported once from
    the primary stack. Ideal reflectivity is evaluated separately for every
    physical beam channel.
    """
    _append_nevot_croce_diagnostic(primary_stack, diagnostics)
    channels = ((primary_stack, primary_wavelength, "Kα₁/单色"),)
    if secondary_stack is not None:
        channels += ((secondary_stack, secondary_wavelength, "Kα₂"),)
    for stack, wavelength, channel in channels:
        _append_ideal_reflectivity_diagnostic(
            stack,
            wavelength,
            theta,
            model_indices,
            channel,
            diagnostics,
        )


def _is_integer_index(index: object) -> bool:
    return (
        isinstance(index, (int, np.integer)),
        isinstance(index, bool),
    ) == (True, False)


def _index_in_range(index: int | np.integer, size: int) -> bool:
    return 0 <= index < size


def _record_physics_diagnostic(
    diagnostics: list[PhysicsDiagnostic],
    model_indices: np.ndarray,
    diagnostic: PhysicsDiagnostic,
) -> None:
    """Map compact physics-row diagnostics back to complete source indices.

    Boolean, noninteger, negative, and out-of-range indices are rejected before
    indexing. Empty diagnostic index sets remain valid and preserve metadata.
    """
    indices = diagnostic.point_indices
    if not all(map(_is_integer_index, indices)):
        raise ValueError("physics diagnostic point index is out of range")
    valid = all(map(partial(_index_in_range, size=model_indices.size), indices))
    if not valid:
        raise ValueError("physics diagnostic point index is out of range")
    diagnostics.append(
        replace(
            diagnostic,
            point_indices=tuple(
                map(
                    int,
                    model_indices[np.asarray(diagnostic.point_indices, dtype=int)],
                )
            ),
        )
    )


def _modeled_reflectivity(
    problem: object,
    theta: np.ndarray,
    model_mask: np.ndarray,
    model_indices: np.ndarray,
    primary_stack: SlabStack,
    secondary_stack: SlabStack | None,
    primary_sigma: np.ndarray | None,
    secondary_sigma: np.ndarray | None,
    instrument: dict[str, float],
    diagnostics: list[PhysicsDiagnostic],
) -> np.ndarray:
    """Evaluate compact positive-angle rows through the primal instrument model.

    Point widths are sliced by the same mask as theta. The callback remaps any
    adaptive-resolution diagnostic before it joins the candidate evidence.
    """
    callback = partial(_record_physics_diagnostic, diagnostics, model_indices)
    # Adaptive quadrature diagnostics are already captured structurally by the
    # callback. Suppressing the repeated warning in the optimizer's inner loop
    # avoids console I/O for every trial while direct physics callers retain the
    # public warning behavior.
    return instrument_reflectivity(
        theta[model_mask],
        primary_stack,
        problem.data.beam,
        secondary_stack=secondary_stack,
        resolution_domain=problem.instrument.resolution_domain,
        sigma_q_a_inv=_masked_optional(primary_sigma, model_mask),
        secondary_sigma_q_a_inv=_masked_optional(secondary_sigma, model_mask),
        diagnostic_callback=callback,
        emit_warning=False,
        **instrument,
    )


def _model_evaluation(
    problem: object,
    unit_vector: np.ndarray,
) -> ModelEvaluation:
    """Assemble one complete immutable primal candidate evaluation.

    Coordinate, reconstruction, expansion, and physics-domain numeric failures
    become the documented constraint error. Full source arrays retain NaN only
    outside modeled positive-angle rows; fitted rows must be finite. Objective
    assembly uses unweighted residuals, region weights, and the optional scale
    prior exactly once.
    """
    try:
        values = values_by_name(problem, unit_vector)
        rebuilt = rebuild_structure(problem.structure, values)
        primary_wavelength, primary_stack, secondary_stack = _expanded_stacks(
            problem,
            rebuilt,
        )
    except (ValueError, FloatingPointError, OverflowError) as error:
        raise EvaluationConstraintError(f"constraint_violation:{type(error).__name__}") from error
    theta, model_mask, model_indices, qz = _angle_layout(problem, values)
    if np.any(problem.data.fit_mask & ~model_mask):
        raise EvaluationConstraintError("nonpositive_fitted_incident_angle")
    angle_offset = values["instrument.angle_offset_deg"]
    primary_sigma = _point_resolution_for_wavelength(
        problem,
        angle_offset,
        primary_wavelength,
    )
    secondary_sigma = (
        _point_resolution_for_wavelength(
            problem,
            angle_offset,
            problem.data.beam.wavelength_2_a,
        )
        if problem.data.beam.kind == "mixed_kalpha"
        else None
    )
    parameters = _parameter_values(problem, values)
    diagnostics: list[PhysicsDiagnostic] = []
    _append_stack_diagnostics(
        primary_stack,
        secondary_stack,
        primary_wavelength,
        problem.data.beam.wavelength_2_a,
        theta[model_mask],
        model_indices,
        diagnostics,
    )
    try:
        modeled = _modeled_reflectivity(
            problem,
            theta,
            model_mask,
            model_indices,
            primary_stack,
            secondary_stack,
            primary_sigma,
            secondary_sigma,
            _instrument_values(parameters),
            diagnostics,
        )
    except (ValueError, FloatingPointError, OverflowError) as error:
        raise EvaluationConstraintError(
            f"constraint_violation:{type(error).__name__}",
            tuple(diagnostics),
        ) from error
    model = np.full(problem.data.qz_a_inv.shape, np.nan, dtype=float)
    model[model_mask] = modeled
    fit_mask = problem.data.fit_mask
    # Residuals remain unweighted; weights participate only in search objective
    # and in the separately published weighted residual vector.
    residual = log_residuals(
        model[fit_mask],
        problem.data.intensity_normalized[fit_mask],
        problem.data.r_floor,
    )
    weighted = problem.weights[fit_mask] * residual
    objective = robust_log_cost(
        residual,
        problem.weights[fit_mask],
        problem.config.c_decades,
    ) + scale_prior_penalty(
        values["instrument.scale"],
        problem.scale_prior_center,
        problem.scale_prior_tau_decades,
        int(np.count_nonzero(fit_mask)),
    )
    # All fitted and modeled values must be finite before immutable publication.
    # NaN outside the model mask is intentional and excluded from this check.
    finite_values = np.concatenate((qz[model_mask], model[model_mask], residual, weighted))
    if not all((np.all(np.isfinite(finite_values)), np.isfinite(objective))):
        raise EvaluationConstraintError("nonfinite_model", tuple(diagnostics))
    return ModelEvaluation(
        valid=True,
        reason="ok",
        parameters=parameters,
        qz_a_inv=qz,
        model_normalized=model,
        fit_log_residuals_decades=residual,
        fit_weighted_residuals=weighted,
        objective=objective,
        expanded_stack=primary_stack,
        diagnostics=tuple(diagnostics),
    )


def evaluate_model(
    problem: FitEvaluationContext,
    unit_vector: np.ndarray,
) -> ModelEvaluation:
    """Evaluate one candidate through the single shared numerical chain.

    This public pure function intentionally exposes no alternate engine or
    injectable operation bundle; fit and analysis share this exact boundary.
    """
    return _model_evaluation(problem, unit_vector)


def expanded_structure_jacobian(
    problem: FitEvaluationContext,
    unit_vector: np.ndarray,
) -> DifferentiableStack:
    """Expand the primary-wavelength stack and its analytic unit tangents.

    This is a structural derivative view only. It does not evaluate reflectivity
    and therefore cannot diverge from or duplicate the primal model result.
    """
    values, value_jacobians = values_and_jacobians(problem, unit_vector)
    return _expand_structure_with_jacobian(
        problem,
        values,
        value_jacobians,
        _primary_wavelength(problem),
    )


def _qz_and_jacobian(
    theta_deg: np.ndarray,
    theta_jacobian: np.ndarray,
    wavelength_a: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert incident angle and its forward tangents to wavelength-specific q.

    Degree-to-radian scaling participates in the derivative explicitly. The
    trailing parameter axis is broadcast against the leading sample shape and
    remains aligned with the flattened physics query.
    """
    theta_rad = np.deg2rad(theta_deg)
    scale = 4.0 * np.pi / wavelength_a
    qz = qz_from_theta_deg(theta_deg, wavelength_a)
    # The derivative of sine is evaluated in radians while the source tangent
    # remains expressed per degree, hence the explicit pi/180 factor.
    qz_jacobian = scale * np.cos(theta_rad)[:, None] * (np.pi / 180.0) * theta_jacobian
    return qz, qz_jacobian


def _point_resolution_with_jacobian(
    problem: object,
    angle_offset_deg: float,
    angle_jacobian: np.ndarray,
    wavelength_a: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return per-row q widths and their angle-offset forward tangents.

    Imported q widths have zero candidate tangent. Angular widths are first
    normalized from FWHM when needed, then differentiated through the absolute
    cosine conversion used by the primal resolution preparation.
    """
    point_resolution = _point_resolution_for_wavelength(
        problem,
        angle_offset_deg,
        wavelength_a,
    )
    if point_resolution is None:
        return None, None
    point_jacobian = np.zeros(
        point_resolution.shape + (len(problem.variables),),
        dtype=float,
    )
    kind = problem.data.column_mapping.resolution_kind
    if kind in {"sigma_q_a_inv", "fwhm_q_a_inv"}:
        return point_resolution, point_jacobian
    raw = np.asarray(problem.data.resolution_raw, dtype=float)
    if kind.startswith("fwhm"):
        raw = raw / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_theta_rad = np.deg2rad(raw / 2.0)
    theta_rad = np.deg2rad(problem.data.two_theta_deg / 2.0 + angle_offset_deg)
    cosine = np.cos(theta_rad)
    # The primal converter uses ``abs(cos(theta))``; its local derivative is
    # represented by sign(cosine) away from the exact cusp.
    derivative = -(4.0 * np.pi / wavelength_a) * np.sign(cosine) * np.sin(theta_rad) * (np.pi / 180.0) * sigma_theta_rad
    return point_resolution, derivative[:, None] * angle_jacobian[None, :]


def _value_or_zeros(value: np.ndarray | None, template: np.ndarray) -> np.ndarray:
    """Materialize an absent optional tangent with a shape-matched zero array.

    Callers may then use one algebraic expression for imported and absent point
    resolution without introducing a second derivative path.
    """
    return np.zeros_like(template) if value is None else value


def _resolution_width_jacobian(
    qz: np.ndarray,
    qz_jacobian: np.ndarray,
    relative_sigma: float,
    relative_sigma_jacobian: np.ndarray,
    absolute_sigma: float,
    absolute_sigma_jacobian: np.ndarray,
    point_sigma: np.ndarray | None,
    point_sigma_jacobian: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine relative, absolute, and imported q-width terms analytically.

    The primal width is the square root of summed variances. Its tangent uses
    the same terms before dividing only positive widths; an exact zero width
    retains an exact zero derivative and bypasses division.
    """
    point = _value_or_zeros(point_sigma, qz)
    point_jacobian = _value_or_zeros(point_sigma_jacobian, qz_jacobian)
    relative_term = relative_sigma * qz
    relative_jacobian = qz[:, None] * relative_sigma_jacobian[None, :] + relative_sigma * qz_jacobian
    variance = relative_term**2 + absolute_sigma**2 + point**2
    variance_jacobian = (
        2.0 * relative_term[:, None] * relative_jacobian
        + 2.0 * absolute_sigma * absolute_sigma_jacobian[None, :]
        + 2.0 * point[:, None] * point_jacobian
    )
    # Zero total variance is a legitimate disabled-resolution state.
    # Its tangent is defined as zero instead of evaluating a 0/0 quotient.
    widths = np.sqrt(variance)
    width_jacobian = np.zeros_like(variance_jacobian)
    positive = widths > 0.0
    width_jacobian[positive] = variance_jacobian[positive] / (2.0 * widths[positive, None])
    return widths, width_jacobian


def _reflectivity_with_jacobian(
    differentiable: DifferentiableStack,
    query_qz: np.ndarray,
    query_qz_jacobian: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Bind a differentiable stack to the low-level Parratt tangent kernel.

    Values and every physical tangent array are forwarded together so periodic
    acceleration and branch conventions remain owned by physics.
    """
    return parratt_reflectivity_jacobian(
        query_qz,
        differentiable.stack,
        query_qz_jacobian,
        differentiable.thickness_jacobian,
        differentiable.sld_jacobian,
        differentiable.roughness_jacobian,
    )


def _theta_reflectivity_with_jacobian(
    differentiable: DifferentiableStack,
    wavelength_a: float,
    query_theta: np.ndarray,
    query_theta_jacobian: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate arbitrary-shaped theta quadrature nodes and restore their shape.

    Smearing supplies a sample-by-node query grid. Flattening keeps Parratt's
    vector interface simple, while reshape restores both node axes and the
    trailing parameter dimension without reordering observations.
    """
    flat_theta = query_theta.ravel()
    flat_jacobian = query_theta_jacobian.reshape(
        -1,
        query_theta_jacobian.shape[-1],
    )
    qz, qz_jacobian = _qz_and_jacobian(
        flat_theta,
        flat_jacobian,
        wavelength_a,
    )
    result, jacobian = _reflectivity_with_jacobian(
        differentiable,
        qz,
        qz_jacobian,
    )
    return (
        result.reshape(query_theta.shape),
        jacobian.reshape(query_theta.shape + (flat_jacobian.shape[1],)),
    )


def _primal_theta_reflectivity(
    stack: SlabStack,
    wavelength_a: float,
    query_theta: np.ndarray,
) -> np.ndarray:
    """Provide the exact primal callback paired with theta-domain tangents.

    Adaptive smearing compares quadrature orders using this value-only function;
    it must therefore share q conversion and Parratt ordering with the tangent
    callback rather than deriving values from Jacobian output.
    """
    query_qz = 4.0 * np.pi * np.sin(np.deg2rad(query_theta)) / wavelength_a
    return parratt_reflectivity(query_qz, stack)


def _single_wavelength_smeared_jacobian(
    problem: object,
    theta_deg: np.ndarray,
    theta_jacobian: np.ndarray,
    differentiable: DifferentiableStack,
    wavelength_a: float,
    values: dict[str, float],
    value_jacobians: dict[str, np.ndarray],
    point_sigma: np.ndarray | None,
    point_sigma_jacobian: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Differentiate adaptive resolution smearing for one physical wavelength.

    Theta-domain resolution smears incident angles with the scalar instrument
    width. Q-domain resolution first converts candidate angles, combines all
    width sources, and smears q. Both paths pass paired primal callbacks so
    convergence decisions match ordinary model evaluation.
    """
    if problem.instrument.resolution_domain == "theta":
        sigma = values["instrument.sigma_theta_deg"]
        widths = np.full(theta_deg.shape, sigma, dtype=float)
        width_jacobian = np.broadcast_to(
            value_jacobians["instrument.sigma_theta_deg"],
            theta_jacobian.shape,
        ).copy()
        return smear_with_widths_jacobian(
            theta_deg,
            theta_jacobian,
            widths,
            width_jacobian,
            partial(
                _theta_reflectivity_with_jacobian,
                differentiable,
                wavelength_a,
            ),
            primal_function=partial(
                _primal_theta_reflectivity,
                differentiable.stack,
                wavelength_a,
            ),
        )
    if problem.instrument.resolution_domain != "q":
        raise ValueError("resolution_domain must be q or theta")
    qz, qz_jacobian = _qz_and_jacobian(
        theta_deg,
        theta_jacobian,
        wavelength_a,
    )
    widths, width_jacobian = _resolution_width_jacobian(
        qz,
        qz_jacobian,
        values["instrument.relative_sigma"],
        value_jacobians["instrument.relative_sigma"],
        values["instrument.absolute_sigma_a_inv"],
        value_jacobians["instrument.absolute_sigma_a_inv"],
        point_sigma,
        point_sigma_jacobian,
    )
    return smear_with_widths_jacobian(
        qz,
        qz_jacobian,
        widths,
        width_jacobian,
        partial(_reflectivity_with_jacobian, differentiable),
        primal_function=partial(parratt_reflectivity, stack=differentiable.stack),
    )


def _smeared_beam_jacobian(
    problem: object,
    theta_deg: np.ndarray,
    theta_jacobian: np.ndarray,
    values: dict[str, float],
    value_jacobians: dict[str, np.ndarray],
    primary_stack: DifferentiableStack,
    secondary_stack: DifferentiableStack | None,
    primary_point_sigma: np.ndarray | None,
    primary_point_sigma_jacobian: np.ndarray | None,
    secondary_point_sigma: np.ndarray | None,
    secondary_point_sigma_jacobian: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine primary and optional K-alpha-2 smeared values and tangents.

    Each wavelength owns its expanded SLD stack and point-width conversion.
    Mixed-beam fractions are fixed beam metadata, so the weighted derivative is
    the same linear combination as the primal reflectivity.
    """
    beam = problem.data.beam
    smeared, smeared_jacobian = _single_wavelength_smeared_jacobian(
        problem,
        theta_deg,
        theta_jacobian,
        primary_stack,
        _primary_wavelength(problem),
        values,
        value_jacobians,
        primary_point_sigma,
        primary_point_sigma_jacobian,
    )
    if beam.kind != "mixed_kalpha":
        return smeared, smeared_jacobian
    if secondary_stack is None:
        raise ValueError("secondary stack is required for mixed_kalpha")
    # Beam fractions are fixed acquisition metadata rather than fit coordinates.
    # Apply the identical linear mixture to primal values and every tangent row.
    secondary, secondary_jacobian = _single_wavelength_smeared_jacobian(
        problem,
        theta_deg,
        theta_jacobian,
        secondary_stack,
        beam.wavelength_2_a,
        values,
        value_jacobians,
        secondary_point_sigma,
        secondary_point_sigma_jacobian,
    )
    ratio = beam.intensity_ratio_21
    return (
        (smeared + ratio * secondary) / (1.0 + ratio),
        (smeared_jacobian + ratio * secondary_jacobian) / (1.0 + ratio),
    )


def _footprint_jacobian(
    theta_deg: np.ndarray,
    theta_jacobian: np.ndarray,
    spill_angle_deg: float,
    spill_angle_jacobian: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Differentiate the capped sine-ratio footprint transmission.

    Zero spill angle disables footprint loss exactly. Below saturation the
    quotient rule includes candidate angle and spill-angle tangents; at or above
    one, the capped transmission has zero local derivative.
    """
    if spill_angle_deg == 0.0:
        return (
            np.ones(theta_deg.size, dtype=float),
            np.zeros_like(theta_jacobian),
        )
    theta_rad = np.deg2rad(theta_deg)
    spill_rad = np.deg2rad(spill_angle_deg)
    numerator = np.sin(theta_rad)
    denominator = np.sin(spill_rad)
    ratio = numerator / denominator
    numerator_jacobian = np.cos(theta_rad)[:, None] * (np.pi / 180.0) * theta_jacobian
    denominator_jacobian = np.cos(spill_rad) * (np.pi / 180.0) * spill_angle_jacobian
    ratio_jacobian = (
        numerator_jacobian * denominator - numerator[:, None] * denominator_jacobian[None, :]
    ) / denominator**2
    # ``minimum`` creates a saturated branch whose derivative is exactly zero.
    # Strictly sub-unity points retain the full sine-ratio quotient tangent.
    footprint = np.minimum(1.0, ratio)
    jacobian = np.zeros_like(ratio_jacobian)
    active = ratio < 1.0
    jacobian[active] = ratio_jacobian[active]
    return footprint, jacobian


def _scaled_signal_jacobian(
    theta_deg: np.ndarray,
    theta_jacobian: np.ndarray,
    smeared: np.ndarray,
    smeared_jacobian: np.ndarray,
    values: dict[str, float],
    value_jacobians: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply scale and footprint to smeared reflectivity with the product rule.

    Scale multiplies both the optical signal and footprint. Background remains
    outside this helper to preserve the frozen instrument order.
    """
    footprint, footprint_jacobian = _footprint_jacobian(
        theta_deg,
        theta_jacobian,
        values["instrument.footprint_spill_angle_deg"],
        value_jacobians["instrument.footprint_spill_angle_deg"],
    )
    scale = values["instrument.scale"]
    scale_jacobian = value_jacobians["instrument.scale"]
    # Keep multiplication grouping aligned with the primal instrument model.
    # This order also fixes the frozen floating-point derivative artifact.
    signal = scale * smeared * footprint
    signal_jacobian = (smeared * footprint)[:, None] * scale_jacobian[None, :] + scale * (
        footprint[:, None] * smeared_jacobian + smeared[:, None] * footprint_jacobian
    )
    return signal, signal_jacobian


def _background_jacobian(
    problem: object,
    theta_deg: np.ndarray,
    theta_jacobian: np.ndarray,
    values: dict[str, float],
    value_jacobians: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Differentiate constant, linear-q, and optional power-law background.

    Q is recomputed at the beam's effective wavelength and carries angle-offset
    tangents. A disabled zero-amplitude power law returns before evaluating
    logarithms or negative powers, matching the primal inactive mode.
    """
    qz, qz_jacobian = _qz_and_jacobian(
        theta_deg,
        theta_jacobian,
        problem.data.beam.effective_wavelength_a,
    )
    background = values["instrument.background"]
    linear = values["instrument.linear_background_per_a_inv"]
    sampled = background + linear * qz
    sampled_jacobian = (
        value_jacobians["instrument.background"][None, :]
        + qz[:, None] * value_jacobians["instrument.linear_background_per_a_inv"][None, :]
        + linear * qz_jacobian
    )
    amplitude = values["instrument.powerlaw_background_amplitude"]
    amplitude_jacobian = value_jacobians["instrument.powerlaw_background_amplitude"]
    inactive = (amplitude <= 0.0, np.all(amplitude_jacobian == 0.0)) == (True, True)
    if inactive:
        return sampled, sampled_jacobian
    # Power-law differentiation includes both exponent and q-angle dependence.
    # This branch is skipped entirely when amplitude and its tangent are zero.
    exponent = values["instrument.powerlaw_background_exponent"]
    exponent_jacobian = value_jacobians["instrument.powerlaw_background_exponent"]
    power = qz**-exponent
    power_jacobian = power[:, None] * (
        -np.log(qz)[:, None] * exponent_jacobian[None, :] - exponent * qz_jacobian / qz[:, None]
    )
    return (
        sampled + amplitude * power,
        sampled_jacobian + power[:, None] * amplitude_jacobian[None, :] + amplitude * power_jacobian,
    )


def _instrument_model_jacobian(
    problem: object,
    theta_deg: np.ndarray,
    theta_jacobian: np.ndarray,
    values: dict[str, float],
    value_jacobians: dict[str, np.ndarray],
    primary_stack: DifferentiableStack,
    secondary_stack: DifferentiableStack | None,
    point_inputs: tuple[
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
    ],
) -> tuple[np.ndarray, np.ndarray]:
    """Compose smearing, scale-footprint signal, and background derivatives.

    This order mirrors ``instrument_reflectivity`` exactly: wavelength-specific
    optics and resolution first, then beam mixing, scale times footprint, and
    finally additive background on the effective q grid.
    """
    smeared, smeared_jacobian = _smeared_beam_jacobian(
        problem,
        theta_deg,
        theta_jacobian,
        values,
        value_jacobians,
        primary_stack,
        secondary_stack,
        *point_inputs,
    )
    signal, signal_jacobian = _scaled_signal_jacobian(
        theta_deg,
        theta_jacobian,
        smeared,
        smeared_jacobian,
        values,
        value_jacobians,
    )
    background, background_jacobian = _background_jacobian(
        problem,
        theta_deg,
        theta_jacobian,
        values,
        value_jacobians,
    )
    return signal + background, signal_jacobian + background_jacobian


def _differentiable_stacks(
    problem: object,
    values: dict[str, float],
    value_jacobians: dict[str, np.ndarray],
) -> tuple[float, DifferentiableStack, DifferentiableStack | None]:
    """Build primary and optional secondary tangent stacks from one value map.

    Separate expansion is required because material SLD depends on wavelength.
    Geometry and its unit tangents remain shared between the two expansions.
    """
    primary_wavelength = _primary_wavelength(problem)
    primary = _expand_structure_with_jacobian(
        problem,
        values,
        value_jacobians,
        primary_wavelength,
    )
    secondary = (
        _expand_structure_with_jacobian(
            problem,
            values,
            value_jacobians,
            problem.data.beam.wavelength_2_a,
        )
        if problem.data.beam.kind == "mixed_kalpha"
        else None
    )
    return primary_wavelength, primary, secondary


def _masked_optional(
    value: np.ndarray | None,
    mask: np.ndarray,
) -> np.ndarray | None:
    """Apply the model-row mask while preserving an absent optional array.

    The same helper slices primal widths and width tangents, preventing compact
    model rows from drifting relative to full source order.
    """
    return None if value is None else value[mask]


def _model_residual_jacobian(
    problem: FitEvaluationContext,
    unit_vector: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Evaluate fitted residuals and their analytic Jacobian in one traversal.

    Values and coordinate tangents are decoded once; this function does not call
    the primal evaluator. Positive-angle rows are differentiated through stack,
    resolution, beam, footprint, scale, and background operations, then inserted
    into full source order. The returned model values supply the residual from the
    same physical traversal that supplied the Jacobian.
    """
    try:
        values, value_jacobians = values_and_jacobians(problem, unit_vector)
        primary_wavelength, primary_stack, secondary_stack = _differentiable_stacks(
            problem,
            values,
            value_jacobians,
        )
    except PhysicalValueError as error:
        raise EvaluationConstraintError(f"constraint_violation:{type(error).__name__}") from error
    theta = problem.data.two_theta_deg / 2.0 + values["instrument.angle_offset_deg"]
    model_mask = np.isfinite(theta) & (theta > 0.0)
    if np.any(problem.data.fit_mask & ~model_mask):
        raise ValueError("cannot differentiate nonpositive fitted angle")
    theta_model = theta[model_mask]
    angle_jacobian = value_jacobians["instrument.angle_offset_deg"]
    theta_jacobian = np.broadcast_to(
        angle_jacobian,
        (theta_model.size, angle_jacobian.size),
    ).copy()
    primary_sigma, primary_sigma_jacobian = _point_resolution_with_jacobian(
        problem,
        values["instrument.angle_offset_deg"],
        angle_jacobian,
        primary_wavelength,
    )
    if problem.data.beam.kind == "mixed_kalpha":
        secondary_sigma, secondary_sigma_jacobian = _point_resolution_with_jacobian(
            problem,
            values["instrument.angle_offset_deg"],
            angle_jacobian,
            problem.data.beam.wavelength_2_a,
        )
    else:
        secondary_sigma, secondary_sigma_jacobian = None, None
    point_inputs = (
        _masked_optional(primary_sigma, model_mask),
        _masked_optional(primary_sigma_jacobian, model_mask),
        _masked_optional(secondary_sigma, model_mask),
        _masked_optional(secondary_sigma_jacobian, model_mask),
    )
    model, model_jacobian = _instrument_model_jacobian(
        problem,
        theta_model,
        theta_jacobian,
        values,
        value_jacobians,
        primary_stack,
        secondary_stack,
        point_inputs,
    )
    full_model = np.full(problem.data.qz_a_inv.shape, np.nan, dtype=float)
    full_model[model_mask] = model
    full_jacobian = np.zeros(
        (problem.data.qz_a_inv.size, len(problem.variables)),
        dtype=float,
    )
    full_jacobian[model_mask] = model_jacobian
    fit_model = full_model[problem.data.fit_mask]
    residual = log_residuals(
        fit_model,
        problem.data.intensity_normalized[problem.data.fit_mask],
        problem.data.r_floor,
    )
    # d(log10(model + floor)) = d(model) / ((model + floor) * ln(10)).
    # Observations are constant, so they contribute no residual tangent.
    with np.errstate(divide="ignore", invalid="ignore"):
        residual_jacobian = full_jacobian[problem.data.fit_mask] / (
            (fit_model + problem.data.r_floor)[:, None] * np.log(10.0)
        )
    finite_values = np.concatenate(
        (
            model.ravel(),
            model_jacobian.ravel(),
            fit_model,
            residual,
            residual_jacobian.ravel(),
        )
    )
    if np.any(~np.isfinite(finite_values)):
        raise FloatingPointError("nonfinite analytic model or residual Jacobian")
    return (
        np.array(residual, dtype=float, copy=True),
        np.array(residual_jacobian, dtype=float, copy=True),
        float(values["instrument.scale"]),
    )


def evaluate_model_jacobian(
    problem: FitEvaluationContext,
    unit_vector: np.ndarray,
) -> np.ndarray:
    """Return the analytic Jacobian of unweighted fitted log residuals."""
    _residual, jacobian, _scale = _model_residual_jacobian(problem, unit_vector)
    # Defensive copying prevents solver or analysis code from mutating a
    # derivative snapshot that may be shared with candidate publication.
    result = np.array(jacobian, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _scale_prior_residual(problem: object, evaluation: ModelEvaluation) -> float | None:
    """Return the signed standardized logarithmic scale displacement.

    A missing plateau center removes the row completely. Active contexts always
    carry the compiled ``instrument.scale`` parameter value, and the same
    decades-based tau is used by least squares and MCMC.

    The lookup uses the published evaluation snapshot rather than decoding the
    unit vector again. This keeps prior accounting tied to the exact candidate
    that produced the model residuals. The sign is retained for the solver row;
    callers that need a penalty square the standardized displacement themselves.

    ``scale_prior_center`` and ``scale_prior_tau_decades`` are compilation-time
    invariants. This boundary therefore does not repair a missing scale value or
    reinterpret nonpositive metadata after optimization has begun.
    """
    scale = next(value.value for value in evaluation.parameters if value.name == "instrument.scale")
    return _scale_prior_residual_from_scale(problem, scale)


def _scale_prior_residual_from_scale(problem: object, scale: float) -> float | None:
    if problem.scale_prior_center is None:
        return None
    return float((np.log10(scale) - np.log10(problem.scale_prior_center)) / problem.scale_prior_tau_decades)


def _least_squares_row_count(problem: object) -> int:
    """Count fitted data rows plus the optional independent prior row.

    The result is the stable solver axis used even when candidate evaluation
    fails. It depends only on compiled fit selection and prior activation, never
    on a candidate's validity, so sentinel residuals cannot change shape between
    calls.
    """
    return int(np.count_nonzero(problem.data.fit_mask)) + int(problem.scale_prior_center is not None)


def least_squares_residual(
    problem: FitEvaluationContext,
    unit_vector: np.ndarray,
    *,
    evaluator: Callable[[FitEvaluationContext, np.ndarray], ModelEvaluation] | None = None,
) -> np.ndarray:
    """Return fitted log residuals followed by the optional scale-prior row.

    Unit coordinates are validated before the evaluator is invoked. Expected
    physical constraints and invalid evaluations produce the fixed ``1e6``
    solver sentinel with the exact compiled row count; they do not masquerade as
    a successful model evaluation.

    Valid data residuals are copied from the immutable model evaluation. Regional
    weights are intentionally absent because ``least_squares_loss`` applies them
    outside soft-L1. When active, the signed prior displacement is appended after
    every fitted data row.

    The evaluator override is a test boundary, not a second production dispatch
    mechanism. It must honor ``evaluate_model``'s result contract. Unexpected
    exceptions remain visible so malformed contexts and unsupported structures
    cannot be misclassified as merely unfavorable candidates.
    """
    unit = _validated_unit(problem, unit_vector)
    evaluate = evaluate_model if evaluator is None else evaluator
    try:
        observed = evaluate(problem, unit)
    except EvaluationConstraintError:
        observed = None
    if observed is None or not observed.valid:
        return np.full(_least_squares_row_count(problem), 1e6, dtype=float)
    residual = np.array(observed.fit_log_residuals_decades, dtype=float, copy=True)
    prior = _scale_prior_residual(problem, observed)
    return residual if prior is None else np.concatenate((residual, np.asarray([prior])))


def _scale_prior_jacobian(problem: object) -> np.ndarray:
    """Differentiate the optional scale-prior row in unit coordinates.

    Log transforms have a constant decades-per-unit derivative. The affine case
    follows the persisted midpoint convention used by the frozen optimizer
    contract. All non-scale coordinates remain exactly zero.

    The row is allocated even when the prior is inactive so its coordinate axis
    remains explicit and testable. Callers append it only for an active prior.
    A compiled problem has at most one free scale coordinate; locked scale values
    consequently leave the full derivative row at zero.

    This derivative intentionally follows the frozen optimizer convention rather
    than the current candidate scale for affine transforms. Changing that rule
    would alter reference search trajectories and checkpoint replay.
    """
    row = np.zeros(len(problem.variables), dtype=float)
    if problem.scale_prior_center is None:
        return row
    for index, coordinate in enumerate(problem.variables):
        if coordinate.name != "instrument.scale":
            continue
        definition = problem.parameter_definitions[coordinate.parameter_index]
        if definition.transform == "log":
            derivative = log(definition.upper / definition.lower) / log(10.0)
        else:
            scale = definition.lower + 0.5 * (definition.upper - definition.lower)
            derivative = (definition.upper - definition.lower) / (scale * log(10.0))
        row[index] = derivative / problem.scale_prior_tau_decades
    return row


def _empty_residual_jacobian(problem: object) -> np.ndarray:
    """Return a zero data-only Jacobian with the compiled solver axes.

    Prior rows are excluded here because they remain analytically valid when the
    physical model cannot be differentiated. The public wrapper appends that row
    after replacing only the failed data block.
    """
    return np.zeros(
        (np.count_nonzero(problem.data.fit_mask), len(problem.variables)),
        dtype=float,
    )


def _expected_derivative_value_error(error: BaseException) -> bool:
    return isinstance(error, ValueError) and str(error) == ("cannot differentiate nonpositive fitted angle")


def least_squares_system(
    problem: FitEvaluationContext,
    unit_vector: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return solver residuals and Jacobian from one analytic model traversal.

    Valid candidates take their residual and derivative from the same modeled
    reflectivity values. Declared physical constraints retain the fixed residual
    sentinel and zero data Jacobian. A derivative-only floating-point failure
    reuses the ordinary residual boundary and disables only the data Jacobian,
    preserving the established solver failure contract. The optional scale prior
    remains the final independent row in both arrays.

    Returned arrays are owned so the one-entry SciPy callback cache cannot expose
    internal analytic workspaces to solver mutation.
    """
    unit = _validated_unit(problem, unit_vector)
    try:
        residual, jacobian, scale = _model_residual_jacobian(problem, unit)
    except EvaluationConstraintError:
        residual = np.full(_least_squares_row_count(problem), 1e6, dtype=float)
        jacobian = _empty_residual_jacobian(problem)
        scale = None
    except (FloatingPointError, ValueError) as error:
        if isinstance(error, ValueError) and not _expected_derivative_value_error(error):
            raise
        residual = least_squares_residual(problem, unit)
        jacobian = _empty_residual_jacobian(problem)
        scale = None
    if problem.scale_prior_center is not None:
        jacobian = np.vstack((jacobian, _scale_prior_jacobian(problem)))
        if scale is not None:
            prior = _scale_prior_residual_from_scale(problem, scale)
            assert prior is not None
            residual = np.concatenate((residual, np.asarray([prior])))
    return (
        np.array(residual, dtype=float, copy=True),
        np.array(jacobian, dtype=float, copy=True),
    )


def cached_least_squares_callbacks(
    system: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """Memoize the last solver point shared by SciPy's two callbacks.

    The supplied callable remains the sole numerical implementation. This
    helper controls callback lifetime and equality only; it neither catches
    evaluation failures nor manufactures residual or Jacobian values.
    """
    state = local()

    def evaluate(unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        value = np.asarray(unit, dtype=float)
        cached_unit = getattr(state, "unit", None)
        if cached_unit is None or not np.array_equal(value, cached_unit):
            residual, jacobian = system(value)
            state.unit = np.array(value, copy=True)
            state.values = (
                np.array(residual, dtype=float, copy=True),
                np.array(jacobian, dtype=float, copy=True),
            )
        return state.values

    def selected(unit: np.ndarray, index: int) -> np.ndarray:
        return np.array(evaluate(unit)[index], copy=True)

    return partial(selected, index=0), partial(selected, index=1)


def least_squares_residual_jacobian(
    problem: FitEvaluationContext,
    unit_vector: np.ndarray,
    *,
    jacobian_evaluator: Callable[[FitEvaluationContext, np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Return the analytic residual Jacobian with the optional prior row.

    The primary evaluator must return one column per compiled free coordinate.
    Expected candidate constraints, numerical derivative failure, and the known
    nonpositive fitted-angle boundary become a zero data Jacobian so SciPy can
    reject the matching residual sentinel deterministically.

    Other value errors remain programming or unsupported-layout failures and are
    re-raised. The prior row is appended only after data-Jacobian normalization,
    preserving the same row order as ``least_squares_residual``.

    The test override must return the data block only. Prior differentiation is
    owned here so production and test evaluators cannot apply it twice. Copying
    through ``np.array`` also detaches the solver-facing matrix from an immutable
    evaluation snapshot before the optional row is stacked.

    No finite-value repair occurs at this layer. The analytic model boundary is
    responsible for rejecting nonfinite tangents, while shape errors remain
    visible through NumPy or SciPy instead of producing a plausible zero matrix.
    """
    unit = _validated_unit(problem, unit_vector)
    evaluate = evaluate_model_jacobian if jacobian_evaluator is None else jacobian_evaluator
    try:
        jacobian = np.array(evaluate(problem, unit), dtype=float, copy=True)
    except (EvaluationConstraintError, FloatingPointError, ValueError) as error:
        expected_value_error = _expected_derivative_value_error(error)
        if isinstance(error, ValueError) and not expected_value_error:
            raise
        jacobian = _empty_residual_jacobian(problem)
    if problem.scale_prior_center is not None:
        jacobian = np.vstack((jacobian, _scale_prior_jacobian(problem)))
    return jacobian


def least_squares_loss(
    problem: FitEvaluationContext,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build SciPy's three-row soft-L1 loss with external region weights.

    SciPy supplies squared residuals and expects loss value plus first and second
    derivatives with respect to those squared values. Data rows use the frozen
    factor-two soft-L1 convention, with regional weights squared outside the
    robust expression.

    Rows after the fitted data count are independent quadratic priors. Their
    derivative rows therefore stay constant and do not inherit regional weights
    or the data robustness scale. The returned closure captures immutable
    compiled weights and the configured decades scale.

    SciPy passes squared residuals, so ``rho[1]`` and ``rho[2]`` differentiate
    with respect to that squared coordinate rather than the signed residual.
    Keeping all three rows together prevents the objective and analytic solver
    derivatives from acquiring different factor-two conventions.

    The closure accepts an arbitrary number of trailing prior rows even though
    the current compiler emits at most one. Every trailing row uses the same
    exact quadratic contract and remains independent of regional balancing.
    """
    weights = np.asarray(problem.weights[problem.data.fit_mask], dtype=float)
    c_decades = problem.config.c_decades
    data_count = weights.size

    def loss(squared: np.ndarray) -> np.ndarray:
        """Evaluate loss value and two derivatives without changing row axes.

        The returned shape is always ``(3, squared.size)`` as required by SciPy's
        callable-loss protocol. Empty data selections and prior-only vectors use
        the same allocation path, avoiding special cases in optimizer dispatch.
        """
        values = np.asarray(squared, dtype=float)
        rho = np.empty((3, values.size), dtype=float)
        data = values[:data_count]
        scaled = 1.0 + data / c_decades**2
        rho[0, :data_count] = 4.0 * weights**2 * c_decades**2 * (np.sqrt(scaled) - 1.0)
        rho[1, :data_count] = 2.0 * weights**2 / np.sqrt(scaled)
        rho[2, :data_count] = -(weights**2 / c_decades**2) * scaled ** (-1.5)
        if values.size > data_count:
            rho[0, data_count:] = 2.0 * values[data_count:]
            rho[1, data_count:] = 2.0
            rho[2, data_count:] = 0.0
        return rho

    return loss


def _normal_log_density(parameters: tuple[float, ...], x: float) -> float:
    mean, std = parameters
    return -0.5 * ((x - mean) / std) ** 2


def _lognormal_log_density(parameters: tuple[float, ...], x: float) -> float:
    # Parameters live in log space; the 1/x Jacobian makes this a density in x.
    log_mean, log_std = parameters
    if x <= 0.0:
        return -np.inf
    return -0.5 * ((log(x) - log_mean) / log_std) ** 2 - log(x)


def _soft_range_log_density(parameters: tuple[float, ...], x: float) -> float:
    # Flat inside [low, high], Gaussian shoulders of scale std outside it.
    low, high, std = parameters
    distance = max(low - x, x - high, 0.0)
    return -0.5 * (distance / std) ** 2


def _uniform_log_density(parameters: tuple[float, ...], x: float) -> float:
    return 0.0


PRIOR_LOG_DENSITY: dict[str, Callable[[tuple[float, ...], float], float]] = {
    "uniform": _uniform_log_density,
    "normal": _normal_log_density,
    "lognormal": _lognormal_log_density,
    "soft_range": _soft_range_log_density,
}


def _unnormalized_density(kind: str, parameters: tuple[float, ...], x: float) -> float:
    return float(np.exp(PRIOR_LOG_DENSITY[kind](parameters, x)))


PRIOR_QUADRATURE_NODES = 2049


PRIOR_GRID_NODES: dict[str, Callable[[tuple[float, ...]], np.ndarray]] = {
    "normal": lambda values: values[0] + values[1] * np.linspace(-12.0, 12.0, 1025),
    "lognormal": lambda values: np.exp(values[0] + values[1] * np.linspace(-12.0, 12.0, 1025)),
    "soft_range": lambda values: np.concatenate(
        (
            values[0] + values[2] * np.linspace(-12.0, 12.0, 513),
            values[1] + values[2] * np.linspace(-12.0, 12.0, 513),
        )
    ),
}


def _prior_grid(
    kind: str,
    parameters: tuple[float, ...],
    lower: float,
    upper: float,
) -> np.ndarray:
    """Return a deterministic grid refined around every narrow feature.

    The original 2049 evenly spaced nodes remain the common integration grid.
    They preserve the established uniform-prior path and provide complete hard
    support coverage for every distribution.

    Normal and lognormal kernels can be much narrower than one base interval.
    Each therefore contributes 1025 nodes spanning twelve standard deviations
    on either side of its center.  Twelve sigmas include essentially all mass
    while retaining deterministic, finite work independent of the hard-box
    width.

    Lognormal nodes are spaced in the distribution's native log coordinate and
    exponentiated only after refinement.  This prevents a narrow positive-space
    peak from disappearing when the physical hard box spans several decades.

    Soft ranges have two independently sharp shoulders.  Both edges receive
    513 nodes across the same twelve-sigma window, so each transition is
    resolved without biasing the flat interior toward either edge.

    Nodes outside the declared box and non-finite exponentiation results are
    removed before the union.  The hard endpoints are inserted explicitly;
    sorting and duplicate removal are delegated to ``np.unique`` so cumulative
    integration always sees a strictly increasing coordinate vector.

    Uniform priors have no local feature and use the base grid unchanged.  The
    dispatch table keeps distribution-specific geometry out of this shared
    assembly path and avoids a branch ladder as future prior kinds are added.

    Refinement changes only numerical resolution.  It does not widen support,
    alter log kernels, change the cached normalization key, or affect problems
    with no parameter priors.

    The generated nodes are deterministic functions of immutable prior
    parameters.  No random generator, mutable cache entry, fitted vector, or
    platform path participates in their construction.

    The normal window is affine in physical coordinates.  Translating its mean
    translates every local node by the same amount; scaling its sigma scales
    every offset without moving the declared hard endpoints.

    The lognormal window applies that same affine construction before the
    exponential map.  Positive support follows from exponentiation, while an
    overflowed tail is discarded by the common finite-node filter.

    The soft-range window treats its lower and upper shoulder symmetrically.
    Concatenation order is irrelevant after sorting, but constructing both with
    one shared offset vector guarantees matching resolution.

    Base and local nodes may coincide at centers, shoulders, or hard endpoints.
    Duplicate removal prevents zero-width trapezoids and therefore avoids a
    distribution-specific special case in cumulative integration.

    Clipping happens before the union rather than after integration.  Thus no
    probability mass is sampled outside the conditioned interval and every CDF
    entry corresponds to a coordinate that is legal for the parameter.

    Keeping the base grid for narrow priors is intentional.  Local nodes resolve
    the peak while base nodes retain tails, broad shoulders, and both support
    endpoints in the same cumulative array.

    Returning the untouched base array for a uniform prior preserves its exact
    historical spacing and cumulative interpolation behavior.

    The cached normalization remains keyed by kind, parameters, and bounds, so
    repeated density and quantile calls reuse this complete refined grid.

    CDF and inverse-CDF interpolation consume the identical returned axes.
    Refinement therefore improves both directions together rather than letting
    density, probability, and quantile calculations drift onto separate grids.

    Endpoint insertion is retained even though the base grid already contains
    both bounds.  It makes the local-grid contract explicit and robust if the
    base construction is later replaced by another deterministic mesh.

    A missing dispatch entry means that the prior has no localized feature.
    Falling back to the base grid is preferable to fabricating arbitrary anchor
    points or silently changing that prior's kernel.

    Twelve-sigma windows are deliberately much wider than conventional display
    intervals.  The extra tail nodes cost little after caching and prevent the
    truncated normalization from depending on one coarse interval around a very
    narrow center.

    All arrays use NumPy's default floating dtype, matching the surrounding
    evaluation boundary and avoiding hidden precision conversions between the
    grid, density, trapezoid, and interpolation stages.

    The helper returns a newly owned array for every uncached construction.
    Callers normalize and retain it without mutating shared module constants.

    Sorted finite nodes also make monotonic CDF construction an invariant rather
    than an assumption left to each prior-specific node generator.

    This single assembly function is the only place that combines global hard
    support with local distribution geometry.

    Its output is ready for direct trapezoidal integration.
    """
    base = np.linspace(lower, upper, PRIOR_QUADRATURE_NODES)
    nodes = PRIOR_GRID_NODES.get(kind)
    if nodes is None:
        return base
    local = nodes(parameters)
    local = local[np.isfinite(local) & (local >= lower) & (local <= upper)]
    return np.unique(np.concatenate((base, local, (lower, upper))))


@lru_cache(maxsize=256)
def _prior_norm(
    kind: str,
    parameters: tuple[float, ...],
    lower: float,
    upper: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Cache the truncation constant, grid, and cumulative mass of one prior.

    Composite trapezoid on a fixed grid keeps every kind on one code path and
    makes ``prior_cdf`` and ``prior_inverse_cdf`` exact inverses by construction.
    Keys are immutable scalars only, so frozen ``PriorSpec`` values are never
    retained and repeated evaluation costs one dict lookup.
    """
    grid = _prior_grid(kind, parameters, lower, upper)
    density = np.array([_unnormalized_density(kind, parameters, float(x)) for x in grid])
    increments = np.diff(grid) * (density[:-1] + density[1:]) / 2.0
    cumulative = np.concatenate(([0.0], np.cumsum(increments)))
    total = float(cumulative[-1])
    if not (isfinite(total) and total > 0.0):
        raise ValueError(f"{kind} prior has no mass inside the declared bounds")
    return total, grid, cumulative / total


def prior_log_density(spec: PriorSpec, x: float, lower: float, upper: float) -> float:
    """Return the log density of ``spec`` truncated and renormalized on the bounds."""
    if not lower <= x <= upper:
        return -np.inf
    total, _, _ = _prior_norm(spec.kind, spec.parameters, lower, upper)
    return PRIOR_LOG_DENSITY[spec.kind](spec.parameters, x) - log(total)


def prior_cdf(spec: PriorSpec, x: float, lower: float, upper: float) -> float:
    """Return the truncated cumulative probability of ``spec`` at ``x``."""
    _, grid, masses = _prior_norm(spec.kind, spec.parameters, lower, upper)
    return float(np.interp(x, grid, masses))


def prior_inverse_cdf(spec: PriorSpec, level: float, lower: float, upper: float) -> float:
    """Return the truncated quantile of ``spec`` at probability ``level``."""
    if not 0.0 <= level <= 1.0:
        raise ValueError("prior cdf level must be within [0, 1]")
    _, grid, masses = _prior_norm(spec.kind, spec.parameters, lower, upper)
    return float(np.interp(level, masses, grid))


def prior_bounds(spec: PriorSpec, lower: float, upper: float) -> tuple[float, float]:
    """Return the support of ``spec``, which truncation pins to the declared bounds."""
    if spec.kind not in PRIOR_LOG_DENSITY:
        raise ValueError(f"unsupported prior kind: {spec.kind}")
    return float(lower), float(upper)


def prior_center_and_spread(spec: PriorSpec) -> tuple[float, float] | None:
    """Return the physical center and spread of ``spec``, or ``None`` when flat.

    ``uniform`` carries no location, so it never takes part in conflict scoring.
    ``lognormal`` parameters are in log space; its physical spread is the
    first-order image of the log-space sigma.
    """
    if spec.kind == "normal":
        mean, std = spec.parameters
        return float(mean), float(std)
    if spec.kind == "lognormal":
        log_mean, log_std = spec.parameters
        center = exp(log_mean)
        return float(center), float(center * log_std)
    if spec.kind == "soft_range":
        low, high, std = spec.parameters
        return float((low + high) / 2.0), float((high - low) / 2.0 + std)
    return None


def _prior_coordinate(
    definition: ParameterDefinition,
    unit_value: float,
) -> tuple[float, float, float]:
    """Return the value and bounds a prior is declared against for one definition.

    ``roughness_fraction`` priors live on the dimensionless unit fraction in
    ``[0, 1]`` because the physical roughness depends on a geometry snapshot this
    scalar path does not own.
    """
    if definition.transform == "roughness_fraction":
        return float(unit_value), 0.0, 1.0
    value = float(unit_to_physical(definition, unit_value))
    return value, float(definition.lower), float(definition.upper)


def _parameter_prior_log_density(
    problem: FitEvaluationContext,
    unit: np.ndarray,
) -> float:
    """Return the summed parameter prior log density, exactly ``0.0`` when unused.

    Returning the literal zero keeps ``problem_log_probability`` bitwise
    identical for priorless problems, since IEEE 754 addition of ``0.0`` is exact.
    """
    total = 0.0
    for index, variable in enumerate(problem.variables):
        definition = problem.parameter_definitions[variable.parameter_index]
        if definition.prior is None:
            continue
        value, lower, upper = _prior_coordinate(definition, float(unit[index]))
        total += prior_log_density(definition.prior, value, lower, upper)
    return total


def problem_log_probability(
    problem: FitEvaluationContext,
    unit_vector: np.ndarray,
) -> float:
    """Return the robust pseudo-posterior density used by uncertainty MCMC.

    This is a deterministic analysis density, not a normalized probability
    distribution. The data term is the same weighted soft-L1 loss used by fit,
    converted from its mean form back to a sum and scaled by ``2*c**2``.

    Invalid unit coordinates, expected physical constraints, and invalid model
    evaluations have negative-infinite density. Unsupported layouts and other
    programming errors are not swallowed. An active scale prior contributes
    ``-0.5 * standardized**2`` independently of fitted-point count.

    Returning ``-inf`` at declared domain boundaries gives ensemble samplers a
    conventional rejection signal without inventing a finite penalty magnitude.
    The accepted path uses the same weighted soft-L1 expression as deterministic
    fitting while retaining the sampler's frozen pointwise summation order. That
    explicit grouping is required for bitwise replay of stored log probabilities.

    This density omits normalization constants because downstream analysis uses
    only relative log probability. It also performs no mutation, caching, or
    random work, making repeated evaluation deterministic for a fixed context
    and unit vector.
    """
    try:
        unit = _validated_unit(problem, unit_vector)
    except ValueError:
        return -np.inf
    try:
        observed = evaluate_model(problem, unit)
    except EvaluationConstraintError:
        return -np.inf
    if not all((observed.valid, isfinite(observed.objective))):
        return -np.inf
    residual = np.asarray(observed.fit_log_residuals_decades, dtype=float)
    weights = np.asarray(problem.weights[problem.data.fit_mask], dtype=float)
    c_decades = problem.config.c_decades
    # Preserve the frozen sampler grouping: sum point losses before dividing by
    # the robust scale. Replacing this with mean * count changes retained log
    # probabilities by a few ULPs even though the expressions are algebraically
    # equivalent, which breaks deterministic checkpoint and reference replay.
    data_loss = np.sum(weights**2 * 2.0 * c_decades**2 * (np.sqrt(1.0 + (residual / c_decades) ** 2) - 1.0))
    log_probability = -float(data_loss) / (2.0 * c_decades**2)
    if problem.scale_prior_center is not None:
        prior = _scale_prior_residual(problem, observed)
        assert prior is not None
        log_probability -= 0.5 * prior**2
    log_probability += _parameter_prior_log_density(problem, unit)
    return float(log_probability)
