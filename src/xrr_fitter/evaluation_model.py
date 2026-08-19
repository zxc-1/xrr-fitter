"""Internal model implementation."""

from __future__ import annotations

from dataclasses import replace
from functools import partial

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
from xrr_fitter.evaluation_objective import log_residuals, robust_log_cost, scale_prior_penalty
from xrr_fitter.evaluation_parameters import EvaluationConstraintError, values_and_jacobians, values_by_name
from xrr_fitter.model.fitting import FitEvaluationContext, ModelEvaluation
from xrr_fitter.model.instrument import PhysicsDiagnostic, resolution_to_sigma_q
from xrr_fitter.model.parameters import (
    ParameterDefinition,
    ParameterValue,
)
from xrr_fitter.model.slab_stack import SlabStack
from xrr_fitter.model.structure import (
    StructureSpec,
)
from xrr_fitter.physics.geometry import DifferentiableStack
from xrr_fitter.physics.parratt import parratt_reflectivity
from xrr_fitter.physics.reflectivity import (
    instrument_reflectivity,
    qz_from_theta_deg,
)
from xrr_fitter.physics.stack import expand_structure, rebuild_structure


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
    gradient_slab_counts = _gradient_slab_counts(problem)
    primary = expand_structure(rebuilt, primary_wavelength, gradient_slab_counts)
    secondary = (
        expand_structure(rebuilt, problem.data.beam.wavelength_2_a, gradient_slab_counts)
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
    a previously prepared q array. Angles outside the grazing-incidence domain
    ``(0, 90]`` remain outside the compact physics model and keep NaN q entries
    in full source order.
    """
    theta = problem.data.two_theta_deg / 2.0 + values["instrument.angle_offset_deg"]
    model_mask = np.isfinite(theta) & (theta > 0.0) & (theta <= 90.0)
    model_indices = np.flatnonzero(model_mask)
    # Full source layout is retained for result publication and diagnostic rows.
    # Only the compact positive-angle subset enters trigonometric physics.
    qz = np.full(problem.data.qz_a_inv.shape, np.nan, dtype=float)
    qz[model_mask] = qz_from_theta_deg(
        theta[model_mask],
        problem.data.beam.effective_wavelength_a,
    )
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
    qz = qz_from_theta_deg(theta, wavelength_a)
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
    try:
        theta, model_mask, model_indices, qz = _angle_layout(problem, values)
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
    except (ValueError, FloatingPointError, OverflowError) as error:
        raise EvaluationConstraintError(f"constraint_violation:{type(error).__name__}") from error
    if np.any(problem.data.fit_mask & ~model_mask):
        raise EvaluationConstraintError("nonpositive_fitted_incident_angle")
    parameters = _parameter_values(problem, values)
    diagnostics: list[PhysicsDiagnostic] = []
    try:
        _append_stack_diagnostics(
            primary_stack,
            secondary_stack,
            primary_wavelength,
            problem.data.beam.wavelength_2_a,
            theta[model_mask],
            model_indices,
            diagnostics,
        )
    except (ValueError, FloatingPointError, OverflowError) as error:
        raise EvaluationConstraintError(
            f"constraint_violation:{type(error).__name__}",
            tuple(diagnostics),
        ) from error
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


def _masked_optional(
    value: np.ndarray | None,
    mask: np.ndarray,
) -> np.ndarray | None:
    """Apply the model-row mask while preserving an absent optional array.

    The same helper slices primal widths and width tangents, preventing compact
    model rows from drifting relative to full source order.
    """
    return None if value is None else value[mask]
