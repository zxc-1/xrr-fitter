"""Internal solver implementation."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from math import isfinite, log
from threading import local

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
from xrr_fitter.evaluation_instrument_jacobian import _model_residual_jacobian, evaluate_model_jacobian
from xrr_fitter.evaluation_model import evaluate_model
from xrr_fitter.evaluation_parameters import EvaluationConstraintError, _validated_unit
from xrr_fitter.model.fitting import FitEvaluationContext, ModelEvaluation
from xrr_fitter.model.parameters import (
    _log10_ratio,
    _log_interval_width,
)


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
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            residual = np.divide(
                _log10_ratio(scale, problem.scale_prior_center),
                problem.scale_prior_tau_decades,
            )
    except FloatingPointError as error:
        raise FloatingPointError("scale prior residual is not finite") from error
    if not isfinite(residual):
        raise FloatingPointError("scale prior residual is not finite")
    return float(residual)


def _least_squares_residual_parts(
    problem: FitEvaluationContext,
    unit: np.ndarray,
    evaluator: Callable[[FitEvaluationContext, np.ndarray], ModelEvaluation],
) -> tuple[np.ndarray, float | None, bool]:
    """Return data rows, the optional prior row, and candidate validity."""
    try:
        observed = evaluator(problem, unit)
    except EvaluationConstraintError:
        return (
            np.full(np.count_nonzero(problem.data.fit_mask), 1e6, dtype=float),
            None,
            False,
        )
    if not observed.valid:
        return (
            np.full(np.count_nonzero(problem.data.fit_mask), 1e6, dtype=float),
            None,
            False,
        )
    residual = np.array(observed.fit_log_residuals_decades, dtype=float, copy=True)
    return residual, _scale_prior_residual(problem, observed), True


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
    residual, prior, valid = _least_squares_residual_parts(problem, unit, evaluate)
    if not valid:
        return np.full(_least_squares_row_count(problem), 1e6, dtype=float)
    return residual if prior is None else np.concatenate((residual, np.asarray([prior])))


def _log_scale_prior_derivative(definition: object) -> float:
    if definition.lower <= 0.0 or definition.upper <= 0.0:
        raise ValueError("scale prior requires positive log bounds")
    try:
        log_span = _log_interval_width(definition.lower, definition.upper)
    except (ValueError, OverflowError) as error:
        raise FloatingPointError("scale prior log span is not representable") from error
    return log_span / log(10.0)


def _affine_scale_prior_derivative(definition: object) -> float:
    span = definition.upper - definition.lower
    if not isfinite(span):
        raise FloatingPointError("scale prior affine parameter span is not representable")
    scale = definition.lower + 0.5 * span
    if not isfinite(scale) or scale <= 0.0:
        raise FloatingPointError("scale prior affine midpoint is not positive and finite")
    return span / (scale * log(10.0))


def _scale_prior_derivative(definition: object) -> float:
    derivative = (
        _log_scale_prior_derivative(definition)
        if definition.transform == "log"
        else _affine_scale_prior_derivative(definition)
    )
    if not isfinite(derivative):
        raise FloatingPointError("scale prior Jacobian is not finite")
    return derivative


def _scaled_prior_derivative(derivative: float, tau_decades: float) -> float:
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            scaled = np.divide(derivative, tau_decades)
    except FloatingPointError as error:
        raise FloatingPointError("scale prior Jacobian is not finite") from error
    if not isfinite(scaled):
        raise FloatingPointError("scale prior Jacobian is not finite")
    return float(scaled)


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
        derivative = _scale_prior_derivative(definition)
        row[index] = _scaled_prior_derivative(
            derivative,
            problem.scale_prior_tau_decades,
        )
    return row


def _empty_residual_jacobian(problem: object) -> np.ndarray:
    """Return a zero data-only Jacobian with the compiled solver axes.

    Prior rows are excluded so callers can append exactly one row after deciding
    whether the candidate itself remains valid.
    """
    return np.zeros(
        (np.count_nonzero(problem.data.fit_mask), len(problem.variables)),
        dtype=float,
    )


def _expected_derivative_value_error(error: BaseException) -> bool:
    return isinstance(error, ValueError) and str(error) == ("cannot differentiate nonpositive fitted angle")


def _solver_data_system(
    problem: FitEvaluationContext,
    unit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float | None, float | None, bool]:
    prior = None
    try:
        residual, jacobian, scale = _model_residual_jacobian(problem, unit)
        return residual, jacobian, prior, scale, True
    except EvaluationConstraintError:
        residual = np.full(
            np.count_nonzero(problem.data.fit_mask),
            1e6,
            dtype=float,
        )
        return residual, _empty_residual_jacobian(problem), prior, None, False
    except (FloatingPointError, ValueError) as error:
        if isinstance(error, ValueError) and not _expected_derivative_value_error(error):
            raise
        residual, prior, valid = _least_squares_residual_parts(
            problem,
            unit,
            evaluate_model,
        )
        return residual, _empty_residual_jacobian(problem), prior, None, valid


def _append_scale_prior_row(
    problem: object,
    residual: np.ndarray,
    jacobian: np.ndarray,
    prior: float | None,
    scale: float | None,
    valid: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if problem.scale_prior_center is None:
        return residual, jacobian
    if valid and scale is not None:
        prior = _scale_prior_residual_from_scale(problem, scale)
        assert prior is not None
        prior_jacobian = _scale_prior_jacobian(problem)
    else:
        prior = prior if valid else 1e6
        prior_jacobian = _scale_prior_jacobian(problem) if valid else np.zeros(len(problem.variables), dtype=float)
    assert prior is not None
    return (
        np.concatenate((residual, np.asarray([prior]))),
        np.vstack((jacobian, prior_jacobian)),
    )


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
    residual, jacobian, prior, scale, valid = _solver_data_system(problem, unit)
    residual, jacobian = _append_scale_prior_row(
        problem,
        residual,
        jacobian,
        prior,
        scale,
        valid,
    )
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
    valid = True
    try:
        jacobian = np.array(evaluate(problem, unit), dtype=float, copy=True)
    except (EvaluationConstraintError, FloatingPointError, ValueError) as error:
        expected_value_error = _expected_derivative_value_error(error)
        if isinstance(error, ValueError) and not expected_value_error:
            raise
        jacobian = _empty_residual_jacobian(problem)
        if isinstance(error, EvaluationConstraintError) or expected_value_error:
            valid = False
        else:
            # A floating-point failure may be derivative-only. Re-evaluate the
            # primal boundary before deciding whether its prior row is valid.
            _residual, _prior, valid = _least_squares_residual_parts(
                problem,
                unit,
                evaluate_model,
            )
    if problem.scale_prior_center is not None:
        prior_jacobian = _scale_prior_jacobian(problem) if valid else np.zeros(len(problem.variables), dtype=float)
        jacobian = np.vstack((jacobian, prior_jacobian))
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
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            scaled_data = data / c_decades**2
            scaled = 1.0 + scaled_data
            root = np.sqrt(scaled)
            rho[0, :data_count] = 4.0 * weights**2 * c_decades**2 * (root - 1.0)
            rho[1, :data_count] = 2.0 * weights**2 / root
            rho[2, :data_count] = -(weights**2 / c_decades**2) * scaled ** (-1.5)
        invalid_columns = np.any(~np.isfinite(rho[:, :data_count]), axis=0)
        near_zero = (data > 0.0) & np.isfinite(scaled_data) & (scaled_data < 1e-8)
        if np.any(invalid_columns | near_zero):
            with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
                root_data = np.sqrt(data)
                radius = np.hypot(c_decades, root_data)
                stable = np.vstack(
                    (
                        4.0 * weights**2 * c_decades * root_data * (root_data / (radius + c_decades)),
                        2.0 * weights**2 * c_decades / radius,
                        -(weights**2) * (((c_decades / radius) / radius) / radius),
                    )
                )
            rho[:3, :data_count] = np.where(invalid_columns[None, :], stable, rho[:3, :data_count])
            rho[0, :data_count] = np.where(near_zero, stable[0], rho[0, :data_count])
        if values.size > data_count:
            rho[0, data_count:] = 2.0 * values[data_count:]
            rho[1, data_count:] = 2.0
            rho[2, data_count:] = 0.0
        return rho

    return loss
