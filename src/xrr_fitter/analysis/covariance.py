"""Total-information covariance with explicit rank and calibration failures."""

from __future__ import annotations

import numpy as np

from xrr_fitter.analysis.derivatives import correlation_from_covariance, physical_parameter_jacobian
from xrr_fitter.evaluation import StatisticalUnavailableError, statistical_information
from xrr_fitter.model.analysis import CovarianceEvidence, ResidualEvidence
from xrr_fitter.model.fitting import FitEvaluationContext

RANK_RTOL = 1e-12


def covariance_summary(evidence: CovarianceEvidence) -> tuple[np.ndarray, np.ndarray | None]:
    matrix = evidence.matrix
    if matrix is None:
        return np.full((len(evidence.names), len(evidence.names)), np.nan), None
    return correlation_from_covariance(matrix), np.sqrt(np.diag(matrix))


def covariance_method(modes: tuple[str, ...]) -> str:
    if all(mode == "gaussian" for mode in modes):
        return "gaussian_known_sigma"
    if all(mode == "poisson" for mode in modes):
        return "poisson_expected_information"
    return "robust_sandwich_iid" if all(mode == "robust_log" for mode in modes) else "mixed_sandwich_iid"


def _matrices(values: tuple[np.ndarray, ...], width: int) -> tuple[np.ndarray, ...]:
    result = tuple(np.asarray(value, dtype=float) for value in values)
    if any(value.shape != (width, width) or np.any(~np.isfinite(value)) for value in result):
        raise ValueError("statistical matrices must be finite and match the parameter axis")
    return result


def _rank_evidence(information: np.ndarray, names: tuple[str, ...]):
    scale = np.sqrt(np.maximum(np.diag(information), 0.0))
    scale[scale == 0.0] = 1.0
    normalized = information / scale[:, None] / scale[None, :]
    _left, singular, right = np.linalg.svd(normalized)
    rank = int(np.count_nonzero(singular > (singular[0] * RANK_RTOL if singular.size else 0)))
    null = right[rank:]
    unidentified = tuple(name for index, name in enumerate(names) if np.any(np.abs(null[:, index]) > 1e-6))
    return rank, unidentified, scale


def covariance_from_matrices(
    names: tuple[str, ...],
    data_information: np.ndarray,
    prior_information: np.ndarray,
    meat: np.ndarray,
    physical_jacobian: np.ndarray,
    *,
    method: str,
    unavailable_reason: str | None = None,
) -> CovarianceEvidence:
    """Invert only full-rank data information; priors never invent observations."""
    data, prior, meat, physical = _matrices((data_information, prior_information, meat, physical_jacobian), len(names))
    rank, unidentified, scales = _rank_evidence(data, names)
    reason = "rank_deficient" if rank < len(names) else unavailable_reason
    if reason is not None:
        return CovarianceEvidence(names, None, method, rank, unidentified, reason)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        bread = (data + prior) / scales[:, None] / scales[None, :]
        variance = meat / scales[:, None] / scales[None, :]
        inverse_mapping = np.linalg.solve(bread, (physical / scales[None, :]).T).T
        matrix = inverse_mapping @ variance @ inverse_mapping.T
    matrix = matrix * 0.5 + matrix.T * 0.5
    if names and np.any(np.diag(matrix) <= 0.0):
        return CovarianceEvidence(names, None, method, rank, (), "zero_score_variation")
    return CovarianceEvidence(names, matrix, method, rank)


def calibration_unavailable_reason(
    modes: tuple[str, ...],
    unit_vector: np.ndarray,
    residuals: tuple[ResidualEvidence, ...],
    boundary_fraction: float,
) -> str | None:
    if np.any((unit_vector <= boundary_fraction) | (unit_vector >= 1 - boundary_fraction)):
        return "boundary_solution"
    if "robust_log" in modes:
        if not residuals or any(not evidence.executed for evidence in residuals):
            return "residual_diagnostics_not_executed"
        if any(evidence.autocorrelation for evidence in residuals):
            return "residual_autocorrelation_requires_block_bootstrap"
    return None


def problem_covariance(
    problem: FitEvaluationContext,
    unit_vector: np.ndarray,
    residuals: tuple[ResidualEvidence, ...],
) -> CovarianceEvidence:
    names = tuple(variable.name for variable in problem.variables)
    modes = (problem.config.noise_model,)
    method = covariance_method(modes)
    reason = calibration_unavailable_reason(modes, unit_vector, residuals, problem.config.confidence.boundary_fraction)
    try:
        data, prior, meat = statistical_information(problem, unit_vector)
        physical = physical_parameter_jacobian(problem, unit_vector)
        return covariance_from_matrices(names, data, prior, meat, physical, method=method, unavailable_reason=reason)
    except (StatisticalUnavailableError, FloatingPointError, np.linalg.LinAlgError) as error:
        return CovarianceEvidence(names, None, method, 0, (), f"{type(error).__name__}:{error}")


def joint_covariance(
    names: tuple[str, ...],
    unit_vector: np.ndarray,
    problems: tuple[FitEvaluationContext, ...],
    layout: tuple,
    residuals: tuple[ResidualEvidence, ...],
) -> CovarianceEvidence:
    """Sum member score contributions after the complete global scatter map."""
    units, scatters, physical, multipliers = layout
    modes = tuple(problem.config.noise_model for problem in problems)
    method = covariance_method(modes)
    reason = calibration_unavailable_reason(
        modes, unit_vector, residuals, problems[0].config.confidence.boundary_fraction
    )
    data, prior, meat = (np.zeros((len(names), len(names))) for _ in range(3))
    try:
        for problem, unit, scatter, alpha in zip(problems, units, scatters, multipliers, strict=True):
            local_data, local_prior, local_meat = statistical_information(problem, unit)
            data += alpha * (scatter.T @ local_data @ scatter)
            prior += scatter.T @ local_prior @ scatter
            meat += alpha**2 * (scatter.T @ local_meat @ scatter)
        return covariance_from_matrices(names, data, prior, meat, physical, method=method, unavailable_reason=reason)
    except (StatisticalUnavailableError, FloatingPointError, np.linalg.LinAlgError) as error:
        return CovarianceEvidence(names, None, method, 0, (), f"{type(error).__name__}:{error}")
