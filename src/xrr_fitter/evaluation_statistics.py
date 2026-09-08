"""Mode-specific residuals and total-data objective statistics shared by domains."""

from __future__ import annotations

import numpy as np

from xrr_fitter.evaluation_objective import log_residuals, robust_log_cost, robust_loss_rho, robust_score_information
from xrr_fitter.evaluation_parameters import EvaluationConstraintError
from xrr_fitter.model.data import PreparedData, log_domain_mask, validate_noise_model
from xrr_fitter.model.fitting import FitEvaluationContext


class StatisticalUnavailableError(ValueError):
    """A valid fitting point does not admit the declared local inference."""


def _gaussian_sigma(data: PreparedData) -> np.ndarray:
    if data.intensity_sigma_normalized is None:
        raise ValueError("Gaussian fitting requires known intensity sigma at every selected point")
    sigma = data.intensity_sigma_normalized[data.fit_mask]
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("Gaussian intensity sigma must be finite and strictly positive at selected points")
    return sigma


def _validate_raw_counts(data: PreparedData) -> None:
    counts = data.intensity_raw[data.fit_mask]
    if np.any(~np.isfinite(counts)) or np.any(counts < 0.0) or np.any(counts != np.floor(counts)):
        raise ValueError("Poisson fitting requires finite nonnegative integer raw counts")
    selected = np.flatnonzero(data.fit_mask)
    if any(len(data.source_row_groups[index]) != 1 for index in selected):
        raise ValueError("Poisson raw counts cannot use merged duplicate rows or averaged counts")
    if not np.allclose(counts / data.normalization, data.intensity_normalized[data.fit_mask], rtol=1e-12, atol=0.0):
        raise ValueError("Poisson raw counts must match the declared intensity normalization")


def validate_noise_data(data: PreparedData, noise_model: str) -> None:
    """Preflight selected rows only; explicit Poisson mode declares raw-count semantics."""
    validate_noise_model(noise_model)
    observed = data.intensity_normalized[data.fit_mask]
    if np.any(~np.isfinite(observed)):
        raise ValueError("selected observed intensity must be finite")
    if noise_model == "gaussian":
        _gaussian_sigma(data)
    elif noise_model == "poisson":
        _validate_raw_counts(data)
    elif not np.all(log_domain_mask(observed, data.r_floor)):
        raise ValueError("selected reflectivity plus floor must be positive for robust_log")


def _poisson_arrays(counts: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts, prediction = np.asarray(counts, dtype=float), np.asarray(prediction, dtype=float)
    if counts.ndim != 1 or counts.shape != prediction.shape:
        raise ValueError("Poisson counts and prediction must be aligned vectors")
    if np.any(~np.isfinite(counts)) or np.any(counts < 0.0) or np.any(counts != np.floor(counts)):
        raise ValueError("Poisson counts must be finite nonnegative integers")
    if np.any(~np.isfinite(prediction)) or np.any(prediction < 0.0) or np.any((counts > 0.0) & (prediction == 0.0)):
        raise ValueError("Poisson prediction must be nonnegative and positive for positive counts")
    return counts, prediction


def _positive_count_deviance(counts: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    difference = prediction - counts
    close = np.abs(difference) < 1e-3 * counts
    result = np.empty(counts.size, dtype=float)
    ratio = difference[close] / counts[close]
    # Series for t-log1p(t) avoids cancellation at the likelihood maximum.
    series = 0.5 + ratio * (-1 / 3 + ratio * (1 / 4 + ratio * (-1 / 5 + ratio * (1 / 6 - ratio / 7))))
    result[close] = 2 * difference[close] * ratio * series
    far = ~close
    result[far] = 2 * (difference[far] + counts[far] * (np.log(counts[far]) - np.log(prediction[far])))
    return result


def poisson_deviance(counts: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Return per-count -2 log likelihood relative to the saturated model."""
    counts, prediction = _poisson_arrays(counts, prediction)
    positive = counts > 0.0
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        result = np.empty(counts.size, dtype=float)
        result[~positive] = 2 * prediction[~positive]
        result[positive] = _positive_count_deviance(counts[positive], prediction[positive])
    if np.any(~np.isfinite(result)) or np.any(result < 0.0):
        raise FloatingPointError("Poisson deviance is not finite and nonnegative")
    return result


def data_residuals(problem: FitEvaluationContext, fit_model: np.ndarray) -> np.ndarray:
    data, mode = problem.data, problem.config.noise_model
    observed = data.intensity_normalized[data.fit_mask]
    if mode == "robust_log":
        return log_residuals(fit_model, observed, data.r_floor)
    if mode == "gaussian":
        return (fit_model - observed) / _gaussian_sigma(data)
    counts = data.intensity_raw[data.fit_mask]
    mu = data.normalization * fit_model
    try:
        return np.sign(mu - counts) * np.sqrt(poisson_deviance(counts, mu))
    except (ValueError, FloatingPointError) as error:
        raise EvaluationConstraintError(f"invalid_poisson_prediction:{error}") from error


def data_residual_model_derivative(
    problem: FitEvaluationContext, fit_model: np.ndarray, residual: np.ndarray
) -> np.ndarray:
    """Differentiate the mode residual with respect to normalized prediction."""
    data, mode = problem.data, problem.config.noise_model
    if mode == "robust_log":
        return 1 / ((fit_model + data.r_floor) * np.log(10.0))
    if mode == "gaussian":
        return 1 / _gaussian_sigma(data)
    counts = data.intensity_raw[data.fit_mask]
    mu = data.normalization * fit_model
    equal = mu == counts
    derivative = np.empty(mu.size, dtype=float)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        derivative[equal] = 1 / np.sqrt(mu[equal])
        derivative[~equal] = ((mu[~equal] - counts[~equal]) / mu[~equal]) / residual[~equal]
        return data.normalization * derivative


def _data_weights(problem: FitEvaluationContext) -> np.ndarray:
    mask = problem.data.fit_mask
    return problem.weights[mask] * np.sqrt(problem.sampling_multipliers[mask])


def data_objective(problem: FitEvaluationContext, residual: np.ndarray) -> float:
    """Return the data contribution to displayed J; priors are assembled once above it."""
    if problem.config.noise_model == "robust_log":
        return robust_log_cost(residual, _data_weights(problem), problem.config.c_decades) * (
            residual.size / problem.objective_point_count
        )
    mass = problem.sampling_multipliers[problem.data.fit_mask]
    return float(np.sum(mass * residual**2) / problem.objective_point_count)


def data_loss_rho(problem: FitEvaluationContext, squared: np.ndarray) -> np.ndarray:
    """Return SciPy data rho whose half-sum is the declared total data Q."""
    if problem.config.noise_model == "robust_log":
        return robust_loss_rho(squared, _data_weights(problem), problem.config.c_decades)
    mass = problem.sampling_multipliers[problem.data.fit_mask]
    return np.vstack((2 * mass * squared, 2 * mass, np.zeros_like(mass)))


def data_score_information(problem: FitEvaluationContext, residual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return dQ/dr and Q/2 residual-coordinate Gauss-Newton curvature."""
    if problem.config.noise_model == "robust_log":
        return robust_score_information(residual, _data_weights(problem), problem.config.c_decades)
    mass = problem.sampling_multipliers[problem.data.fit_mask]
    return 2 * mass * residual, mass


def data_statistical_information(
    problem: FitEvaluationContext,
    fit_model: np.ndarray,
    residual: np.ndarray,
    jacobian: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return total data bread and independent-score meat in unit coordinates.

    Likelihood modes use expected information, not deviance-residual curvature.
    Robust weights enter the estimating equation once and its variance twice.
    Only full observations support this independent-observation calibration.
    """
    if problem.config.noise_model == "robust_log":
        score, curvature = data_score_information(problem, residual)
        bread = jacobian.T @ (curvature[:, None] * jacobian)
        influence = (score / 2)[:, None] * jacobian
        return bread, influence.T @ influence
    if problem.config.noise_model == "poisson":
        mu = problem.data.normalization * fit_model
        if np.any(mu <= 0.0):
            raise StatisticalUnavailableError("poisson_nonregular_zero_mean")
        derivative = data_residual_model_derivative(problem, fit_model, residual)
        jacobian = jacobian * (problem.data.normalization / np.sqrt(mu) / derivative)[:, None]
    information = jacobian.T @ jacobian
    return information, information.copy()
