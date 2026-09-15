"""Strict full-observation statistical information, separate from solver sentinels."""

from __future__ import annotations

import numpy as np

from xrr_fitter.evaluation_instrument_jacobian import evaluate_model_jacobian
from xrr_fitter.evaluation_model import evaluate_model
from xrr_fitter.evaluation_solver import _scale_prior_jacobian
from xrr_fitter.evaluation_statistics import StatisticalUnavailableError, data_statistical_information
from xrr_fitter.model.fitting import FitEvaluationContext


def _validate_full_observations(problem: FitEvaluationContext) -> None:
    selected = problem.data.fit_mask
    if np.count_nonzero(selected) != problem.objective_point_count or np.any(
        problem.sampling_multipliers[selected] != 1.0
    ):
        raise StatisticalUnavailableError("covariance_requires_full_observations")


def _inference_evaluation(problem: FitEvaluationContext, unit_vector: np.ndarray):
    result = evaluate_model(problem, unit_vector)
    if not result.valid:
        raise StatisticalUnavailableError(f"invalid_inference_point:{result.reason}")
    model = result.model_normalized[problem.data.fit_mask]
    if problem.config.noise_model == "poisson" and np.any(model <= 0.0):
        raise StatisticalUnavailableError("poisson_nonregular_zero_mean")
    return result, model


def statistical_information(
    problem: FitEvaluationContext, unit_vector: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return data bread, prior bread, and score meat; never use solver fallbacks."""
    _validate_full_observations(problem)
    result, model = _inference_evaluation(problem, unit_vector)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        jacobian = evaluate_model_jacobian(problem, unit_vector)
        data_bread, meat = data_statistical_information(problem, model, result.fit_residuals, jacobian)
        prior_jacobian = _scale_prior_jacobian(problem, unit_vector)
        prior_bread = np.outer(prior_jacobian, prior_jacobian)
    matrices = (data_bread, prior_bread, meat)
    if any(np.any(~np.isfinite(matrix)) for matrix in matrices):
        raise FloatingPointError("statistical information is not finite")
    for matrix in matrices:
        matrix.setflags(write=False)
    return matrices
