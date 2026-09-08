"""Eligibility and total-to-display scaling for profile thresholds."""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from xrr_fitter.analysis.covariance import problem_covariance
from xrr_fitter.analysis.diagnostics import build_residual_evidence
from xrr_fitter.evaluation import evaluate_model
from xrr_fitter.model.fitting import FitEvaluationContext


def _unavailable_reason(problem, covariance, residual) -> str | None:
    if problem.scale_prior_center is not None or any(item.prior is not None for item in problem.parameter_definitions):
        return "prior_present"
    if covariance.matrix is None:
        return covariance.unavailable_reason
    return _diagnostic_reason(residual)


def _diagnostic_reason(residual) -> str | None:
    if not residual.executed:
        return "residual_diagnostics_not_executed"
    if residual.systematic or residual.autocorrelation or residual.diagnostics:
        return "residual_diagnostics_failed"
    return None


def _calibration_reason(problem, unit) -> str | None:
    if problem.config.noise_model == "robust_log":
        return "robust_objective_is_not_a_likelihood"
    evaluation = evaluate_model(problem, unit)
    values = np.full(problem.data.fit_mask.shape, np.nan)
    if evaluation.valid:
        values[problem.data.fit_mask] = evaluation.fit_residuals
    residual = build_residual_evidence(problem, values, evaluation.diagnostics)
    covariance = problem_covariance(problem, unit, (residual,))
    return _unavailable_reason(problem, covariance, residual)


def problem_profile_options(problem: FitEvaluationContext, unit: np.ndarray) -> dict[str, object]:
    reason = _calibration_reason(problem, unit)
    metadata = {
        "interval_kind": "likelihood_ratio" if reason is None else "loss_support",
        "confidence_level": 0.95 if reason is None else None,
        "method": "chi_square_1df_asymptotic" if reason is None else "objective_tolerance",
        "unavailable_reason": reason,
        "objective_point_count": problem.objective_point_count,
    }
    delta = float(chi2.ppf(0.95, 1)) / problem.objective_point_count if reason is None else None
    return {"objective_delta": delta, "interval_options": metadata}
