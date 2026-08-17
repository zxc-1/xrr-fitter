"""Fit-specific invalid-candidate policy over the shared evaluation chain."""

from __future__ import annotations

import numpy as np

import xrr_fitter.evaluation as evaluation
from xrr_fitter.model.fitting import ModelEvaluation
from xrr_fitter.model.parameters import PhysicalValueError
from xrr_fitter.model.structure import ExpandedSlabLimitError


def _invalid_evaluation(
    problem: object,
    error: evaluation.EvaluationConstraintError,
) -> ModelEvaluation:
    qz = np.nan_to_num(
        problem.data.qz_a_inv,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    model = np.zeros_like(qz)
    residual = np.full(
        np.count_nonzero(problem.data.fit_mask),
        1e6,
        dtype=float,
    )
    weighted = problem.weights[problem.data.fit_mask] * residual
    return ModelEvaluation(
        valid=False,
        reason=error.reason,
        parameters=(),
        qz_a_inv=qz,
        model_normalized=model,
        fit_log_residuals_decades=residual,
        fit_weighted_residuals=weighted,
        objective=float("inf"),
        expanded_stack=None,
        diagnostics=error.diagnostics,
    )


def evaluate_vector(problem: object, unit_vector: np.ndarray) -> ModelEvaluation:
    """Evaluate a candidate and convert only declared physical failures."""
    try:
        return evaluation.evaluate_model(problem, unit_vector)
    except evaluation.EvaluationConstraintError as error:
        return _invalid_evaluation(problem, error)


def evaluate_declared_initial(problem: object) -> ModelEvaluation:
    """Evaluate the compiled declaration defaults through the real fit path."""
    try:
        unit = evaluation.encode_physical_vector(problem, {})
    except (
        evaluation.EvaluationConstraintError,
        PhysicalValueError,
        ExpandedSlabLimitError,
    ) as error:
        if not isinstance(error, evaluation.EvaluationConstraintError):
            error = evaluation.EvaluationConstraintError(f"constraint_violation:{type(error).__name__}")
        return _invalid_evaluation(problem, error)
    return evaluate_vector(problem, unit)


def evaluate_jacobian(problem: object, unit_vector: np.ndarray) -> np.ndarray:
    """Return the shared analytic fitted-residual Jacobian."""
    return evaluation.evaluate_model_jacobian(problem, unit_vector)
