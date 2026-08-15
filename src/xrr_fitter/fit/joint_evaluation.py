"""Raw joint residuals, equal-dataset loss, and analytic global Jacobian."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite

import numpy as np

from xrr_fitter.evaluation import EvaluationConstraintError
from xrr_fitter.fit.joint_roughness import SHARED_ROUGHNESS_TRANSFORM
from xrr_fitter.fit.joint_scatter_jacobian import (
    joint_scatter_jacobians as _joint_scatter_jacobians,
)
from xrr_fitter.fit.joint_sharing import _raw_scatter, scatter_joint_vector
from xrr_fitter.fit.local_search import local_jacobian
from xrr_fitter.fit.objective import _invalid_evaluation, evaluate_vector
from xrr_fitter.model.fitting import ModelEvaluation


def _has_cross_dataset_constraints(problem: object) -> bool:
    return bool(problem.joint_constraint_rules)


def _readonly(value: object) -> np.ndarray:
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class JointEvaluation:
    valid: bool
    objective: float
    local_unit_vectors: tuple[np.ndarray, ...]
    local_evaluations: tuple[ModelEvaluation, ...]
    residuals: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "local_unit_vectors", tuple(_readonly(value) for value in self.local_unit_vectors))
        object.__setattr__(self, "local_evaluations", tuple(self.local_evaluations))
        object.__setattr__(self, "residuals", _readonly(self.residuals))


def _prior_residual(problem: object, evaluation: ModelEvaluation) -> float | None:
    if problem.scale_prior_center is None:
        return None
    if not evaluation.valid:
        return 0.0
    scale = next(value.value for value in evaluation.parameters if value.name == "instrument.scale")
    return (np.log10(scale) - np.log10(problem.scale_prior_center)) / problem.scale_prior_tau_decades


def _evaluation_residual(problem: object, evaluation: ModelEvaluation) -> np.ndarray:
    values = np.asarray(evaluation.fit_log_residuals_decades, dtype=float)
    prior = _prior_residual(problem, evaluation)
    return values if prior is None else np.concatenate((values, np.asarray([prior])))


def evaluate_joint_vector(problem: object, global_unit: np.ndarray) -> JointEvaluation:
    """Evaluate every local projection and report their arithmetic mean cost."""
    try:
        local_units = scatter_joint_vector(problem, global_unit)
    except EvaluationConstraintError as error:
        local_units = tuple(
            _raw_scatter(
                problem,
                np.asarray(global_unit, dtype=float),
            )
        )
        evaluations = tuple(_invalid_evaluation(local_problem, error) for local_problem in problem.problems)
        residuals = np.concatenate(
            tuple(
                _evaluation_residual(local_problem, evaluation)
                for local_problem, evaluation in zip(
                    problem.problems,
                    evaluations,
                    strict=True,
                )
            )
        )
        return JointEvaluation(
            False,
            float("inf"),
            local_units,
            evaluations,
            residuals,
        )
    evaluations = tuple(
        evaluate_vector(local_problem, unit) for local_problem, unit in zip(problem.problems, local_units, strict=True)
    )
    valid = all(value.valid for value in evaluations)
    objective = float(np.mean([value.objective for value in evaluations])) if valid else float("inf")
    residuals = np.concatenate(
        tuple(
            _evaluation_residual(local_problem, evaluation)
            for local_problem, evaluation in zip(problem.problems, evaluations, strict=True)
        )
    )
    return JointEvaluation(valid, objective, local_units, evaluations, residuals)


def _loss_block(
    squared: np.ndarray,
    weights: np.ndarray,
    c_decades: float,
    alpha: float,
) -> np.ndarray:
    scaled = 1.0 + squared / c_decades**2
    return np.vstack(
        (
            4.0 * alpha * weights**2 * c_decades**2 * (np.sqrt(scaled) - 1.0),
            2.0 * alpha * weights**2 / np.sqrt(scaled),
            -(alpha * weights**2 / c_decades**2) * scaled ** (-1.5),
        )
    )


def _loss_layout(local_problem: object) -> tuple[int, np.ndarray, float, bool]:
    fit_mask = np.asarray(local_problem.data.fit_mask)
    weights = np.asarray(local_problem.weights, dtype=float)
    if fit_mask.ndim != 1 or weights.ndim != 1 or weights.shape != fit_mask.shape:
        raise ValueError("joint loss weight layout must match each one-dimensional dataset")
    size = int(np.count_nonzero(fit_mask))
    if size == 0:
        raise ValueError("joint loss dataset layout must contain fitted rows")
    fit_weights = np.array(weights[fit_mask], dtype=float, copy=True)
    if not np.all(np.isfinite(fit_weights)) or np.any(fit_weights <= 0.0):
        raise ValueError("joint loss weights must be finite and strictly positive")
    c_decades = local_problem.config.c_decades
    if not isfinite(c_decades) or c_decades <= 0.0:
        raise ValueError("joint loss c_decades must be positive and finite")
    return size, fit_weights, float(c_decades), local_problem.scale_prior_center is not None


def _validated_squared_residuals(squared: np.ndarray, row_count: int) -> np.ndarray:
    values = np.asarray(squared, dtype=float)
    if values.ndim != 1 or values.size != row_count or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("joint loss squared residual rows must be finite, nonnegative, and match the compiled layout")
    return values


def joint_least_squares_loss(problem: object) -> Callable[[np.ndarray], np.ndarray]:
    """Return SciPy's rho callable with exact equal dataset mass."""
    problems = tuple(problem.problems)
    if not problems:
        raise ValueError("joint loss requires a nonempty dataset layout")
    frozen_layouts = tuple(_loss_layout(local_problem) for local_problem in problems)
    sizes = tuple(layout[0] for layout in frozen_layouts)
    total_data = sum(sizes)
    row_count = sum(size + int(has_prior) for size, _weights, _c_decades, has_prior in frozen_layouts)

    def loss(squared: np.ndarray) -> np.ndarray:
        values = _validated_squared_residuals(squared, row_count)
        blocks: list[np.ndarray] = []
        offset = 0
        for size, weights, c_decades, has_prior in frozen_layouts:
            alpha = total_data / (len(sizes) * size)
            data = values[offset : offset + size]
            blocks.append(_loss_block(data, weights, c_decades, alpha))
            offset += size
            if has_prior:
                prior = values[offset]
                blocks.append(np.asarray(((2.0 * alpha * prior,), (2.0 * alpha,), (0.0,))))
                offset += 1
        if offset != values.size:
            raise ValueError("joint loss row layout mismatch")
        return np.hstack(blocks)

    return loss


def evaluate_joint_jacobian(problem: object, global_unit: np.ndarray) -> np.ndarray:
    """Scatter local analytic columns into the exact global coordinate order."""
    try:
        local_units = scatter_joint_vector(problem, global_unit)
    except EvaluationConstraintError:
        row_count = sum(
            int(np.count_nonzero(local_problem.data.fit_mask)) + int(local_problem.scale_prior_center is not None)
            for local_problem in problem.problems
        )
        return _readonly(np.zeros((row_count, len(problem.global_variables)), dtype=float))
    rows = []
    width = len(problem.global_variables)
    physical_roughness = any(variable.transform == SHARED_ROUGHNESS_TRANSFORM for variable in problem.global_variables)
    scatter_jacobians = (
        _joint_scatter_jacobians(problem, global_unit)
        if physical_roughness or _has_cross_dataset_constraints(problem)
        else None
    )
    for dataset_index, (local_problem, unit, scatter) in enumerate(
        zip(
            problem.problems,
            local_units,
            problem.scatter_maps,
            strict=True,
        )
    ):
        local = local_jacobian(local_problem, unit)
        if scatter_jacobians is not None:
            rows.append(local @ scatter_jacobians[dataset_index])
            continue
        block = np.zeros((local.shape[0], width), dtype=float)
        for local_index, global_index in enumerate(scatter):
            if global_index < 0:
                continue
            block[:, global_index] += local[:, local_index]
        rows.append(block)
    return _readonly(np.vstack(rows))
