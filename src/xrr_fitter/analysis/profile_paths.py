"""Recorded profile support and objective-barrier tests between candidate basins."""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from xrr_fitter.evaluation import evaluate_model
from xrr_fitter.model.analysis import ParameterProfile

Scalar = Callable[[np.ndarray], float]


def profile_covers_value(
    profile: ParameterProfile,
    value: float,
    *,
    objective_delta: float | None = None,
) -> bool:
    values = np.asarray(profile.values, dtype=float)
    objectives = np.asarray(profile.objectives, dtype=float)
    finite = np.isfinite(values) & np.isfinite(objectives)
    if not np.any(finite):
        return False
    order = np.argsort(values[finite], kind="stable")
    x, y = values[finite][order], objectives[finite][order]
    best = float(np.min(y))
    delta = profile.objective_delta if objective_delta is None else objective_delta
    if delta is None:
        return False
    if value < x[0] or value > x[-1]:
        return False
    interpolated = float(np.interp(value, x, np.sqrt(np.maximum(0.0, y - best))))
    return interpolated <= np.sqrt(delta) + 32.0 * np.finfo(float).eps


def _path_objective(problem_or_objective: object) -> Scalar:
    if callable(problem_or_objective):
        return problem_or_objective

    def objective(unit: np.ndarray) -> float:
        evaluation = evaluate_model(problem_or_objective, unit, fit_only=True)
        return evaluation.objective if evaluation.valid else np.inf

    return objective


def default_profile_path_merge(
    problem_or_objective: object,
    first_unit: np.ndarray,
    second_unit: np.ndarray,
    threshold: float,
) -> bool:
    """Prove that two candidate endpoints share a sub-threshold profile path.

    Eleven fixed fractions establish the coarse path. At each fraction SLSQP may
    relax coordinates only on the hyperplane normal to the endpoint direction.
    Bounded scalar maximization then searches every fraction interval for a
    missed objective barrier.
    """
    first, second = np.asarray(first_unit, dtype=float), np.asarray(second_unit, dtype=float)
    direction = second - first
    if direction.shape != first.shape:
        raise ValueError("profile path endpoints must have matching shapes")
    if float(direction @ direction) <= 1e-24:
        return True
    objective = _path_objective(problem_or_objective)
    penalty = threshold + max(1.0, abs(threshold))

    def cost(fraction: float) -> float:
        base = (1.0 - fraction) * first + fraction * second

        def safe(unit: np.ndarray) -> float:
            value = float(objective(np.asarray(unit, dtype=float)))
            return value if isfinite(value) else penalty

        constraint = {
            "type": "eq",
            "fun": lambda unit: float((np.asarray(unit) - base) @ direction),
        }
        optimized = minimize(
            safe,
            base,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * base.size,
            constraints=(constraint,),
            options={"ftol": 1e-10, "maxiter": 100},
        )
        candidates = [base]
        if optimized.x.shape == base.shape and np.all(np.isfinite(optimized.x)):
            candidates.append(np.clip(optimized.x, 0.0, 1.0))
        return min(safe(candidate) for candidate in candidates)

    fractions = np.linspace(0.0, 1.0, 11)
    if any(cost(float(fraction)) > threshold for fraction in fractions):
        return False
    for lower, upper in zip(fractions[:-1], fractions[1:], strict=True):
        maximum = minimize_scalar(
            lambda fraction: -cost(float(fraction)),
            bounds=(float(lower), float(upper)),
            method="bounded",
            options={"xatol": 1e-5, "maxiter": 32},
        )
        if isfinite(maximum.fun) and -float(maximum.fun) > threshold:
            return False
    return True
