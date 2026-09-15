"""Fixed complete-score localization followed by bounded TRF refinement.

Localization never certifies the final fit. Every path reserves a charged full
handoff validation and refinement within its original budget, including when
L-BFGS-B stops abnormally or reaches its localization cap. Its last accepted
coordinate is the only handoff; returned fun and rejected trials are not used.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares, minimize

from xrr_fitter.evaluation import EvaluationConstraintError, cached_least_squares_callbacks
from xrr_fitter.fit.joint_solvers import poll
from xrr_fitter.model.diagnostic_work import DiagnosticSolverExit as StageExit
from xrr_fitter.model.parameters import PhysicalValueError
from xrr_fitter.model.structure import ExpandedSlabLimitError

NUMERICAL_ERRORS = (
    EvaluationConstraintError,
    PhysicalValueError,
    ExpandedSlabLimitError,
    FloatingPointError,
    OverflowError,
    np.linalg.LinAlgError,
)


def numerical_reason(error: Exception) -> str:
    return f"diagnostic_refit_numerical_failure:{type(error).__name__}:{error}"


class _BudgetReached(Exception):
    """Internal control flow: no evaluation has occurred beyond the hard cap."""


def _unit_copy(unit, shape):
    value = np.array(unit, dtype=float, copy=True)
    if value.shape != shape or value.ndim != 1:
        raise ValueError("diagnostic solver coordinate has an invalid shape")
    if np.any(~np.isfinite(value)):
        raise FloatingPointError("nonfinite diagnostic coordinate")
    if np.any((value < 0.0) | (value > 1.0)):
        raise EvaluationConstraintError("diagnostic coordinate is outside unit bounds")
    return value


class TwoStageSolver:
    """One path's numerical state; publication remains the refitter's job."""

    def __init__(self, system, loss, validate, *, budget, poisson, cancelled=None):
        if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
            raise ValueError("diagnostic solver budget must be a positive integer")
        self._system = system
        self._residual, self._jacobian = cached_least_squares_callbacks(self._system_at)
        self._loss, self._validate = loss, validate
        self._cancelled, self._poisson = cancelled, poisson
        self.budget = budget
        self.nfev = 0
        self._limit = 0
        self._exits = []
        self._initial = self._accepted = self._last_evaluated = self._solution = None
        self._last_residual = None

    @property
    def exits(self):
        return tuple(self._exits)

    def _request(self):
        poll(self._cancelled)
        if self.nfev >= self._limit:
            raise _BudgetReached()
        self.nfev += 1

    def _system_at(self, unit):
        residual, jacobian = self._system(unit)
        poll(self._cancelled)
        if residual.ndim != 1 or jacobian.shape != (residual.size, self._initial.size):
            raise ValueError("diagnostic system axes must match residual and coordinate dimensions")
        if np.any(~np.isfinite(residual)):
            raise FloatingPointError("nonfinite diagnostic residual")
        # TRF may reject a trial without asking for J. Validate the entire
        # computed system before caching, not just the component it requests.
        if np.any(~np.isfinite(jacobian)):
            raise FloatingPointError("nonfinite diagnostic Jacobian")
        return residual, jacobian

    def residual(self, unit):
        self._request()
        value = self._residual(unit)
        poll(self._cancelled)
        if value.ndim != 1:
            raise ValueError("diagnostic residual must be one-dimensional")
        if np.any(~np.isfinite(value)):
            raise FloatingPointError("nonfinite diagnostic residual")
        self._last_residual = _unit_copy(unit, self._initial.shape)
        return value

    def jacobian(self, unit):
        poll(self._cancelled)
        if not np.array_equal(unit, self._last_residual):
            raise RuntimeError("diagnostic Jacobian requires a matching charged residual request")
        value = self._jacobian(unit)
        poll(self._cancelled)
        if value.ndim != 2 or value.shape[1] != self._initial.size:
            raise ValueError("diagnostic Jacobian must match the coordinate dimension")
        if np.any(~np.isfinite(value)):
            raise FloatingPointError("nonfinite diagnostic Jacobian")
        return value

    def _value_gradient(self, unit):
        residual, jacobian = self.residual(unit), self.jacobian(unit)
        rho = self._loss(residual * residual)
        score = float(0.5 * np.sum(rho[0]))
        gradient = jacobian.T @ (rho[1] * residual)
        if not np.isfinite(score) or np.any(~np.isfinite(gradient)):
            raise FloatingPointError("nonfinite diagnostic complete score or gradient")
        self._last_evaluated = _unit_copy(unit, self._initial.shape)
        if self._accepted is None:
            if not np.array_equal(unit, self._initial):
                raise RuntimeError("localization must first evaluate its declared start")
            self._accepted = self._last_evaluated.copy()
        return score, gradient

    def _accept(self, unit):
        poll(self._cancelled)
        accepted = _unit_copy(unit, self._initial.shape)
        if not np.array_equal(accepted, self._last_evaluated):
            raise RuntimeError("localization accepted a coordinate without its matching evaluation")
        self._accepted = accepted
        poll(self._cancelled)

    def _stage(self, phase, operation):
        before = self.nfev
        poll(self._cancelled)
        try:
            with np.errstate(over="raise", invalid="raise"):
                status, success, message, kind = operation()
        except NUMERICAL_ERRORS as error:
            poll(self._cancelled)
            self._exits.append(
                StageExit(phase, None, False, numerical_reason(error), self.nfev - before, "numerical_failure")
            )
            raise
        poll(self._cancelled)
        result = StageExit(phase, status, success, message, self.nfev - before, kind)
        self._exits.append(result)
        return result

    def _localize(self):
        self._limit = (self.budget - 1) // 2
        try:
            solved = minimize(
                self._value_gradient,
                self._initial.copy(),
                jac=True,
                bounds=[(0.0, 1.0)] * self._initial.size,
                method="L-BFGS-B",
                options={"ftol": 1e-10, "gtol": 1e-10, "maxfun": self._limit, "maxiter": self._limit},
                callback=self._accept,
            )
        except _BudgetReached:
            return None, False, "diagnostic_localization_budget_exhausted", "budget"
        poll(self._cancelled)
        if self._accepted is None:
            raise RuntimeError("localization returned without evaluating its declared start")
        return int(solved.status), bool(solved.success), str(solved.message), "native"

    def _handoff(self):
        self._limit = self.budget
        self._request()
        reason = self._validate(self._accepted.copy())
        poll(self._cancelled)
        return None, reason is None, "diagnostic_handoff_validated" if reason is None else reason, "validation"

    def _refine(self):
        try:
            solved = least_squares(
                self.residual,
                self._accepted.copy(),
                jac=self.jacobian,
                bounds=(0.0, 1.0),
                method="trf",
                loss=self._loss,
                x_scale="jac",
                ftol=None if self._poisson else 1e-10,
                xtol=1e-10,
                gtol=1e-10,
                max_nfev=self.budget - self.nfev,
                callback=lambda *_args, **_kwargs: poll(self._cancelled),
            )
        except _BudgetReached:
            return None, False, "diagnostic_refinement_budget_exhausted", "budget"
        poll(self._cancelled)
        if solved.success:
            self._solution = _unit_copy(solved.x, self._initial.shape)
        return int(solved.status), bool(solved.success), str(solved.message), "native"

    def solve(self, start):
        poll(self._cancelled)
        self._initial = _unit_copy(start, np.asarray(start).shape)
        if not self._initial.size:
            return self._initial.copy(), None
        if self.budget < 3:
            reason = "diagnostic_refit_insufficient_two_stage_budget"
            self._exits.append(StageExit("localize", None, False, reason, 0, "budget"))
            return None, reason
        self._stage("localize", self._localize)
        handoff = self._stage("handoff", self._handoff)
        if not handoff.success:
            return None, handoff.message
        refined = self._stage("refine", self._refine)
        if not refined.success:
            return None, f"diagnostic_refit_nonconverged:{refined.message}"
        poll(self._cancelled)
        return self._solution.copy(), None
