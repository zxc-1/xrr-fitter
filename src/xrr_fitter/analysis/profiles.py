"""Profile-likelihood scans and profile-path basin evidence.

A profile fixes one unit-space coordinate and reoptimizes every nuisance
coordinate. The reported x-axis may remain that unit coordinate or may be
mapped into a physical or binary-derived value. Objective values always come
from the shared evaluation boundary, so fitting and analysis use identical
residual, weighting, prior, and invalid-candidate conventions.

Each scan starts at the converged center and walks outward in both directions.
The previous nuisance optimum warms the next point, while the center and a
neutral midpoint remain deterministic fallback starts. Scalar minimization is
used for generic objectives; problem profiles use the analytic residual and
Jacobian path through bounded least squares.

Coarse support transitions are refined at their midpoints. A lower basin found
by a coarse point is recentered with all coordinates free before interval
closure is decided. Refinement is retained only when it finds threshold support
or expands the reported support envelope, preventing extra probes from changing
a stable profile without evidence.

The analysis domain reports a ``ProfileBasinDecision`` when the refined minimum
materially improves on the supplied center. It does not execute reconvergence.
Fit owns the four-path continuation and independently revalidates the decision,
which preserves the package dependency direction.

Cancellation is polled around every optimizer boundary and scan step. No partial
profile or basin decision is published after interruption. Stable ordering and
explicit objective deltas make the same profile replayable under frozen seeds.

Uncertainty analysis prepares every requested profile before scheduling work.
The lower and upper walks are independent because each begins with the center
nuisance state; all such walks are therefore flattened into one ordered task
batch. Their results retain explicit unit-grid positions, so completion timing
cannot alter array order. Refinement and mapping form a second ordered batch,
allowing expensive threshold probes from different parameters to overlap
without nesting calls into the shared thread pool. A single-profile caller uses
the identical plan and merge path serially.

Only service-owned code supplies the task runner. Plans contain process-local
closures and never enter a request, checkpoint, result, or project value. The
runner returns values in declaration order, and the final profile tuple follows
the requested parameter order even when directions or refinements complete in
the opposite order.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite

import numpy as np
from scipy.optimize import least_squares, minimize, minimize_scalar

from xrr_fitter.analysis.profile_tasks import (
    build_problem_profiles as _build_problem_profiles,
)
from xrr_fitter.evaluation import (
    EvaluationConstraintError,
    cached_least_squares_callbacks,
    evaluate_model,
    least_squares_loss,
    least_squares_system,
    values_by_name,
)
from xrr_fitter.model.analysis import ParameterProfile, ProfileBasinDecision
from xrr_fitter.model.fitting import FitEvaluationContext

Scalar = Callable[[np.ndarray], float]
Vector = Callable[[np.ndarray], np.ndarray]
# Profile closure never resolves an objective delta below 1e-5.
PROFILE_SOLVER_TOLERANCE = 1e-6
EXHAUSTIVE_PROFILE_LIMIT = 11


@dataclass(frozen=True, slots=True)
class _Callbacks:
    """Validated numerical hooks shared by every scan operation.

    The optional residual pair is all-or-nothing. Cancellation remains a
    separate poll operation so optimizers cannot accidentally treat it as an
    objective value.
    """

    objective: Scalar
    value_mapper: Scalar | None
    gradient: Vector | None
    residual: Vector | None
    residual_jacobian: Vector | None
    cancelled: Callable[[], bool] | None

    def poll(self) -> None:
        if self.cancelled is not None and self.cancelled():
            raise InterruptedError("cancelled")


@dataclass(frozen=True, slots=True)
class _Setup:
    """Immutable geometry and threshold state for one parameter profile.

    ``grid`` includes the supplied center exactly. ``nuisance`` preserves source
    coordinate order, and ``penalty`` is a finite optimizer-only replacement for
    nonfinite objective probes.
    """

    center: np.ndarray
    index: int
    lower: float
    upper: float
    grid: np.ndarray
    nuisance: tuple[int, ...]
    center_objective: float
    delta: float
    penalty: float
    name: str


@dataclass(frozen=True, slots=True)
class _ProfilePlan:
    """One validated scan whose two walks can execute independently."""

    callbacks: _Callbacks
    setup: _Setup
    residual_loss: object
    least_squares_max_nfev: int | None
    seek_basin: bool = False


def _profile_center(center_unit: np.ndarray) -> np.ndarray:
    center = np.asarray(center_unit, dtype=float)
    valid = (
        center.ndim == 1
        and center.size > 0
        and np.all(np.isfinite(center))
        and np.all((center >= 0.0) & (center <= 1.0))
    )
    if not valid:
        raise ValueError("profile center must be a nonempty finite unit vector")
    return center


def _validate_profile_index(parameter_index: int, size: int) -> None:
    if isinstance(parameter_index, bool) or not 0 <= int(parameter_index) < size:
        raise ValueError("profile parameter_index is out of range")


def _validate_profile_bounds(
    center: np.ndarray,
    parameter_index: int,
    lower: float,
    upper: float,
) -> None:
    if not all((isfinite(lower), isfinite(upper), 0.0 <= lower < upper <= 1.0)):
        raise ValueError("profile bounds must satisfy 0 <= lower < upper <= 1")
    if not lower <= center[parameter_index] <= upper:
        raise ValueError("profile center is outside requested bounds")


def _validate_profile_steps(steps: int) -> None:
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 5:
        raise ValueError("profile steps must be an integer of at least five")


def _validate_profile_callbacks(callbacks: _Callbacks) -> None:
    if (callbacks.residual is None) != (callbacks.residual_jacobian is None):
        raise ValueError("profile residual and residual_jacobian must be supplied together")


def _profile_delta(center_objective: float, objective_delta: float | None) -> float:
    delta = max(0.02 * abs(center_objective), 1e-5) if objective_delta is None else float(objective_delta)
    if not isfinite(delta) or delta <= 0.0:
        raise ValueError("profile objective_delta must be positive and finite")
    return delta


def _profile_grid(lower: float, upper: float, steps: int, center: float) -> np.ndarray:
    grid = np.linspace(lower, upper, int(steps))
    if not np.any(np.isclose(grid, center, rtol=0.0, atol=1e-15)):
        grid = np.sort(np.append(grid, center))
    return grid


def _prepare(
    callbacks: _Callbacks,
    center_unit: np.ndarray,
    parameter_index: int,
    lower: float,
    upper: float,
    steps: int,
    name: str | None,
    objective_delta: float | None,
) -> _Setup:
    """Validate scan declarations and evaluate the authoritative center.

    Bounds and step count are checked before invoking user callbacks. The center
    objective is evaluated once, cancellation is polled again, and the default
    material-improvement delta is derived from that finite value.
    """
    callbacks.poll()
    center = _profile_center(center_unit)
    _validate_profile_index(parameter_index, center.size)
    _validate_profile_bounds(center, parameter_index, lower, upper)
    _validate_profile_steps(steps)
    _validate_profile_callbacks(callbacks)
    center_objective = float(callbacks.objective(center.copy()))
    callbacks.poll()
    if not isfinite(center_objective):
        raise ValueError("profile center objective must be finite")
    delta = _profile_delta(center_objective, objective_delta)
    grid = _profile_grid(lower, upper, steps, float(center[parameter_index]))
    nuisance = tuple(index for index in range(center.size) if index != parameter_index)
    penalty = center_objective + max(1.0, abs(center_objective), 1000.0 * delta)
    return _Setup(
        center.copy(),
        int(parameter_index),
        float(lower),
        float(upper),
        grid,
        nuisance,
        center_objective,
        delta,
        penalty,
        str(parameter_index) if name is None else name,
    )


def _trial(setup: _Setup, fixed: float, nuisance: np.ndarray | None = None) -> np.ndarray:
    value = setup.center.copy()
    value[setup.index] = fixed
    if nuisance is not None:
        value[list(setup.nuisance)] = nuisance
    return value


def _finite_objective(callbacks: _Callbacks, setup: _Setup, unit: np.ndarray) -> float:
    callbacks.poll()
    value = float(callbacks.objective(unit))
    return value if isfinite(value) else setup.penalty


def _best_start(
    callbacks: _Callbacks,
    setup: _Setup,
    fixed: float,
    warm: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Choose the best distinct deterministic nuisance start.

    Warm, center, and midpoint starts are evaluated without optimization first.
    Duplicate vectors are skipped so equal starts do not consume extra work or
    perturb tie ordering.
    """
    starts = (
        warm,
        setup.center[list(setup.nuisance)],
        np.full(len(setup.nuisance), 0.5),
    )
    best, objective = np.array(starts[0], copy=True), np.inf
    seen: list[np.ndarray] = []
    for start in starts:
        callbacks.poll()
        if any(np.array_equal(start, previous) for previous in seen):
            continue
        seen.append(np.array(start, copy=True))
        observed = float(callbacks.objective(_trial(setup, fixed, start)))
        if isfinite(observed) and observed < objective:
            best, objective = np.array(start, copy=True), observed
    return best, objective


def _scalar_nuisance(
    callbacks: _Callbacks,
    setup: _Setup,
    fixed: float,
    start: np.ndarray,
):
    def objective(nuisance: np.ndarray) -> float:
        return _finite_objective(callbacks, setup, _trial(setup, fixed, nuisance))

    def gradient(nuisance: np.ndarray) -> np.ndarray:
        callbacks.poll()
        full = np.asarray(callbacks.gradient(_trial(setup, fixed, nuisance)), dtype=float)
        if full.shape != setup.center.shape or np.any(~np.isfinite(full)):
            raise ValueError("profile gradient must return a matching finite vector")
        return full[list(setup.nuisance)]

    return minimize(
        objective,
        start,
        method="L-BFGS-B",
        jac=None if callbacks.gradient is None else gradient,
        bounds=[(0.0, 1.0)] * start.size,
    )


def _residual_nuisance(
    callbacks: _Callbacks,
    setup: _Setup,
    fixed: float,
    start: np.ndarray,
    residual_loss: object,
    max_nfev: int | None,
):
    def residual(nuisance: np.ndarray) -> np.ndarray:
        callbacks.poll()
        return np.asarray(callbacks.residual(_trial(setup, fixed, nuisance)), dtype=float)

    def jacobian(nuisance: np.ndarray) -> np.ndarray:
        callbacks.poll()
        full = np.asarray(callbacks.residual_jacobian(_trial(setup, fixed, nuisance)), dtype=float)
        if full.ndim != 2 or full.shape[1] != setup.center.size:
            raise ValueError("profile residual_jacobian must have one column per parameter")
        return full[:, list(setup.nuisance)]

    return least_squares(
        residual,
        start,
        jac=jacobian,
        bounds=(0.0, 1.0),
        loss=residual_loss,
        ftol=PROFILE_SOLVER_TOLERANCE,
        xtol=PROFILE_SOLVER_TOLERANCE,
        gtol=PROFILE_SOLVER_TOLERANCE,
        x_scale="jac",
        max_nfev=max_nfev,
    )


def _profile_point(
    callbacks: _Callbacks,
    setup: _Setup,
    fixed: float,
    warm: np.ndarray,
    residual_loss: object,
    max_nfev: int | None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Optimize nuisance coordinates for one fixed profile coordinate.

    A no-nuisance profile evaluates directly. Otherwise the configured scalar or
    residual solver may improve the best deterministic start, but malformed or
    nonfinite optimizer output cannot replace that finite fallback.
    """
    callbacks.poll()
    if not setup.nuisance:
        trial = _trial(setup, fixed)
        return trial, float(callbacks.objective(trial)), warm
    start, best = _best_start(callbacks, setup, fixed, warm)
    optimized = (
        _scalar_nuisance(callbacks, setup, fixed, start)
        if callbacks.residual is None
        else _residual_nuisance(callbacks, setup, fixed, start, residual_loss, max_nfev)
    )
    callbacks.poll()
    nuisance = np.asarray(optimized.x, dtype=float)
    if nuisance.shape == start.shape and np.all(np.isfinite(nuisance)):
        nuisance = np.clip(nuisance, 0.0, 1.0)
        observed = float(callbacks.objective(_trial(setup, fixed, nuisance)))
        if isfinite(observed) and observed < best:
            start, best = nuisance, observed
    return _trial(setup, fixed, start), best, start


def _scan_direction(
    callbacks: _Callbacks,
    setup: _Setup,
    residual_loss: object,
    max_nfev: int | None,
    direction: int,
) -> tuple[tuple[int, ...], np.ndarray, np.ndarray]:
    """Walk one independent profile direction with its own warm state.

    The descending direction includes the supplied center; the ascending
    direction begins at the next grid point. Both initialize nuisance coordinates
    from the center, so neither can inherit a local basin found by the other.
    Explicit grid indices make later merging independent of task completion.
    """
    center_index = int(np.argmin(np.abs(setup.grid - setup.center[setup.index])))
    indices = tuple(range(center_index, -1, -1) if direction < 0 else range(center_index + 1, setup.grid.size))
    vectors = np.empty((len(indices), setup.center.size), dtype=float)
    objectives = np.full(len(indices), np.inf)
    warm = setup.center[list(setup.nuisance)]
    for offset, position in enumerate(indices):
        vector, objective, nuisance = _profile_point(
            callbacks,
            setup,
            float(setup.grid[position]),
            warm,
            residual_loss,
            max_nfev,
        )
        vectors[offset], objectives[offset] = vector, objective
        if isfinite(objective):
            warm = nuisance
    return indices, vectors, objectives


def _merge_direction_scans(
    setup: _Setup,
    scans: tuple[tuple[tuple[int, ...], np.ndarray, np.ndarray], ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Restore directional rows to their authoritative unit-grid positions.

    Assignment uses the carried integer positions rather than concatenation.
    Consequently reversed completion order produces the same warm-path values,
    objective array, and stable refinement input as the serial scan.
    """
    vectors = np.empty((setup.grid.size, setup.center.size), dtype=float)
    objectives = np.full(setup.grid.size, np.inf)
    for indices, directional_vectors, directional_objectives in scans:
        positions = np.asarray(indices, dtype=int)
        vectors[positions] = directional_vectors
        objectives[positions] = directional_objectives
    return vectors, objectives


def _scan(
    callbacks: _Callbacks,
    setup: _Setup,
    residual_loss: object,
    max_nfev: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Walk both independent directions and merge them in unit-grid order."""
    scans = tuple(
        _scan_direction(
            callbacks,
            setup,
            residual_loss,
            max_nfev,
            direction,
        )
        for direction in (-1, 1)
    )
    return _merge_direction_scans(setup, scans)


def _profile_minimum(objectives: np.ndarray) -> float:
    finite = objectives[np.isfinite(objectives)]
    return np.inf if finite.size == 0 else float(np.min(finite))


def _supported_points(objectives: np.ndarray, threshold: float) -> np.ndarray:
    return np.isfinite(objectives) & (objectives <= threshold)


def _reported_values(
    callbacks: _Callbacks,
    unit_values: np.ndarray,
    vectors: np.ndarray,
    cache: dict[bytes, float] | None = None,
) -> np.ndarray:
    if callbacks.value_mapper is None:
        return unit_values
    values = []
    for vector in vectors:
        key = np.asarray(vector, dtype=float).tobytes()
        if cache is None or key not in cache:
            mapped = float(callbacks.value_mapper(vector))
            if cache is not None:
                cache[key] = mapped
        values.append(mapped if cache is None else cache[key])
    values = np.asarray(values, dtype=float)
    callbacks.poll()
    return values


def _support_envelope(
    values: np.ndarray,
    objectives: np.ndarray,
    delta: float,
) -> tuple[float, float] | None:
    """Interpolate the outer reported interval supported within ``delta``.

    Crossing interpolation occurs in square-root objective height, matching the
    likelihood-ratio scale used by profile closure. Stable sorting permits
    derived values whose physical ordering differs from unit-grid ordering.
    """
    finite = np.isfinite(values) & np.isfinite(objectives)
    if not np.any(finite):
        return None
    order = np.argsort(values[finite], kind="stable")
    reported = values[finite][order]
    costs = objectives[finite][order]
    best = float(np.min(costs))
    supported = _supported_points(costs, best + delta)
    indices = np.flatnonzero(supported[:-1] != supported[1:])
    left_heights = np.sqrt(np.maximum(0.0, costs[indices] - best))
    right_heights = np.sqrt(np.maximum(0.0, costs[indices + 1] - best))
    with np.errstate(divide="ignore", invalid="ignore"):
        fractions = (np.sqrt(delta) - left_heights) / (right_heights - left_heights)
    fractions = np.where(np.isfinite(fractions), fractions, np.where(supported[indices], 0.0, 1.0))
    fractions = np.clip(fractions, 0.0, 1.0)
    crossings = reported[indices] + fractions * (reported[indices + 1] - reported[indices])
    supported_values = reported[supported]
    lower = np.concatenate((supported_values, crossings[~supported[indices]]))
    upper = np.concatenate((supported_values, crossings[supported[indices]]))
    return float(np.min(lower)), float(np.max(upper))


def _support_expanded(
    callbacks: _Callbacks,
    original: tuple[np.ndarray, np.ndarray, np.ndarray],
    refined: tuple[np.ndarray, np.ndarray, np.ndarray],
    delta: float,
    value_cache: dict[bytes, float],
) -> bool:
    original_envelope = _support_envelope(
        _reported_values(callbacks, original[0], original[1], value_cache),
        original[2],
        delta,
    )
    refined_envelope = _support_envelope(
        _reported_values(callbacks, refined[0], refined[1], value_cache),
        refined[2],
        delta,
    )
    if original_envelope is None or refined_envelope is None:
        return False
    values = np.asarray((*original_envelope, *refined_envelope), dtype=float)
    tolerance = 32.0 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(values))))
    return bool(
        original_envelope[0] - refined_envelope[0] > tolerance or refined_envelope[1] - original_envelope[1] > tolerance
    )


def _solve_recentered_profile(
    callbacks: _Callbacks,
    setup: _Setup,
    start: np.ndarray,
    residual_loss: object,
    max_nfev: int | None,
):
    if callbacks.residual is None:

        def objective(vector: np.ndarray) -> float:
            return _finite_objective(callbacks, setup, vector)

        def gradient(vector: np.ndarray) -> np.ndarray:
            callbacks.poll()
            value = np.asarray(callbacks.gradient(vector), dtype=float)
            if value.shape != setup.center.shape or np.any(~np.isfinite(value)):
                raise ValueError("profile gradient must return a matching finite vector")
            return value

        return minimize(
            objective,
            start,
            method="L-BFGS-B",
            jac=None if callbacks.gradient is None else gradient,
            bounds=[(0.0, 1.0)] * start.size,
        )

    def residual(vector: np.ndarray) -> np.ndarray:
        callbacks.poll()
        return np.asarray(callbacks.residual(vector), dtype=float)

    def jacobian(vector: np.ndarray) -> np.ndarray:
        callbacks.poll()
        value = np.asarray(callbacks.residual_jacobian(vector), dtype=float)
        if value.ndim != 2 or value.shape[1] != setup.center.size:
            raise ValueError("profile residual_jacobian must have one column per parameter")
        return value

    return least_squares(
        residual,
        start,
        bounds=(0.0, 1.0),
        jac=jacobian,
        loss=residual_loss,
        ftol=PROFILE_SOLVER_TOLERANCE,
        xtol=PROFILE_SOLVER_TOLERANCE,
        gtol=PROFILE_SOLVER_TOLERANCE,
        x_scale="jac",
        max_nfev=max_nfev,
    )


def _recentered_point(
    callbacks: _Callbacks,
    setup: _Setup,
    start: np.ndarray,
    coarse_objective: float,
    residual_loss: object,
    max_nfev: int | None,
) -> tuple[float, np.ndarray, float] | None:
    optimized = _solve_recentered_profile(
        callbacks,
        setup,
        start,
        residual_loss,
        max_nfev,
    )
    callbacks.poll()
    candidate = np.asarray(optimized.x, dtype=float)
    if candidate.shape != start.shape or np.any(~np.isfinite(candidate)):
        return None
    candidate = np.clip(candidate, 0.0, 1.0)
    objective = float(callbacks.objective(candidate))
    moved = not np.isclose(
        candidate[setup.index],
        start[setup.index],
        rtol=0.0,
        atol=1e-15,
    )
    if not isfinite(objective) or objective >= coarse_objective or not moved:
        return None
    return float(candidate[setup.index]), candidate, objective


def _merge_profile_points(
    values: np.ndarray,
    vectors: np.ndarray,
    objectives: np.ndarray,
    additions: tuple[tuple[float, np.ndarray, float], ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.concatenate((values, np.asarray([item[0] for item in additions])))
    vectors = np.vstack((vectors, np.vstack([item[1] for item in additions])))
    objectives = np.concatenate((objectives, np.asarray([item[2] for item in additions])))
    order = np.argsort(values, kind="stable")
    return values[order], vectors[order], objectives[order]


def _refine_transitions(
    callbacks: _Callbacks,
    setup: _Setup,
    vectors: np.ndarray,
    objectives: np.ndarray,
    residual_loss: object,
    max_nfev: int | None,
    value_cache: dict[bytes, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Refine threshold crossings and retain only evidence-bearing probes.

    A materially better coarse point first receives a full-dimensional recenter.
    Up to two midpoint depths then probe every support transition. The refined
    set replaces the coarse set only when a probe is supported or when mapped
    interval support expands beyond floating-point tolerance.
    """
    values = setup.grid
    best = _profile_minimum(objectives)
    if best + setup.delta < setup.center_objective:
        best_index = int(np.argmin(np.where(np.isfinite(objectives), objectives, np.inf)))
        addition = _recentered_point(
            callbacks,
            setup,
            vectors[best_index],
            float(objectives[best_index]),
            residual_loss,
            max_nfev,
        )
        if addition is not None:
            values, vectors, objectives = _merge_profile_points(
                values,
                vectors,
                objectives,
                (addition,),
            )
    original = values, vectors, objectives
    probe_objectives: list[float] = []
    for _depth in range(2):
        best = _profile_minimum(objectives)
        supported = _supported_points(objectives, best + setup.delta)
        crossings = np.flatnonzero(supported[:-1] != supported[1:])
        if crossings.size == 0:
            break
        additions = []
        for left_value in crossings:
            left = int(left_value)
            right = left + 1
            fixed = float(0.5 * (values[left] + values[right]))
            warm_index = left if supported[left] else right
            warm = vectors[warm_index, list(setup.nuisance)]
            vector, objective, _state = _profile_point(callbacks, setup, fixed, warm, residual_loss, max_nfev)
            additions.append((fixed, vector, objective))
            probe_objectives.append(objective)
        values, vectors, objectives = _merge_profile_points(
            values,
            vectors,
            objectives,
            tuple(additions),
        )
    refined = values, vectors, objectives
    discovered = bool(
        probe_objectives
        and np.any(
            _supported_points(
                np.asarray(probe_objectives, dtype=float),
                _profile_minimum(objectives) + setup.delta,
            )
        )
    )
    return (
        refined
        if discovered
        or _support_expanded(
            callbacks,
            original,
            refined,
            setup.delta,
            value_cache,
        )
        else original
    )


def _dense_basin_search(
    callbacks: _Callbacks,
    setup: _Setup,
    current_best: float,
) -> tuple[np.ndarray, float] | None:
    if setup.center.size != 1:
        return None
    grid = np.linspace(setup.lower, setup.upper, 257)
    costs = np.asarray([float(callbacks.objective(np.asarray([value]))) for value in grid])
    finite = np.where(np.isfinite(costs), costs, np.inf)
    index = int(np.argmin(finite))
    lower = grid[max(0, index - 1)]
    upper = grid[min(grid.size - 1, index + 1)]
    optimized = minimize_scalar(
        lambda value: _finite_objective(callbacks, setup, np.asarray([value])),
        bounds=(float(lower), float(upper)),
        method="bounded",
        options={"xatol": 1e-10, "maxiter": 64},
    )
    unit = np.asarray([float(optimized.x)])
    objective = float(callbacks.objective(unit))
    if isfinite(objective) and objective + setup.delta < current_best:
        return unit, objective
    return None


def _closed_sides(setup: _Setup, units: np.ndarray, objectives: np.ndarray) -> tuple[bool, bool]:
    threshold = setup.center_objective + setup.delta
    center = setup.center[setup.index]
    supported = np.isfinite(objectives) & (objectives >= threshold)
    return (
        bool(np.any(supported & (units >= setup.lower) & (units < center))),
        bool(np.any(supported & (units <= setup.upper) & (units > center))),
    )


def _prepare_profile_plan(
    objective: Scalar,
    center_unit: np.ndarray,
    *,
    parameter_index: int,
    lower: float = 0.0,
    upper: float = 1.0,
    steps: int = 41,
    name: str | None = None,
    objective_delta: float | None = None,
    value_mapper: Scalar | None = None,
    gradient: Vector | None = None,
    residual: Vector | None = None,
    residual_jacobian: Vector | None = None,
    residual_loss: object = "linear",
    least_squares_max_nfev: int | None = None,
    cancelled: Callable[[], bool] | None = None,
    seek_basin: bool = False,
) -> _ProfilePlan:
    """Validate callbacks and freeze all state needed after task submission."""
    callbacks = _Callbacks(objective, value_mapper, gradient, residual, residual_jacobian, cancelled)
    setup = _prepare(
        callbacks,
        center_unit,
        parameter_index,
        lower,
        upper,
        steps,
        name,
        objective_delta,
    )
    return _ProfilePlan(
        callbacks,
        setup,
        residual_loss,
        least_squares_max_nfev,
        seek_basin,
    )


def _scan_plan_direction(
    plan: _ProfilePlan,
    direction: int,
) -> tuple[tuple[int, ...], np.ndarray, np.ndarray]:
    """Adapt one immutable plan to the zero-argument task-runner boundary."""
    return _scan_direction(
        plan.callbacks,
        plan.setup,
        plan.residual_loss,
        plan.least_squares_max_nfev,
        direction,
    )


def _finish_profile_plan(
    plan: _ProfilePlan,
    scans: tuple[tuple[tuple[int, ...], np.ndarray, np.ndarray], ...],
) -> tuple[ParameterProfile, np.ndarray, float, float]:
    """Refine, map, close, and publish one pair of completed direction scans.

    Refinement remains a single task because threshold probes depend on the
    merged support envelope. Mapping and closure run only after every retained
    probe is fixed, so no partial profile can escape on cancellation or failure.
    """
    callbacks, setup = plan.callbacks, plan.setup
    vectors, objectives = _merge_direction_scans(setup, scans)
    value_cache: dict[bytes, float] = {}
    units, vectors, objectives = _refine_transitions(
        callbacks,
        setup,
        vectors,
        objectives,
        plan.residual_loss,
        plan.least_squares_max_nfev,
        value_cache,
    )
    finite = np.where(np.isfinite(objectives), objectives, np.inf)
    best_index = int(np.argmin(finite))
    best_unit = np.array(vectors[best_index], copy=True)
    best_objective = float(objectives[best_index])
    if plan.seek_basin:
        dense = _dense_basin_search(callbacks, setup, best_objective)
        if dense is not None:
            best_unit, best_objective = dense
            units = np.append(units, best_unit[setup.index])
            vectors = np.vstack((vectors, best_unit))
            objectives = np.append(objectives, best_objective)
            order = np.argsort(units, kind="stable")
            units, vectors, objectives = units[order], vectors[order], objectives[order]
    values = _reported_values(callbacks, units, vectors, value_cache)
    callbacks.poll()
    if np.any(~np.isfinite(values)):
        raise ValueError("profile value_mapper returned a nonfinite value")
    lower_closed, upper_closed = _closed_sides(setup, units, objectives)
    profile = ParameterProfile(setup.name, values, objectives, lower_closed, upper_closed)
    return profile, best_unit, best_objective, setup.delta


def _profile_scan(
    objective: Scalar,
    center_unit: np.ndarray,
    **options,
) -> tuple[ParameterProfile, np.ndarray, float, float]:
    """Run the complete scan, refinement, mapping, and closure pipeline."""
    plan = _prepare_profile_plan(objective, center_unit, **options)
    scans = tuple(_scan_plan_direction(plan, direction) for direction in (-1, 1))
    return _finish_profile_plan(plan, scans)


def profile_parameter(
    objective: Scalar,
    center_unit: np.ndarray,
    *,
    parameter_index: int,
    lower: float = 0.0,
    upper: float = 1.0,
    steps: int = 41,
    name: str | None = None,
    objective_delta: float | None = None,
    value_mapper: Scalar | None = None,
    gradient: Vector | None = None,
    residual: Vector | None = None,
    residual_jacobian: Vector | None = None,
    residual_loss: object = "linear",
    least_squares_max_nfev: int | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> ParameterProfile:
    """Build one profile without requesting a fitting continuation.

    Callers may supply either a scalar objective or a paired residual/Jacobian
    solver contract. The returned value contains only mapped coordinates,
    objectives, and closure flags.
    """
    return _profile_scan(
        objective,
        center_unit,
        parameter_index=parameter_index,
        lower=lower,
        upper=upper,
        steps=steps,
        name=name,
        objective_delta=objective_delta,
        value_mapper=value_mapper,
        gradient=gradient,
        residual=residual,
        residual_jacobian=residual_jacobian,
        residual_loss=residual_loss,
        least_squares_max_nfev=least_squares_max_nfev,
        cancelled=cancelled,
        seek_basin=False,
    )[0]


def profile_parameter_with_decision(
    objective: Scalar,
    center_unit: np.ndarray,
    **kwargs,
) -> tuple[ParameterProfile, ProfileBasinDecision | None]:
    """Build a profile and report materially better basin evidence.

    Dense basin search is enabled, unknown options are rejected, and the original
    center is re-evaluated for the final comparison. The decision carries no fit
    callback and performs no state mutation.
    """
    profile, best_unit, best_objective, delta = _profile_scan(
        objective,
        center_unit,
        parameter_index=kwargs.pop("parameter_index"),
        lower=kwargs.pop("lower", 0.0),
        upper=kwargs.pop("upper", 1.0),
        steps=kwargs.pop("steps", 41),
        name=kwargs.pop("name", None),
        objective_delta=kwargs.pop("objective_delta", None),
        value_mapper=kwargs.pop("value_mapper", None),
        gradient=kwargs.pop("gradient", None),
        residual=kwargs.pop("residual", None),
        residual_jacobian=kwargs.pop("residual_jacobian", None),
        residual_loss=kwargs.pop("residual_loss", "linear"),
        least_squares_max_nfev=kwargs.pop("least_squares_max_nfev", None),
        cancelled=kwargs.pop("cancelled", None),
        seek_basin=True,
    )
    if kwargs:
        raise TypeError(f"unexpected profile options: {tuple(kwargs)}")
    center_objective = float(objective(np.asarray(center_unit, dtype=float)))
    if best_objective + delta >= center_objective:
        return profile, None
    return profile, ProfileBasinDecision(
        profile.name,
        best_unit,
        best_objective,
        ("materially_better_profile_basin",),
    )


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
    delta = max(0.02 * abs(best), 1e-5) if objective_delta is None else objective_delta
    if value < x[0] or value > x[-1]:
        return False
    interpolated = float(np.interp(value, x, np.sqrt(np.maximum(0.0, y - best))))
    return interpolated <= np.sqrt(delta) + 32.0 * np.finfo(float).eps


def _path_objective(problem_or_objective: object) -> Scalar:
    if callable(problem_or_objective):
        return problem_or_objective

    def objective(unit: np.ndarray) -> float:
        evaluation = evaluate_model(problem_or_objective, unit)
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


def build_problem_profiles(
    problem: FitEvaluationContext,
    unit_vector: np.ndarray,
    names: tuple[str, ...],
    *,
    cancelled: Callable[[], bool] | None = None,
    task_runner=None,
) -> tuple[ParameterProfile, ...]:
    """Build ordered problem profiles through the shared two-phase task graph."""
    return _build_problem_profiles(
        problem,
        unit_vector,
        names,
        prepare_plan=_prepare_profile_plan,
        scan_plan_direction=_scan_plan_direction,
        finish_plan=_finish_profile_plan,
        evaluate=evaluate_model,
        cache_callbacks=cached_least_squares_callbacks,
        least_squares_system=least_squares_system,
        least_squares_loss=least_squares_loss,
        values_by_name=values_by_name,
        cancelled=cancelled,
        task_runner=task_runner,
    )


def build_problem_profile(
    problem: FitEvaluationContext,
    unit_vector: np.ndarray,
    name: str,
    cancelled: Callable[[], bool] | None = None,
) -> ParameterProfile:
    """Bind a compiled problem to the shared residual and Jacobian profile path."""
    return build_problem_profiles(
        problem,
        unit_vector,
        (name,),
        cancelled=cancelled,
    )[0]


def _structural_profile_names(names: tuple[str, ...]) -> set[str]:
    fragments = ("thickness", "period", "density", "roughness")
    return {
        name
        for name in names
        if name == "instrument.angle_offset_deg" or any(fragment in name for fragment in fragments)
    }


def _reported_profile_names(preliminary_report: object | None) -> set[str]:
    if preliminary_report is None:
        return set()
    selected = set(preliminary_report.boundary_hits)
    for first, second, _value in preliminary_report.strong_correlations:
        selected.update((first, second))
    return selected


def _degeneracy_profile_names(warnings: tuple[str, ...]) -> set[str]:
    selected: set[str] = set()
    for warning in warnings:
        if warning.startswith("\u539a\u5ea6-\u5bc6\u5ea6\u7b80\u5e76:"):
            selected.update(warning.split(":", 2)[1:])
    return selected


def _evidence_focused_layout(problem: object) -> bool:
    return len(problem.variables) > EXHAUSTIVE_PROFILE_LIMIT


def select_profile_names(
    problem: FitEvaluationContext,
    preliminary_report: object | None = None,
    *,
    degeneracy_warnings: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Select exhaustive small-problem or evidence-focused large-problem names.

    Large layouts retain structural coordinates, angle offset, boundary hits,
    correlated pairs, and thickness-density degeneracy participants. Derived
    binary profiles are appended in their declaration order.
    """
    from xrr_fitter.analysis.binary_profiles import binary_derived_profiles

    names = tuple(variable.name for variable in problem.variables)
    derived = tuple(item.name for item in binary_derived_profiles(problem))
    if not _evidence_focused_layout(problem):
        return names + derived
    required = _structural_profile_names(names)
    required.update(_reported_profile_names(preliminary_report))
    required.update(_degeneracy_profile_names(degeneracy_warnings))
    return tuple(name for name in names if name in required) + derived


def recover_profile_basin(
    problem: FitEvaluationContext,
    candidate: object,
    *,
    cancelled: Callable[[], bool] | None = None,
) -> ProfileBasinDecision | None:
    """Search thickness profiles for the first materially better basin.

    Already-good or invalid candidates do not trigger expensive recovery. Each
    eligible thickness coordinate is scanned in compiler order, and the first
    decision is returned for fit-owned four-path reconvergence.
    """
    unit = np.asarray(candidate.unit_vector, dtype=float)
    if not candidate.valid or not isfinite(candidate.objective):
        return None
    trigger = max(1e-3, 100.0 * problem.config.confidence.equivalent_cost_floor)
    if candidate.objective <= trigger:
        return None
    for name in (variable.name for variable in problem.variables if variable.name.endswith(".thickness_a")):
        index = tuple(variable.name for variable in problem.variables).index(name)

        def objective(value: np.ndarray) -> float:
            try:
                evaluation = evaluate_model(problem, value)
            except EvaluationConstraintError:
                return np.inf
            return evaluation.objective if evaluation.valid else np.inf

        _profile, decision = profile_parameter_with_decision(
            objective,
            unit,
            parameter_index=index,
            name=name,
            steps=11 if problem.config.budget.bootstrap_samples < 100 else 41,
            cancelled=cancelled,
        )
        if decision is not None:
            return decision
    return None
