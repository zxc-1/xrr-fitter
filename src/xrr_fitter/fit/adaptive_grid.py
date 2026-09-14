"""Deterministic grid, full-cost review, and search-budget decisions."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite

import numpy as np

from xrr_fitter.fit.feature_grid import feature_grid_indices
from xrr_fitter.fit.screening import fringe_extrema_qz


def grid_levels(full_count: int) -> tuple[int, ...]:
    if isinstance(full_count, bool) or not isinstance(full_count, (int, np.integer)) or full_count < 1:
        raise ValueError("full_count must be a positive integer")
    return tuple(level for level in (128, 256, 512) if level < full_count) + (int(full_count),)


def initial_grid_points(data: object) -> int:
    """Choose the first grid resolving every detected full fringe, or all rows."""
    mask = data.fit_mask
    count = int(np.count_nonzero(mask))
    extrema = fringe_extrema_qz(data.qz_a_inv[mask], data.intensity_normalized[mask], data.r_floor)
    for level in grid_levels(count)[:-1]:
        selected_qz = np.sort(data.qz_a_inv[feature_grid_indices(data, level)])
        starts = np.searchsorted(selected_qz, extrema[:-2], side="left")
        stops = np.searchsorted(selected_qz, extrema[2:], side="right")
        if np.all(stops - starts >= 8):
            return level
    return count


def _cost_pair(coarse: np.ndarray, full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first, second = np.asarray(coarse, dtype=float), np.asarray(full, dtype=float)
    if first.ndim != 1 or second.shape != first.shape or first.size == 0:
        raise ValueError("review costs must be nonempty aligned vectors")
    if np.any(np.isnan(first)) or np.any(np.isnan(second)) or np.any(first < 0) or np.any(second < 0):
        raise ValueError("review costs must be nonnegative or positive infinity")
    return first, second


def should_promote(
    coarse: np.ndarray | None = None,
    full: np.ndarray | None = None,
    *,
    relative_error: float | None = None,
    winner_changed: bool = False,
) -> bool:
    """Return whether a denser grid is required by cost or ranking drift.

    The scalar keyword form is useful at the decision boundary and mirrors the
    vector form used by the live review. It deliberately has no hidden scale or
    tolerance state, so replaying a decision only needs the recorded values.
    """
    if relative_error is not None:
        return _scalar_promotion(relative_error, winner_changed)
    if coarse is None or full is None:
        raise TypeError("coarse and full costs are required")
    first, second = _cost_pair(coarse, full)
    return _vector_promotion(first, second)


def _scalar_promotion(relative_error: float, winner_changed: bool) -> bool:
    if not isfinite(relative_error) or relative_error < 0.0:
        raise ValueError("relative_error must be finite and nonnegative")
    if not isinstance(winner_changed, bool):
        raise TypeError("winner_changed must be bool")
    return bool(relative_error > 0.05 or winner_changed)


def _cost_drift(first: np.ndarray, second: np.ndarray) -> bool:
    finite = np.isfinite(second)
    if np.any(np.isfinite(first) != finite):
        return True
    if not np.any(finite):
        return False
    # Comparing translated thresholds avoids cancellation at the exact 5% boundary.
    delta = 0.05 * np.maximum(np.abs(second[finite]), 1e-12)
    with np.errstate(over="ignore"):
        if np.any(first[finite] > second[finite] + delta) or np.any(first[finite] < second[finite] - delta):
            return True
    return False


def _ranking_drift(first: np.ndarray, second: np.ndarray) -> bool:
    coarse_winner, full_winner = int(np.argmin(first)), int(np.argmin(second))
    margin = max(0.02 * abs(second[full_winner]), 1e-12)
    return bool(coarse_winner != full_winner and second[coarse_winner] > second[full_winner] + margin)


def _vector_promotion(first: np.ndarray, second: np.ndarray) -> bool:
    return _cost_drift(first, second) or _ranking_drift(first, second)


def review_candidate_indices(
    unit_vectors: np.ndarray,
    objectives: np.ndarray,
    candidate_ids: tuple[str, ...],
) -> tuple[int, ...]:
    """Review four cost leaders plus four diverse representatives, or every row."""
    units = np.asarray(unit_vectors, dtype=float)
    costs = np.asarray(objectives, dtype=float)
    ids = tuple(candidate_ids)
    if units.ndim != 2 or costs.shape != (len(ids),) or units.shape[0] != len(ids):
        raise ValueError("review candidate axes must align")
    if len(set(ids)) != len(ids) or not np.all(np.isfinite(units)):
        raise ValueError("review requires unique candidate IDs and finite unit vectors")
    if np.any(np.isnan(costs)) or np.any(costs < 0):
        raise ValueError("review costs must be nonnegative or positive infinity")
    ranked = _ranked_indices(costs, ids)
    limit = min(8, len(ranked))
    if len(ranked) <= 8:
        return tuple(ranked)
    selected = ranked[:4]
    return _diverse_indices(units, costs, ids, ranked, selected, limit)


def _ranked_indices(costs: np.ndarray, ids: tuple[str, ...]) -> list[int]:
    return sorted(range(len(ids)), key=lambda index: (costs[index], ids[index]))


def _diverse_indices(
    units: np.ndarray,
    costs: np.ndarray,
    ids: tuple[str, ...],
    ranked: list[int],
    selected: list[int],
    limit: int,
) -> tuple[int, ...]:
    while len(selected) < limit:
        remaining = (index for index in ranked if index not in selected)

        def priority(index: int) -> tuple[float, float, str]:
            distances = np.linalg.norm(units[selected] - units[index], axis=1)
            return -float(np.min(distances)), float(costs[index]), ids[index]

        selected.append(min(remaining, key=priority))
    return tuple(sorted(selected, key=lambda index: (costs[index], ids[index])))


def next_budget_lineage(costs: Mapping[str, float], allocated_this_round: tuple[str, ...]) -> str | None:
    eligible = [key for key, cost in costs.items() if key not in allocated_this_round and isfinite(cost)]
    return min(eligible, key=lambda key: (costs[key], key), default=None)


class GenerationStagnation:
    """Track total-objective gain and new unit-space representatives per generation."""

    def __init__(self) -> None:
        self.best = float("inf")
        self.previous = float("inf")
        self.stagnant_generations = 0
        self.stopped = False
        self._representatives: list[np.ndarray] = []
        self._novel = False

    def observe(self, unit: np.ndarray, objective: float) -> None:
        if not isfinite(objective):
            return
        self.best = min(self.best, objective)
        value = np.asarray(unit, dtype=float)
        if value.ndim != 1 or not np.all(np.isfinite(value)):
            raise ValueError("generation candidates must be finite unit vectors")
        threshold = 0.05
        if not self._representatives or all(
            np.linalg.norm(value - previous) >= threshold for previous in self._representatives
        ):
            self._representatives.append(np.array(value, copy=True))
            self._novel = True

    def start_generations(self) -> None:
        self.previous = self.best
        self._novel = False

    def finish_generation(self) -> bool:
        small_gain = isfinite(self.previous) and self.previous - self.best < 1e-4 * max(abs(self.previous), 1e-12)
        self.stagnant_generations = self.stagnant_generations + 1 if small_gain and not self._novel else 0
        self.stopped = self.stagnant_generations >= 3
        self.previous = self.best
        self._novel = False
        return self.stopped
