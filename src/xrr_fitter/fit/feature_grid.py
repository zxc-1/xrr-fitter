"""Select real measured rows for every feature-preserving search grid.

Fitted endpoints and sharp log-reflectivity features are protected while the
remaining rows provide uniform coverage in q order. Both adaptive resolution
checks and compiled grid contexts use this single deterministic sampler.
"""

from __future__ import annotations

import numpy as np


def _ranked_feature_positions(data: object, by_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qz = data.qz_a_inv[by_q]
    intensity = np.log10(np.clip(data.intensity_normalized[by_q], data.r_floor, np.inf))
    span = qz[2:] - qz[:-2]
    fraction = np.divide(
        qz[1:-1] - qz[:-2],
        span,
        out=np.full(span.shape, 0.5),
        where=span > 0.0,
    )
    baseline = intensity[:-2] + fraction * (intensity[2:] - intensity[:-2])
    scores = np.abs(intensity[1:-1] - baseline)
    median = float(np.median(scores))
    deviation = float(np.median(np.abs(scores - median)))
    numerical_floor = np.finfo(float).eps * max(1.0, float(np.max(np.abs(intensity))))
    reliable = np.flatnonzero(scores > median + max(5.0 * deviation, numerical_floor)) + 1
    return qz, reliable[np.lexsort((reliable, -scores[reliable - 1]))]


def _spaced_features(ranked: np.ndarray, budget: int) -> tuple[int, ...]:
    selected: list[int] = []
    for value in ranked:
        position = int(value)
        if any(abs(position - chosen) <= 1 for chosen in selected):
            continue
        selected.append(position)
        if len(selected) == budget:
            break
    return tuple(selected)


def _insert_features(
    selected: set[int],
    protected: set[int],
    features: tuple[int, ...],
    qz: np.ndarray,
) -> None:
    for position in features:
        if position in selected:
            protected.add(position)
            continue
        replaceable = selected - protected
        if not replaceable:
            return
        nearest = min(
            replaceable,
            key=lambda current: (abs(qz[current] - qz[position]), current),
        )
        selected.remove(nearest)
        selected.add(position)
        protected.add(position)


def feature_grid_indices(data: object, max_points: int = 128) -> np.ndarray:
    """Select a bounded q grid while retaining endpoints and sharp features."""
    if isinstance(max_points, bool) or not isinstance(max_points, (int, np.integer)):
        raise ValueError("max_points must be an integer of at least two")
    if max_points < 2:
        raise ValueError("max_points must be an integer of at least two")
    fit_indices = np.flatnonzero(data.fit_mask)
    if fit_indices.size == 0:
        raise ValueError("fit data contain no enabled points")
    if fit_indices.size <= max_points:
        return np.sort(fit_indices)
    by_q = fit_indices[np.argsort(data.qz_a_inv[fit_indices], kind="stable")]
    positions = np.rint(np.linspace(0, by_q.size - 1, max_points)).astype(int)
    selected = {int(position) for position in positions}
    protected = {0, by_q.size - 1}
    qz, ranked = _ranked_feature_positions(data, by_q)
    budget = min(max(1, max_points // 4), max_points - 2)
    _insert_features(selected, protected, _spaced_features(ranked, budget), qz)
    return np.sort(by_q[np.asarray(sorted(selected), dtype=int)])
