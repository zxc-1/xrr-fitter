"""Reliable fringe-window detection and deterministic candidate screening."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from xrr_fitter.fit.initialization import structure_evidence
from xrr_fitter.model.fitting import FitCandidate


@dataclass(frozen=True, slots=True)
class FringeScreenResult:
    candidates: tuple[FitCandidate, ...]
    disabled: bool
    warnings: tuple[str, ...] = ()
    stop_reason: str | None = None


def fringe_extrema_qz(
    qz_a_inv: np.ndarray,
    normalized: np.ndarray,
    r_floor: float,
) -> np.ndarray:
    """Return robust detrended q^4 R maxima and minima on a uniform q grid."""
    qz = np.asarray(qz_a_inv, dtype=float)
    curve = np.asarray(normalized, dtype=float)
    if qz.ndim != 1 or curve.shape != qz.shape or qz.size < 16:
        return np.empty(0, dtype=float)
    finite = np.isfinite(qz) & np.isfinite(curve)
    if np.count_nonzero(finite) < 16:
        return np.empty(0, dtype=float)
    order = np.argsort(qz[finite], kind="stable")
    qz = qz[finite][order]
    curve = curve[finite][order]
    uniform_qz = np.linspace(qz[0], qz[-1], qz.size)
    transformed = uniform_qz**4 * np.maximum(
        np.interp(uniform_qz, qz, curve),
        r_floor,
    )
    detrended = signal.detrend(transformed, type="linear")
    differences = np.diff(detrended)
    baseline = float(np.median(differences))
    mad = float(np.median(np.abs(differences - baseline)))
    numerical_floor = np.finfo(float).eps * max(
        1.0,
        float(np.max(np.abs(detrended))),
    )
    prominence = 5.0 * max(mad, numerical_floor)
    maxima, _ = signal.find_peaks(detrended, prominence=prominence)
    minima, _ = signal.find_peaks(-detrended, prominence=prominence)
    return uniform_qz[np.sort(np.concatenate((maxima, minima)))]


def _longest_accepted_run(accepted: np.ndarray) -> tuple[int, int]:
    best_start = 0
    best_stop = 0
    run_start = 0
    for gap_index, is_accepted in enumerate(accepted):
        if is_accepted:
            continue
        if gap_index - run_start > best_stop - best_start:
            best_start, best_stop = run_start, gap_index
        run_start = gap_index + 1
    if accepted.size - run_start > best_stop - best_start:
        best_start, best_stop = run_start, accepted.size
    return best_start, best_stop


def _best_fringe_span(
    extrema_qz: np.ndarray,
    peak_positions_a: tuple[float, ...],
) -> tuple[int, int]:
    best_start = 0
    best_stop = 0
    gaps = np.diff(extrema_qz)
    for thickness_a in peak_positions_a:
        expected = np.pi / thickness_a
        accepted = (gaps >= 0.45 * expected) & (gaps <= 1.75 * expected)
        start, stop = _longest_accepted_run(accepted)
        if stop - start > best_stop - best_start:
            best_start, best_stop = start, stop
    return best_start, best_stop


def reliable_fringe_window(problem: object) -> tuple[float, float, int] | None:
    """Return a structure-supported window containing at least four extrema."""
    fit_mask = problem.data.fit_mask
    qz = problem.data.qz_a_inv[fit_mask]
    observed = problem.data.intensity_normalized[fit_mask]
    order = np.argsort(qz, kind="stable")
    qz = qz[order]
    observed = observed[order]
    if qz.size < 16 or np.ptp(qz) <= 0.0:
        return None
    evidence = structure_evidence(problem.data, problem.structure)
    if not evidence.peak_positions_a:
        return None
    extrema = fringe_extrema_qz(qz, observed, problem.data.r_floor)
    if extrema.size < 4:
        return None
    start, stop = _best_fringe_span(extrema, evidence.peak_positions_a)
    count = stop - start + 1
    if count < 4:
        return None
    return float(extrema[start]), float(extrema[stop]), count


def _candidate_extrema_count(
    problem: object,
    candidate: FitCandidate,
    q_min: float,
    q_max: float,
) -> int:
    fit_mask = problem.data.fit_mask
    extrema = fringe_extrema_qz(
        candidate.qz_a_inv[fit_mask],
        candidate.model_normalized[fit_mask],
        problem.data.r_floor,
    )
    return int(np.count_nonzero((extrema >= q_min) & (extrema <= q_max)))


def fringe_count_screen(
    problem: object,
    candidates: tuple[FitCandidate, ...],
) -> FringeScreenResult:
    """Keep candidates matching the reliable observed fringe count."""
    candidate_tuple = tuple(candidates)
    window = reliable_fringe_window(problem)
    if window is None:
        return FringeScreenResult(
            candidate_tuple,
            True,
            warnings=("fringe_count_screen_disabled",),
        )
    q_min, q_max, observed_count = window
    tolerance = max(1, int(np.ceil(0.10 * observed_count)))
    survivors = tuple(
        candidate
        for candidate in candidate_tuple
        if candidate.valid
        and abs(
            _candidate_extrema_count(problem, candidate, q_min, q_max)
            - observed_count
        )
        <= tolerance
    )
    return FringeScreenResult(
        survivors,
        False,
        stop_reason=(
            "stage_a_all_candidates_rejected"
            if candidate_tuple and not survivors
            else None
        ),
    )
