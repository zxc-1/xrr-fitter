"""Bounded local refits used by the automatic fitting policy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from math import isfinite

import xrr_fitter.evaluation as evaluation
from xrr_fitter.fit.candidates import (
    best_candidate_index,
    candidate_from_evaluation,
)
from xrr_fitter.fit.local_search import solve_local
from xrr_fitter.fit.objective import evaluate_vector
from xrr_fitter.model.fitting import FitSearchResult, FitStageSummary
from xrr_fitter.model.provenance import fit_search_provenance_sha256


def _bounded_nfev(problem: object, max_nfev: int) -> int:
    if isinstance(max_nfev, bool) or not isinstance(max_nfev, int) or max_nfev <= 0:
        raise ValueError("max_nfev must be a positive integer")
    configured = int(problem.config.budget.local_min_nfev)
    return max(1, min(max_nfev, configured))


def _start_mapping(start: object) -> dict[object, object]:
    """Copy a mapping-like start while preserving the public error contract."""
    if isinstance(start, Mapping):
        return dict(start)
    try:
        return dict(start)
    except (TypeError, ValueError) as error:
        raise TypeError("automatic starts must be physical mappings") from error


def _validate_start_mapping(problem: object, values: dict[object, object]) -> None:
    """Reject unknown, non-string, or nonfinite physical start entries."""
    names = {definition.name for definition in problem.parameter_definitions}
    unknown = sorted(set(values) - names)
    if unknown:
        raise ValueError(f"automatic start contains unknown parameters: {unknown[0]}")
    if any(not isinstance(name, str) for name in values):
        raise TypeError("automatic start parameter names must be strings")
    if any(not isfinite(float(value)) for value in values.values()):
        raise ValueError("automatic start values must be finite")


def _physical_start(problem: object, start: object) -> dict[str, float]:
    values = _start_mapping(start)
    _validate_start_mapping(problem, values)
    return {name: float(value) for name, value in values.items()}


def candidate_from_physical_values(
    problem: object,
    physical_values: object,
    template: object,
    *,
    stop_reason: str | None = None,
    nfev: int | None = None,
):
    """Re-evaluate one candidate identity after a physical-value handoff."""
    physical = _physical_start(problem, physical_values)
    unit = evaluation.encode_physical_vector(problem, physical)
    evaluated = evaluate_vector(problem, unit)
    candidate = candidate_from_evaluation(
        problem,
        unit,
        evaluated,
        candidate_id=template.candidate_id,
        seed_index=template.seed_index,
        stop_reason=template.stop_reason if stop_reason is None else stop_reason,
        nfev=template.nfev if nfev is None else nfev,
    )
    if template.ranking_objective is None:
        return candidate
    prior_cost = template.ranking_objective - template.objective
    return replace(
        candidate,
        ranking_objective=candidate.objective + prior_cost,
    )


def refit_from_physical_values(
    problem: object,
    starts: object,
    max_nfev: int,
    cancelled=None,
) -> FitSearchResult:
    """Run a deterministic, bounded local refit from physical-value starts."""
    starts = tuple(starts)
    if not starts:
        raise ValueError("automatic refit requires at least one start")
    maximum = _bounded_nfev(problem, max_nfev)
    candidates = []
    for index, start in enumerate(starts):
        physical = _physical_start(problem, start)
        unit = evaluation.encode_physical_vector(problem, physical)
        solved = solve_local(
            problem,
            unit,
            max_nfev=maximum,
            cancelled=cancelled,
        )
        candidates.append(
            candidate_from_evaluation(
                problem,
                solved.unit_vector,
                solved.evaluation,
                candidate_id=f"automatic-refit-{index}",
                seed_index=index,
                stop_reason=solved.stop_reason,
                nfev=solved.nfev + 1,
            )
        )
    values = tuple(candidates)
    candidate_ids = tuple(candidate.candidate_id for candidate in values)
    selectable = best_candidate_index(values)
    best_objective = (
        float("inf") if selectable is None else values[selectable].objective
    )
    summary = FitStageSummary(
        "automatic-refit",
        candidate_ids,
        best_objective,
        sum(candidate.nfev for candidate in values),
        tuple(candidate.stop_reason for candidate in values),
    )
    result = FitSearchResult(
        parameter_definitions=problem.parameter_definitions,
        candidates=values,
        best_index=selectable,
        warnings=tuple(problem.warnings),
        child_seeds=tuple(range(len(values))),
        stage_summaries=(summary,),
        region_labels=problem.region_labels,
        region_weights=problem.weights,
    )
    return replace(
        result,
        provenance_sha256=fit_search_provenance_sha256(problem, result),
    )
