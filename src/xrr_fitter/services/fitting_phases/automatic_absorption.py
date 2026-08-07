"""Automatic fast analysis and absorption recovery."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from xrr_fitter.model.fitting import (
    FitEvaluationContext,
    FitSearchResult,
    FitStageSummary,
    candidate_selection_objective,
)
from xrr_fitter.model.parameters import ParameterSetting
from xrr_fitter.model.provenance import fit_search_provenance_sha256

from .base import _scale_prior
from .common import CancellationProbe, PreparedDatasetFit

def _automatic_absorption_problem(
    problem: FitEvaluationContext,
    names: tuple[str, ...],
    values: dict[str, float],
    *,
    compile_fit_problem: Callable,
) -> FitEvaluationContext:
    released = frozenset(names)
    settings = tuple(
        (
            ParameterSetting(
                definition.name,
                values.get(definition.name, definition.initial),
                definition.lower,
                definition.upper,
                locked=False,
            )
            if definition.name in released
            else ParameterSetting(
                definition.name,
                values.get(definition.name, definition.initial),
                values.get(definition.name, definition.initial),
                values.get(definition.name, definition.initial),
                locked=True,
            )
        )
        for definition in problem.parameter_definitions
    )
    return compile_fit_problem(
        problem.data,
        problem.structure,
        problem.instrument,
        problem.config,
        settings,
    )


def _fixed_absorption_problem(
    problem: FitEvaluationContext,
    names: tuple[str, ...],
    values: dict[str, float],
    *,
    compile_fit_problem: Callable,
) -> FitEvaluationContext:
    fixed = frozenset(names)
    settings = tuple(
        ParameterSetting(
            definition.name,
            values[definition.name]
            if definition.name in fixed
            else definition.initial,
            definition.lower,
            definition.upper,
            locked=True if definition.name in fixed else definition.locked,
        )
        for definition in problem.parameter_definitions
    )
    return compile_fit_problem(
        problem.data,
        problem.structure,
        problem.instrument,
        problem.config,
        settings,
    )


def _fixed_absorption_settings(
    prepared: PreparedDatasetFit,
    problem: FitEvaluationContext,
    names: tuple[str, ...],
    values: dict[str, float],
) -> tuple[ParameterSetting, ...]:
    fixed = frozenset(names)
    definitions = {
        definition.name: definition for definition in problem.parameter_definitions
    }
    existing = {setting.name for setting in prepared.updated_dataset.parameter_settings}

    def updated(setting: ParameterSetting) -> ParameterSetting:
        if setting.name not in fixed:
            return setting
        definition = definitions[setting.name]
        return ParameterSetting(
            setting.name,
            values[setting.name],
            definition.lower,
            definition.upper,
            locked=True,
        )

    retained = tuple(updated(setting) for setting in prepared.updated_dataset.parameter_settings)
    appended = tuple(
        ParameterSetting(
            definition.name,
            values[definition.name],
            definition.lower,
            definition.upper,
            locked=True,
        )
        for definition in problem.parameter_definitions
        if definition.name in fixed and definition.name not in existing
    )
    return (*retained, *appended)


def _candidate_for_problem(
    problem: FitEvaluationContext,
    candidate,
    unit,
    *,
    evaluate_vector: Callable,
    candidate_from_evaluation: Callable,
    stop_reason: str | None = None,
    nfev: int | None = None,
):
    evaluation = evaluate_vector(problem, unit)
    replacement = candidate_from_evaluation(
        problem,
        unit,
        evaluation,
        candidate_id=candidate.candidate_id,
        seed_index=candidate.seed_index,
        stop_reason=candidate.stop_reason if stop_reason is None else stop_reason,
        nfev=candidate.nfev if nfev is None else nfev,
    )
    if candidate.ranking_objective is None:
        return replacement
    prior_cost = candidate.ranking_objective - candidate.objective
    return replace(
        replacement,
        ranking_objective=replacement.objective + prior_cost,
    )


def _last_stage_e_index(summaries: tuple[FitStageSummary, ...]) -> int | None:
    return next(
        (
            index
            for index in range(len(summaries) - 1, -1, -1)
            if summaries[index].stage in {"E", "stage-e"}
        ),
        None,
    )


def _replacement_stage_e_summary(
    summary: FitStageSummary,
    candidates: tuple[object, ...],
    *,
    best_candidate_index: Callable,
) -> FitStageSummary:
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    scoped = tuple(by_id[candidate_id] for candidate_id in summary.candidate_ids)
    selectable = best_candidate_index(candidates, eligible_ids=summary.candidate_ids)
    best_objective = (
        float("inf")
        if selectable is None
        else candidate_selection_objective(candidates[selectable])
    )
    return FitStageSummary(
        summary.stage,
        summary.candidate_ids,
        best_objective,
        sum(candidate.nfev for candidate in scoped),
        tuple(candidate.stop_reason for candidate in scoped),
    )


def _updated_stage_e_summaries(
    search: FitSearchResult,
    candidates: tuple[object, ...],
    *,
    best_candidate_index: Callable,
) -> tuple[FitStageSummary, ...]:
    stage_index = _last_stage_e_index(search.stage_summaries)
    if stage_index is None:
        return search.stage_summaries
    original = search.stage_summaries[stage_index]
    replacement = _replacement_stage_e_summary(
        original,
        candidates,
        best_candidate_index=best_candidate_index,
    )
    return tuple(
        replacement if index == stage_index else summary
        for index, summary in enumerate(search.stage_summaries)
    )


def _absorption_trial_starts(
    active: tuple[str, ...],
    baseline_values: dict[str, float],
    definitions: dict[str, object],
) -> tuple[dict[str, float], ...]:
    return tuple({name: baseline_values[name]} for name in active) + tuple(
        {name: min(2e-6, definitions[name].upper)} for name in active
    )


def _absorption_inputs(
    problem: FitEvaluationContext,
    baseline,
    names: tuple[str, ...],
) -> tuple[dict[str, float], dict[str, object], tuple[str, ...]] | None:
    if baseline is None or not names:
        return None
    baseline_values = {value.name: value.value for value in baseline.parameters}
    definitions = {
        definition.name: definition for definition in problem.parameter_definitions
    }
    active = tuple(name for name in names if name in definitions)
    if not active:
        return None
    return baseline_values, definitions, active


def _absorption_gain_is_meaningful(problem, baseline, winner) -> bool:
    thresholds = problem.config.confidence
    required = max(
        abs(baseline.objective) * thresholds.equivalent_cost_fraction,
        thresholds.equivalent_cost_floor,
    )
    return baseline.objective - winner.objective > required


def _accepted_absorption_result(
    problem: FitEvaluationContext,
    search: FitSearchResult,
    baseline,
    active: tuple[str, ...],
    winner,
    *,
    compile_fit_problem: Callable,
    candidate_from_physical_values: Callable,
    evaluate_vector: Callable,
    candidate_from_evaluation: Callable,
    best_candidate_index: Callable,
    fit_search_provenance_sha256: Callable,
) -> tuple[FitEvaluationContext, FitSearchResult, dict[str, float]] | None:
    winner_values = {value.name: value.value for value in winner.parameters}
    accepted_problem = _fixed_absorption_problem(
        problem,
        active,
        winner_values,
        compile_fit_problem=compile_fit_problem,
    )
    replacement = candidate_from_physical_values(
        accepted_problem,
        winner_values,
        baseline,
        stop_reason=winner.stop_reason,
        nfev=baseline.nfev + winner.nfev,
    )
    if not replacement.valid:
        return None
    candidates = tuple(
        replacement
        if candidate.candidate_id == baseline.candidate_id
        else _candidate_for_problem(
            accepted_problem,
            candidate,
            candidate.unit_vector,
            evaluate_vector=evaluate_vector,
            candidate_from_evaluation=candidate_from_evaluation,
        )
        for candidate in search.candidates
    )
    eligible_ids = next(
        (
            summary.candidate_ids
            for summary in reversed(search.stage_summaries)
            if summary.stage in {"E", "stage-e"}
        ),
        None,
    )
    best_index = best_candidate_index(candidates, eligible_ids=eligible_ids)
    result = FitSearchResult(
        parameter_definitions=accepted_problem.parameter_definitions,
        candidates=candidates,
        best_index=best_index,
        warnings=search.warnings,
        child_seeds=search.child_seeds,
        stage_summaries=_updated_stage_e_summaries(
            search,
            candidates,
            best_candidate_index=best_candidate_index,
        ),
        region_labels=search.region_labels,
        region_weights=search.region_weights,
    )
    result = replace(
        result,
        provenance_sha256=fit_search_provenance_sha256(accepted_problem, result),
    )
    return accepted_problem, result, winner_values


def _automatic_absorption_search(
    prepared: PreparedDatasetFit,
    search: FitSearchResult,
    names: tuple[str, ...],
    *,
    cancelled: CancellationProbe | None,
    compile_fit_problem: Callable,
    refit_from_physical_values: Callable,
    candidate_from_physical_values: Callable,
    evaluate_vector: Callable,
    candidate_from_evaluation: Callable,
    best_candidate_index: Callable,
    fit_search_provenance_sha256: Callable,
) -> tuple[PreparedDatasetFit, FitSearchResult]:
    problem = prepared.problem
    baseline = search.best_candidate
    inputs = _absorption_inputs(problem, baseline, names)
    if inputs is None:
        return prepared, search
    baseline_values, definitions, active = inputs
    trial_problem = _automatic_absorption_problem(
        problem,
        active,
        baseline_values,
        compile_fit_problem=compile_fit_problem,
    )
    trial = refit_from_physical_values(
        trial_problem,
        _absorption_trial_starts(active, baseline_values, definitions),
        max_nfev=problem.config.budget.local_min_nfev,
        cancelled=cancelled,
    )
    winner = trial.best_candidate
    if winner is None or not winner.valid:
        return prepared, search
    if not _absorption_gain_is_meaningful(problem, baseline, winner):
        return prepared, search
    accepted = _accepted_absorption_result(
        problem,
        search,
        baseline,
        active,
        winner,
        compile_fit_problem=compile_fit_problem,
        candidate_from_physical_values=candidate_from_physical_values,
        evaluate_vector=evaluate_vector,
        candidate_from_evaluation=candidate_from_evaluation,
        best_candidate_index=best_candidate_index,
        fit_search_provenance_sha256=fit_search_provenance_sha256,
    )
    if accepted is None:
        return prepared, search
    accepted_problem, result, winner_values = accepted
    updated_dataset = replace(
        prepared.updated_dataset,
        parameter_settings=_fixed_absorption_settings(
            prepared,
            accepted_problem,
            active,
            winner_values,
        ),
        scale_prior=_scale_prior(accepted_problem),
        last_valid_result=None,
        checkpoint=None,
    )
    return replace(
        prepared,
        updated_dataset=updated_dataset,
        problem=accepted_problem,
    ), result
