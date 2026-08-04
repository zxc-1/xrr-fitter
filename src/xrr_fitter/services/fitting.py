"""Fit, profile-recovery, analysis, and MCMC composition services.

Services own runtime resources while fit and analysis remain pure calculation
domains. One independent dataset normally receives the configured local thread
budget for its complete search and uncertainty lifetime. Batch orchestration may
pass a smaller positive share so several datasets can run concurrently without
multiplying the total physics worker count. That runtime share does not modify
the compiled problem, seed tree, checkpoint identity, or persisted project
configuration.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from xrr_fitter.analysis.automatic import assess_automatic_quality
from xrr_fitter.analysis.joint import analyze_joint_ensemble
from xrr_fitter.analysis.mcmc import run_problem_mcmc
from xrr_fitter.analysis.profiles import recover_profile_basin
from xrr_fitter.analysis.report import AnalysisRequest, run_analysis
from xrr_fitter.fit.automatic import (
    candidate_from_physical_values,
    refit_from_physical_values,
)
from xrr_fitter.fit.candidates import best_candidate_index, candidate_from_evaluation
from xrr_fitter.fit.initialization import structure_evidence
from xrr_fitter.fit.joint_pipeline import JointFitRequest, run_joint_fit
from xrr_fitter.fit.joint_problem import compile_joint_problem
from xrr_fitter.fit.joint_sharing import joint_candidate_vectors
from xrr_fitter.fit.objective import evaluate_vector
from xrr_fitter.fit.parameters import (
    apply_parameter_settings,
    default_parameter_definitions,
)
from xrr_fitter.fit.pipeline import (
    FitSearchRequest,
    continue_profile_basin,
    run_fit_search,
)
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.analysis import FitResult, McmcConfig, StructureEvidence
from xrr_fitter.model.automation import AutomaticRole, AutomaticStatus
from xrr_fitter.model.fitting import (
    FitCheckpoint,
    FitEvaluationContext,
    FitProgress,
    FitSearchResult,
    FitStageSummary,
    candidate_selection_objective,
)
from xrr_fitter.model.operations import FitReadiness, ProjectFitResult
from xrr_fitter.model.parameters import ParameterSetting
from xrr_fitter.model.project import DatasetProject, ScalePriorState, XrrProject
from xrr_fitter.model.provenance import fit_search_provenance_sha256
from xrr_fitter.services.datasets import (
    SERVICE_SEED_TREE_VERSION,  # noqa: F401 - preserved service compatibility export
    _prepared_current,
    mcmc_candidate_seed,
    service_seed_branches,
)
from xrr_fitter.services.parallel import OrderedTaskRunner
from xrr_fitter.services.projects import inspect_sources

ProgressCallback = Callable[[FitProgress], None]
CheckpointCallback = Callable[[XrrProject], None]
CancellationProbe = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class PreparedDatasetFit:
    """One source-checked, compiled dataset ready for service execution."""

    dataset_id: str
    dataset_index: int
    updated_dataset: DatasetProject
    problem: FitEvaluationContext


@dataclass(frozen=True, slots=True)
class AutomaticPreparedResult:
    """Automatic fit output plus the quality gate that owns publication."""

    prepared: PreparedDatasetFit
    fit_result: FitResult
    passed: bool
    reason: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.fit_result, FitResult):
            raise TypeError("fit_result must be FitResult")
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be bool")
        if self.passed and self.reason is not None:
            raise ValueError("passed result must not have a reason")
        if not self.passed and not self.reason:
            raise ValueError("failed quality decision requires a reason")


def _scale_prior(problem: FitEvaluationContext) -> ScalePriorState:
    return ScalePriorState(
        enabled=problem.scale_prior_center is not None,
        s_hat=problem.scale_prior_center,
        tau_s_decades=problem.scale_prior_tau_decades,
        reason=problem.scale_prior_reason,
    )


def _structure_evidence(problem: FitEvaluationContext) -> StructureEvidence:
    return structure_evidence_for(problem.data, problem.structure)


def structure_evidence_for(data, structure) -> StructureEvidence:
    """Translate fit-owned evidence into the public model value."""
    evidence = structure_evidence(data, structure)
    return StructureEvidence(
        evidence.m_data,
        evidence.m_model,
        evidence.warning,
        evidence.peak_positions_a,
    )


def parameter_definitions_for(data, structure, instrument, config):
    """Expose the canonical declarations without duplicating fit rules."""
    return default_parameter_definitions(data, structure, instrument, config)


def compiled_parameter_definitions(
    data,
    structure,
    instrument,
    config,
    settings,
):
    """Compile settings through the canonical fit problem boundary."""
    return compile_fit_problem(
        data,
        structure,
        instrument,
        config,
        tuple(settings),
    ).parameter_definitions


def validate_parameter_setting_declarations(definitions, settings) -> None:
    """Apply fit-owned setting validation without returning a fit value."""
    apply_parameter_settings(tuple(definitions), tuple(settings))


def _dataset_index(project: XrrProject, dataset_id: str) -> int:
    try:
        return next(
            index
            for index, dataset in enumerate(project.datasets)
            if dataset.dataset_id == dataset_id
        )
    except StopIteration as error:
        raise ValueError(f"unknown dataset_id: {dataset_id}") from error


def _compile_dataset(
    project: XrrProject,
    dataset_id: str,
    *,
    master_seed: int,
) -> PreparedDatasetFit:
    index = _dataset_index(project, dataset_id)
    dataset = project.datasets[index]
    if dataset.structure is None:
        raise ValueError(f"dataset {dataset_id} has no structure")
    data = _prepared_current(project, dataset)
    config = replace(project.fit_config, master_seed=master_seed)
    problem = compile_fit_problem(
        data,
        dataset.structure,
        dataset.instrument,
        config,
        dataset.parameter_settings,
    )
    updated = replace(
        dataset,
        structure_evidence=_structure_evidence(problem),
        scale_prior=_scale_prior(problem),
    )
    return PreparedDatasetFit(dataset_id, index, updated, problem)


def prepare_dataset_fit(
    project: XrrProject,
    dataset_id: str,
    seed: int,
) -> PreparedDatasetFit:
    """Parse and compile one dataset against its persisted source identity."""
    return _compile_dataset(project, dataset_id, master_seed=seed)


def _search_with_profile_recovery(
    prepared: PreparedDatasetFit,
    *,
    progress: ProgressCallback | None,
    cancelled: CancellationProbe | None,
    checkpoint: Callable[[FitCheckpoint], None] | None,
    task_runner: Callable,
):
    search = run_fit_search(
        FitSearchRequest(
            prepared.dataset_id,
            prepared.problem,
            prepared.updated_dataset.checkpoint,
        ),
        cancelled=cancelled,
        progress=progress,
        checkpoint=checkpoint,
        task_runner=task_runner,
    )
    candidate = search.best_candidate
    if candidate is None:
        return search
    objective = candidate_selection_objective(candidate)
    if progress is not None:
        progress(
            FitProgress(
                prepared.dataset_id,
                "basin-recovery",
                0,
                1,
                objective,
                "checking profile basins",
            )
        )
    decision = recover_profile_basin(
        prepared.problem,
        candidate,
        cancelled=cancelled,
    )
    if decision is None:
        if progress is not None:
            progress(
                FitProgress(
                    prepared.dataset_id,
                    "basin-recovery",
                    1,
                    1,
                    objective,
                    "basin recovery completed",
                )
            )
        return search
    continued = continue_profile_basin(
        prepared.problem,
        search,
        decision.unit_vector,
        parameter_name=decision.parameter_name,
        cancelled=cancelled,
        checkpoint=checkpoint,
        task_runner=task_runner,
    )
    if progress is not None:
        progress(
            FitProgress(
                prepared.dataset_id,
                "basin-recovery",
                1,
                1,
                objective,
                "basin recovery completed",
            )
        )
    return continued


def fit_prepared_dataset(
    prepared: PreparedDatasetFit,
    *,
    progress: ProgressCallback | None = None,
    cancelled: CancellationProbe | None = None,
    checkpoint: Callable[[FitCheckpoint], None] | None = None,
    local_workers: int | None = None,
) -> FitResult:
    """Run one independent search, optional recovery, and final analysis."""
    workers = prepared.problem.config.local_workers if local_workers is None else local_workers
    if local_workers is not None and local_workers > prepared.problem.config.local_workers:
        raise ValueError("local_workers must fit within the configured worker budget")
    with OrderedTaskRunner(workers) as runner:
        search = _search_with_profile_recovery(
            prepared,
            progress=progress,
            cancelled=cancelled,
            checkpoint=checkpoint,
            task_runner=runner.run,
        )
        return run_analysis(
            AnalysisRequest(prepared.dataset_id, prepared.problem, search),
            cancelled=cancelled,
            progress=progress,
            task_runner=runner.run,
        )


def _automatic_fast_analysis(
    prepared: PreparedDatasetFit,
    search: FitSearchResult,
    *,
    progress: ProgressCallback | None,
    cancelled: CancellationProbe | None,
    task_runner: Callable,
) -> FitResult:
    return run_analysis(
        AnalysisRequest(
            prepared.dataset_id,
            prepared.problem,
            search,
            profile_names=(),
            bootstrap_enabled=False,
        ),
        cancelled=cancelled,
        progress=progress,
        task_runner=task_runner,
    )


def _automatic_absorption_problem(
    problem: FitEvaluationContext,
    names: tuple[str, ...],
    values: dict[str, float],
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


def _updated_stage_e_summaries(
    search: FitSearchResult,
    candidates: tuple[object, ...],
) -> tuple[FitStageSummary, ...]:
    stage_index = next(
        (
            index
            for index in range(len(search.stage_summaries) - 1, -1, -1)
            if search.stage_summaries[index].stage in {"E", "stage-e"}
        ),
        None,
    )
    if stage_index is None:
        return search.stage_summaries
    original = search.stage_summaries[stage_index]
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    scoped = tuple(by_id[candidate_id] for candidate_id in original.candidate_ids)
    selectable = best_candidate_index(candidates, eligible_ids=original.candidate_ids)
    best_objective = (
        float("inf")
        if selectable is None
        else candidate_selection_objective(candidates[selectable])
    )
    replacement = FitStageSummary(
        original.stage,
        original.candidate_ids,
        best_objective,
        sum(candidate.nfev for candidate in scoped),
        tuple(candidate.stop_reason for candidate in scoped),
    )
    return tuple(
        replacement if index == stage_index else summary
        for index, summary in enumerate(search.stage_summaries)
    )


def _automatic_absorption_search(
    prepared: PreparedDatasetFit,
    search: FitSearchResult,
    names: tuple[str, ...],
    *,
    cancelled: CancellationProbe | None,
) -> tuple[PreparedDatasetFit, FitSearchResult]:
    problem = prepared.problem
    baseline = search.best_candidate
    if baseline is None or not names:
        return prepared, search
    baseline_values = {value.name: value.value for value in baseline.parameters}
    definitions = {definition.name: definition for definition in problem.parameter_definitions}
    active = tuple(name for name in names if name in definitions)
    if not active:
        return prepared, search
    trial_problem = _automatic_absorption_problem(problem, active, baseline_values)
    starts = tuple(
        {name: baseline_values[name]}
        for name in active
    ) + tuple(
        {name: min(2e-6, definitions[name].upper)}
        for name in active
    )
    trial = refit_from_physical_values(
        trial_problem,
        starts,
        max_nfev=problem.config.budget.local_min_nfev,
        cancelled=cancelled,
    )
    winner = trial.best_candidate
    if winner is None or not winner.valid:
        return prepared, search
    gain = baseline.objective - winner.objective
    thresholds = problem.config.confidence
    required = max(
        abs(baseline.objective) * thresholds.equivalent_cost_fraction,
        thresholds.equivalent_cost_floor,
    )
    if not (winner.valid and gain > required):
        return prepared, search
    winner_values = {value.name: value.value for value in winner.parameters}
    accepted_problem = _fixed_absorption_problem(problem, active, winner_values)
    replacement = candidate_from_physical_values(
        accepted_problem,
        winner_values,
        baseline,
        stop_reason=winner.stop_reason,
        nfev=baseline.nfev + winner.nfev,
    )
    if not replacement.valid:
        return prepared, search
    candidates = tuple(
        replacement
        if candidate.candidate_id == baseline.candidate_id
        else _candidate_for_problem(
            accepted_problem,
            candidate,
            candidate.unit_vector,
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
        stage_summaries=_updated_stage_e_summaries(search, candidates),
        region_labels=search.region_labels,
        region_weights=search.region_weights,
    )
    result = replace(
        result,
        provenance_sha256=fit_search_provenance_sha256(accepted_problem, result),
    )
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


def _automatic_profile_recovery(
    prepared: PreparedDatasetFit,
    search: FitSearchResult,
    *,
    progress: ProgressCallback | None,
    cancelled: CancellationProbe | None,
    checkpoint: Callable[[FitCheckpoint], None] | None,
    task_runner: Callable,
) -> FitSearchResult:
    candidate = search.best_candidate
    if candidate is None:
        return search
    decision = recover_profile_basin(
        prepared.problem,
        candidate,
        cancelled=cancelled,
    )
    if decision is None:
        return search
    return continue_profile_basin(
        prepared.problem,
        search,
        decision.unit_vector,
        parameter_name=decision.parameter_name,
        cancelled=cancelled,
        checkpoint=checkpoint,
        task_runner=task_runner,
    )


def fit_automatic_prepared_dataset(
    prepared: PreparedDatasetFit,
    *,
    progress: ProgressCallback | None = None,
    cancelled: CancellationProbe | None = None,
    checkpoint: Callable[[FitCheckpoint], None] | None = None,
    local_workers: int | None = None,
) -> AutomaticPreparedResult:
    """Run the bounded automatic search, quality gates, and final report."""
    workers = prepared.problem.config.local_workers if local_workers is None else local_workers
    if local_workers is not None and local_workers > prepared.problem.config.local_workers:
        raise ValueError("local_workers must fit within the configured worker budget")
    with OrderedTaskRunner(workers) as runner:
        search = run_fit_search(
            FitSearchRequest(
                prepared.dataset_id,
                prepared.problem,
                prepared.updated_dataset.checkpoint,
            ),
            cancelled=cancelled,
            progress=progress,
            checkpoint=checkpoint,
            task_runner=runner.run,
        )
        fast_result = _automatic_fast_analysis(
            prepared,
            search,
            progress=progress,
            cancelled=cancelled,
            task_runner=runner.run,
        )
        decision = assess_automatic_quality(prepared.problem, fast_result)
        if decision.search_upgrade:
            search = _automatic_profile_recovery(
                prepared,
                search,
                progress=progress,
                cancelled=cancelled,
                checkpoint=checkpoint,
                task_runner=runner.run,
            )
            fast_result = _automatic_fast_analysis(
                prepared,
                search,
                progress=progress,
                cancelled=cancelled,
                task_runner=runner.run,
            )
            decision = assess_automatic_quality(prepared.problem, fast_result)
        if decision.absorption_names:
            updated_prepared, updated = _automatic_absorption_search(
                prepared,
                search,
                decision.absorption_names,
                cancelled=cancelled,
            )
            if updated is not search:
                prepared = updated_prepared
                search = updated
                fast_result = _automatic_fast_analysis(
                    prepared,
                    search,
                    progress=progress,
                    cancelled=cancelled,
                    task_runner=runner.run,
                )
                decision = assess_automatic_quality(prepared.problem, fast_result)
        final_result = run_analysis(
            AnalysisRequest(
                prepared.dataset_id,
                prepared.problem,
                search,
                profile_names=decision.profile_names,
                bootstrap_enabled=False,
            ),
            cancelled=cancelled,
            progress=progress,
            task_runner=runner.run,
        )
        final_decision = assess_automatic_quality(prepared.problem, final_result)
        reason = None if final_decision.passed else "; ".join(final_decision.reasons)
        if not final_decision.passed and not reason:
            reason = "automatic quality review required"
        return AutomaticPreparedResult(prepared, final_result, final_decision.passed, reason)


def _joint_checkpoints(
    prepared: tuple[PreparedDatasetFit, ...],
) -> tuple[FitCheckpoint, ...] | None:
    values = tuple(item.updated_dataset.checkpoint for item in prepared)
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("joint resume requires checkpoints for all datasets")
    return tuple(value for value in values if value is not None)


def _joint_final_ids(searches: tuple[object, ...]) -> tuple[str, ...]:
    summaries = tuple(
        next(summary for summary in reversed(search.stage_summaries) if summary.stage == "E")
        for search in searches
    )
    if any(summary != summaries[0] for summary in summaries[1:]):
        raise ValueError("joint Stage-E history is not aligned")
    return summaries[0].candidate_ids


def _joint_candidate_maps(searches: tuple[object, ...]) -> tuple[dict[str, object], ...]:
    return tuple(
        {candidate.candidate_id: candidate for candidate in search.candidates}
        for search in searches
    )


def _joint_candidate_rows(
    candidate_maps: tuple[dict[str, object], ...],
    candidate_ids: tuple[str, ...],
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        tuple(candidate_map[candidate_id] for candidate_map in candidate_maps)
        for candidate_id in candidate_ids
    )


def _joint_objectives(rows: tuple[tuple[object, ...], ...]) -> tuple[float, ...]:
    return tuple(float(candidates[0].ranking_objective) for candidates in rows)


def _joint_validity(rows: tuple[tuple[object, ...], ...]) -> tuple[bool, ...]:
    return tuple(all(candidate.valid for candidate in candidates) for candidates in rows)


def _joint_diagnostics(
    rows: tuple[tuple[object, ...], ...],
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        tuple(
            diagnostic
            for candidate in candidates
            for diagnostic in candidate.diagnostics
        )
        for candidates in rows
    )


def _joint_physical_values(
    problem: object,
    candidate_maps: tuple[dict[str, object], ...],
    candidate_ids: tuple[str, ...],
) -> tuple[tuple[float, ...], ...]:
    dataset_indices = {
        dataset_id: index for index, dataset_id in enumerate(problem.dataset_ids)
    }
    rows = []
    for candidate_id in candidate_ids:
        values = []
        for variable in problem.global_variables:
            member = variable.members[0]
            candidate = candidate_maps[dataset_indices[member.dataset_id]][candidate_id]
            parameter = next(
                value for value in candidate.parameters if value.name == member.parameter_name
            )
            values.append(parameter.value)
        rows.append(tuple(values))
    return tuple(rows)


def _analyze_joint_searches(
    problem: object,
    searches: tuple[object, ...],
) -> tuple[FitResult, ...]:
    candidate_ids = _joint_final_ids(searches)
    candidate_maps = _joint_candidate_maps(searches)
    vectors = joint_candidate_vectors(
        problem,
        tuple(search.candidates for search in searches),
        candidate_ids,
    )
    aligned = _joint_candidate_rows(candidate_maps, candidate_ids)
    report, confidence, evidence = analyze_joint_ensemble(
        variable_names=tuple(variable.name for variable in problem.global_variables),
        candidate_ids=candidate_ids,
        unit_vectors=vectors,
        physical_values=_joint_physical_values(problem, candidate_maps, candidate_ids),
        objectives=_joint_objectives(aligned),
        valid=_joint_validity(aligned),
        diagnostics=_joint_diagnostics(aligned),
        thresholds=problem.problems[0].config.confidence,
    )
    return tuple(
        FitResult.from_search(
            search,
            confidence=confidence,
            uncertainty=report,
            classification_evidence=evidence,
        )
        for search in searches
    )


def fit_joint_datasets(
    prepared: tuple[PreparedDatasetFit, ...],
    sharing_rules: tuple,
    *,
    progress: ProgressCallback | None = None,
    cancelled: CancellationProbe | None = None,
    checkpoint: Callable[[tuple[FitCheckpoint, ...]], None] | None = None,
) -> tuple[FitResult, ...]:
    """Run and analyze one joint graph without independent fallback."""
    values = tuple(prepared)
    problem = compile_joint_problem(
        tuple(item.dataset_id for item in values),
        tuple(item.problem for item in values),
        tuple(sharing_rules),
    )
    searches = run_joint_fit(
        JointFitRequest(problem, _joint_checkpoints(values)),
        cancelled=cancelled,
        progress=progress,
        checkpoint=checkpoint,
    )
    best = searches[0].best_candidate if searches else None
    objective = float("inf") if best is None else candidate_selection_objective(best)
    if progress is not None:
        progress(
            FitProgress(
                None,
                "finalizing",
                0,
                1,
                objective,
                "finalizing joint fit",
            )
        )
    results = _analyze_joint_searches(problem, searches)
    if progress is not None:
        progress(FitProgress(None, "finalizing", 1, 1, objective, "completed"))
    return results


def _source_failure(validation) -> str | None:
    if validation.valid:
        return None
    if validation.issues:
        return validation.issues[0].message
    record = next((item for item in validation.datasets if item.status.value != "ok"), None)
    return "source validation failed" if record is None else record.message


def _preflight_seeds(project: XrrProject) -> dict[str, int]:
    independent, joint, _mcmc = service_seed_branches(project)
    if project.batch_mode == "independent":
        return independent
    return {dataset.dataset_id: joint for dataset in project.datasets}


def _compile_preflight_fit(project: XrrProject) -> None:
    seeds = _preflight_seeds(project)
    prepared = tuple(
        prepare_dataset_fit(project, dataset.dataset_id, seeds[dataset.dataset_id])
        for dataset in project.datasets
    )
    if project.batch_mode == "joint":
        compile_joint_problem(
            tuple(item.dataset_id for item in prepared),
            tuple(item.problem for item in prepared),
            project.sharing_rules,
        )


def preflight_fit(project: XrrProject) -> FitReadiness:
    """Load and compile the complete declared fit without mutating the project."""
    if not project.datasets:
        return FitReadiness(False, "project has no datasets")
    try:
        validation = inspect_sources(project)
        failure = _source_failure(validation)
        if failure is not None:
            return FitReadiness(False, failure)
        _compile_preflight_fit(project)
    except Exception as error:
        return FitReadiness(False, str(error) or type(error).__name__)
    return FitReadiness(True, "ready")


_AUTOMATIC_RUNNABLE = frozenset(
    {AutomaticStatus.PENDING, AutomaticStatus.REFINING, AutomaticStatus.REVIEW}
)


def _automatic_dataset_ids(
    project: XrrProject,
    import_batch_id: str | None,
) -> tuple[str, ...]:
    return tuple(
        dataset.dataset_id
        for dataset in project.datasets
        if dataset.automation.role is not AutomaticRole.MANUAL
        and dataset.automation.status in _AUTOMATIC_RUNNABLE
        and (
            import_batch_id is None
            or dataset.automation.import_batch_id == import_batch_id
        )
    )


def preflight_automatic_fit(
    project: XrrProject,
    import_batch_id: str | None = None,
) -> FitReadiness:
    """Validate only runnable automatic datasets without mutating state."""
    if project.measurement_preset is None:
        return FitReadiness(False, "automatic fit requires a measurement preset")
    dataset_ids = _automatic_dataset_ids(project, import_batch_id)
    if not dataset_ids:
        return FitReadiness(False, "no runnable automatic datasets")
    try:
        records = {
            record.dataset_id: record
            for record in inspect_sources(project).datasets
        }
        seeds, _joint_seed, _mcmc_seed = service_seed_branches(project)
        for dataset_id in dataset_ids:
            record = records[dataset_id]
            if record.status.value != "ok":
                return FitReadiness(False, record.message)
            prepare_dataset_fit(project, dataset_id, seeds[dataset_id])
    except Exception as error:
        return FitReadiness(False, str(error) or type(error).__name__)
    return FitReadiness(True, "ready")


def fit_automatically(
    project: XrrProject,
    import_batch_id: str | None = None,
    progress_callback: ProgressCallback | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
) -> ProjectFitResult:
    """Run the persisted automatic route through the batch transaction."""
    readiness = preflight_automatic_fit(project, import_batch_id)
    if not readiness.ready:
        raise ValueError(readiness.message)
    from xrr_fitter.services.batch import fit_automatic_transaction

    return fit_automatic_transaction(
        project,
        import_batch_id,
        progress_callback,
        checkpoint_callback,
        None,
        seed_branches=service_seed_branches,
        prepare_dataset=prepare_dataset_fit,
        fit_dataset=fit_automatic_prepared_dataset,
    )


def fit_project(
    project: XrrProject,
    progress_callback: ProgressCallback | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
) -> ProjectFitResult:
    """Dispatch a synchronous fit through the batch transaction owner."""
    return _dispatch_project(project, progress_callback, checkpoint_callback, None)


def _dispatch_project(
    project: XrrProject,
    progress_callback: ProgressCallback | None,
    checkpoint_callback: CheckpointCallback | None,
    cancelled: CancellationProbe | None,
) -> ProjectFitResult:
    from xrr_fitter.services.batch import fit_project_transaction

    return fit_project_transaction(
        project,
        progress_callback,
        checkpoint_callback,
        cancelled,
        seed_branches=service_seed_branches,
        prepare_dataset=prepare_dataset_fit,
        fit_dataset=fit_prepared_dataset,
        fit_joint=fit_joint_datasets,
    )


def _mcmc_problem(project: XrrProject, dataset_id: str):
    prepared = _compile_dataset(project, dataset_id, master_seed=project.master_seed)
    result = prepared.updated_dataset.last_valid_result
    if result is None or result.uncertainty is None:
        raise ValueError(f"dataset has no valid uncertainty result: {dataset_id}")
    if prepared.problem.parameter_definitions != result.parameter_definitions:
        raise ValueError(f"parameter definitions changed: {dataset_id}")
    return prepared, result


def _run_mcmc(
    project: XrrProject,
    dataset_id: str,
    candidate_id: str,
    config: McmcConfig,
    progress_callback: ProgressCallback | None,
    cancelled: CancellationProbe | None,
) -> XrrProject:
    validation = inspect_sources(project)
    failure = _source_failure(validation)
    if failure is not None:
        raise ValueError(failure)
    prepared, result = _mcmc_problem(project, dataset_id)
    candidate = next(
        (item for item in result.candidates if item.candidate_id == candidate_id),
        None,
    )
    if candidate is None or not candidate.valid:
        raise ValueError(f"invalid MCMC candidate: {dataset_id}/{candidate_id}")
    seed = mcmc_candidate_seed(
        project,
        dataset_id,
        tuple(item.candidate_id for item in result.candidates),
        candidate_id,
    )

    def progress(completed: int, total: int) -> None:
        if progress_callback is not None:
            progress_callback(
                FitProgress(
                    dataset_id,
                    "MCMC",
                    completed,
                    total,
                    candidate.objective,
                    "MCMC sampling",
                )
            )

    report = run_problem_mcmc(
        prepared.problem,
        candidate,
        config,
        child_seed=seed,
        progress=progress,
        cancelled=cancelled,
    )
    updated_result = replace(
        result,
        uncertainty=replace(result.uncertainty, mcmc=report),
    )
    datasets = tuple(
        replace(dataset, last_valid_result=updated_result)
        if dataset.dataset_id == dataset_id
        else dataset
        for dataset in project.datasets
    )
    return replace(project, datasets=datasets)


def run_mcmc(
    project: XrrProject,
    dataset_id: str,
    candidate_id: str,
    config: McmcConfig,
    progress_callback: ProgressCallback | None = None,
) -> XrrProject:
    return _run_mcmc(
        project,
        dataset_id,
        candidate_id,
        config,
        progress_callback,
        None,
    )


def fit_worker_handler(
    project: XrrProject,
    progress_callback: ProgressCallback | None,
    checkpoint_callback: CheckpointCallback | None,
    cancelled: CancellationProbe | None,
) -> ProjectFitResult:
    return _dispatch_project(
        project,
        progress_callback,
        checkpoint_callback,
        cancelled,
    )


def mcmc_worker_handler(
    project: XrrProject,
    dataset_id: str,
    candidate_id: str,
    config: McmcConfig,
    progress_callback: ProgressCallback | None,
    cancelled: CancellationProbe | None,
) -> XrrProject:
    return _run_mcmc(
        project,
        dataset_id,
        candidate_id,
        config,
        progress_callback,
        cancelled,
    )
