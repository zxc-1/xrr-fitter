"""Prepared dataset compilation and ordinary fitting services."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from xrr_fitter.model.analysis import FitResult, StructureEvidence
from xrr_fitter.model.fitting import (
    FitCheckpoint,
    FitEvaluationContext,
    FitProgress,
    candidate_selection_objective,
)
from xrr_fitter.model.project import ScalePriorState, XrrProject
from xrr_fitter.services.datasets import _prepared_current
from xrr_fitter.services.parallel import OrderedTaskRunner

from .common import CancellationProbe, PreparedDatasetFit, ProgressCallback


def _scale_prior(problem: FitEvaluationContext) -> ScalePriorState:
    return ScalePriorState(
        enabled=problem.scale_prior_center is not None,
        s_hat=problem.scale_prior_center,
        tau_s_decades=problem.scale_prior_tau_decades,
        reason=problem.scale_prior_reason,
    )


def _structure_evidence(
    problem: FitEvaluationContext,
    *,
    structure_evidence: Callable,
) -> StructureEvidence:
    return structure_evidence_for(
        problem.data,
        problem.structure,
        structure_evidence=structure_evidence,
    )


def structure_evidence_for(
    data,
    structure,
    *,
    structure_evidence: Callable,
) -> StructureEvidence:
    """Translate fit-owned evidence into the public model value."""
    evidence = structure_evidence(data, structure)
    return StructureEvidence(
        evidence.m_data,
        evidence.m_model,
        evidence.warning,
        evidence.peak_positions_a,
    )


def parameter_definitions_for(
    data,
    structure,
    instrument,
    config,
    *,
    default_parameter_definitions: Callable,
):
    """Expose the canonical declarations without duplicating fit rules."""
    return default_parameter_definitions(data, structure, instrument, config)


def compiled_parameter_definitions(
    data,
    structure,
    instrument,
    config,
    settings,
    *,
    compile_fit_problem: Callable,
):
    """Compile settings through the canonical fit problem boundary."""
    return compile_fit_problem(
        data,
        structure,
        instrument,
        config,
        tuple(settings),
    ).parameter_definitions


def validate_parameter_setting_declarations(
    definitions,
    settings,
    *,
    apply_parameter_settings: Callable,
) -> None:
    """Apply fit-owned setting validation without returning a fit value."""
    apply_parameter_settings(tuple(definitions), tuple(settings))


def _dataset_index(project: XrrProject, dataset_id: str) -> int:
    try:
        return next(index for index, dataset in enumerate(project.datasets) if dataset.dataset_id == dataset_id)
    except StopIteration as error:
        raise ValueError(f"unknown dataset_id: {dataset_id}") from error


def _compile_dataset(
    project: XrrProject,
    dataset_id: str,
    *,
    master_seed: int,
    compile_fit_problem: Callable,
    structure_evidence: Callable,
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
        structure_evidence=_structure_evidence(
            problem,
            structure_evidence=structure_evidence,
        ),
        scale_prior=_scale_prior(problem),
    )
    return PreparedDatasetFit(dataset_id, index, updated, problem)


def prepare_dataset_fit(
    project: XrrProject,
    dataset_id: str,
    seed: int,
    *,
    compile_fit_problem: Callable,
    structure_evidence: Callable,
) -> PreparedDatasetFit:
    """Parse and compile one dataset against its persisted source identity."""
    return _compile_dataset(
        project,
        dataset_id,
        master_seed=seed,
        compile_fit_problem=compile_fit_problem,
        structure_evidence=structure_evidence,
    )


def _search_with_profile_recovery(
    prepared: PreparedDatasetFit,
    *,
    progress: ProgressCallback | None,
    cancelled: CancellationProbe | None,
    checkpoint: Callable[[FitCheckpoint], None] | None,
    task_runner: Callable,
    fit_search_request: Callable,
    run_fit_search: Callable,
    recover_profile_basin: Callable,
    continue_profile_basin: Callable,
):
    search = run_fit_search(
        fit_search_request(
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
    profile_names: tuple[str, ...] | None = None,
    fit_search_request: Callable,
    run_fit_search: Callable,
    recover_profile_basin: Callable,
    continue_profile_basin: Callable,
    analysis_request: Callable,
    run_analysis: Callable,
    task_runner_factory: Callable = OrderedTaskRunner,
) -> FitResult:
    """Run one independent search, optional recovery, and final analysis."""
    workers = prepared.problem.config.local_workers if local_workers is None else local_workers
    if local_workers is not None and local_workers > prepared.problem.config.local_workers:
        raise ValueError("local_workers must fit within the configured worker budget")
    with task_runner_factory(workers) as runner:
        search = _search_with_profile_recovery(
            prepared,
            progress=progress,
            cancelled=cancelled,
            checkpoint=checkpoint,
            task_runner=runner.run,
            fit_search_request=fit_search_request,
            run_fit_search=run_fit_search,
            recover_profile_basin=recover_profile_basin,
            continue_profile_basin=continue_profile_basin,
        )
        return run_analysis(
            analysis_request(
                prepared.dataset_id,
                prepared.problem,
                search,
                profile_names=profile_names,
                parameter_priors=prepared.updated_dataset.parameter_priors,
            ),
            cancelled=cancelled,
            progress=progress,
            task_runner=runner.run,
        )
