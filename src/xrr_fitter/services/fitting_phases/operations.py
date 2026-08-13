"""Public fitting, preflight, dispatch, and worker handlers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from xrr_fitter.model.analysis import McmcConfig
from xrr_fitter.model.automation import AutomaticRole, AutomaticStatus
from xrr_fitter.model.fitting import FitProgress
from xrr_fitter.model.operations import FitReadiness, ProjectFitResult
from xrr_fitter.model.project import XrrProject
from xrr_fitter.services.datasets import mcmc_candidate_seed, service_seed_branches
from xrr_fitter.services.projects import inspect_sources

from .common import CancellationProbe, CheckpointCallback, ProgressCallback

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


def _compile_preflight_fit(
    project: XrrProject,
    *,
    prepare_dataset_fit: Callable,
    compile_joint_problem: Callable,
) -> None:
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


def preflight_fit(
    project: XrrProject,
    *,
    prepare_dataset_fit: Callable,
    compile_joint_problem: Callable,
) -> FitReadiness:
    """Load and compile the complete declared fit without mutating the project."""
    if not project.datasets:
        return FitReadiness(False, "project has no datasets")
    try:
        validation = inspect_sources(project)
        failure = _source_failure(validation)
        if failure is not None:
            return FitReadiness(False, failure)
        _compile_preflight_fit(
            project,
            prepare_dataset_fit=prepare_dataset_fit,
            compile_joint_problem=compile_joint_problem,
        )
    except Exception as error:
        return FitReadiness(False, str(error) or type(error).__name__)
    return FitReadiness(True, "ready")


AUTOMATIC_RUNNABLE = frozenset(
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
        and dataset.automation.status in AUTOMATIC_RUNNABLE
        and (
            import_batch_id is None
            or dataset.automation.import_batch_id == import_batch_id
        )
    )


def preflight_automatic_fit(
    project: XrrProject,
    import_batch_id: str | None = None,
    *,
    prepare_dataset_fit: Callable,
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
    *,
    fit_automatic_transaction: Callable,
    prepare_dataset_fit: Callable,
    fit_automatic_prepared_dataset: Callable,
    fit_automatic_joint_group: Callable,
) -> ProjectFitResult:
    """Run the persisted automatic route through the batch transaction."""
    readiness = preflight_automatic_fit(
        project,
        import_batch_id,
        prepare_dataset_fit=prepare_dataset_fit,
    )
    if not readiness.ready:
        raise ValueError(readiness.message)
    return fit_automatic_transaction(
        project,
        import_batch_id,
        progress_callback,
        checkpoint_callback,
        None,
        seed_branches=service_seed_branches,
        prepare_dataset=prepare_dataset_fit,
        fit_dataset=fit_automatic_prepared_dataset,
        fit_joint=fit_automatic_joint_group,
    )


def fit_project(
    project: XrrProject,
    progress_callback: ProgressCallback | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
    *,
    fit_project_transaction: Callable,
    prepare_dataset_fit: Callable,
    fit_prepared_dataset: Callable,
    fit_joint_datasets: Callable,
) -> ProjectFitResult:
    """Dispatch a synchronous fit through the batch transaction owner."""
    return _dispatch_project(
        project,
        progress_callback,
        checkpoint_callback,
        None,
        fit_project_transaction=fit_project_transaction,
        prepare_dataset_fit=prepare_dataset_fit,
        fit_prepared_dataset=fit_prepared_dataset,
        fit_joint_datasets=fit_joint_datasets,
    )


def _dispatch_project(
    project: XrrProject,
    progress_callback: ProgressCallback | None,
    checkpoint_callback: CheckpointCallback | None,
    cancelled: CancellationProbe | None,
    *,
    fit_project_transaction: Callable,
    prepare_dataset_fit: Callable,
    fit_prepared_dataset: Callable,
    fit_joint_datasets: Callable,
) -> ProjectFitResult:
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


def _mcmc_problem(
    project: XrrProject,
    dataset_id: str,
    *,
    compile_dataset: Callable,
):
    prepared = compile_dataset(project, dataset_id, master_seed=project.master_seed)
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
    *,
    compile_dataset: Callable,
    run_problem_mcmc: Callable,
) -> XrrProject:
    validation = inspect_sources(project)
    failure = _source_failure(validation)
    if failure is not None:
        raise ValueError(failure)
    prepared, result = _mcmc_problem(
        project,
        dataset_id,
        compile_dataset=compile_dataset,
    )
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
    *,
    compile_dataset: Callable,
    run_problem_mcmc: Callable,
) -> XrrProject:
    return _run_mcmc(
        project,
        dataset_id,
        candidate_id,
        config,
        progress_callback,
        None,
        compile_dataset=compile_dataset,
        run_problem_mcmc=run_problem_mcmc,
    )


def fit_worker_handler(
    project: XrrProject,
    progress_callback: ProgressCallback | None,
    checkpoint_callback: CheckpointCallback | None,
    cancelled: CancellationProbe | None,
    *,
    fit_project_transaction: Callable,
    prepare_dataset_fit: Callable,
    fit_prepared_dataset: Callable,
    fit_joint_datasets: Callable,
) -> ProjectFitResult:
    return _dispatch_project(
        project,
        progress_callback,
        checkpoint_callback,
        cancelled,
        fit_project_transaction=fit_project_transaction,
        prepare_dataset_fit=prepare_dataset_fit,
        fit_prepared_dataset=fit_prepared_dataset,
        fit_joint_datasets=fit_joint_datasets,
    )


def automatic_worker_handler(
    project: XrrProject,
    import_batch_id: str | None,
    progress_callback: ProgressCallback | None,
    checkpoint_callback: CheckpointCallback | None,
    cancelled: CancellationProbe | None,
    *,
    fit_automatic_transaction: Callable,
    prepare_dataset_fit: Callable,
    fit_automatic_prepared_dataset: Callable,
    fit_automatic_joint_group: Callable,
) -> ProjectFitResult:
    return fit_automatic_transaction(
        project,
        import_batch_id,
        progress_callback,
        checkpoint_callback,
        cancelled,
        seed_branches=service_seed_branches,
        prepare_dataset=prepare_dataset_fit,
        fit_dataset=fit_automatic_prepared_dataset,
        fit_joint=fit_automatic_joint_group,
    )


def mcmc_worker_handler(
    project: XrrProject,
    dataset_id: str,
    candidate_id: str,
    config: McmcConfig,
    progress_callback: ProgressCallback | None,
    cancelled: CancellationProbe | None,
    *,
    compile_dataset: Callable,
    run_problem_mcmc: Callable,
) -> XrrProject:
    return _run_mcmc(
        project,
        dataset_id,
        candidate_id,
        config,
        progress_callback,
        cancelled,
        compile_dataset=compile_dataset,
        run_problem_mcmc=run_problem_mcmc,
    )
