"""Sole service composition root for fitting and analysis domains.

The phase modules own reviewable orchestration logic without importing either
calculation domain. This module binds their explicit callable boundaries and
keeps every process entry point pickle-safe at module scope.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from types import SimpleNamespace

from xrr_fitter.analysis import sld_bands as _bands
from xrr_fitter.analysis.automatic import assess_automatic_quality
from xrr_fitter.analysis.joint import analyze_joint_ensemble
from xrr_fitter.analysis.mcmc import (
    prior_conflicts,
    run_problem_mcmc,
    with_parameter_priors,
)
from xrr_fitter.analysis.profiles import recover_profile_basin
from xrr_fitter.analysis.report import AnalysisRequest, run_analysis
from xrr_fitter.fit.automatic import (
    candidate_from_physical_values,
    refit_from_physical_values,
)
from xrr_fitter.fit.candidates import best_candidate_index, candidate_from_evaluation
from xrr_fitter.fit.initialization import structure_evidence
from xrr_fitter.fit.joint_candidates import (
    consensus_joint_vector,
    joint_candidate_vectors,
)
from xrr_fitter.fit.joint_evaluation import evaluate_joint_vector
from xrr_fitter.fit.joint_pipeline import JointFitRequest, run_joint_fit
from xrr_fitter.fit.joint_problem import compile_joint_problem
from xrr_fitter.fit.joint_sharing import (
    initial_joint_vector,
)
from xrr_fitter.fit.joint_sharing import (
    validate_sharing_rules as validate_compiled_sharing_rules,
)
from xrr_fitter.fit.local_search import SearchCancelled
from xrr_fitter.fit.objective import evaluate_declared_initial, evaluate_vector
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
from xrr_fitter.model.fitting import FitCheckpoint
from xrr_fitter.model.operations import FitReadiness, ProjectFitResult
from xrr_fitter.model.parameters import ParameterCoordinate
from xrr_fitter.model.project import XrrProject
from xrr_fitter.model.provenance import fit_search_provenance_sha256
from xrr_fitter.services.datasets import (
    SERVICE_SEED_TREE_VERSION,
    _prepared_current,
    service_seed_branches,
)
from xrr_fitter.services.fitting_phases import automatic_absorption as _absorption
from xrr_fitter.services.fitting_phases import automatic_dataset as _automatic
from xrr_fitter.services.fitting_phases import base as _base
from xrr_fitter.services.fitting_phases import joint_analysis as _joint_analysis
from xrr_fitter.services.fitting_phases import joint_execution as _joint_execution
from xrr_fitter.services.fitting_phases import operations as _operations
from xrr_fitter.services.fitting_phases.common import (
    AutomaticPreparedResult,
    CancellationProbe,
    CheckpointCallback,
    PreparedDatasetFit,
    ProgressCallback,
)
from xrr_fitter.services.fitting_phases.sharing import automatic_sharing_rules


def _validate_parameter_priors(definitions, priors) -> None:
    """Reject stale/duplicate prior sidecars during fit preflight."""
    if len({prior.name for prior in priors}) != len(priors):
        raise ValueError("parameter prior names must be unique")
    by_name = {definition.name: definition for definition in definitions}
    for prior in priors:
        definition = by_name.get(prior.name)
        if definition is None:
            raise ValueError(f"unknown parameter name: {prior.name}")
        if definition.constrained:
            raise ValueError(f"cannot assign a prior to constrained parameter: {prior.name}")
        replace(definition, prior=prior.prior)


def _dataset_index(project: XrrProject, dataset_id: str) -> int:
    try:
        return next(index for index, dataset in enumerate(project.datasets) if dataset.dataset_id == dataset_id)
    except StopIteration as error:
        raise ValueError(f"unknown dataset_id: {dataset_id}") from error


def _project_without_parameter_sidecars(
    project: XrrProject,
    index: int,
) -> XrrProject:
    datasets = list(project.datasets)
    datasets[index] = replace(
        datasets[index],
        parameter_settings=(),
        parameter_priors=(),
    )
    return replace(project, datasets=tuple(datasets))


def _reconciled_settings(
    project: XrrProject,
    index: int,
    settings,
) -> tuple:
    dataset = project.datasets[index]
    retained = []
    seen: set[str] = set()
    for setting in settings:
        if setting.name in seen:
            continue
        candidate = (*retained, setting)
        try:
            compiled_parameter_definitions(
                _prepared_current(project, dataset),
                dataset.structure,
                dataset.instrument,
                project.fit_config,
                candidate,
            )
        except ValueError:
            continue
        retained.append(setting)
        seen.add(setting.name)
    return tuple(retained)


def _reconciled_priors(definitions, priors) -> tuple:
    by_name = {definition.name: definition for definition in definitions}
    retained = []
    seen: set[str] = set()
    for prior in priors:
        if prior.name in seen:
            continue
        definition = by_name.get(prior.name)
        if definition is None or definition.locked or definition.constrained:
            continue
        try:
            replace(definition, prior=prior.prior)
        except ValueError:
            continue
        retained.append(prior)
        seen.add(prior.name)
    return tuple(retained)


def _reconcile_parameter_sidecars(project: XrrProject, dataset_id: str) -> XrrProject:
    """Keep only sidecars that compile against current dataset declarations."""
    index = _dataset_index(project, dataset_id)
    dataset = project.datasets[index]
    if dataset.structure is None:
        reconciled = replace(dataset, parameter_settings=(), parameter_priors=())
    elif not dataset.parameter_settings and not dataset.parameter_priors:
        return project
    else:
        clean = _project_without_parameter_sidecars(project, index)
        settings = _reconciled_settings(
            clean,
            index,
            dataset.parameter_settings,
        )
        definitions = compiled_parameter_definitions(
            _prepared_current(clean, clean.datasets[index]),
            dataset.structure,
            dataset.instrument,
            project.fit_config,
            settings,
        )
        reconciled = replace(
            dataset,
            parameter_settings=settings,
            parameter_priors=_reconciled_priors(
                definitions,
                dataset.parameter_priors,
            ),
        )
    datasets = list(project.datasets)
    datasets[index] = reconciled
    return replace(project, datasets=tuple(datasets))


def _sharing_problem_view(project: XrrProject, dataset: object) -> object:
    if dataset.structure is None:
        definitions = ()
    else:
        definitions = compiled_parameter_definitions(
            _prepared_current(project, dataset),
            dataset.structure,
            dataset.instrument,
            project.fit_config,
            dataset.parameter_settings,
        )
    variables = tuple(
        ParameterCoordinate(index, definition.name, definition.transform)
        for index, definition in enumerate(definitions)
        if not (definition.locked or definition.constrained)
    )
    return SimpleNamespace(
        parameter_definitions=definitions,
        variables=variables,
        instrument=dataset.instrument,
    )


def reconciled_sharing_rules(project: XrrProject) -> tuple:
    """Retain only sharing rules valid for the current effective declarations."""
    if not project.sharing_rules:
        return ()
    dataset_ids = tuple(dataset.dataset_id for dataset in project.datasets)
    problems = tuple(_sharing_problem_view(project, dataset) for dataset in project.datasets)
    retained = []
    for rule in project.sharing_rules:
        try:
            validate_compiled_sharing_rules(dataset_ids, problems, (rule,))
        except ValueError:
            continue
        retained.append(rule)
    return tuple(retained)


def structure_evidence_for(data, structure) -> StructureEvidence:
    """Translate fit-owned evidence into the public model value."""
    return _base.structure_evidence_for(
        data,
        structure,
        structure_evidence=structure_evidence,
    )


def parameter_definitions_for(data, structure, instrument, config):
    """Expose canonical parameter declarations through the service boundary."""
    return _base.parameter_definitions_for(
        data,
        structure,
        instrument,
        config,
        default_parameter_definitions=default_parameter_definitions,
    )


def compiled_parameter_definitions(
    data,
    structure,
    instrument,
    config,
    settings,
    constraint_rules=(),
):
    """Compile settings through the canonical fit problem boundary."""
    return _base.compiled_parameter_definitions(
        data,
        structure,
        instrument,
        config,
        settings,
        constraint_rules,
        compile_fit_problem=compile_fit_problem,
    )


def validate_parameter_setting_declarations(definitions, settings) -> None:
    """Apply fit-owned setting validation without returning a fit value."""
    _base.validate_parameter_setting_declarations(
        definitions,
        settings,
        apply_parameter_settings=apply_parameter_settings,
    )


def effective_parameter_definitions(definitions, settings):
    """Apply valid setting overrides without invoking dataset preflight."""
    return apply_parameter_settings(tuple(definitions), tuple(settings))


def _compile_dataset(
    project: XrrProject,
    dataset_id: str,
    *,
    master_seed: int,
) -> PreparedDatasetFit:
    return _base._compile_dataset(
        project,
        dataset_id,
        master_seed=master_seed,
        compile_fit_problem=compile_fit_problem,
        structure_evidence=structure_evidence,
    )


def prepare_dataset_fit(
    project: XrrProject,
    dataset_id: str,
    seed: int,
) -> PreparedDatasetFit:
    """Parse and compile one dataset against its persisted source identity."""
    return _base.prepare_dataset_fit(
        project,
        dataset_id,
        seed,
        compile_fit_problem=compile_fit_problem,
        structure_evidence=structure_evidence,
    )


def fit_prepared_dataset(
    prepared: PreparedDatasetFit,
    *,
    progress: ProgressCallback | None = None,
    cancelled: CancellationProbe | None = None,
    checkpoint: Callable[[FitCheckpoint], None] | None = None,
    local_workers: int | None = None,
    profile_names: tuple[str, ...] | None = None,
) -> FitResult:
    """Run one independent search, optional recovery, and final analysis."""
    return _base.fit_prepared_dataset(
        prepared,
        progress=progress,
        cancelled=cancelled,
        checkpoint=checkpoint,
        local_workers=local_workers,
        profile_names=profile_names,
        fit_search_request=FitSearchRequest,
        run_fit_search=run_fit_search,
        recover_profile_basin=recover_profile_basin,
        continue_profile_basin=continue_profile_basin,
        analysis_request=AnalysisRequest,
        run_analysis=run_analysis,
    )


def _automatic_profile_recovery(
    prepared: PreparedDatasetFit,
    search,
    *,
    progress: ProgressCallback | None,
    cancelled: CancellationProbe | None,
    checkpoint: Callable[[FitCheckpoint], None] | None,
    task_runner: Callable,
):
    return _automatic._automatic_profile_recovery(
        prepared,
        search,
        progress=progress,
        cancelled=cancelled,
        checkpoint=checkpoint,
        task_runner=task_runner,
        recover_profile_basin=recover_profile_basin,
        continue_profile_basin=continue_profile_basin,
    )


def _automatic_absorption_search(
    prepared: PreparedDatasetFit,
    search,
    names: tuple[str, ...],
    *,
    cancelled: CancellationProbe | None,
):
    return _absorption._automatic_absorption_search(
        prepared,
        search,
        names,
        cancelled=cancelled,
        compile_fit_problem=compile_fit_problem,
        refit_from_physical_values=refit_from_physical_values,
        candidate_from_physical_values=candidate_from_physical_values,
        evaluate_vector=evaluate_vector,
        candidate_from_evaluation=candidate_from_evaluation,
        best_candidate_index=best_candidate_index,
        fit_search_provenance_sha256=fit_search_provenance_sha256,
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
    return _automatic.fit_automatic_prepared_dataset(
        prepared,
        progress=progress,
        cancelled=cancelled,
        checkpoint=checkpoint,
        local_workers=local_workers,
        fit_search_request=FitSearchRequest,
        run_fit_search=run_fit_search,
        analysis_request=AnalysisRequest,
        run_analysis=run_analysis,
        assess_automatic_quality=assess_automatic_quality,
        automatic_profile_recovery=_automatic_profile_recovery,
        automatic_absorption_search=_automatic_absorption_search,
    )


def _analyze_joint_searches(problem, searches, priors) -> tuple[FitResult, ...]:
    return _joint_analysis._analyze_joint_searches(
        problem,
        searches,
        priors,
        joint_candidate_vectors=joint_candidate_vectors,
        analyze_joint_ensemble=analyze_joint_ensemble,
        with_parameter_priors=with_parameter_priors,
        prior_conflicts=prior_conflicts,
    )


def fit_automatic_joint_group(
    prepared: tuple[PreparedDatasetFit, ...],
    prefits: tuple[AutomaticPreparedResult, ...],
    fit_group_id: str,
    *,
    progress: ProgressCallback | None = None,
    cancelled: CancellationProbe | None = None,
    checkpoint: Callable[[tuple[FitCheckpoint, ...]], None] | None = None,
) -> tuple[AutomaticPreparedResult, ...]:
    """Refine qualified prefits jointly and retry isolated points independently."""
    return _joint_execution.fit_automatic_joint_group(
        prepared,
        prefits,
        fit_group_id,
        progress=progress,
        cancelled=cancelled,
        checkpoint=checkpoint,
        compile_fit_problem=compile_fit_problem,
        compile_joint_problem=compile_joint_problem,
        consensus_joint_vector=consensus_joint_vector,
        joint_fit_request=JointFitRequest,
        run_joint_fit=run_joint_fit,
        analysis_request=AnalysisRequest,
        run_analysis=run_analysis,
        assess_automatic_quality=assess_automatic_quality,
        analyze_joint_searches=_analyze_joint_searches,
        fit_automatic_prepared_dataset=fit_automatic_prepared_dataset,
        cancellation_exceptions=(SearchCancelled, InterruptedError),
    )


def fit_joint_datasets(
    prepared: tuple[PreparedDatasetFit, ...],
    sharing_rules: tuple,
    constraint_rules: tuple = (),
    *,
    progress: ProgressCallback | None = None,
    cancelled: CancellationProbe | None = None,
    checkpoint: Callable[[tuple[FitCheckpoint, ...]], None] | None = None,
) -> tuple[FitResult, ...]:
    """Run and analyze one joint graph without independent fallback."""
    return _joint_execution.fit_joint_datasets(
        prepared,
        sharing_rules,
        constraint_rules,
        progress=progress,
        cancelled=cancelled,
        checkpoint=checkpoint,
        compile_joint_problem=compile_joint_problem,
        joint_fit_request=JointFitRequest,
        run_joint_fit=run_joint_fit,
        analyze_joint_searches=_analyze_joint_searches,
    )


def preflight_fit(project: XrrProject) -> FitReadiness:
    """Load and compile the complete declared fit without mutating the project."""
    return _operations.preflight_fit(
        project,
        prepare_dataset_fit=prepare_dataset_fit,
        compile_joint_problem=compile_joint_problem,
        validate_parameter_priors=_validate_parameter_priors,
        evaluate_declared_initial=evaluate_declared_initial,
        initial_joint_vector=initial_joint_vector,
        evaluate_joint_vector=evaluate_joint_vector,
    )


def _automatic_dataset_ids(
    project: XrrProject,
    import_batch_id: str | None,
) -> tuple[str, ...]:
    return _operations._automatic_dataset_ids(project, import_batch_id)


def preflight_automatic_fit(
    project: XrrProject,
    import_batch_id: str | None = None,
) -> FitReadiness:
    """Validate only runnable automatic datasets without mutating state."""
    return _operations.preflight_automatic_fit(
        project,
        import_batch_id,
        prepare_dataset_fit=prepare_dataset_fit,
        validate_parameter_priors=_validate_parameter_priors,
        evaluate_declared_initial=evaluate_declared_initial,
    )


def fit_automatically(
    project: XrrProject,
    import_batch_id: str | None = None,
    progress_callback: ProgressCallback | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
) -> ProjectFitResult:
    """Run the persisted automatic route through the batch transaction."""
    from xrr_fitter.services.batch import fit_automatic_transaction

    return _operations.fit_automatically(
        project,
        import_batch_id,
        progress_callback,
        checkpoint_callback,
        fit_automatic_transaction=fit_automatic_transaction,
        prepare_dataset_fit=prepare_dataset_fit,
        validate_parameter_priors=_validate_parameter_priors,
        evaluate_declared_initial=evaluate_declared_initial,
        fit_automatic_prepared_dataset=fit_automatic_prepared_dataset,
        fit_automatic_joint_group=fit_automatic_joint_group,
    )


def _dispatch_project(
    project: XrrProject,
    progress_callback: ProgressCallback | None,
    checkpoint_callback: CheckpointCallback | None,
    cancelled: CancellationProbe | None,
) -> ProjectFitResult:
    from xrr_fitter.services.batch import fit_project_transaction

    return _operations._dispatch_project(
        project,
        progress_callback,
        checkpoint_callback,
        cancelled,
        preflight_fit=preflight_fit,
        fit_project_transaction=fit_project_transaction,
        prepare_dataset_fit=prepare_dataset_fit,
        fit_prepared_dataset=fit_prepared_dataset,
        fit_joint_datasets=fit_joint_datasets,
    )


def fit_project(
    project: XrrProject,
    progress_callback: ProgressCallback | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
) -> ProjectFitResult:
    """Dispatch a synchronous fit through the batch transaction owner."""
    return _dispatch_project(project, progress_callback, checkpoint_callback, None)


def _mcmc_problem(project: XrrProject, dataset_id: str):
    return _operations._mcmc_problem(
        project,
        dataset_id,
        compile_dataset=_compile_dataset,
    )


def _sld_bands(structure, report, wavelength_a):
    """Replay retained samples into SLD bands, folding failures into warnings."""
    if report is None:
        return None, None
    try:
        bands = _bands.sld_uncertainty_bands(structure, report, wavelength_a=wavelength_a)
    except ValueError as error:
        return None, replace(report, warnings=(*report.warnings, str(error)))
    return bands, report


def _run_mcmc(
    project: XrrProject,
    dataset_id: str,
    candidate_id: str,
    config: McmcConfig,
    progress_callback: ProgressCallback | None,
    cancelled: CancellationProbe | None,
) -> XrrProject:
    return _operations._run_mcmc(
        project,
        dataset_id,
        candidate_id,
        config,
        progress_callback,
        cancelled,
        compile_dataset=_compile_dataset,
        with_parameter_priors=with_parameter_priors,
        run_problem_mcmc=run_problem_mcmc,
        sld_bands=_sld_bands,
    )


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


def sld_uncertainty_bands(structure, report, *, wavelength_a, align="backing"):
    """Recompute view-only bands for an alignment the user picked."""
    return _bands.sld_uncertainty_bands(structure, report, wavelength_a=wavelength_a, align=align)


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


def automatic_worker_handler(
    project: XrrProject,
    import_batch_id: str | None,
    progress_callback: ProgressCallback | None,
    checkpoint_callback: CheckpointCallback | None,
    cancelled: CancellationProbe | None,
) -> ProjectFitResult:
    from xrr_fitter.services.batch import fit_automatic_transaction

    return _operations.automatic_worker_handler(
        project,
        import_batch_id,
        progress_callback,
        checkpoint_callback,
        cancelled,
        fit_automatic_transaction=fit_automatic_transaction,
        prepare_dataset_fit=prepare_dataset_fit,
        validate_parameter_priors=_validate_parameter_priors,
        evaluate_declared_initial=evaluate_declared_initial,
        fit_automatic_prepared_dataset=fit_automatic_prepared_dataset,
        fit_automatic_joint_group=fit_automatic_joint_group,
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


__all__ = (
    "AutomaticPreparedResult",
    "PreparedDatasetFit",
    "SERVICE_SEED_TREE_VERSION",
    "automatic_sharing_rules",
    "automatic_worker_handler",
    "compiled_parameter_definitions",
    "fit_automatic_joint_group",
    "fit_automatic_prepared_dataset",
    "fit_automatically",
    "fit_joint_datasets",
    "fit_prepared_dataset",
    "fit_project",
    "fit_worker_handler",
    "mcmc_worker_handler",
    "parameter_definitions_for",
    "preflight_automatic_fit",
    "preflight_fit",
    "prepare_dataset_fit",
    "run_mcmc",
    "service_seed_branches",
    "sld_uncertainty_bands",
    "structure_evidence_for",
    "validate_parameter_setting_declarations",
)
