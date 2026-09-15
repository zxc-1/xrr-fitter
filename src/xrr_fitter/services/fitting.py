"""Sole service composition root for fitting and analysis domains.

The phase modules own reviewable orchestration logic without importing either
calculation domain. This module binds their explicit callable boundaries and
keeps every process entry point pickle-safe at module scope.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import partial
from threading import Lock
from time import sleep as _sleep

from xrr_fitter.analysis import sld_bands as _bands
from xrr_fitter.analysis.automatic import assess_automatic_quality
from xrr_fitter.analysis.bootstrap_generation import poll_cancelled
from xrr_fitter.analysis.joint import analyze_joint_ensemble, analyze_joint_point
from xrr_fitter.analysis.joint_bootstrap import bootstrap_joint_local
from xrr_fitter.analysis.mcmc import (
    prior_conflicts,
    run_problem_mcmc,
    with_parameter_priors,
)
from xrr_fitter.analysis.profiles import recover_profile_basin
from xrr_fitter.analysis.report import AnalysisRequest, uncertainty_seed
from xrr_fitter.analysis.report import run_analysis as _run_analysis
from xrr_fitter.analysis.residual_calibration import calibrate_residuals, qualify_poisson_bootstrap
from xrr_fitter.fit.automatic import (
    candidate_from_physical_values,
    refit_from_physical_values,
)
from xrr_fitter.fit.candidates import best_candidate_index, candidate_from_evaluation
from xrr_fitter.fit.diagnostic_refit import refit_diagnostic_joint, refit_diagnostic_single
from xrr_fitter.fit.initialization import structure_evidence
from xrr_fitter.fit.joint_candidates import (
    consensus_joint_vector,
    joint_candidate_vectors,
)
from xrr_fitter.fit.joint_evaluation import evaluate_joint_vector, joint_inference_layout
from xrr_fitter.fit.joint_pipeline import JointFitRequest, run_joint_fit
from xrr_fitter.fit.joint_problem import compile_joint_problem
from xrr_fitter.fit.joint_sharing import (
    initial_joint_vector,
)
from xrr_fitter.fit.joint_sharing import (
    validate_sharing_rules as validate_compiled_sharing_rules,
)
from xrr_fitter.fit.joint_solvers import refit_resampled_joint
from xrr_fitter.fit.local_search import SearchCancelled, StageSkipped
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
from xrr_fitter.fit.problem import compile_fit_problem, recompile_resampled_problem
from xrr_fitter.model.analysis import FitResult, McmcConfig, StructureEvidence
from xrr_fitter.model.fitting import FitCheckpoint, FitProgress
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.joint_bootstrap_provenance import joint_bootstrap_owner_sha256
from xrr_fitter.model.operations import FitReadiness, ProjectFitResult
from xrr_fitter.model.parameters import (
    ParameterCoordinate,
    ParameterDefinition,
    ParameterPrior,
    ParameterSetting,
    SharingRule,
)
from xrr_fitter.model.project import DatasetProject, XrrProject
from xrr_fitter.model.provenance import fit_search_provenance_sha256, joint_residual_owner_sha256
from xrr_fitter.services import bootstrap_ownership as _bootstrap_ownership
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

# 暂停期间探针醒来查看的周期。取消要在这个量级内被听见，而不是等到下一个阶段边界。
PAUSE_POLL_SECONDS = 0.05


def run_analysis(request, *, cancelled=None, progress=None, task_runner=None):
    return _run_analysis(
        request,
        cancelled=cancelled,
        progress=progress,
        task_runner=task_runner,
        recompile=recompile_resampled_problem,
        diagnostic_refit=lambda members: refit_diagnostic_single(members[0], cancelled=cancelled),
    )


def pause_aware_probe(cancellation, pause, skip=None, *, sleep=_sleep):
    """Poll cancellation first, consume one stage skip, then honor pause."""
    skip_lock = Lock()

    def probe() -> bool:
        while True:
            if cancellation.is_set():
                return True
            with skip_lock:
                if skip is not None and skip.is_set():
                    skip.clear()
                    raise StageSkipped("stage skipped by request")
            if pause is None or not pause.is_set():
                return False
            sleep(PAUSE_POLL_SECONDS)

    return probe


class _WorkerCancellation:
    """Give solver-owned search scopes a skip-capable view of worker control."""

    def __init__(self, cancellation, pause, skip) -> None:
        self._skip = skip
        self._normal_probe = pause_aware_probe(cancellation, pause)
        self._search_probe = pause_aware_probe(cancellation, pause, skip)

    def _discard_skip(self) -> None:
        if self._skip is not None:
            self._skip.clear()

    def __call__(self) -> bool:
        self._discard_skip()
        return self._normal_probe()

    @contextmanager
    def search_scope(self):
        # Requests made before or after a search must not skip the next dataset.
        self._discard_skip()
        try:
            yield self._search_probe
        finally:
            self._discard_skip()


def build_cancellation_probe(cancellation, pause, skip=None):
    """Build normal worker control; only a search boundary enables stage skip."""
    if pause is None and skip is None:
        return cancellation.is_set
    return _WorkerCancellation(cancellation, pause, skip)


def _run_skippable_search(run_search, request, *, cancelled=None, **kwargs):
    if isinstance(cancelled, _WorkerCancellation):
        with cancelled.search_scope() as probe:
            return run_search(request, cancelled=probe, **kwargs)
    return run_search(request, cancelled=cancelled, **kwargs)


@dataclass(frozen=True, slots=True)
class _SharingProblemView:
    parameter_definitions: tuple[ParameterDefinition, ...]
    variables: tuple[ParameterCoordinate, ...]
    instrument: InstrumentSpec


def _validate_parameter_priors(
    definitions: tuple[ParameterDefinition, ...],
    priors: tuple[ParameterPrior, ...],
) -> None:
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
    settings: tuple[ParameterSetting, ...],
) -> tuple[ParameterSetting, ...]:
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


def _reconciled_priors(
    definitions: tuple[ParameterDefinition, ...],
    priors: tuple[ParameterPrior, ...],
) -> tuple[ParameterPrior, ...]:
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


def _sharing_problem_view(project: XrrProject, dataset: DatasetProject) -> _SharingProblemView:
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
    return _SharingProblemView(
        parameter_definitions=definitions,
        variables=variables,
        instrument=dataset.instrument,
    )


def reconciled_sharing_rules(project: XrrProject) -> tuple[SharingRule, ...]:
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


def validate_project_bootstrap_ownership(project: XrrProject) -> None:
    """Rebuild joint persistence owners through the sole numerical composition root."""
    _bootstrap_ownership.validate_project_bootstrap_ownership(
        project,
        prepare_dataset=prepare_dataset_fit,
        compile_joint_problem=compile_joint_problem,
        joint_candidate_vectors=joint_candidate_vectors,
        uncertainty_seed=uncertainty_seed,
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
        run_fit_search=partial(_run_skippable_search, run_fit_search),
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
        run_fit_search=partial(_run_skippable_search, run_fit_search),
        analysis_request=AnalysisRequest,
        run_analysis=run_analysis,
        assess_automatic_quality=assess_automatic_quality,
        automatic_profile_recovery=_automatic_profile_recovery,
        automatic_absorption_search=_automatic_absorption_search,
    )


def _joint_diagnostic_progress(progress, objective):
    def publish(completed, total):
        if progress is not None:
            progress(
                FitProgress(
                    None,
                    "diagnostics",
                    completed,
                    total,
                    objective,
                    f"Poisson diagnostic calibration {completed}/{total}",
                )
            )

    return publish


def _joint_point_evidence(problem, vector, *, cancelled=None, progress=None, task_runner=None, cache=None):
    poll_cancelled(cancelled)
    evaluation = evaluate_joint_vector(problem, vector)
    owner = joint_residual_owner_sha256(
        problem.problems,
        problem.dataset_ids,
        vector,
        evaluation.local_evaluations,
        problem.layout_fingerprint,
    )
    poll_cancelled(cancelled)
    if cache is not None and owner in cache:
        return cache[owner]
    residuals = calibrate_residuals(
        problem.problems,
        problem.dataset_ids,
        vector,
        evaluation.local_evaluations,
        owner_sha256=owner,
        refit=partial(refit_diagnostic_joint, problem, cancelled=cancelled),
        recompile=recompile_resampled_problem,
        cancelled=cancelled,
        task_runner=task_runner,
        progress=_joint_diagnostic_progress(progress, evaluation.objective),
    )
    evidence = analyze_joint_point(
        tuple(variable.name for variable in problem.global_variables),
        problem.dataset_ids,
        problem.problems,
        vector,
        evaluation.local_evaluations,
        lambda: joint_inference_layout(problem, vector),
        residual_evidence=residuals,
        layout_fingerprint=problem.layout_fingerprint,
    )
    poll_cancelled(cancelled)
    if cache is not None:
        cache[owner] = evidence
    return evidence


def _joint_winner_candidates(searches, candidate_id):
    return tuple(
        next(candidate for candidate in search.candidates if candidate.candidate_id == candidate_id)
        for search in searches
    )


def _joint_bootstrap_owner(problem, searches, candidate_id, vector):
    return joint_bootstrap_owner_sha256(
        problem,
        _joint_winner_candidates(searches, candidate_id),
        vector,
        uncertainty_seed(problem.problems[0].config),
    )


def _joint_bootstrap(
    problem, searches, candidate_id, vector, *, cancelled=None, progress=None, task_runner=None, point_evidence=None
):
    evaluation = evaluate_joint_vector(problem, vector)
    config = problem.problems[0].config
    point = point_evidence or partial(
        _joint_point_evidence,
        cancelled=cancelled,
        progress=progress,
        task_runner=task_runner,
    )
    _covariance, residuals = point(problem, vector)

    def publish(completed, total):
        if progress is not None:
            progress(
                FitProgress(
                    None, "bootstrap", completed, total, evaluation.objective, f"joint bootstrap {completed}/{total}"
                )
            )

    publish(0, config.budget.bootstrap_samples)
    sampling = bootstrap_joint_local(
        problem,
        _joint_winner_candidates(searches, candidate_id),
        vector,
        sample_count=config.budget.bootstrap_samples,
        child_seed=uncertainty_seed(config),
        recompile=recompile_resampled_problem,
        refit=partial(refit_resampled_joint, problem, vector, cancelled=cancelled),
        cancelled=cancelled,
        progress=publish,
        task_runner=task_runner,
    )
    return qualify_poisson_bootstrap(sampling, problem.problems, residuals)


def _analyze_joint_searches(
    problem,
    searches,
    priors,
    *,
    bootstrap_enabled=False,
    cancelled=None,
    progress=None,
    task_runner=None,
) -> tuple[FitResult, ...]:
    # This operation alone owns the cache; even the bootstrap checks the exact
    # numerical owner before reusing the common family and covariance evidence.
    point_evidence = partial(
        _joint_point_evidence,
        cancelled=cancelled,
        progress=progress,
        task_runner=task_runner,
        cache={},
    )
    bootstrap = (
        partial(
            _joint_bootstrap,
            problem,
            searches,
            cancelled=cancelled,
            progress=progress,
            task_runner=task_runner,
            point_evidence=point_evidence,
        )
        if bootstrap_enabled
        else None
    )
    return _joint_analysis._analyze_joint_searches(
        problem,
        searches,
        priors,
        joint_candidate_vectors=joint_candidate_vectors,
        analyze_joint_ensemble=analyze_joint_ensemble,
        joint_point_evidence=point_evidence,
        with_parameter_priors=with_parameter_priors,
        prior_conflicts=prior_conflicts,
        bootstrap=bootstrap,
        bootstrap_owner=partial(_joint_bootstrap_owner, problem, searches),
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
        run_joint_fit=partial(_run_skippable_search, run_joint_fit),
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
        run_joint_fit=partial(_run_skippable_search, run_joint_fit),
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
