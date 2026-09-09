"""Immutable dataset result and checkpoint publication for batch transactions.

The transaction owner determines execution order and cancellation boundaries.
These helpers update dataset values without scheduling work or calling solvers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from xrr_fitter.model.analysis import FitResult
from xrr_fitter.model.automation import AutomaticRole, AutomaticStatus
from xrr_fitter.model.fitting import FitCheckpoint
from xrr_fitter.model.operations import DatasetFitResult
from xrr_fitter.model.parameters import ParameterSetting
from xrr_fitter.model.project import DatasetProject, ScalePriorState, XrrProject
from xrr_fitter.services.fitting_phases.common import AutomaticPreparedResult, PreparedDatasetFit


def _clear_dataset(dataset: DatasetProject) -> DatasetProject:
    """Remove numerical state while retaining declared dataset inputs.

    Source, structure, instrument, and parameter declarations remain intact.
    """

    return replace(
        dataset,
        scale_prior=ScalePriorState(enabled=False),
        last_valid_result=None,
        checkpoint=None,
    )


def _replace_dataset(project: XrrProject, index: int, dataset: DatasetProject) -> XrrProject:
    """Publish one dataset replacement without mutating the project.

    Dataset order is a persisted contract and is never recomputed here.
    """

    datasets = list(project.datasets)
    datasets[index] = dataset
    return replace(project, datasets=tuple(datasets))


def _clear_all(project: XrrProject) -> XrrProject:
    """Invalidate every result participating in an expert joint graph.

    A joint result is not publishable when any member transaction fails.
    """

    return replace(project, datasets=tuple(map(_clear_dataset, project.datasets)))


def _failure_result(dataset: DatasetProject, error: BaseException | str) -> FitResult:
    """Represent a transaction failure as an untrusted dataset result.

    Region arrays retain source length so downstream rendering stays aligned.
    """

    message = error if isinstance(error, str) else f"{type(error).__name__}: {error}"
    return FitResult.failed(message, len(dataset.fit_mask))


def _warnings(results: tuple[DatasetFitResult, ...]) -> tuple[str, ...]:
    """Flatten dataset warnings in publication order.

    The project result preserves dataset order and warning order within rows.
    """

    return tuple(warning for item in results for warning in item.fit_result.warnings)


def _checkpoint_with_result_diagnostics(
    checkpoint: FitCheckpoint | None,
    result: FitResult,
) -> FitCheckpoint | None:
    """Project final candidate diagnostics back onto a saved checkpoint.

    Candidate identity, rather than tuple position, controls the replacement.
    """

    if checkpoint is None:
        return None
    result_candidates = {candidate.candidate_id: candidate for candidate in result.candidates}
    candidates = tuple(
        replace(
            candidate,
            diagnostics=result_candidates[candidate.candidate_id].diagnostics,
        )
        if candidate.candidate_id in result_candidates
        else candidate
        for candidate in checkpoint.candidates
    )
    return replace(checkpoint, candidates=candidates)


def _replay_checkpoints(
    working: XrrProject,
    index: int,
    checkpoints: tuple[FitCheckpoint | None, ...],
    checkpoint_callback: Callable[[XrrProject], None] | None,
) -> XrrProject:
    """Commit one dataset's checkpoints in project order after it finishes."""
    for value in checkpoints:
        dataset = replace(working.datasets[index], checkpoint=value)
        working = _replace_dataset(working, index, dataset)
        if checkpoint_callback is not None:
            checkpoint_callback(working)
    return working


def _commit_success(
    working: XrrProject,
    index: int,
    fit_result: FitResult,
) -> XrrProject:
    """Publish one successful independent fit and its final diagnostics.

    The latest checkpoint is retained as resumable provenance for the result.
    """

    dataset = working.datasets[index]
    dataset = replace(
        dataset,
        last_valid_result=fit_result,
        checkpoint=_checkpoint_with_result_diagnostics(
            dataset.checkpoint,
            fit_result,
        ),
    )
    return _replace_dataset(working, index, dataset)


def _automatic_fit_parts(result: AutomaticPreparedResult) -> tuple[PreparedDatasetFit, FitResult, bool, str | None]:
    """Validate the structural result contract returned by a fit service.

    Batch publication depends only on these four service-owned fields.
    """

    prepared = getattr(result, "prepared", None)
    fit_result = getattr(result, "fit_result", None)
    passed = getattr(result, "passed", None)
    reason = getattr(result, "reason", None)
    if prepared is None or not isinstance(fit_result, FitResult):
        raise TypeError("automatic fit must return AutomaticPreparedResult")
    if not isinstance(passed, bool):
        raise TypeError("automatic fit result passed flag must be bool")
    return prepared, fit_result, passed, reason


def _winner_settings(
    current: tuple[ParameterSetting, ...],
    fit_result: FitResult,
) -> tuple[ParameterSetting, ...]:
    """Freeze the winning physical vector into persisted parameter settings.

    Incomplete candidate vectors leave the caller's declarations unchanged.
    """

    best = fit_result.best_candidate
    if best is None or not fit_result.parameter_definitions:
        return current
    values = {parameter.name: parameter.value for parameter in best.parameters}
    if any(definition.name not in values for definition in fit_result.parameter_definitions):
        return current
    return tuple(
        ParameterSetting(
            definition.name,
            values[definition.name],
            definition.lower,
            definition.upper,
            locked=definition.locked,
        )
        for definition in fit_result.parameter_definitions
    )


def _automatic_status(
    fit_result: FitResult,
    passed: bool,
    refining: bool,
) -> AutomaticStatus:
    """Map candidate validity and quality state onto publication status.

    Joint prefits remain ``REFINING`` until their group decision completes.
    """

    best = fit_result.best_candidate
    if best is None or not best.valid:
        return AutomaticStatus.FAILED
    if refining:
        return AutomaticStatus.REFINING
    return AutomaticStatus.PASSED if passed else AutomaticStatus.REVIEW


def _automatic_reason(
    status: AutomaticStatus,
    reason: str | None,
    role: AutomaticRole,
) -> str | None:
    """Normalize the audit reason associated with an automatic status.

    Successful isolated retries retain their isolation reason for provenance.
    """

    if status is AutomaticStatus.PASSED:
        return reason if role is AutomaticRole.ISOLATED_RETRY else None
    if status is AutomaticStatus.REFINING:
        return reason
    if reason:
        return reason
    if status is AutomaticStatus.FAILED:
        return "no valid automatic candidate"
    return "automatic quality review required"


def _commit_automatic_result(
    working: XrrProject,
    index: int,
    previous_prepared: PreparedDatasetFit | None,
    result: AutomaticPreparedResult,
    *,
    refining: bool,
) -> tuple[XrrProject, DatasetFitResult]:
    """Publish one automatic result with coherent settings and checkpoints.

    Parameter changes invalidate stale checkpoints; invalid winners clear the
    publishable result while preserving the prepared declaration state.
    """

    prepared, fit_result, passed, reason = _automatic_fit_parts(result)
    current = working.datasets[index]
    prepared_dataset = prepared.updated_dataset
    winner_settings = _winner_settings(
        prepared_dataset.parameter_settings,
        fit_result,
    )
    changed_settings = (
        previous_prepared is not None
        and prepared_dataset.parameter_settings != previous_prepared.updated_dataset.parameter_settings
    )
    status = _automatic_status(fit_result, passed, refining)
    settings_changed = winner_settings != prepared_dataset.parameter_settings
    if status is AutomaticStatus.FAILED:
        checkpoint = None
        last_valid_result = None
        persisted_settings = prepared_dataset.parameter_settings
    else:
        checkpoint = (
            None if settings_changed else prepared_dataset.checkpoint if changed_settings else current.checkpoint
        )
        last_valid_result = fit_result
        persisted_settings = winner_settings
    automation = replace(
        prepared_dataset.automation,
        status=status,
        statistics_member=status is AutomaticStatus.PASSED,
        reason=_automatic_reason(
            status,
            (
                prepared_dataset.automation.reason
                if status is AutomaticStatus.PASSED
                else reason or prepared_dataset.automation.reason
            ),
            prepared_dataset.automation.role,
        ),
    )
    dataset = replace(
        prepared_dataset,
        automation=automation,
        parameter_settings=persisted_settings,
        last_valid_result=last_valid_result,
        checkpoint=_checkpoint_with_result_diagnostics(checkpoint, fit_result),
    )
    updated = _replace_dataset(working, index, dataset)
    return updated, DatasetFitResult(dataset.dataset_id, fit_result)


def _commit_automatic_failure(
    working: XrrProject,
    index: int,
    original: DatasetProject,
    error: BaseException,
) -> tuple[XrrProject, DatasetFitResult]:
    """Publish a row-level automatic failure without a stale candidate.

    The exception type and message become both warning and automation reason.
    """

    fit_result = _failure_result(original, error)
    message = f"{type(error).__name__}: {error}"
    automation = replace(
        working.datasets[index].automation,
        status=AutomaticStatus.FAILED,
        statistics_member=False,
        reason=message,
    )
    dataset = replace(
        working.datasets[index],
        automation=automation,
        last_valid_result=None,
        checkpoint=None,
    )
    updated = _replace_dataset(working, index, dataset)
    return updated, DatasetFitResult(dataset.dataset_id, fit_result)
