"""Independent, joint, and automatic project fit transactions.

The module coordinates project transactions around numerical service calls;
dataset value updates belong to ``batch_publication``.
Preparation failures remain dataset-scoped for independent and automatic work,
while expert joint failures invalidate the complete joint result graph.

Project checkpoints and terminal results are committed in stable dataset or
fit-group order. Automatic results become statistics members only after their
final quality decision is ``PASSED``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from functools import partial
from threading import Lock
from typing import Protocol, cast

from xrr_fitter.model.analysis import FitResult
from xrr_fitter.model.automation import AutomaticRole, AutomaticStatus
from xrr_fitter.model.fitting import FitCheckpoint, FitProgress
from xrr_fitter.model.operations import DatasetFitResult, ProjectFitResult
from xrr_fitter.model.parameters import SharingRule
from xrr_fitter.model.project import DatasetProject, DatasetSourceValidation, ProjectValidation, XrrProject
from xrr_fitter.services.batch_publication import (
    _automatic_fit_parts,
    _checkpoint_with_result_diagnostics,
    _clear_all,
    _clear_dataset,
    _commit_automatic_failure,
    _commit_automatic_result,
    _commit_success,
    _failure_result,
    _replace_dataset,
    _replay_checkpoints,
    _warnings,
)
from xrr_fitter.services.batch_routing import automatic_group_id as _automatic_group_id
from xrr_fitter.services.batch_routing import automatic_physical_signature
from xrr_fitter.services.datasets import ServiceSeedBranches
from xrr_fitter.services.fitting_phases.common import AutomaticPreparedResult, PreparedDatasetFit
from xrr_fitter.services.parallel import OrderedTaskRunner
from xrr_fitter.services.projects import inspect_sources

_SeedBranches = Callable[[XrrProject], ServiceSeedBranches]
_PrepareDataset = Callable[[XrrProject, str, int], PreparedDatasetFit]


class _DatasetFit[Result](Protocol):
    """The keyword callback contract shared by expert and automatic searches."""

    def __call__(
        self,
        prepared: PreparedDatasetFit,
        /,
        *,
        progress: Callable[[FitProgress], None] | None,
        cancelled: Callable[[], bool] | None,
        checkpoint: Callable[[FitCheckpoint | None], None] | None,
        local_workers: int,
    ) -> Result: ...


class _JointFit(Protocol):
    def __call__(
        self,
        prepared: tuple[PreparedDatasetFit, ...],
        sharing_rules: tuple[SharingRule, ...],
        /,
        *,
        progress: Callable[[FitProgress], None] | None,
        cancelled: Callable[[], bool] | None,
        checkpoint: Callable[[tuple[FitCheckpoint, ...]], None] | None,
    ) -> tuple[FitResult, ...]: ...


class _AutomaticJointFit(Protocol):
    def __call__(
        self,
        prepared: tuple[PreparedDatasetFit, ...],
        prefits: tuple[AutomaticPreparedResult, ...],
        fit_group_id: str,
        /,
        *,
        progress: Callable[[FitProgress], None] | None,
        cancelled: Callable[[], bool] | None,
        checkpoint: Callable[[tuple[FitCheckpoint, ...]], None] | None,
    ) -> tuple[AutomaticPreparedResult, ...]: ...


@dataclass(frozen=True, slots=True)
class _IndependentPreparation:
    """Capture one independent row before numerical execution.

    Exactly one of ``prepared`` and ``error`` is populated after preparation.
    """

    index: int
    original: DatasetProject
    prepared: PreparedDatasetFit | None = None
    error: Exception | None = None


@dataclass(frozen=True, slots=True)
class _BufferedFit[Result]:
    """Retain a worker outcome until ordered project publication.

    Checkpoints are buffered because they mutate the immutable project value.
    """

    result: Result | None
    error: Exception | None
    checkpoints: tuple[FitCheckpoint | None, ...]


@dataclass(frozen=True, slots=True)
class _AutomaticPreparation:
    """Bind one automatic row to its deterministic routing decision.

    Group identity survives preparation failure so publication stays auditable.
    """

    index: int
    original: DatasetProject
    fit_group_id: str
    group_size: int
    prepared: PreparedDatasetFit | None = None
    error: Exception | None = None


def _source_records(validation: ProjectValidation) -> dict[str, DatasetSourceValidation]:
    """Index source validation records by persisted dataset identity.

    Transaction preparation consumes this snapshot without re-reading sources.
    """

    return {record.dataset_id: record for record in validation.datasets}


def _source_error(records: dict[str, DatasetSourceValidation], dataset_id: str) -> ValueError | None:
    """Translate a non-current source record into a preparation error.

    Missing records are treated as unavailable only when a record says so.
    """

    record = records.get(dataset_id)
    if record is None or record.status.value == "ok":
        return None
    return ValueError(f"source status {record.status.value}: {record.message}")


def _cancelled(error: BaseException) -> bool:
    """Recognize cooperative cancellation across exception subclasses.

    Solvers may mark domain-specific exceptions without sharing their type here.
    """

    return isinstance(error, InterruptedError) or bool(getattr(type(error), "_xrr_cooperative_cancellation", False))


def _prepare_independent_rows(
    project: XrrProject,
    records: dict[str, DatasetSourceValidation],
    seeds: dict[str, int],
    prepare_dataset: _PrepareDataset,
) -> tuple[_IndependentPreparation, ...]:
    """Prepare independent rows while retaining every row-level failure.

    Successful preparation state feeds later rows through the working project.
    """

    working = project
    rows = []
    for index, original in enumerate(project.datasets):
        error: Exception | None = _source_error(records, original.dataset_id)
        prepared = None
        if error is None:
            try:
                prepared = prepare_dataset(
                    working,
                    original.dataset_id,
                    seeds[original.dataset_id],
                )
            except Exception as caught:
                error = caught
            else:
                working = _replace_dataset(working, index, prepared.updated_dataset)
        rows.append(_IndependentPreparation(index, original, prepared, error))
    return tuple(rows)


def _worker_allocations(total_workers: int, count: int) -> tuple[int, ...]:
    """Divide one total worker budget across concurrent dataset tasks.

    Each runnable task receives at least one worker and remainders go first.
    """

    if count == 0:
        return ()
    if count >= total_workers:
        return (1,) * count
    base, remainder = divmod(total_workers, count)
    return tuple(base + int(index < remainder) for index in range(count))


def _dataset_fit[Result](
    prepared: PreparedDatasetFit,
    local_workers: int,
    fit_dataset: _DatasetFit[Result],
    cancelled: Callable[[], bool] | None,
    progress: Callable[[FitProgress], None] | None,
) -> _BufferedFit[Result]:
    """Publish progress as it happens while deferring checkpoint commits.

    Progress is a pure notification, so it reaches the caller immediately even
    when several datasets run concurrently; each value carries its own dataset
    id. Checkpoints are withheld because replaying one mutates the accumulated
    immutable project, which must stay serialized in project order.
    """
    checkpoints: list[FitCheckpoint | None] = []
    try:
        if cancelled is not None and cancelled():
            raise InterruptedError("cancelled")
        result = fit_dataset(
            prepared,
            progress=progress,
            cancelled=cancelled,
            checkpoint=checkpoints.append,
            local_workers=local_workers,
        )
    except Exception as error:
        return _BufferedFit(None, error, tuple(checkpoints))
    return _BufferedFit(result, None, tuple(checkpoints))


def _serialized_progress(
    progress: Callable[[FitProgress], None] | None,
) -> Callable[[FitProgress], None] | None:
    """Serialize concurrent progress so an arbitrary callback sees one value."""
    if progress is None:
        return None
    lock = Lock()

    def publish(value: FitProgress) -> None:
        with lock:
            progress(value)

    return publish


def _run_independent_rows(
    rows: tuple[_IndependentPreparation, ...],
    total_workers: int,
    fit_dataset: _DatasetFit[FitResult],
    cancelled: Callable[[], bool] | None,
    progress: Callable[[FitProgress], None] | None,
) -> dict[int, _BufferedFit[FitResult]]:
    runnable = tuple(row for row in rows if row.prepared is not None)
    allocations = _worker_allocations(total_workers, len(runnable))
    published = _serialized_progress(progress)
    tasks: tuple[Callable[[], _BufferedFit[FitResult]], ...] = tuple(
        partial(
            _dataset_fit,
            cast(PreparedDatasetFit, row.prepared),
            workers,
            fit_dataset,
            cancelled,
            published,
        )
        for row, workers in zip(runnable, allocations, strict=True)
    )
    if not tasks:
        return {}
    concurrency = min(len(tasks), total_workers)
    with OrderedTaskRunner(concurrency) as runner:
        buffered = runner.run(tasks)
    return {row.index: result for row, result in zip(runnable, buffered, strict=True)}


AUTOMATIC_RUNNABLE_STATUSES = frozenset({AutomaticStatus.PENDING, AutomaticStatus.REFINING, AutomaticStatus.REVIEW})


def _automatic_indices(
    project: XrrProject,
    import_batch_id: str | None,
) -> tuple[int, ...]:
    """Select runnable automatic rows in persisted project order.

    An optional import batch limits retries without affecting manual datasets.
    """

    return tuple(
        index
        for index, dataset in enumerate(project.datasets)
        if dataset.automation.role is not AutomaticRole.MANUAL
        and dataset.automation.status in AUTOMATIC_RUNNABLE_STATUSES
        and (import_batch_id is None or dataset.automation.import_batch_id == import_batch_id)
    )


def _automatic_routes(
    project: XrrProject,
    indices: tuple[int, ...],
) -> tuple[XrrProject, dict[int, tuple[str, int]], dict[int, Exception]]:
    """Assign automatic rows to deterministic single or joint routes.

    Invalid rows receive stable fallback signatures so failures stay ordered.
    """

    preset = project.measurement_preset
    if preset is None:
        raise ValueError("automatic fit requires a measurement preset")
    grouped: dict[tuple[str, str], list[int]] = {}
    errors: dict[int, Exception] = {}
    for index in indices:
        dataset = project.datasets[index]
        batch_id = dataset.automation.import_batch_id
        if batch_id is None:
            errors[index] = ValueError("automatic dataset requires import_batch_id")
            batch_id = f"invalid-{dataset.dataset_id}"
        try:
            signature = automatic_physical_signature(dataset, preset)
        except Exception as error:
            errors[index] = error
            signature = hashlib.sha256(dataset.dataset_id.encode("utf-8")).hexdigest()
        grouped.setdefault((batch_id, signature), []).append(index)

    routes: dict[int, tuple[str, int]] = {}
    datasets = list(project.datasets)
    for (batch_id, signature), members in grouped.items():
        fit_group_id = _automatic_group_id(batch_id, signature)
        role = AutomaticRole.SINGLE if len(members) == 1 else AutomaticRole.JOINT
        for index in members:
            routes[index] = (fit_group_id, len(members))
            automation = replace(
                datasets[index].automation,
                fit_group_id=fit_group_id,
                role=role,
                status=AutomaticStatus.REFINING,
                statistics_member=False,
                reason=None,
            )
            datasets[index] = replace(datasets[index], automation=automation)
    return replace(project, datasets=tuple(datasets)), routes, errors


def _automatic_preparations(
    working: XrrProject,
    indices: tuple[int, ...],
    routes: dict[int, tuple[str, int]],
    route_errors: dict[int, Exception],
    records: dict[str, DatasetSourceValidation],
    seeds: dict[str, int],
    prepare_dataset: _PrepareDataset,
) -> tuple[_AutomaticPreparation, ...]:
    """Compile each routed automatic row and preserve preparation errors.

    Routing state is already published before compilation begins.
    """

    rows = []
    for index in indices:
        original = working.datasets[index]
        fit_group_id, group_size = routes[index]
        error = route_errors.get(index) or _source_error(records, original.dataset_id)
        prepared = None
        if error is None:
            try:
                prepared = prepare_dataset(
                    working,
                    original.dataset_id,
                    seeds[original.dataset_id],
                )
            except Exception as caught:
                error = caught
        rows.append(
            _AutomaticPreparation(
                index,
                original,
                fit_group_id,
                group_size,
                prepared,
                error,
            )
        )
    return tuple(rows)


def _independent_fit(
    project: XrrProject,
    validation: ProjectValidation,
    progress: Callable[[FitProgress], None] | None,
    checkpoint_callback: Callable[[XrrProject], None] | None,
    cancelled: Callable[[], bool] | None,
    seed_branches: _SeedBranches,
    prepare_dataset: _PrepareDataset,
    fit_dataset: _DatasetFit[FitResult],
) -> ProjectFitResult:
    seeds, _joint, _mcmc = seed_branches(project)
    records = _source_records(validation)
    rows = _prepare_independent_rows(project, records, seeds, prepare_dataset)
    buffered = _run_independent_rows(
        rows,
        project.fit_config.local_workers,
        fit_dataset,
        cancelled,
        progress,
    )
    working = project
    results: list[DatasetFitResult] = []
    was_cancelled = False
    for row in rows:
        index, original = row.index, row.original
        dataset_id = original.dataset_id
        if row.error is not None:
            working = _replace_dataset(working, index, _clear_dataset(working.datasets[index]))
            fit_result = _failure_result(original, row.error)
        else:
            prepared = cast(PreparedDatasetFit, row.prepared)
            working = _replace_dataset(working, index, prepared.updated_dataset)
            outcome = buffered[index]
            working = _replay_checkpoints(
                working,
                index,
                outcome.checkpoints,
                checkpoint_callback,
            )
            if outcome.error is not None:
                fit_result = _failure_result(original, outcome.error)
                was_cancelled = _cancelled(outcome.error)
            else:
                assert outcome.result is not None
                fit_result = outcome.result
                working = _commit_success(working, index, fit_result)
        results.append(DatasetFitResult(dataset_id, fit_result))
        if was_cancelled or (cancelled is not None and cancelled()):
            was_cancelled = True
            break
    values = tuple(results)
    return ProjectFitResult(
        "independent",
        values,
        _warnings(values),
        working,
        was_cancelled,
    )


def _joint_failure(project: XrrProject, error: BaseException) -> ProjectFitResult:
    """Invalidate an expert joint graph after any member failure.

    Every member receives the same terminal error and cancellation classification.
    """

    cleared = _clear_all(project)
    values = tuple(
        DatasetFitResult(dataset.dataset_id, _failure_result(dataset, error)) for dataset in cleared.datasets
    )
    return ProjectFitResult(
        "joint",
        values,
        _warnings(values),
        cleared,
        _cancelled(error),
    )


def _joint_fit(
    project: XrrProject,
    validation: ProjectValidation,
    progress: Callable[[FitProgress], None] | None,
    checkpoint_callback: Callable[[XrrProject], None] | None,
    cancelled: Callable[[], bool] | None,
    seed_branches: _SeedBranches,
    prepare_dataset: _PrepareDataset,
    fit_joint: _JointFit,
) -> ProjectFitResult:
    """Execute one all-or-nothing expert joint transaction.

    Checkpoint batches and final results must remain aligned with project order.
    """

    records = _source_records(validation)
    source_error = next(
        (error for dataset in project.datasets if (error := _source_error(records, dataset.dataset_id)) is not None),
        None,
    )
    if source_error is not None:
        return _joint_failure(project, source_error)
    _independent, seed, _mcmc = seed_branches(project)
    try:
        prepared = tuple(prepare_dataset(project, dataset.dataset_id, seed) for dataset in project.datasets)
        working = replace(
            project,
            datasets=tuple(item.updated_dataset for item in prepared),
        )

        def publish_checkpoints(values: Iterable[FitCheckpoint | None]) -> None:
            nonlocal working
            checkpoints = tuple(values)
            if len(checkpoints) != len(working.datasets):
                raise ValueError("joint checkpoint batch size mismatch")
            datasets = tuple(
                replace(dataset, checkpoint=checkpoint)
                for dataset, checkpoint in zip(working.datasets, checkpoints, strict=True)
            )
            working = replace(working, datasets=datasets)
            if checkpoint_callback is not None:
                checkpoint_callback(working)

        fit_results = fit_joint(
            prepared,
            project.sharing_rules,
            progress=progress,
            cancelled=cancelled,
            checkpoint=publish_checkpoints,
        )
        if len(fit_results) != len(working.datasets):
            raise ValueError("joint result batch size mismatch")
        datasets = tuple(
            replace(
                dataset,
                last_valid_result=result,
                checkpoint=_checkpoint_with_result_diagnostics(
                    dataset.checkpoint,
                    result,
                ),
            )
            for dataset, result in zip(working.datasets, fit_results, strict=True)
        )
        working = replace(working, datasets=datasets)
    except Exception as error:
        return _joint_failure(project, error)
    values = tuple(
        DatasetFitResult(dataset.dataset_id, result)
        for dataset, result in zip(working.datasets, fit_results, strict=True)
    )
    return ProjectFitResult("joint", values, _warnings(values), working)


def _publish_automatic_preparation_failures(
    working: XrrProject,
    rows: tuple[_AutomaticPreparation, ...],
    published: dict[int, DatasetFitResult],
    checkpoint_callback: Callable[[XrrProject], None] | None,
) -> XrrProject:
    """Commit automatic preparation failures before numerical workers start.

    Early publication exposes deterministic row failures through checkpoints.
    """

    for row in rows:
        if row.error is None:
            continue
        working, published[row.index] = _commit_automatic_failure(
            working,
            row.index,
            row.original,
            row.error,
        )
        if checkpoint_callback is not None:
            checkpoint_callback(working)
    return working


def _run_automatic_prefits(
    working: XrrProject,
    rows: tuple[_AutomaticPreparation, ...],
    total_workers: int,
    published: dict[int, DatasetFitResult],
    *,
    fit_dataset: _DatasetFit[AutomaticPreparedResult],
    progress_callback: Callable[[FitProgress], None] | None,
    checkpoint_callback: Callable[[XrrProject], None] | None,
    cancelled: Callable[[], bool] | None,
) -> tuple[
    XrrProject,
    tuple[_AutomaticPreparation, ...],
    dict[int, AutomaticPreparedResult],
    bool,
    Callable[[FitProgress], None] | None,
]:
    runnable = tuple(row for row in rows if row.prepared is not None)
    allocations = _worker_allocations(total_workers, len(runnable))
    published_progress = _serialized_progress(progress_callback)
    tasks: tuple[Callable[[], _BufferedFit[AutomaticPreparedResult]], ...] = tuple(
        partial(
            _dataset_fit,
            cast(PreparedDatasetFit, row.prepared),
            workers,
            fit_dataset,
            cancelled,
            published_progress,
        )
        for row, workers in zip(runnable, allocations, strict=True)
    )
    prefit_results: dict[int, AutomaticPreparedResult] = {}
    was_cancelled = False

    def publish_prefit(position: int, outcome: _BufferedFit[AutomaticPreparedResult]) -> None:
        nonlocal working, was_cancelled
        row = runnable[position]
        working = _replay_checkpoints(
            working,
            row.index,
            outcome.checkpoints,
            checkpoint_callback,
        )
        if outcome.error is None:
            assert outcome.result is not None
            prefit_results[row.index] = outcome.result
            working, published[row.index] = _commit_automatic_result(
                working,
                row.index,
                row.prepared,
                outcome.result,
                refining=row.group_size > 1,
            )
        else:
            working, published[row.index] = _commit_automatic_failure(
                working,
                row.index,
                row.original,
                outcome.error,
            )
            was_cancelled = was_cancelled or _cancelled(outcome.error)
        if checkpoint_callback is not None:
            checkpoint_callback(working)

    if tasks:
        with OrderedTaskRunner(min(len(tasks), total_workers)) as runner:
            runner.run(tasks, completed=publish_prefit)
    return (
        working,
        runnable,
        prefit_results,
        was_cancelled,
        published_progress,
    )


def _automatic_joint_groups(
    runnable: tuple[_AutomaticPreparation, ...],
    prefit_results: dict[int, AutomaticPreparedResult],
) -> dict[str, tuple[_AutomaticPreparation, ...]]:
    """Collect successful prefits that still require joint refinement.

    Singleton routes are already final and do not enter this mapping.
    """

    grouped: dict[str, list[_AutomaticPreparation]] = {}
    for row in runnable:
        if row.group_size > 1 and row.index in prefit_results:
            grouped.setdefault(row.fit_group_id, []).append(row)
    return {key: tuple(members) for key, members in grouped.items()}


def _commit_incomplete_automatic_group(
    working: XrrProject,
    row: _AutomaticPreparation,
    prefit: AutomaticPreparedResult,
) -> tuple[XrrProject, DatasetFitResult]:
    """Demote a lone surviving group member to an auditable review result.

    A joint decision requires at least two qualified prepared datasets.
    """

    review = replace(
        prefit,
        passed=False,
        reason="insufficient qualified points for joint refinement",
    )
    return _commit_automatic_result(
        working,
        row.index,
        row.prepared,
        review,
        refining=False,
    )


def _automatic_joint_checkpoint_project(
    working: XrrProject,
    member_rows: tuple[_AutomaticPreparation, ...],
    values: Iterable[FitCheckpoint | None],
) -> XrrProject:
    """Apply one aligned joint checkpoint batch to its member rows.

    Group order is validated before any immutable project replacement occurs.
    """

    checkpoints = tuple(values)
    if len(checkpoints) != len(member_rows):
        raise ValueError("automatic joint checkpoint batch size mismatch")
    datasets = list(working.datasets)
    for row, checkpoint in zip(member_rows, checkpoints, strict=True):
        datasets[row.index] = replace(datasets[row.index], checkpoint=checkpoint)
    return replace(working, datasets=tuple(datasets))


def _commit_automatic_joint_success(
    working: XrrProject,
    member_rows: tuple[_AutomaticPreparation, ...],
    joint_results: tuple[AutomaticPreparedResult, ...],
) -> tuple[XrrProject, dict[int, DatasetFitResult]]:
    """Commit an aligned automatic joint result batch by original row index.

    Returned prepared identities guard against accidental cross-row publication.
    """

    if len(joint_results) != len(member_rows):
        raise ValueError("automatic joint result batch size mismatch")
    published = {}
    for row, result in zip(member_rows, joint_results, strict=True):
        returned_prepared = _automatic_fit_parts(result)[0]
        if returned_prepared.dataset_id != row.original.dataset_id:
            raise ValueError("automatic joint result dataset order mismatch")
        working, published[row.index] = _commit_automatic_result(
            working,
            row.index,
            row.prepared,
            result,
            refining=False,
        )
    return working, published


def _fit_automatic_joint_transaction_group(
    working: XrrProject,
    fit_group_id: str,
    member_rows: tuple[_AutomaticPreparation, ...],
    prefit_results: dict[int, AutomaticPreparedResult],
    *,
    fit_joint: _AutomaticJointFit,
    progress: Callable[[FitProgress], None] | None,
    checkpoint_callback: Callable[[XrrProject], None] | None,
    cancelled: Callable[[], bool] | None,
) -> tuple[XrrProject, dict[int, DatasetFitResult], bool]:
    """Refine and atomically publish one automatic fit group.

    Cancellation restores the pre-group baseline; ordinary failures publish a
    terminal failure for every member without affecting other groups.
    """

    if len(member_rows) == 1:
        row = member_rows[0]
        working, result = _commit_incomplete_automatic_group(
            working,
            row,
            prefit_results[row.index],
        )
        return working, {row.index: result}, False
    member_prefits = tuple(prefit_results[row.index] for row in member_rows)
    member_prepared = tuple(_automatic_fit_parts(prefit)[0] for prefit in member_prefits)
    group_baseline = working

    def publish_checkpoints(values: Iterable[FitCheckpoint | None]) -> None:
        nonlocal working
        working = _automatic_joint_checkpoint_project(
            working,
            member_rows,
            values,
        )
        if checkpoint_callback is not None:
            checkpoint_callback(working)

    try:
        joint_results = tuple(
            fit_joint(
                member_prepared,
                member_prefits,
                fit_group_id,
                progress=progress,
                cancelled=cancelled,
                checkpoint=publish_checkpoints,
            )
        )
        working, published = _commit_automatic_joint_success(
            working,
            member_rows,
            joint_results,
        )
    except Exception as error:
        if _cancelled(error):
            return group_baseline, {}, True
        published = {}
        for row in member_rows:
            working, published[row.index] = _commit_automatic_failure(
                working,
                row.index,
                row.original,
                error,
            )
        was_cancelled = _cancelled(error)
    else:
        was_cancelled = False
    return working, published, was_cancelled


def fit_automatic_transaction(
    project: XrrProject,
    import_batch_id: str | None,
    progress_callback: Callable[[FitProgress], None] | None,
    checkpoint_callback: Callable[[XrrProject], None] | None,
    cancelled: Callable[[], bool] | None,
    *,
    seed_branches: _SeedBranches,
    prepare_dataset: _PrepareDataset,
    fit_dataset: _DatasetFit[AutomaticPreparedResult],
    fit_joint: _AutomaticJointFit,
) -> ProjectFitResult:
    """Route automatic prefits, joint groups, and isolated final results.

    Preparation and prefit outcomes publish incrementally, while each joint
    group remains atomic with respect to cancellation.
    """
    indices = _automatic_indices(project, import_batch_id)
    if not indices:
        raise ValueError("no runnable automatic datasets")
    working, routes, route_errors = _automatic_routes(project, indices)
    validation = inspect_sources(working)
    records = _source_records(validation)
    seeds, _joint_seed, _mcmc_seed = seed_branches(working)
    rows = _automatic_preparations(
        working,
        indices,
        routes,
        route_errors,
        records,
        seeds,
        prepare_dataset,
    )
    published: dict[int, DatasetFitResult] = {}
    working = _publish_automatic_preparation_failures(
        working,
        rows,
        published,
        checkpoint_callback,
    )
    (
        working,
        runnable,
        prefit_results,
        was_cancelled,
        published_progress,
    ) = _run_automatic_prefits(
        working,
        rows,
        project.fit_config.local_workers,
        published,
        fit_dataset=fit_dataset,
        progress_callback=progress_callback,
        checkpoint_callback=checkpoint_callback,
        cancelled=cancelled,
    )
    grouped = _automatic_joint_groups(runnable, prefit_results)
    for fit_group_id, member_rows in grouped.items():
        if was_cancelled or (cancelled is not None and cancelled()):
            was_cancelled = True
            break
        working, group_results, group_cancelled = _fit_automatic_joint_transaction_group(
            working,
            fit_group_id,
            member_rows,
            prefit_results,
            fit_joint=fit_joint,
            progress=published_progress,
            checkpoint_callback=checkpoint_callback,
            cancelled=cancelled,
        )
        published.update(group_results)
        was_cancelled = was_cancelled or group_cancelled
        if checkpoint_callback is not None:
            checkpoint_callback(working)

    values = tuple(published[index] for index in indices)
    return ProjectFitResult(
        "automatic",
        values,
        _warnings(values),
        working,
        was_cancelled,
    )


def fit_project_transaction(
    project: XrrProject,
    progress_callback: Callable[[FitProgress], None] | None = None,
    checkpoint_callback: Callable[[XrrProject], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    *,
    seed_branches: _SeedBranches,
    prepare_dataset: _PrepareDataset,
    fit_dataset: _DatasetFit[FitResult],
    fit_joint: _JointFit,
) -> ProjectFitResult:
    """Dispatch exactly the persisted independent or joint batch mode.

    The persisted mode is authoritative; this boundary never infers a fallback.
    """
    if not project.datasets:
        raise ValueError("project has no datasets")
    validation = inspect_sources(project)
    if project.batch_mode == "independent":
        return _independent_fit(
            project,
            validation,
            progress_callback,
            checkpoint_callback,
            cancelled,
            seed_branches,
            prepare_dataset,
            fit_dataset,
        )
    if project.batch_mode == "joint":
        return _joint_fit(
            project,
            validation,
            progress_callback,
            checkpoint_callback,
            cancelled,
            seed_branches,
            prepare_dataset,
            fit_joint,
        )
    raise ValueError("fit batch mode must be independent or joint")
