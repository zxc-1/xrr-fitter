"""Independent and joint project fit transactions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from xrr_fitter.model.analysis import ConfidenceClass, FitResult
from xrr_fitter.model.fitting import FitProgress
from xrr_fitter.model.operations import DatasetFitResult, ProjectFitResult
from xrr_fitter.model.project import ScalePriorState, XrrProject
from xrr_fitter.services.parallel import OrderedTaskRunner
from xrr_fitter.services.projects import inspect_sources


@dataclass(frozen=True, slots=True)
class _IndependentPreparation:
    index: int
    original: object
    prepared: object | None = None
    error: Exception | None = None


@dataclass(frozen=True, slots=True)
class _BufferedFit:
    result: FitResult | None
    error: Exception | None
    events: tuple[tuple[str, object], ...]


def _clear_dataset(dataset):
    return replace(
        dataset,
        scale_prior=ScalePriorState(enabled=False),
        last_valid_result=None,
        checkpoint=None,
    )


def _replace_dataset(project: XrrProject, index: int, dataset) -> XrrProject:
    datasets = list(project.datasets)
    datasets[index] = dataset
    return replace(project, datasets=tuple(datasets))


def _clear_all(project: XrrProject) -> XrrProject:
    return replace(project, datasets=tuple(map(_clear_dataset, project.datasets)))


def _failure_result(dataset, error: BaseException | str) -> FitResult:
    message = error if isinstance(error, str) else f"{type(error).__name__}: {error}"
    return FitResult(
        parameter_definitions=(),
        candidates=(),
        best_index=None,
        confidence=ConfidenceClass.UNTRUSTED,
        warnings=(message,),
        child_seeds=(),
        stage_summaries=(),
        region_labels=(-1,) * len(dataset.fit_mask),
        region_weights=(0.0,) * len(dataset.fit_mask),
        uncertainty=None,
    )


def _source_records(validation) -> dict[str, object]:
    return {record.dataset_id: record for record in validation.datasets}


def _source_error(records: dict[str, object], dataset_id: str) -> ValueError | None:
    record = records.get(dataset_id)
    if record is None or record.status.value == "ok":
        return None
    return ValueError(f"source status {record.status.value}: {record.message}")


def _warnings(results: tuple[DatasetFitResult, ...]) -> tuple[str, ...]:
    return tuple(
        warning
        for item in results
        for warning in item.fit_result.warnings
    )


def _checkpoint_with_result_diagnostics(checkpoint, result: FitResult):
    if checkpoint is None:
        return None
    result_candidates = {
        candidate.candidate_id: candidate for candidate in result.candidates
    }
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


def _cancelled(error: BaseException) -> bool:
    return type(error).__name__ in {"SearchCancelled", "InterruptedError"}


def _prepare_independent_rows(
    project: XrrProject,
    records: dict[str, object],
    seeds: dict[str, int],
    prepare_dataset,
) -> tuple[_IndependentPreparation, ...]:
    working = project
    rows = []
    for index, original in enumerate(project.datasets):
        error = _source_error(records, original.dataset_id)
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
    if count == 0:
        return ()
    if count >= total_workers:
        return (1,) * count
    base, remainder = divmod(total_workers, count)
    return tuple(base + int(index < remainder) for index in range(count))


def _buffered_dataset_fit(
    prepared,
    local_workers: int,
    fit_dataset,
    cancelled,
) -> _BufferedFit:
    events: list[tuple[str, object]] = []
    try:
        if cancelled is not None and cancelled():
            raise InterruptedError("cancelled")
        result = fit_dataset(
            prepared,
            progress=lambda value: events.append(("progress", value)),
            cancelled=cancelled,
            checkpoint=lambda value: events.append(("checkpoint", value)),
            local_workers=local_workers,
        )
    except Exception as error:
        return _BufferedFit(None, error, tuple(events))
    return _BufferedFit(result, None, tuple(events))


def _run_independent_rows(
    rows: tuple[_IndependentPreparation, ...],
    total_workers: int,
    fit_dataset,
    cancelled,
) -> dict[int, _BufferedFit]:
    runnable = tuple(row for row in rows if row.prepared is not None)
    allocations = _worker_allocations(total_workers, len(runnable))
    tasks = tuple(
        lambda row=row, workers=workers: _buffered_dataset_fit(
            row.prepared,
            workers,
            fit_dataset,
            cancelled,
        )
        for row, workers in zip(runnable, allocations, strict=True)
    )
    if not tasks:
        return {}
    concurrency = min(len(tasks), total_workers)
    with OrderedTaskRunner(concurrency) as runner:
        buffered = runner.run(tasks)
    return {
        row.index: result
        for row, result in zip(runnable, buffered, strict=True)
    }


def _replay_buffered_events(
    working: XrrProject,
    index: int,
    events: tuple[tuple[str, object], ...],
    progress: Callable[[FitProgress], None] | None,
    checkpoint_callback: Callable[[XrrProject], None] | None,
) -> XrrProject:
    for kind, value in events:
        if kind == "progress":
            if progress is not None:
                progress(value)
            continue
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


def _independent_fit(
    project: XrrProject,
    validation,
    progress: Callable[[FitProgress], None] | None,
    checkpoint_callback: Callable[[XrrProject], None] | None,
    cancelled: Callable[[], bool] | None,
    seed_branches,
    prepare_dataset,
    fit_dataset,
) -> ProjectFitResult:
    seeds, _joint, _mcmc = seed_branches(project)
    records = _source_records(validation)
    rows = _prepare_independent_rows(project, records, seeds, prepare_dataset)
    buffered = _run_independent_rows(
        rows,
        project.fit_config.local_workers,
        fit_dataset,
        cancelled,
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
            working = _replace_dataset(working, index, row.prepared.updated_dataset)
            outcome = buffered[index]
            working = _replay_buffered_events(
                working,
                index,
                outcome.events,
                progress,
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
    cleared = _clear_all(project)
    values = tuple(
        DatasetFitResult(dataset.dataset_id, _failure_result(dataset, error))
        for dataset in cleared.datasets
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
    validation,
    progress: Callable[[FitProgress], None] | None,
    checkpoint_callback: Callable[[XrrProject], None] | None,
    cancelled: Callable[[], bool] | None,
    seed_branches,
    prepare_dataset,
    fit_joint,
) -> ProjectFitResult:
    records = _source_records(validation)
    source_error = next(
        (
            error
            for dataset in project.datasets
            if (error := _source_error(records, dataset.dataset_id)) is not None
        ),
        None,
    )
    if source_error is not None:
        return _joint_failure(project, source_error)
    _independent, seed, _mcmc = seed_branches(project)
    try:
        prepared = tuple(
            prepare_dataset(project, dataset.dataset_id, seed)
            for dataset in project.datasets
        )
        working = replace(
            project,
            datasets=tuple(item.updated_dataset for item in prepared),
        )

        def publish_checkpoints(values):
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


def fit_project_transaction(
    project: XrrProject,
    progress_callback: Callable[[FitProgress], None] | None = None,
    checkpoint_callback: Callable[[XrrProject], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    *,
    seed_branches,
    prepare_dataset,
    fit_dataset,
    fit_joint,
) -> ProjectFitResult:
    """Dispatch exactly the persisted independent or joint batch mode."""
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
