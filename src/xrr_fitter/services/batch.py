"""Independent and joint project fit transactions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from xrr_fitter.model.analysis import ConfidenceClass, FitResult
from xrr_fitter.model.fitting import FitProgress
from xrr_fitter.model.operations import DatasetFitResult, ProjectFitResult
from xrr_fitter.model.project import ScalePriorState, XrrProject
from xrr_fitter.services.projects import inspect_sources


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
    working = project
    results: list[DatasetFitResult] = []
    was_cancelled = False
    for index, original in enumerate(project.datasets):
        dataset_id = original.dataset_id
        source_error = _source_error(records, dataset_id)
        if source_error is not None:
            working = _replace_dataset(working, index, _clear_dataset(working.datasets[index]))
            results.append(DatasetFitResult(dataset_id, _failure_result(original, source_error)))
            continue
        try:
            prepared = prepare_dataset(working, dataset_id, seeds[dataset_id])
        except Exception as error:
            working = _replace_dataset(working, index, _clear_dataset(working.datasets[index]))
            results.append(DatasetFitResult(dataset_id, _failure_result(original, error)))
            continue
        working = _replace_dataset(working, index, prepared.updated_dataset)

        def publish_checkpoint(value, *, dataset_index=index):
            nonlocal working
            dataset = replace(working.datasets[dataset_index], checkpoint=value)
            working = _replace_dataset(working, dataset_index, dataset)
            if checkpoint_callback is not None:
                checkpoint_callback(working)

        try:
            fit_result = fit_dataset(
                prepared,
                progress=progress,
                cancelled=cancelled,
                checkpoint=publish_checkpoint,
            )
        except Exception as error:
            fit_result = _failure_result(original, error)
            was_cancelled = _cancelled(error)
        else:
            dataset = working.datasets[index]
            dataset = replace(
                dataset,
                last_valid_result=fit_result,
                checkpoint=_checkpoint_with_result_diagnostics(
                    dataset.checkpoint,
                    fit_result,
                ),
            )
            working = _replace_dataset(working, index, dataset)
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
