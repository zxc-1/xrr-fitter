"""Independent and joint project fit transactions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, replace

from xrr_fitter.model.analysis import ConfidenceClass, FitResult
from xrr_fitter.model.automation import AutomaticRole, AutomaticStatus, MeasurementPreset
from xrr_fitter.model.fitting import FitProgress
from xrr_fitter.model.operations import DatasetFitResult, ProjectFitResult
from xrr_fitter.model.parameters import ParameterSetting
from xrr_fitter.model.project import ScalePriorState, XrrProject
from xrr_fitter.model.structure import GradientLayerSpec, LayerSpec, PeriodicBlock
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
    result: object | None
    error: Exception | None
    events: tuple[tuple[str, object], ...]


@dataclass(frozen=True, slots=True)
class _AutomaticPreparation:
    index: int
    original: object
    fit_group_id: str
    group_size: int
    prepared: object | None = None
    error: Exception | None = None


def _material_signature(material) -> tuple[object, ...]:
    return (
        material.name,
        material.formula,
        material.sld_override_a2 is not None,
    )


def _component_signature(component) -> tuple[object, ...]:
    if isinstance(component, LayerSpec):
        return (component.name, *_material_signature(component.material))
    if isinstance(component, PeriodicBlock):
        return (
            component.name,
            tuple((layer.name, *_material_signature(layer.material)) for layer in component.layers),
            component.repeats,
            component.top_roughness_a is not None,
        )
    if isinstance(component, GradientLayerSpec):
        return (
            component.name,
            (component.upper_sld_a2.real, component.upper_sld_a2.imag),
            (component.lower_sld_a2.real, component.lower_sld_a2.imag),
            component.microslab_max_a,
        )
    raise TypeError(f"unsupported automatic structure component: {type(component).__name__}")


def _dataclass_values(value) -> tuple[object, ...]:
    return tuple(getattr(value, field) for field in value.__dataclass_fields__)


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def automatic_physical_signature(dataset, preset: MeasurementPreset) -> str:
    """Hash the behavior-changing automatic grouping contract."""
    if dataset.structure is None:
        raise ValueError(f"dataset {dataset.dataset_id} has no structure")
    if not isinstance(preset, MeasurementPreset):
        raise TypeError("preset must be MeasurementPreset")
    structure = dataset.structure
    payload = {
        "layers": tuple(_component_signature(component) for component in structure.components),
        "backing": _material_signature(structure.backing),
        "beam": _dataclass_values(dataset.beam),
        "import_angle_offset_deg": dataset.import_angle_offset_deg,
        "instrument": _dataclass_values(dataset.instrument),
        "preset": (
            _dataclass_values(preset.beam),
            _dataclass_values(preset.instrument),
            preset.import_angle_offset_deg,
        ),
        "structure_modes": tuple(
            type(component).__name__ for component in structure.components
        ),
    }
    return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()


def _automatic_group_id(import_batch_id: str, signature: str) -> str:
    payload = {"import_batch_id": import_batch_id, "physical_signature": signature}
    digest = hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()
    return f"automatic-{digest[:24]}"


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


AUTOMATIC_RUNNABLE_STATUSES = frozenset(
    {AutomaticStatus.PENDING, AutomaticStatus.REFINING, AutomaticStatus.REVIEW}
)


def _automatic_indices(
    project: XrrProject,
    import_batch_id: str | None,
) -> tuple[int, ...]:
    return tuple(
        index
        for index, dataset in enumerate(project.datasets)
        if dataset.automation.role is not AutomaticRole.MANUAL
        and dataset.automation.status in AUTOMATIC_RUNNABLE_STATUSES
        and (
            import_batch_id is None
            or dataset.automation.import_batch_id == import_batch_id
        )
    )


def _automatic_routes(
    project: XrrProject,
    indices: tuple[int, ...],
) -> tuple[XrrProject, dict[int, tuple[str, int]], dict[int, Exception]]:
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
    records: dict[str, object],
    seeds: dict[str, int],
    prepare_dataset,
) -> tuple[_AutomaticPreparation, ...]:
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


def _automatic_fit_parts(result: object) -> tuple[object, FitResult, bool, str | None]:
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
    best = fit_result.best_candidate
    if best is None or not best.valid:
        return AutomaticStatus.FAILED
    if refining:
        return AutomaticStatus.REFINING
    return AutomaticStatus.PASSED if passed else AutomaticStatus.REVIEW


def _automatic_reason(
    status: AutomaticStatus,
    reason: str | None,
) -> str | None:
    if status is AutomaticStatus.PASSED:
        return None
    if status is AutomaticStatus.REFINING:
        return reason
    if reason:
        return reason
    if status is AutomaticStatus.FAILED:
        return "no valid automatic candidate"
    return "automatic quality review required"


def _commit_automatic_result(
    working: XrrProject,
    row: _AutomaticPreparation,
    result: object,
    *,
    refining: bool,
) -> tuple[XrrProject, DatasetFitResult]:
    prepared, fit_result, passed, reason = _automatic_fit_parts(result)
    current = working.datasets[row.index]
    prepared_dataset = prepared.updated_dataset
    changed_settings = (
        row.prepared is not None
        and prepared_dataset.parameter_settings
        != row.prepared.updated_dataset.parameter_settings
    )
    checkpoint = prepared_dataset.checkpoint if changed_settings else current.checkpoint
    status = _automatic_status(fit_result, passed, refining)
    automation = replace(
        current.automation,
        status=status,
        statistics_member=status is AutomaticStatus.PASSED,
        reason=_automatic_reason(status, reason),
    )
    dataset = replace(
        prepared_dataset,
        automation=automation,
        parameter_settings=_winner_settings(
            prepared_dataset.parameter_settings,
            fit_result,
        ),
        last_valid_result=fit_result,
        checkpoint=_checkpoint_with_result_diagnostics(checkpoint, fit_result),
    )
    updated = _replace_dataset(working, row.index, dataset)
    return updated, DatasetFitResult(dataset.dataset_id, fit_result)


def _commit_automatic_failure(
    working: XrrProject,
    row: _AutomaticPreparation,
    error: BaseException,
) -> tuple[XrrProject, DatasetFitResult]:
    fit_result = _failure_result(row.original, error)
    message = f"{type(error).__name__}: {error}"
    automation = replace(
        working.datasets[row.index].automation,
        status=AutomaticStatus.FAILED,
        statistics_member=False,
        reason=message,
    )
    dataset = replace(
        working.datasets[row.index],
        automation=automation,
        last_valid_result=None,
        checkpoint=None,
    )
    updated = _replace_dataset(working, row.index, dataset)
    return updated, DatasetFitResult(dataset.dataset_id, fit_result)


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


def fit_automatic_transaction(
    project: XrrProject,
    import_batch_id: str | None,
    progress_callback: Callable[[FitProgress], None] | None,
    checkpoint_callback: Callable[[XrrProject], None] | None,
    cancelled: Callable[[], bool] | None,
    *,
    seed_branches,
    prepare_dataset,
    fit_dataset,
) -> ProjectFitResult:
    """Route and publish automatic prefits under one project CPU budget."""
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
    was_cancelled = False

    for row in rows:
        if row.error is None:
            continue
        working, published[row.index] = _commit_automatic_failure(
            working,
            row,
            row.error,
        )
        if checkpoint_callback is not None:
            checkpoint_callback(working)

    runnable = tuple(row for row in rows if row.prepared is not None)
    allocations = _worker_allocations(
        project.fit_config.local_workers,
        len(runnable),
    )
    tasks = tuple(
        lambda row=row, workers=workers: _buffered_dataset_fit(
            row.prepared,
            workers,
            fit_dataset,
            cancelled,
        )
        for row, workers in zip(runnable, allocations, strict=True)
    )

    def publish_prefit(position: int, outcome: _BufferedFit) -> None:
        nonlocal working, was_cancelled
        row = runnable[position]
        working = _replay_buffered_events(
            working,
            row.index,
            outcome.events,
            progress_callback,
            checkpoint_callback,
        )
        if outcome.error is not None:
            working, published[row.index] = _commit_automatic_failure(
                working,
                row,
                outcome.error,
            )
            was_cancelled = was_cancelled or _cancelled(outcome.error)
        else:
            assert outcome.result is not None
            working, published[row.index] = _commit_automatic_result(
                working,
                row,
                outcome.result,
                refining=row.group_size > 1,
            )
        if checkpoint_callback is not None:
            checkpoint_callback(working)

    if tasks:
        concurrency = min(len(tasks), project.fit_config.local_workers)
        with OrderedTaskRunner(concurrency) as runner:
            runner.run(tasks, completed=publish_prefit)

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
