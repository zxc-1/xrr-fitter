"""Independent, joint, and automatic project fit transactions.

The module owns immutable project publication around numerical service calls.
Preparation failures remain dataset-scoped for independent and automatic work,
while expert joint failures invalidate the complete joint result graph.

Project checkpoints and terminal results are committed in stable dataset or
fit-group order. Automatic results become statistics members only after their
final quality decision is ``PASSED``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from threading import Lock

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
    """Capture one independent row before numerical execution.

    Exactly one of ``prepared`` and ``error`` is populated after preparation.
    """

    index: int
    original: object
    prepared: object | None = None
    error: Exception | None = None


@dataclass(frozen=True, slots=True)
class _BufferedFit:
    """Retain a worker outcome until ordered project publication.

    Checkpoints are buffered because they mutate the immutable project value.
    """

    result: object | None
    error: Exception | None
    checkpoints: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class _AutomaticPreparation:
    """Bind one automatic row to its deterministic routing decision.

    Group identity survives preparation failure so publication stays auditable.
    """

    index: int
    original: object
    fit_group_id: str
    group_size: int
    prepared: object | None = None
    error: Exception | None = None


def _material_signature(material) -> tuple[object, ...]:
    """Return the material identity relevant to automatic sharing.

    Numerical density values remain fit parameters rather than group identity.
    """

    return (
        material.name,
        material.formula,
        material.sld_override_a2 is not None,
    )


def _component_signature(component) -> tuple[object, ...]:
    """Describe one structure component for physical-route hashing.

    The signature records topology and material identity, not fitted values.
    """

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
    """Read declared dataclass fields in their stable definition order.

    This avoids representation-dependent hashes for beam and instrument state.
    """

    return tuple(getattr(value, field) for field in value.__dataclass_fields__)


def _canonical_json(value: object) -> str:
    """Serialize a signature payload with deterministic key ordering.

    ASCII output makes the subsequent digest independent of locale settings.
    """

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def automatic_physical_signature(dataset, preset: MeasurementPreset) -> str:
    """Hash the behavior-changing automatic grouping contract.

    Only datasets with the same structure, backing, beam, and instrument route
    into one joint automatic fit.
    """
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
    """Derive a stable group identifier from batch and physical identity.

    Including the import batch prevents unrelated acquisitions from coalescing.
    """

    payload = {"import_batch_id": import_batch_id, "physical_signature": signature}
    digest = hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()
    return f"automatic-{digest[:24]}"


def _clear_dataset(dataset):
    """Remove numerical state while retaining declared dataset inputs.

    Source, structure, instrument, and parameter declarations remain intact.
    """

    return replace(
        dataset,
        scale_prior=ScalePriorState(enabled=False),
        last_valid_result=None,
        checkpoint=None,
    )


def _replace_dataset(project: XrrProject, index: int, dataset) -> XrrProject:
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


def _failure_result(dataset, error: BaseException | str) -> FitResult:
    """Represent a transaction failure as an untrusted dataset result.

    Region arrays retain source length so downstream rendering stays aligned.
    """

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
    """Index source validation records by persisted dataset identity.

    Transaction preparation consumes this snapshot without re-reading sources.
    """

    return {record.dataset_id: record for record in validation.datasets}


def _source_error(records: dict[str, object], dataset_id: str) -> ValueError | None:
    """Translate a non-current source record into a preparation error.

    Missing records are treated as unavailable only when a record says so.
    """

    record = records.get(dataset_id)
    if record is None or record.status.value == "ok":
        return None
    return ValueError(f"source status {record.status.value}: {record.message}")


def _warnings(results: tuple[DatasetFitResult, ...]) -> tuple[str, ...]:
    """Flatten dataset warnings in publication order.

    The project result preserves dataset order and warning order within rows.
    """

    return tuple(
        warning
        for item in results
        for warning in item.fit_result.warnings
    )


def _checkpoint_with_result_diagnostics(checkpoint, result: FitResult):
    """Project final candidate diagnostics back onto a saved checkpoint.

    Candidate identity, rather than tuple position, controls the replacement.
    """

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
    """Recognize cooperative cancellation across exception subclasses.

    Solvers may mark domain-specific exceptions without sharing their type here.
    """

    return isinstance(error, InterruptedError) or bool(
        getattr(type(error), "_xrr_cooperative_cancellation", False)
    )


def _prepare_independent_rows(
    project: XrrProject,
    records: dict[str, object],
    seeds: dict[str, int],
    prepare_dataset,
) -> tuple[_IndependentPreparation, ...]:
    """Prepare independent rows while retaining every row-level failure.

    Successful preparation state feeds later rows through the working project.
    """

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
    """Divide one total worker budget across concurrent dataset tasks.

    Each runnable task receives at least one worker and remainders go first.
    """

    if count == 0:
        return ()
    if count >= total_workers:
        return (1,) * count
    base, remainder = divmod(total_workers, count)
    return tuple(base + int(index < remainder) for index in range(count))


def _dataset_fit(
    prepared,
    local_workers: int,
    fit_dataset,
    cancelled,
    progress: Callable[[FitProgress], None] | None,
) -> _BufferedFit:
    """Publish progress as it happens while deferring checkpoint commits.

    Progress is a pure notification, so it reaches the caller immediately even
    when several datasets run concurrently; each value carries its own dataset
    id. Checkpoints are withheld because replaying one mutates the accumulated
    immutable project, which must stay serialized in project order.
    """
    checkpoints: list[object] = []
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
    fit_dataset,
    cancelled,
    progress: Callable[[FitProgress], None] | None,
) -> dict[int, _BufferedFit]:
    runnable = tuple(row for row in rows if row.prepared is not None)
    allocations = _worker_allocations(total_workers, len(runnable))
    published = _serialized_progress(progress)
    tasks = tuple(
        lambda row=row, workers=workers: _dataset_fit(
            row.prepared,
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
    return {
        row.index: result
        for row, result in zip(runnable, buffered, strict=True)
    }


def _replay_checkpoints(
    working: XrrProject,
    index: int,
    checkpoints: tuple[object, ...],
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


AUTOMATIC_RUNNABLE_STATUSES = frozenset(
    {AutomaticStatus.PENDING, AutomaticStatus.REFINING, AutomaticStatus.REVIEW}
)


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
        and (
            import_batch_id is None
            or dataset.automation.import_batch_id == import_batch_id
        )
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
    records: dict[str, object],
    seeds: dict[str, int],
    prepare_dataset,
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


def _automatic_fit_parts(result: object) -> tuple[object, FitResult, bool, str | None]:
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
    row: _AutomaticPreparation,
    result: object,
    *,
    refining: bool,
) -> tuple[XrrProject, DatasetFitResult]:
    """Publish one automatic result with coherent settings and checkpoints.

    Parameter changes invalidate stale checkpoints; invalid winners clear the
    publishable result while preserving the prepared declaration state.
    """

    prepared, fit_result, passed, reason = _automatic_fit_parts(result)
    current = working.datasets[row.index]
    prepared_dataset = prepared.updated_dataset
    winner_settings = _winner_settings(
        prepared_dataset.parameter_settings,
        fit_result,
    )
    changed_settings = (
        row.prepared is not None
        and prepared_dataset.parameter_settings
        != row.prepared.updated_dataset.parameter_settings
    )
    status = _automatic_status(fit_result, passed, refining)
    settings_changed = winner_settings != prepared_dataset.parameter_settings
    if status is AutomaticStatus.FAILED:
        checkpoint = None
        last_valid_result = None
        persisted_settings = prepared_dataset.parameter_settings
    else:
        checkpoint = (
            None
            if settings_changed
            else prepared_dataset.checkpoint
            if changed_settings
            else current.checkpoint
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
    updated = _replace_dataset(working, row.index, dataset)
    return updated, DatasetFitResult(dataset.dataset_id, fit_result)


def _commit_automatic_failure(
    working: XrrProject,
    row: _AutomaticPreparation,
    error: BaseException,
) -> tuple[XrrProject, DatasetFitResult]:
    """Publish a row-level automatic failure without a stale candidate.

    The exception type and message become both warning and automation reason.
    """

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
            working = _replace_dataset(working, index, row.prepared.updated_dataset)
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
    """Execute one all-or-nothing expert joint transaction.

    Checkpoint batches and final results must remain aligned with project order.
    """

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
            row,
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
    fit_dataset,
    progress_callback: Callable[[FitProgress], None] | None,
    checkpoint_callback: Callable[[XrrProject], None] | None,
    cancelled: Callable[[], bool] | None,
) -> tuple[
    XrrProject,
    tuple[_AutomaticPreparation, ...],
    dict[int, object],
    bool,
    Callable[[FitProgress], None] | None,
]:
    runnable = tuple(row for row in rows if row.prepared is not None)
    allocations = _worker_allocations(total_workers, len(runnable))
    published_progress = _serialized_progress(progress_callback)
    tasks = tuple(
        lambda row=row, workers=workers: _dataset_fit(
            row.prepared,
            workers,
            fit_dataset,
            cancelled,
            published_progress,
        )
        for row, workers in zip(runnable, allocations, strict=True)
    )
    prefit_results: dict[int, object] = {}
    was_cancelled = False

    def publish_prefit(position: int, outcome: _BufferedFit) -> None:
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
                row,
                outcome.result,
                refining=row.group_size > 1,
            )
        else:
            working, published[row.index] = _commit_automatic_failure(
                working,
                row,
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
    prefit_results: dict[int, object],
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
    prefit: object,
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
        row,
        review,
        refining=False,
    )


def _automatic_joint_checkpoint_project(
    working: XrrProject,
    member_rows: tuple[_AutomaticPreparation, ...],
    values: object,
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
    joint_results: tuple[object, ...],
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
            row,
            result,
            refining=False,
        )
    return working, published


def _fit_automatic_joint_transaction_group(
    working: XrrProject,
    fit_group_id: str,
    member_rows: tuple[_AutomaticPreparation, ...],
    prefit_results: dict[int, object],
    *,
    fit_joint,
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
    member_prepared = tuple(
        _automatic_fit_parts(prefit)[0]
        for prefit in member_prefits
    )
    group_baseline = working

    def publish_checkpoints(values) -> None:
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
                row,
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
    seed_branches,
    prepare_dataset,
    fit_dataset,
    fit_joint,
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
        working, group_results, group_cancelled = (
            _fit_automatic_joint_transaction_group(
                working,
                fit_group_id,
                member_rows,
                prefit_results,
                fit_joint=fit_joint,
                progress=published_progress,
                checkpoint_callback=checkpoint_callback,
                cancelled=cancelled,
            )
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
    seed_branches,
    prepare_dataset,
    fit_dataset,
    fit_joint,
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
