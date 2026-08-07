"""Project persistence and workspace-state use cases."""

from __future__ import annotations

import os
import secrets
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Literal

from xrr_fitter.io.project_codec import load_project as _load_project
from xrr_fitter.io.project_codec import save_project as _save_project
from xrr_fitter.io.source import (
    dataset_by_id,
    resolve_source_path,
    validate_sources,
)
from xrr_fitter.io.xy import read_xy
from xrr_fitter.model.data import with_fit_mask
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.parameters import JointFitLayout
from xrr_fitter.model.project import (
    DatasetProject,
    ProjectUiState,
    ProjectValidation,
    SourceStatus,
    XrrProject,
    with_active_dataset,
    with_workspace_state,
)
from xrr_fitter.services.datasets import _cleared, _dependent_fit_ids

SOURCE_RESTORE_ATTEMPTS = 2


def new_project() -> XrrProject:
    """Create an empty project with a fresh persisted 64-bit seed."""
    project = XrrProject.new((), master_seed=secrets.randbits(64))
    return replace(project, fit_config=FitConfig.fast(project.master_seed))


def load_project(path: str | Path) -> XrrProject:
    """Load, source-validate, and invalidate one persisted workspace."""
    return _prepare_source_snapshots(_load_project(path))


def save_project(project: XrrProject, path: str | Path) -> None:
    """Revalidate, relocate, and atomically persist a complete project."""
    target = Path(path)
    current = _prepare_source_snapshots(project)
    _save_project(_rebased_project(current, target.resolve().parent), target)


def inspect_sources(project: XrrProject) -> ProjectValidation:
    """Inspect every declared source without mutating the project."""
    return validate_sources(project)


def _source_records(validation: ProjectValidation) -> dict[str, object]:
    records = {record.dataset_id: record for record in validation.datasets}
    if len(records) != len(validation.datasets):
        raise ValueError("source validation dataset IDs must be unique")
    return records


def _validated_source_records(
    project: XrrProject,
    validation: ProjectValidation,
) -> dict[str, object]:
    records = _source_records(validation)
    if set(records) != {dataset.dataset_id for dataset in project.datasets}:
        raise ValueError("source validation must cover every project dataset")
    return records


def _affected_source_ids(
    project: XrrProject,
    records: dict[str, object],
) -> set[str]:
    invalid = {
        dataset_id
        for dataset_id, record in records.items()
        if record.status is not SourceStatus.OK
    }
    return _dependent_fit_ids(project, invalid)


def _invalidated_sources(
    project: XrrProject,
    validation: ProjectValidation,
) -> XrrProject:
    records = _validated_source_records(project, validation)
    affected = _affected_source_ids(project, records)
    if not affected:
        return project
    datasets = tuple(
        _cleared(dataset, clear_evidence=True)
        if dataset.dataset_id in affected
        else dataset
        for dataset in project.datasets
    )
    selected = tuple(
        item
        for item in project.ui_state.selected_candidate_ids
        if item[0] not in affected
    )
    return replace(
        project,
        datasets=datasets,
        ui_state=replace(project.ui_state, selected_candidate_ids=selected),
    )


def _raced_source_ids(
    project: XrrProject,
    validation: ProjectValidation,
) -> frozenset[str]:
    records = _source_records(validation)
    raced: set[str] = set()
    for dataset in project.datasets:
        record = records[dataset.dataset_id]
        if record.status is not SourceStatus.OK:
            continue
        try:
            data = read_xy(
                resolve_source_path(project, dataset),
                dataset.beam,
                dataset.import_angle_offset_deg,
                dataset.column_mapping,
            )
        except OSError:
            raced.add(dataset.dataset_id)
            continue
        if data.source_sha256 != record.actual_sha256:
            raced.add(dataset.dataset_id)
            continue
        with_fit_mask(data, dataset.fit_mask)
    return frozenset(raced)


def _prepare_source_snapshots(project: XrrProject) -> XrrProject:
    validation = validate_sources(project)
    current = project
    for attempt in range(SOURCE_RESTORE_ATTEMPTS):
        current = _invalidated_sources(current, validation)
        raced = _raced_source_ids(current, validation)
        if not raced:
            return current
        fresh = validate_sources(current)
        if attempt == SOURCE_RESTORE_ATTEMPTS - 1:
            records = _source_records(fresh)
            if all(
                records[dataset_id].status is not SourceStatus.OK
                for dataset_id in raced
            ):
                return _invalidated_sources(current, fresh)
            raise RuntimeError("source changed repeatedly during project restore")
        validation = fresh
    raise AssertionError("unreachable source restore state")


def _rebased_project(project: XrrProject, target_directory: Path) -> XrrProject:
    target = target_directory.resolve()
    source_base = (
        Path(project.base_directory).resolve()
        if project.base_directory is not None
        else Path.cwd().resolve()
    )
    if source_base == target:
        return replace(project, base_directory=str(target))
    datasets = tuple(
        dataset
        if Path(dataset.source_path).is_absolute()
        else replace(
            dataset,
            source_path=str(
                Path(
                    os.path.relpath(
                        (source_base / dataset.source_path).resolve(),
                        target,
                    )
                )
            ),
        )
        for dataset in project.datasets
    )
    return replace(project, datasets=datasets, base_directory=str(target))


def select_active_dataset(project: XrrProject, dataset_id: str | None) -> XrrProject:
    return with_active_dataset(project, dataset_id)


def select_candidate(
    project: XrrProject,
    dataset_id: str,
    candidate_id: str | None,
) -> XrrProject:
    dataset = dataset_by_id(project, dataset_id)
    selected = tuple(
        item for item in project.ui_state.selected_candidate_ids if item[0] != dataset_id
    )
    if candidate_id is not None:
        result = dataset.last_valid_result
        if result is None or candidate_id not in {
            candidate.candidate_id for candidate in result.candidates
        }:
            raise ValueError("candidate_id is absent from the persisted result")
        selected = (*selected, (dataset_id, candidate_id))
    return replace(
        project,
        ui_state=replace(project.ui_state, selected_candidate_ids=selected),
    )


def set_expert_mode(project: XrrProject, enabled: bool) -> XrrProject:
    if not isinstance(enabled, bool):
        raise TypeError("enabled must be bool")
    return replace(
        project,
        ui_state=replace(project.ui_state, expert_mode=enabled),
    )


def set_workspace_state(project: XrrProject, state: ProjectUiState) -> XrrProject:
    return with_workspace_state(project, state)


def _joint_structure_template(
    datasets: tuple[DatasetProject, ...],
    active_id: str | None,
) -> DatasetProject | None:
    active = next(
        (
            dataset
            for dataset in datasets
            if dataset.dataset_id == active_id and dataset.structure is not None
        ),
        None,
    )
    if active is not None:
        return active
    return next(
        (dataset for dataset in datasets if dataset.structure is not None),
        None,
    )


def _fill_missing_joint_structures(
    datasets: tuple[DatasetProject, ...],
    active_id: str | None,
) -> tuple[DatasetProject, ...]:
    template = _joint_structure_template(datasets, active_id)
    if template is None:
        return datasets
    return tuple(
        replace(
            dataset,
            structure=template.structure,
            structure_evidence=None,
            parameter_settings=(),
        )
        if dataset.structure is None
        else dataset
        for dataset in datasets
    )


def clear_fit_results(
    project: XrrProject,
    dataset_ids: Sequence[str],
) -> XrrProject:
    requested = tuple(dataset_ids)
    if len(requested) != len(set(requested)):
        raise ValueError("dataset_ids must be unique")
    known = {dataset.dataset_id for dataset in project.datasets}
    unknown = set(requested) - known
    if unknown:
        raise ValueError(f"unknown dataset_id: {sorted(unknown)[0]}")
    affected = _dependent_fit_ids(project, set(requested))
    datasets = tuple(
        _cleared(dataset, clear_evidence=False)
        if dataset.dataset_id in affected
        else dataset
        for dataset in project.datasets
    )
    selected = tuple(
        item
        for item in project.ui_state.selected_candidate_ids
        if item[0] not in affected
    )
    return replace(
        project,
        datasets=datasets,
        ui_state=replace(project.ui_state, selected_candidate_ids=selected),
    )


def set_batch_mode(
    project: XrrProject,
    mode: Literal["independent", "joint"],
) -> XrrProject:
    if mode not in {"independent", "joint"}:
        raise ValueError(f"unsupported batch_mode: {mode}")
    if mode == "joint" and len(project.datasets) < 2:
        raise ValueError("joint batch mode requires at least two datasets")
    if mode == project.batch_mode:
        return project
    datasets = tuple(
        _cleared(dataset, clear_evidence=False) for dataset in project.datasets
    )
    if mode == "joint":
        datasets = _fill_missing_joint_structures(
            datasets,
            project.ui_state.active_dataset_id,
        )
    updated = replace(
        project,
        batch_mode=mode,
        datasets=datasets,
        ui_state=replace(project.ui_state, selected_candidate_ids=()),
    )
    return updated


def describe_joint_layout(project: XrrProject) -> JointFitLayout:
    """Report a joint project's participating datasets and shared parameters.

    This is a pure read of persisted fields: it never compiles a joint problem
    or loads source data, so it stays cheap enough for a progress view to call
    on every refresh.  It reports the declared sharing structure as-is and does
    not verify that each shared parameter still exists or is free.
    """
    if project.batch_mode != "joint":
        raise ValueError("joint layout requires the joint batch mode")
    dataset_ids = tuple(dataset.dataset_id for dataset in project.datasets)
    return JointFitLayout(dataset_ids, tuple(project.sharing_rules))
