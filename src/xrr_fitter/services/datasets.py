"""Dataset import, source acceptance, and derived-state invalidation.

This service owns deterministic dataset IDs and seed branches, source parsing,
measurement metadata, filename-derived automatic structures, and immutable
project insertion. Batch previews remain descriptive until this module imports
each valid row in source order; one malformed row becomes an ``ImportFailure``
without rolling back successful siblings.

Any mutation that changes fitted physics clears candidates, checkpoints,
evidence, and scale priors through one invalidation path. Expert joint projects
invalidate as a complete graph, while automatic projects invalidate only the
matching fit group. Manual rows retain their automation state and unrelated
automatic groups remain publishable.

Source replacement is split into preview and acceptance so callers can display
hash drift before adopting bytes. Fit masks, instruments, and structures use the
same derived-state contract rather than maintaining parallel cleanup rules.
"""

from __future__ import annotations

import json
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np

from xrr_fitter.io.source import dataset_index, resolve_source_path
from xrr_fitter.io.xy import read_xy, read_xy_bytes
from xrr_fitter.model.automation import (
    AutomaticRole,
    AutomaticStatus,
    DatasetAutomation,
    ImportBatchPreview,
    ImportFailure,
    ImportFilePreview,
    MeasurementPreset,
)
from xrr_fitter.model.data import (
    BeamSpec,
    DataColumnMapping,
    PreparedData,
    with_fit_mask,
)
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.operations import ProjectImportResult
from xrr_fitter.model.parameters import ParameterSetting
from xrr_fitter.model.project import (
    DatasetProject,
    ScalePriorState,
    SourceUpdatePreview,
    XrrProject,
)
from xrr_fitter.model.structure import LayerSpec, StructureSpec
from xrr_fitter.services.materials import automatic_structure, initial_structure

SERVICE_SEED_TREE_VERSION = 1


def _uint64_seed(sequence: np.random.SeedSequence) -> int:
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def service_seed_branches(
    project: XrrProject,
) -> tuple[dict[str, int], int, dict[str, np.random.SeedSequence]]:
    """Derive stable independent, joint, and MCMC service branches."""
    independent_root, joint_root, mcmc_root = np.random.SeedSequence(
        project.master_seed,
        spawn_key=(SERVICE_SEED_TREE_VERSION,),
    ).spawn(3)
    dataset_ids = tuple(sorted(dataset.dataset_id for dataset in project.datasets))
    independent_children = independent_root.spawn(len(dataset_ids))
    mcmc_children = mcmc_root.spawn(len(dataset_ids))
    return (
        dict(
            zip(
                dataset_ids,
                map(_uint64_seed, independent_children),
                strict=True,
            )
        ),
        _uint64_seed(joint_root),
        dict(zip(dataset_ids, mcmc_children, strict=True)),
    )


def mcmc_candidate_seed(
    project: XrrProject,
    dataset_id: str,
    candidate_ids: tuple[str, ...],
    candidate_id: str,
) -> int:
    """Derive one candidate stream from sorted dataset and candidate IDs."""
    _independent, _joint, roots = service_seed_branches(project)
    ordered = tuple(sorted(candidate_ids))
    if len(ordered) != len(set(ordered)) or candidate_id not in ordered:
        raise ValueError(f"invalid MCMC candidate: {dataset_id}/{candidate_id}")
    root = roots.get(dataset_id)
    if root is None:
        raise ValueError(f"unknown dataset_id: {dataset_id}")
    children = root.spawn(len(ordered))
    return _uint64_seed(children[ordered.index(candidate_id)])


def import_data(
    path: str | Path,
    beam: BeamSpec,
    import_angle_offset_deg: float = 0.0,
    column_mapping: DataColumnMapping | None = None,
) -> PreparedData:
    """Import one source through the authoritative XY reader."""
    return read_xy(path, beam, import_angle_offset_deg, column_mapping)


def _dataset_id(project: XrrProject, stem: str) -> str:
    reserved = {dataset.dataset_id for dataset in project.datasets}
    candidate = stem
    suffix = 2
    while candidate in reserved:
        candidate = f"{stem}-{suffix}"
        suffix += 1
    return candidate


def _strict_filename_materials(stem: str) -> tuple[str, tuple[str, ...]]:
    parts = stem.rsplit(maxsplit=1)
    if len(parts) != 2:
        raise ValueError("filename must end with a space-separated material stack")
    sample_id, material_segment = parts
    tokens = tuple(value.strip() for value in material_segment.split("+"))
    if not sample_id.strip() or not tokens or any(not value for value in tokens):
        raise ValueError("filename material stack contains an empty token")
    return sample_id, tokens


def _strict_source_materials(path: Path) -> tuple[str, tuple[str, ...]]:
    """Prefer a matching parent stack while preserving the wafer-point ID.

    Only ``W<digits>_exported`` loses the transport suffix. Other stems retain
    the strict filename parser so malformed or unrelated names still fail.
    """
    try:
        folder_sample_id, folder_tokens = _strict_filename_materials(path.parent.name)
    except ValueError:
        pass
    else:
        if path.stem.startswith(f"{folder_sample_id} "):
            point_stem = path.stem[len(folder_sample_id) + 1 :]
            exported_suffix = "_exported"
            if (
                point_stem.startswith("W")
                and point_stem.endswith(exported_suffix)
                and point_stem[1 : -len(exported_suffix)].isdigit()
            ):
                point_stem = point_stem[: -len(exported_suffix)]
            return f"{folder_sample_id} {point_stem}", folder_tokens
    return _strict_filename_materials(path.stem)


def _substrate_group_id(tokens: tuple[str, ...]) -> str:
    encoded = json.dumps(tokens, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    return sha256(encoded).hexdigest()[:20]


def preview_import_batch(
    paths: Sequence[str | Path],
    preset: MeasurementPreset,
    import_batch_id: str | None = None,
) -> ImportBatchPreview:
    """Parse filename or matching parent-folder stacks without reading sources."""
    if not isinstance(preset, MeasurementPreset):
        raise TypeError("preset must be MeasurementPreset")
    batch_id = secrets.token_hex(16) if import_batch_id is None else import_batch_id
    if not batch_id.strip():
        raise ValueError("import_batch_id must not be empty")
    files = []
    for declaration in paths:
        path = Path(declaration)
        try:
            dataset_id_stem, tokens = _strict_source_materials(path)
        except ValueError as error:
            files.append(
                ImportFilePreview(
                    str(path),
                    path.stem,
                    None,
                    (),
                    None,
                    False,
                    str(error),
                )
            )
            continue
        group_id = _substrate_group_id(tokens)
        files.append(
            ImportFilePreview(
                str(path),
                dataset_id_stem,
                dataset_id_stem,
                tokens,
                group_id,
                tokens[0] == "Si",
            )
        )
    return ImportBatchPreview(batch_id, preset, tuple(files))


def _fit_range(data: PreparedData, mask: np.ndarray | None = None) -> tuple[float, float]:
    selected = np.asarray(data.fit_mask if mask is None else mask, dtype=bool)
    finite = np.asarray(data.two_theta_deg, dtype=float)[selected]
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("fit mask must retain at least one finite angle")
    return float(np.min(finite)), float(np.max(finite))


def _from_prepared(
    dataset_id: str,
    display_name: str,
    data: PreparedData,
    instrument: InstrumentSpec,
    *,
    source_path: str,
    structure: StructureSpec | None = None,
    parameter_settings: tuple[ParameterSetting, ...] = (),
    automation: DatasetAutomation | None = None,
) -> DatasetProject:
    return DatasetProject(
        dataset_id=dataset_id,
        source_path=source_path,
        source_sha256=data.source_sha256,
        beam=data.beam,
        import_angle_offset_deg=data.import_angle_offset_deg,
        column_mapping=data.column_mapping,
        fit_mask=tuple(bool(value) for value in data.fit_mask),
        fit_range_two_theta_deg=_fit_range(data),
        structure=structure,
        instrument=instrument,
        parameter_settings=parameter_settings,
        display_name=display_name,
        automation=DatasetAutomation() if automation is None else automation,
    )


def _import_failure(row: ImportFilePreview, error: BaseException) -> ImportFailure:
    return ImportFailure(
        source_path=row.source_path,
        message=f"{type(error).__name__}: {error}",
        recovery_action=(
            "rename the file and retry or open manual structure editing"
            if row.error is not None
            else "choose the data columns for this file and retry"
        ),
    )


def _automatic_dataset(
    project: XrrProject,
    row: ImportFilePreview,
    preview: ImportBatchPreview,
    backing_token: str,
    column_mapping: DataColumnMapping | None,
) -> DatasetProject:
    if row.dataset_id_stem is None:
        raise ValueError("valid import preview requires a dataset ID stem")
    structure, automatic_settings = automatic_structure(
        tuple(reversed(row.layers_backing_to_surface)),
        backing_token,
    )
    data = import_data(
        row.source_path,
        preview.preset.beam,
        preview.preset.import_angle_offset_deg,
        column_mapping,
    )
    return _from_prepared(
        _dataset_id(project, row.dataset_id_stem),
        row.display_name,
        data,
        preview.preset.instrument,
        source_path=row.source_path,
        structure=structure,
        parameter_settings=automatic_settings,
        automation=DatasetAutomation(
            import_batch_id=preview.import_batch_id,
            role=AutomaticRole.UNROUTED,
            status=AutomaticStatus.PENDING,
        ),
    )


def _append_imported_dataset(
    project: XrrProject,
    dataset: DatasetProject,
) -> XrrProject:
    datasets = project.datasets
    state = project.ui_state
    if project.batch_mode == "joint":
        datasets = tuple(_cleared(value, clear_evidence=False) for value in project.datasets)
        state = replace(state, selected_candidate_ids=())
    if state.active_dataset_id is None:
        state = replace(state, active_dataset_id=dataset.dataset_id)
    return replace(project, datasets=(*datasets, dataset), ui_state=state)


def _import_preview_row(
    project: XrrProject,
    row: ImportFilePreview,
    preview: ImportBatchPreview,
    choices: Mapping[str, str],
    mappings: Mapping[str, DataColumnMapping],
) -> DatasetProject:
    if row.error is not None:
        raise ValueError(row.error)
    backing_token = "Si"
    if row.requires_substrate_choice:
        if row.substrate_group_id not in choices:
            raise ValueError("substrate choice is required for this structure group")
        backing_token = choices[row.substrate_group_id]
    return _automatic_dataset(
        project,
        row,
        preview,
        backing_token,
        mappings.get(row.source_path),
    )


def import_dataset_batch(
    project: XrrProject,
    preview: ImportBatchPreview,
    substrate_choices: Mapping[str, str] | None = None,
    column_mappings: Mapping[str, DataColumnMapping] | None = None,
) -> ProjectImportResult:
    """Append each valid source independently while preserving preview order."""
    if not isinstance(project, XrrProject):
        raise TypeError("project must be an XrrProject")
    if not isinstance(preview, ImportBatchPreview):
        raise TypeError("preview must be ImportBatchPreview")
    choices = {} if substrate_choices is None else substrate_choices
    mappings = {} if column_mappings is None else column_mappings
    updated = project
    imported = []
    failures = []
    for row in preview.files:
        try:
            dataset = _import_preview_row(
                updated,
                row,
                preview,
                choices,
                mappings,
            )
            updated = _append_imported_dataset(updated, dataset)
            imported.append(dataset.dataset_id)
        except Exception as error:
            failures.append(_import_failure(row, error))
    if imported:
        updated = replace(updated, measurement_preset=preview.preset)
    return ProjectImportResult(
        updated,
        preview.import_batch_id,
        tuple(imported),
        tuple(failures),
    )


def _filename_materials(stem: str) -> tuple[str, tuple[str, ...] | None]:
    parts = stem.rsplit(maxsplit=1)
    if len(parts) != 2 or "+" not in parts[1]:
        return stem, None
    backing_to_surface = tuple(value.strip() for value in parts[1].split("+"))
    if any(not value for value in backing_to_surface):
        return stem, None
    return parts[0], tuple(reversed(backing_to_surface))


def _ordinary_layer_formulas(structure: StructureSpec | None) -> tuple[str, ...] | None:
    if structure is None or any(not isinstance(component, LayerSpec) for component in structure.components):
        return None
    return tuple((component.material.formula or component.material.name).strip() for component in structure.components)


def _filename_structure(
    project: XrrProject,
    formulas: tuple[str, ...] | None,
) -> StructureSpec | None:
    if formulas is None:
        return None
    matching = next(
        (dataset.structure for dataset in project.datasets if _ordinary_layer_formulas(dataset.structure) == formulas),
        None,
    )
    return initial_structure(formulas) if matching is None else matching


def add_dataset(
    project: XrrProject,
    source_path: str | Path,
    instrument: InstrumentSpec,
    display_name: str | None = None,
    column_mapping: DataColumnMapping | None = None,
    import_angle_offset_deg: float = 0.0,
    beam: BeamSpec | None = None,
) -> XrrProject:
    """Import and append a dataset with a stable source-stem identifier."""
    if not isinstance(project, XrrProject):
        raise TypeError("project must be an XrrProject")
    if not isinstance(instrument, InstrumentSpec):
        raise TypeError("instrument must be an InstrumentSpec")
    beam_value = BeamSpec(kind="monochromatic") if beam is None else beam
    if not isinstance(beam_value, BeamSpec):
        raise TypeError("beam must be a BeamSpec")
    source = Path(source_path)
    stem = source.stem
    identifier_stem, filename_materials = _filename_materials(stem)
    data = import_data(
        source,
        beam_value,
        import_angle_offset_deg,
        column_mapping,
    )
    dataset = _from_prepared(
        _dataset_id(project, identifier_stem),
        stem if display_name is None else display_name,
        data,
        instrument,
        source_path=str(source_path),
        structure=_filename_structure(project, filename_materials),
    )
    state = project.ui_state
    if state.active_dataset_id is None:
        state = replace(state, active_dataset_id=dataset.dataset_id)
    return replace(project, datasets=(*project.datasets, dataset), ui_state=state)


def _cleared(dataset: DatasetProject, *, clear_evidence: bool) -> DatasetProject:
    return replace(
        dataset,
        structure_evidence=None if clear_evidence else dataset.structure_evidence,
        scale_prior=ScalePriorState(enabled=False),
        last_valid_result=None,
        checkpoint=None,
        automation=_reset_automatic_state(dataset),
    )


def _reset_automatic_state(dataset: DatasetProject) -> DatasetAutomation:
    state = dataset.automation
    if state.role is AutomaticRole.MANUAL:
        return state
    return replace(
        state,
        status=AutomaticStatus.PENDING,
        statistics_member=False,
        reason=None,
    )


def _automatic_fit_group_member(
    dataset: DatasetProject,
    groups: set[str],
) -> bool:
    state = dataset.automation
    return (
        state.role is not AutomaticRole.MANUAL
        and state.fit_group_id in groups
    )


def _dependent_fit_ids(project: XrrProject, changed_ids: set[str]) -> set[str]:
    if project.batch_mode == "joint" and changed_ids:
        return {dataset.dataset_id for dataset in project.datasets}
    groups = {
        dataset.automation.fit_group_id
        for dataset in project.datasets
        if dataset.dataset_id in changed_ids
        and dataset.automation.fit_group_id is not None
    }
    automatic = {
        dataset.dataset_id
        for dataset in project.datasets
        if _automatic_fit_group_member(dataset, groups)
    }
    if automatic:
        return changed_ids | automatic
    return changed_ids


def _replace_invalidated(
    project: XrrProject,
    index: int,
    updated: DatasetProject,
    *,
    clear_evidence: bool,
) -> XrrProject:
    datasets = list(project.datasets)
    datasets[index] = updated
    affected = _dependent_fit_ids(project, {updated.dataset_id})
    datasets = [
        _cleared(dataset, clear_evidence=clear_evidence)
        if dataset.dataset_id in affected
        else dataset
        for dataset in datasets
    ]
    selected = tuple(
        item
        for item in project.ui_state.selected_candidate_ids
        if item[0] not in affected
    )
    return replace(
        project,
        datasets=tuple(datasets),
        ui_state=replace(project.ui_state, selected_candidate_ids=selected),
    )


def _ui_after_removal(
    project: XrrProject,
    datasets: tuple[DatasetProject, ...],
    remaining_ids: set[str],
    index: int,
    dataset_id: str,
):
    active = project.ui_state.active_dataset_id
    if active == dataset_id:
        active = datasets[min(index, len(datasets) - 1)].dataset_id if datasets else None
    selected = tuple(item for item in project.ui_state.selected_candidate_ids if item[0] in remaining_ids)
    return replace(
        project.ui_state,
        active_dataset_id=active,
        selected_candidate_ids=selected,
    )


def _sharing_after_removal(
    project: XrrProject,
    remaining_ids: set[str],
):
    return tuple(
        rule for rule in project.sharing_rules if all(member.dataset_id in remaining_ids for member in rule.members)
    )


def remove_dataset(project: XrrProject, dataset_id: str) -> XrrProject:
    """Remove one dataset and every cross-project reference to it."""
    index = dataset_index(project, dataset_id)
    datasets = (*project.datasets[:index], *project.datasets[index + 1 :])
    if project.batch_mode == "joint":
        datasets = tuple(_cleared(dataset, clear_evidence=False) for dataset in datasets)
    remaining_ids = {dataset.dataset_id for dataset in datasets}
    mode = "independent" if project.batch_mode == "joint" and len(datasets) < 2 else project.batch_mode
    return replace(
        project,
        batch_mode=mode,
        datasets=datasets,
        sharing_rules=_sharing_after_removal(project, remaining_ids),
        ui_state=_ui_after_removal(project, datasets, remaining_ids, index, dataset_id),
    )


def _read_current(project: XrrProject, dataset: DatasetProject) -> PreparedData:
    data = import_data(
        resolve_source_path(project, dataset),
        dataset.beam,
        dataset.import_angle_offset_deg,
        dataset.column_mapping,
    )
    if data.source_sha256 != dataset.source_sha256:
        raise ValueError(f"source changed for dataset {dataset.dataset_id}")
    return data


def _prepared_current(project: XrrProject, dataset: DatasetProject) -> PreparedData:
    data = _read_current(project, dataset)
    return with_fit_mask(data, np.asarray(dataset.fit_mask, dtype=bool))


def set_fit_mask(
    project: XrrProject,
    dataset_id: str,
    mask: np.ndarray,
) -> XrrProject:
    """Persist a validated mask and invalidate source-derived state."""
    if not isinstance(mask, np.ndarray) or mask.ndim != 1 or mask.dtype != np.bool_:
        raise TypeError("mask must be a one-dimensional bool numpy.ndarray")
    index = dataset_index(project, dataset_id)
    dataset = project.datasets[index]
    data = _read_current(project, dataset)
    masked = with_fit_mask(data, mask)
    updated = replace(
        dataset,
        fit_mask=tuple(bool(value) for value in masked.fit_mask),
        fit_range_two_theta_deg=_fit_range(masked),
    )
    return _replace_invalidated(
        project,
        index,
        updated,
        clear_evidence=True,
    )


def set_instrument(
    project: XrrProject,
    dataset_id: str,
    instrument: InstrumentSpec,
) -> XrrProject:
    """Persist an instrument declaration and invalidate fit-derived state."""
    if not isinstance(instrument, InstrumentSpec):
        raise TypeError("instrument must be an InstrumentSpec")
    index = dataset_index(project, dataset_id)
    dataset = project.datasets[index]
    if instrument == dataset.instrument:
        return project
    return _replace_invalidated(
        project,
        index,
        replace(dataset, instrument=instrument),
        clear_evidence=False,
    )


def _proposed_source(
    project: XrrProject,
    dataset: DatasetProject,
    declaration: str,
) -> Path:
    path = Path(declaration)
    if path.is_absolute():
        return path
    if declaration == dataset.source_path:
        return resolve_source_path(project, dataset)
    if project.base_directory:
        return Path(project.base_directory) / path
    return path


def preview_source_update(
    project: XrrProject,
    dataset_id: str,
    new_path: str | Path | None = None,
) -> SourceUpdatePreview:
    """Hash a proposed source without changing project state."""
    dataset = project.datasets[dataset_index(project, dataset_id)]
    declaration = dataset.source_path if new_path is None else str(new_path)
    observed = sha256(_proposed_source(project, dataset, declaration).read_bytes()).hexdigest()
    return SourceUpdatePreview(
        dataset_id=dataset.dataset_id,
        current_source_path=dataset.source_path,
        proposed_source_path=declaration,
        expected_sha256=dataset.source_sha256,
        observed_sha256=observed,
        changed=observed != dataset.source_sha256 or declaration != dataset.source_path,
    )


def _accepted_source_dataset(
    project: XrrProject,
    preview: SourceUpdatePreview,
) -> tuple[int, DatasetProject]:
    """Read exactly the source bytes observed by a prior preview."""
    if not isinstance(preview, SourceUpdatePreview):
        raise TypeError("preview must be a SourceUpdatePreview")
    index = dataset_index(project, preview.dataset_id)
    dataset = project.datasets[index]
    if preview.current_source_path != dataset.source_path or preview.expected_sha256 != dataset.source_sha256:
        raise ValueError("source update preview is stale")
    source = _proposed_source(project, dataset, preview.proposed_source_path)
    content = source.read_bytes()
    if sha256(content).hexdigest() != preview.observed_sha256:
        raise ValueError("proposed source changed after preview")
    data = read_xy_bytes(
        content,
        source_path=source,
        beam=dataset.beam,
        import_angle_offset_deg=dataset.import_angle_offset_deg,
        column_mapping=dataset.column_mapping,
    )
    return index, replace(
        dataset,
        source_path=preview.proposed_source_path,
        source_sha256=data.source_sha256,
        fit_mask=tuple(bool(value) for value in data.fit_mask),
        fit_range_two_theta_deg=_fit_range(data),
    )
