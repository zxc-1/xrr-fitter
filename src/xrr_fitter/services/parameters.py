"""Parameter declarations, settings, and pure sharing validation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from xrr_fitter.io.source import dataset_index
from xrr_fitter.model.parameters import (
    ParameterDefinition,
    ParameterReference,
    ParameterSetting,
    SharingRule,
)
from xrr_fitter.model.project import DatasetProject, SourceUpdatePreview, XrrProject
from xrr_fitter.model.structure import StructureSpec
from xrr_fitter.services.datasets import (
    _accepted_source_dataset,
    _cleared,
    _prepared_current,
    _replace_invalidated,
)
from xrr_fitter.services import fitting


ROLE_FIELDS = (
    "display_name",
    "unit",
    "category",
    "transform",
    "locked",
    "integer",
    "expert_only",
    "sharing_key",
)


def _default_definitions(
    project: XrrProject,
    dataset,
    data,
    structure: StructureSpec,
) -> tuple[ParameterDefinition, ...]:
    return fitting.parameter_definitions_for(
        data,
        structure,
        dataset.instrument,
        project.fit_config,
    )


def describe_parameters(
    project: XrrProject,
    dataset_id: str,
) -> tuple[ParameterDefinition, ...]:
    """Compile the current dataset and expose all parameter declarations."""
    dataset = project.datasets[dataset_index(project, dataset_id)]
    if dataset.structure is None:
        raise ValueError(f"dataset has no structure: {dataset_id}")
    data = _prepared_current(project, dataset)
    return fitting.compiled_parameter_definitions(
        data,
        dataset.structure,
        dataset.instrument,
        project.fit_config,
        dataset.parameter_settings,
    )


def validate_parameter_settings(
    definitions: Sequence[ParameterDefinition],
    settings: Sequence[ParameterSetting],
) -> tuple[ParameterSetting, ...]:
    """Validate settings against immutable definition metadata in input order."""
    declaration_values = tuple(definitions)
    setting_values = tuple(settings)
    fitting.validate_parameter_setting_declarations(declaration_values, setting_values)
    return setting_values


def _reconciled_source_settings(
    project: XrrProject,
    index: int,
    updated: DatasetProject,
) -> tuple[ParameterSetting, ...]:
    settings = updated.parameter_settings
    if not settings or updated.structure is None:
        return () if updated.structure is None else settings
    datasets = list(project.datasets)
    datasets[index] = replace(updated, parameter_settings=())
    definition_project = replace(project, datasets=tuple(datasets))
    definitions = describe_parameters(definition_project, updated.dataset_id)
    retained = []
    seen: set[str] = set()
    for setting in settings:
        if setting.name in seen:
            continue
        try:
            validate_parameter_settings(definitions, (setting,))
        except ValueError:
            continue
        seen.add(setting.name)
        retained.append(setting)
    return validate_parameter_settings(definitions, retained)


def accept_source_update(
    project: XrrProject,
    preview: SourceUpdatePreview,
) -> XrrProject:
    """Accept previewed bytes and retain only compatible parameter settings."""
    index, updated = _accepted_source_dataset(project, preview)
    reconciled = replace(
        updated,
        parameter_settings=_reconciled_source_settings(project, index, updated),
    )
    return _replace_invalidated(
        project,
        index,
        reconciled,
        clear_evidence=True,
    )


def _role_signature(definition: ParameterDefinition) -> tuple[object, ...]:
    return tuple(getattr(definition, field) for field in ROLE_FIELDS)


def _topology_changed(
    previous: StructureSpec | None,
    updated: StructureSpec,
) -> bool:
    if previous is None or len(previous.components) != len(updated.components):
        return True
    changed = sum(
        old is not new
        for old, new in zip(previous.components, updated.components, strict=True)
    )
    return changed > 1


def _reconciled_parameter_settings(
    project: XrrProject,
    dataset_id: str,
    structure: StructureSpec,
) -> tuple[tuple[ParameterSetting, ...], frozenset[str]]:
    index = dataset_index(project, dataset_id)
    dataset = project.datasets[index]
    data = _prepared_current(project, dataset)
    old_definitions = (
        ()
        if dataset.structure is None
        else _default_definitions(project, dataset, data, dataset.structure)
    )
    new_definitions = _default_definitions(project, dataset, data, structure)
    new_by_name = {definition.name: definition for definition in new_definitions}
    topology_changed = _topology_changed(dataset.structure, structure)
    compatible = frozenset(
        old.name
        for old in old_definitions
        if old.name in new_by_name
        and (not topology_changed or not old.name.startswith("component."))
        and _role_signature(old) == _role_signature(new_by_name[old.name])
    )
    retained = tuple(
        setting
        for setting in dataset.parameter_settings
        if setting.name in compatible
    )
    return validate_parameter_settings(new_definitions, retained), compatible


def set_parameter_settings(
    project: XrrProject,
    dataset_id: str,
    settings: Sequence[ParameterSetting],
) -> XrrProject:
    """Persist validated settings and invalidate their dependent fit state."""
    index = dataset_index(project, dataset_id)
    dataset = project.datasets[index]
    if dataset.structure is None:
        raise ValueError(f"dataset has no structure: {dataset_id}")
    data = _prepared_current(project, dataset)
    definitions = _default_definitions(project, dataset, data, dataset.structure)
    validated = validate_parameter_settings(definitions, settings)
    if validated == dataset.parameter_settings:
        return project
    updated = replace(dataset, parameter_settings=validated)
    return _replace_invalidated(
        project,
        index,
        updated,
        clear_evidence=False,
    )


def validate_sharing_rules(
    project: XrrProject,
    rules: Sequence[SharingRule],
) -> tuple[SharingRule, ...]:
    """Validate sharing declarations without loading or compiling source data."""
    values = tuple(rules)
    if any(not isinstance(rule, SharingRule) for rule in values):
        raise TypeError("rules must contain SharingRule values")
    replace(project, sharing_rules=values)
    owners: set[ParameterReference] = set()
    for rule in values:
        dataset_ids = tuple(member.dataset_id for member in rule.members)
        if len(dataset_ids) != len(set(dataset_ids)):
            raise ValueError("sharing group may contain at most one member per dataset")
        for member in rule.members:
            if member in owners:
                raise ValueError("sharing coordinate has multiple ownership")
            owners.add(member)
    return values


def _rule_dataset_ids(rules: Sequence[SharingRule]) -> set[str]:
    return {
        member.dataset_id
        for rule in rules
        for member in rule.members
    }


def set_sharing_rules(
    project: XrrProject,
    rules: Sequence[SharingRule],
) -> XrrProject:
    """Persist sharing declarations and invalidate the affected result graph."""
    validated = validate_sharing_rules(project, rules)
    if validated == project.sharing_rules:
        return project
    affected = _rule_dataset_ids((*project.sharing_rules, *validated))
    if project.batch_mode == "joint" and affected:
        affected = {dataset.dataset_id for dataset in project.datasets}
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
        sharing_rules=validated,
        ui_state=replace(project.ui_state, selected_candidate_ids=selected),
    )
