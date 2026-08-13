"""Parameter declarations, settings, and pure sharing validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from xrr_fitter.io.source import dataset_index
from xrr_fitter.model.parameters import (
    ParameterDefinition,
    ParameterPrior,
    ParameterReference,
    ParameterSetting,
    SharingRule,
)
from xrr_fitter.model.project import DatasetProject, SourceUpdatePreview, XrrProject
from xrr_fitter.model.structure import StructureSpec
from xrr_fitter.services import fitting
from xrr_fitter.services.datasets import (
    _accepted_source_dataset,
    _cleared,
    _prepared_current,
    _replace_invalidated,
)

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


def _with_priors(
    definitions: tuple[ParameterDefinition, ...],
    priors: Sequence[ParameterPrior],
) -> tuple[ParameterDefinition, ...]:
    """Overlay stored priors onto compiled definitions without touching bounds.

    Only the ``prior`` field is replaced, so definitions without a stored prior
    remain byte-identical and the safety net of ``parameter_priors == ()`` keeps
    every declaration exactly as compiled.
    """
    if not priors:
        return definitions
    overlay = {value.name: value.prior for value in priors}
    return tuple(
        replace(definition, prior=overlay[definition.name]) if definition.name in overlay else definition
        for definition in definitions
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
    definitions = fitting.compiled_parameter_definitions(
        data,
        dataset.structure,
        dataset.instrument,
        project.fit_config,
        dataset.parameter_settings,
    )
    return _with_priors(definitions, dataset.parameter_priors)


def validate_parameter_settings(
    definitions: Sequence[ParameterDefinition],
    settings: Sequence[ParameterSetting],
) -> tuple[ParameterSetting, ...]:
    """Validate settings against immutable definition metadata in input order."""
    declaration_values = tuple(definitions)
    setting_values = tuple(settings)
    fitting.validate_parameter_setting_declarations(declaration_values, setting_values)
    return setting_values


def _reconciled_source_sidecars(
    project: XrrProject,
    index: int,
    updated: DatasetProject,
) -> tuple[tuple[ParameterSetting, ...], tuple[ParameterPrior, ...]]:
    settings = updated.parameter_settings
    priors = updated.parameter_priors
    if updated.structure is None:
        return (), ()
    if not settings and not priors:
        return (), ()
    datasets = list(project.datasets)
    datasets[index] = replace(updated, parameter_settings=(), parameter_priors=())
    definition_project = replace(project, datasets=tuple(datasets))
    data = _prepared_current(definition_project, updated)
    definitions = _default_definitions(
        definition_project,
        updated,
        data,
        updated.structure,
    )
    return _reconciled_parameter_sidecars(definitions, settings, priors)


def accept_source_update(
    project: XrrProject,
    preview: SourceUpdatePreview,
) -> XrrProject:
    """Accept previewed bytes and retain only compatible parameter settings."""
    index, updated = _accepted_source_dataset(project, preview)
    settings, priors = _reconciled_source_sidecars(project, index, updated)
    reconciled = replace(updated, parameter_settings=settings, parameter_priors=priors)
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
    changed = sum(old is not new for old, new in zip(previous.components, updated.components, strict=True))
    return changed > 1


def _compatible_definition_names(
    old_definitions: Sequence[ParameterDefinition],
    new_definitions: Sequence[ParameterDefinition],
    *,
    topology_changed: bool,
) -> frozenset[str]:
    new_by_name = {definition.name: definition for definition in new_definitions}
    return frozenset(
        old.name
        for old in old_definitions
        if _definition_is_compatible(
            old,
            new_by_name,
            topology_changed=topology_changed,
        )
    )


def _definition_is_compatible(
    old: ParameterDefinition,
    new_by_name: Mapping[str, ParameterDefinition],
    *,
    topology_changed: bool,
) -> bool:
    updated = new_by_name.get(old.name)
    return (
        updated is not None
        and _topology_allows_parameter(old.name, topology_changed)
        and _role_signature(old) == _role_signature(updated)
    )


def _topology_allows_parameter(name: str, topology_changed: bool) -> bool:
    return not topology_changed or not name.startswith("component.")


def _retained_named(values, compatible: frozenset[str]) -> tuple:
    return tuple(value for value in values if value.name in compatible)


def _default_definitions_for_existing_structure(
    project: XrrProject,
    dataset,
    data,
) -> tuple[ParameterDefinition, ...]:
    if dataset.structure is None:
        return ()
    return _default_definitions(project, dataset, data, dataset.structure)


def _reconciled_parameter_settings(
    project: XrrProject,
    dataset_id: str,
    structure: StructureSpec,
) -> tuple[
    tuple[ParameterSetting, ...],
    tuple[ParameterPrior, ...],
    frozenset[str],
]:
    index = dataset_index(project, dataset_id)
    dataset = project.datasets[index]
    data = _prepared_current(project, dataset)
    old_definitions = _default_definitions_for_existing_structure(project, dataset, data)
    new_definitions = _default_definitions(project, dataset, data, structure)
    compatible = _compatible_definition_names(
        old_definitions,
        new_definitions,
        topology_changed=_topology_changed(dataset.structure, structure),
    )
    settings, priors = _reconciled_parameter_sidecars(
        new_definitions,
        _retained_named(dataset.parameter_settings, compatible),
        _retained_named(dataset.parameter_priors, compatible),
    )
    return settings, priors, compatible


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
    effective = fitting.effective_parameter_definitions(definitions, validated)
    _, priors = _reconciled_parameter_sidecars(
        effective,
        (),
        dataset.parameter_priors,
    )
    if validated == dataset.parameter_settings and priors == dataset.parameter_priors:
        return project
    updated = replace(
        dataset,
        parameter_settings=validated,
        parameter_priors=priors,
    )
    return _replace_invalidated(
        project,
        index,
        updated,
        clear_evidence=False,
    )


def validate_parameter_priors(
    definitions: Sequence[ParameterDefinition],
    priors: Sequence[ParameterPrior],
) -> tuple[ParameterPrior, ...]:
    """Validate priors against definition metadata, rejecting unknown names.

    Binding each prior with ``replace(definition, prior=...)`` reuses the
    declaration's own center-within-bounds check, so a prior can never sit
    outside the interval it constrains.
    """
    values = tuple(priors)
    if any(not isinstance(value, ParameterPrior) for value in values):
        raise TypeError("priors must contain ParameterPrior values")
    if len({value.name for value in values}) != len(values):
        raise ValueError("parameter prior names must be unique")
    by_name = {definition.name: definition for definition in definitions}
    for value in values:
        try:
            definition = by_name[value.name]
        except KeyError as error:
            raise ValueError(f"unknown parameter name: {value.name}") from error
        replace(definition, prior=value.prior)
    return values


def _reconciled_parameter_sidecars(
    definitions: Sequence[ParameterDefinition],
    settings: Sequence[ParameterSetting],
    priors: Sequence[ParameterPrior],
) -> tuple[tuple[ParameterSetting, ...], tuple[ParameterPrior, ...]]:
    """Keep the first valid sidecar for each still-compatible declaration."""
    retained_settings: list[ParameterSetting] = []
    seen_settings: set[str] = set()
    for setting in settings:
        if setting.name in seen_settings:
            continue
        try:
            validate_parameter_settings(definitions, (setting,))
        except ValueError:
            continue
        seen_settings.add(setting.name)
        retained_settings.append(setting)

    prior_definitions = fitting.effective_parameter_definitions(
        definitions,
        retained_settings,
    )

    retained_priors: list[ParameterPrior] = []
    seen_priors: set[str] = set()
    for prior in priors:
        if prior.name in seen_priors:
            continue
        try:
            validate_parameter_priors(prior_definitions, (prior,))
        except ValueError:
            continue
        seen_priors.add(prior.name)
        retained_priors.append(prior)

    return (
        validate_parameter_settings(definitions, retained_settings),
        validate_parameter_priors(prior_definitions, retained_priors),
    )


def set_parameter_priors(
    project: XrrProject,
    dataset_id: str,
    priors: Sequence[ParameterPrior],
) -> XrrProject:
    """Persist validated priors and invalidate their dependent fit state."""
    index = dataset_index(project, dataset_id)
    dataset = project.datasets[index]
    if dataset.structure is None:
        raise ValueError(f"dataset has no structure: {dataset_id}")
    definitions = describe_parameters(project, dataset_id)
    validated = validate_parameter_priors(definitions, priors)
    if validated == dataset.parameter_priors:
        return project
    updated = replace(dataset, parameter_priors=validated)
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
        for member in rule.members:
            if member in owners:
                raise ValueError("sharing coordinate has multiple ownership")
            owners.add(member)
    return values


def _rule_dataset_ids(rules: Sequence[SharingRule]) -> set[str]:
    return {member.dataset_id for rule in rules for member in rule.members}


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
        _cleared(dataset, clear_evidence=False) if dataset.dataset_id in affected else dataset
        for dataset in project.datasets
    )
    selected = tuple(item for item in project.ui_state.selected_candidate_ids if item[0] not in affected)
    return replace(
        project,
        datasets=datasets,
        sharing_rules=validated,
        ui_state=replace(project.ui_state, selected_candidate_ids=selected),
    )
