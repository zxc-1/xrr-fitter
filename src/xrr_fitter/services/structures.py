"""Structure validation, evidence, and versioned native-oxide workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType

from xrr_fitter.io.source import dataset_index
from xrr_fitter.model.analysis import StructureEvidence
from xrr_fitter.model.data import BeamSpec
from xrr_fitter.model.parameters import (
    ConstraintRule,
    ParameterSetting,
    SharingRule,
    _iter_references,
)
from xrr_fitter.model.project import OxideDecision, XrrProject
from xrr_fitter.model.structure import (
    GradientLayerSpec,
    LayerSpec,
    MaterialSpec,
    OxideSuggestion,
    PeriodicBlock,
    StructureSpec,
)
from xrr_fitter.physics.sld_profile import sld_depth_profile
from xrr_fitter.physics.stack import expand_structure
from xrr_fitter.services import fitting
from xrr_fitter.services.datasets import _prepared_current, _replace_invalidated
from xrr_fitter.services.parameters import (
    _reconciled_parameter_settings,
    describe_parameters,
    set_constraint_rules,
    validate_parameter_settings,
)

OXIDE_TABLE_VERSION = "oxide-table-v1"
OXIDE_TABLE: Mapping[str, tuple[str, float]] = MappingProxyType(
    {
        "Si": ("SiO2", 2.20),
        "Mo": ("MoO3", 4.69),
        "Al": ("Al2O3", 3.95),
        "Ti": ("TiO2", 4.23),
        "Cu": ("Cu2O", 6.00),
    }
)


def validate_structure(structure: StructureSpec, beam: BeamSpec) -> None:
    """Expand a declaration at every wavelength carried by its beam."""
    if not isinstance(structure, StructureSpec):
        raise TypeError("structure must be a StructureSpec")
    if not isinstance(beam, BeamSpec):
        raise TypeError("beam must be a BeamSpec")
    wavelengths = (beam.wavelength_a,) if beam.kind == "monochromatic" else (beam.wavelength_1_a, beam.wavelength_2_a)
    for wavelength in wavelengths:
        expand_structure(structure, wavelength)


def sld_nominal_profile(structure, *, wavelength_a, step_a=0.5):
    """Return the view-only SLD profile of a declaration at its nominal values."""
    return sld_depth_profile(expand_structure(structure, wavelength_a), step_a=step_a)


def analyze_structure(
    project: XrrProject,
    dataset_id: str,
) -> StructureEvidence:
    """Return detached evidence for the current hash-bound dataset state."""
    dataset = project.datasets[dataset_index(project, dataset_id)]
    if dataset.structure is None:
        raise ValueError(f"dataset has no structure: {dataset_id}")
    return fitting.structure_evidence_for(
        _prepared_current(project, dataset),
        dataset.structure,
    )


def _trimmed_formula(material: MaterialSpec) -> str | None:
    return material.formula.strip() if material.formula is not None else None


def _surface_material(structure: StructureSpec) -> MaterialSpec | None:
    if not structure.components:
        return None
    component = structure.components[0]
    if isinstance(component, LayerSpec):
        return component.material
    if isinstance(component, PeriodicBlock):
        return component.layers[0].material
    return None


def _backing_adjacent_formula(structure: StructureSpec) -> str | None:
    if not structure.components:
        return None
    component = structure.components[-1]
    if isinstance(component, LayerSpec):
        return _trimmed_formula(component.material)
    if isinstance(component, PeriodicBlock):
        return _trimmed_formula(component.layers[-1].material)
    if isinstance(component, GradientLayerSpec):
        return None
    raise TypeError("unsupported structure component")


def _suggestion(
    material: MaterialSpec,
    location: str,
    adjacent_formula: str | None,
) -> OxideSuggestion | None:
    formula = _trimmed_formula(material)
    if formula is None or formula not in OXIDE_TABLE:
        return None
    oxide_formula, density = OXIDE_TABLE[formula]
    if adjacent_formula == oxide_formula:
        return None
    return OxideSuggestion(
        base_material=formula,
        oxide_material=MaterialSpec(oxide_formula, oxide_formula, density),
        density_locked=True,
        thickness_initial_a=10.0,
        thickness_bounds_a=(2.0, 50.0),
        oxide_table_version=OXIDE_TABLE_VERSION,
        location=location,
    )


def suggest_oxide_layers(
    structure: StructureSpec,
) -> tuple[OxideSuggestion, ...]:
    """Return exact table-backed proposals in surface-to-backing order."""
    if not isinstance(structure, StructureSpec):
        raise TypeError("structure must be a StructureSpec")
    suggestions = []
    surface = _surface_material(structure)
    if surface is not None:
        proposal = _suggestion(surface, "surface", None)
        if proposal is not None:
            suggestions.append(proposal)
    backing = _suggestion(
        structure.backing,
        "backing",
        _backing_adjacent_formula(structure),
    )
    if backing is not None:
        suggestions.append(backing)
    return tuple(suggestions)


def _reconciled_sharing(
    rules: tuple[SharingRule, ...],
    dataset_id: str,
    compatible_names: frozenset[str],
) -> tuple[SharingRule, ...]:
    retained = []
    for rule in rules:
        members = tuple(
            member
            for member in rule.members
            if member.dataset_id != dataset_id or member.parameter_name in compatible_names
        )
        if len(members) >= 2:
            retained.append(replace(rule, members=members))
    return tuple(retained)


def _reconciled_constraints(
    rules: tuple[ConstraintRule, ...],
    dataset_id: str,
    compatible_names: frozenset[str],
) -> tuple[ConstraintRule, ...]:
    return tuple(
        rule
        for rule in rules
        if all(
            reference.dataset_id != dataset_id or reference.parameter_name in compatible_names
            for reference in (rule.target, *_iter_references(rule.expression))
        )
    )


def _set_dataset_structure(
    project: XrrProject,
    dataset_id: str,
    structure: StructureSpec,
) -> XrrProject:
    index = dataset_index(project, dataset_id)
    dataset = project.datasets[index]
    validate_structure(structure, dataset.beam)
    if structure == dataset.structure:
        return project
    settings, priors, compatible = _reconciled_parameter_settings(
        project,
        dataset_id,
        structure,
    )
    updated = replace(
        dataset,
        structure=structure,
        structure_evidence=None,
        parameter_settings=settings,
        parameter_priors=priors,
    )
    invalidated = _replace_invalidated(
        project,
        index,
        updated,
        clear_evidence=False,
    )
    reconciled = replace(
        invalidated,
        sharing_rules=_reconciled_sharing(
            invalidated.sharing_rules,
            dataset_id,
            compatible,
        ),
    )
    return set_constraint_rules(
        reconciled,
        _reconciled_constraints(
            invalidated.constraint_rules,
            dataset_id,
            compatible,
        ),
    )


def set_structure(
    project: XrrProject,
    dataset_id: str,
    structure: StructureSpec,
) -> XrrProject:
    """Persist one structure, propagating its topology across a joint batch."""
    dataset_index(project, dataset_id)
    target_ids = (
        tuple(dataset.dataset_id for dataset in project.datasets) if project.batch_mode == "joint" else (dataset_id,)
    )
    updated = project
    for target_id in target_ids:
        updated = _set_dataset_structure(updated, target_id, structure)
    return updated


def _oxide_identity(value: OxideSuggestion | OxideDecision) -> tuple[str, str, str, str]:
    if isinstance(value, OxideSuggestion):
        formula = value.oxide_material.formula
        if formula is None:
            raise ValueError("oxide suggestion requires a formula-backed material")
        return value.base_material, formula, value.location, value.oxide_table_version
    return (
        value.base_material,
        value.oxide_material,
        value.location,
        value.oxide_table_version,
    )


def _with_decision(
    project: XrrProject,
    index: int,
    decision: OxideDecision,
) -> XrrProject:
    dataset = project.datasets[index]
    identity = _oxide_identity(decision)
    decisions = tuple(item for item in dataset.oxide_decisions if _oxide_identity(item) != identity)
    datasets = list(project.datasets)
    datasets[index] = replace(dataset, oxide_decisions=(*decisions, decision))
    return replace(project, datasets=tuple(datasets))


def record_oxide_decision(
    project: XrrProject,
    dataset_id: str,
    decision: OxideDecision,
) -> XrrProject:
    """Record an exact rejection without invalidating structure or fit state."""
    if not isinstance(decision, OxideDecision):
        raise TypeError("decision must be an OxideDecision")
    if decision.accepted:
        raise ValueError("accepted oxide decisions require accept_oxide_suggestion")
    index = dataset_index(project, dataset_id)
    structure = project.datasets[index].structure
    if structure is None:
        raise ValueError(f"dataset has no structure: {dataset_id}")
    identities = {_oxide_identity(item) for item in suggest_oxide_layers(structure)}
    if _oxide_identity(decision) not in identities:
        raise ValueError("oxide decision does not match a current suggestion")
    return _with_decision(project, index, decision)


def _with_oxide_layer(
    structure: StructureSpec,
    suggestion: OxideSuggestion,
) -> StructureSpec:
    formula = suggestion.oxide_material.formula
    if formula is None:
        raise ValueError("accepted oxide requires a formula-backed material")
    layer = LayerSpec(
        f"{formula} native oxide",
        suggestion.oxide_material,
        suggestion.thickness_initial_a,
    )
    components = list(structure.components)
    if suggestion.location == "surface":
        components.insert(0, layer)
    elif suggestion.location == "backing":
        components.append(layer)
    else:
        raise ValueError(f"unsupported oxide location: {suggestion.location}")
    return replace(structure, components=tuple(components))


def accept_oxide_suggestion(
    project: XrrProject,
    dataset_id: str,
    suggestion: OxideSuggestion,
) -> XrrProject:
    """Insert one current proposal with exact provenance and parameter policy."""
    if not isinstance(suggestion, OxideSuggestion):
        raise TypeError("suggestion must be an OxideSuggestion")
    index = dataset_index(project, dataset_id)
    structure = project.datasets[index].structure
    if structure is None:
        raise ValueError(f"dataset has no structure: {dataset_id}")
    if suggestion not in suggest_oxide_layers(structure):
        raise ValueError("oxide suggestion is stale or does not match the structure")
    if not suggestion.density_locked:
        raise ValueError("accepted oxide density must be locked")
    updated = set_structure(
        project,
        dataset_id,
        _with_oxide_layer(structure, suggestion),
    )
    index = dataset_index(updated, dataset_id)
    dataset = updated.datasets[index]
    component_index = 0 if suggestion.location == "surface" else len(dataset.structure.components) - 1
    prefix = f"component.{component_index}"
    lower, upper = suggestion.thickness_bounds_a
    inserted_names = {
        f"{prefix}.thickness_a",
        f"{prefix}.density_scale",
        f"{prefix}.roughness_a",
    }
    retained = tuple(setting for setting in dataset.parameter_settings if setting.name not in inserted_names)
    settings = validate_parameter_settings(
        describe_parameters(updated, dataset_id),
        (
            *retained,
            ParameterSetting(
                f"{prefix}.thickness_a",
                suggestion.thickness_initial_a,
                lower,
                upper,
            ),
            ParameterSetting(
                f"{prefix}.density_scale",
                1.0,
                1.0,
                1.0,
                locked=True,
            ),
        ),
    )
    formula = suggestion.oxide_material.formula
    assert formula is not None
    decision = OxideDecision(
        suggestion.base_material,
        formula,
        suggestion.location,
        True,
        suggestion.oxide_table_version,
    )
    datasets = list(updated.datasets)
    datasets[index] = replace(dataset, parameter_settings=settings)
    return _with_decision(replace(updated, datasets=tuple(datasets)), index, decision)
