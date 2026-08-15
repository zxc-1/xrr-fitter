from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest
from tests.support.model_cases import final_fit_result, simple_structure

from xrr_fitter.io.xy import xy_bytes
from xrr_fitter.model.analysis import StructureEvidence
from xrr_fitter.model.data import BeamSpec
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import (
    ConstraintNode,
    ConstraintRule,
    ParameterPrior,
    ParameterReference,
    PriorSpec,
)
from xrr_fitter.model.project import OxideDecision, ProjectUiState
from xrr_fitter.model.structure import (
    GradientLayerSpec,
    LayerSpec,
    MaterialSpec,
    PeriodicBlock,
    StructureSpec,
)
from xrr_fitter.services.datasets import add_dataset
from xrr_fitter.services.parameters import (
    describe_parameters,
    set_constraint_rules,
    set_parameter_priors,
)
from xrr_fitter.services.projects import (
    load_project,
    new_project,
    save_project,
    set_batch_mode,
)
from xrr_fitter.services.structures import (
    OXIDE_TABLE,
    OXIDE_TABLE_VERSION,
    accept_oxide_suggestion,
    analyze_structure,
    record_oxide_decision,
    set_structure,
    suggest_oxide_layers,
    validate_structure,
)

AIR = MaterialSpec("Air", None, None, 0.0j)


def _source(path: Path) -> Path:
    angles = np.linspace(0.1, 3.2, 64)
    path.write_bytes(xy_bytes(angles, np.geomspace(1.0, 1e-5, angles.size)))
    return path


def _bare(formula: str = "Si", density: float = 2.329) -> StructureSpec:
    return StructureSpec(AIR, (), MaterialSpec(formula, formula, density))


def _project_with_structure(tmp_path: Path, structure: StructureSpec):
    value = add_dataset(
        new_project(),
        _source(tmp_path / "curve.xy"),
        InstrumentSpec(instrument_id="structure-service"),
    )
    return set_structure(value, "curve", structure)


def test_validate_and_analyze_structure_use_persisted_beam_and_mask(tmp_path: Path) -> None:
    value = _project_with_structure(tmp_path, simple_structure())

    assert validate_structure(value.datasets[0].structure, BeamSpec("mixed_kalpha")) is None
    evidence = analyze_structure(value, "curve")

    assert isinstance(evidence, StructureEvidence)
    assert evidence.m_model == 1
    assert len(evidence.peak_positions_a) == evidence.m_data


def test_structure_change_reconciles_settings_and_invalidates_derived_state(
    tmp_path: Path,
) -> None:
    value = _project_with_structure(tmp_path, simple_structure())
    dataset = replace(
        value.datasets[0],
        structure_evidence=StructureEvidence(0, 1, None, ()),
        last_valid_result=final_fit_result(),
    )
    value = replace(
        value,
        datasets=(dataset,),
        ui_state=ProjectUiState(
            active_dataset_id="curve",
            selected_candidate_ids=(("curve", "candidate-0"),),
        ),
    )
    changed = replace(
        dataset.structure,
        components=(replace(dataset.structure.components[0], thickness_a=120.0),),
    )

    updated = set_structure(value, "curve", changed)

    assert updated.datasets[0].structure == changed
    assert updated.datasets[0].structure_evidence is None
    assert updated.datasets[0].last_valid_result is None
    assert updated.datasets[0].checkpoint is None
    assert updated.ui_state.selected_candidate_ids == ()


def test_structure_change_reconciles_parameter_priors(tmp_path: Path) -> None:
    value = _project_with_structure(tmp_path, simple_structure())
    definitions = describe_parameters(value, "curve")
    thickness = next(item for item in definitions if item.name == "component.0.thickness_a")
    scale = next(item for item in definitions if item.name == "instrument.scale")
    thickness_prior = ParameterPrior(
        thickness.name,
        PriorSpec("normal", (thickness.initial, 5.0)),
    )
    scale_prior = ParameterPrior(scale.name, PriorSpec("normal", (scale.initial, 0.1)))
    value = set_parameter_priors(value, "curve", (thickness_prior, scale_prior))

    updated = set_structure(value, "curve", _bare())

    assert updated.datasets[0].parameter_priors == (scale_prior,)


def test_structure_change_drops_only_constraints_with_incompatible_parameters(
    tmp_path: Path,
) -> None:
    value = _project_with_structure(tmp_path, simple_structure())
    structural = ConstraintRule(
        ParameterReference("curve", "component.0.density_scale"),
        ConstraintNode(
            "ref",
            reference=ParameterReference("curve", "component.0.thickness_a"),
        ),
    )
    instrument = ConstraintRule(
        ParameterReference("curve", "instrument.scale"),
        ConstraintNode("const", value=1.0),
    )
    value = set_constraint_rules(value, (structural, instrument))

    updated = set_structure(value, "curve", _bare())

    assert updated.constraint_rules == (instrument,)
    definitions = describe_parameters(updated, "curve")
    assert next(item for item in definitions if item.name == "instrument.scale").constrained


def test_structure_change_invalidates_results_for_datasets_in_dropped_constraints(
    tmp_path: Path,
) -> None:
    project = new_project()
    for dataset_id in ("source", "target"):
        project = add_dataset(
            project,
            _source(tmp_path / f"{dataset_id}.xy"),
            InstrumentSpec(instrument_id="structure-service"),
        )
        project = set_structure(project, dataset_id, simple_structure())
    rule = ConstraintRule(
        ParameterReference("target", "instrument.scale"),
        ConstraintNode(
            "ref",
            reference=ParameterReference("source", "component.0.thickness_a"),
        ),
    )
    project = set_constraint_rules(project, (rule,))
    project = replace(
        project,
        datasets=tuple(replace(dataset, last_valid_result=final_fit_result()) for dataset in project.datasets),
        ui_state=replace(
            project.ui_state,
            selected_candidate_ids=(("source", "candidate-0"), ("target", "candidate-0")),
        ),
    )

    updated = set_structure(project, "source", _bare())

    assert updated.constraint_rules == ()
    assert tuple(dataset.last_valid_result for dataset in updated.datasets) == (None, None)
    assert updated.ui_state.selected_candidate_ids == ()


def test_joint_structure_edit_applies_one_topology_to_every_dataset(
    tmp_path: Path,
) -> None:
    project = new_project()
    for dataset_id in ("first", "second"):
        project = add_dataset(
            project,
            _source(tmp_path / f"{dataset_id}.xy"),
            InstrumentSpec(instrument_id="joint-structure"),
        )
        project = set_structure(project, dataset_id, _bare())
    project = set_batch_mode(project, "joint")
    film = LayerSpec("Si3N4", MaterialSpec("Si3N4", "Si3N4", 3.17), 100.0)
    shared_topology = StructureSpec(AIR, (film,), _bare().backing)

    updated = set_structure(project, "first", shared_topology)

    assert tuple(dataset.structure for dataset in updated.datasets) == (
        shared_topology,
        shared_topology,
    )


def test_entering_joint_mode_reuses_the_active_dataset_structure_for_batch(
    tmp_path: Path,
) -> None:
    project = new_project()
    for dataset_id in ("first", "second"):
        project = add_dataset(
            project,
            _source(tmp_path / f"{dataset_id}.xy"),
            InstrumentSpec(instrument_id="joint-structure"),
        )
    template = simple_structure()
    project = set_structure(project, "first", template)
    assert project.datasets[1].structure is None

    updated = set_batch_mode(project, "joint")

    assert tuple(dataset.structure for dataset in updated.datasets) == (
        template,
        template,
    )


def test_bare_si_substrate_triggers_sio2_suggestion() -> None:
    suggestion = suggest_oxide_layers(_bare())[0]

    assert suggestion.base_material == "Si"
    assert suggestion.oxide_material.formula == "SiO2"
    assert suggestion.oxide_material.bulk_density_g_cm3 == 2.20
    assert suggestion.density_locked is True
    assert suggestion.thickness_initial_a == 10.0
    assert suggestion.thickness_bounds_a == (2.0, 50.0)
    assert suggestion.oxide_table_version == OXIDE_TABLE_VERSION
    assert suggestion.location == "backing"


def test_existing_sio2_layer_suppresses_suggestion() -> None:
    silica = MaterialSpec("SiO2", "SiO2", 2.20)
    structure = StructureSpec(AIR, (LayerSpec("native oxide", silica, 5.0),), _bare().backing)

    assert suggest_oxide_layers(structure) == ()


@pytest.mark.parametrize(
    ("base_formula", "oxide_formula", "density"),
    (
        ("Mo", "MoO3", 4.69),
        ("Al", "Al2O3", 3.95),
        ("Ti", "TiO2", 4.23),
        ("Cu", "Cu2O", 6.00),
    ),
)
def test_oxide_table_covers_versioned_materials(
    base_formula: str,
    oxide_formula: str,
    density: float,
) -> None:
    suggestion = suggest_oxide_layers(_bare(base_formula, density))[0]

    assert suggestion.oxide_material.formula == oxide_formula
    assert suggestion.oxide_material.bulk_density_g_cm3 == density
    assert suggestion.oxide_table_version == OXIDE_TABLE_VERSION


def test_oxide_suggestions_cover_surface_and_backing_in_order() -> None:
    silicon = MaterialSpec("Si", "Si", 2.329)
    molybdenum = MaterialSpec("Mo", "Mo", 10.28)
    structure = StructureSpec(
        AIR,
        (LayerSpec("silicon surface", silicon, 8.0),),
        molybdenum,
    )

    suggestions = suggest_oxide_layers(structure)

    assert [(item.location, item.base_material) for item in suggestions] == [
        ("surface", "Si"),
        ("backing", "Mo"),
    ]


def test_oxide_surface_adjacency_suppresses_only_surface_proposal() -> None:
    silica = MaterialSpec("SiO2", "SiO2", 2.20)
    molybdenum = MaterialSpec("Mo", "Mo", 10.28)
    structure = StructureSpec(
        AIR,
        (LayerSpec("native surface oxide", silica, 5.0),),
        molybdenum,
    )

    assert [item.location for item in suggest_oxide_layers(structure)] == ["backing"]


def test_oxide_periodic_block_first_layer_supports_surface_proposal() -> None:
    silicon = MaterialSpec("Si", "Si", 2.329)
    molybdenum = MaterialSpec("Mo", "Mo", 10.28)
    block = PeriodicBlock("Si/Mo", (LayerSpec("Si", silicon, 8.0),), repeats=2)

    suggestions = suggest_oxide_layers(StructureSpec(AIR, (block,), molybdenum))

    assert [(item.location, item.base_material) for item in suggestions] == [
        ("surface", "Si"),
        ("backing", "Mo"),
    ]


def test_oxide_gradient_component_is_not_guessed() -> None:
    gradient = GradientLayerSpec(
        "graded",
        upper_sld_a2=1e-6 + 0.1e-6j,
        lower_sld_a2=2e-6 + 0.1e-6j,
        thickness_a=8.0,
    )

    suggestions = suggest_oxide_layers(StructureSpec(AIR, (gradient,), _bare().backing))

    assert [(item.location, item.base_material) for item in suggestions] == [("backing", "Si")]


def test_oxide_formula_matching_trims_exact_adjacency() -> None:
    silica = MaterialSpec("native", " SiO2 ", 2.20)

    assert suggest_oxide_layers(StructureSpec(AIR, (LayerSpec("native", silica, 5.0),), _bare().backing)) == ()


def test_oxide_matching_ignores_names_and_fuzzy_or_case_variants() -> None:
    named_only = MaterialSpec("Silicon", None, None, 2e-6 + 0.1e-6j)
    lower_case = MaterialSpec("silicon", "si", 2.329)

    assert suggest_oxide_layers(StructureSpec(AIR, (), named_only)) == ()
    assert suggest_oxide_layers(StructureSpec(AIR, (), lower_case)) == ()


def test_oxide_suggestions_are_pure_and_immutable() -> None:
    structure = _bare()
    before_hash = hash(structure)

    suggestion = suggest_oxide_layers(structure)[0]

    assert hash(structure) == before_hash
    with pytest.raises(FrozenInstanceError):
        suggestion.location = "surface"


def test_oxide_table_is_immutable() -> None:
    with pytest.raises(TypeError):
        OXIDE_TABLE["Si"] = ("bad", 0.0)


def test_rejected_oxide_decision_round_trips_in_project(tmp_path: Path) -> None:
    value = _project_with_structure(tmp_path, _bare())
    suggestion = suggest_oxide_layers(value.datasets[0].structure)[0]
    decision = OxideDecision(
        suggestion.base_material,
        suggestion.oxide_material.formula,
        suggestion.location,
        False,
        suggestion.oxide_table_version,
    )
    rejected = record_oxide_decision(value, "curve", decision)
    target = tmp_path / "oxide.xrrproj.json"

    save_project(rejected, target)

    assert load_project(target).datasets[0].oxide_decisions == (decision,)


def test_refusal_preserves_fit_state_and_acceptance_inserts_exact_policy(
    tmp_path: Path,
) -> None:
    value = _project_with_structure(tmp_path, _bare())
    result = final_fit_result()
    value = replace(value, datasets=(replace(value.datasets[0], last_valid_result=result),))
    suggestion = suggest_oxide_layers(value.datasets[0].structure)[0]
    refusal = OxideDecision(
        suggestion.base_material,
        suggestion.oxide_material.formula,
        suggestion.location,
        False,
        suggestion.oxide_table_version,
    )

    refused = record_oxide_decision(value, "curve", refusal)
    accepted = accept_oxide_suggestion(value, "curve", suggestion)

    assert refused.datasets[0].last_valid_result is result
    assert accepted.datasets[0].last_valid_result is None
    inserted = accepted.datasets[0].structure.components[-1]
    assert inserted.material == suggestion.oxide_material
    assert accepted.datasets[0].oxide_decisions[-1].accepted is True
    settings = {item.name: item for item in accepted.datasets[0].parameter_settings}
    assert settings["component.0.thickness_a"].lower == 2.0
    assert settings["component.0.thickness_a"].upper == 50.0
    assert settings["component.0.density_scale"].locked is True
