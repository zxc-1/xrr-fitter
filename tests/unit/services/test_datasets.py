from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from tests.support.model_cases import final_fit_result, simple_structure

import xrr_fitter.api as api
from xrr_fitter.io.xy import xy_bytes
from xrr_fitter.model.analysis import StructureEvidence
from xrr_fitter.model.automation import (
    AutomaticRole,
    AutomaticStatus,
    DatasetAutomation,
)
from xrr_fitter.model.data import BeamSpec
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import (
    RESERVED_DATASET_ID,
    ConstraintNode,
    ConstraintRule,
    ParameterPrior,
    ParameterReference,
    ParameterSetting,
    PriorSpec,
    SharingRule,
)
from xrr_fitter.model.project import ProjectUiState, ScalePriorState
from xrr_fitter.services.datasets import (
    _dataset_id as _allocate_dataset_id,
)
from xrr_fitter.services.datasets import (
    add_dataset,
    preview_source_update,
    remove_dataset,
    set_fit_mask,
    set_instrument,
)
from xrr_fitter.services.parameters import (
    accept_source_update,
    describe_parameters,
)
from xrr_fitter.services.projects import new_project
from xrr_fitter.services.structures import set_structure

# These service contracts intentionally exercise complete immutable project
# transitions rather than isolated setters.  Each scenario builds real source
# bytes, changes one declaration, and checks the exact result graph that must be
# retained or invalidated.  Keeping that end-to-end evidence together makes
# data-derived bounds, sidecars, automatic-fit groups, and UI selection state
# observable at the same public boundary.


def _write_curve(path: Path, *, scale: float = 1.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    angles = np.linspace(0.1, 3.2, 32)
    intensity = scale * np.geomspace(1.0, 1e-4, angles.size)
    path.write_bytes(xy_bytes(angles, intensity))
    return path


def _instrument() -> InstrumentSpec:
    return InstrumentSpec(instrument_id="service-test")


def _with_automatic_results(project, groups: tuple[str, ...]):
    return replace(
        project,
        datasets=tuple(
            replace(
                dataset,
                last_valid_result=final_fit_result(),
                automation=DatasetAutomation(
                    import_batch_id="batch-1",
                    fit_group_id=group_id,
                    role=AutomaticRole.JOINT,
                    status=AutomaticStatus.PASSED,
                    statistics_member=True,
                    reason="previous result",
                ),
            )
            for dataset, group_id in zip(project.datasets, groups, strict=True)
        ),
    )


def _filename_structure_snapshot(dataset) -> tuple[object, ...]:
    structure = dataset.structure
    assert structure is not None
    materials = tuple(layer.material for layer in structure.components)
    return (
        dataset.dataset_id,
        dataset.display_name,
        structure.fronting.name,
        structure.backing.formula,
        tuple(material.formula for material in materials),
        materials[0].bulk_density_g_cm3,
        tuple(material.bulk_density_g_cm3 is not None for material in materials),
        tuple(material.sld_override_a2 is not None for material in materials),
    )


def test_dataset_id_allocator_rejects_reserved_drift_namespace() -> None:
    with pytest.raises(ValueError, match="reserved dataset_id"):
        _allocate_dataset_id(new_project(), RESERVED_DATASET_ID)


def test_add_dataset_uses_source_stem_namespace_and_lowest_available_suffix(
    tmp_path: Path,
) -> None:
    project = new_project()
    sources = tuple(
        _write_curve(tmp_path / directory / "sample.xy", scale=index + 1.0)
        for index, directory in enumerate(("a", "b", "c"))
    )
    names = ("first display", None, "third display")
    for source, display_name in zip(sources, names, strict=True):
        project = add_dataset(
            project,
            source,
            _instrument(),
            display_name=display_name,
        )

    assert (
        tuple(item.dataset_id for item in project.datasets),
        tuple(item.display_name for item in project.datasets),
    ) == (
        ("sample", "sample-2", "sample-3"),
        ("first display", "sample", "third display"),
    )

    project = remove_dataset(project, "sample-2")
    replacement = _write_curve(tmp_path / "d" / "sample.xy", scale=4.0)
    project = add_dataset(
        project,
        replacement,
        _instrument(),
        display_name="replacement display",
    )

    assert (
        tuple(item.dataset_id for item in project.datasets),
        project.datasets[-1].display_name,
    ) == (("sample", "sample-3", "sample-2"), "replacement display")


def test_remove_dataset_drops_constraints_that_reference_it_and_invalidates_remaining_target(
    tmp_path: Path,
) -> None:
    first_source = _write_curve(tmp_path / "first" / "first.xy")
    second_source = _write_curve(tmp_path / "second" / "second.xy", scale=2.0)
    project = new_project()
    project = add_dataset(project, first_source, _instrument())
    project = add_dataset(project, second_source, _instrument())
    first_id, second_id = (dataset.dataset_id for dataset in project.datasets)
    target = ParameterReference(first_id, "component.0.thickness_a")
    source = ParameterReference(second_id, "instrument.scale")
    rule = ConstraintRule(target, ConstraintNode("ref", reference=source))
    datasets = tuple(
        replace(
            dataset,
            structure=simple_structure(),
            last_valid_result=final_fit_result() if dataset.dataset_id == first_id else None,
        )
        for dataset in project.datasets
    )
    project = replace(
        project,
        datasets=datasets,
        constraint_rules=(rule,),
        ui_state=replace(
            project.ui_state,
            selected_candidate_ids=((first_id, "candidate-0"),),
        ),
    )

    updated = remove_dataset(project, second_id)

    assert updated.constraint_rules == ()
    assert len(updated.datasets) == 1
    assert updated.datasets[0].last_valid_result is None
    assert updated.ui_state.selected_candidate_ids == ()


def test_add_dataset_preserves_an_explicit_mixed_kalpha_beam(tmp_path: Path) -> None:
    source = _write_curve(tmp_path / "mixed.xy")
    beam = BeamSpec(
        kind="mixed_kalpha",
        wavelength_1_a=1.54056,
        wavelength_2_a=1.54439,
        intensity_ratio_21=0.5,
    )

    updated = add_dataset(
        new_project(),
        source,
        _instrument(),
        beam=beam,
    )

    assert updated.datasets[0].beam is beam


def test_add_dataset_interprets_filename_materials_from_backing_to_surface(
    tmp_path: Path,
) -> None:
    cases = (
        (
            "S300-1_250904-2 Si3N4+Si+Zr",
            "S300-1_250904-2",
            ("Zr", "Si", "Si3N4"),
        ),
        (
            "S300-1-260424-2 CrSiC+SiCMo+TaN",
            "S300-1-260424-2",
            ("TaN", "SiCMo", "CrSiC"),
        ),
    )
    project = new_project()

    for index, (stem, _dataset_id, _formulas) in enumerate(cases):
        project = add_dataset(
            project,
            _write_curve(tmp_path / f"{stem}.xy", scale=index + 1.0),
            _instrument(),
        )

    assert tuple(_filename_structure_snapshot(dataset) for dataset in project.datasets) == (
        (
            cases[0][1],
            cases[0][0],
            "Air",
            "Si",
            ("Zr", "Si", "Si3N4"),
            6.52,
            (True, True, True),
            (False, False, False),
        ),
        (
            cases[1][1],
            cases[1][0],
            "Air",
            "Si",
            ("TaN", None, None),
            14.30,
            (True, False, False),
            (False, True, True),
        ),
    )


def test_add_dataset_reuses_matching_filename_structure_for_batch(
    tmp_path: Path,
) -> None:
    project = new_project()
    for index, sample in enumerate(("S300-1", "S300-2")):
        project = add_dataset(
            project,
            _write_curve(
                tmp_path / f"{sample} Si3N4+Si+Zr.xy",
                scale=index + 1.0,
            ),
            _instrument(),
        )

    assert project.datasets[0].structure is not None
    assert project.datasets[1].structure is project.datasets[0].structure


def test_add_dataset_reuses_matching_direct_sld_filename_structure_for_batch(
    tmp_path: Path,
) -> None:
    project = new_project()
    for index, sample in enumerate(("S300-1", "S300-2")):
        project = add_dataset(
            project,
            _write_curve(
                tmp_path / f"{sample} CrSiC+SiCMo+TaN.xy",
                scale=index + 1.0,
            ),
            _instrument(),
        )

    assert project.datasets[0].structure is not None
    assert project.datasets[1].structure is project.datasets[0].structure


def test_fit_mask_change_clears_only_derived_state_and_candidate_selection(
    tmp_path: Path,
) -> None:
    source = _write_curve(tmp_path / "curve.xy")
    project = add_dataset(new_project(), source, _instrument())
    result = final_fit_result()
    dataset = replace(
        project.datasets[0],
        structure_evidence=StructureEvidence(1, 1, None, (20.0,)),
        scale_prior=ScalePriorState(enabled=True, s_hat=1.0, tau_s_decades=0.1),
        last_valid_result=result,
    )
    project = replace(
        project,
        datasets=(dataset,),
        ui_state=ProjectUiState(
            active_dataset_id=dataset.dataset_id,
            selected_candidate_ids=((dataset.dataset_id, result.candidates[0].candidate_id),),
        ),
    )
    mask = np.ones(len(dataset.fit_mask), dtype=bool)
    mask[-1] = False

    updated = set_fit_mask(project, dataset.dataset_id, mask)
    changed = updated.datasets[0]

    assert (
        changed.source_path,
        changed.source_sha256,
        changed.structure is dataset.structure,
        changed.instrument is dataset.instrument,
        changed.fit_mask[-1],
        changed.structure_evidence,
        changed.scale_prior,
        changed.last_valid_result,
        changed.checkpoint,
        updated.ui_state.active_dataset_id,
        updated.ui_state.selected_candidate_ids,
    ) == (
        dataset.source_path,
        dataset.source_sha256,
        True,
        True,
        False,
        None,
        ScalePriorState(enabled=False),
        None,
        None,
        dataset.dataset_id,
        (),
    )


def test_automatic_fit_mask_change_clears_the_matching_fit_group(
    tmp_path: Path,
) -> None:
    project = new_project()
    for name, scale in (("a", 1.0), ("b", 2.0), ("c", 3.0)):
        project = add_dataset(
            project,
            _write_curve(tmp_path / f"{name}.xy", scale=scale),
            _instrument(),
        )
    project = _with_automatic_results(project, ("g1", "g1", "g2"))
    mask = np.asarray(project.datasets[0].fit_mask, dtype=bool)
    mask[-1] = False

    changed = set_fit_mask(project, "a", mask)

    by_id = {dataset.dataset_id: dataset for dataset in changed.datasets}
    assert by_id["a"].last_valid_result is None
    assert by_id["b"].last_valid_result is None
    assert by_id["c"].last_valid_result is not None
    assert by_id["b"].automation == replace(
        project.datasets[1].automation,
        status=AutomaticStatus.PENDING,
        statistics_member=False,
        reason=None,
    )


def test_fit_mask_change_reconciles_data_derived_parameter_priors(
    tmp_path: Path,
) -> None:
    project = add_dataset(
        new_project(),
        _write_curve(tmp_path / "mask-priors.xy"),
        _instrument(),
    )
    dataset = replace(project.datasets[0], structure=simple_structure())
    project = replace(project, datasets=(dataset,))
    definitions = {definition.name: definition for definition in describe_parameters(project, dataset.dataset_id)}
    thickness = definitions["component.0.thickness_a"]
    scale = definitions["instrument.scale"]
    thickness_prior = ParameterPrior(
        thickness.name,
        PriorSpec("normal", (thickness.lower, max(thickness.lower / 10.0, 1e-6))),
    )
    scale_prior = ParameterPrior(
        scale.name,
        PriorSpec("normal", (scale.initial, 0.1)),
    )
    project = replace(
        project,
        datasets=(
            replace(
                dataset,
                parameter_priors=(thickness_prior, scale_prior),
            ),
        ),
    )
    mask = np.asarray(dataset.fit_mask, dtype=bool)
    mask[-2:] = False
    new_lower = thickness.lower
    for cutoff in range(mask.size - 2, 29, -1):
        candidate = np.zeros_like(mask)
        candidate[:cutoff] = True
        probe = api.set_fit_mask(project, dataset.dataset_id, candidate)
        probe_thickness = next(
            definition
            for definition in describe_parameters(probe, dataset.dataset_id)
            if definition.name == thickness.name
        )
        mask = candidate
        new_lower = probe_thickness.lower
        if new_lower > thickness.lower:
            break

    assert new_lower > thickness.lower

    updated = api.set_fit_mask(project, dataset.dataset_id, mask)

    new_definitions = {definition.name: definition for definition in describe_parameters(updated, dataset.dataset_id)}
    assert new_definitions[thickness.name].lower == new_lower
    assert updated.datasets[0].parameter_priors == (scale_prior,)


def test_source_update_is_previewed_then_accepted_with_fresh_declarations(
    tmp_path: Path,
) -> None:
    original = _write_curve(tmp_path / "original.xy")
    project = add_dataset(new_project(), original, _instrument())
    dataset = replace(project.datasets[0], last_valid_result=final_fit_result())
    project = replace(project, datasets=(dataset,))
    replacement = _write_curve(tmp_path / "replacement.xy", scale=2.0)

    preview = preview_source_update(project, dataset.dataset_id, replacement)

    assert (
        preview.dataset_id,
        preview.current_source_path,
        preview.proposed_source_path,
        preview.expected_sha256,
        preview.observed_sha256 != dataset.source_sha256,
        preview.changed,
        project.datasets[0] is dataset,
    ) == (
        dataset.dataset_id,
        str(original),
        str(replacement),
        dataset.source_sha256,
        True,
        True,
        True,
    )

    updated = accept_source_update(project, preview)
    accepted = updated.datasets[0]

    assert (
        accepted.dataset_id,
        accepted.display_name,
        accepted.source_path,
        accepted.source_sha256,
        accepted.beam,
        accepted.instrument,
        accepted.structure,
        accepted.last_valid_result,
        accepted.checkpoint,
    ) == (
        dataset.dataset_id,
        dataset.display_name,
        str(replacement),
        preview.observed_sha256,
        dataset.beam,
        dataset.instrument,
        dataset.structure,
        None,
        None,
    )


def test_automatic_source_acceptance_clears_the_matching_fit_group(
    tmp_path: Path,
) -> None:
    value = new_project()
    sources = (
        _write_curve(tmp_path / "a.xy"),
        _write_curve(tmp_path / "b.xy", scale=2.0),
        _write_curve(tmp_path / "c.xy", scale=3.0),
    )
    for source in sources:
        value = add_dataset(value, source, _instrument())
    value = _with_automatic_results(value, ("g1", "g1", "g2"))
    sources[0].write_bytes(sources[0].read_bytes() + b"# changed\n")

    changed = accept_source_update(value, preview_source_update(value, "a"))

    by_id = {dataset.dataset_id: dataset for dataset in changed.datasets}
    assert by_id["a"].last_valid_result is None
    assert by_id["b"].last_valid_result is None
    assert by_id["c"].last_valid_result is not None
    assert by_id["a"].automation == replace(
        value.datasets[0].automation,
        status=AutomaticStatus.PENDING,
        statistics_member=False,
        reason=None,
    )


def test_automatic_invalidation_preserves_manual_dataset_with_matching_group_id(
    tmp_path: Path,
) -> None:
    value = new_project()
    for name, scale in (("auto", 1.0), ("manual", 2.0)):
        value = add_dataset(
            value,
            _write_curve(tmp_path / f"{name}.xy", scale=scale),
            _instrument(),
        )
    value = _with_automatic_results(value, ("shared", "shared"))
    manual = replace(
        value.datasets[1],
        automation=DatasetAutomation(fit_group_id="shared"),
    )
    value = replace(value, datasets=(value.datasets[0], manual))

    changed = set_instrument(
        value,
        "auto",
        replace(value.datasets[0].instrument, background_kind="linear"),
    )

    assert changed.datasets[0].last_valid_result is None
    assert changed.datasets[1] is manual


def test_automatic_structure_change_clears_the_matching_fit_group(
    tmp_path: Path,
) -> None:
    project = new_project()
    for name, scale in (("a", 1.0), ("b", 2.0), ("c", 3.0)):
        project = add_dataset(
            project,
            _write_curve(tmp_path / f"{name}.xy", scale=scale),
            _instrument(),
        )
    project = _with_automatic_results(project, ("g1", "g1", "g2"))
    structure = replace(simple_structure(), backing_roughness_a=4.0)

    changed = set_structure(project, "a", structure)

    by_id = {dataset.dataset_id: dataset for dataset in changed.datasets}
    assert by_id["a"].structure == structure
    assert by_id["a"].last_valid_result is None
    assert by_id["b"].last_valid_result is None
    assert by_id["c"].last_valid_result is not None


def test_source_update_retains_only_parameter_sidecars_valid_for_current_definitions(
    tmp_path: Path,
) -> None:
    source = _write_curve(tmp_path / "reconciled.xy")
    project = add_dataset(new_project(), source, _instrument())
    dataset = replace(project.datasets[0], structure=simple_structure())
    project = replace(project, datasets=(dataset,))
    definitions = {definition.name: definition for definition in describe_parameters(project, dataset.dataset_id)}
    scale = definitions["instrument.scale"]
    compatible = ParameterSetting(
        scale.name,
        scale.initial,
        scale.lower,
        scale.upper,
        scale.locked,
    )
    missing = ParameterSetting(
        "component.99.thickness_a",
        10.0,
        2.0,
        20.0,
    )
    compatible_prior = ParameterPrior(
        scale.name,
        PriorSpec("normal", (scale.initial, 0.1)),
    )
    missing_prior = ParameterPrior(
        "component.99.thickness_a",
        PriorSpec("uniform"),
    )
    project = replace(
        project,
        datasets=(
            replace(
                dataset,
                parameter_settings=(compatible, missing),
                parameter_priors=(compatible_prior, missing_prior),
            ),
        ),
    )
    source.write_bytes(source.read_bytes() + b"# accepted replacement\n")

    preview = preview_source_update(project, dataset.dataset_id)
    updated = accept_source_update(project, preview)

    assert updated.datasets[0].parameter_settings == (compatible,)
    assert updated.datasets[0].parameter_priors == (compatible_prior,)


def test_instrument_change_preserves_source_and_structure_but_invalidates_fit_state(
    tmp_path: Path,
) -> None:
    project = add_dataset(
        new_project(),
        _write_curve(tmp_path / "instrument.xy"),
        _instrument(),
    )
    result = final_fit_result()
    evidence = StructureEvidence(1, 1, None, (20.0,))
    dataset = replace(
        project.datasets[0],
        structure_evidence=evidence,
        scale_prior=ScalePriorState(enabled=True, s_hat=1.0, tau_s_decades=0.1),
        last_valid_result=result,
    )
    project = replace(
        project,
        datasets=(dataset,),
        ui_state=replace(
            project.ui_state,
            selected_candidate_ids=((dataset.dataset_id, "candidate-0"),),
        ),
    )
    instrument = replace(dataset.instrument, background_kind="linear")

    updated = set_instrument(project, dataset.dataset_id, instrument)
    changed = updated.datasets[0]

    assert (
        changed.source_path,
        changed.source_sha256,
        changed.structure is dataset.structure,
        changed.structure_evidence is evidence,
        changed.instrument is instrument,
        changed.scale_prior,
        changed.last_valid_result,
        changed.checkpoint,
        updated.ui_state.selected_candidate_ids,
        set_instrument(updated, dataset.dataset_id, instrument) is updated,
    ) == (
        dataset.source_path,
        dataset.source_sha256,
        True,
        True,
        True,
        ScalePriorState(enabled=False),
        None,
        None,
        (),
        True,
    )


def test_instrument_change_reconciles_parameter_sidecars(tmp_path: Path) -> None:
    project = add_dataset(
        new_project(),
        _write_curve(tmp_path / "instrument-priors.xy"),
        replace(_instrument(), footprint_mode="fit"),
    )
    dataset = replace(project.datasets[0], structure=simple_structure())
    project = replace(project, datasets=(dataset,))
    definitions = {definition.name: definition for definition in describe_parameters(project, dataset.dataset_id)}
    scale = definitions["instrument.scale"]
    footprint = definitions["instrument.footprint_spill_angle_deg"]
    scale_setting = ParameterSetting(
        scale.name,
        scale.initial,
        scale.lower,
        scale.upper,
        scale.locked,
    )
    footprint_setting = ParameterSetting(
        footprint.name,
        footprint.initial,
        footprint.lower,
        footprint.upper,
        footprint.locked,
    )
    scale_prior = ParameterPrior(
        scale.name,
        PriorSpec("normal", (scale.initial, 0.1)),
    )
    footprint_prior = ParameterPrior(
        footprint.name,
        PriorSpec("normal", (footprint.initial, 0.1)),
    )
    project = replace(
        project,
        datasets=(
            replace(
                dataset,
                parameter_settings=(scale_setting, footprint_setting),
                parameter_priors=(scale_prior, footprint_prior),
            ),
        ),
    )

    updated = api.set_instrument(
        project,
        dataset.dataset_id,
        replace(dataset.instrument, footprint_mode="none"),
    )

    assert updated.datasets[0].parameter_settings == (scale_setting,)
    assert updated.datasets[0].parameter_priors == (scale_prior,)


def test_instrument_domain_change_drops_sharing_rule_for_locked_coordinate(
    tmp_path: Path,
) -> None:
    project = new_project()
    instrument = InstrumentSpec(instrument_id="shared-resolution", resolution_domain="q")
    for dataset_id in ("first", "second"):
        project = add_dataset(
            project,
            _write_curve(tmp_path / f"{dataset_id}.xy"),
            instrument,
        )
        project = set_structure(project, dataset_id, simple_structure())
    name = "instrument.relative_sigma"
    rule = SharingRule(
        "shared-resolution",
        (
            ParameterReference("first", name),
            ParameterReference("second", name),
        ),
    )
    project = api.set_sharing_rules(project, (rule,))

    updated = api.set_instrument(
        project,
        "first",
        replace(instrument, resolution_domain="theta"),
    )

    assert updated.sharing_rules == ()


def test_source_update_drops_sharing_when_data_derived_bounds_diverge(
    tmp_path: Path,
) -> None:
    project = new_project()
    for dataset_id in ("first", "second"):
        project = add_dataset(
            project,
            _write_curve(tmp_path / f"{dataset_id}.xy"),
            _instrument(),
        )
        project = set_structure(project, dataset_id, simple_structure())
    name = "component.0.thickness_a"
    project = api.set_sharing_rules(
        project,
        (
            SharingRule(
                "shared-thickness-source-update",
                (
                    ParameterReference("first", name),
                    ParameterReference("second", name),
                ),
            ),
        ),
    )

    replacement = tmp_path / "first-replacement.xy"
    replacement.write_bytes(
        xy_bytes(
            np.linspace(0.1, 60.0, 32),
            np.geomspace(1.0, 1e-4, 32),
        )
    )
    updated = accept_source_update(
        project,
        preview_source_update(project, "first", replacement),
    )

    assert updated.sharing_rules == ()


def test_source_update_drops_constraint_when_new_bounds_reject_target(
    tmp_path: Path,
) -> None:
    project = add_dataset(new_project(), _write_curve(tmp_path / "curve.xy"), _instrument())
    project = set_structure(project, "curve", simple_structure())
    rule = ConstraintRule(
        ParameterReference("curve", "component.0.thickness_a"),
        ConstraintNode("const", value=200.0),
    )
    project = api.set_constraint_rules(project, (rule,))

    replacement = tmp_path / "curve-replacement.xy"
    replacement.write_bytes(
        xy_bytes(
            np.linspace(0.1, 60.0, 32),
            np.geomspace(1.0, 1e-4, 32),
        )
    )
    updated = accept_source_update(
        project,
        preview_source_update(project, "curve", replacement),
    )

    assert updated.constraint_rules == ()
