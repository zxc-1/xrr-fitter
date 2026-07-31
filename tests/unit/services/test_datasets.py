from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from tests.support.model_cases import final_fit_result, simple_structure
from xrr_fitter.io.xy import xy_bytes
from xrr_fitter.model.analysis import StructureEvidence
from xrr_fitter.model.data import BeamSpec
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import ParameterSetting
from xrr_fitter.model.project import ProjectUiState, ScalePriorState
from xrr_fitter.services.datasets import (
    add_dataset,
    preview_source_update,
    remove_dataset,
    set_fit_mask,
    set_instrument,
)
from xrr_fitter.services.parameters import accept_source_update, describe_parameters
from xrr_fitter.services.projects import new_project


def _write_curve(path: Path, *, scale: float = 1.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    angles = np.linspace(0.1, 3.2, 32)
    intensity = scale * np.geomspace(1.0, 1e-4, angles.size)
    path.write_bytes(xy_bytes(angles, intensity))
    return path


def _instrument() -> InstrumentSpec:
    return InstrumentSpec(instrument_id="service-test")


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


def test_add_dataset_builds_structure_from_material_suffix_in_source_name(
    tmp_path: Path,
) -> None:
    cases = (
        (
            "S300-1_250904-2 Si3N4+Si+Zr",
            "S300-1_250904-2",
            ("Si3N4", "Si", "Zr"),
        ),
        (
            "S300-1-260424-2 CrSiC+SiCMo+TaN",
            "S300-1-260424-2",
            ("CrSiC", "SiCMo", "TaN"),
        ),
    )
    project = new_project()

    for index, (stem, _dataset_id, _formulas) in enumerate(cases):
        project = add_dataset(
            project,
            _write_curve(tmp_path / f"{stem}.xy", scale=index + 1.0),
            _instrument(),
        )

    assert tuple(dataset.dataset_id for dataset in project.datasets) == tuple(
        case[1] for case in cases
    )
    assert tuple(dataset.display_name for dataset in project.datasets) == tuple(
        case[0] for case in cases
    )
    for dataset, (_stem, _dataset_id, formulas) in zip(
        project.datasets,
        cases,
        strict=True,
    ):
        assert dataset.structure is not None
        assert dataset.structure.fronting.name == "Air"
        assert dataset.structure.backing.formula == "Si"
        assert tuple(
            component.material.formula for component in dataset.structure.components
        ) == formulas
        assert all(
            component.material.bulk_density_g_cm3 > 0.0
            for component in dataset.structure.components
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


def test_source_update_retains_only_parameter_settings_valid_for_current_definitions(
    tmp_path: Path,
) -> None:
    source = _write_curve(tmp_path / "reconciled.xy")
    project = add_dataset(new_project(), source, _instrument())
    dataset = replace(project.datasets[0], structure=simple_structure())
    project = replace(project, datasets=(dataset,))
    definitions = {
        definition.name: definition
        for definition in describe_parameters(project, dataset.dataset_id)
    }
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
    project = replace(
        project,
        datasets=(
            replace(
                dataset,
                parameter_settings=(compatible, missing),
            ),
        ),
    )
    source.write_bytes(source.read_bytes() + b"# accepted replacement\n")

    preview = preview_source_update(project, dataset.dataset_id)
    updated = accept_source_update(project, preview)

    assert updated.datasets[0].parameter_settings == (compatible,)


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
