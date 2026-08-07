from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from tests.support.model_cases import dataset_project, final_fit_result, project, simple_structure

from xrr_fitter.io.xy import xy_bytes
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import (
    ParameterReference,
    ParameterSetting,
    SharingRule,
)
from xrr_fitter.model.project import ProjectUiState
from xrr_fitter.services.datasets import add_dataset
from xrr_fitter.services.parameters import (
    describe_parameters,
    set_parameter_settings,
    set_sharing_rules,
    validate_parameter_settings,
    validate_sharing_rules,
)
from xrr_fitter.services.projects import new_project
from xrr_fitter.services.structures import set_structure


def _source(path: Path) -> Path:
    angles = np.linspace(0.1, 3.2, 64)
    path.write_bytes(xy_bytes(angles, np.geomspace(1.0, 1e-5, angles.size)))
    return path


def _structured_project(tmp_path: Path):
    value = add_dataset(
        new_project(),
        _source(tmp_path / "curve.xy"),
        InstrumentSpec(instrument_id="parameter-service"),
    )
    return set_structure(value, "curve", simple_structure())


def test_parameter_settings_validate_without_reordering_and_reject_unknown(
    tmp_path: Path,
) -> None:
    value = _structured_project(tmp_path)
    definitions = describe_parameters(value, "curve")
    thickness = next(item for item in definitions if item.name == "component.0.thickness_a")
    setting = ParameterSetting(
        thickness.name,
        thickness.initial,
        thickness.lower,
        thickness.upper,
    )

    assert validate_parameter_settings(definitions, (setting,)) == (setting,)
    with pytest.raises(ValueError, match="unknown parameter setting"):
        validate_parameter_settings(
            definitions,
            (ParameterSetting("unknown", 1.0, 0.0, 2.0),),
        )


def test_set_parameter_settings_persists_and_invalidates_only_fit_state(
    tmp_path: Path,
) -> None:
    value = _structured_project(tmp_path)
    result = final_fit_result()
    dataset = replace(value.datasets[0], last_valid_result=result)
    value = replace(
        value,
        datasets=(dataset,),
        ui_state=ProjectUiState(
            active_dataset_id="curve",
            selected_candidate_ids=(("curve", "candidate-0"),),
        ),
    )
    definition = next(
        item
        for item in describe_parameters(value, "curve")
        if item.name == "component.0.thickness_a"
    )
    setting = ParameterSetting(definition.name, 90.0, 20.0, 180.0)

    updated = set_parameter_settings(value, "curve", (setting,))

    assert updated.datasets[0].parameter_settings == (setting,)
    assert updated.datasets[0].structure is dataset.structure
    assert updated.datasets[0].structure_evidence is dataset.structure_evidence
    assert updated.datasets[0].last_valid_result is None
    assert updated.ui_state.selected_candidate_ids == ()


def test_sharing_validation_is_pure_and_does_not_read_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = project(dataset_project("first"), dataset_project("second"))
    rule = SharingRule(
        "shared-thickness",
        (
            ParameterReference("first", "component.0.thickness_a"),
            ParameterReference("second", "component.0.thickness_a"),
        ),
    )

    def fail_read(_path: Path) -> bytes:
        raise AssertionError("sharing declaration validation read source")

    monkeypatch.setattr(Path, "read_bytes", fail_read)

    assert validate_sharing_rules(value, (rule,)) == (rule,)


def test_sharing_validation_rejects_duplicate_ownership_and_allows_same_dataset() -> None:
    value = project(dataset_project("first"), dataset_project("second"))
    shared = ParameterReference("first", "component.0.thickness_a")
    first = SharingRule(
        "first-rule",
        (shared, ParameterReference("second", "component.0.thickness_a")),
    )
    second = SharingRule(
        "second-rule",
        (shared, ParameterReference("second", "component.0.density_scale")),
    )
    same_dataset = SharingRule(
        "same-dataset",
        (
            ParameterReference("first", "component.0.thickness_a"),
            ParameterReference("first", "component.0.density_scale"),
        ),
    )

    with pytest.raises(ValueError, match="multiple|ownership"):
        validate_sharing_rules(value, (first, second))
    assert validate_sharing_rules(value, (same_dataset,)) == (same_dataset,)


def test_set_sharing_rules_invalidates_affected_fit_state() -> None:
    result = final_fit_result()
    value = project(
        dataset_project("first", result=result),
        dataset_project("second", result=result),
    )
    value = replace(
        value,
        ui_state=ProjectUiState(
            selected_candidate_ids=(
                ("first", "candidate-0"),
                ("second", "candidate-0"),
            )
        ),
    )
    rule = SharingRule(
        "shared-thickness",
        (
            ParameterReference("first", "component.0.thickness_a"),
            ParameterReference("second", "component.0.thickness_a"),
        ),
    )

    updated = set_sharing_rules(value, (rule,))

    assert updated.sharing_rules == (rule,)
    assert all(dataset.last_valid_result is None for dataset in updated.datasets)
    assert updated.ui_state.selected_candidate_ids == ()
