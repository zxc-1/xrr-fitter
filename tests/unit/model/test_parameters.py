from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from xrr_fitter.model.parameters import (
    ParameterCoordinate,
    ParameterDefinition,
    ParameterReference,
    ParameterSetting,
    ParameterValue,
    SharingRule,
)


def test_parameter_definitions_validate_coordinates_bounds_and_names() -> None:
    definition = ParameterDefinition(
        name="layer.0.thickness_a",
        display_name="Thickness",
        unit="A",
        category="structure",
        initial=20.0,
        lower=2.0,
        upper=200.0,
        transform="linear",
        locked=False,
    )
    coordinate = ParameterCoordinate(0, definition.name, "linear")

    assert coordinate.parameter_index == 0
    with pytest.raises(ValueError, match="name"):
        ParameterDefinition("", "x", "A", "structure", 1.0, 0.0, 2.0, "linear", False)
    with pytest.raises(ValueError, match="bounds"):
        ParameterDefinition("x", "x", "", "fit", 3.0, 0.0, 2.0, "linear", False)


def test_parameter_coordinates_use_declared_r22_transforms() -> None:
    coordinate = ParameterCoordinate(0, "interface.roughness_a", "roughness_fraction")

    assert coordinate.transform == "roughness_fraction"
    with pytest.raises(ValueError, match="unsupported parameter transform"):
        ParameterCoordinate(0, "interface.roughness_a", "logit")


def test_parameter_settings_and_values_are_finite_immutable_values() -> None:
    setting = ParameterSetting("scale", 1.0, 0.5, 1.5)
    value = ParameterValue("scale", 1.1, 0.5, 1.5)

    assert value.value == 1.1
    with pytest.raises(FrozenInstanceError):
        setting.initial = 2.0
    with pytest.raises(ValueError, match="finite"):
        ParameterSetting("scale", float("nan"), 0.5, 1.5)


def test_sharing_rules_copy_members_and_reject_malformed_references() -> None:
    members = [
        ParameterReference("first", "scale"),
        ParameterReference("second", "scale"),
    ]
    rule = SharingRule("shared-scale", members)

    members.clear()

    assert len(rule.members) == 2
    with pytest.raises(ValueError, match="dataset_id"):
        ParameterReference("", "scale")
    with pytest.raises(ValueError, match="at least two"):
        SharingRule("one", (ParameterReference("first", "scale"),))
