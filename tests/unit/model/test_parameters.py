from __future__ import annotations

from dataclasses import FrozenInstanceError
from math import log

import numpy as np
import pytest

from xrr_fitter.model.parameters import (
    PRIOR_KINDS,
    ParameterCoordinate,
    ParameterDefinition,
    ParameterReference,
    ParameterSetting,
    ParameterValue,
    PriorSpec,
    SharingRule,
    unit_to_physical,
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


def test_log_unit_interior_roundoff_stays_within_physical_bounds() -> None:
    definition = ParameterDefinition(
        name="component.0.thickness_a",
        display_name="Thickness",
        unit="A",
        category="structure",
        initial=180.0,
        lower=58.46351284627307,
        upper=69454.62186224229,
        transform="log",
        locked=False,
    )

    value = unit_to_physical(definition, np.nextafter(0.0, 1.0))

    assert definition.lower <= value <= definition.upper


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


def test_prior_spec_accepts_each_kind_with_correct_arity() -> None:
    assert PRIOR_KINDS == frozenset({"uniform", "normal", "lognormal", "soft_range"})

    uniform = PriorSpec("uniform")
    normal = PriorSpec("normal", [20.0, 2.0])
    lognormal = PriorSpec("lognormal", (log(20.0), 0.25))
    soft_range = PriorSpec("soft_range", (10.0, 30.0, 1.5))

    assert uniform.parameters == ()
    assert normal.parameters == (20.0, 2.0)
    assert lognormal.parameters == (log(20.0), 0.25)
    assert soft_range.parameters == (10.0, 30.0, 1.5)
    with pytest.raises(FrozenInstanceError):
        normal.kind = "uniform"


def test_prior_spec_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unsupported prior kind: gaussian"):
        PriorSpec("gaussian", (20.0, 2.0))


def test_prior_spec_rejects_wrong_arity() -> None:
    with pytest.raises(ValueError, match="normal prior requires 2 parameters"):
        PriorSpec("normal", (20.0,))
    with pytest.raises(ValueError, match="uniform prior requires 0 parameters"):
        PriorSpec("uniform", (20.0,))
    with pytest.raises(ValueError, match="soft_range prior requires 3 parameters"):
        PriorSpec("soft_range", (10.0, 30.0))


def test_prior_spec_rejects_nonfinite_parameters() -> None:
    with pytest.raises(ValueError, match="prior parameters must be finite"):
        PriorSpec("normal", (float("nan"), 2.0))
    with pytest.raises(ValueError, match="prior parameters must be finite"):
        PriorSpec("normal", (20.0, float("inf")))


def test_prior_spec_rejects_nonpositive_scale() -> None:
    with pytest.raises(ValueError, match="normal prior scale must be positive"):
        PriorSpec("normal", (20.0, 0.0))
    with pytest.raises(ValueError, match="lognormal prior scale must be positive"):
        PriorSpec("lognormal", (log(20.0), -0.25))
    with pytest.raises(ValueError, match="soft_range prior scale must be positive"):
        PriorSpec("soft_range", (10.0, 30.0, 0.0))
    with pytest.raises(ValueError, match="soft_range prior requires low < high"):
        PriorSpec("soft_range", (30.0, 10.0, 1.5))


def test_parameter_definition_defaults_prior_to_none() -> None:
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

    assert definition.prior is None


def test_parameter_definition_accepts_a_prior() -> None:
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
        prior=PriorSpec("normal", (20.0, 2.0)),
    )

    assert definition.prior == PriorSpec("normal", (20.0, 2.0))


def test_parameter_definition_rejects_prior_center_outside_bounds() -> None:
    with pytest.raises(ValueError, match="prior center must be within bounds"):
        ParameterDefinition(
            name="layer.0.thickness_a",
            display_name="Thickness",
            unit="A",
            category="structure",
            initial=20.0,
            lower=2.0,
            upper=200.0,
            transform="linear",
            locked=False,
            prior=PriorSpec("normal", (500.0, 2.0)),
        )
