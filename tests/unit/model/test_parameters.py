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


def test_roughness_fraction_prior_center_is_validated_in_unit_fraction() -> None:
    # roughness_fraction 先验在消费侧(evaluation._prior_coordinate)恒落在无量纲单位
    # 分数 [0,1] 上,而非定义里以 Å 计的物理界 [lower, upper]。构造期校验必须用同一
    # 坐标系:一个看似落在 Å 界内(25.0∈[0,50] Å)的中心其实是非法分数(>1),必须被拒,
    # 否则它会在 fit/MCMC 时落到 [0,1] 截断先验之外(先验无质量)而在运行期才炸。
    with pytest.raises(ValueError, match="prior center must be within bounds"):
        ParameterDefinition(
            name="component.0.roughness_a",
            display_name="Roughness",
            unit="Å",
            category="interface",
            initial=10.0,
            lower=0.0,
            upper=50.0,
            transform="roughness_fraction",
            locked=False,
            prior=PriorSpec("normal", (25.0, 1.0)),
        )


def test_roughness_fraction_prior_accepts_fractional_center_below_physical_lower() -> None:
    # 反向:物理 Å 下界 > 1 时,一个合法的分数中心(0.3∈[0,1])不得因 0.3 < 2.0 Å 被误拒
    # ——先验坐标系是单位分数,与 Å 下界无关。
    definition = ParameterDefinition(
        name="component.0.roughness_a",
        display_name="Roughness",
        unit="Å",
        category="interface",
        initial=10.0,
        lower=2.0,
        upper=50.0,
        transform="roughness_fraction",
        locked=False,
        prior=PriorSpec("normal", (0.3, 0.1)),
    )

    assert definition.prior == PriorSpec("normal", (0.3, 0.1))


def test_lognormal_prior_rejects_a_negative_lower_bound() -> None:
    # lognormal 支撑集是 (0, ∞);挂到会取负值的参数(如 sld_real,下界 <0)上时,负半轴
    # 被静默赋零先验概率——用户以为设了先验,实则禁掉了半个物理域。构造期必须直接拒绝,
    # 而不是让 _prior_norm 在运行期对负区密度取 0 后悄悄误归一化。
    with pytest.raises(ValueError, match="lognormal prior requires a positive lower bound"):
        ParameterDefinition(
            name="component.0.sld_real_a2",
            display_name="SLD real",
            unit="Å⁻²",
            category="material",
            initial=0.0,
            lower=-150e-6,
            upper=150e-6,
            transform="linear",
            locked=False,
            prior=PriorSpec("lognormal", (log(50e-6), 0.25)),
        )


def test_lognormal_prior_rejects_a_zero_lower_bound() -> None:
    # The frozen prior contract rejects any declaration whose support may
    # include zero: lognormal density is defined only for strictly positive x.
    with pytest.raises(ValueError, match="lognormal prior requires a positive lower bound"):
        ParameterDefinition(
            name="component.0.roughness_a",
            display_name="Roughness",
            unit="Å",
            category="interface",
            initial=10.0,
            lower=0.0,
            upper=50.0,
            transform="roughness_fraction",
            locked=False,
            prior=PriorSpec("lognormal", (log(0.3), 0.25)),
        )
