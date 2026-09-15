"""Current import declarations, parameter freedom and explicit profile metadata.

Acquisition fields retain their established optional encoding. V2 profile
evidence instead requires every inference field, including an explicit null
when the actual closure threshold is unavailable; no older evidence is inferred.
"""

from __future__ import annotations

from dataclasses import replace
from importlib import import_module

import numpy as np
import pytest
from tests.support.model_cases import dataset_project, final_fit_result, fit_candidate, project

from xrr_fitter.io.project_codec import (
    ProjectSchemaError,
    project_from_bytes,
    project_from_dict,
    project_to_bytes,
    project_to_dict,
)
from xrr_fitter.model.analysis import ParameterProfile, UncertaintyReport
from xrr_fitter.model.parameters import ParameterFreedom, ParameterSetting


def _project_with_profile(threshold: float | None):
    """一份带单条 profile 曲线的工程——``objective_threshold`` 是这里唯一在变的东西。"""
    candidate = fit_candidate()
    profile = ParameterProfile(
        name="scale",
        values=np.array([0.9, 1.0, 1.1]),
        objectives=np.array([1.4, 1.0, 1.4]),
        lower_closed=True,
        upper_closed=True,
        objective_threshold=threshold,
    )
    uncertainty = UncertaintyReport(
        correlation_names=("scale",),
        correlation_matrix=np.eye(1),
        profiles=(profile,),
        bootstrap_intervals=(),
        bootstrap_failure_rate=0.0,
        boundary_hits=(),
        strong_correlations=(),
        systematic_residual=False,
        diagnostics=(),
        candidate_id=candidate.candidate_id,
    )
    result = replace(final_fit_result(candidate), uncertainty=uncertainty)
    return project(dataset_project("sample-1", result=result))


def _restored_profile(payload: dict[str, object]) -> ParameterProfile:
    return project_from_dict(payload).datasets[0].last_valid_result.uncertainty.profiles[0]


def test_codec_omits_the_default_angle_convention_and_round_trips_theta() -> None:
    """默认约定不落键，非默认约定必须往返。"""
    default_document = project_to_dict(project(dataset_project("sample")))
    assert "angle_convention" not in default_document["datasets"][0]

    incident = project(replace(dataset_project("sample"), angle_convention="theta"))
    document = project_to_dict(incident)
    assert document["datasets"][0]["angle_convention"] == "theta"

    restored = project_from_bytes(project_to_bytes(incident))
    assert restored.datasets[0].angle_convention == "theta"
    assert restored == incident


def test_old_project_without_angle_convention_decodes_to_two_theta() -> None:
    payload = project_to_dict(project())
    payload["datasets"][0].pop("angle_convention", None)

    restored = project_from_dict(payload)

    assert restored.datasets[0].angle_convention == "two_theta"


def test_codec_round_trips_the_preset_angle_convention_without_writing_the_default() -> None:
    """Preset 里的约定跟项目一起存，默认值照旧不落键。

    preset 是「下一批数据按什么读」的声明，它必须活过一次保存-打开：约定丢了，同一台仪器
    的下一次导入就悄悄退回按 2θ 读，角度掉回一半。
    """
    from xrr_fitter.model.automation import MeasurementPreset
    from xrr_fitter.model.data import BeamSpec
    from xrr_fitter.model.instrument import InstrumentSpec

    preset = MeasurementPreset(
        "cu-kalpha",
        BeamSpec(kind="monochromatic", wavelength_a=1.5406),
        InstrumentSpec(instrument_id="lab"),
    )
    default_document = project_to_dict(replace(project(), measurement_preset=preset))
    assert "angle_convention" not in default_document["measurement_preset"]

    incident = replace(project(), measurement_preset=replace(preset, angle_convention="theta"))
    document = project_to_dict(incident)
    assert document["measurement_preset"]["angle_convention"] == "theta"

    restored = project_from_bytes(project_to_bytes(incident))
    assert restored.measurement_preset.angle_convention == "theta"
    assert restored == incident

    document["measurement_preset"].pop("angle_convention")
    assert project_from_dict(document).measurement_preset.angle_convention == "two_theta"


def test_profile_roundtrips_the_interval_closure_threshold() -> None:
    """存下来的两个闭合标志，只有跟它比过的那个目标值放在一起才读得懂。"""
    tuned = _project_with_profile(0.225)

    assert _restored_profile(project_to_dict(tuned)).objective_threshold == 0.225


def test_profile_without_a_closure_threshold_encodes_explicit_v2_metadata() -> None:
    """Unavailable thresholds are explicit rather than inferred from older shapes."""
    results = import_module("xrr_fitter.io.codec_results")
    profile = _project_with_profile(None).datasets[0].last_valid_result.uncertainty.profiles[0]

    payload = results._profile_to_dict(profile)

    assert profile.objective_threshold is None
    assert payload["objective_threshold"] is None
    assert set(payload) == {
        "name",
        "values",
        "objectives",
        "lower_closed",
        "upper_closed",
        "interval_kind",
        "confidence_level",
        "method",
        "unavailable_reason",
        "delta_total",
        "objective_point_count",
        "objective_threshold",
    }


def test_profile_without_the_threshold_key_is_incomplete_v2_evidence() -> None:
    """Current evidence must declare whether its closure threshold is available."""
    payload = project_to_dict(_project_with_profile(0.225))
    for dataset in payload["datasets"]:
        result = dataset["last_valid_result"]
        if result is not None and result["uncertainty"] is not None:
            for profile in result["uncertainty"]["profiles"]:
                assert profile.pop("objective_threshold") == 0.225

    with pytest.raises(ProjectSchemaError, match="objective_threshold"):
        _restored_profile(payload)


def _project_with_setting(freedom: ParameterFreedom):
    """一份带单条 setting 的工程——``freedom`` 是这里唯一在变的东西。"""
    setting = ParameterSetting("instrument.scale", 1.0, 0.5, 2.0, freedom)
    return project(replace(dataset_project("sample-1"), parameter_settings=(setting,)))


def _restored_setting(payload: dict[str, object]) -> ParameterSetting:
    return project_from_dict(payload).datasets[0].parameter_settings[0]


def test_codec_omits_the_free_freedom_and_round_trips_the_other_two() -> None:
    """FREE 的规范编码省略默认键，FIXED / RANGE_ONLY 必须保留声明并完整往返。"""
    free = _project_with_setting(ParameterFreedom.FREE)
    default_document = project_to_dict(free)
    assert "freedom" not in default_document["datasets"][0]["parameter_settings"][0]
    assert project_from_bytes(project_to_bytes(free)) == free

    default_document["datasets"][0]["parameter_settings"][0]["freedom"] = "free"
    assert _restored_setting(default_document).freedom is ParameterFreedom.FREE

    for freedom in (ParameterFreedom.FIXED, ParameterFreedom.RANGE_ONLY):
        pinned = _project_with_setting(freedom)
        document = project_to_dict(pinned)

        assert document["datasets"][0]["parameter_settings"][0]["freedom"] == freedom.value
        restored = project_from_bytes(project_to_bytes(pinned))
        assert restored.datasets[0].parameter_settings[0].freedom is freedom
        assert restored == pinned


@pytest.mark.parametrize("locked", [False, True])
def test_setting_rejects_locked_in_current_schema(locked: bool) -> None:
    payload = project_to_dict(_project_with_setting(ParameterFreedom.FREE))
    stored = payload["datasets"][0]["parameter_settings"][0]
    assert "freedom" not in stored
    assert _restored_setting(payload).freedom is ParameterFreedom.FREE

    stored["locked"] = locked

    with pytest.raises(ProjectSchemaError, match=r"extra=\['locked'\]"):
        project_from_dict(payload)


@pytest.mark.parametrize("locked", [False, True])
@pytest.mark.parametrize("freedom", list(ParameterFreedom))
def test_a_setting_carrying_both_keys_is_refused_rather_than_silently_picking_one(
    freedom: ParameterFreedom, locked: bool
) -> None:
    payload = project_to_dict(_project_with_setting(freedom))
    stored = payload["datasets"][0]["parameter_settings"][0]
    stored["freedom"] = freedom.value
    stored["locked"] = locked

    with pytest.raises(ProjectSchemaError, match=r"extra=\['locked'\]"):
        project_from_dict(payload)


@pytest.mark.parametrize("freedom", [False, 1, 1.0, [], {}])
def test_a_non_string_freedom_is_refused(freedom: object) -> None:
    payload = project_to_dict(_project_with_setting(ParameterFreedom.FIXED))
    payload["datasets"][0]["parameter_settings"][0]["freedom"] = freedom

    with pytest.raises(ProjectSchemaError, match="parameter setting freedom must be a string"):
        project_from_dict(payload)


def test_an_unknown_freedom_names_itself_instead_of_falling_back() -> None:
    """认不出来的档位要报出那个词。落回「自由」会让新版写的文件在旧版里悄悄放开一个量。"""
    payload = project_to_dict(_project_with_setting(ParameterFreedom.FIXED))
    payload["datasets"][0]["parameter_settings"][0]["freedom"] = "soft"

    with pytest.raises(ProjectSchemaError, match="unsupported parameter setting freedom"):
        project_from_dict(payload)
