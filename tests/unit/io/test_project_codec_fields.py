"""新增可选字段过 codec 的三条规矩：默认不落键、非默认往返、老文件缺键落回默认。

R22/R23 往两处加了字段——数据集与 preset 的 ``angle_convention``，以及 profile 曲线的
``objective_threshold``；参数 setting 的 ``freedom`` 是第三处，它还多一条：旧文件里同一件事
存的是 ``locked: bool``，所以那个键得继续读得进来。几处的规矩是同一条，所以摆在一个文件里：

1. **默认值不写出来。** 老项目文件重存一遍必须逐位相同，凭空多一个 ``"angle_convention":
   "two_theta"`` 就不是了；导出的逐位不变是这套 codec 对得住 provenance 的根据。
2. **非默认值必须活过一次往返。** ``"theta"`` 说的是源文件那一列是入射角，重读时不照原样
   解释一次，角度就掉回一半——而这个错不会报，只会让层厚整体差一倍。
3. **缺键的老文件照旧能开。** 字段是后加的，此前存下的文件里没有这个键；解码得落回默认，
   而不是拿字段集校验把文件判成坏的。

这三条一起才管得住一个可选字段：只守往返，默认值会被写进文件；只守不落键，非默认值可能
根本没存；只守这两条，老文件会打不开。
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


def test_profile_without_a_closure_threshold_omits_the_key() -> None:
    """扫描没报阈值时那个键不写出来——写一个 ``null`` 进去，老文件重存就不再逐位相同。"""
    results = import_module("xrr_fitter.io.codec_results")
    profile = _project_with_profile(None).datasets[0].last_valid_result.uncertainty.profiles[0]

    payload = results._profile_to_dict(profile)

    assert profile.objective_threshold is None
    assert set(payload) == {
        "name",
        "values",
        "objectives",
        "lower_closed",
        "upper_closed",
    }


def test_old_profile_without_the_threshold_key_still_decodes() -> None:
    """字段是后加的：此前存下的 profile 没有这个键，解码得落回 ``None`` 而不是判文件坏。"""
    payload = project_to_dict(_project_with_profile(0.225))
    for dataset in payload["datasets"]:
        result = dataset["last_valid_result"]
        if result is not None and result["uncertainty"] is not None:
            for profile in result["uncertainty"]["profiles"]:
                # 造的这份里键确实在，拿掉它才是「老文件」——原本就没有的话这一步会空转。
                assert profile.pop("objective_threshold") == 0.225

    assert _restored_profile(payload).objective_threshold is None


def _project_with_setting(freedom: ParameterFreedom):
    """一份带单条 setting 的工程——``freedom`` 是这里唯一在变的东西。"""
    setting = ParameterSetting("instrument.scale", 1.0, 0.5, 2.0, freedom)
    return project(replace(dataset_project("sample-1"), parameter_settings=(setting,)))


def _restored_setting(payload: dict[str, object]) -> ParameterSetting:
    return project_from_dict(payload).datasets[0].parameter_settings[0]


def test_codec_omits_the_free_freedom_and_round_trips_the_other_two() -> None:
    """``FREE`` 不落键，另外两档必须往返。

    第三态是往 v2 里加的：默认那一档写进文件的话，既有项目重存一遍就不再逐位相同；而
    ``fixed``/``range_only`` 掉了键会静默读成「自由」——拟合器于是去推一个读者钉死了的量，
    或者把「优先待在这段区间内」整条丢掉，两种都不报错。
    """
    default_document = project_to_dict(_project_with_setting(ParameterFreedom.FREE))
    assert "freedom" not in default_document["datasets"][0]["parameter_settings"][0]

    for freedom in (ParameterFreedom.FIXED, ParameterFreedom.RANGE_ONLY):
        pinned = _project_with_setting(freedom)
        document = project_to_dict(pinned)

        assert document["datasets"][0]["parameter_settings"][0]["freedom"] == freedom.value
        restored = project_from_bytes(project_to_bytes(pinned))
        assert restored.datasets[0].parameter_settings[0].freedom is freedom
        assert restored == pinned


def test_old_setting_written_with_locked_decodes_into_the_matching_gear() -> None:
    """旧文件里第三态还不存在，自由度是一个 ``locked: bool``——两个值都得落到对应那一档。

    这个键在 v2 里原本是必填的，现在的写出路径不再产生它；把它读成「未知字段」会让此前
    存下的每一份工程都打不开。
    """
    payload = project_to_dict(_project_with_setting(ParameterFreedom.FREE))
    stored = payload["datasets"][0]["parameter_settings"][0]
    assert "freedom" not in stored

    stored["locked"] = True
    assert _restored_setting(payload).freedom is ParameterFreedom.FIXED

    stored["locked"] = False
    assert _restored_setting(payload).freedom is ParameterFreedom.FREE

    stored.pop("locked")
    assert _restored_setting(payload).freedom is ParameterFreedom.FREE


def test_a_setting_carrying_both_keys_is_refused_rather_than_silently_picking_one() -> None:
    """两个键同时在的文件没法确定读者的本意，所以判坏而不是挑一个。

    挑 ``freedom`` 会让一份 ``locked: true`` + ``freedom: "free"`` 的文件把钉死的量放开，
    挑 ``locked`` 会让「仅范围」退成两态；两种都不报错，而这种文件只能是被手改或被两个
    版本先后写过，本身就该拦下来。
    """
    payload = project_to_dict(_project_with_setting(ParameterFreedom.FIXED))
    payload["datasets"][0]["parameter_settings"][0]["locked"] = True

    with pytest.raises(ProjectSchemaError, match="both freedom and locked"):
        project_from_dict(payload)


def test_an_unknown_freedom_names_itself_instead_of_falling_back() -> None:
    """认不出来的档位要报出那个词。落回「自由」会让新版写的文件在旧版里悄悄放开一个量。"""
    payload = project_to_dict(_project_with_setting(ParameterFreedom.FIXED))
    payload["datasets"][0]["parameter_settings"][0]["freedom"] = "soft"

    with pytest.raises(ProjectSchemaError, match="unsupported parameter setting freedom"):
        project_from_dict(payload)
