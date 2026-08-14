"""ORSO ``.ort`` export contract: three-layer validation and NaN policy.

The export reuses ``PreparedData.validation_mask`` to drop non-finite and
non-positive-``Qz`` rows from the data segment (修正 8) and records the dropped
count in the ``xrr_fitter.confidence`` extension. Data columns and parameter
values survive an orsopy round-trip bit-for-bit because orsopy writes 17
significant digits, so equality is asserted with ``==`` not ``approx``.
"""

from __future__ import annotations

import io
from dataclasses import replace

import numpy as np
from orsopy import fileio
from orsopy.fileio.base import _read_header_data, _validate_header_data
from tests.support.model_cases import (
    dataset_project,
    final_fit_result,
    fit_candidate,
    prepared_data,
    project,
)

from xrr_fitter.io.export_tables import DatasetExportData, ExportReplayIdentity
from xrr_fitter.io.orso import orso_bytes
from xrr_fitter.model.analysis import UncertaintyReport
from xrr_fitter.model.parameters import ParameterValue


def _orso_context(
    *,
    two_theta_deg: np.ndarray | None = None,
    with_sigma: bool = True,
) -> DatasetExportData:
    data = prepared_data(two_theta_deg=two_theta_deg)
    if with_sigma:
        data = replace(data, intensity_sigma_normalized=data.intensity_normalized * 0.05)
    model = data.intensity_normalized * 0.97
    residual = np.log10(model + data.r_floor) - np.log10(data.intensity_normalized + data.r_floor)
    base = fit_candidate()
    candidate = replace(
        base,
        parameters=(*base.parameters, ParameterValue("instrument.background", 2.5e-7, 0.0, 1.0e-5)),
        qz_a_inv=data.qz_a_inv,
        model_normalized=model,
        log_residuals_decades=residual,
        weighted_residuals=residual / 0.05,
    )
    uncertainty = UncertaintyReport(
        correlation_names=("scale", "instrument.background"),
        correlation_matrix=np.eye(2),
        profiles=(),
        bootstrap_intervals=(("scale", 0.9, 1.1), ("instrument.background", 2.0e-7, 3.0e-7)),
        bootstrap_failure_rate=0.0,
        boundary_hits=(),
        strong_correlations=(),
        systematic_residual=False,
        diagnostics=(),
        parameter_sigma=np.array([0.05, 1.0e-8]),
    )
    result = replace(
        final_fit_result(candidate),
        uncertainty=uncertainty,
        classification_evidence=("boundary_hit",),
    )
    # dataset 声明的 fit_mask 必须与 prepared data 一致（DatasetExportData
    # 身份校验），负角度构造下 validation/fit mask 含 False，故对齐后再装配。
    dataset = dataset_project("curve", result=result)
    dataset = replace(dataset, fit_mask=tuple(bool(value) for value in data.fit_mask))
    return DatasetExportData(
        project=project(dataset),
        dataset=dataset,
        data=data,
        directory_mapping=(("curve", "001-curve-aaaaaaaa"),),
        selected=candidate,
        replay_identity=ExportReplayIdentity(1, 10101, 20202),
        matching_surface_oxide_rejection=False,
    )


def _load_single(raw: bytes) -> fileio.OrsoDataset:
    datasets = fileio.load_orso(io.StringIO(raw.decode("utf-8")))
    assert len(datasets) == 1
    return datasets[0]


def test_orso_bytes_roundtrips_data_parameters_and_extension() -> None:
    context = _orso_context()
    sigma = context.result.uncertainty.parameter_sigma
    covariance = np.diag(sigma**2)

    raw = orso_bytes(context, covariance=covariance)
    loaded = _load_single(raw)

    # 层 2: header 通过 ORSO schema 校验。``Orso.to_dict()`` 直出 datetime，
    # 需经 orsopy reader 得到字符串化 header 再校验（与 orso_bytes 内部同一路径）。
    header_dicts, _rows, _version = _read_header_data(io.StringIO(raw.decode("utf-8")))
    _validate_header_data(header_dicts)

    # 层 3: 数据列逐位相等（未被排除的行，全正角度 -> mask 全 True）。
    # 前四列位置被 ORSO schema 固定为 Qz/R/sR/sQz，这里只发出 Qz、R、sR；
    # 模型与残差是拟合产物，落在 xrr_fitter.model 扩展段（同样逐位核验）。
    mask = context.data.validation_mask
    table = loaded.data
    assert table.shape[1] == 3
    assert np.array_equal(table[:, 0], context.data.qz_a_inv[mask])
    assert np.array_equal(table[:, 1], context.data.intensity_normalized[mask])
    assert np.array_equal(table[:, 2], context.data.intensity_sigma_normalized[mask])

    model = loaded.info.user_data["xrr_fitter.model"]
    assert np.array_equal(np.array(model["reflectivity"]), context.selected.model_normalized[mask])
    assert np.array_equal(np.array(model["residual_decades"]), context.selected.log_residuals_decades[mask])

    confidence = loaded.info.user_data["xrr_fitter.confidence"]
    assert confidence["class_name"] == "TRUSTED"
    assert confidence["display"] == "可信"
    assert confidence["reason_codes"] == ["boundary_hit"]
    params = confidence["parameters"]
    assert [item["name"] for item in params] == ["scale", "instrument.background"]
    assert params[0]["value"] == context.selected.parameters[0].value
    assert params[0]["lower"] == context.selected.parameters[0].lower
    assert [bar["name"] for bar in confidence["error_bars"]] == ["scale", "instrument.background"]
    assert confidence["covariance"]["names"] == ["scale", "instrument.background"]
    assert np.array_equal(np.array(confidence["covariance"]["matrix"]), covariance)
    assert confidence["excluded_rows"]["count"] == 0

    reduction = loaded.info.user_data["xrr_fitter.reduction"]
    assert reduction["service_seed_tree_version"] == context.replay_identity.service_seed_tree_version
    assert reduction["fit_config"]["master_seed"] == context.project.fit_config.master_seed


def test_orso_bytes_excludes_nonpositive_angle_rows() -> None:
    context = _orso_context(two_theta_deg=np.linspace(-0.2, 3.2, 32))
    mask = context.data.validation_mask
    assert not bool(mask.all())  # 构造确实含被排除的非正角度行

    raw = orso_bytes(context, covariance=None)
    loaded = _load_single(raw)

    assert loaded.data.shape[0] == int(mask.sum())
    excluded = loaded.info.user_data["xrr_fitter.confidence"]["excluded_rows"]
    assert excluded["count"] == int((~mask).sum())
    assert excluded["reason"]

    # 扩展段的模型/残差也应与保留行对齐。
    model = loaded.info.user_data["xrr_fitter.model"]
    assert len(model["reflectivity"]) == int(mask.sum())
    assert len(model["residual_decades"]) == int(mask.sum())


def test_orso_bytes_omits_covariance_when_none() -> None:
    context = _orso_context()

    raw = orso_bytes(context, covariance=None)
    confidence = _load_single(raw).info.user_data["xrr_fitter.confidence"]

    assert "covariance" not in confidence
    assert confidence["covariance_absent_reason"]
