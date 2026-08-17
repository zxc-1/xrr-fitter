"""ORSO ``.ort`` export contract: three-layer validation and NaN policy.

The export reuses ``PreparedData.validation_mask`` to drop non-finite and
non-positive-``Qz`` rows from the data segment (修正 8) and records the dropped
count in the ``xrr_fitter.confidence`` extension. Data columns and parameter
values survive an orsopy round-trip bit-for-bit because orsopy writes 17
significant digits, so equality is asserted with ``==`` not ``approx``.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tomllib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from jsonschema import ValidationError
from orsopy import fileio
from orsopy.fileio.base import _read_header_data, _validate_header_data
from tests.support.model_cases import (
    dataset_project,
    final_fit_result,
    fit_candidate,
    prepared_data,
    project,
)

import xrr_fitter.io.orso as orso_module
from xrr_fitter.io.export_tables import DatasetExportData, ExportReplayIdentity
from xrr_fitter.io.orso import orso_bytes
from xrr_fitter.model.analysis import UncertaintyReport
from xrr_fitter.model.data import BeamSpec
from xrr_fitter.model.parameters import ParameterValue

ROOT = Path(__file__).resolve().parents[3]


def _declared_package_version() -> str:
    from xrr_fitter.version import __version__

    return __version__


def test_api_import_does_not_require_installed_distribution_metadata() -> None:
    script = """
import importlib.metadata

def missing_distribution(name):
    raise importlib.metadata.PackageNotFoundError(name)

importlib.metadata.version = missing_distribution
import xrr_fitter.api
"""
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))

    completed = subprocess.run(
        (sys.executable, "-c", script),
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_orso_bytes_does_not_require_installed_distribution_metadata() -> None:
    script = """
import importlib.metadata

def missing_distribution(name):
    raise importlib.metadata.PackageNotFoundError(name)

importlib.metadata.version = missing_distribution

from tests.unit.io.test_orso_export import _load_single, _orso_context
from xrr_fitter.io.orso import orso_bytes
from xrr_fitter.version import __version__

loaded = _load_single(orso_bytes(_orso_context(), covariance=None))
assert loaded.info.reduction.software.version == __version__
"""
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))

    completed = subprocess.run(
        (sys.executable, "-c", script),
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_orsopy_private_header_helpers_are_exactly_pinned_and_importable() -> None:
    dependencies = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
    assert "orsopy==1.2.3" in dependencies
    assert callable(_read_header_data)
    assert callable(_validate_header_data)


def _orso_context(
    *,
    two_theta_deg: np.ndarray | None = None,
    with_sigma: bool = True,
    sigma_q_a_inv: np.ndarray | None = None,
    beam: BeamSpec | None = None,
    fit_mask: np.ndarray | None = None,
) -> DatasetExportData:
    data = prepared_data(two_theta_deg=two_theta_deg, beam=beam, fit_mask=fit_mask)
    if with_sigma:
        data = replace(data, intensity_sigma_normalized=data.intensity_normalized * 0.05)
    if sigma_q_a_inv is not None:
        data = replace(data, sigma_q_a_inv=np.asarray(sigma_q_a_inv, dtype=float))
    model = data.intensity_normalized * 0.97
    fit_residual = np.log10(model + data.r_floor) - np.log10(data.intensity_normalized + data.r_floor)
    residual = np.full(data.qz_a_inv.shape, np.nan, dtype=float)
    residual[data.fit_mask] = fit_residual[data.fit_mask]
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
        candidate_id=candidate.candidate_id,
    )
    result = replace(
        final_fit_result(candidate),
        uncertainty=uncertainty,
        classification_evidence=("boundary_hit",),
    )
    # dataset 声明的 fit_mask 必须与 prepared data 一致（DatasetExportData
    # 身份校验），负角度构造下 validation/fit mask 含 False，故对齐后再装配。
    dataset = dataset_project("curve", result=result)
    dataset = replace(
        dataset,
        beam=data.beam,
        fit_mask=tuple(bool(value) for value in data.fit_mask),
    )
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


def _assert_roundtrip_data(loaded, context, mask) -> None:
    table = loaded.data
    assert table.shape[1] == 3
    np.testing.assert_array_equal(table[:, 0], context.data.qz_a_inv[mask])
    np.testing.assert_array_equal(table[:, 1], context.data.intensity_normalized[mask])
    np.testing.assert_array_equal(table[:, 2], context.data.intensity_sigma_normalized[mask])


def _assert_roundtrip_model(loaded, context, mask) -> None:
    model = loaded.info.user_data["xrr_fitter.model"]
    np.testing.assert_array_equal(model["qz_a_inv"], context.selected.qz_a_inv[mask])
    np.testing.assert_array_equal(model["reflectivity"], context.selected.model_normalized[mask])
    np.testing.assert_array_equal(model["residual_decades"], context.selected.log_residuals_decades[mask])


def _assert_roundtrip_confidence(loaded, context, covariance) -> None:
    confidence = loaded.info.user_data["xrr_fitter.confidence"]
    parameters = confidence["parameters"]
    assert (
        confidence["class_name"],
        confidence["display"],
        confidence["reason_codes"],
        [item["name"] for item in parameters],
        parameters[0]["value"],
        parameters[0]["lower"],
        [bar["name"] for bar in confidence["error_bars"]],
        confidence["covariance"]["names"],
        confidence["excluded_rows"]["count"],
    ) == (
        "TRUSTED",
        "可信",
        ["boundary_hit"],
        ["scale", "instrument.background"],
        context.selected.parameters[0].value,
        context.selected.parameters[0].lower,
        ["scale", "instrument.background"],
        ["scale", "instrument.background"],
        0,
    )
    np.testing.assert_array_equal(confidence["covariance"]["matrix"], covariance)


def _assert_roundtrip_reduction(loaded, context) -> None:
    reduction = loaded.info.user_data["xrr_fitter.reduction"]
    assert (
        reduction["service_seed_tree_version"],
        reduction["fit_config"]["master_seed"],
    ) == (
        context.replay_identity.service_seed_tree_version,
        context.project.fit_config.master_seed,
    )


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
    _assert_roundtrip_data(loaded, context, mask)
    _assert_roundtrip_model(loaded, context, mask)
    _assert_roundtrip_confidence(loaded, context, covariance)
    _assert_roundtrip_reduction(loaded, context)


def test_orso_bytes_roundtrips_pointwise_q_resolution_as_standard_error_column() -> None:
    base = _orso_context()
    sigma_q = np.linspace(1.0e-4, 4.0e-4, base.data.qz_a_inv.size)
    context = _orso_context(sigma_q_a_inv=sigma_q)

    raw = orso_bytes(context, covariance=None)
    loaded = _load_single(raw)

    header_dicts, _rows, _version = _read_header_data(io.StringIO(raw.decode("utf-8")))
    _validate_header_data(header_dicts)
    assert [column.to_dict() for column in loaded.info.columns] == [
        {"name": "Qz", "unit": "1/angstrom"},
        {"name": "R"},
        {"error_of": "R", "error_type": "uncertainty", "value_is": "sigma"},
        {"error_of": "Qz", "error_type": "resolution", "value_is": "sigma"},
    ]
    mask = context.data.validation_mask
    assert loaded.data.shape[1] == 4
    np.testing.assert_array_equal(loaded.data[:, 0], context.data.qz_a_inv[mask])
    np.testing.assert_array_equal(loaded.data[:, 1], context.data.intensity_normalized[mask])
    np.testing.assert_array_equal(loaded.data[:, 2], context.data.intensity_sigma_normalized[mask])
    np.testing.assert_array_equal(loaded.data[:, 3], context.data.sigma_q_a_inv[mask])


def test_orso_bytes_excludes_rows_without_finite_pointwise_q_resolution() -> None:
    base = _orso_context()
    sigma_q = np.linspace(1.0e-4, 4.0e-4, base.data.qz_a_inv.size)
    sigma_q[7] = np.nan
    context = _orso_context(sigma_q_a_inv=sigma_q)

    loaded = _load_single(orso_bytes(context, covariance=None))

    expected_mask = context.data.validation_mask & np.isfinite(context.data.sigma_q_a_inv)
    assert loaded.data.shape[0] == int(expected_mask.sum())
    np.testing.assert_array_equal(loaded.data[:, 3], context.data.sigma_q_a_inv[expected_mask])
    excluded = loaded.info.user_data["xrr_fitter.confidence"]["excluded_rows"]
    assert excluded["count"] == int(np.count_nonzero(~expected_mask))
    assert excluded["reason"] == "non_finite_or_nonpositive_export_row"


def test_orso_bytes_keeps_q_resolution_in_reduction_extension_without_reflectivity_sigma() -> None:
    base = _orso_context(with_sigma=False)
    sigma_q = np.linspace(1.0e-4, 4.0e-4, base.data.qz_a_inv.size)
    context = _orso_context(with_sigma=False, sigma_q_a_inv=sigma_q)

    loaded = _load_single(orso_bytes(context, covariance=None))

    assert [column.to_dict() for column in loaded.info.columns] == [
        {"name": "Qz", "unit": "1/angstrom"},
        {"name": "R"},
    ]
    assert loaded.data.shape[1] == 2
    pointwise = loaded.info.user_data["xrr_fitter.reduction"]["pointwise_resolution"]
    assert {key: pointwise[key] for key in ("error_of", "error_type", "value_is", "unit")} == {
        "error_of": "Qz",
        "error_type": "resolution",
        "value_is": "sigma",
        "unit": "1/angstrom",
    }
    np.testing.assert_array_equal(pointwise["values"], context.data.sigma_q_a_inv[context.data.validation_mask])


def test_orso_bytes_keeps_prepared_and_fitted_qz_axes_distinct() -> None:
    context = _orso_context()
    fitted_qz = context.selected.qz_a_inv * 1.01
    selected = replace(context.selected, qz_a_inv=fitted_qz)
    result = replace(context.result, candidates=(selected,))
    dataset = replace(context.dataset, last_valid_result=result)
    changed = replace(
        context,
        project=replace(context.project, datasets=(dataset,)),
        dataset=dataset,
        selected=selected,
    )

    loaded = _load_single(orso_bytes(changed, covariance=None))
    mask = changed.data.validation_mask
    assert np.array_equal(loaded.data[:, 0], changed.data.qz_a_inv[mask])
    assert np.array_equal(
        np.asarray(loaded.info.user_data["xrr_fitter.model"]["qz_a_inv"]),
        fitted_qz[mask],
    )


def test_orso_bytes_normalizes_schema_validation_failure(monkeypatch) -> None:
    context = _orso_context()

    def reject_schema(_headers) -> None:
        raise ValidationError("invalid ORSO header")

    monkeypatch.setattr(orso_module, "_validate_header_data", reject_schema)

    with pytest.raises(ValueError, match="ORSO schema validation failed"):
        orso_bytes(context, covariance=None)


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
    assert len(model["qz_a_inv"]) == int(mask.sum())
    assert len(model["reflectivity"]) == int(mask.sum())
    assert len(model["residual_decades"]) == int(mask.sum())


def test_orso_bytes_excludes_valid_rows_without_finite_fit_residuals() -> None:
    fit_mask = np.ones(32, dtype=bool)
    fit_mask[5] = False
    context = _orso_context(fit_mask=fit_mask)

    loaded = _load_single(orso_bytes(context, covariance=None))
    model = loaded.info.user_data["xrr_fitter.model"]
    excluded = loaded.info.user_data["xrr_fitter.confidence"]["excluded_rows"]

    assert loaded.data.shape[0] == 31
    assert excluded["count"] == 1
    assert excluded["reason"] == "non_finite_or_nonpositive_export_row"
    assert np.all(np.isfinite(model["residual_decades"]))


def test_orso_bytes_omits_covariance_when_none() -> None:
    context = _orso_context()

    raw = orso_bytes(context, covariance=None)
    confidence = _load_single(raw).info.user_data["xrr_fitter.confidence"]

    assert "covariance" not in confidence
    assert confidence["covariance_absent_reason"]


def test_orso_bytes_rejects_nonfinite_bootstrap_interval() -> None:
    context = _orso_context()
    uncertainty = replace(
        context.result.uncertainty,
        bootstrap_intervals=(
            ("scale", float("nan"), 1.1),
            ("instrument.background", 2.0e-7, 3.0e-7),
        ),
    )
    result = replace(context.result, uncertainty=uncertainty)
    dataset = replace(context.dataset, last_valid_result=result)
    invalid = replace(
        context,
        project=replace(context.project, datasets=(dataset,)),
        dataset=dataset,
    )

    with pytest.raises(ValueError, match="bootstrap intervals"):
        orso_bytes(invalid, covariance=uncertainty.covariance)


@pytest.mark.parametrize(
    "covariance",
    (
        np.ones((2, 3)),
        np.array([[1.0, np.nan], [0.0, 1.0]]),
        np.array([[1.0, 0.25], [0.1, 1.0]]),
        np.array([[1.0, 2.0], [2.0, 1.0]]),
        np.array([[0.0, 1e-12], [1e-12, 0.0]]),
    ),
)
def test_orso_bytes_rejects_invalid_covariance(covariance: np.ndarray) -> None:
    context = _orso_context()

    with pytest.raises(ValueError, match="covariance"):
        orso_bytes(context, covariance=covariance)


def test_orso_bytes_symmetrizes_covariance_within_numeric_tolerance() -> None:
    context = _orso_context()
    covariance = np.array(
        [
            [1.0, 0.2 + 5e-13],
            [0.2, 1.0],
        ]
    )

    loaded = _load_single(orso_bytes(context, covariance=covariance))
    exported = np.asarray(
        loaded.info.user_data["xrr_fitter.confidence"]["covariance"]["matrix"],
        dtype=float,
    )

    np.testing.assert_array_equal(exported, exported.T)
    np.testing.assert_allclose(
        exported,
        (covariance + covariance.T) / 2.0,
        rtol=0.0,
        atol=0.0,
    )


def test_orso_bytes_omits_uncertainty_owned_by_another_candidate() -> None:
    context = _orso_context()
    first = context.selected
    selected = replace(
        first,
        candidate_id="E-1",
        parameters=tuple(replace(parameter, value=parameter.value * 0.9) for parameter in first.parameters),
    )
    uncertainty = replace(
        context.result.uncertainty,
        candidate_id=first.candidate_id,
    )
    result = replace(
        context.result,
        candidates=(first, selected),
        uncertainty=uncertainty,
    )
    dataset = replace(context.dataset, last_valid_result=result)
    mismatched = replace(
        context,
        project=project(dataset),
        dataset=dataset,
        selected=selected,
    )

    raw = orso_bytes(mismatched, covariance=np.eye(2))
    confidence = _load_single(raw).info.user_data["xrr_fitter.confidence"]

    assert confidence["parameters"][0]["value"] == selected.parameters[0].value
    assert confidence["error_bars"] == []
    assert "covariance" not in confidence
    assert first.candidate_id in confidence["covariance_absent_reason"]
    assert selected.candidate_id in confidence["covariance_absent_reason"]


def test_orso_bytes_records_source_hash_package_version_and_mixed_kalpha() -> None:
    beam = BeamSpec(
        kind="mixed_kalpha",
        wavelength_1_a=1.54056,
        wavelength_2_a=1.54439,
        intensity_ratio_21=0.42,
    )
    context = _orso_context(beam=beam)

    loaded = _load_single(orso_bytes(context, covariance=None))
    source = loaded.info.data_source.measurement.data_files[0]
    reduction = loaded.info.user_data["xrr_fitter.reduction"]

    assert isinstance(source, fileio.File)
    assert source.hash.algorithm == "sha256"
    assert source.hash.digest == context.data.source_sha256
    assert loaded.info.reduction.software.version == _declared_package_version()
    assert reduction["beam"] == {
        "kind": "mixed_kalpha",
        "effective_wavelength_a": beam.effective_wavelength_a,
        "wavelength_1_a": beam.wavelength_1_a,
        "wavelength_2_a": beam.wavelength_2_a,
        "intensity_ratio_21": beam.intensity_ratio_21,
    }
