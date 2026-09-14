"""Saved V2 inference has the same meaning in every export projection."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from orsopy import fileio
from tests.support.bootstrap_cases import bootstrap_evidence
from tests.unit.io.test_export_tables import _context_with_parameter_metadata

from xrr_fitter.io.codec_inference import covariance_to_dict, residual_to_dict
from xrr_fitter.io.export_log import run_log_bytes
from xrr_fitter.io.export_tables import (
    DatasetExportData,
    batch_workbook_bytes,
    compatibility_workbook_bytes,
    dataset_json_bytes,
    dataset_workbook_bytes,
    parameters_csv_bytes,
)
from xrr_fitter.io.orso import orso_bytes
from xrr_fitter.io.project_codec import project_from_dict, project_to_dict
from xrr_fitter.model.analysis import (
    BootstrapResult,
    CovarianceEvidence,
    ParameterProfile,
    ResidualEvidence,
    UncertaintyReport,
)

MODES = (
    ("robust_log", "log_reflectivity", "decade"),
    ("gaussian", "standardized_intensity", "1"),
    ("poisson", "signed_poisson_deviance", "1"),
)


def _sampling(names: tuple[str, ...], *, formal: bool) -> BootstrapResult:
    if formal:
        return bootstrap_evidence(((names[0], 0.8, 1.2), (names[1], 1e-7, 4e-7)))
    return BootstrapResult(
        names,
        np.column_stack((np.linspace(0.8, 1.2, 8), np.linspace(1e-7, 4e-7, 8))),
        (),
        0.0,
        8,
        method="gaussian_parametric",
        unavailable_reason="insufficient_successful_samples",
    )


def _profile(name: str, *, formal: bool) -> ParameterProfile:
    return ParameterProfile(
        name,
        np.array([0.8, 1.0, 1.2]),
        np.array([1.3, 1.0, 1.3]),
        True,
        True,
        interval_kind="likelihood_ratio" if formal else "loss_support",
        confidence_level=0.95 if formal else None,
        method="chi_square_likelihood_ratio" if formal else "objective_tolerance",
        unavailable_reason=None if formal else "scale_prior_active",
        delta_total=3.841458820694124 if formal else None,
        objective_point_count=32,
    )


def _report(context: DatasetExportData, *, available: bool, formal: bool) -> UncertaintyReport:
    names = ("scale", "instrument.background")
    covariance = CovarianceEvidence(
        names,
        np.diag([0.0625, 1e-16]) if available else None,
        "gaussian_known_sigma",
        2 if available else 1,
        () if available else (names[1],),
        None if available else "rank_deficient",
    )
    sampling = _sampling(names, formal=formal)
    return UncertaintyReport(
        names,
        np.eye(2) if available else np.full((2, 2), np.nan),
        (_profile(names[0], formal=formal),),
        sampling.intervals,
        sampling.failure_rate,
        (),
        (),
        None,
        (),
        residual_autocorrelation=None,
        candidate_id=context.selected.candidate_id,
        bootstrap_performed=True,
        parameter_sigma=np.array([0.25, 1e-8]) if available else None,
        covariance_evidence=covariance,
        member_residuals=(ResidualEvidence("curve", False, None, None, 32, (), "not_requested"),),
        search_parameter_spread=np.array([7.0, 8.0]),
        bootstrap_evidence=sampling,
    )


def evidence_context(
    noise_model: str = "gaussian", *, available: bool = False, formal: bool = False
) -> DatasetExportData:
    original = _context_with_parameter_metadata()
    candidate = replace(
        original.selected,
        noise_model=noise_model,
        residuals=np.linspace(-2.0, 2.0, original.data.qz_a_inv.size),
    )
    report = _report(original, available=available, formal=formal)
    result = replace(original.result, candidates=(candidate,), uncertainty=report)
    dataset = replace(original.dataset, last_valid_result=result)
    project = replace(
        original.project,
        datasets=(dataset,),
        fit_config=replace(original.project.fit_config, noise_model=noise_model),
    )
    return replace(original, project=project, dataset=dataset, selected=candidate)


def _workbook_frame(context: DatasetExportData, sheet: str) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(dataset_workbook_bytes(context)), sheet_name=sheet, keep_default_na=False)


def _orso(context: DatasetExportData) -> fileio.OrsoDataset:
    return fileio.load_orso(io.StringIO(orso_bytes(context).decode()))[0]


def _metadata(context: DatasetExportData, kind: str) -> dict:
    if kind == "json":
        return json.loads(dataset_json_bytes(context))["run_info"]["inference"]
    if kind == "workbook":
        return json.loads(_workbook_frame(context, "RunInfo").iloc[0]["inference"])
    if kind == "csv":
        row = next(csv.DictReader(io.StringIO(parameters_csv_bytes(context).decode())))
        return json.loads(row["inference_json"])
    if kind == "orso":
        return _orso(context).info.user_data["xrr_fitter.confidence"]["inference"]
    lines = run_log_bytes(context).decode().splitlines()
    return json.loads(next(line.removeprefix("inference: ") for line in lines if line.startswith("inference: ")))


@pytest.mark.parametrize(("mode", "name", "unit"), MODES)
def test_json_model_projection_preserves_actual_residual_values_and_units(mode, name, unit) -> None:
    context = evidence_context(mode)
    model = json.loads(dataset_json_bytes(context))["model_residuals"]

    assert (model["noise_model"], model["residual_name"], model["residual_unit"]) == (mode, name, unit)
    np.testing.assert_array_equal(model["residuals"], context.selected.residuals)
    assert not np.array_equal(context.selected.residuals, context.selected.log_residuals_decades)


@pytest.mark.parametrize(("mode", "name", "unit"), MODES)
def test_workbook_preserves_actual_residual_values_and_units(mode, name, unit) -> None:
    context = evidence_context(mode)
    model = _workbook_frame(context, "ModelResiduals")
    info = _workbook_frame(context, "RunInfo").iloc[0]

    assert (info["noise_model"], info["residual_name"], str(info["residual_unit"])) == (mode, name, unit)
    np.testing.assert_allclose(model["residuals"], context.selected.residuals, atol=1e-15, rtol=1e-15)


@pytest.mark.parametrize("kind", ("json", "csv", "workbook", "log", "orso"))
@pytest.mark.parametrize("available", (False, True))
def test_export_metadata_uses_saved_covariance_and_unexecuted_diagnostics(kind, available) -> None:
    context = evidence_context(available=available)
    report = context.selected_uncertainty
    metadata = _metadata(context, kind)

    covariance = covariance_to_dict(report.covariance_evidence)
    assert metadata["covariance_evidence"] == {key: value for key, value in covariance.items() if key != "matrix"}
    residual = residual_to_dict(report.member_residuals[0])
    residual["diagnostic_count"] = len(residual.pop("diagnostics"))
    assert metadata["member_residuals"] == [residual]
    assert metadata["systematic_residual"] is None
    assert metadata["residual_autocorrelation"] is None


@pytest.mark.parametrize("kind", ("json", "csv", "workbook", "log", "orso"))
@pytest.mark.parametrize("formal", (False, True))
def test_export_interval_semantics_come_from_sampling_and_profile_evidence(kind, formal) -> None:
    context = evidence_context(formal=formal)
    report = context.selected_uncertainty
    metadata = _metadata(context, kind)
    bootstrap = metadata["bootstrap_evidence"]
    profile = metadata["profiles"][0]

    expected = report.bootstrap_evidence
    expected_bootstrap = {
        "interval_kind": expected.interval_kind,
        "confidence_level": expected.confidence_level,
        "method": expected.method,
        "unavailable_reason": expected.unavailable_reason,
        "successful_samples": expected.successful_samples,
        "intervals": [list(row) for row in expected.intervals],
    }
    assert {key: bootstrap[key] for key in expected_bootstrap} == expected_bootstrap
    expected_profile = {
        "interval_kind": report.profiles[0].interval_kind,
        "confidence_level": report.profiles[0].confidence_level,
        "method": report.profiles[0].method,
        "unavailable_reason": report.profiles[0].unavailable_reason,
    }
    assert {key: profile[key] for key in expected_profile} == expected_profile


@pytest.mark.parametrize("available", (False, True))
def test_csv_and_workbook_sigma_is_saved_statistical_sigma_or_blank(available) -> None:
    context = evidence_context(available=available)
    csv_rows = list(csv.DictReader(io.StringIO(parameters_csv_bytes(context).decode())))
    sheet = _workbook_frame(context, "Parameters")

    expected = [0.25, 1e-8] if available else ["", ""]
    observed = [float(row["sigma"]) if row["sigma"] else "" for row in csv_rows]
    assert observed == expected
    assert sheet["sigma"].tolist() == expected
    assert csv_rows[0]["covariance_method"] == "gaussian_known_sigma"
    assert int(csv_rows[0]["covariance_rank"]) == (2 if available else 1)
    assert csv_rows[0]["sigma_unavailable_reason"] == ("" if available else "rank_deficient")


def test_workbook_profile_rows_include_explicit_interval_metadata() -> None:
    context = evidence_context()
    profiles = _workbook_frame(context, "Profiles")

    assert profiles["interval_kind"].tolist() == ["loss_support"] * 3
    assert profiles["confidence_level"].tolist() == [""] * 3
    assert profiles["method"].tolist() == ["objective_tolerance"] * 3
    assert profiles["unavailable_reason"].tolist() == ["scale_prior_active"] * 3


@pytest.mark.parametrize("serializer", (compatibility_workbook_bytes, batch_workbook_bytes))
def test_summary_workbooks_keep_mode_and_inference_metadata(serializer) -> None:
    context = evidence_context()
    summary = pd.read_excel(io.BytesIO(serializer((context,))), sheet_name="Summary", keep_default_na=False)

    assert summary.iloc[0]["noise_model"] == "gaussian"
    assert json.loads(summary.iloc[0]["inference"]) == _metadata(context, "json")


@pytest.mark.parametrize(("mode", "observed"), (("gaussian", -0.01), ("poisson", 0.0)))
def test_orso_keeps_mode_valid_observations_even_when_log_residual_is_undefined(mode, observed) -> None:
    context = evidence_context(mode)
    data_values = context.data.intensity_normalized.copy()
    data_values[0] = observed
    data = replace(
        context.data, intensity_normalized=data_values, intensity_raw=data_values * context.data.normalization
    )
    log_residual = context.selected.log_residuals_decades.copy()
    log_residual[0] = np.nan
    candidate = replace(context.selected, log_residuals_decades=log_residual)
    dataset = replace(context.dataset, last_valid_result=replace(context.result, candidates=(candidate,)))
    changed = replace(
        context, project=replace(context.project, datasets=(dataset,)), dataset=dataset, data=data, selected=candidate
    )

    exported = _orso(changed)

    assert exported.data.shape[0] == context.data.qz_a_inv.size
    assert exported.data[0, 1] == observed
    model = exported.info.user_data["xrr_fitter.model"]
    assert model["noise_model"] == mode
    np.testing.assert_array_equal(model["residuals"], candidate.residuals)
    assert model["residual_decades"][0] is None


def test_export_metadata_is_unchanged_after_strict_project_roundtrip() -> None:
    context = evidence_context(available=True, formal=True)
    restored = project_from_dict(project_to_dict(context.project))
    dataset = restored.datasets[0]
    loaded = replace(context, project=restored, dataset=dataset, selected=dataset.last_valid_result.best_candidate)

    assert dataset_json_bytes(loaded) == dataset_json_bytes(context)
    assert parameters_csv_bytes(loaded) == parameters_csv_bytes(context)
    assert dataset_workbook_bytes(loaded) == dataset_workbook_bytes(context)
    assert run_log_bytes(loaded) == run_log_bytes(context)
    assert _metadata(loaded, "orso") == _metadata(context, "json")


def test_workbook_inference_summary_is_not_truncated_by_a_large_covariance() -> None:
    context = evidence_context()
    names = (*context.selected_uncertainty.correlation_names, *(f"parameter-{index}" for index in range(62)))
    matrix = np.full((64, 64), 0.10000000000000003)
    np.fill_diagonal(matrix, 0.25)
    report = replace(
        context.selected_uncertainty,
        correlation_names=names,
        correlation_matrix=matrix / 0.25,
        parameter_sigma=np.full(64, 0.5),
        covariance_evidence=CovarianceEvidence(names, matrix, "gaussian_known_sigma", 64),
        bootstrap_evidence=None,
        bootstrap_performed=False,
        search_parameter_spread=None,
    )
    dataset = replace(context.dataset, last_valid_result=replace(context.result, uncertainty=report))
    changed = replace(context, project=replace(context.project, datasets=(dataset,)), dataset=dataset)

    cell = _workbook_frame(changed, "RunInfo").iloc[0]["inference"]

    assert cell.endswith("}"), "workbook truncated the inference JSON at Excel's cell limit"
    metadata = json.loads(cell)
    assert metadata["covariance_evidence"]["rank"] == 64
    assert metadata["covariance_evidence"]["names"] == list(names)
