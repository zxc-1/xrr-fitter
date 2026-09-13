"""Backward-compatible codec coverage for ORSO parameter uncertainty."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from tests.support.model_cases import dataset_project, final_fit_result, fit_candidate, project

from xrr_fitter.io.project_codec import project_from_dict, project_to_dict
from xrr_fitter.model.analysis import UncertaintyReport


def _project_with_parameter_sigma(sigma: np.ndarray | None):
    candidate = fit_candidate()
    uncertainty = UncertaintyReport(
        correlation_names=("scale",),
        correlation_matrix=np.eye(1),
        profiles=(),
        bootstrap_intervals=(),
        bootstrap_failure_rate=0.0,
        boundary_hits=(),
        strong_correlations=(),
        systematic_residual=False,
        diagnostics=(),
        candidate_id=candidate.candidate_id,
        parameter_sigma=sigma,
    )
    result = replace(final_fit_result(candidate), uncertainty=uncertainty)
    return project(dataset_project("sample-1", result=result))


def _uncertainty_payload(value):
    return project_to_dict(value)["datasets"][0]["last_valid_result"]["uncertainty"]


def test_project_roundtrip_preserves_parameter_sigma() -> None:
    original = _project_with_parameter_sigma(np.array([2.0]))
    restored = project_from_dict(project_to_dict(original))
    before = original.datasets[0].last_valid_result.uncertainty.parameter_sigma
    after = restored.datasets[0].last_valid_result.uncertainty.parameter_sigma

    np.testing.assert_array_equal(after, before)


def test_result_without_parameter_sigma_omits_key() -> None:
    uncertainty = _uncertainty_payload(_project_with_parameter_sigma(None))

    assert "parameter_sigma" not in uncertainty


def test_result_without_parameter_sigma_key_still_decodes() -> None:
    payload = project_to_dict(_project_with_parameter_sigma(np.array([2.0])))
    payload["datasets"][0]["last_valid_result"]["uncertainty"].pop("parameter_sigma")

    restored = project_from_dict(payload)
    report = restored.datasets[0].last_valid_result.uncertainty

    assert report.parameter_sigma is None


def _project_with_resample_count(count: int):
    candidate = fit_candidate()
    uncertainty = UncertaintyReport(
        correlation_names=("scale",),
        correlation_matrix=np.eye(1),
        profiles=(),
        bootstrap_intervals=(),
        bootstrap_failure_rate=0.0,
        boundary_hits=(),
        strong_correlations=(),
        systematic_residual=False,
        diagnostics=(),
        candidate_id=candidate.candidate_id,
        bootstrap_sample_count=count,
    )
    result = replace(final_fit_result(candidate), uncertainty=uncertainty)
    return project(dataset_project("sample-1", result=result))


def test_project_roundtrip_preserves_the_bootstrap_resample_count() -> None:
    """次数存不住，重开工程后失败率就没了基数——「丢了 2%」是 200 次里的 4 次还是 20 次里的 0.4 次。"""
    original = _project_with_resample_count(200)
    restored = project_from_dict(project_to_dict(original))

    assert restored.datasets[0].last_valid_result.uncertainty.bootstrap_sample_count == 200


def test_result_without_a_resample_count_omits_key() -> None:
    """没跑过自助抽样的报告不该多出这个键：旧工程文件重编码要逐位不变，旧读者也要照旧读得动。"""
    uncertainty = _uncertainty_payload(_project_with_resample_count(0))

    assert "bootstrap_sample_count" not in uncertainty


def test_result_without_a_resample_count_key_still_decodes() -> None:
    """这个字段之前的工程文件里没有这个键，读到的应当是 0（未记录），不是报错。"""
    payload = project_to_dict(_project_with_resample_count(200))
    payload["datasets"][0]["last_valid_result"]["uncertainty"].pop("bootstrap_sample_count")

    restored = project_from_dict(payload)

    assert restored.datasets[0].last_valid_result.uncertainty.bootstrap_sample_count == 0
