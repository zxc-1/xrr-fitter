"""Interval metadata must survive storage without inventing calibration."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from tests.unit.analysis.covariance_cases import scale_candidate, scale_problem

from xrr_fitter.analysis.bootstrap import bootstrap_local
from xrr_fitter.analysis.report import build_uncertainty_report
from xrr_fitter.io.codec_declarations import _fit_config_from_dict, _fit_config_to_dict
from xrr_fitter.io.codec_results import _uncertainty_from_dict, _uncertainty_to_dict
from xrr_fitter.model.fitting import FitConfig


def _report(count=8):
    problem = scale_problem("gaussian")
    bootstrap = bootstrap_local(
        lambda rng, _i: np.array([0.5 + rng.normal(0, 0.01)]), ("instrument.scale",), sample_count=count, child_seed=17
    )
    return build_uncertainty_report(
        problem, (scale_candidate(problem),), profile_names=("instrument.scale",), bootstrap=bootstrap
    )


def test_config_round_trip_keeps_independent_profile_resolution() -> None:
    config = replace(FitConfig.fast(17), profile_steps=23)
    assert _fit_config_from_dict(_fit_config_to_dict(config)).profile_steps == 23


@pytest.mark.parametrize("count", [8, 200])
def test_report_round_trip_keeps_profile_threshold_and_bootstrap_evidence(count) -> None:
    report = _report(count)
    assert hasattr(report, "bootstrap_evidence"), "reports must retain interval calibration evidence"
    decoded = _uncertainty_from_dict(json.loads(json.dumps(_uncertainty_to_dict(report))))
    assert decoded.profiles[0].interval_kind == "likelihood_ratio"
    assert decoded.profiles[0].delta_total == report.profiles[0].delta_total
    assert decoded.bootstrap_evidence.successful_samples == count
    assert decoded.bootstrap_evidence.confidence_level == (0.95 if count >= 200 else None)
    assert decoded.bootstrap_evidence.method == "custom_resampling"
    np.testing.assert_array_equal(decoded.bootstrap_evidence.samples, report.bootstrap_evidence.samples)


def test_report_rejects_summary_disagreeing_with_bootstrap_evidence() -> None:
    report = _report()
    assert hasattr(report, "bootstrap_evidence")
    with pytest.raises(ValueError, match="bootstrap"):
        replace(report, bootstrap_performed=False)
    with pytest.raises(ValueError, match="bootstrap"):
        replace(report, bootstrap_failure_rate=0.3)


def test_codec_requires_interval_metadata_in_v2_reports() -> None:
    payload = _uncertainty_to_dict(_report())
    assert "bootstrap_evidence" in payload
    del payload["bootstrap_evidence"]
    with pytest.raises(ValueError, match="bootstrap_evidence"):
        _uncertainty_from_dict(payload)


def test_report_round_trip_retains_every_failed_attempt_and_the_empty_sample_axis() -> None:
    report = _report()
    failed = bootstrap_local(
        lambda _rng, index: f"nonconvergence:{index}", ("instrument.scale",), sample_count=8, child_seed=17
    )
    report = replace(report, bootstrap_evidence=failed, bootstrap_failure_rate=1.0)
    decoded = _uncertainty_from_dict(json.loads(json.dumps(_uncertainty_to_dict(report))))
    evidence = decoded.bootstrap_evidence
    assert evidence.samples.shape == (0, 1)
    assert evidence.attempted_count == 8
    assert evidence.failure_reasons == tuple((index, f"nonconvergence:{index}") for index in range(8))
    assert evidence.interval_kind == "unavailable"
    assert evidence.confidence_level is None
