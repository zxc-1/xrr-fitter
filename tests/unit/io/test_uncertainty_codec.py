"""Explicit V2 codec coverage for available and unavailable uncertainty."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
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


def test_result_without_parameter_sigma_emits_explicit_unavailable_value() -> None:
    uncertainty = _uncertainty_payload(_project_with_parameter_sigma(None))

    assert uncertainty["parameter_sigma"] is None


def test_result_without_parameter_sigma_key_is_incomplete_v2_evidence() -> None:
    payload = project_to_dict(_project_with_parameter_sigma(np.array([2.0])))
    payload["datasets"][0]["last_valid_result"]["uncertainty"].pop("parameter_sigma")

    with pytest.raises(ValueError, match="parameter_sigma"):
        project_from_dict(payload)
