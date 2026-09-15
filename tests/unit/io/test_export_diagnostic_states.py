"""Unknown, negative, positive, and unestimable are distinct export states."""

from __future__ import annotations

import csv
import io
from dataclasses import replace

import numpy as np
import pytest
from tests.unit.io.test_export_summary_limits import _with_report
from tests.unit.io.test_export_v2_evidence import _metadata, _orso, _workbook_frame, evidence_context

from xrr_fitter.io.export_tables import parameters_csv_bytes
from xrr_fitter.model.analysis import BootstrapResult, ResidualEvidence
from xrr_fitter.model.parameters import ParameterValue


@pytest.mark.parametrize("kind", ("json", "csv", "workbook", "log", "orso"))
@pytest.mark.parametrize("systematic", (False, True))
def test_executed_residual_conclusions_are_not_coerced_to_unknown(kind, systematic) -> None:
    context = evidence_context()
    member = ResidualEvidence("curve", True, systematic, False, 32)
    report = replace(
        context.selected_uncertainty,
        systematic_residual=systematic,
        residual_autocorrelation=False,
        member_residuals=(member,),
    )

    metadata = _metadata(_with_report(context, report), kind)

    assert metadata["systematic_residual"] is systematic
    assert metadata["residual_autocorrelation"] is False
    assert metadata["member_residuals"][0] == {
        "dataset_id": "curve",
        "executed": True,
        "systematic": systematic,
        "autocorrelation": False,
        "point_count": 32,
        "unavailable_reason": None,
        "diagnostic_count": 0,
        "raw_systematic": None,
        "raw_autocorrelation": None,
        "advisories": [],
        "calibration": None,
        "owner_sha256": None,
    }


@pytest.mark.parametrize("kind", ("json", "csv", "workbook", "log", "orso"))
def test_failed_bootstrap_work_remains_unavailable_instead_of_becoming_a_confidence_interval(kind) -> None:
    context = evidence_context()
    sampling = BootstrapResult(
        context.selected_uncertainty.correlation_names,
        np.empty((0, 2)),
        (),
        1.0,
        8,
        tuple((index, "fit_failed") for index in range(8)),
        "parametric_poisson",
        "excessive_fit_failures",
    )
    report = replace(context.selected_uncertainty, bootstrap_evidence=sampling, bootstrap_failure_rate=1.0)

    metadata = _metadata(_with_report(context, report), kind)["bootstrap_evidence"]

    assert metadata["interval_kind"] == "unavailable"
    assert metadata["confidence_level"] is None
    assert metadata["method"] == "parametric_poisson"
    assert metadata["unavailable_reason"] == "excessive_fit_failures"
    assert metadata["intervals"] == []


def test_sigma_without_covariance_provenance_is_not_published_as_an_error_estimate() -> None:
    context = evidence_context(available=True)
    report = replace(context.selected_uncertainty, covariance_evidence=None)
    changed = _with_report(context, report)
    rows = list(csv.DictReader(io.StringIO(parameters_csv_bytes(changed).decode())))
    parameters = _orso(changed).info.user_data["xrr_fitter.confidence"]["parameters"]

    assert rows[0]["sigma"] == ""
    assert rows[0]["sigma_unavailable_reason"] == "covariance not estimated for this fit result"
    assert parameters[0]["sigma"] is None


def test_parameters_outside_the_covariance_axis_have_blank_sigma_not_zero() -> None:
    context = evidence_context(available=True)
    parameter = ParameterValue("instrument.angle_offset_deg", 0.0, -0.1, 0.1)
    candidate = replace(context.selected, parameters=(*context.selected.parameters, parameter))
    definition = replace(
        context.result.parameter_definitions[-1],
        name=parameter.name,
        initial=parameter.value,
        lower=parameter.lower,
        upper=parameter.upper,
        locked=True,
    )
    result = replace(
        context.result,
        candidates=(candidate,),
        parameter_definitions=(*context.result.parameter_definitions, definition),
    )
    dataset = replace(context.dataset, last_valid_result=result)
    changed = replace(
        context, project=replace(context.project, datasets=(dataset,)), dataset=dataset, selected=candidate
    )

    row = _workbook_frame(changed, "Parameters").iloc[-1]

    assert row["sigma"] == ""
    assert row["sigma_unavailable_reason"] == "parameter not in covariance axis"
