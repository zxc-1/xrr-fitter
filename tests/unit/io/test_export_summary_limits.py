"""Workbook summaries stay legal JSON instead of silently losing large evidence."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from tests.unit.io.test_export_tables import _context_with_parameter_metadata
from tests.unit.io.test_export_v2_evidence import _workbook_frame, evidence_context

from xrr_fitter.io.export_tables import dataset_json_bytes, dataset_workbook_bytes
from xrr_fitter.model.analysis import BootstrapResult, ResidualEvidence
from xrr_fitter.model.instrument import PhysicsDiagnostic


def _with_report(context, report):
    dataset = replace(context.dataset, last_valid_result=replace(context.result, uncertainty=report))
    return replace(context, project=replace(context.project, datasets=(dataset,)), dataset=dataset)


def _workbook_summary(context):
    cell = _workbook_frame(context, "RunInfo").iloc[0]["inference"]
    assert cell.endswith("}"), "workbook silently truncated its evidence JSON"
    return json.loads(cell)


def test_workbook_summarizes_dense_diagnostics_but_full_json_keeps_point_identities() -> None:
    context = evidence_context()
    diagnostic = PhysicsDiagnostic("systematic", "dense residual evidence", tuple(range(10000)))
    member = ResidualEvidence("curve", True, True, False, 10000, (diagnostic,))
    report = replace(
        context.selected_uncertainty,
        member_residuals=(member,),
        systematic_residual=True,
        residual_autocorrelation=False,
    )
    changed = _with_report(context, report)

    summary = _workbook_summary(changed)
    full = json.loads(dataset_json_bytes(changed))["fit_result"]["uncertainty"]["member_residuals"][0]

    assert summary["member_residuals"][0]["diagnostic_count"] == 1
    assert summary["member_residuals"][0]["point_count"] == 10000
    assert full["diagnostics"][0]["point_indices"] == list(range(10000))


def test_workbook_summarizes_failed_bootstrap_work_but_full_json_keeps_failure_reasons() -> None:
    context = evidence_context()
    failures = tuple((index, "fit did not converge: " + "x" * 100) for index in range(300))
    evidence = BootstrapResult(
        context.selected_uncertainty.correlation_names,
        np.empty((0, 2)),
        (),
        1.0,
        300,
        failures,
        "gaussian_parametric",
        "excessive_fit_failures",
    )
    report = replace(context.selected_uncertainty, bootstrap_evidence=evidence, bootstrap_failure_rate=1.0)
    changed = _with_report(context, report)

    summary = _workbook_summary(changed)["bootstrap_evidence"]
    full = json.loads(dataset_json_bytes(changed))["fit_result"]["uncertainty"]["bootstrap_evidence"]

    assert summary["attempted_count"] == 300
    assert summary["successful_samples"] == 0
    assert summary["unavailable_reason"] == "excessive_fit_failures"
    assert full["failure_reasons"] == [list(row) for row in failures]


def test_workbook_rejects_remaining_oversize_text_instead_of_truncating_it() -> None:
    context = _context_with_parameter_metadata(dataset_id="x" * 32768)

    with pytest.raises(ValueError, match="Excel cell limit"):
        dataset_workbook_bytes(context)
