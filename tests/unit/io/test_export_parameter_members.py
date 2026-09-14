"""Global covariance axes project only onto their saved dataset members."""

from __future__ import annotations

import csv
import io
from dataclasses import replace

import pandas as pd
import pytest
from tests.unit.io.test_export_v2_evidence import _metadata, _orso, _workbook_frame, evidence_context

from xrr_fitter.io.codec_inference import parameter_members_to_list
from xrr_fitter.io.export_tables import DatasetExportData, batch_workbook_bytes, parameters_csv_bytes
from xrr_fitter.model.parameters import ParameterReference


def _member_context(names, members) -> DatasetExportData:
    context = evidence_context(available=True)
    report = replace(
        context.selected_uncertainty,
        correlation_names=names,
        parameter_members=members,
        covariance_evidence=replace(context.selected_uncertainty.covariance_evidence, names=names),
        profiles=(),
        bootstrap_evidence=None,
        bootstrap_performed=False,
        bootstrap_intervals=(),
    )
    dataset = replace(context.dataset, last_valid_result=replace(context.result, uncertainty=report))
    return replace(context, dataset=dataset, project=replace(context.project, datasets=(dataset,)))


def _shared_context() -> DatasetExportData:
    return _member_context(
        ("opaque-axis-one", "opaque-axis-two"),
        tuple(
            tuple(ParameterReference(dataset_id, name) for dataset_id in ("curve", "peer"))
            for name in ("scale", "instrument.background")
        ),
    )


def _sigma_projection(context: DatasetExportData, kind: str) -> dict[str, dict]:
    if kind == "csv":
        rows = list(csv.DictReader(io.StringIO(parameters_csv_bytes(context).decode())))
    elif kind == "workbook":
        rows = _workbook_frame(context, "Parameters").to_dict("records")
    elif kind == "batch":
        rows = pd.read_excel(
            io.BytesIO(batch_workbook_bytes((context,))), sheet_name="Parameters", keep_default_na=False
        ).to_dict("records")
    else:
        rows = _orso(context).info.user_data["xrr_fitter.confidence"]["parameters"]
    return {row.get("name", row.get("parameter_name")): row for row in rows}


@pytest.mark.parametrize("kind", ("csv", "workbook", "batch", "orso"))
def test_shared_sigma_uses_exact_saved_member_identity_at_every_export(kind) -> None:
    context = _shared_context()
    rows = _sigma_projection(context, kind)

    assert rows["scale"]["sigma"] not in ("", None)
    assert rows["instrument.background"]["sigma"] not in ("", None)
    assert float(rows["scale"]["sigma"]) == context.selected_uncertainty.parameter_sigma[0]
    assert float(rows["instrument.background"]["sigma"]) == context.selected_uncertainty.parameter_sigma[1]
    assert rows["scale"]["sigma_unavailable_reason"] in ("", None)


@pytest.mark.parametrize("kind", ("json", "csv", "workbook", "log", "orso"))
def test_global_axis_members_are_in_the_common_inference_summary(kind) -> None:
    context = _shared_context()
    metadata = _metadata(context, kind)

    assert "parameter_members" in metadata
    assert metadata["parameter_members"] == parameter_members_to_list(context.selected_uncertainty.parameter_members)


@pytest.mark.parametrize("kind", ("csv", "workbook", "batch", "orso"))
def test_nonshared_sigma_never_uses_another_members_same_named_global_axis(kind) -> None:
    context = _member_context(
        ("scale", "instrument.background"),
        ((ParameterReference("peer", "scale"),), (ParameterReference("curve", "scale"),)),
    )
    rows = _sigma_projection(context, kind)

    assert float(rows["scale"]["sigma"]) == context.selected_uncertainty.parameter_sigma[1]
    assert rows["instrument.background"]["sigma"] in ("", None)
    assert rows["instrument.background"]["sigma_unavailable_reason"] == "parameter not in covariance axis"


@pytest.mark.parametrize("kind", ("csv", "workbook", "batch", "orso"))
@pytest.mark.parametrize("state", ("missing", "locked", "constrained"))
def test_absent_member_has_no_sigma_even_when_another_global_axis_has_the_local_name(kind, state) -> None:
    context = _member_context(
        ("scale", "instrument.background"),
        ((ParameterReference("peer", "scale"),), (ParameterReference("peer", "instrument.background"),)),
    )
    definition = replace(
        context.result.parameter_definitions[0], locked=state == "locked", constrained=state == "constrained"
    )
    result = replace(context.result, parameter_definitions=(definition, *context.result.parameter_definitions[1:]))
    dataset = replace(context.dataset, last_valid_result=result)
    context = replace(context, dataset=dataset, project=replace(context.project, datasets=(dataset,)))

    row = _sigma_projection(context, kind)["scale"]

    assert row["sigma"] in ("", None)
    assert row["sigma_unavailable_reason"] == "parameter not in covariance axis"


@pytest.mark.parametrize("kind", ("csv", "workbook", "batch", "orso"))
def test_shared_membership_does_not_override_candidate_ownership(kind) -> None:
    context = _shared_context()
    alternate = replace(context.selected, candidate_id="another-candidate", objective=context.selected.objective + 1.0)
    result = replace(context.result, candidates=(context.selected, alternate))
    dataset = replace(context.dataset, last_valid_result=result)
    changed = replace(
        context, dataset=dataset, selected=alternate, project=replace(context.project, datasets=(dataset,))
    )

    row = _sigma_projection(changed, kind)["scale"]

    assert row["sigma"] in ("", None)
    assert "candidate mismatch" in row["sigma_unavailable_reason"]
