"""Deterministic export schema, replay identity, and workbook contracts.

Fixtures cover synthetic aligned curves, parser-owned raw provenance, and
nonempty stochastic replay state across strict JSON and workbook views.
"""

from __future__ import annotations

from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from openpyxl import load_workbook

from tests.support.model_cases import (
    dataset_project,
    final_fit_result,
    fit_candidate,
    prepared_data,
    project,
)
from xrr_fitter.io.export_tables import (
    DatasetExportData,
    ExportReplayIdentity,
    batch_workbook_bytes,
    compatibility_workbook_bytes,
    dataset_json_bytes,
    dataset_workbook_bytes,
    json_text,
    parameters_csv_bytes,
    run_log_bytes,
)
from xrr_fitter.io.project_codec import project_to_dict
from xrr_fitter.io.xy import read_xy_bytes
from xrr_fitter.model.analysis import McmcConfig, McmcReport, UncertaintyReport
from xrr_fitter.model.data import PreparedData
from xrr_fitter.model.fitting import FitStageSummary
from xrr_fitter.model.instrument import PhysicsDiagnostic
from xrr_fitter.model.parameters import ParameterDefinition, ParameterValue
from xrr_fitter.model.project import OxideDecision


SOURCE_FIXTURE = (
    Path(__file__).resolve().parents[2] / "fixtures/source/header_and_duplicates.xy"
)


def _context(dataset_id: str = "curve") -> DatasetExportData:
    data = prepared_data()
    model = data.intensity_normalized * 0.97
    residual = np.log10(model + data.r_floor) - np.log10(
        data.intensity_normalized + data.r_floor
    )
    source_candidate = fit_candidate()
    candidate = replace(
        source_candidate,
        parameters=(
            *source_candidate.parameters,
            ParameterValue("instrument.background", 2.5e-7, 0.0, 1.0e-5),
        ),
        qz_a_inv=data.qz_a_inv,
        model_normalized=model,
        log_residuals_decades=residual,
        weighted_residuals=residual / 0.05,
        sld_depth_a=np.array([0.0, 20.0, 40.0]),
        sld_profile_a2=np.array([0.0, 2.0e-5, 1.0e-5], dtype=complex),
    )
    result = final_fit_result(candidate)
    dataset = dataset_project(dataset_id, result=result)
    dataset = replace(
        dataset,
        source_path=data.source_path.as_posix(),
        source_sha256=data.source_sha256,
        fit_mask=tuple(bool(value) for value in data.fit_mask),
        fit_range_two_theta_deg=(
            float(data.two_theta_deg[0]),
            float(data.two_theta_deg[-1]),
        ),
    )
    value = project(dataset)
    return DatasetExportData(
        project=value,
        dataset=dataset,
        data=data,
        directory_mapping=((dataset_id, "001-curve-aaaaaaaa"),),
        selected=candidate,
        replay_identity=ExportReplayIdentity(1, 10101, 20202),
        matching_surface_oxide_rejection=False,
    )


def _source_groups(payload: dict[str, object]) -> tuple[tuple[int, ...], ...]:
    raw_data = payload["raw_data"]
    return tuple(tuple(value) for value in raw_data["source_row_groups"])


def _context_with_parameter_metadata() -> DatasetExportData:
    original = _context()
    definitions = (
        ParameterDefinition(
            "scale",
            "Scale",
            "",
            "instrument",
            1.1,
            0.25,
            2.5,
            "linear",
            False,
            sharing_key="shared-scale",
        ),
        ParameterDefinition(
            "instrument.background",
            "Background",
            "reflectivity",
            "instrument",
            1.0e-7,
            0.0,
            1.0e-5,
            "linear",
            False,
            expert_only=True,
        ),
    )
    result = replace(
        original.result,
        parameter_definitions=definitions,
        warnings=("review-warning",),
    )
    dataset = replace(original.dataset, last_valid_result=result)
    value = replace(original.project, datasets=(dataset,))
    return DatasetExportData(
        project=value,
        dataset=dataset,
        data=original.data,
        directory_mapping=original.directory_mapping,
        selected=original.selected,
        replay_identity=original.replay_identity,
        matching_surface_oxide_rejection=original.matching_surface_oxide_rejection,
    )


def _context_with_length_parameter(dataset_id: str = "curve") -> DatasetExportData:
    original = _context(dataset_id)
    selected = replace(
        original.selected,
        parameters=(
            *original.selected.parameters,
            ParameterValue("component.0.thickness_a", 25.0, 2.0, 100.0),
        ),
    )
    result = replace(
        original.result,
        candidates=(selected,),
        warnings=("length-warning",),
    )
    dataset = replace(original.dataset, last_valid_result=result)
    value = replace(original.project, datasets=(dataset,))
    return DatasetExportData(
        project=value,
        dataset=dataset,
        data=original.data,
        directory_mapping=original.directory_mapping,
        selected=selected,
        replay_identity=original.replay_identity,
        matching_surface_oxide_rejection=original.matching_surface_oxide_rejection,
    )


def _project_contexts(*dataset_ids: str) -> tuple[DatasetExportData, ...]:
    originals = tuple(_context(dataset_id) for dataset_id in dataset_ids)
    value = project(*(context.dataset for context in originals))
    mapping = tuple(
        (dataset_id, f"{index:03d}-dataset-aaaaaaaa")
        for index, dataset_id in enumerate(dataset_ids, start=1)
    )
    return tuple(
        DatasetExportData(
            project=value,
            dataset=context.dataset,
            data=context.data,
            directory_mapping=mapping,
            selected=context.selected,
            replay_identity=context.replay_identity,
            matching_surface_oxide_rejection=(
                context.matching_surface_oxide_rejection
            ),
        )
        for context in originals
    )


def _context_with_mcmc_diagnostics() -> DatasetExportData:
    original = _context()
    diagnostics = (
        PhysicsDiagnostic("review-code", "review diagnostic", (0, 2, 4)),
        PhysicsDiagnostic("surface_thin_layer_residual", "surface residual", (1,)),
    )
    uncertainty_diagnostic = PhysicsDiagnostic(
        "uncertainty-code",
        "uncertainty diagnostic",
        (3,),
    )
    selected = replace(original.selected, diagnostics=diagnostics)
    stage = FitStageSummary(
        stage="review-stage",
        candidate_ids=(selected.candidate_id,),
        best_objective=selected.objective,
        total_nfev=17,
        stop_reasons=("review-stop",),
    )
    mcmc = McmcReport(
        config=McmcConfig(walkers=2, burn_in=0, production_steps=2),
        child_seed=30303,
        parameter_names=("scale",),
        samples_physical=np.ones((2, 1)),
        log_probability=np.zeros(2),
        acceptance_fraction=np.full(2, 0.5),
        split_rhat=np.ones(1),
        effective_sample_size=np.ones(1),
        boundary_hits=(),
        candidate_id=selected.candidate_id,
    )
    uncertainty = UncertaintyReport(
        correlation_names=(),
        correlation_matrix=np.empty((0, 0)),
        profiles=(),
        bootstrap_intervals=(),
        bootstrap_failure_rate=0.0,
        boundary_hits=(),
        strong_correlations=(),
        systematic_residual=False,
        diagnostics=(uncertainty_diagnostic,),
        mcmc=mcmc,
        candidate_id=selected.candidate_id,
    )
    result = replace(
        original.result,
        candidates=(selected,),
        warnings=("review-warning",),
        child_seeds=(101, 202),
        stage_summaries=(stage,),
        uncertainty=uncertainty,
    )
    dataset = replace(
        original.dataset,
        last_valid_result=result,
        oxide_decisions=(
            OxideDecision("Si", "SiO2", "surface", False, "oxide-table-v1"),
        ),
    )
    value = replace(original.project, datasets=(dataset,))
    return DatasetExportData(
        project=value,
        dataset=dataset,
        data=original.data,
        directory_mapping=original.directory_mapping,
        selected=selected,
        replay_identity=original.replay_identity,
        matching_surface_oxide_rejection=True,
    )


def _context_from_prepared_data(data: PreparedData) -> DatasetExportData:
    original = _context()
    selected = replace(
        original.selected,
        qz_a_inv=data.qz_a_inv,
        model_normalized=np.maximum(data.intensity_normalized, data.r_floor),
        log_residuals_decades=np.zeros(data.qz_a_inv.size),
        weighted_residuals=np.zeros(data.qz_a_inv.size),
    )
    result = replace(original.result, candidates=(selected,))
    dataset = replace(
        original.dataset,
        source_path=data.source_path.as_posix(),
        source_sha256=data.source_sha256,
        beam=data.beam,
        import_angle_offset_deg=data.import_angle_offset_deg,
        column_mapping=data.column_mapping,
        fit_mask=tuple(bool(value) for value in data.fit_mask),
        fit_range_two_theta_deg=(
            float(data.two_theta_deg[0]),
            float(data.two_theta_deg[-1]),
        ),
        last_valid_result=result,
    )
    value = replace(original.project, datasets=(dataset,))
    return DatasetExportData(
        project=value,
        dataset=dataset,
        data=data,
        directory_mapping=original.directory_mapping,
        selected=selected,
        replay_identity=original.replay_identity,
        matching_surface_oxide_rejection=False,
    )


def _assert_json_identity(payload: dict[str, object], context: DatasetExportData) -> None:
    info = payload["run_info"]
    observed = {
        "project": payload["project"],
        "dataset_id": payload["dataset_id"],
        "source_sha256": payload["source_sha256"],
        "directory_mapping": info["dataset_directory_mapping"],
        "fit_config": info["fit_config"],
        "service_seed_tree_version": info["service_seed_tree_version"],
        "independent_root_child": info["independent_root_child"],
        "joint_root_child": info["joint_root_child"],
        "mcmc_child_seed": info["mcmc_child_seed"],
        "fitted_instrument_parameters": info["fitted_instrument_parameters"],
    }
    expected = {
        "project": project_to_dict(context.project),
        "dataset_id": context.dataset.dataset_id,
        "source_sha256": context.data.source_sha256,
        "directory_mapping": {"curve": "001-curve-aaaaaaaa"},
        "fit_config": payload["project"]["fit_config"],
        "service_seed_tree_version": 1,
        "independent_root_child": 10101,
        "joint_root_child": 20202,
        "mcmc_child_seed": None,
        "fitted_instrument_parameters": {"instrument.background": 2.5e-7},
    }
    assert observed == expected


def _assert_json_provenance(payload: dict[str, object], context: DatasetExportData) -> None:
    assert payload["model_residuals"]["qz_a_inv"] == pytest.approx(
        context.selected.qz_a_inv
    )
    observed = {
        "raw_rows": payload["raw_data"]["raw_rows"],
        "source_groups": _source_groups(payload),
        "beam": payload["beam"],
        "instrument": payload["instrument"],
        "scale_prior": payload["scale_prior"],
        "structure_evidence": payload["structure_evidence"],
        "oxide_decisions": payload["oxide_decisions"],
        "archived": payload["candidates"][0]["archived"],
        "convergence": payload["convergence"],
    }
    expected = {
        "raw_rows": list(context.data.raw_rows),
        "source_groups": context.data.source_row_groups,
        "beam": payload["run_info"]["beam"],
        "instrument": payload["run_info"]["instrument"],
        "scale_prior": payload["run_info"]["scale_prior"],
        "structure_evidence": payload["run_info"]["structure_evidence"],
        "oxide_decisions": payload["run_info"]["oxide_decisions"],
        "archived": False,
        "convergence": {
            "candidate_ids": [context.selected.candidate_id],
            "objectives": [context.selected.objective],
        },
    }
    assert observed == expected


def test_export_json_uses_project_codec_and_complete_provenance() -> None:
    context = _context()

    first = dataset_json_bytes(context)
    second = dataset_json_bytes(context)
    payload = json.loads(first)

    assert first == second
    assert first.endswith(b"\n")
    assert b"NaN" not in first and b"Infinity" not in first
    _assert_json_identity(payload, context)
    _assert_json_provenance(payload, context)


def test_export_json_encodes_nonfinite_excluded_points_as_null() -> None:
    original = _context()
    mask = np.array(original.data.fit_mask, copy=True)
    mask[0] = False
    data = replace(original.data, fit_mask=mask)
    candidate = original.selected
    model = np.array(candidate.model_normalized, copy=True)
    residual = np.array(candidate.log_residuals_decades, copy=True)
    weighted = np.array(candidate.weighted_residuals, copy=True)
    model[0] = residual[0] = weighted[0] = np.nan
    updated = replace(
        candidate,
        model_normalized=model,
        log_residuals_decades=residual,
        weighted_residuals=weighted,
    )
    result = replace(original.result, candidates=(updated,))
    dataset = replace(
        original.dataset,
        fit_mask=tuple(bool(value) for value in mask),
        last_valid_result=result,
    )
    value = replace(original.project, datasets=(dataset,))
    context = DatasetExportData(
        project=value,
        dataset=dataset,
        data=data,
        directory_mapping=original.directory_mapping,
        selected=updated,
        replay_identity=original.replay_identity,
        matching_surface_oxide_rejection=original.matching_surface_oxide_rejection,
    )

    payload = json.loads(dataset_json_bytes(context))

    assert payload["model_residuals"]["model_normalized"][0] is None
    assert payload["model_residuals"]["log_residuals_decades"][0] is None


def test_export_uses_explicit_selected_candidate_without_rewriting_result() -> None:
    original = _context()
    selected = replace(
        original.selected,
        candidate_id="candidate-1",
        objective=original.selected.objective + 1.0,
        model_normalized=original.selected.model_normalized * 0.9,
    )
    result = replace(
        original.result,
        candidates=(original.selected, selected),
        best_index=0,
    )
    dataset = replace(original.dataset, last_valid_result=result)
    ui_state = replace(
        original.project.ui_state,
        selected_candidate_ids=((dataset.dataset_id, selected.candidate_id),),
    )
    value = replace(original.project, datasets=(dataset,), ui_state=ui_state)
    context = DatasetExportData(
        project=value,
        dataset=dataset,
        data=original.data,
        directory_mapping=original.directory_mapping,
        selected=selected,
        replay_identity=original.replay_identity,
        matching_surface_oxide_rejection=original.matching_surface_oxide_rejection,
    )

    payload = json.loads(dataset_json_bytes(context))

    assert payload["model_residuals"]["candidate_id"] == selected.candidate_id
    assert payload["model_residuals"]["model_normalized"] == pytest.approx(
        selected.model_normalized
    )
    assert payload["fit_result"]["best_index"] == 0
    assert payload["project"]["ui_state"]["selected_candidate_ids"] == [
        [dataset.dataset_id, selected.candidate_id]
    ]


def test_export_rejects_selected_candidate_outside_persisted_result() -> None:
    original = _context()
    outsider = replace(original.selected)

    assert outsider.candidate_id == original.selected.candidate_id
    assert outsider is not original.selected

    with pytest.raises(ValueError, match="selected candidate must belong"):
        DatasetExportData(
            project=original.project,
            dataset=original.dataset,
            data=original.data,
            directory_mapping=original.directory_mapping,
            selected=outsider,
            replay_identity=original.replay_identity,
            matching_surface_oxide_rejection=original.matching_surface_oxide_rejection,
        )


def test_export_dataset_workbook_has_complete_aligned_sheets() -> None:
    context = _context()

    first = dataset_workbook_bytes(context)
    second = dataset_workbook_bytes(context)

    workbook = pd.ExcelFile(BytesIO(first))
    raw_data = pd.read_excel(BytesIO(first), sheet_name="RawData")
    model = pd.read_excel(BytesIO(first), sheet_name="ModelResiduals")
    required_raw_columns = {
        "row_index",
        "raw_row_index",
        "raw_row_text",
        "raw_parse_status",
        "source_row_group_json",
        "source_path",
        "source_sha256",
        "beam_json",
        "column_mapping_json",
        "normalization",
        "r_floor",
        "fit_ready",
        "warnings_json",
    }
    observed_layout = {
        "deterministic": first == second,
        "sheets": workbook.sheet_names,
        "raw_rows": len(raw_data),
        "model_rows": len(model),
        "raw_columns_present": required_raw_columns <= set(raw_data.columns),
        "fit_mask": model["fit_included"].tolist(),
    }
    expected_layout = {
        "deterministic": True,
        "sheets": [
            "Parameters",
            "Candidates",
            "RawData",
            "ModelResiduals",
            "Correlation",
            "Profiles",
            "RunInfo",
        ],
        "raw_rows": len(context.data.raw_rows),
        "model_rows": context.data.two_theta_deg.size,
        "raw_columns_present": True,
        "fit_mask": list(context.dataset.fit_mask),
    }
    assert observed_layout == expected_layout
    first_raw = raw_data.iloc[0]
    observed_provenance = {
        "source_path": first_raw["source_path"],
        "source_sha256": first_raw["source_sha256"],
        "beam": json.loads(first_raw["beam_json"]),
        "angle_offset": first_raw["import_angle_offset_deg"],
        "column_mapping": json.loads(first_raw["column_mapping_json"]),
        "normalization": first_raw["normalization"],
        "r_floor": first_raw["r_floor"],
        "fit_ready": bool(first_raw["fit_ready"]),
        "warnings": json.loads(first_raw["warnings_json"]),
    }
    expected_provenance = {
        "source_path": str(context.data.source_path),
        "source_sha256": context.data.source_sha256,
        "beam": payload_dataset_beam(context),
        "angle_offset": context.data.import_angle_offset_deg,
        "column_mapping": payload_column_mapping(context),
        "normalization": context.data.normalization,
        "r_floor": context.data.r_floor,
        "fit_ready": context.data.fit_ready,
        "warnings": list(context.data.warnings),
    }
    assert observed_provenance == expected_provenance


def test_export_workbook_preserves_real_parser_raw_provenance() -> None:
    source_bytes = SOURCE_FIXTURE.read_bytes()
    data = read_xy_bytes(
        source_bytes,
        source_path=SOURCE_FIXTURE.name,
        beam=_context().dataset.beam,
    )
    context = _context_from_prepared_data(data)

    raw_data = pd.read_excel(
        BytesIO(dataset_workbook_bytes(context)),
        sheet_name="RawData",
    )

    assert len(data.raw_rows) == 9
    assert data.two_theta_deg.size == 4
    assert len(raw_data) == len(data.raw_rows)
    assert raw_data["raw_row_text"].tolist() == list(data.raw_rows)
    assert raw_data["raw_parse_status"].tolist() == list(data.raw_parse_status)
    assert raw_data["row_index"].dropna().astype(int).tolist() == [0, 1, 2, 3]
    assert [
        json.loads(value)
        for value in raw_data["source_row_group_json"].dropna()
    ] == [[3], [4, 5], [7], [8]]

    metadata_columns = (
        "source_path",
        "source_sha256",
        "beam_json",
        "import_angle_offset_deg",
        "column_mapping_json",
        "normalization",
        "r_floor",
        "fit_ready",
        "warnings_json",
    )
    assert {
        column: int(raw_data[column].notna().sum())
        for column in metadata_columns
    } == {column: 1 for column in metadata_columns}
    first = raw_data.iloc[0]
    assert {
        "source_path": first["source_path"],
        "source_sha256": first["source_sha256"],
        "beam": json.loads(first["beam_json"]),
        "angle_offset": first["import_angle_offset_deg"],
        "column_mapping": json.loads(first["column_mapping_json"]),
        "normalization": first["normalization"],
        "r_floor": first["r_floor"],
        "fit_ready": bool(first["fit_ready"]),
        "warnings": json.loads(first["warnings_json"]),
    } == {
        "source_path": str(data.source_path),
        "source_sha256": data.source_sha256,
        "beam": payload_dataset_beam(context),
        "angle_offset": data.import_angle_offset_deg,
        "column_mapping": payload_column_mapping(context),
        "normalization": data.normalization,
        "r_floor": data.r_floor,
        "fit_ready": data.fit_ready,
        "warnings": list(data.warnings),
    }


def payload_dataset_beam(context: DatasetExportData) -> dict[str, object]:
    document = project_to_dict(context.project)
    return document["datasets"][0]["beam"]


def payload_column_mapping(context: DatasetExportData) -> dict[str, object]:
    document = project_to_dict(context.project)
    return document["datasets"][0]["column_mapping"]


def test_export_dataset_parameters_preserve_definition_metadata() -> None:
    context = _context_with_parameter_metadata()

    parameters = pd.read_excel(
        BytesIO(dataset_workbook_bytes(context)),
        sheet_name="Parameters",
    )

    scale = parameters.iloc[0]
    observed = {
        "columns": parameters.columns.tolist(),
        "name": scale["name"],
        "display_name": scale["display_name"],
        "category": scale["category"],
        "value": scale["value"],
        "lower": scale["lower"],
        "upper": scale["upper"],
        "transform": scale["transform"],
        "locked": bool(scale["locked"]),
        "integer": bool(scale["integer"]),
        "expert_only": bool(scale["expert_only"]),
        "sharing_key": scale["sharing_key"],
        "selected_candidate_id": scale["selected_candidate_id"],
        "background_expert_only": bool(parameters.iloc[1]["expert_only"]),
    }
    expected = {
        "columns": [
            "name",
            "display_name",
            "category",
            "value",
            "lower",
            "upper",
            "unit",
            "transform",
            "locked",
            "integer",
            "expert_only",
            "sharing_key",
            "selected_candidate_id",
        ],
        "name": "scale",
        "display_name": "Scale",
        "category": "instrument",
        "value": context.selected.parameters[0].value,
        "lower": context.selected.parameters[0].lower,
        "upper": context.selected.parameters[0].upper,
        "transform": "linear",
        "locked": False,
        "integer": False,
        "expert_only": False,
        "sharing_key": "shared-scale",
        "selected_candidate_id": context.selected.candidate_id,
        "background_expert_only": True,
    }
    assert observed == expected


def test_export_workbook_run_info_matches_json_and_keeps_strings_literal() -> None:
    context = _context("=1+1")
    payload = json.loads(dataset_json_bytes(context))
    workbook_bytes = dataset_workbook_bytes(context)
    row = pd.read_excel(BytesIO(workbook_bytes), sheet_name="RunInfo").iloc[0]

    assert row["dataset_id"] == "=1+1"
    assert json.loads(row["dataset_directory_mapping"]) == payload["run_info"][
        "dataset_directory_mapping"
    ]
    assert json.loads(row["beam"]) == payload["run_info"]["beam"]
    workbook = load_workbook(BytesIO(workbook_bytes), data_only=False)
    sheet = workbook["RunInfo"]
    columns = {cell.value: cell.column for cell in sheet[1]}
    dataset_cell = sheet.cell(row=2, column=columns["dataset_id"])
    assert dataset_cell.value == "=1+1"
    assert dataset_cell.data_type == "s"
    assert dataset_cell.hyperlink is None


def test_export_json_and_workbook_retain_nonempty_mcmc_replay_identity() -> None:
    context = _context_with_mcmc_diagnostics()
    payload = json.loads(dataset_json_bytes(context))
    row = pd.read_excel(
        BytesIO(dataset_workbook_bytes(context)),
        sheet_name="RunInfo",
    ).iloc[0]
    expected = {
        "fit_config": project_to_dict(context.project)["fit_config"],
        "service_seed_tree_version": 1,
        "independent_root_child": 10101,
        "joint_root_child": 20202,
        "optimizer_child_seeds": [101, 202],
        "mcmc_child_seed": 30303,
        "fitted_instrument_parameters": {"instrument.background": 2.5e-7},
    }

    json_info = payload["run_info"]
    observed_json = {key: json_info[key] for key in expected}
    observed_workbook = {
        "fit_config": json.loads(row["fit_config"]),
        "service_seed_tree_version": row["service_seed_tree_version"],
        "independent_root_child": row["independent_root_child"],
        "joint_root_child": row["joint_root_child"],
        "optimizer_child_seeds": json.loads(row["optimizer_child_seeds"]),
        "mcmc_child_seed": row["mcmc_child_seed"],
        "fitted_instrument_parameters": json.loads(
            row["fitted_instrument_parameters"]
        ),
    }
    assert observed_json == expected
    assert observed_workbook == expected


def test_export_workbook_json_codec_rejects_unknown_objects() -> None:
    with pytest.raises(TypeError):
        json_text({"unsupported": object()})


def test_export_csv_and_compatibility_workbook_are_deterministic() -> None:
    context = _context_with_length_parameter()

    csv_first = parameters_csv_bytes(context)
    csv_second = parameters_csv_bytes(context)
    workbook_first = compatibility_workbook_bytes((context,))
    workbook_second = compatibility_workbook_bytes((context,))

    workbook = pd.ExcelFile(BytesIO(workbook_first))
    summary = pd.read_excel(BytesIO(workbook_first), sheet_name="Summary")
    parameters_nm = pd.read_excel(BytesIO(workbook_first), sheet_name="Parameters_nm")
    curves = pd.read_excel(BytesIO(workbook_first), sheet_name="Curves")
    observed = {
        "csv_deterministic": csv_first == csv_second,
        "csv_has_crlf": b"\r\n" in csv_first,
        "csv_header": csv_first.splitlines()[0],
        "workbook_deterministic": workbook_first == workbook_second,
        "sheets": workbook.sheet_names,
        "summary_columns": summary.columns.tolist(),
        "warnings": json.loads(summary.loc[0, "warnings"]),
        "length_columns": parameters_nm.columns.tolist(),
        "value_angstrom": parameters_nm.loc[0, "value_angstrom"],
        "value_nm": parameters_nm.loc[0, "value_nm"],
    }
    expected = {
        "csv_deterministic": True,
        "csv_has_crlf": False,
        "csv_header": b"parameter_name,value,lower,upper",
        "workbook_deterministic": True,
        "sheets": ["Summary", "Parameters_nm", "Curves"],
        "summary_columns": [
            "dataset_id",
            "confidence",
            "objective",
            "selected_candidate_id",
            "warnings",
        ],
        "warnings": ["length-warning"],
        "length_columns": [
            "dataset_id",
            "parameter_name",
            "value_angstrom",
            "value_nm",
        ],
        "value_angstrom": 25.0,
        "value_nm": 2.5,
    }
    assert observed == expected
    np.testing.assert_allclose(curves["two_theta_deg"], context.data.two_theta_deg)
    np.testing.assert_allclose(curves["intensity_raw"], context.data.intensity_raw)
    np.testing.assert_allclose(curves["model_normalized"], context.selected.model_normalized)


def test_export_batch_parameters_put_dataset_identity_first() -> None:
    first, second = _project_contexts("first", "second")

    workbook = batch_workbook_bytes((second, first))
    parameters = pd.read_excel(BytesIO(workbook), sheet_name="Parameters")

    assert parameters.columns.tolist() == [
        "dataset_id",
        "parameter_name",
        "value",
        "lower",
        "upper",
    ]
    assert parameters["dataset_id"].tolist() == [
        "first",
        "first",
        "second",
        "second",
    ]


def test_export_batch_rejects_contexts_from_different_projects() -> None:
    with pytest.raises(ValueError, match="same project"):
        batch_workbook_bytes((_context("first"), _context("second")))


def test_export_batch_rejects_non_export_contexts() -> None:
    with pytest.raises(TypeError, match="DatasetExportData"):
        batch_workbook_bytes((object(),))


def test_export_batch_requires_every_project_dataset_once() -> None:
    first, second = _project_contexts("first", "second")

    with pytest.raises(ValueError, match="exactly once"):
        batch_workbook_bytes((first,))
    with pytest.raises(ValueError, match="exactly once"):
        batch_workbook_bytes((first, first))


def test_export_log_records_warnings_seed_tree_stages_and_diagnostics() -> None:
    context = _context_with_mcmc_diagnostics()

    text = run_log_bytes(context).decode("utf-8")

    expected_fragments = (
        "warning: review-warning",
        "optimizer_child_seeds: [101,202]",
        "stage review-stage:",
        'stop_reasons=["review-stop"]',
        "review-code",
        "uncertainty-code",
        "full_data_indices=[0,2,4]",
        "project_master_seed: 1201",
        "independent_root_child: 10101",
        "joint_root_child: 20202",
        "mcmc_child_seed: 30303",
        "suspected_unmodeled_surface_oxide_after_rejection",
    )
    missing = tuple(value for value in expected_fragments if value not in text)
    assert missing == ()


def test_export_log_ignores_unrelated_surface_oxide_rejection() -> None:
    original = _context()
    selected = replace(
        original.selected,
        diagnostics=(
            PhysicsDiagnostic(
                "surface_thin_layer_residual",
                "surface residual",
                (1,),
            ),
        ),
    )
    result = replace(original.result, candidates=(selected,))
    dataset = replace(
        original.dataset,
        last_valid_result=result,
        oxide_decisions=(
            OxideDecision("Cu", "Cu2O", "surface", False, "unrelated-version"),
        ),
    )
    value = replace(original.project, datasets=(dataset,))
    context = DatasetExportData(
        project=value,
        dataset=dataset,
        data=original.data,
        directory_mapping=original.directory_mapping,
        selected=selected,
        replay_identity=original.replay_identity,
        matching_surface_oxide_rejection=False,
    )

    text = run_log_bytes(context).decode("utf-8")

    assert "surface_thin_layer_residual" in text
    assert "suspected_unmodeled_surface_oxide_after_rejection" not in text


def test_export_log_retains_stale_diagnostic_indices_without_indexing_them() -> None:
    original = _context()
    stale_index = original.data.qz_a_inv.size + 4
    diagnostic = PhysicsDiagnostic(
        "stale-index",
        "retained diagnostic",
        (0, stale_index),
    )
    selected = replace(original.selected, diagnostics=(diagnostic,))
    result = replace(original.result, candidates=(selected,))
    dataset = replace(original.dataset, last_valid_result=result)
    value = replace(original.project, datasets=(dataset,))
    context = DatasetExportData(
        project=value,
        dataset=dataset,
        data=original.data,
        directory_mapping=original.directory_mapping,
        selected=selected,
        replay_identity=original.replay_identity,
        matching_surface_oxide_rejection=original.matching_surface_oxide_rejection,
    )

    text = run_log_bytes(context).decode("utf-8")

    qz = float(original.data.qz_a_inv[0])
    assert f"full_data_indices=[0,{stale_index}]" in text
    assert f"qz_a_inv_range=[{qz:.17g},{qz:.17g}]" in text
