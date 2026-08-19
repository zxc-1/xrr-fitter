from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from tests.unit.io.test_export_tables import PROJECT_REFERENCE, _context

from xrr_fitter.io import export_tables
from xrr_fitter.io.export_tables import DatasetExportData, dataset_json_bytes, json_text
from xrr_fitter.io.project_codec import project_to_dict


def _source_groups(payload: dict[str, object]) -> tuple[tuple[int, ...], ...]:
    raw_data = payload["raw_data"]
    return tuple(tuple(value) for value in raw_data["source_row_groups"])


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
        "project": {
            "path": context.project_reference.path,
            "size": context.project_reference.size,
            "sha256": context.project_reference.sha256,
        },
        "dataset_id": context.dataset.dataset_id,
        "source_sha256": context.data.source_sha256,
        "directory_mapping": {"curve": "001-curve-aaaaaaaa"},
        "fit_config": project_to_dict(context.project)["fit_config"],
        "service_seed_tree_version": 1,
        "independent_root_child": 10101,
        "joint_root_child": 20202,
        "mcmc_child_seed": None,
        "fitted_instrument_parameters": {"instrument.background": 2.5e-7},
    }
    assert observed == expected


def _assert_json_provenance(payload: dict[str, object], context: DatasetExportData) -> None:
    assert payload["model_residuals"]["qz_a_inv"] == pytest.approx(context.selected.qz_a_inv)
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


def test_export_json_does_not_reencode_the_complete_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    monkeypatch.setattr(
        export_tables,
        "project_to_dict",
        lambda _project: pytest.fail("dataset export re-encoded the complete project"),
        raising=False,
    )

    payload = json.loads(dataset_json_bytes(context))

    assert payload["dataset_id"] == context.dataset.dataset_id


def test_export_json_v2_keeps_dataset_provenance_shape() -> None:
    original = _context()
    dataset = replace(original.dataset, display_name="Measured curve")
    value = replace(original.project, datasets=(dataset,))
    context = replace(original, project=value, dataset=dataset)

    payload = json.loads(dataset_json_bytes(context))

    assert project_to_dict(value)["datasets"][0]["display_name"] == "Measured curve"
    assert payload["project"] == {
        "path": context.project_reference.path,
        "size": context.project_reference.size,
        "sha256": context.project_reference.sha256,
    }
    assert payload["raw_data"]["beam_kind"] == context.data.beam.kind
    assert "beam" not in payload["raw_data"]
    assert "candidate_count" not in payload["run_info"]
    assert "fit_mask" not in payload["model_residuals"]
    assert "diagnostics" not in payload["model_residuals"]


def test_export_json_v2_references_one_shared_project_snapshot() -> None:
    context = _context()

    payload = json.loads(dataset_json_bytes(context))

    assert tuple(payload) == (
        "export_schema_version",
        "dataset_id",
        "source_path",
        "source_sha256",
        "beam",
        "instrument",
        "scale_prior",
        "structure_evidence",
        "oxide_decisions",
        "raw_data",
        "model_residuals",
        "fit_result",
        "project",
        "candidates",
        "convergence",
        "run_info",
    )
    assert payload["export_schema_version"] == 2
    assert payload["project"] == {
        "path": PROJECT_REFERENCE.path,
        "size": PROJECT_REFERENCE.size,
        "sha256": PROJECT_REFERENCE.sha256,
    }
    assert "datasets" not in payload["project"]


def test_export_json_v2_preserves_the_documented_field_order() -> None:
    content = dataset_json_bytes(_context())
    payload = json.loads(content)

    assert tuple(payload) == (
        "export_schema_version",
        "dataset_id",
        "source_path",
        "source_sha256",
        "beam",
        "instrument",
        "scale_prior",
        "structure_evidence",
        "oxide_decisions",
        "raw_data",
        "model_residuals",
        "fit_result",
        "project",
        "candidates",
        "convergence",
        "run_info",
    )
    assert tuple(payload["model_residuals"]) == (
        "qz_a_inv",
        "model_normalized",
        "log_residuals_decades",
        "weighted_residuals",
        "candidate_id",
    )
    assert tuple(payload["run_info"]) == (
        "schema_version",
        "algorithm_version",
        "fit_config",
        "project_master_seed",
        "service_seed_tree_version",
        "independent_root_child",
        "joint_root_child",
        "optimizer_child_seeds",
        "mcmc_child_seed",
        "selected_candidate_id",
        "uncertainty_absent_reason",
        "fitted_instrument_parameters",
        "dataset_id",
        "source_path",
        "source_sha256",
        "beam",
        "instrument",
        "scale_prior",
        "structure_evidence",
        "oxide_decisions",
        "confidence",
        "warnings",
        "fringe_screen_threshold_version",
        "budget_reclaim_threshold_version",
        "downsample_rule_version",
        "jacobian_version",
        "dataset_directory",
        "dataset_directory_mapping",
    )
    assert json_text({"second": 2, "first": 1}) == '{"second": 2, "first": 1}'


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
        project_reference=original.project_reference,
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
        project_reference=original.project_reference,
    )

    payload = json.loads(dataset_json_bytes(context))

    assert payload["model_residuals"]["candidate_id"] == selected.candidate_id
    assert payload["model_residuals"]["model_normalized"] == pytest.approx(selected.model_normalized)
    assert payload["fit_result"]["best_index"] == 0
    assert value.ui_state.selected_candidate_ids == ((dataset.dataset_id, selected.candidate_id),)
    assert payload["project"]["path"] == PROJECT_REFERENCE.path


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
            project_reference=original.project_reference,
        )
