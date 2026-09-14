from __future__ import annotations

import csv
import io
import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from orsopy import fileio
from orsopy.fileio.base import _read_header_data, _validate_header_data
from tests.support.export_sigma_cases import _assert_thickness_sigma, _single_free_setting

import xrr_fitter.api as api
from xrr_fitter.io.codec_results import fit_result_to_dict

ROOT = Path(__file__).resolve().parents[2]
DATASET_EXPORT_NAMES = {
    "fit_result.xlsx",
    "fit_result.json",
    "parameters.csv",
    "fit_overview.png",
    "sld_profile.png",
    "residuals.png",
    "run_log.txt",
}


def _fitted_project() -> api.XrrProject:
    value = api.load_project(ROOT / "examples/single-layer.xrrproj.json")
    budget = replace(
        value.fit_config.budget,
        short_de_maxiter=0,
        full_de_maxiter=0,
        local_min_nfev=5,
        local_nfev_per_parameter=1,
        bootstrap_samples=1,
    )
    value = replace(
        value,
        fit_config=replace(
            api.FitConfig.fast(value.master_seed),
            budget=budget,
            local_workers=1,
            scale_prior_enabled=False,
        ),
    )
    value = _thickness_only(value, value.datasets[0].dataset_id)
    return api.fit_project(value).updated_project


def _root_file_paths(manifest: api.ExportManifest) -> tuple[str, ...]:
    return tuple(record.path for record in manifest.root_files)


def _export_tree(manifest: api.ExportManifest) -> dict[str, object]:
    datasets = {}
    for dataset in manifest.datasets:
        document_path = next(
            manifest.run_directory / record.path for record in dataset.files if record.path.endswith("fit_result.json")
        )
        datasets[dataset.dataset_id] = {
            "files": {Path(record.path).name for record in dataset.files},
            "document_dataset_id": json.loads(document_path.read_text(encoding="utf-8"))["dataset_id"],
        }
    return {
        "parent": manifest.run_directory.parent,
        "is_directory": manifest.run_directory.is_dir(),
        "partial_directories": tuple(manifest.run_directory.parent.glob(".partial-*")),
        "dataset_order": tuple(item.dataset_id for item in manifest.datasets),
        "root_files": _root_file_paths(manifest),
        "datasets": datasets,
    }


def test_export_multi_dataset_writes_complete_atomic_artifact_tree(
    tmp_path: Path,
) -> None:
    fitted = _fitted_project()
    original = fitted.datasets[0]
    second = replace(original, dataset_id="second", display_name="second")
    value = replace(fitted, datasets=(original, second))

    manifest = api.export_result(value, tmp_path / "exports")

    snapshot = manifest.run_directory / "project_snapshot.xrrproj.json"
    assert snapshot.is_file()
    reopened = api.load_project(snapshot)
    assert tuple(item.dataset_id for item in reopened.datasets) == tuple(item.dataset_id for item in value.datasets)
    assert all(item.last_valid_result is not None for item in reopened.datasets)
    assert reopened.ui_state.selected_candidate_ids == value.ui_state.selected_candidate_ids
    assert all(Path(item.source_path).is_absolute() for item in reopened.datasets)

    assert _export_tree(manifest) == {
        "parent": tmp_path / "exports",
        "is_directory": True,
        "partial_directories": (),
        "dataset_order": (original.dataset_id, "second"),
        "root_files": (
            "batch_summary.xlsx",
            "compatibility_summary.xlsx",
            "export_manifest.json",
            "parameter_trends.png",
            "project_snapshot.xrrproj.json",
        ),
        "datasets": {
            original.dataset_id: {
                "files": DATASET_EXPORT_NAMES,
                "document_dataset_id": original.dataset_id,
            },
            "second": {
                "files": DATASET_EXPORT_NAMES,
                "document_dataset_id": "second",
            },
        },
    }


def test_export_publishes_parameter_csv_with_saved_evidence_and_manifest_hash(tmp_path: Path) -> None:
    fitted = _fitted_project()
    manifest = api.export_result(fitted, tmp_path / "csv-export")
    files = {Path(record.path).name: record for record in manifest.datasets[0].files}

    assert "parameters.csv" in files
    record = files["parameters.csv"]
    content = (manifest.run_directory / record.path).read_bytes()
    assert record.sha256 == sha256(content).hexdigest()
    assert record.size == len(content)
    rows = list(csv.DictReader(content.decode().splitlines()))
    assert rows[0]["noise_model"] == fitted.fit_config.noise_model
    inference = json.loads(rows[0]["inference_json"])
    assert inference["candidate_id"] == fitted.datasets[0].last_valid_result.uncertainty.candidate_id


def _mode_source(tmp_path: Path, noise_model: str) -> tuple[Path, api.DataColumnMapping]:
    values = np.loadtxt(ROOT / "examples/single-layer.xy")[::20]
    mapping = api.DataColumnMapping()
    if noise_model == "poisson":
        values[:, 1] = np.rint(values[:, 1] * 1e7)
    if noise_model == "gaussian":
        sigma = np.maximum(values[:, 1] * 0.02, 1e-7)
        values[-1, 1] = -sigma[-1]
        values = np.column_stack((values, sigma))
        mapping = api.DataColumnMapping(intensity_sigma=2)
    source = tmp_path / "observations.xy"
    np.savetxt(source, values, fmt="%.17g")
    return source, mapping


def _thickness_only(value: api.XrrProject, dataset_id: str) -> api.XrrProject:
    name = "component.0.thickness_a"
    settings = tuple(
        _single_free_setting(definition, name) for definition in api.describe_parameters(value, dataset_id)
    )
    return api.set_parameter_settings(value, dataset_id, settings)


def _mode_project(tmp_path: Path, noise_model: str, joint: bool) -> api.XrrProject:
    template = api.load_project(ROOT / "examples/single-layer.xrrproj.json")
    original = template.datasets[0]
    value = replace(template, datasets=(), ui_state=api.ProjectUiState())
    budget = replace(
        value.fit_config.budget,
        short_de_maxiter=0,
        full_de_maxiter=0,
        local_min_nfev=5,
        local_nfev_per_parameter=1,
        bootstrap_samples=8,
    )
    config = replace(
        api.FitConfig.fast(value.master_seed),
        budget=budget,
        profile_steps=5,
        local_workers=1,
        noise_model=noise_model,
        scale_prior_enabled=False,
    )
    value = api.set_fit_config(value, config)
    source, mapping = _mode_source(tmp_path, noise_model)
    for _ in range(2 if joint else 1):
        value = api.add_dataset(value, source, original.instrument, column_mapping=mapping, beam=original.beam)
        dataset_id = value.datasets[-1].dataset_id
        value = api.set_structure(value, dataset_id, original.structure)
        value = _thickness_only(value, dataset_id)
    if joint:
        rule = api.SharingRule(
            "shared-thickness",
            tuple(api.ParameterReference(dataset.dataset_id, "component.0.thickness_a") for dataset in value.datasets),
        )
        value = api.set_sharing_rules(value, (rule,))
        value = api.set_batch_mode(value, "joint")
    return value


def _export_inference_views(run_directory: Path, records) -> tuple[dict, ...]:
    paths = {Path(record.path).name: run_directory / record.path for record in records}
    payload = json.loads(paths["fit_result.json"].read_bytes())
    table = pd.read_excel(paths["fit_result.xlsx"], sheet_name="RunInfo").iloc[0]
    with paths["parameters.csv"].open(newline="", encoding="utf-8") as stream:
        csv_row = next(csv.DictReader(stream))
    text = paths["fit_result.ort"].read_text(encoding="utf-8")
    headers, _rows, _version = _read_header_data(io.StringIO(text))
    _validate_header_data(headers)
    orso = fileio.load_orso(io.StringIO(text))[0]
    lines = paths["run_log.txt"].read_text(encoding="utf-8").splitlines()
    log_metadata = next(line.removeprefix("inference: ") for line in lines if line.startswith("inference: "))
    return (
        payload["run_info"]["inference"],
        json.loads(table["inference"]),
        json.loads(csv_row["inference_json"]),
        orso.info.user_data["xrr_fitter.confidence"]["inference"],
        json.loads(log_metadata),
    )


def _assert_export_matches_saved_result(manifest: api.ExportManifest, dataset: api.DatasetProject) -> None:
    record = next(item for item in manifest.datasets if item.dataset_id == dataset.dataset_id)
    json_record = next(item for item in record.files if item.path.endswith("fit_result.json"))
    payload = json.loads((manifest.run_directory / json_record.path).read_bytes())
    encoded = json.loads(json.dumps(fit_result_to_dict(dataset.last_valid_result), allow_nan=False))
    views = _export_inference_views(manifest.run_directory, record.files)

    assert payload["fit_result"] == encoded
    assert all(view == views[0] for view in views)
    _assert_inference_summary(views[0], encoded["uncertainty"])
    selected = encoded["candidates"][encoded["best_index"]]
    assert payload["model_residuals"]["residuals"] == selected["residuals"]
    assert payload["model_residuals"]["noise_model"] == selected["noise_model"]


def _assert_inference_summary(summary: dict, encoded: dict) -> None:
    assert summary["parameter_members"] == encoded["parameter_members"]
    members = []
    for value in encoded["member_residuals"]:
        member = dict(value)
        member["diagnostic_count"] = len(member.pop("diagnostics"))
        advisories = []
        for item in member["advisories"]:
            advisory = dict(item)
            advisory["point_count"] = len(advisory.pop("point_indices"))
            advisories.append(advisory)
        member["advisories"] = advisories
        members.append(member)
    assert summary["member_residuals"] == members
    covariance = encoded["covariance_evidence"]
    assert summary["covariance_evidence"] == {key: value for key, value in covariance.items() if key != "matrix"}


@pytest.mark.parametrize("noise_model", ("robust_log", "gaussian", "poisson"))
@pytest.mark.parametrize("joint", (False, True))
def test_three_modes_fit_save_load_export_and_refit_keep_the_same_evidence(tmp_path, noise_model, joint) -> None:
    value = _mode_project(tmp_path, noise_model, joint)
    fitted = api.fit_project(value).updated_project
    project_path = tmp_path / "fitted.xrrproj.json"
    api.save_project(fitted, project_path)
    loaded = api.load_project(project_path)
    manifest = api.export_result(loaded, tmp_path / "exports", include_ort=True)

    assert loaded.fit_config.noise_model == noise_model
    assert loaded.algorithm_version == "xrr-fit-v2-poisson-5"
    assert loaded.schema_version == 5
    for dataset in loaded.datasets:
        _assert_export_matches_saved_result(manifest, dataset)
        _assert_thickness_sigma(manifest, dataset, "shared-thickness" if joint else "component.0.thickness_a")
    cleared = api.clear_fit_results(loaded, tuple(dataset.dataset_id for dataset in loaded.datasets))
    refitted = api.fit_project(cleared).updated_project
    assert [fit_result_to_dict(dataset.last_valid_result) for dataset in refitted.datasets] == [
        fit_result_to_dict(dataset.last_valid_result) for dataset in fitted.datasets
    ]
