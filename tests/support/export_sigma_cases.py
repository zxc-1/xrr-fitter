"""Fixed generated curves and exact saved-sigma artifact assertions.

Automatic observations use the declared forward curve and fixed seeds 17/18.
Reader normalization is applied to fixed scale/background, not to shared
material density. No generated or design-fit uncertainty is reused.
"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from orsopy import fileio

import xrr_fitter.api as api

ROOT = Path(__file__).resolve().parents[2]


def _single_free_setting(definition, name: str, normalization: float = 1.0) -> api.ParameterSetting:
    initial = definition.initial
    if definition.name in {"instrument.scale", "instrument.background"}:
        initial /= normalization
    return api.ParameterSetting(
        definition.name,
        initial,
        definition.lower if definition.name == name else initial,
        definition.upper if definition.name == name else initial,
        locked=definition.name != name,
    )


def _sigma_rows(rows, name_field: str) -> dict[str, float | None]:
    return {row[name_field]: None if row["sigma"] in ("", None) else float(row["sigma"]) for row in rows}


def _parameter_sigma_views(manifest: api.ExportManifest, dataset_id: str) -> dict[str, dict]:
    record = next(item for item in manifest.datasets if item.dataset_id == dataset_id)
    paths = {Path(item.path).name: manifest.run_directory / item.path for item in record.files}
    csv_rows = list(csv.DictReader(paths["parameters.csv"].read_text(encoding="utf-8").splitlines()))
    dataset = pd.read_excel(paths["fit_result.xlsx"], sheet_name="Parameters", keep_default_na=False)
    orso = fileio.load_orso(paths["fit_result.ort"])[0].info.user_data["xrr_fitter.confidence"]["parameters"]
    views = {
        "csv": _sigma_rows(csv_rows, "parameter_name"),
        "dataset_xlsx": _sigma_rows(dataset.to_dict("records"), "name"),
        "orso": _sigma_rows(orso, "name"),
    }
    if len(manifest.datasets) > 1:
        batch = pd.read_excel(
            manifest.run_directory / "batch_summary.xlsx", sheet_name="Parameters", keep_default_na=False
        )
        views["batch_xlsx"] = _sigma_rows(batch[batch["dataset_id"] == dataset_id].to_dict("records"), "parameter_name")
    return views


def _assert_saved_sigmas(manifest: api.ExportManifest, dataset_id: str, expected: dict) -> None:
    for kind, view in _parameter_sigma_views(manifest, dataset_id).items():
        # XlsxWriter's existing numeric cell encoding is 16G; this is an exact
        # file-format projection, not a tolerance on the fitted uncertainty.
        serialized = expected
        if kind.endswith("xlsx"):
            serialized = {
                name: None if sigma is None else float(format(sigma, ".16g")) for name, sigma in expected.items()
            }
        assert view == serialized, kind


def _assert_thickness_sigma(manifest: api.ExportManifest, dataset: api.DatasetProject, axis_name: str) -> None:
    report = dataset.last_valid_result.uncertainty
    expected = {parameter.name: None for parameter in dataset.last_valid_result.best_candidate.parameters}
    if report.covariance is not None:
        expected["component.0.thickness_a"] = float(report.parameter_sigma[report.correlation_names.index(axis_name)])
    _assert_saved_sigmas(manifest, dataset.dataset_id, expected)


def _automatic_sigma_design(tmp_path: Path):
    template = api.load_project(ROOT / "examples/single-layer.xrrproj.json")
    original = template.datasets[0]
    instrument = replace(original.instrument, footprint_mode="none")
    budget = replace(
        template.fit_config.budget,
        short_de_maxiter=0,
        full_de_maxiter=0,
        local_min_nfev=20,
        local_nfev_per_parameter=1,
        bootstrap_samples=8,
    )
    config = replace(
        api.FitConfig.fast(1701),
        budget=budget,
        profile_steps=5,
        local_workers=1,
        noise_model="gaussian",
        scale_prior_enabled=False,
    )
    base = replace(template, datasets=(), ui_state=api.ProjectUiState(), fit_config=config)
    angles = np.linspace(0.1, 3.2, 80)
    source = tmp_path / "design.xy"
    np.savetxt(source, np.column_stack((angles, np.ones(80), np.full(80, 0.02))), fmt="%.17g")
    value = api.add_dataset(
        base, source, instrument, beam=original.beam, column_mapping=api.DataColumnMapping(intensity_sigma=2)
    )
    dataset_id = value.datasets[0].dataset_id
    value = api.set_structure(value, dataset_id, original.structure)
    settings = tuple(
        api.ParameterSetting(definition.name, definition.initial, definition.initial, definition.initial, locked=True)
        for definition in api.describe_parameters(value, dataset_id)
    )
    operation = api.fit_project(api.set_parameter_settings(value, dataset_id, settings))
    result = operation.updated_project.datasets[0].last_valid_result
    assert result is not None, operation.warnings
    # Locked design parameters publish the declared forward curve regardless of
    # search candidate deduplication; its uncertainty is not reused below.
    assert result.best_candidate is not None
    prediction = result.best_candidate.model_normalized
    return base, original, instrument, angles, prediction


def _add_automatic_sigma_member(value, source, original, instrument, normalization):
    value = api.add_dataset(
        value, source, instrument, beam=original.beam, column_mapping=api.DataColumnMapping(intensity_sigma=2)
    )
    dataset_id = value.datasets[-1].dataset_id
    value = api.set_structure(value, dataset_id, original.structure)
    settings = tuple(
        _single_free_setting(definition, "component.0.density_scale", normalization)
        for definition in api.describe_parameters(value, dataset_id)
    )
    return api.set_parameter_settings(value, dataset_id, settings)


def _automatic_sigma_project(tmp_path: Path) -> api.XrrProject:
    value, original, instrument, angles, prediction = _automatic_sigma_design(tmp_path)
    sigma = np.maximum(prediction * 0.02, 1e-8)
    for seed in (17, 18):
        observed = prediction + np.random.default_rng(seed).normal(0, sigma)
        source = tmp_path / f"auto-{seed}.xy"
        np.savetxt(source, np.column_stack((angles, observed, sigma)), fmt="%.17g")
        data = api.import_data(
            source, original.beam, column_mapping=api.DataColumnMapping(intensity_sigma=2), noise_model="gaussian"
        )
        value = _add_automatic_sigma_member(value, source, original, instrument, data.normalization)
    automation = api.DatasetAutomation(
        import_batch_id="sigma-auto", role=api.AutomaticRole.UNROUTED, status=api.AutomaticStatus.PENDING
    )
    return replace(
        value,
        measurement_preset=api.MeasurementPreset("sigma-auto", original.beam, instrument),
        datasets=tuple(replace(dataset, automation=automation) for dataset in value.datasets),
    )
