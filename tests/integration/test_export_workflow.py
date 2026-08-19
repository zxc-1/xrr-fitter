from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import xrr_fitter.api as api

ROOT = Path(__file__).resolve().parents[2]
DATASET_EXPORT_NAMES = {
    "fit_result.xlsx",
    "fit_result.json",
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
    dataset_id = value.datasets[0].dataset_id
    free_name = "component.0.thickness_a"
    value = api.set_parameter_settings(
        value,
        dataset_id,
        tuple(
            api.ParameterSetting(
                definition.name,
                definition.initial,
                definition.lower if definition.name == free_name else definition.initial,
                definition.upper if definition.name == free_name else definition.initial,
                locked=definition.name != free_name,
            )
            for definition in api.describe_parameters(value, dataset_id)
        ),
    )
    return api.fit_project(value).updated_project


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
        "root_files": tuple(record.path for record in manifest.root_files),
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

    assert _export_tree(manifest) == {
        "parent": tmp_path / "exports",
        "is_directory": True,
        "partial_directories": (),
        "dataset_order": (original.dataset_id, "second"),
        "root_files": (
            "batch_summary.xlsx",
            "compatibility_summary.xlsx",
            "parameter_trends.png",
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
