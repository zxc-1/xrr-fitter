from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from tests.support.model_cases import final_fit_result, fit_candidate, simple_structure

from xrr_fitter.io.xy import xy_bytes
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.services import exports
from xrr_fitter.services.datasets import add_dataset
from xrr_fitter.services.projects import inspect_sources, new_project
from xrr_fitter.services.structures import set_structure

DATASET_FILES = (
    "fit_overview.png",
    "fit_result.json",
    "fit_result.xlsx",
    "residuals.png",
    "run_log.txt",
    "sld_profile.png",
)

# ``DatasetArtifacts`` sorts payloads by path, so the opt-in ``.ort`` lands
# between ``fit_result.json`` and ``fit_result.xlsx`` (j < o < x).
DATASET_FILES_WITH_ORT = (
    "fit_overview.png",
    "fit_result.json",
    "fit_result.ort",
    "fit_result.xlsx",
    "residuals.png",
    "run_log.txt",
    "sld_profile.png",
)


def _fitted_project(tmp_path: Path):
    source = tmp_path / "curve.xy"
    angles = np.linspace(0.1, 3.2, 40)
    source.write_bytes(xy_bytes(angles, np.geomspace(1.0, 1e-5, angles.size)))
    value = add_dataset(
        new_project(),
        source,
        InstrumentSpec(instrument_id="export-service", footprint_mode="none"),
    )
    value = set_structure(value, "curve", simple_structure())
    data = exports.load_export_data(value, value.datasets[0])
    candidate = replace(
        fit_candidate(),
        qz_a_inv=data.qz_a_inv,
        model_normalized=data.intensity_normalized,
        log_residuals_decades=np.zeros(data.qz_a_inv.size),
        weighted_residuals=np.zeros(data.qz_a_inv.size),
    )
    result = final_fit_result(candidate)
    return replace(value, datasets=(replace(value.datasets[0], last_valid_result=result),))


def _stub_serializers(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "dataset_json_bytes",
        "dataset_workbook_bytes",
        "fit_overview_png",
        "sld_profile_png",
        "residuals_png",
        "run_log_bytes",
        "compatibility_workbook_bytes",
        "batch_workbook_bytes",
        "parameter_trends_png",
    ):
        monkeypatch.setattr(exports, name, lambda *_args, _name=name: _name.encode())


def test_export_builds_the_fixed_artifact_set_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _fitted_project(tmp_path)
    captured: dict[str, object] = {}
    _stub_serializers(monkeypatch)

    def publish(output_dir, datasets, root_files):
        captured.update(output_dir=output_dir, datasets=datasets, root_files=root_files)
        return "manifest"

    monkeypatch.setattr(exports, "publish_export_run", publish)

    result = exports.export_result(value, tmp_path / "exports")

    assert result == "manifest"
    dataset = captured["datasets"][0]
    assert tuple(item.path for item in dataset.files) == DATASET_FILES
    assert tuple(item.path for item in captured["root_files"]) == ("compatibility_summary.xlsx",)


def test_export_rechecks_source_after_initial_inspection_before_allocating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _fitted_project(tmp_path)
    source = Path(value.datasets[0].source_path)
    original_inspect = inspect_sources

    def inspect_then_mutate(project):
        validation = original_inspect(project)
        source.write_bytes(source.read_bytes() + b"\n0.0 1.0\n")
        return validation

    monkeypatch.setattr(exports, "inspect_sources", inspect_then_mutate)
    monkeypatch.setattr(
        exports,
        "publish_export_run",
        lambda *_args, **_kwargs: pytest.fail("stale export allocated a run"),
    )

    with pytest.raises(ValueError, match="source|hash|changed"):
        exports.export_result(value, tmp_path / "exports")

    assert not (tmp_path / "exports").exists()


def _capture_publish(captured: dict[str, object]):
    def publish(output_dir, datasets, root_files):
        captured.update(output_dir=output_dir, datasets=datasets, root_files=root_files)
        return "manifest"

    return publish


def test_export_omits_ort_when_not_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _fitted_project(tmp_path)
    captured: dict[str, object] = {}
    _stub_serializers(monkeypatch)
    monkeypatch.setattr(exports, "publish_export_run", _capture_publish(captured))

    exports.export_result(value, tmp_path / "exports", include_ort=False)

    dataset = captured["datasets"][0]
    assert tuple(item.path for item in dataset.files) == DATASET_FILES


def test_export_appends_single_ort_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _fitted_project(tmp_path)
    captured: dict[str, object] = {}
    _stub_serializers(monkeypatch)
    monkeypatch.setattr(exports, "publish_export_run", _capture_publish(captured))
    monkeypatch.setattr(exports, "orso_bytes", lambda *_args, **_kwargs: b"orso-document", raising=False)

    exports.export_result(value, tmp_path / "exports", include_ort=True)

    dataset = captured["datasets"][0]
    assert tuple(item.path for item in dataset.files) == DATASET_FILES_WITH_ORT
    assert tuple(item.path for item in captured["root_files"]) == ("compatibility_summary.xlsx",)
