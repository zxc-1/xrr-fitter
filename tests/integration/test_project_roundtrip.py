from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

import xrr_fitter.api as api
from xrr_fitter.model.automation import (
    AutomaticRole,
    AutomaticStatus,
    DatasetAutomation,
    MeasurementPreset,
)
from xrr_fitter.model.data import BeamSpec


def _source(path: Path, scale: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    angles = np.linspace(0.1, 3.2, 32)
    path.write_text(
        "\n".join(
            f"{angle:.17g} {value:.17g}"
            for angle, value in zip(
                angles,
                scale * np.geomspace(1.0, 1e-5, angles.size),
                strict=True,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_project_roundtrip_preserves_allocated_ids_and_source_identity(
    tmp_path: Path,
) -> None:
    value = api.new_project()
    for directory, display, scale in (
        ("first", "first display", 1.0),
        ("second", "second display", 0.9),
    ):
        value = api.add_dataset(
            value,
            _source(tmp_path / directory / "sample.xy", scale),
            api.InstrumentSpec(instrument_id=directory),
            display_name=display,
        )
    automatic = DatasetAutomation(
        import_batch_id="batch-1",
        fit_group_id="group-1",
        role=AutomaticRole.JOINT,
        status=AutomaticStatus.PASSED,
        statistics_member=True,
    )
    value = replace(
        value,
        datasets=tuple(replace(dataset, automation=automatic) for dataset in value.datasets),
        measurement_preset=MeasurementPreset(
            "lab-cu-kalpha",
            BeamSpec(kind="monochromatic", wavelength_a=1.5406),
            api.InstrumentSpec(instrument_id="lab", footprint_mode="fit"),
            0.012,
        ),
    )
    target = tmp_path / "project.xrrproj.json"

    api.save_project(value, target)
    loaded = api.load_project(target)

    assert tuple(dataset.dataset_id for dataset in loaded.datasets) == (
        "sample",
        "sample-2",
    )
    assert tuple(dataset.display_name for dataset in loaded.datasets) == (
        "first display",
        "second display",
    )
    assert loaded == value
    assert loaded.base_directory == str(tmp_path)
    assert api.inspect_sources(loaded).valid is True
