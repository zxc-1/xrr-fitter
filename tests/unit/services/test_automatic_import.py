from dataclasses import replace
from pathlib import Path

import numpy as np
from tests.support.model_cases import dataset_project, final_fit_result, fit_candidate, project

from xrr_fitter.io.xy import xy_bytes
from xrr_fitter.model.automation import (
    AutomaticRole,
    AutomaticStatus,
    MeasurementPreset,
)
from xrr_fitter.model.data import BeamSpec, DataColumnMapping
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.services.datasets import import_dataset_batch, preview_import_batch
from xrr_fitter.services.projects import new_project


def _curve(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        xy_bytes(
            np.linspace(0.1, 3.0, 32),
            np.geomspace(1.0, 1e-5, 32),
        )
    )
    return path


def _preset() -> MeasurementPreset:
    return MeasurementPreset(
        "cu-kalpha",
        BeamSpec(kind="monochromatic", wavelength_a=1.5406),
        InstrumentSpec(instrument_id="lab"),
    )


def test_preview_accepts_single_layer_and_reverses_multilayer_once(
    tmp_path: Path,
) -> None:
    paths = (
        _curve(tmp_path / "P1 Zr.xy"),
        _curve(tmp_path / "P2 Si3N4+Si+Zr.xy"),
    )

    preview = preview_import_batch(paths, _preset(), import_batch_id="batch-1")

    assert preview.files[0].layers_backing_to_surface == ("Zr",)
    assert preview.files[1].layers_backing_to_surface == ("Si3N4", "Si", "Zr")
    original = new_project()
    result = import_dataset_batch(original, preview)
    assert tuple(layer.name for layer in result.updated_project.datasets[1].structure.components[:3]) == (
        "Zr",
        "Si",
        "Si3N4",
    )
    assert original.datasets == ()
    assert original.measurement_preset is None
    assert result.updated_project.measurement_preset is preview.preset
    assert result.updated_project.ui_state.active_dataset_id == "P1"
    assert all(
        dataset.automation.import_batch_id == "batch-1"
        and dataset.automation.role is AutomaticRole.UNROUTED
        and dataset.automation.status is AutomaticStatus.PENDING
        and dataset.parameter_settings
        for dataset in result.updated_project.datasets
    )


def test_preview_uses_parent_folder_stack_for_exported_point_files(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "S300-1-260424-2 CrSiC+SiCMo+TaN"
    paths = (
        _curve(folder / "S300-1-260424-2 W2_exported.xy"),
        _curve(folder / "S300-1-260424-2 W02_exported.xy"),
    )

    preview = preview_import_batch(paths, _preset(), import_batch_id="batch-folder")

    assert tuple(
        (
            row.dataset_id_stem,
            row.layers_backing_to_surface,
            row.requires_substrate_choice,
        )
        for row in preview.files
    ) == (
        (
            "S300-1-260424-2 W2",
            ("CrSiC", "SiCMo", "TaN"),
            False,
        ),
        (
            "S300-1-260424-2 W02",
            ("CrSiC", "SiCMo", "TaN"),
            False,
        ),
    )

    result = import_dataset_batch(new_project(), preview)

    assert result.imported_dataset_ids == (
        "S300-1-260424-2 W2",
        "S300-1-260424-2 W02",
    )
    assert tuple(
        tuple(layer.name for layer in dataset.structure.components[:3])
        for dataset in result.updated_project.datasets
    ) == (("TaN", "SiCMo", "CrSiC"),) * 2


def test_leftmost_si_requests_one_substrate_choice_per_structure_group(
    tmp_path: Path,
) -> None:
    paths = (
        _curve(tmp_path / "P1 Si+Zr.xy"),
        _curve(tmp_path / "P2 Si+Zr.xy"),
    )

    preview = preview_import_batch(paths, _preset(), import_batch_id="batch-2")

    groups = {item.substrate_group_id for item in preview.files}
    assert len(groups) == 1
    assert all(item.requires_substrate_choice for item in preview.files)
    group_id = next(iter(groups))
    result = import_dataset_batch(new_project(), preview, {group_id: "Al2O3"})
    assert all(dataset.structure.backing.name == "Al2O3" for dataset in result.updated_project.datasets)


def test_bad_filename_and_bad_data_do_not_block_valid_files(tmp_path: Path) -> None:
    valid = _curve(tmp_path / "good Zr.xy")
    missing_stack = _curve(tmp_path / "missing-stack.xy")
    unreadable = tmp_path / "bad Zr.xy"
    unreadable.write_text("not numeric\n", encoding="utf-8")

    preview = preview_import_batch(
        (missing_stack, valid, unreadable),
        _preset(),
        import_batch_id="batch-3",
    )
    result = import_dataset_batch(new_project(), preview)

    assert result.imported_dataset_ids == ("good",)
    assert len(result.failures) == 2
    assert {Path(item.source_path).name for item in result.failures} == {
        "missing-stack.xy",
        "bad Zr.xy",
    }


def test_all_failures_leave_project_and_measurement_preset_unchanged(
    tmp_path: Path,
) -> None:
    invalid = _curve(tmp_path / "missing-stack.xy")
    project = new_project()
    preview = preview_import_batch((invalid,), _preset(), import_batch_id="batch-4")

    result = import_dataset_batch(project, preview)

    assert result.updated_project is project
    assert result.imported_dataset_ids == ()
    assert result.updated_project.measurement_preset is None
    assert result.failures[0].recovery_action == ("rename the file and retry or open manual structure editing")


def test_batch_allocates_duplicate_ids_and_uses_each_files_column_mapping(
    tmp_path: Path,
) -> None:
    paths = tuple(_curve(tmp_path / directory / "sample Zr.xy") for directory in ("a", "b"))
    mapping = DataColumnMapping(1, 0)
    preview = preview_import_batch(paths, _preset(), import_batch_id="batch-5")

    result = import_dataset_batch(
        new_project(),
        preview,
        column_mappings={str(paths[1]): mapping},
    )

    assert result.imported_dataset_ids == ("sample", "sample-2")
    assert tuple(dataset.column_mapping for dataset in result.updated_project.datasets) == (
        DataColumnMapping(),
        mapping,
    )
    assert result.updated_project.ui_state.active_dataset_id == "sample"


def test_unexpected_file_error_does_not_block_later_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = (
        _curve(tmp_path / "first Zr.xy"),
        _curve(tmp_path / "second Zr.xy"),
    )
    preview = preview_import_batch(paths, _preset(), import_batch_id="batch-6")
    from xrr_fitter.services import datasets

    original = datasets._automatic_dataset

    def fail_first(project, row, batch_preview, backing_token, column_mapping):
        if row.source_path == str(paths[0]):
            raise RuntimeError("file-specific failure")
        return original(
            project,
            row,
            batch_preview,
            backing_token,
            column_mapping,
        )

    monkeypatch.setattr(datasets, "_automatic_dataset", fail_first)

    result = import_dataset_batch(new_project(), preview)

    assert result.imported_dataset_ids == ("second",)
    assert result.failures[0].message == "RuntimeError: file-specific failure"


def test_import_into_completed_joint_project_invalidates_previous_joint_results(
    tmp_path: Path,
) -> None:
    prior = final_fit_result(replace(fit_candidate(), ranking_objective=1.0))
    original = replace(
        project(
            dataset_project("first", result=prior),
            dataset_project("second", result=prior),
        ),
        batch_mode="joint",
    )
    preview = preview_import_batch(
        (_curve(tmp_path / "third Zr.xy"),),
        _preset(),
        import_batch_id="batch-7",
    )

    result = import_dataset_batch(original, preview)

    assert result.imported_dataset_ids == ("third",)
    assert result.failures == ()
    assert all(
        dataset.last_valid_result is None and dataset.checkpoint is None for dataset in result.updated_project.datasets
    )
    assert original.datasets[0].last_valid_result is prior
