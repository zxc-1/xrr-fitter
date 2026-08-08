"""GUI acceptance for filename-derived structures and one joint batch."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QLabel,
    QLineEdit,
    QPushButton,
    QTreeWidget,
)

import xrr_fitter.api as api
from tests.integration.test_gui_synthetic_xy_workflow import (
    _configure_import,
    _lock_all_but_first_thickness,
    _run_gui_fit,
    _run_modal,
    _select_dataset,
    _show_window,
    _write_synthetic_xy,
)


def _filename_batch_inputs(tmp_path: Path) -> tuple[Path, Path]:
    layers = tuple(
        api.LayerSpec(
            formula,
            api.MaterialSpec(formula, formula, density),
            100.0,
            roughness_a=3.0,
        )
        for formula, density in (
            ("Zr", 6.52),
            ("Si", 2.329),
            ("Si3N4", 3.17),
        )
    )
    return (
        _write_synthetic_xy(
            tmp_path / "S300-1_250904-2 Si3N4+Si+Zr.xy",
            layers,
        ),
        _write_synthetic_xy(
            tmp_path / "S300-2_250904-2 Si3N4+Si+Zr.xy",
            layers,
        ),
    )


def _import_filename_batch(window) -> None:
    import_button = window.findChild(QPushButton, "importFilesButton")
    _run_modal(
        lambda: QTest.mouseClick(import_button, Qt.MouseButton.LeftButton),
        "importDialog",
        _configure_import,
    )
    project = window.document.project
    snapshot = (
        tuple(dataset.dataset_id for dataset in project.datasets),
        tuple(dataset.display_name for dataset in project.datasets),
        tuple(dataset.column_mapping for dataset in project.datasets),
        tuple(
            tuple(layer.material.formula for layer in dataset.structure.components)
            for dataset in project.datasets
        ),
        window.findChild(QPushButton, "initializeStructureButton").isHidden(),
    )
    assert snapshot == (
        ("S300-1_250904-2", "S300-2_250904-2"),
        # The display name is the wafer point identifier, not the full stem: a
        # folder-declared stack keeps the point ID and drops the material stack,
        # which is what "preserve wafer import names" (3a37a39) established.
        ("S300-1_250904-2", "S300-2_250904-2"),
        (api.DataColumnMapping(0, 1), api.DataColumnMapping(0, 1)),
        # A Si substrate gains a 10 A SiO2 native oxide unless the adjacent layer
        # is already exactly SiO2. Here the substrate-side layer is Si3N4, so the
        # oxide is inserted and the stack carries four layers, surface first.
        (
            ("Zr", "Si", "Si3N4", "SiO2"),
            ("Zr", "Si", "Si3N4", "SiO2"),
        ),
        True,
    )
    # Equality, not object identity: joint routing groups datasets by physical
    # signature (services/batch.py `_component_signature`), which compares values.
    # Batch import builds each structure from the declared stack, so two rows with
    # the same stack are equal without necessarily being the same instance.
    assert project.datasets[0].structure == project.datasets[1].structure


def _select_joint_mode(window) -> None:
    selector = window.findChild(QComboBox, "batchModeSelector")
    selector.setCurrentIndex(selector.findData("joint"))
    QApplication.processEvents()
    assert window.document.project.batch_mode == "joint"


def _exercise_recoverable_layer_dialog(window) -> None:
    add_button = window.findChild(QPushButton, "addLayerButton")
    observed: list[tuple[bool, bool, str, tuple[int, ...]]] = []

    def configure(dialog: QDialog) -> None:
        dialog.findChild(QLineEdit, "layerNameInput").setText("validation-layer")
        dialog.findChild(QLineEdit, "layerFormulaInput").setText("SiO2")
        dialog.findChild(QDoubleSpinBox, "layerDensityInput").setValue(2.2)
        dialog.findChild(QDoubleSpinBox, "layerThicknessInput").setValue(1.0)
        roughness = dialog.findChild(QDoubleSpinBox, "layerRoughnessInput")
        roughness.setValue(1.0)
        buttons = dialog.findChild(QDialogButtonBox, "layerDialogButtons")
        ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        QTest.mouseClick(ok, Qt.MouseButton.LeftButton)
        QApplication.processEvents()
        error = dialog.findChild(QLabel, "layerDialogError")
        observed.append(
            (
                dialog.isVisible(),
                error.isVisible(),
                error.text(),
                tuple(
                    len(dataset.structure.components)
                    for dataset in window.document.project.datasets
                ),
            )
        )
        roughness.setValue(0.1)
        QTest.mouseClick(ok, Qt.MouseButton.LeftButton)

    _run_modal(
        lambda: QTest.mouseClick(add_button, Qt.MouseButton.LeftButton),
        "layerDialog",
        configure,
    )
    # The native oxide makes this a four-layer stack, so the rejected interface is
    # index 4 rather than 3.
    assert observed == [
        (
            True,
            True,
            "interface.4.roughness_a must be below 4.9 A",
            (4, 4),
        )
    ]
    assert _component_names(window) == (
        ("Zr", "Si", "Si3N4", "SiO2 native oxide", "validation-layer"),
        ("Zr", "Si", "Si3N4", "SiO2 native oxide", "validation-layer"),
    )

    # Locate the added layer by name: the tree also shows the ambient row and the
    # native oxide, so a fixed row index silently targets the wrong component
    # whenever the stack changes.
    tree = window.findChild(QTreeWidget, "structureTree")
    target = next(
        tree.topLevelItem(row)
        for row in range(tree.topLevelItemCount())
        if tree.topLevelItem(row).text(0) == "validation-layer"
    )
    tree.setCurrentItem(target)
    remove = window.findChild(QPushButton, "removeComponentButton")
    QTest.mouseClick(remove, Qt.MouseButton.LeftButton)
    QApplication.processEvents()
    assert _component_names(window) == (
        ("Zr", "Si", "Si3N4", "SiO2 native oxide"),
        ("Zr", "Si", "Si3N4", "SiO2 native oxide"),
    )


def _component_names(window) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(layer.name for layer in dataset.structure.components)
        for dataset in window.document.project.datasets
    )


def _prepare_filename_batch_parameters(window) -> None:
    tree = window.findChild(QTreeWidget, "datasetTree")
    for row in range(2):
        _select_dataset(tree, row)
        _lock_all_but_first_thickness(window, initial_nm=9.5)


def _assert_joint_progress(events: list[api.OperationEvent]) -> None:
    stages = tuple(
        event.progress.stage
        for event in events
        if event.kind == "progress" and event.progress is not None
    )
    finalizing = stages.index("finalizing")
    assert "E" in stages
    assert max(index for index, stage in enumerate(stages) if stage == "E") < finalizing
    assert stages[-1] == "finalizing"


@pytest.mark.spawn
def test_filename_batch_auto_structure_joint_gui_workflow(
    qtbot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _filename_batch_inputs(tmp_path)
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileNames",
        lambda *_args, **_kwargs: ([str(path) for path in sources], ""),
    )
    window = _show_window(qtbot, 1600, 900)
    _import_filename_batch(window)
    _select_joint_mode(window)
    _exercise_recoverable_layer_dialog(window)
    _prepare_filename_batch_parameters(window)
    events: list[api.OperationEvent] = []
    window.fit_panel.controller.event_received.connect(events.append)

    _run_gui_fit(window, qtbot)

    _assert_joint_progress(events)
    assert all(
        dataset.last_valid_result is not None
        for dataset in window.document.project.datasets
    )
