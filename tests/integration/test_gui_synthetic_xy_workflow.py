"""End-to-end GUI workflow over generated standard two-column XY files."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from matplotlib.backend_bases import MouseEvent
from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QLineEdit,
    QListWidget,
    QPushButton,
    QTreeWidget,
)

import xrr_fitter.api as api
from xrr_fitter.gui.results.candidates import candidate_is_selectable
from xrr_fitter.io.xy import xy_bytes
from xrr_fitter.physics.reflectivity import instrument_reflectivity
from xrr_fitter.physics.stack import expand_structure

AIR = api.MaterialSpec("Air", None, None, 0.0j)
SI = api.MaterialSpec("Si", "Si", 2.329)
SIO2 = api.MaterialSpec("SiO2", "SiO2", 2.20)
AL2O3 = api.MaterialSpec("Al2O3", "Al2O3", 3.95)
BEAM = api.BeamSpec("monochromatic")


def _write_synthetic_xy(
    path: Path,
    components: tuple[api.LayerSpec, ...],
) -> Path:
    theta_deg = np.linspace(0.03, 3.0, 180)
    structure = api.StructureSpec(AIR, components, SI)
    intensity = instrument_reflectivity(
        theta_deg,
        expand_structure(structure, BEAM.wavelength_a),
        BEAM,
    )
    path.write_bytes(xy_bytes(2.0 * theta_deg, intensity))
    return path


def _run_modal(
    trigger,
    object_name: str,
    configure,
) -> None:
    seen: list[QDialog] = []
    errors: list[BaseException] = []

    def interact() -> None:
        dialog = QApplication.activeModalWidget()
        if dialog is None or dialog.objectName() != object_name:
            QTimer.singleShot(10, interact)
            return
        seen.append(dialog)
        try:
            configure(dialog)
        except BaseException as error:
            errors.append(error)
            dialog.reject()

    QTimer.singleShot(0, interact)
    trigger()
    QApplication.processEvents()
    if errors:
        raise errors[0]
    assert seen, f"modal dialog did not open: {object_name}"


def _run_modals(trigger, steps) -> None:
    """Drive a fixed sequence of nested modal dialogs opened by one trigger."""
    seen: list[str] = []
    errors: list[BaseException] = []
    pending = list(steps)

    def interact() -> None:
        if not pending:
            return
        object_name, configure = pending[0]
        dialog = QApplication.activeModalWidget()
        if dialog is None or dialog.objectName() != object_name:
            QTimer.singleShot(10, interact)
            return
        pending.pop(0)
        seen.append(object_name)
        try:
            configure(dialog)
        except BaseException as error:
            errors.append(error)
            dialog.reject()
            return
        if pending:
            QTimer.singleShot(10, interact)

    QTimer.singleShot(0, interact)
    trigger()
    QApplication.processEvents()
    if errors:
        raise errors[0]
    assert seen == [name for name, _ in steps], f"modal sequence incomplete: {seen}"


def _select_dataset(tree: QTreeWidget, row: int) -> str:
    item = tree.topLevelItem(row)
    dataset_id = str(item.data(0, Qt.ItemDataRole.UserRole))
    tree.scrollToItem(item)
    QTest.mouseClick(
        tree.viewport(),
        Qt.MouseButton.LeftButton,
        pos=QPoint(10, tree.visualItemRect(item).center().y()),
    )
    QApplication.processEvents()
    return dataset_id


def _add_layer(window, layer: api.LayerSpec) -> None:
    button = window.findChild(QPushButton, "addLayerButton")
    assert button is not None and button.isEnabled()

    def configure(dialog: QDialog) -> None:
        dialog.findChild(QLineEdit, "layerNameInput").setText(layer.name)
        dialog.findChild(QLineEdit, "layerFormulaInput").setText(layer.material.formula)
        dialog.findChild(QDoubleSpinBox, "layerDensityInput").setValue(layer.material.bulk_density_g_cm3)
        dialog.findChild(QDoubleSpinBox, "layerThicknessInput").setValue(layer.thickness_a / 10.0)
        dialog.findChild(QDoubleSpinBox, "layerRoughnessInput").setValue(layer.roughness_a / 10.0)
        buttons = dialog.findChild(QDialogButtonBox, "layerDialogButtons")
        QTest.mouseClick(
            buttons.button(QDialogButtonBox.StandardButton.Ok),
            Qt.MouseButton.LeftButton,
        )

    _run_modal(
        lambda: QTest.mouseClick(button, Qt.MouseButton.LeftButton),
        "layerDialog",
        configure,
    )


def _configure_import(dialog: QDialog) -> None:
    dialog.mono_button.click()
    QApplication.processEvents()
    instrument_id = dialog.findChild(QLineEdit, "instrumentIdEditor")
    instrument_id.setText("synthetic-gui")
    footprint = dialog.findChild(QComboBox, "footprintModeEditor")
    footprint.setCurrentIndex(footprint.findData("none"))
    assert dialog.column_mapping() is None
    assert dialog.import_button().isEnabled()
    dialog.import_button().click()


def _lock_all_but_first_thickness(window, initial_nm: float) -> None:
    panel = window.parameters_panel
    expert = panel.findChild(QCheckBox, "expertModeToggle")
    if not expert.isChecked():
        QTest.mouseClick(expert, Qt.MouseButton.LeftButton)
        QApplication.processEvents()
    table = panel.parameter_table
    free_name = "component.0.thickness_a"
    row = panel.row_names.index(free_name)
    table.item(row, 1).setText(f"{initial_nm:.8g}")
    QApplication.processEvents()
    for name in tuple(panel.row_names):
        if not name:
            # Group caption rows carry no parameter and no lock cell; row_names
            # holds a blank placeholder for them to keep physical row indices
            # aligned, so there is nothing to lock here.
            continue
        row = panel.row_names.index(name)
        item = table.item(row, 5)
        desired = Qt.CheckState.Unchecked if name == free_name else Qt.CheckState.Checked
        if item.checkState() != desired:
            item.setCheckState(desired)
            QApplication.processEvents()
    definitions = api.describe_parameters(
        window.document.project,
        window.document.active_dataset_id,
    )
    assert tuple(item.name for item in definitions if not item.locked) == (free_name,)


def _mouse_event(name: str, window, xdata: float) -> MouseEvent:
    view = window.plot_panel.view("raw")
    view.canvas.draw()
    ydata = float(np.nanmedian(view.axes.lines[0].get_ydata()))
    x, y = view.axes.transData.transform((xdata, ydata))
    return MouseEvent(name, view.canvas, x, y, button=1)


def _drag_fit_range(window, lower: float, upper: float) -> None:
    canvas = window.plot_panel.view("raw").canvas
    canvas.callbacks.process(
        "button_press_event",
        _mouse_event("button_press_event", window, lower),
    )
    canvas.callbacks.process(
        "motion_notify_event",
        _mouse_event("motion_notify_event", window, upper),
    )
    canvas.callbacks.process(
        "button_release_event",
        _mouse_event("button_release_event", window, upper),
    )


def _click_point_mask(window, angle: float) -> None:
    canvas = window.plot_panel.view("raw").canvas
    canvas.callbacks.process(
        "button_press_event",
        _mouse_event("button_press_event", window, angle),
    )


def _install_fast_fit_config(window) -> None:
    project = window.document.project
    budget = replace(
        project.fit_config.budget,
        short_de_maxiter=0,
        full_de_maxiter=0,
        local_min_nfev=5,
        local_nfev_per_parameter=1,
        bootstrap_samples=1,
    )
    config = replace(
        api.FitConfig.fast(project.master_seed),
        budget=budget,
        local_workers=1,
        scale_prior_enabled=False,
    )
    window.document.replace_project(replace(project, fit_config=config))


def _workflow_inputs(
    tmp_path: Path,
) -> tuple[
    tuple[Path, Path],
    tuple[tuple[api.LayerSpec, ...], tuple[api.LayerSpec, ...]],
    Path,
    Path,
]:
    single_layer = api.LayerSpec("SiO2", SIO2, 85.0, roughness_a=3.0)
    double_layers = (
        api.LayerSpec("Al2O3", AL2O3, 45.0, roughness_a=2.0),
        api.LayerSpec("SiO2", SIO2, 110.0, roughness_a=3.0),
    )
    sources = (
        _write_synthetic_xy(tmp_path / "S1 SiO2.xy", (single_layer,)),
        _write_synthetic_xy(tmp_path / "S2 SiO2+Al2O3.xy", double_layers),
    )
    return (
        sources,
        ((single_layer,), double_layers),
        tmp_path / "synthetic-workflow.xrrproj.json",
        tmp_path / "exports",
    )


def _patch_file_dialogs(
    monkeypatch: pytest.MonkeyPatch,
    sources: tuple[Path, Path],
    project_path: Path,
    export_root: Path,
) -> None:
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileNames",
        lambda *_args, **_kwargs: ([str(path) for path in sources], ""),
    )
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(project_path), ""),
    )
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (str(project_path), ""),
    )
    monkeypatch.setattr(
        QFileDialog,
        "getExistingDirectory",
        lambda *_args, **_kwargs: str(export_root),
    )


def _show_window(qtbot, width: int, height: int):
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(width, height)
    window.show()
    # This workflow drives the panels directly, so it needs the expert surface;
    # the guided opening surface deliberately hides the docks that hold them.
    window.set_guidance_visible(False)
    qtbot.wait(1)
    return window


def _import_sources(window) -> None:
    import_button = window.findChild(QPushButton, "importFilesButton")
    _run_modal(
        lambda: QTest.mouseClick(import_button, Qt.MouseButton.LeftButton),
        "importDialog",
        _configure_import,
    )

    assert window.data_panel.dataset_ids == ("S1", "S2")
    assert all(dataset.column_mapping == api.DataColumnMapping(0, 1) for dataset in window.document.project.datasets)
    assert all(window.data_panel.status_text(dataset_id) == "可拟合" for dataset_id in window.data_panel.dataset_ids)


def _adopt_dataset_structures(
    window,
    expected_layers: tuple[
        tuple[api.LayerSpec, ...],
        tuple[api.LayerSpec, ...],
    ],
) -> None:
    """Confirm the filename-declared stack, then narrow the fitted parameters.

    These filenames declare their layer stacks, so import builds each structure
    on the way in and the manual "initialize structure" entry is correctly
    absent. The workflow therefore adopts what import produced and asserts the
    declared layer order survived, rather than rebuilding it by hand.
    """
    tree = window.findChild(QTreeWidget, "datasetTree")
    for row, layers in enumerate(expected_layers):
        dataset_id = _select_dataset(tree, row)
        assert window.document.active_dataset_id == dataset_id
        components = window.structure_panel.structure.components
        assert tuple(component.name for component in components) == tuple(layer.name for layer in layers)
        _lock_all_but_first_thickness(
            window,
            initial_nm=layers[0].thickness_a / 10.0 * 0.95,
        )


def _edit_plot_mask(window, source: Path) -> None:
    active = window.document.project.datasets[1]
    prepared = api.import_data(source, BEAM)
    lower = float(prepared.two_theta_deg[5])
    upper = float(prepared.two_theta_deg[-6])
    range_button = window.plot_panel.mode_buttons()["range"]
    QTest.mouseClick(range_button, Qt.MouseButton.LeftButton)
    _drag_fit_range(window, lower, upper)
    stored_range = window.document.project.datasets[1].fit_range_two_theta_deg
    assert lower <= stored_range[0] < stored_range[1] <= upper
    assert np.any(np.isclose(prepared.two_theta_deg, stored_range[0]))
    assert np.any(np.isclose(prepared.two_theta_deg, stored_range[1]))

    mask_button = window.plot_panel.mode_buttons()["mask"]
    QTest.mouseClick(mask_button, Qt.MouseButton.LeftButton)
    masked_index = 20
    _click_point_mask(window, float(prepared.two_theta_deg[masked_index]))
    assert window.document.project.datasets[1].fit_mask[masked_index] is False
    assert window.document.project.datasets[1].source_sha256 == active.source_sha256


def _run_gui_fit(window, qtbot) -> None:
    _install_fast_fit_config(window)
    readiness = api.preflight_fit(window.document.project)
    assert readiness == api.FitReadiness(True, "ready")
    start = window.findChild(QPushButton, "startFitButton")
    assert start is not None and start.isEnabled()
    QTest.mouseClick(start, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(
        lambda: (
            not window.fit_panel.is_running
            and all(dataset.last_valid_result is not None for dataset in window.document.project.datasets)
        ),
        timeout=120_000,
    )
    assert window.fit_panel.status_text() == "拟合完成"


def _select_alternate_candidate(window) -> str:
    result = window.document.project.datasets[1].last_valid_result
    selectable = tuple(index for index, candidate in enumerate(result.candidates) if candidate_is_selectable(candidate))
    candidates = window.findChild(QListWidget, "candidateList")
    if candidates is None:
        candidates = window.result_panel.candidates
    target_row = next(
        (index for index in selectable if index != candidates.currentRow()),
        selectable[0],
    )
    target_id = result.candidates[target_row].candidate_id
    item = candidates.item(target_row)
    QTest.mouseClick(
        candidates.viewport(),
        Qt.MouseButton.LeftButton,
        pos=candidates.visualItemRect(item).center(),
    )
    QApplication.processEvents()
    assert window.result_panel.selected_candidate_id() == target_id
    assert window.plot_panel.selected_candidate_id() == target_id
    return target_id


def _save_and_export(window, project_path: Path) -> None:
    save_as = window.findChild(QPushButton, "saveAsProjectButton")
    QTest.mouseClick(save_as, Qt.MouseButton.LeftButton)
    assert project_path.is_file()
    assert window.document.is_dirty is False

    def close_export_summary(dialog: QDialog) -> None:
        buttons = dialog.findChild(QDialogButtonBox, "exportSummaryButtons")
        QTest.mouseClick(
            buttons.button(QDialogButtonBox.StandardButton.Close),
            Qt.MouseButton.LeftButton,
        )

    def accept_ort_option(dialog: QDialog) -> None:
        buttons = dialog.findChild(QDialogButtonBox, "ortOptionButtons")
        QTest.mouseClick(
            buttons.button(QDialogButtonBox.StandardButton.Ok),
            Qt.MouseButton.LeftButton,
        )

    _run_modals(
        lambda: QTest.mouseClick(window.export_button, Qt.MouseButton.LeftButton),
        [
            ("ortOptionDialog", accept_ort_option),
            ("exportSummaryDialog", close_export_summary),
        ],
    )
    manifest = window.export_workflow.manifest
    assert manifest is not None and manifest.run_directory.is_dir()
    assert all((manifest.run_directory / record.path).is_file() for record in manifest.files)
    assert any(str(record.path).endswith(".ort") for record in manifest.files)


def _screenshot_geometry() -> tuple[int, int]:
    """Let the screenshot run pick the window size it needs to document."""
    raw = os.environ.get("XRR_GUI_E2E_SCREENSHOT_SIZE")
    if not raw:
        return (1280, 760)
    width, _, height = raw.partition("x")
    return (int(width), int(height))


DARK_PALETTE_ROLES = (
    ("Window", "#1E1F22"),
    ("WindowText", "#E8E8EA"),
    ("Base", "#26282C"),
    ("AlternateBase", "#2E3034"),
    ("Text", "#E8E8EA"),
    ("Button", "#2A2C30"),
    ("ButtonText", "#E8E8EA"),
    ("ToolTipBase", "#26282C"),
    ("ToolTipText", "#E8E8EA"),
)


def _apply_screenshot_palette() -> None:
    """Switch the application to a dark palette when the run asks for it.

    The theme resolves its tokens from the live QPalette, so documenting dark
    mode means setting the palette before the window is built rather than
    threading a mode through the GUI.
    """
    if not os.environ.get("XRR_GUI_E2E_SCREENSHOT_DARK"):
        return
    from PySide6.QtGui import QColor, QPalette

    from xrr_fitter.gui.theme import apply_theme

    application = QApplication.instance()
    palette = QPalette()
    for role, value in DARK_PALETTE_ROLES:
        palette.setColor(getattr(QPalette.ColorRole, role), QColor(value))
    application.setPalette(palette)
    apply_theme(application)


def _reopen_and_verify(qtbot, target_id: str) -> None:
    _apply_screenshot_palette()
    reopened = _show_window(qtbot, *_screenshot_geometry())
    open_button = reopened.findChild(QPushButton, "openProjectButton")
    QTest.mouseClick(open_button, Qt.MouseButton.LeftButton)
    qtbot.wait(1)

    assert reopened.data_panel.dataset_ids == ("S1", "S2")
    assert tuple(len(dataset.structure.components) for dataset in reopened.document.project.datasets) == (1, 2)
    assert all(dataset.last_valid_result is not None for dataset in reopened.document.project.datasets)
    assert reopened.document.project.ui_state.selected_candidate_ids
    assert reopened.plot_panel.selected_candidate_id() == target_id
    screenshot = os.environ.get("XRR_GUI_E2E_SCREENSHOT")
    if screenshot:
        # The documented screenshots cover standard and expert state; expert mode
        # is what reveals the SLD companion pane and the full parameter table.
        if os.environ.get("XRR_GUI_E2E_SCREENSHOT_EXPERT"):
            reopened.parameters_panel.set_expert_mode(True)
            qtbot.wait(1)
        QApplication.processEvents()
        assert reopened.grab().save(screenshot)


@pytest.mark.spawn
def test_generated_single_and_double_layer_xy_complete_gui_workflow(
    qtbot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources, expected_layers, project_path, export_root = _workflow_inputs(tmp_path)
    _patch_file_dialogs(monkeypatch, sources, project_path, export_root)
    window = _show_window(qtbot, 1600, 900)
    _import_sources(window)
    _adopt_dataset_structures(window, expected_layers)
    _edit_plot_mask(window, sources[1])
    _run_gui_fit(window, qtbot)
    target_id = _select_alternate_candidate(window)
    _save_and_export(window, project_path)
    _reopen_and_verify(qtbot, target_id)
