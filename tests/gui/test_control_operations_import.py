"""Import surface input sweeps through the real MainWindow entry points."""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QShortcut
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialogButtonBox
from tests.gui.control_operation_driver import setup_project, source_curve

import xrr_fitter.api as api


def _cycle_instrument_modes(driver, dialog):
    for kind in ("linear", "powerlaw", "constant"):
        driver.select(dialog.background, dialog.background.findData(kind))
        assert dialog.instrument_spec().background_kind == kind
    for domain in ("theta", "q"):
        driver.select(dialog.resolution_domain, dialog.resolution_domain.findData(domain))
        assert dialog.instrument_spec().resolution_domain == domain
    for mode in ("none", "fit", "geometry"):
        driver.select(dialog.footprint, dialog.footprint.findData(mode))
        assert dialog.instrument_spec().footprint_mode == mode
        assert dialog.sample_length.isEnabled() is (mode == "geometry")


def _import_fields(driver, dialog, *, mixed=False):
    driver.click(dialog.mixed_button if mixed else dialog.mono_button)
    driver.type(dialog.mono_wavelength, "1.1")
    driver.type(dialog.wavelength_1, "1.54056")
    driver.type(dialog.wavelength_2, "1.54439")
    driver.type(dialog.intensity_ratio, "0.42")
    driver.type(dialog.instrument_id, "qtest-import")
    driver.select(dialog.footprint, dialog.footprint.findData("geometry"))
    driver.type(dialog.sample_length, "12.0")
    driver.type(dialog.beam_width, "0.12")
    _cycle_instrument_modes(driver, dialog)
    assert dialog.instrument_spec().sample_length_mm == pytest.approx(12.0)
    assert dialog.instrument_spec().beam_width_mm == pytest.approx(0.12)
    assert dialog.instrument_spec().instrument_id == "qtest-import"


@pytest.mark.parametrize("mixed", (False, True))
def test_import_beam_instrument_and_angle_inputs_commit(controls, tmp_path, mixed):
    window = controls.window()
    path = source_curve(tmp_path / "manual.xy", header="# theta intensity\n")

    def configure(dialog):
        _import_fields(controls, dialog, mixed=mixed)
        controls.click(dialog.two_theta_convention)
        assert dialog.angle_convention() == "two_theta"
        controls.click(dialog.theta_convention)
        assert dialog.angle_convention() == "theta"
        controls.click(dialog.two_theta_convention)
        controls.click(dialog.detect_convention)
        assert dialog.angle_convention() == "theta"
        assert dialog.preview_table.item(0, 1).text() == "θ"
        controls.buttons(dialog, QDialogButtonBox.StandardButton.Ok)

    controls.modals(
        lambda: controls.named_click(window, "datasetAddButton"),
        [lambda dialog: controls.file(dialog, path), configure],
    )
    dataset = window.document.project.datasets[0]
    assert dataset.dataset_id == "manual"
    assert dataset.angle_convention == "theta"
    assert dataset.beam.kind == ("mixed_kalpha" if mixed else "monochromatic")
    assert (
        dataset.beam.intensity_ratio_21 == pytest.approx(0.42)
        if mixed
        else dataset.beam.wavelength_a == pytest.approx(1.1)
    )
    assert dataset.instrument.sample_length_mm == pytest.approx(12.0)
    assert window.document.project.measurement_preset is not None
    controls.record(
        ("datasetAddButton", "datasetAddAction", "fileDialog:ImportFiles", "importDialog:Ok"),
        "click +; type source in real Qt chooser; configure and click Import",
        "dataset and reusable measurement preset persist",
        setup="empty MainWindow; generated 64-line parser fixture",
    )
    controls.record(
        (
            "beamRadio:mixed" if mixed else "beamRadio:monochromatic",
            "monochromaticWavelengthEditor",
            "mixedWavelength1Editor",
            "mixedWavelength2Editor",
            "mixedIntensityRatioEditor",
            "instrumentIdEditor",
            "sampleLengthEditor",
            "beamWidthEditor",
            "footprintModeEditor",
            "backgroundModelEditor",
            "resolutionDomainEditor",
            "twoThetaConventionOption",
            "thetaConventionOption",
            "detectAngleConventionButton",
        ),
        "mouse/radio clicks; keyboard numeric entry; popup Home/Down/Return selection",
        "beam/instrument declarations and detected theta convention persist correctly",
    )


def test_import_column_mapping_apply_reopen_cancel(controls, tmp_path):
    window = controls.window()
    path = source_curve(tmp_path / "mapped.xy", columns=4)

    def configure(dialog):
        controls.click(dialog.mono_button)

        def mapping_fields(mapping):
            controls.type(mapping.two_theta, "1")
            controls.type(mapping.intensity, "1")
            controls.buttons(mapping, QDialogButtonBox.StandardButton.Ok)
            assert mapping.isVisible() and mapping.error_label.isVisible()
            assert "distinct" in mapping.error_label.text()
            controls.type(mapping.intensity, "2")
            controls.click(mapping.sigma_enabled)
            controls.type(mapping.intensity_sigma, "3")
            controls.click(mapping.resolution_enabled)
            controls.type(mapping.resolution, "4")
            for index in range(mapping.resolution_kind.count()):
                controls.select(mapping.resolution_kind, index)
                assert mapping.mapping().resolution_kind == mapping.resolution_kind.itemData(index)
            controls.select(mapping.resolution_kind, 0)
            controls.buttons(mapping, QDialogButtonBox.StandardButton.Ok)

        controls.modals(lambda: controls.click(dialog.mapping_button), [mapping_fields])
        expected = api.DataColumnMapping(0, 1, 2, 3, "sigma_q_a_inv")
        assert dialog.column_mapping() == expected

        def cancel_mapping(mapping):
            assert mapping.mapping() == expected
            controls.type(mapping.two_theta, "2")
            controls.click(mapping.sigma_enabled)
            controls.click(mapping.resolution_enabled)
            assert not mapping.intensity_sigma.isEnabled() and not mapping.resolution.isEnabled()
            controls.buttons(mapping, QDialogButtonBox.StandardButton.Cancel)

        controls.modals(lambda: controls.click(dialog.mapping_button), [cancel_mapping])
        assert dialog.column_mapping() == expected
        controls.buttons(dialog, QDialogButtonBox.StandardButton.Ok)

    controls.modals(
        lambda: controls.menu(window, "importFilesMenuAction"),
        [lambda dialog: controls.file(dialog, path), configure],
    )
    assert window.document.project.datasets[0].column_mapping == api.DataColumnMapping(0, 1, 2, 3, "sigma_q_a_inv")
    controls.record(
        (
            "importFilesMenuAction",
            "columnMappingButton",
            "twoThetaColumnEditor",
            "intensityColumnEditor",
            "intensitySigmaColumnEditor",
            "resolutionColumnEditor",
            "resolutionKindEditor",
            "intensitySigmaEnabled",
            "resolutionEnabled",
            "columnMappingDialog:Ok",
            "columnMappingDialog:Cancel",
        ),
        "real File menu import; mapping dialog typed/selected/applied; reopened, changed and cancelled",
        "duplicate columns refused inline; optional mapping persists; cancellation does not overwrite it",
    )


def test_folder_recursion_checkbox_changes_actual_import(controls, tmp_path):
    folder = tmp_path / "batch"
    child = folder / "child"
    child.mkdir(parents=True)
    source_curve(folder / "first.xy")
    source_curve(child / "second.xy")
    window = controls.window()

    def configure(dialog):
        assert dialog.preview_table.rowCount() == 1
        controls.click(dialog.recursive_check)
        assert dialog.preview_table.rowCount() == 2
        controls.click(dialog.recursive_check)
        assert dialog.preview_table.rowCount() == 1
        controls.click(dialog.recursive_check)
        controls.click(dialog.mono_button)
        controls.buttons(dialog, QDialogButtonBox.StandardButton.Ok)

    controls.modals(
        lambda: controls.named_click(window, "importFolderQuickButton"),
        [lambda dialog: controls.file(dialog, folder), configure],
    )
    assert {value.dataset_id for value in window.document.project.datasets} == {"first", "second"}
    tree = window.data_panel.tree
    for index in (1, 0):
        item = tree.topLevelItem(index)
        identity = item.data(0, Qt.ItemDataRole.UserRole)
        controls.qtbot.mouseClick(tree.viewport(), Qt.MouseButton.LeftButton, pos=tree.visualItemRect(item).center())
        assert window.document.active_dataset_id == identity
    controls.record(
        ("importFolderQuickButton", "fileDialog:ImportFolder", "recursiveFolderImportCheck"),
        "real folder picker; checkbox on/off/on",
        "preview and committed import contain exactly two recursive sources",
    )
    controls.record("datasetTree", "click both data cards", "active dataset follows actual selection")


@pytest.mark.parametrize(
    "entry",
    (
        "emptyStateImportButton",
        "emptyStateImportFolderButton",
        "importFolderMenuAction",
        "importFilesShortcut",
        "importFolderShortcut",
    ),
)
def test_import_alternative_entry_cancel_preserves_empty_project(controls, entry):
    window = controls.window()
    original = window.document.project

    def activate():
        if entry.endswith("MenuAction"):
            controls.menu(window, entry)
        elif entry.endswith("Shortcut"):
            shortcut = window.findChild(QShortcut, entry)
            window.data_panel.tree.setFocus()
            QTest.keySequence(window.data_panel.tree, shortcut.key())
        else:
            controls.named_click(window, entry)

    controls.modals(activate, [lambda dialog: controls.file(dialog, None)])
    assert window.document.project is original
    controls.record(
        entry, "real button/menu/keyboard activation; cancel Qt file dialog", "no dataset or preset is committed"
    )


def test_import_preview_cancel_does_not_create_preset(controls, tmp_path):
    window = controls.window()
    path = source_curve(tmp_path / "cancel.xy")
    original = window.document.project

    def cancel_import(dialog):
        controls.click(dialog.mixed_button)
        controls.type(dialog.intensity_ratio, "0.33")
        controls.buttons(dialog, QDialogButtonBox.StandardButton.Cancel)

    controls.modals(
        lambda: controls.named_click(window, "datasetAddButton"),
        [lambda dialog: controls.file(dialog, path), cancel_import],
    )
    assert window.document.project is original
    assert window.document.project.measurement_preset is None
    controls.record(
        "importDialog:Cancel",
        "select file; edit preview declarations; Cancel",
        "neither dataset nor measurement preset changes",
    )


def test_batch_segment_modes_commit(controls, tmp_path):
    window = controls.window(setup_project(tmp_path, datasets=2))
    controls.named_click(window, "pipelineStep_数据")
    for name, mode in (("batchModeJoint", "joint"), ("batchModeIndependent", "independent")):
        controls.named_click(window, name)
        assert window.document.project.batch_mode == mode
    controls.record(
        ("batchModeJoint", "batchModeIndependent"),
        "click both batch mode segments",
        "two-dataset project persists requested mode",
    )
