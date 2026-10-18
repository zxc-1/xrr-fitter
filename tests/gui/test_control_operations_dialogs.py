"""Previously unrecorded menu widgets, cancellation and read-only affordances."""

from dataclasses import replace

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QDialogButtonBox, QLineEdit
from tests.gui.control_operation_driver import controls as controls
from tests.gui.control_operation_driver import setup_project, source_curve
from tests.gui.test_control_operations_structure import _row_menu, _select_component

import xrr_fitter.api as api


def preset_project():
    preset = api.MeasurementPreset("control-preset", api.BeamSpec("monochromatic"), api.InstrumentSpec())
    return api.set_expert_mode(replace(api.new_project(), measurement_preset=preset), True)


@pytest.mark.parametrize("entry", ("importFilesButton", "importFolderButton", "changeMeasurementPresetButton"))
def test_dataset_add_popup_entries_cancel_without_mutation(controls, entry):
    window = controls.window(preset_project())
    before = window.document.project
    button = controls.widget(window, "datasetAddButton")

    def choose(menu):
        controls.named_click(menu, entry)

    def cancel(dialog):
        assert window.data_panel._force_preset_dialog is (entry == "changeMeasurementPresetButton")
        controls.file(dialog, None)

    try:
        controls.modals(
            lambda: controls.popup(
                lambda: controls.qtbot.mousePress(button, Qt.MouseButton.LeftButton, pos=button.rect().center()), choose
            ),
            [cancel],
        )
    finally:
        controls.qtbot.mouseRelease(button, Qt.MouseButton.LeftButton, pos=QPoint(-5, -5))
    assert window.document.project is before
    assert not window.data_panel._force_preset_dialog
    controls.record(
        ("datasetAddMenu:DelayedPopup", entry),
        "QTest hold opens delayed menu; QTest click embedded button; cancel chooser; release opening press",
        "menu route is live and cancelling keeps the stored preset/project",
    )


def test_ambiguous_import_substrate_fields_and_accept(controls, tmp_path):
    window = controls.window(preset_project())
    source = source_curve(tmp_path / "P1 Si+Zr.xy")

    def choose(dialog):
        assert dialog.objectName() == "substrateDialog"
        controls.buttons(dialog, QDialogButtonBox.StandardButton.Ok)
        assert dialog.error_label.isVisible()
        controls.type(dialog.substrate_editor, "Al2O3")
        controls.buttons(dialog, QDialogButtonBox.StandardButton.Ok)

    controls.modals(
        lambda: controls.named_click(window, "datasetAddButton"), [lambda dialog: controls.file(dialog, source), choose]
    )
    assert len(window.document.project.datasets) == 1
    assert window.document.project.datasets[0].structure.backing.name == "Al2O3"
    controls.record(
        ("substrateMaterialEditor", "substrateDialog:Ok"),
        "empty OK rejected; type actual substrate; OK",
        "chosen substrate is committed through the real import flow",
    )


def test_substrate_cancel_does_not_accept_a_material(controls):
    from xrr_fitter.gui.data.substrate_dialog import SubstrateDialog

    dialog = SubstrateDialog(("Si", "Zr"))
    controls.qtbot.addWidget(dialog)
    dialog.show()
    controls.type(dialog.substrate_editor, "Al2O3")
    controls.buttons(dialog, QDialogButtonBox.StandardButton.Cancel)
    assert dialog.result() == dialog.DialogCode.Rejected
    controls.record(
        "substrateDialog:Cancel",
        "type substrate and Cancel",
        "selection dialog is rejected, not accepted",
        setup="standalone production dialog; import rollback covered separately",
    )


@pytest.mark.parametrize("kind", ("backing", "periodic"))
def test_structure_dialog_cancel_keeps_exact_project(controls, tmp_path, kind):
    window = controls.window(setup_project(tmp_path, layers=2))
    controls.named_click(window, "pipelineStep_结构")
    before = window.document.project
    tree = _select_component(controls, window, 0)

    def trigger():
        if kind == "backing":
            return _row_menu(controls, tree, "editBackingAction")
        return controls.named_click(window, "addPeriodicBlockButton")

    def cancel(dialog):
        editor = dialog.formula_editor if kind == "backing" else dialog.name_editor
        controls.type(editor, "Al2O3")
        controls.buttons(dialog, QDialogButtonBox.StandardButton.Cancel)

    controls.modals(trigger, [cancel])
    assert window.document.project is before
    controls.record(
        f"{kind}DialogButtons:Cancel", f"edit {kind} dialog then Cancel", "project object and structure are unchanged"
    )


def preview_values(table):
    return tuple(
        table.item(row, col).text()
        for row in range(table.rowCount())
        for col in range(table.columnCount())
        if table.item(row, col)
    )


def test_import_reserved_controls_and_preview_remain_readonly(controls, tmp_path):
    from xrr_fitter.gui.data.import_dialog import ImportDialog

    dialog = ImportDialog((source_curve(tmp_path / "curve.xy"),))
    controls.qtbot.addWidget(dialog)
    dialog.show()
    header = controls.widget(dialog, "headerSkipEditor", QLineEdit)
    before = header.text()
    controls.qtbot.keyClicks(header, "42")
    assert header.isReadOnly() and header.text() == before
    delimiter = dialog.delimiter_select
    previous = delimiter.currentIndex()
    controls.qtbot.keyClick(delimiter, Qt.Key.Key_Down)
    assert not delimiter.isEnabled() and delimiter.currentIndex() == previous
    check = dialog.log_intensity_check
    controls.qtbot.mouseClick(check, Qt.MouseButton.LeftButton)
    assert not check.isEnabled() and not check.isChecked()
    table = dialog.preview_table
    text = preview_values(table)
    controls.qtbot.mouseDClick(
        table.viewport(), Qt.MouseButton.LeftButton, pos=table.visualRect(table.model().index(0, 0)).center()
    )
    controls.qtbot.keyClicks(table, "not numeric")
    after = preview_values(table)
    assert text == after and not [w for w in table.findChildren(QLineEdit) if w.isVisible()]
    controls.record(
        ("columnDelimiterEditor", "headerSkipEditor", "logIntensityCheck", "readOnly:importPreviewTable"),
        "attempt reserved field edits and preview table editing",
        "declared read-only/disabled affordances do not mutate input",
    )
    controls.buttons(dialog, QDialogButtonBox.StandardButton.Cancel)
