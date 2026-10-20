"""Audit helpers must fail boundedly rather than hang at an unexpected modal."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFileDialog
from tests.gui.control_operation_driver import controls as controls
from tests.gui.control_operation_driver import source_curve
from tests.gui.test_control_operations_dialogs import preset_project


def test_unexpected_second_modal_is_rejected_with_its_identity(controls):
    first = QDialog()
    second = QDialog()
    second.setObjectName("unplannedDialog")
    controls.qtbot.addWidget(first)
    controls.qtbot.addWidget(second)
    with pytest.raises(AssertionError, match="unexpected modal dialog: unplannedDialog"):
        controls.modals(lambda: (first.exec(), second.exec()), [lambda dialog: dialog.reject()])
    assert not second.isVisible()


def test_file_chooser_replaces_a_selected_directory(controls, tmp_path):
    directory = tmp_path / "previous"
    directory.mkdir()
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    source = incoming / "P1 Si+Zr.xy"
    source.write_text("0.1 1\n", encoding="utf-8")
    dialog = QFileDialog()
    controls.qtbot.addWidget(dialog)
    dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
    dialog.setDirectory(str(tmp_path))
    dialog.selectFile(str(directory))
    dialog.show()
    assert dialog.selectedFiles() == [str(directory)]

    controls.file(dialog, source)

    assert dialog.result() == QDialog.DialogCode.Accepted
    assert dialog.selectedFiles() == [str(source)]


def test_file_chooser_accepts_system_temporary_path(controls):
    window = controls.window(preset_project())
    reached = []

    def choose_substrate(dialog):
        reached.append(dialog.objectName())
        controls.type(dialog.substrate_editor, "Al2O3")
        controls.buttons(dialog, QDialogButtonBox.StandardButton.Ok)

    with TemporaryDirectory(prefix="xrr-r23-file-dialog-") as directory:
        source = source_curve(Path(directory).resolve() / "P1 Si+Zr.xy")
        controls.modals(
            lambda: controls.named_click(window, "datasetAddButton"),
            [lambda dialog: controls.file(dialog, source), choose_substrate],
        )
        assert len(window.document.project.datasets) == 1
        assert window.document.project.datasets[0].structure.backing.name == "Al2O3"
        window.close()
    assert reached == ["substrateDialog"]


def test_handled_modal_left_open_is_rejected_with_its_identity(controls):
    dialog = QDialog()
    dialog.setObjectName("stuckDialog")
    controls.qtbot.addWidget(dialog)
    forced_cleanup = []
    fallback = QTimer(dialog)
    fallback.setSingleShot(True)

    def cleanup():
        forced_cleanup.append(True)
        dialog.reject()

    fallback.timeout.connect(cleanup)
    fallback.start(4500)
    try:
        with pytest.raises(AssertionError, match="modal dialog did not close: stuckDialog"):
            controls.modals(dialog.exec, [lambda shown: None])
        assert not forced_cleanup
        assert not dialog.isVisible()
    finally:
        fallback.stop()
