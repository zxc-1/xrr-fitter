"""Source recovery and draft dialogs are operated, not replaced by answers."""

import pytest
from PySide6.QtWidgets import QMessageBox
from tests.gui.control_operation_driver import setup_project, source_curve

import xrr_fitter.api as api


def answer(driver, dialog, accept):
    button = QMessageBox.StandardButton.Yes if accept else QMessageBox.StandardButton.No
    driver.click(dialog.button(button))


def assert_source_outcome(window, original, expected, accept):
    if accept:
        assert window.document.project.datasets[0].source_sha256 == expected
        assert window.source_hash_status("curve") == "ok"
        assert window.document.is_dirty
    else:
        assert window.document.project is original


@pytest.mark.parametrize("operation", ("reload", "relink"))
@pytest.mark.parametrize("accept", (False, True))
def test_source_recovery_menu_and_confirmation(controls, tmp_path, operation, accept):
    project = setup_project(tmp_path)
    window = controls.window(project)
    original = window.document.project
    old_hash = original.datasets[0].source_sha256
    source = tmp_path / "curve.xy"
    target = source if operation == "reload" else tmp_path / "replacement.xy"
    source_curve(target, header="# revised source\n")
    expected = api.preview_source_update(original, "curve", None if operation == "reload" else target).observed_sha256

    def confirm(dialog):
        assert old_hash in dialog.text() and expected in dialog.text()
        answer(controls, dialog, accept)

    handlers = [confirm]
    if operation == "relink":
        source.rename(tmp_path / "original-missing.xy")
        handlers.insert(0, lambda dialog: controls.file(dialog, target))
    controls.modals(lambda: controls.menu(window, f"{operation}SourceAction"), handlers)
    assert_source_outcome(window, original, expected, accept)
    controls.record(
        (f"{operation}SourceAction", f"sourceUpdate:{'Yes' if accept else 'No'}"),
        f"{operation}; inspect source hashes; answer {accept}",
        "confirmed source adopted or exact project preserved",
    )
    if operation == "relink":
        controls.record(
            "fileDialog:RelinkSource",
            "type replacement path into real Qt chooser",
            "chosen replacement is the source shown for confirmation",
        )


def test_relink_file_chooser_cancel_preserves_project(controls, tmp_path):
    window = controls.window(setup_project(tmp_path))
    original = window.document.project
    controls.modals(lambda: controls.menu(window, "relinkSourceAction"), [lambda dialog: controls.file(dialog, None)])
    assert window.document.project is original
    controls.record("fileDialog:RelinkSource", "Cancel replacement chooser", "project unchanged")


@pytest.mark.parametrize("accept", (False, True))
def test_recover_draft_uses_real_open_and_confirmation(controls, tmp_path, accept):
    base = setup_project(tmp_path)
    draft = api.set_expert_mode(base, True)
    path = tmp_path / "saved.xrrproj.json"
    api.save_project(base, path)
    sidecar = path.with_name(path.name + ".autosave")
    api.save_project(draft, sidecar)
    original_bytes = path.read_bytes()
    window = controls.window()
    controls.modals(
        lambda: controls.named_click(window, "openProjectButton"),
        [lambda dialog: controls.file(dialog, path), lambda dialog: answer(controls, dialog, accept)],
    )
    assert window.document.project.ui_state.expert_mode is accept
    assert window.document.is_dirty is accept
    assert path.read_bytes() == original_bytes
    assert sidecar.exists() is accept
    controls.record(
        f"recoverDraft:{'Yes' if accept else 'No'}",
        "Open project; answer recovery question",
        "adopted draft is dirty; decline preserves saved project; saved bytes unchanged",
    )


def test_workflow_error_can_be_closed_without_publication(controls, tmp_path):
    window = controls.window(setup_project(tmp_path))
    original = window.document.project
    (tmp_path / "curve.xy").rename(tmp_path / "missing.xy")

    def close_error(dialog):
        assert isinstance(dialog, QMessageBox)
        assert "检查数据源" in dialog.text()
        controls.click(dialog.button(QMessageBox.StandardButton.Ok))

    controls.modals(lambda: controls.menu(window, "reloadSourceAction"), [close_error])
    assert window.document.project is original
    controls.record(
        ("reloadSourceAction", "workflowError:Close"),
        "reload missing source; close error",
        "visible failure does not change project",
    )
