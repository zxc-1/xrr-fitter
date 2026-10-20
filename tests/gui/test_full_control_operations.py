"""Shell and project controls exercised through real Qt input and dialogs."""

import pytest
from PySide6.QtWidgets import QApplication, QDialogButtonBox
from tests.gui.control_operation_driver import setup_project


def _choose_project_file(controls, trigger, path):
    controls.modals(trigger, [lambda dialog: controls.file(dialog, path)])


def _answer_unsaved(controls, window, text):
    controls.modals(
        lambda: controls.named_click(window, "newProjectButton"),
        [lambda dialog: controls.click(next(button for button in dialog.buttons() if button.text() == text))],
    )


def test_shell_mode_theme_and_about_real_inputs(controls):
    window = controls.window()
    original = window.document.project
    controls.named_click(window, "workspaceModeGuided")
    assert window.guidance_is_visible()
    controls.named_click(window, "workspaceModeExpert")
    assert not window.guidance_is_visible()
    assert window.document.project is original and not window.document.is_dirty
    controls.record(
        ("workspaceModeGuided", "workspaceModeExpert"),
        "mouse click each segment",
        "surface changes without document mutation",
    )
    controls.menu(window, "guidanceModeAction")
    assert window.guidance_is_visible()
    controls.menu(window, "guidanceModeAction")
    assert not window.guidance_is_visible()
    controls.record("guidanceModeAction", "open View menu and click twice", "guided surface toggles twice")
    before = QApplication.palette().window().color()
    controls.named_click(window, "appearanceToggleButton")
    assert QApplication.palette().window().color() != before
    controls.named_click(window, "appearanceToggleButton")
    assert QApplication.palette().window().color() == before
    controls.record("appearanceToggleButton", "two mouse clicks", "application palette changes then restores")

    def close_about(dialog):
        assert dialog.windowTitle() == "关于 XRR Fitter"
        controls.buttons(dialog, QDialogButtonBox.StandardButton.Close)

    controls.modals(lambda: controls.menu(window, "aboutAction"), [close_about])
    controls.record(
        ("aboutAction", "aboutDialog:Close"), "Help menu then Close button", "real about dialog opens and closes"
    )


def test_project_toolbar_save_new_open_and_save_as_menu(controls, tmp_path):
    window = controls.window(setup_project(tmp_path))
    target = tmp_path / "saved.xrrproj.json"
    _choose_project_file(controls, lambda: controls.named_click(window, "saveProjectButton"), target)
    assert target.is_file() and window.document.path == target
    assert not window.document.is_dirty
    controls.record(
        ("saveProjectButton", "fileDialog:SaveProject"),
        "toolbar Save; type path; click Save",
        "real project file exists and document path matches",
    )
    controls.named_click(window, "newProjectButton")
    assert not window.document.project.datasets and window.document.path is None
    controls.record("newProjectButton", "toolbar New", "project becomes empty")
    _choose_project_file(controls, lambda: controls.named_click(window, "openProjectButton"), target)
    assert window.document.active_dataset_id == "curve" and window.document.path == target
    controls.record(
        ("openProjectButton", "fileDialog:OpenProject"),
        "toolbar Open; type saved path; click Open",
        "saved curve project reopens",
    )
    copy = tmp_path / "copy.xrrproj.json"
    _choose_project_file(controls, lambda: controls.menu(window, "saveAsProjectAction"), copy)
    assert copy.is_file() and window.document.path == copy
    controls.record(
        ("saveAsProjectAction", "fileDialog:SaveAs"),
        "File menu Save As; type path; click Save",
        "copy exists and document path updates",
    )


@pytest.mark.parametrize("surface", ("menu", "shortcut"))
def test_project_save_open_new_and_cancel_paths(controls, tmp_path, surface):
    window = controls.window(setup_project(tmp_path))
    activate = controls.menu if surface == "menu" else controls.shortcut
    target = tmp_path / "keys.xrrproj.json"
    _choose_project_file(controls, lambda: activate(window, "saveProjectAction"), target)
    assert target.is_file()
    activate(window, "newProjectAction")
    assert not window.document.project.datasets
    _choose_project_file(controls, lambda: activate(window, "openProjectAction"), target)
    assert window.document.active_dataset_id == "curve"
    original = window.document.project
    _choose_project_file(controls, lambda: activate(window, "saveAsProjectAction"), None)
    _choose_project_file(controls, lambda: activate(window, "openProjectAction"), None)
    assert window.document.project is original and window.document.path == target
    ids = ("newProjectAction", "openProjectAction", "saveProjectAction", "saveAsProjectAction")
    if surface == "shortcut":
        ids = tuple(f"projectShortcut:{name}" for name in ("New", "Open", "Save", "SaveAs"))
    controls.record(
        ids,
        f"real {surface} activation; real Qt file dialogs",
        "save/new/open roundtrip; cancelled Open/Save As preserve project",
    )


def test_discard_confirmation_and_history_menu(controls, tmp_path):
    window = controls.window(setup_project(tmp_path))
    controls.menu(window, "expertModeAction")
    assert window.document.project.ui_state.expert_mode and window.document.is_dirty
    controls.record(
        "expertModeAction", "click View menu advanced option", "expert depth persists and dirty state changes"
    )
    controls.menu(window, "undoAction")
    assert not window.document.project.ui_state.expert_mode
    controls.menu(window, "redoAction")
    assert window.document.project.ui_state.expert_mode
    controls.record(("undoAction", "redoAction"), "Edit menu Undo/Redo", "expert setting reverses and reapplies")
    original = window.document.project
    _answer_unsaved(controls, window, "取消")
    assert window.document.project is original
    controls.record("unsavedChanges:Cancel", "click New and answer 取消", "dirty project remains unchanged")
    _answer_unsaved(controls, window, "不保存")
    assert not window.document.project.datasets and not window.document.is_dirty
    controls.record(
        "unsavedChanges:Discard", "click New and answer 不保存", "dirty project is replaced only after confirmation"
    )
