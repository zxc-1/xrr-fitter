"""An explicit replacement retires only the outgoing, discarded draft."""

from pathlib import Path

import pytest
from PySide6.QtWidgets import QFileDialog, QMessageBox

import xrr_fitter.api as api


@pytest.fixture
def dirty_window(qtbot, tmp_path):
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    path = tmp_path / "original.xrrproj.json"
    window.save_project(path)
    original = path.read_bytes()
    window.document.replace_project(api.set_expert_mode(window.document.project, True))
    assert window.autosave.autosave_now()
    draft = window.autosave.draft_path()
    assert draft is not None and draft.exists()
    return window, path, draft, original


def _open_dialog(monkeypatch, target):
    questions = []
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (str(target) if target is not None else "", ""),
    )
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda _parent, title, *_args: (
            questions.append(title),
            QMessageBox.StandardButton.Yes,
        )[1],
    )
    return questions


def _assert_current_draft_preserved(dirty_window, project, draft_original):
    window, path, draft, original = dirty_window
    assert window.document.project is project
    assert window.document.path == path
    assert window.document.is_dirty is True
    assert draft.read_bytes() == draft_original
    assert path.read_bytes() == original


def test_reopen_does_not_offer_the_draft_just_explicitly_discarded(dirty_window, monkeypatch):
    window, path, draft, original = dirty_window
    questions = _open_dialog(monkeypatch, path)

    window.open_project_dialog()

    assert questions == ["未保存的项目更改"]
    assert not draft.exists()
    assert window.document.is_dirty is False
    assert window.document.project.ui_state.expert_mode is False
    assert path.read_bytes() == original


def test_new_project_retires_the_explicitly_discarded_draft(dirty_window, monkeypatch):
    window, path, draft, original = dirty_window
    questions = _open_dialog(monkeypatch, None)

    window.new_project_dialog()

    assert questions == ["未保存的项目更改"]
    assert not draft.exists()
    assert window.document.path is None
    assert window.document.is_dirty is False
    assert path.read_bytes() == original


def test_replacement_keeps_a_different_targets_recoverable_draft(dirty_window, monkeypatch, tmp_path):
    window, path, draft, original = dirty_window
    target = tmp_path / "target.xrrproj.json"
    target_draft = Path(str(target) + ".autosave")
    target_project = api.new_project()
    api.save_project(target_project, target)
    api.save_project(api.set_expert_mode(target_project, True), target_draft)
    target_original = target.read_bytes()
    target_draft_original = target_draft.read_bytes()
    questions = _open_dialog(monkeypatch, target)

    window.open_project_dialog()

    assert questions == ["未保存的项目更改", "恢复自动保存草稿"]
    assert not draft.exists()
    assert target_draft.read_bytes() == target_draft_original
    assert window.document.path == target
    assert window.document.is_dirty is True
    assert window.document.project.ui_state.expert_mode is True
    assert target.read_bytes() == target_original
    assert path.read_bytes() == original


def test_cancelled_file_selection_keeps_the_current_draft(dirty_window, monkeypatch):
    window, path, draft, original = dirty_window
    project = window.document.project
    draft_original = draft.read_bytes()
    questions = _open_dialog(monkeypatch, None)

    window.open_project_dialog()

    assert questions == ["未保存的项目更改"]
    _assert_current_draft_preserved(dirty_window, project, draft_original)


def test_failed_open_keeps_the_current_draft(dirty_window, tmp_path):
    window, path, draft, original = dirty_window
    project = window.document.project
    draft_original = draft.read_bytes()
    broken = tmp_path / "broken.xrrproj.json"
    broken.write_text("not JSON", encoding="utf-8")

    with pytest.raises(ValueError):
        window.open_project(broken, discard_unsaved=True)

    _assert_current_draft_preserved(dirty_window, project, draft_original)


@pytest.mark.parametrize("operation", ["open", "new"])
def test_failed_projection_keeps_the_current_draft(dirty_window, operation):
    window, path, draft, original = dirty_window
    project = window.document.project
    draft_original = draft.read_bytes()

    def fail_projection(_project):
        raise RuntimeError("intentional projection failure")

    window.document.register_project_projection(fail_projection)
    try:
        with pytest.raises(RuntimeError, match="intentional projection failure"):
            if operation == "open":
                window.open_project(path, discard_unsaved=True)
            else:
                window.new_project(discard_unsaved=True)
    finally:
        window.document.unregister_project_projection(fail_projection)

    _assert_current_draft_preserved(dirty_window, project, draft_original)


def test_no_explicit_discard_keeps_the_current_draft(dirty_window):
    window, path, draft, original = dirty_window
    project = window.document.project
    draft_original = draft.read_bytes()

    with pytest.raises(RuntimeError, match="explicit discard"):
        window.open_project(path)

    _assert_current_draft_preserved(dirty_window, project, draft_original)


def test_clean_open_still_offers_a_previous_sessions_draft(qtbot, tmp_path, monkeypatch):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    path = tmp_path / "recovery.xrrproj.json"
    draft = Path(str(path) + ".autosave")
    project = api.new_project()
    api.save_project(project, path)
    api.save_project(api.set_expert_mode(project, True), draft)
    draft_original = draft.read_bytes()
    document = ProjectDocument()
    document.open(path)
    window = MainWindow(document)
    qtbot.addWidget(window)
    assert window.document.is_dirty is False
    questions = _open_dialog(monkeypatch, path)

    window.open_project_dialog()

    assert questions == ["恢复自动保存草稿"]
    assert window.document.is_dirty is True
    assert window.document.project.ui_state.expert_mode is True
    assert draft.read_bytes() == draft_original
