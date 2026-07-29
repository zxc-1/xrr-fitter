from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QFileDialog, QMessageBox, QPushButton

import xrr_fitter.api as api


def _action(window, object_name: str) -> QAction:
    action = window.findChild(QAction, object_name)
    assert action is not None, f"missing action {object_name}"
    return action


def _button(window, object_name: str) -> QPushButton:
    button = window.findChild(QPushButton, object_name)
    assert button is not None, f"missing button {object_name}"
    return button


def test_project_and_fit_commands_expose_standard_qactions_and_shortcuts(qtbot) -> None:
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    expected = {
        "newProjectAction": QKeySequence(QKeySequence.StandardKey.New),
        "openProjectAction": QKeySequence(QKeySequence.StandardKey.Open),
        "saveProjectAction": QKeySequence(QKeySequence.StandardKey.Save),
        "saveAsProjectAction": QKeySequence(QKeySequence.StandardKey.SaveAs),
    }

    for object_name, shortcut in expected.items():
        action = _action(window, object_name)
        assert action.text()
        assert action.toolTip()
        assert action.statusTip()
        assert action.shortcut() == shortcut


def test_project_action_shortcuts_trigger_public_workflows(qtbot, monkeypatch) -> None:
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    calls: list[str] = []
    monkeypatch.setattr(window, "new_project_dialog", lambda: calls.append("new"))
    monkeypatch.setattr(window, "open_project_dialog", lambda: calls.append("open"))
    monkeypatch.setattr(window, "save_project_dialog", lambda: calls.append("save"))
    monkeypatch.setattr(
        window,
        "save_project_as_dialog",
        lambda: calls.append("save-as"),
    )

    for name in (
        "newProjectAction",
        "openProjectAction",
        "saveProjectAction",
        "saveAsProjectAction",
    ):
        _action(window, name).trigger()

    assert calls == ["new", "open", "save", "save-as"]


def test_project_buttons_route_through_public_main_window_workflows(
    qtbot,
    monkeypatch,
) -> None:
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    calls: list[str] = []
    callbacks = {
        "newProjectButton": ("new_project_dialog", "new"),
        "openProjectButton": ("open_project_dialog", "open"),
        "saveProjectButton": ("save_project_dialog", "save"),
        "saveAsProjectButton": ("save_project_as_dialog", "save-as"),
        "reloadSourceButton": ("reload_source_dialog", "reload"),
        "relinkSourceButton": ("relink_source_dialog", "relink"),
    }
    for callback, value in callbacks.values():
        monkeypatch.setattr(window, callback, lambda value=value: calls.append(value))

    for object_name in callbacks:
        button = _button(window, object_name)
        button.setEnabled(True)
        button.click()

    assert calls == ["new", "open", "save", "save-as", "reload", "relink"]


def test_project_file_dialogs_call_document_workflows(qtbot, tmp_path, monkeypatch) -> None:
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    opened = tmp_path / "opened.xrrproj.json"
    saved = tmp_path / "saved.xrrproj.json"
    saved_as = tmp_path / "saved-as.xrrproj.json"
    open_names = iter(((str(opened), ""),))
    save_names = iter(((str(saved), ""), (str(saved_as), "")))
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: next(open_names),
    )
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: next(save_names),
    )
    monkeypatch.setattr(window.document, "open", lambda path: calls.append(("open", path)))
    monkeypatch.setattr(window.document, "save", lambda path=None: calls.append(("save", path)))

    window.open_project_dialog()
    window.save_project_dialog()
    window.save_project_as_dialog()

    assert calls == [
        ("open", opened),
        ("save", saved),
        ("save", saved_as),
    ]


def test_dirty_project_declining_discard_blocks_new_and_open(qtbot, monkeypatch) -> None:
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.document.mark_dirty()
    questions: list[str] = []
    file_dialogs: list[bool] = []
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda _parent, title, *_args: (
            questions.append(title),
            QMessageBox.StandardButton.No,
        )[1],
    )
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (file_dialogs.append(True), ("", ""))[1],
    )

    window.new_project_dialog()
    window.open_project_dialog()

    assert questions == ["未保存的项目更改", "未保存的项目更改"]
    assert file_dialogs == []
    assert window.document.is_dirty is True


def test_project_dialog_reports_a_known_open_error_once(qtbot, tmp_path, monkeypatch) -> None:
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    target = tmp_path / "broken.xrrproj.json"
    messages: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (str(target), ""),
    )
    monkeypatch.setattr(
        window.document,
        "open",
        lambda _path: (_ for _ in ()).throw(ValueError("invalid project JSON")),
    )
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda _parent, title, message: messages.append((title, message)),
    )

    window.open_project_dialog()

    assert len(messages) == 1
    assert messages[0][0] == "打开项目失败"
    assert "ValueError: invalid project JSON" in messages[0][1]
    assert "检查项目文件" in messages[0][1]


def test_active_dataset_selection_updates_project_ui_state(monkeypatch) -> None:
    from xrr_fitter.gui.document import ProjectDocument

    original = api.new_project()
    updated = api.set_expert_mode(api.new_project(), True)
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        api,
        "select_active_dataset",
        lambda project, dataset_id: (
            calls.append((project, dataset_id)),
            updated,
        )[1],
    )
    document = ProjectDocument(original)

    document.select_active_dataset("sample")

    assert calls == [(original, "sample")]
    assert document.project is updated
    assert document.is_dirty is True
