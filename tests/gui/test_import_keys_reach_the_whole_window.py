"""Ctrl+I / Ctrl+Shift+I 在整个窗口都能导入，菜单里也写着它。

审计实测：两条导入快捷键挂在数据面板自己的 ``QShortcut`` 上（只在焦点落在左栏时生效），而画布
空状态那句提示、文件菜单都没给出可用的按键。焦点停在画布或检视器时按 Ctrl+I 没有任何反应。
这里钉住：快捷键挂在文件菜单那两条命令上、整个窗口可用；菜单项自己显示按键。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QFileDialog


def _window(qtbot):
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.set_guidance_visible(False)
    window.resize(1400, 900)
    window.show()
    window.activateWindow()
    qtbot.waitExposed(window)
    qtbot.waitUntil(lambda: window.isActiveWindow())
    return window


def test_ctrl_i_imports_with_focus_outside_the_data_panel(qtbot, monkeypatch) -> None:
    window = _window(qtbot)
    opened: list[str] = []
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", lambda *_a, **_k: (opened.append("files"), ([], ""))[1])
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_a, **_k: (opened.append("folder"), "")[1])
    target = window.result_panel
    target.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    target.setFocus()
    assert not window.data_panel.isAncestorOf(window.focusWidget())

    qtbot.keyClick(target, Qt.Key.Key_I, Qt.KeyboardModifier.ControlModifier)
    qtbot.keyClick(target, Qt.Key.Key_I, Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)

    assert opened == ["files", "folder"]


def test_the_file_menu_shows_the_import_keys(qtbot) -> None:
    window = _window(qtbot)

    assert window.chrome_actions["importFilesMenuAction"].shortcut() == QKeySequence("Ctrl+I")
    assert window.chrome_actions["importFolderMenuAction"].shortcut() == QKeySequence("Ctrl+Shift+I")
