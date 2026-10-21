"""帮助菜单给出用户手册（F51）。

wheel 与 exe 都不带 ``docs/``，手册只在公开仓库里；按装的这一版打开对应标签下的
``docs/user-guide.md``，读者看到的才是自己这一版的界面。打不开时要说出地址，而不是什么都不发生。
"""

from __future__ import annotations

from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QMenu, QMessageBox
from tests.gui.test_shell_menus import _menu_action, _window

from xrr_fitter.version import __version__

GUIDE = f"https://github.com/zxc-1/xrr-fitter/blob/v{__version__}/docs/user-guide.md"


def test_the_help_menu_opens_the_guide_for_the_installed_version(qtbot, monkeypatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: opened.append(url.toString()) or True)
    window = _window(qtbot)

    action = _menu_action(window, "userGuideAction")
    assert action in window.findChild(QMenu, "helpMenu").actions()
    action.trigger()

    assert opened == [GUIDE]


def test_a_guide_the_system_cannot_open_names_the_address(qtbot, monkeypatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda _url: False)
    monkeypatch.setattr(QMessageBox, "warning", lambda _parent, _title, text: warnings.append(text))
    window = _window(qtbot)

    _menu_action(window, "userGuideAction").trigger()

    assert len(warnings) == 1 and GUIDE in warnings[0]
