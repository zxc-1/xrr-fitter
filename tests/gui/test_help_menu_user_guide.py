"""帮助菜单给出用户手册（F51）。

wheel 与 exe 都不带 ``docs/``，手册指向公开仓库的固定提交，不依赖候选版尚未创建的
标签。系统不能打开浏览器时，提示中保留尝试访问的地址。
"""

from __future__ import annotations

import re

from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QMenu, QMessageBox
from tests.gui.test_shell_menus import _menu_action, _window


def test_the_help_menu_opens_a_fixed_guide_without_a_release_tag(qtbot, monkeypatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: opened.append(url.toString()) or True)
    window = _window(qtbot)

    action = _menu_action(window, "userGuideAction")
    assert action in window.findChild(QMenu, "helpMenu").actions()
    action.trigger()

    assert len(opened) == 1
    assert re.fullmatch(r"https://github\.com/zxc-1/xrr-fitter/blob/[0-9a-f]{40}/docs/user-guide\.md", opened[0])


def test_a_guide_the_system_cannot_open_names_the_address(qtbot, monkeypatch) -> None:
    opened: list[str] = []
    warnings: list[str] = []
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: opened.append(url.toString()) or False)
    monkeypatch.setattr(QMessageBox, "warning", lambda _parent, _title, text: warnings.append(text))
    window = _window(qtbot)

    _menu_action(window, "userGuideAction").trigger()

    assert len(opened) == 1
    assert len(warnings) == 1
    assert opened[0] in warnings[0]
