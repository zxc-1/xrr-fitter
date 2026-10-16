"""关于框写出版本号（F51）：读者报问题、对照发行说明时，第一件要说的就是自己装的是哪一版。"""

from __future__ import annotations

from PySide6.QtWidgets import QLabel
from tests.gui.test_shell_menus import _menu_action, _window

from xrr_fitter.version import __version__


def test_the_about_box_names_the_installed_version(qtbot, monkeypatch) -> None:
    from xrr_fitter.gui.dialog_template import StyledDialog

    shown: list[StyledDialog] = []
    monkeypatch.setattr(StyledDialog, "exec", lambda dialog: shown.append(dialog) or 0)
    window = _window(qtbot)

    _menu_action(window, "aboutAction").trigger()

    texts = " ".join(label.text() for label in shown[0].findChildren(QLabel))
    assert f"版本 {__version__}" in texts, texts
