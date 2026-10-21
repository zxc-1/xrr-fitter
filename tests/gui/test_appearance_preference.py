"""F52：明暗外观记得住读者选的那一档，从没选过时跟随系统。

偏好存在 ``QSettings`` 里；``tests/conftest.py`` 把它重定向到本次会话的临时目录，
并在每条用例之后清空，所以这里读写的不是开发机的用户配置。
"""

from __future__ import annotations

from PySide6.QtCore import QSettings


def _window(qtbot):
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.set_guidance_visible(False)
    return window


def test_a_dark_choice_comes_back_on_the_next_launch(qtbot, qapp) -> None:
    from xrr_fitter.gui import theme

    first = _window(qtbot)
    first.appearance_toggle.click()
    assert first.appearance_toggle.isChecked()

    # 下一次启动：``create_application`` 先装上发货时的亮色板，再建窗口。
    theme.set_dark_appearance(qapp, False)
    second = _window(qtbot)

    assert second.appearance_toggle.isChecked()
    assert qapp.palette().window().color() == theme.dark_palette().window().color()


def test_the_first_launch_follows_the_system_without_pinning_it(qtbot, qapp, monkeypatch) -> None:
    from xrr_fitter.gui import chrome, theme

    monkeypatch.setattr(chrome, "_system_prefers_dark", lambda: True)
    window = _window(qtbot)

    assert window.appearance_toggle.isChecked()
    assert qapp.palette().window().color() == theme.dark_palette().window().color()
    # 跟随系统的那一次不落盘：系统换回亮色，下次启动也跟着回来；只有读者亲手按的才记住。
    assert QSettings().value(chrome.APPEARANCE_SETTINGS_KEY) is None
