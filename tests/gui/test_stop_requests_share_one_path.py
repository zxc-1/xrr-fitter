"""停止一次拟合只有一条路：拟合卡自己的请求函数。

审计实测：在主窗口任何位置按 Esc 都会停掉正在跑的拟合（输入框里按 Esc 取消编辑也算）；菜单
「取消拟合」、命令栏 ⏹、关窗和「强制停止拟合」各自直接喊控制器，拟合卡于是继续写着「运行中」、
停止键还亮着、强制停止那一枚永远不露面。这里钉住：Esc 只在拟合卡里起作用；每个入口都走卡的
请求函数；菜单、命令栏与卡上的可用状态同步。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from tests.gui.test_fit_progress import _FakeJob, _project

import xrr_fitter.api as api


def _running_window(qtbot, tmp_path, monkeypatch):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow(ProjectDocument(_project(tmp_path)))
    qtbot.addWidget(window)
    window.set_guidance_visible(False)
    window.resize(1400, 900)
    window.show()
    window.activateWindow()
    qtbot.waitExposed(window)
    # 窗口级快捷键只派给活动窗口；不激活的话，Esc 那一例在修之前也是绿的。
    qtbot.waitUntil(lambda: window.isActiveWindow())
    job = _FakeJob()
    calls: list[str] = []
    job.cancel = lambda: calls.append("cancel")
    job.force_stop = lambda: calls.append("force")
    monkeypatch.setattr(api, "start_fit_job", lambda *_a, **_k: job)
    assert window.fit_panel.start_fit() is True
    return window, calls


def _action(window, name):
    return window.chrome_actions.get(name) or window._workflow_actions[name]


def test_escape_anywhere_in_the_window_stops_through_the_fit_card_once(qtbot, tmp_path, monkeypatch) -> None:
    """Esc 归窗口那一条命令（已批准的停止入口方案）：焦点在哪都只发一次取消，拟合卡跟着进停止态。"""
    window, calls = _running_window(qtbot, tmp_path, monkeypatch)
    try:
        window.data_panel.tree.setFocus()
        qtbot.keyClick(window.data_panel.tree, Qt.Key.Key_Escape)
        qtbot.keyClick(window.data_panel.tree, Qt.Key.Key_Escape)

        assert calls == ["cancel"]
        assert window.fit_panel.cancel_button.isEnabled() is False
        assert _action(window, "cancelFitAction").shortcut().toString() == "Esc"
    finally:
        window.fit_panel.controller._finish()


def test_the_menu_cancel_is_acknowledged_by_the_fit_card(qtbot, tmp_path, monkeypatch) -> None:
    window, calls = _running_window(qtbot, tmp_path, monkeypatch)
    try:
        _action(window, "cancelFitAction").trigger()

        assert calls == ["cancel"]
        panel = window.fit_panel
        assert panel.cancel_button.isEnabled() is False
        assert panel.force_button.isVisible() and panel.force_button.isEnabled()
        assert _action(window, "cancelFitAction").isEnabled() is False
    finally:
        window.fit_panel.controller._finish()


def test_the_toolbar_stop_and_menu_follow_the_card(qtbot, tmp_path, monkeypatch) -> None:
    window, calls = _running_window(qtbot, tmp_path, monkeypatch)
    try:
        force = _action(window, "forceStopFitAction")
        assert force.isEnabled() is False

        qtbot.mouseClick(window.fit_panel.cancel_button, Qt.MouseButton.LeftButton)

        assert calls == ["cancel"]
        assert window.toolbar_stop_button.isEnabled() is False
        assert _action(window, "cancelFitAction").isEnabled() is False
        assert force.isEnabled() is True
    finally:
        window.fit_panel.controller._finish()


def test_the_menu_force_stop_escalates_through_the_card(qtbot, tmp_path, monkeypatch) -> None:
    window, calls = _running_window(qtbot, tmp_path, monkeypatch)
    try:
        _action(window, "cancelFitAction").trigger()
        _action(window, "forceStopFitAction").trigger()

        assert calls == ["cancel", "force"]
        assert window.fit_panel.force_button.isEnabled() is False
        assert _action(window, "forceStopFitAction").isEnabled() is False
    finally:
        window.fit_panel.controller._finish()


def test_closing_mid_run_asks_the_fit_card_to_stop(qtbot, tmp_path, monkeypatch) -> None:
    window, calls = _running_window(qtbot, tmp_path, monkeypatch)
    try:
        window.close()

        assert calls == ["cancel"]
        assert window.close_pending is True
        assert window.fit_panel.cancel_button.isEnabled() is False
    finally:
        window.fit_panel.controller._finish()
