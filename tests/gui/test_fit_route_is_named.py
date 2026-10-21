"""「一键拟合」要说出它这一次走哪条路线。

审计实测：参数卡上那枚「显示高级选项」暗中决定一键拟合走自动路线（按导入预设搜索有界结构）
还是手动路线（按参数表），而命令栏、菜单和状态栏都不说。勾一下复选框，同一枚按钮就换了一套
做法。这里钉住：命令的提示写出当前路线；路线跟着项目改口；开跑时状态栏报出走的是哪条。
"""

from __future__ import annotations

from dataclasses import replace

from tests.gui.test_fit_progress import _FakeJob, _project

import xrr_fitter.api as api


def _automatic(project: api.XrrProject) -> api.XrrProject:
    automation = api.DatasetAutomation(
        import_batch_id="batch", role=api.AutomaticRole.UNROUTED, status=api.AutomaticStatus.PENDING
    )
    return replace(project, datasets=tuple(replace(item, automation=automation) for item in project.datasets))


def _window(qtbot, project):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow(ProjectDocument(project))
    qtbot.addWidget(window)
    window.set_guidance_visible(False)
    return window


def _tip(window) -> str:
    return window._workflow_actions["startFitAction"].toolTip()


def test_a_manual_project_names_the_parameter_table_route(qtbot, tmp_path) -> None:
    window = _window(qtbot, _project(tmp_path))

    assert "手动拟合" in _tip(window)
    assert "参数表" in _tip(window)


def test_the_route_follows_the_advanced_switch(qtbot, tmp_path) -> None:
    window = _window(qtbot, _automatic(_project(tmp_path)))
    assert "自动拟合" in _tip(window)

    window.document.replace_project(api.set_expert_mode(window.document.project, True))

    assert "手动拟合" in _tip(window)
    assert "自动拟合" in _tip(window), "the tip must say how to get the automatic route back"


def test_a_joint_project_names_the_joint_route(qtbot, tmp_path) -> None:
    window = _window(qtbot, _project(tmp_path, count=2))

    window.document.replace_project(api.set_batch_mode(window.document.project, "joint"))

    assert "联合拟合" in _tip(window)


def test_starting_announces_the_route_in_the_status_bar(qtbot, tmp_path, monkeypatch) -> None:
    window = _window(qtbot, _project(tmp_path))
    monkeypatch.setattr(api, "start_fit_job", lambda *_a, **_k: _FakeJob())
    try:
        assert window.start_fit() is True

        assert "手动拟合" in window.statusBar().currentMessage()
    finally:
        window.fit_panel.controller._finish()
