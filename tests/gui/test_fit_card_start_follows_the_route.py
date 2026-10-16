"""拟合卡上那枚启动键就是命令栏「一键拟合」走的那条路线，不是第二个入口（F56）。

审计实测：未开高级选项时，卡片只给「自动拟合」。项目走自动路线时它与命令栏的一键拟合是同一件事、
两个名字；项目走手动或联合路线时，一键拟合跑的是手动拟合，卡片上的「自动拟合」却是另一条路、
往往还灰着，手动那枚又藏起来——卡片上没有一枚键与「本次」那一行说的路线对得上。
这里钉住：未开高级选项时卡片只露出本次路线那一枚；开了高级选项，两条路都摆出来供专家挑。
"""

from __future__ import annotations

from tests.gui.test_fit_progress import _project
from tests.gui.test_fit_route_is_named import _automatic, _window

import xrr_fitter.api as api


def _shown(window) -> tuple[str, ...]:
    panel = window.fit_panel
    buttons = (panel.automatic_button, panel.start_button)
    return tuple(button.objectName() for button in buttons if not button.isHidden())


def test_an_automatic_project_offers_only_the_automatic_start(qtbot, tmp_path) -> None:
    window = _window(qtbot, _automatic(_project(tmp_path)))

    assert window.fit_route()[0] == "自动拟合"
    assert _shown(window) == ("startAutomaticFitButton",)


def test_a_manual_project_offers_the_start_the_command_bar_runs(qtbot, tmp_path) -> None:
    window = _window(qtbot, _project(tmp_path))

    assert window.fit_route()[0] == "手动拟合"
    assert _shown(window) == ("startFitButton",)


def test_a_joint_project_offers_the_start_the_command_bar_runs(qtbot, tmp_path) -> None:
    window = _window(qtbot, _automatic(_project(tmp_path, count=2)))
    window.document.replace_project(api.set_batch_mode(window.document.project, "joint"))

    assert window.fit_route()[0] == "联合拟合"
    assert _shown(window) == ("startFitButton",)


def test_the_advanced_switch_lays_out_both_routes_for_the_expert(qtbot, tmp_path) -> None:
    window = _window(qtbot, _automatic(_project(tmp_path)))

    window.document.replace_project(api.set_expert_mode(window.document.project, True))

    assert _shown(window) == ("startAutomaticFitButton", "startFitButton")
