"""SLD 剖面上的把手：拖动才提交、单击不提交；抓取按像素判、界面圆点优先；悬停有反馈；Esc 取消。

审计里复现过：在剖面上随手点一下就提交了一次结构修改，而结构修改会清空拟合结果。层把手只看
x 不看 y，整层宽度都是它的抓取区；松开时没有位移阈值；在界面圆点右侧按下，抓到的是粗糙度。
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from matplotlib.backend_bases import MouseButton, MouseEvent
from PySide6.QtCore import Qt
from tests.gui.plot_cases_5 import _structure_only_panel
from tests.support.model_cases import simple_structure


def _sld(panel):
    view = panel.view("sld")
    view.canvas.draw()
    return view


def _handle(view, label):
    return next(line for line in view.axes.lines if line.get_label() == label)


def _pixel(view, x_data, y_data):
    return tuple(float(value) for value in view.axes.transData.transform((x_data, y_data)))


def _event(view, name, x, y, *, button=MouseButton.LEFT, buttons=None):
    view.canvas.callbacks.process(name, MouseEvent(name, view.canvas, x, y, button=button, buttons=buttons))


def _gesture(view, start, end):
    """按下、拖到、松开，三个事件都给像素坐标——读者的准头是按像素算的。"""
    _event(view, "button_press_event", *start)
    _event(view, "motion_notify_event", *end, buttons={MouseButton.LEFT})
    _event(view, "button_release_event", *end)


def _dot(view, index=0):
    interface = _handle(view, f"_interface_{index}")
    depth = float(interface.get_xdata()[1])
    return depth, _pixel(view, depth, float(interface.get_ydata()[1]))


def _edits(panel):
    edits: list[object] = []
    panel.structure_edit_requested.connect(edits.append)
    return edits


def test_a_click_that_never_crosses_the_drag_threshold_commits_nothing(qtbot, tmp_path) -> None:
    panel = _structure_only_panel(qtbot, tmp_path)
    edits = _edits(panel)
    view = _sld(panel)
    depth, (x, y) = _dot(view)

    # 手抖了 1px：没越过拖动阈值，这一下只是单击，不该提交结构，更不该清掉拟合结果。
    _gesture(view, (x, y), (x + 1.0, y))

    assert edits == []
    assert float(_handle(view, "_interface_0").get_xdata()[0]) == pytest.approx(depth)


def test_a_level_handle_is_grabbed_only_near_its_own_line(qtbot, tmp_path) -> None:
    panel = _structure_only_panel(qtbot, tmp_path)
    edits = _edits(panel)
    view = _sld(panel)
    level = float(_handle(view, "_level_0").get_ydata()[0])
    lower, _upper = view.axes.get_ylim()
    x, y_line = _pixel(view, 1.0, level)
    _x, y_far = _pixel(view, 1.0, lower + 0.1 * (level - lower))

    # 层中段、远在电平线下方按下再拖：此前整层宽度都是层把手的抓取区，这一下会改密度。
    _gesture(view, (x, y_far), (x, y_far + 30.0))
    assert edits == []

    _gesture(view, (x, y_line), (x, y_line - 30.0))
    assert len(edits) == 1
    assert edits[0].components[0].density_scale < 1.0


def test_pressing_on_the_interface_dot_drags_the_interface_not_the_roughness(qtbot, tmp_path) -> None:
    # σ 只有 1 Å 时须线几乎整个缩在圆点里；圆点右边那一小片此前归粗糙度，按在点上拖出来改的却是粗糙度。
    structure = replace(simple_structure(), backing_roughness_a=1.0)
    panel = _structure_only_panel(qtbot, tmp_path, structure=structure)
    edits = _edits(panel)
    view = _sld(panel)
    _depth, (x, y) = _dot(view)

    _gesture(view, (x + 3.0, y), (x + 40.0, y))

    assert len(edits) == 1
    assert edits[0].backing_roughness_a == pytest.approx(1.0)
    assert edits[0].components[0].thickness_a > 20.0


def test_the_roughness_whisker_is_still_grabbed_beyond_the_dot(qtbot, tmp_path) -> None:
    panel = _structure_only_panel(qtbot, tmp_path)
    edits = _edits(panel)
    view = _sld(panel)
    whisker = _handle(view, "_roughness_0")
    tip_x, tip_y = _pixel(view, float(whisker.get_xdata()[1]), float(whisker.get_ydata()[1]))

    _gesture(view, (tip_x, tip_y), (tip_x + 20.0, tip_y))

    assert len(edits) == 1
    assert edits[0].backing_roughness_a > 3.0
    assert edits[0].components[0].thickness_a == pytest.approx(20.0)


def test_hovering_a_handle_highlights_it_and_changes_the_cursor(qtbot, tmp_path) -> None:
    panel = _structure_only_panel(qtbot, tmp_path)
    view = _sld(panel)
    resting = _handle(view, "_interface_0").get_linewidth()
    _depth, (x, y) = _dot(view)

    _event(view, "motion_notify_event", x, y, button=None)

    assert view.canvas.cursor().shape() == Qt.CursorShape.SizeHorCursor
    assert _handle(view, "_interface_0").get_linewidth() > resting

    lower, _upper = view.axes.get_ylim()
    away_x, away_y = _pixel(view, 1.0, lower)
    _event(view, "motion_notify_event", away_x, away_y + 5.0, button=None)

    assert view.canvas.cursor().shape() == Qt.CursorShape.ArrowCursor
    assert _handle(view, "_interface_0").get_linewidth() == pytest.approx(resting)


def test_hovering_a_level_line_offers_a_vertical_resize(qtbot, tmp_path) -> None:
    panel = _structure_only_panel(qtbot, tmp_path)
    view = _sld(panel)
    x, y = _pixel(view, 1.0, float(_handle(view, "_level_0").get_ydata()[0]))

    _event(view, "motion_notify_event", x, y, button=None)

    assert view.canvas.cursor().shape() == Qt.CursorShape.SizeVerCursor


def test_escape_cancels_a_drag_in_flight_and_restores_the_handle(qtbot, tmp_path) -> None:
    panel = _structure_only_panel(qtbot, tmp_path)
    edits = _edits(panel)
    view = _sld(panel)
    depth, (x, y) = _dot(view)
    _event(view, "button_press_event", x, y)
    _event(view, "motion_notify_event", x + 40.0, y, buttons={MouseButton.LEFT})
    assert float(_handle(view, "_interface_0").get_xdata()[0]) > depth

    qtbot.keyClick(view.canvas, Qt.Key.Key_Escape)
    _event(view, "button_release_event", x + 40.0, y)

    assert edits == []
    assert float(_handle(view, "_interface_0").get_xdata()[0]) == pytest.approx(depth)
