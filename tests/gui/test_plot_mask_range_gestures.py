"""掩膜点按的是读者指着的那个点，选框从当前范围起手、两页一起动，范围模式里照样能缩放。

审计实测：掩膜点击只看横坐标、挑最近的那个点——点在曲线正上方很远处也会排掉一个点；悬停没有任何
提示；对数页把被排除的点和拟合点画成同一种蓝点；范围选框每次都从 0–1 起手，而不是当前范围；原始页
拖了选框，对数页那只不跟；进了范围模式滚轮缩放就失灵。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QLabel, QWidget
from tests.gui.plot_support import _drag_range, _panel
from tests.support.model_cases import prepared_data


def _click(scene_pos: QPointF) -> SimpleNamespace:
    """场景点击事件只被读这两样；方法名照 pyqtgraph 的事件接口取，所以用关键字传。"""
    return SimpleNamespace(button=lambda: Qt.MouseButton.LeftButton, scenePos=lambda: scene_pos)


class _Wheel:
    def __init__(self, pos: QPointF, delta: int) -> None:
        self._pos, self._delta, self.accepted = pos, delta, False

    def delta(self) -> int:
        return self._delta

    def pos(self) -> QPointF:
        return self._pos

    def accept(self) -> None:
        self.accepted = True

    def ignore(self) -> None:
        self.accepted = False


def _laid_out(qtbot, key: str, data=None):
    data = prepared_data(size=4) if data is None else data
    panel = _panel(qtbot, data=data)
    panel.resize(1000, 800)
    panel.show()
    qtbot.waitExposed(panel)
    panel.select_view(key)
    qtbot.wait(10)
    return panel, data


def _scene_of(pane, x: float, y: float) -> QPointF:
    view_y = np.log10(y) if pane.plot_item.ctrl.logYCheck.isChecked() else y
    return pane.plot_item.vb.mapViewToScene(QPointF(x, view_y))


def test_a_mask_click_takes_the_point_under_the_pointer_not_the_nearest_column(qtbot) -> None:
    panel, data = _laid_out(qtbot, "raw")
    emitted: list[int] = []
    panel.point_mask_requested.connect(emitted.append)
    panel.set_interaction_mode("mask")
    raw = panel.view("raw")
    target = _scene_of(raw, float(data.two_theta_deg[2]), float(data.intensity_raw[2]))

    raw._on_scene_clicked(_click(target + QPointF(2.0, -2.0)))
    raw._on_scene_clicked(_click(target + QPointF(0.0, -40.0)))

    assert emitted == [2]


def test_hovering_a_maskable_point_rings_it_and_offers_a_hand(qtbot) -> None:
    panel, data = _laid_out(qtbot, "raw")
    panel.set_interaction_mode("mask")
    raw = panel.view("raw")
    target = _scene_of(raw, float(data.two_theta_deg[1]), float(data.intensity_raw[1]))

    raw._on_scene_moved(target)
    x, _y = raw.mask_hover_item.getData()
    assert np.allclose(x, [data.two_theta_deg[1]])
    assert raw.cursor().shape() == Qt.CursorShape.PointingHandCursor

    raw._on_scene_moved(target + QPointF(0.0, -60.0))
    x, _y = raw.mask_hover_item.getData()
    assert x is None or len(x) == 0
    assert raw.cursor().shape() != Qt.CursorShape.PointingHandCursor


def test_the_log_page_marks_excluded_points_and_keys_them(qtbot) -> None:
    mask = np.array([True, False, True, True])
    data = prepared_data(size=4, fit_mask=mask)
    panel, data = _laid_out(qtbot, "log", data)
    log = panel.view("log")

    shown, _y = log.observed_item.getData()
    excluded, _y = log.excluded_item.getData()
    assert np.allclose(shown, data.two_theta_deg[mask])
    assert np.allclose(excluded, data.two_theta_deg[~mask])
    legend = panel.findChild(QWidget, "plotCard:logLegend")
    assert any("掩膜排除点" in label.text() for label in legend.findChildren(QLabel))


def test_the_range_selector_starts_on_the_current_range_and_both_pages_move_together(qtbot) -> None:
    panel, data = _laid_out(qtbot, "raw")
    low, high = float(data.two_theta_deg[1]), float(data.two_theta_deg[2])
    panel.show_range(low, high)

    panel.set_interaction_mode("range")
    for key in ("raw", "log"):
        assert np.allclose(sorted(panel.view(key).range_item.getRegion()), (low, high)), key

    new_high = float(data.two_theta_deg[3])
    _drag_range(panel, low, new_high)
    assert np.allclose(sorted(panel.view("log").range_item.getRegion()), (low, new_high))


def test_the_wheel_still_zooms_while_the_range_selector_is_armed(qtbot) -> None:
    panel, _data = _laid_out(qtbot, "raw")
    raw = panel.view("raw")
    panel.set_interaction_mode("range")
    view_box = raw.plot_item.vb
    (x0, x1), _y = view_box.viewRange()

    wheel = _Wheel(QPointF(view_box.width() / 2.0, view_box.height() / 2.0), 120)
    view_box.wheelEvent(wheel)

    (z0, z1), _y = view_box.viewRange()
    assert wheel.accepted
    assert (z1 - z0) < (x1 - x0)
    assert isinstance(raw.range_item, pg.LinearRegionItem)
