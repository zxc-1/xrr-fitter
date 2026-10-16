"""叠加对比跟着当前数据集换，色标写清每条曲线是谁、乘了多少，颜色不撞拟合窗口那抹橙，一条错开一个数量级。

审计实测：叠加只在勾选那一刻建一次——换了当前数据集，画着的仍是旧视角下的「其他曲线」，当前那条也混在
里面；色标里一个字都没有；第二条曲线用的正是拟合窗口那抹橙；文档说「每条错开一个数量级」，代码里没有偏移。
"""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QLabel, QWidget
from tests.gui.plot_support import _panel
from tests.support.model_cases import prepared_data

from xrr_fitter.gui import theme


def _second():
    return prepared_data(size=6, intensity_raw=np.geomspace(800.0, 1.0, 6))


def _third():
    return prepared_data(size=6, intensity_raw=np.geomspace(600.0, 0.5, 6))


def _three_dataset_panel(qtbot):
    panel = _panel(qtbot, data=prepared_data(size=6))
    panel.set_dataset("second", _second())
    panel.set_dataset("third", _third())
    panel.select_dataset("curve")
    panel.toolbar.overlay_action.setChecked(True)
    return panel


def _key_captions(panel) -> tuple[str, ...]:
    row = panel.findChild(QWidget, "plotCard:logSeriesKey")
    assert row is not None, "the reflectivity card has no series key row"
    if row.isHidden():
        return ()
    return tuple(text for label in row.findChildren(QLabel) if (text := label.text()))


def test_the_overlay_follows_the_active_dataset_and_keys_each_curve(qtbot) -> None:
    panel = _three_dataset_panel(qtbot)
    assert _key_captions(panel) == ("second", "third ×10⁻¹")

    panel.select_dataset("second")

    assert _key_captions(panel) == ("curve", "third ×10⁻¹")


def test_each_overlay_curve_sits_one_decade_below_the_last(qtbot) -> None:
    panel = _three_dataset_panel(qtbot)
    curves = panel.view("log")._overlay_items

    assert len(curves) == 2
    np.testing.assert_allclose(curves[0].yData, _second().intensity_normalized)
    np.testing.assert_allclose(curves[1].yData, _third().intensity_normalized * 0.1)


def test_overlay_colours_stay_clear_of_the_marks_the_pane_already_uses(qtbot) -> None:
    panel = _three_dataset_panel(qtbot)
    reserved = {theme.DATA_OBSERVED, theme.DATA_CANDIDATE, theme.DATA_PREVIEW, theme.DATA_RANGE}
    pens = [item.opts["pen"] for item in panel.view("log")._overlay_items]

    assert not {pen.color().name().upper() for pen in pens} & {value.upper() for value in reserved}
    assert len({(pen.color().name(), pen.style()) for pen in pens}) == len(pens)


def test_switching_the_overlay_off_drops_the_curves_and_their_key(qtbot) -> None:
    panel = _three_dataset_panel(qtbot)

    panel.toolbar.overlay_action.setChecked(False)

    assert panel.view("log")._overlay_items == []
    assert _key_captions(panel) == ()
